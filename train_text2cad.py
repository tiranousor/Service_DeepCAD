"""Train Text -> DeepCAD latent mapping.

The pretrained DeepCAD autoencoder is frozen. Only the text encoder is trained.
The objective combines latent alignment with CAD command/argument reconstruction
through the frozen decoder, so the predicted latent code must not only be close
to the CAD latent target but also decode to the correct engineering history.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cadlib.macro import EOS_VEC, MAX_TOTAL_LEN, CMD_ARGS_MASK
from config.configAE import ConfigAE
from trainer.trainerAE import TrainerAE
from text2cad_ml import TextVocabulary, TextLatentEncoder, latent_alignment_loss


class CaptionCADDataset(Dataset):
    def __init__(self, rows, data_root, vocab, max_text_len=128):
        self.rows = rows
        self.data_root = data_root
        self.vocab = vocab
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = os.path.join(self.data_root, "cad_vec", row["id"] + ".h5")
        with h5py.File(path, "r") as fp:
            vec = fp["vec"][:]

        if vec.shape[0] > MAX_TOTAL_LEN:
            raise ValueError("CAD sequence exceeds MAX_TOTAL_LEN: {}".format(row["id"]))
        pad_len = MAX_TOTAL_LEN - vec.shape[0]
        if pad_len:
            vec = np.concatenate(
                [vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)],
                axis=0,
            )

        return {
            "text": torch.tensor(self.vocab.encode(row["text"], self.max_text_len), dtype=torch.long),
            "command": torch.tensor(vec[:, 0], dtype=torch.long),
            "args": torch.tensor(vec[:, 1:], dtype=torch.long),
            "id": row["id"],
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(path, split):
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "text" not in row:
                raise ValueError("Each annotation must contain id and text")
            if row.get("split", "train") == split:
                rows.append(row)
    return rows


def load_frozen_autoencoder(args):
    # ConfigAE parses sys.argv internally. Hide Text2CAD CLI arguments from it.
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], "-m", "enc"]
        cfg = ConfigAE("test")
    finally:
        sys.argv = saved_argv

    cfg.data_root = args.data_root
    cfg.proj_dir = args.ae_proj_dir
    cfg.exp_name = args.ae_exp_name
    cfg.exp_dir = os.path.join(cfg.proj_dir, cfg.exp_name)
    cfg.model_dir = os.path.join(cfg.exp_dir, "model")

    ae = TrainerAE(cfg)
    ae.load_ckpt(args.ae_ckpt)
    ae.net.eval()
    for parameter in ae.net.parameters():
        parameter.requires_grad = False
    return cfg, ae


def _normalize_decoder_shapes(outputs, commands):
    command_logits = outputs["command_logits"]
    args_logits = outputs["args_logits"]

    # Some DeepCAD variants keep a singleton group dimension.
    while command_logits.dim() > commands.dim() + 1 and command_logits.size(1) == 1:
        command_logits = command_logits.squeeze(1)
    while args_logits.dim() > commands.dim() + 2 and args_logits.size(1) == 1:
        args_logits = args_logits.squeeze(1)
    return command_logits, args_logits


def reconstruction_loss(ae, pred_z, commands, cad_args):
    """Differentiable loss through the frozen DeepCAD decoder.

    DeepCAD represents CAD arguments -1..255 as classifier classes 0..256,
    hence the +1 target shift used below (the same inverse shift appears in
    TrainerAE.logits2vec).
    """
    decoder_z = pred_z.unsqueeze(1) if pred_z.dim() == 2 else pred_z
    outputs = ae.decode(decoder_z)
    command_logits, args_logits = _normalize_decoder_shapes(outputs, commands)

    command_loss = F.cross_entropy(
        command_logits.contiguous().view(-1, command_logits.size(-1)),
        commands.contiguous().view(-1),
    )

    mask_table = torch.tensor(CMD_ARGS_MASK, dtype=torch.bool, device=commands.device)
    arg_mask = mask_table[commands.long()]
    arg_targets = (cad_args + 1).long()
    if arg_mask.any().item():
        args_loss = F.cross_entropy(args_logits[arg_mask], arg_targets[arg_mask])
    else:
        args_loss = command_loss * 0.0

    with torch.no_grad():
        command_acc = (command_logits.argmax(dim=-1) == commands).float().mean()
        if arg_mask.any().item():
            pred_args = args_logits.argmax(dim=-1) - 1
            args_acc = (pred_args[arg_mask] == cad_args[arg_mask]).float().mean()
        else:
            args_acc = torch.tensor(0.0, device=commands.device)

    return command_loss, args_loss, command_acc, args_acc


def compute_batch_loss(model, ae, batch, args, training):
    commands = batch["command"].cuda(non_blocking=True)
    cad_args = batch["args"].cuda(non_blocking=True)
    text = batch["text"].cuda(non_blocking=True)

    with torch.no_grad():
        target_z = ae.encode({"command": commands, "args": cad_args}, is_batch=True)

    pred_z = model(text)
    latent_loss = latent_alignment_loss(pred_z, target_z)
    command_loss, args_loss, command_acc, args_acc = reconstruction_loss(
        ae, pred_z, commands, cad_args
    )

    total = (
        args.lambda_latent * latent_loss
        + args.lambda_command * command_loss
        + args.lambda_args * args_loss
    )
    return total, {
        "latent_loss": latent_loss,
        "command_loss": command_loss,
        "args_loss": args_loss,
        "command_acc": command_acc,
        "args_acc": args_acc,
    }, text.size(0)


def evaluate(model, ae, loader, args):
    model.eval()
    totals = {
        "loss": 0.0,
        "latent_loss": 0.0,
        "command_loss": 0.0,
        "args_loss": 0.0,
        "command_acc": 0.0,
        "args_acc": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            loss, parts, batch_size = compute_batch_loss(model, ae, batch, args, training=False)
            totals["loss"] += loss.item() * batch_size
            for key in parts:
                totals[key] += parts[key].item() * batch_size
            count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(path, epoch, model, optimizer, scheduler, val_metrics, vocab, args, best_val):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_metrics["loss"],
        "val_metrics": val_metrics,
        "best_val": best_val,
        "vocab_path": "vocab.json",
        "max_text_len": args.max_text_len,
        "dim_z": args.dim_z,
        "vocab_size": len(vocab),
        "model_config": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
        },
        "loss_config": {
            "lambda_latent": args.lambda_latent,
            "lambda_command": args.lambda_command,
            "lambda_args": args.lambda_args,
        },
    }
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ae-proj-dir", default="proj_log")
    parser.add_argument("--ae-exp-name", default="DeepCAD_Optimized")
    parser.add_argument("--ae-ckpt", default="500")
    parser.add_argument("--output", default="proj_log/Text2CAD")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-text-len", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-token-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--dim-z", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lambda-latent", type=float, default=1.0)
    parser.add_argument("--lambda-command", type=float, default=0.1)
    parser.add_argument("--lambda-args", type=float, default=0.25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Text2CAD neural training requires an NVIDIA CUDA environment")

    set_seed(args.seed)
    train_rows = load_rows(args.annotations, "train")
    val_rows = load_rows(args.annotations, "validation")
    if not train_rows:
        raise ValueError("No train annotations found")
    if not val_rows:
        n_val = max(1, int(0.05 * len(train_rows)))
        val_rows = train_rows[-n_val:]
        train_rows = train_rows[:-n_val]

    vocab = TextVocabulary.build((r["text"] for r in train_rows), args.min_token_freq)
    os.makedirs(args.output, exist_ok=True)
    vocab_path = os.path.join(args.output, "vocab.json")
    vocab.save(vocab_path)

    cfg_ae, ae = load_frozen_autoencoder(args)
    args.dim_z = cfg_ae.dim_z

    train_ds = CaptionCADDataset(train_rows, args.data_root, vocab, args.max_text_len)
    val_ds = CaptionCADDataset(val_rows, args.data_root, vocab, args.max_text_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TextLatentEncoder(
        vocab_size=len(vocab),
        dim_z=args.dim_z,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_len=args.max_text_len,
        dropout=args.dropout,
    ).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_loss", best_val)))
        print("Resumed from {} at epoch {}".format(args.resume, start_epoch))

    metrics_path = os.path.join(args.output, "metrics.jsonl")
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc="Text2CAD epoch {}/{}".format(epoch, args.epochs))
        for batch in pbar:
            loss, parts, batch_size = compute_batch_loss(model, ae, batch, args, training=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * batch_size
            seen += batch_size
            pbar.set_postfix(
                loss=running / max(seen, 1),
                cmd=parts["command_acc"].item(),
                args=parts["args_acc"].item(),
            )

        train_loss = running / max(seen, 1)
        val_metrics = evaluate(model, ae, val_loader, args)
        scheduler.step(val_metrics["loss"])
        lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": lr,
            "val": val_metrics,
        }
        print(json.dumps(record, ensure_ascii=False))
        with open(metrics_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        latest_path = os.path.join(args.output, "latest.pth")
        save_checkpoint(
            latest_path, epoch, model, optimizer, scheduler, val_metrics, vocab, args, best_val
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            epochs_without_improvement = 0
            best_path = os.path.join(args.output, "best.pth")
            save_checkpoint(
                best_path, epoch, model, optimizer, scheduler, val_metrics, vocab, args, best_val
            )
            print("New best checkpoint: {} (val_loss={:.6f})".format(best_path, best_val))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping after {} epochs without validation improvement".format(args.patience))
            break


if __name__ == "__main__":
    main()
