"""Train Text -> DeepCAD latent mapping.

Expected annotations (JSONL), one caption per line:
    {"id": "0000/00001234", "text": "a cylindrical sleeve ...", "split": "train"}

``id`` must match the existing DeepCAD ``data/cad_vec/<id>.h5`` file.  The
pretrained autoencoder is frozen and provides target latent codes.  Only the
text encoder is optimized.

Example:
    python train_text2cad.py \
      --annotations data/text2cad_annotations.jsonl \
      --data-root data \
      --ae-exp-name DeepCAD_Optimized \
      --ae-ckpt 500 \
      --output proj_log/Text2CAD
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cadlib.macro import EOS_VEC, MAX_TOTAL_LEN
from config.configAE import ConfigAE
from trainer.trainerAE import TrainerAE
from text2cad_ml import TextVocabulary, TextLatentEncoder, latent_alignment_loss


class CaptionCADDataset(Dataset):
    def __init__(self, rows, data_root, vocab, max_text_len=64):
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
    # ConfigAE uses argparse internally. Isolate it from this script's CLI.
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], "-m", "enc"]
        cfg = ConfigAE("test")
    finally:
        sys.argv = saved_argv

    cfg.data_root = args.data_root
    cfg.exp_name = args.ae_exp_name
    cfg.exp_dir = os.path.join(cfg.proj_dir, cfg.exp_name)
    cfg.model_dir = os.path.join(cfg.exp_dir, "model")

    ae = TrainerAE(cfg)
    ae.load_ckpt(args.ae_ckpt)
    ae.net.eval()
    for parameter in ae.net.parameters():
        parameter.requires_grad = False
    return cfg, ae


def evaluate(model, ae, loader):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            commands = batch["command"].cuda(non_blocking=True)
            cad_args = batch["args"].cuda(non_blocking=True)
            text = batch["text"].cuda(non_blocking=True)
            target_z = ae.encode({"command": commands, "args": cad_args}, is_batch=True)
            pred_z = model(text)
            loss = latent_alignment_loss(pred_z, target_z)
            total += loss.item() * text.size(0)
            count += text.size(0)
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ae-exp-name", default="DeepCAD_Optimized")
    parser.add_argument("--ae-ckpt", default="500")
    parser.add_argument("--output", default="proj_log/Text2CAD")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-token-freq", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The current DeepCAD implementation requires CUDA")

    train_rows = load_rows(args.annotations, "train")
    val_rows = load_rows(args.annotations, "validation")
    if not train_rows:
        raise ValueError("No train annotations found")
    if not val_rows:
        # Allow a simple dataset to run; use a deterministic tail as validation.
        n_val = max(1, int(0.05 * len(train_rows)))
        val_rows = train_rows[-n_val:]
        train_rows = train_rows[:-n_val]

    vocab = TextVocabulary.build((r["text"] for r in train_rows), args.min_token_freq)
    os.makedirs(args.output, exist_ok=True)
    vocab_path = os.path.join(args.output, "vocab.json")
    vocab.save(vocab_path)

    cfg_ae, ae = load_frozen_autoencoder(args)
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
        dim_z=cfg_ae.dim_z,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        max_len=args.max_text_len,
        dropout=0.1,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc="Text2CAD epoch {}/{}".format(epoch, args.epochs))
        for batch in pbar:
            commands = batch["command"].cuda(non_blocking=True)
            cad_args = batch["args"].cuda(non_blocking=True)
            text = batch["text"].cuda(non_blocking=True)

            with torch.no_grad():
                target_z = ae.encode({"command": commands, "args": cad_args}, is_batch=True)

            pred_z = model(text)
            loss = latent_alignment_loss(pred_z, target_z)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * text.size(0)
            seen += text.size(0)
            pbar.set_postfix(loss=running / max(seen, 1))

        val_loss = evaluate(model, ae, val_loader)
        print("epoch={} train_loss={:.6f} val_loss={:.6f}".format(
            epoch, running / max(seen, 1), val_loss
        ))

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "vocab_path": "vocab.json",
            "max_text_len": args.max_text_len,
            "dim_z": cfg_ae.dim_z,
            "vocab_size": len(vocab),
        }
        torch.save(state, os.path.join(args.output, "latest.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, os.path.join(args.output, "best.pth"))


if __name__ == "__main__":
    main()
