"""Evaluate trained Text -> DeepCAD mapping on held-out annotations.

Reports latent alignment, CAD command/argument accuracy, exact command-sequence
accuracy, syntactic CADSequence validity and (optionally) OpenCASCADE solid
validity. The test split is never used by train_text2cad.py for optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cadlib.extrude import CADSequence
from cadlib.macro import CMD_ARGS_MASK, EOS_IDX
from cadlib.visualize import create_CAD
from text2cad_ml import TextVocabulary, TextLatentEncoder
from train_text2cad import (
    CaptionCADDataset,
    _normalize_decoder_shapes,
    load_frozen_autoencoder,
    load_rows,
)


def active_sequence_mask(commands):
    """Mask from first command through first EOS, excluding EOS padding tail."""
    mask = torch.zeros_like(commands, dtype=torch.bool)
    for i in range(commands.size(0)):
        eos = (commands[i] == EOS_IDX).nonzero()
        end = int(eos[0].item()) + 1 if eos.numel() else commands.size(1)
        mask[i, :end] = True
    return mask


def load_text_model(checkpoint_path, vocab_path=None):
    checkpoint = torch.load(checkpoint_path)
    vocab_path = vocab_path or os.path.join(
        os.path.dirname(checkpoint_path), checkpoint.get("vocab_path", "vocab.json")
    )
    vocab = TextVocabulary.load(vocab_path)
    cfg = checkpoint.get("model_config", {})
    model = TextLatentEncoder(
        vocab_size=len(vocab),
        dim_z=int(checkpoint.get("dim_z", 256)),
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 4)),
        dim_feedforward=int(cfg.get("dim_feedforward", 512)),
        max_len=int(checkpoint.get("max_text_len", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, vocab, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default="proj_log/Text2CAD/best.pth")
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--output", default="proj_log/Text2CAD/evaluation.json")
    parser.add_argument("--ae-proj-dir", default="proj_log")
    parser.add_argument("--ae-exp-name", default="DeepCAD_Optimized")
    parser.add_argument("--ae-ckpt", default="500")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires the CUDA DeepCAD environment")

    checkpoint, vocab, model = load_text_model(args.checkpoint, args.vocab)
    args.max_text_len = int(checkpoint.get("max_text_len", 128))
    _, ae = load_frozen_autoencoder(args)

    rows = load_rows(args.annotations, args.split)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No annotations found for split {}".format(args.split))

    dataset = CaptionCADDataset(rows, args.data_root, vocab, args.max_text_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    count = 0
    latent_mse_sum = 0.0
    cosine_sum = 0.0
    command_correct = 0
    command_total = 0
    args_correct = 0
    args_total = 0
    exact_command_sequences = 0
    sequence_valid = 0
    solid_valid = 0
    geometry_attempted = 0
    inference_seconds = 0.0

    mask_table = torch.tensor(CMD_ARGS_MASK, dtype=torch.bool, device="cuda")

    with torch.no_grad():
        for batch in loader:
            commands = batch["command"].cuda(non_blocking=True)
            cad_args = batch["args"].cuda(non_blocking=True)
            text = batch["text"].cuda(non_blocking=True)

            target_z = ae.encode({"command": commands, "args": cad_args}, is_batch=True)
            if target_z.dim() == 3 and target_z.size(1) == 1:
                target_z_flat = target_z[:, 0]
            else:
                target_z_flat = target_z

            torch.cuda.synchronize()
            started = time.time()
            pred_z = model(text)
            decoder_z = pred_z.unsqueeze(1) if pred_z.dim() == 2 else pred_z
            outputs = ae.decode(decoder_z)
            torch.cuda.synchronize()
            inference_seconds += time.time() - started

            command_logits, args_logits = _normalize_decoder_shapes(outputs, commands)
            pred_commands = command_logits.argmax(dim=-1)
            pred_args = args_logits.argmax(dim=-1) - 1

            batch_size = text.size(0)
            count += batch_size
            latent_mse_sum += torch.mean((pred_z - target_z_flat) ** 2, dim=-1).sum().item()
            cosine_sum += torch.nn.functional.cosine_similarity(
                pred_z, target_z_flat, dim=-1
            ).sum().item()

            active = active_sequence_mask(commands)
            command_correct += ((pred_commands == commands) & active).sum().item()
            command_total += active.sum().item()

            arg_mask = mask_table[commands.long()] & active.unsqueeze(-1)
            args_correct += ((pred_args == cad_args) & arg_mask).sum().item()
            args_total += arg_mask.sum().item()

            for i in range(batch_size):
                row_mask = active[i]
                if torch.equal(pred_commands[i][row_mask], commands[i][row_mask]):
                    exact_command_sequences += 1

            vectors = ae.logits2vec(outputs, refill_pad=True, to_numpy=True)
            for vector in vectors:
                try:
                    cad_seq = CADSequence.from_vector(vector, is_numerical=True, n=256)
                    if not cad_seq.seq:
                        raise ValueError("empty CAD sequence")
                    sequence_valid += 1
                    if args.geometry:
                        geometry_attempted += 1
                        create_CAD(cad_seq)
                        solid_valid += 1
                except Exception:
                    if args.geometry and "cad_seq" in locals() and getattr(cad_seq, "seq", None):
                        geometry_attempted += 1
                    continue

    report = {
        "split": args.split,
        "samples": count,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "latent_mse": latent_mse_sum / max(count, 1),
        "latent_cosine_similarity": cosine_sum / max(count, 1),
        "command_accuracy": command_correct / max(command_total, 1),
        "argument_accuracy": args_correct / max(args_total, 1),
        "exact_command_sequence_accuracy": exact_command_sequences / max(count, 1),
        "valid_cad_sequence_rate": sequence_valid / max(count, 1),
        "invalidity_ratio": 1.0 - sequence_valid / max(count, 1),
        "valid_solid_rate": (
            solid_valid / max(geometry_attempted, 1) if args.geometry else None
        ),
        "mean_inference_ms_per_sample": 1000.0 * inference_seconds / max(count, 1),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
