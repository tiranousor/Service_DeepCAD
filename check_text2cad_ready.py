"""Diagnostic for Text2CAD data, checkpoints and runtime.

Safe to run on macOS or inside the GPU Docker image. It does not initialize the
DeepCAD network; it only reports whether the files needed for the next stage
are present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def count_files(root, suffix):
    if not os.path.isdir(root):
        return 0
    count = 0
    for _, _, files in os.walk(root):
        count += sum(1 for name in files if name.endswith(suffix))
    return count


def annotation_stats(path):
    stats = {"train": 0, "validation": 0, "test": 0, "other": 0}
    if not os.path.isfile(path):
        return stats
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split", "train")
            if split in stats:
                stats[split] += 1
            else:
                stats["other"] += 1
    return stats


def find_ae_checkpoint(exp_name, checkpoint):
    candidates = [
        os.path.join("proj_log", exp_name, "model", "ckpt_epoch{}.pth".format(checkpoint)),
        os.path.join("/workspace/proj_log", exp_name, "model", "ckpt_epoch{}.pth".format(checkpoint)),
        os.path.join("/app/proj_log", exp_name, "model", "ckpt_epoch{}.pth".format(checkpoint)),
        os.path.join("/app/DeepCAD/proj_log", exp_name, "model", "ckpt_epoch{}.pth".format(checkpoint)),
    ]
    existing = [path for path in candidates if os.path.isfile(path)]
    return candidates, existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--annotations", default="data/text2cad_annotations.jsonl")
    parser.add_argument("--text-output", default="proj_log/Text2CAD")
    parser.add_argument("--ae-exp-name", default="DeepCAD_Optimized")
    parser.add_argument("--ae-ckpt", default="500")
    args = parser.parse_args()

    cad_json = os.path.join(args.data_root, "cad_json")
    cad_vec = os.path.join(args.data_root, "cad_vec")
    split = os.path.join(args.data_root, "train_val_test_split.json")
    best = os.path.join(args.text_output, "best.pth")
    vocab = os.path.join(args.text_output, "vocab.json")
    candidates, ae_existing = find_ae_checkpoint(args.ae_exp_name, args.ae_ckpt)

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        torch_version = None
        cuda_available = False
        torch_error = str(exc)
    else:
        torch_error = None

    report = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "torch_error": torch_error,
        "cuda_available": cuda_available,
        "data": {
            "cad_json_dir": cad_json,
            "cad_json_exists": os.path.isdir(cad_json),
            "cad_json_files": count_files(cad_json, ".json"),
            "cad_vec_dir": cad_vec,
            "cad_vec_exists": os.path.isdir(cad_vec),
            "cad_vec_files": count_files(cad_vec, ".h5"),
            "split_file": split,
            "split_exists": os.path.isfile(split),
        },
        "annotations": {
            "path": args.annotations,
            "exists": os.path.isfile(args.annotations),
            "splits": annotation_stats(args.annotations),
        },
        "deepcad_autoencoder": {
            "experiment": args.ae_exp_name,
            "checkpoint": args.ae_ckpt,
            "found": ae_existing,
            "searched": candidates,
        },
        "text_model": {
            "best_checkpoint": best,
            "best_exists": os.path.isfile(best),
            "vocab": vocab,
            "vocab_exists": os.path.isfile(vocab),
        },
    }

    data_ready = (
        report["data"]["cad_json_exists"]
        and report["data"]["cad_vec_exists"]
        and report["data"]["split_exists"]
    )
    annotations_ready = report["annotations"]["exists"]
    ae_ready = bool(ae_existing)
    neural_weights_ready = report["text_model"]["best_exists"] and report["text_model"]["vocab_exists"]
    report["stage"] = {
        "can_build_annotations": report["data"]["cad_json_exists"] and report["data"]["split_exists"],
        "can_train_here": data_ready and annotations_ready and ae_ready and cuda_available,
        "neural_weights_ready": neural_weights_ready,
        "can_run_neural_here": neural_weights_ready and ae_ready and cuda_available,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
