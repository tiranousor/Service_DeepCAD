#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-help}"
ARG2="${2:-}"

case "$COMMAND" in
  diagnose)
    python3 check_text2cad_ready.py
    ;;

  mac)
    docker compose -f docker-compose.mac.yml up --build
    ;;

  prepare)
    python3 build_text2cad_annotations.py \
      --data-root data \
      --output data/text2cad_annotations.jsonl \
      --language ru \
      --variants 3
    python3 check_text2cad_ready.py
    ;;

  prepare-bilingual)
    python3 build_text2cad_annotations.py \
      --data-root data \
      --output data/text2cad_annotations.jsonl \
      --language both \
      --variants 3
    python3 check_text2cad_ready.py
    ;;

  prepare-official)
    if [ -z "$ARG2" ] || [ ! -f "$ARG2" ]; then
      echo "Usage: bash text2cad_pipeline.sh prepare-official /path/to/text2cad_v1.1.csv"
      echo "Download the official CSV only after accepting the Text2CAD dataset license."
      exit 1
    fi
    python3 build_text2cad_annotations.py \
      --data-root data \
      --output data/text2cad_generated.jsonl \
      --language ru \
      --variants 3
    python3 import_text2cad_annotations.py \
      --csv "$ARG2" \
      --split-json data/train_val_test_split.json \
      --output data/text2cad_official.jsonl
    python3 merge_text2cad_annotations.py \
      data/text2cad_generated.jsonl \
      data/text2cad_official.jsonl \
      --output data/text2cad_annotations.jsonl
    python3 check_text2cad_ready.py
    ;;

  train)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "ERROR: nvidia-smi is not available. Run neural training on Linux with an NVIDIA GPU."
      exit 1
    fi
    nvidia-smi
    docker compose -f docker-compose.gpu.yml --profile train run --rm trainer
    ;;

  eval)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "ERROR: nvidia-smi is not available. Run neural evaluation on Linux with an NVIDIA GPU."
      exit 1
    fi
    docker compose -f docker-compose.gpu.yml --profile eval run --rm evaluator
    ;;

  serve-neural)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "ERROR: nvidia-smi is not available. Neural DeepCAD inference requires Linux + NVIDIA GPU."
      exit 1
    fi
    docker compose -f docker-compose.gpu.yml up --build neural
    ;;

  stop-mac)
    docker compose -f docker-compose.mac.yml down
    ;;

  stop-neural)
    docker compose -f docker-compose.gpu.yml down
    ;;

  help|*)
    cat <<'EOF'
Text2CAD pipeline

  bash text2cad_pipeline.sh diagnose
      Check dataset/checkpoints/runtime.

  bash text2cad_pipeline.sh mac
      Run deterministic Text-to-CAD on macOS.

  bash text2cad_pipeline.sh prepare
      Build 3 Russian captions per DeepCAD model.

  bash text2cad_pipeline.sh prepare-bilingual
      Build generated Russian + English caption variants.

  bash text2cad_pipeline.sh prepare-official /path/to/text2cad_v1.1.csv
      Merge Russian generated captions with licensed official Text2CAD captions.

  bash text2cad_pipeline.sh train
      Train Text Transformer on Linux + NVIDIA GPU.

  bash text2cad_pipeline.sh eval
      Evaluate best.pth on held-out DeepCAD test split, including solid validity.

  bash text2cad_pipeline.sh serve-neural
      Start neural API at http://localhost:8001/docs.

  bash text2cad_pipeline.sh stop-mac
  bash text2cad_pipeline.sh stop-neural
EOF
    ;;
esac
