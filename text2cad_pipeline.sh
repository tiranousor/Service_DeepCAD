#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-help}"

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

  bash text2cad_pipeline.sh diagnose          Check dataset/checkpoints/runtime
  bash text2cad_pipeline.sh mac               Run deterministic Text-to-CAD on macOS
  bash text2cad_pipeline.sh prepare           Build 3 Russian captions per DeepCAD model
  bash text2cad_pipeline.sh prepare-bilingual Build RU+EN caption variants
  bash text2cad_pipeline.sh train             Train Text Transformer on Linux + NVIDIA GPU
  bash text2cad_pipeline.sh eval              Evaluate best.pth on held-out test data
  bash text2cad_pipeline.sh serve-neural      Start neural API at http://localhost:8001/docs
  bash text2cad_pipeline.sh stop-mac          Stop macOS deterministic service
  bash text2cad_pipeline.sh stop-neural       Stop neural GPU service
EOF
    ;;
esac
