# Service_DeepCAD: reproducible Text-to-CAD workflow

This file is the operational checklist for the Text-to-CAD thesis extension.
The deterministic macOS service and the neural CUDA service are intentionally
separate so the original DeepCAD CUDA stack does not block local development.

## 0. Current architecture

Deterministic baseline:

```text
text -> constrained engineering parser -> DeepCAD JSON -> CADSequence
     -> OpenCASCADE -> STEP
```

Neural method:

```text
text -> Text Transformer -> DeepCAD latent z -> frozen DeepCAD decoder
     -> Line/Arc/Circle/SOL/Ext/EOS -> CADSequence -> OpenCASCADE -> STEP
```

The old DeepCAD Autoencoder is NOT retrained by the Text-to-CAD training script.
Only the new text encoder receives optimizer updates.

## 1. Pull the research branch

```bash
git switch agent/text-to-cad
git pull origin agent/text-to-cad
```

Do not merge PR #1 before neural training/evaluation is completed.

## 2. macOS deterministic baseline

```bash
docker compose -f docker-compose.mac.yml up --build
```

Open:

```text
http://localhost:8000/docs
```

Important endpoints:

- `POST /text_to_cad/json` - inspect parsed parameters and DeepCAD history;
- `POST /text_to_cad` - create a STEP file.

This service does not load neural weights and does not require CUDA.

## 3. DeepCAD training data

The neural mapper needs the original DeepCAD data layout:

```text
data/
  train_val_test_split.json
  cad_json/
    0000/....json
  cad_vec/
    0000/....h5
```

`cad_json` is used for generating captions. `cad_vec` is used as the CAD target
for the frozen DeepCAD encoder.

Check readiness:

```bash
bash text2cad_pipeline.sh diagnose
```

## 4A. Reproducible Russian training captions

```bash
bash text2cad_pipeline.sh prepare
```

This creates three wording variants for each DeepCAD object while keeping the
same train/validation/test split and exactly the same geometric parameters.

Result:

```text
data/text2cad_annotations.jsonl
```

## 4B. Recommended final dataset: generated Russian + official Text2CAD captions

The official Text2CAD dataset is distributed separately under its own license.
After accepting the dataset terms and downloading `text2cad_v1.0.csv` or
`text2cad_v1.1.csv`, run:

```bash
bash text2cad_pipeline.sh prepare-official /path/to/text2cad_v1.1.csv
```

This performs three operations:

1. creates Russian DeepCAD-history captions;
2. converts official `abstract/beginner/intermediate/expert/description`
   captions to this project's JSONL format using the shared DeepCAD UID;
3. merges/de-duplicates the two sources into
   `data/text2cad_annotations.jsonl`.

No dataset or checkpoint is committed to Git.

## 5. Neural training environment

Training requires Linux + NVIDIA GPU + NVIDIA Container Toolkit. Check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

The project reuses `tiranousor/deepcad:latest` and automatically detects the
Python environment inside that image containing `numpy + h5py + torch + OCC`.

Build the GPU overlay once:

```bash
docker compose -f docker-compose.gpu.yml build
```

## 6. Train Text Transformer

```bash
bash text2cad_pipeline.sh train
```

Default research configuration:

- frozen DeepCAD AE checkpoint: `DeepCAD_Optimized / 500`;
- text encoder: 4 Transformer encoder layers;
- latent dimension: 256;
- batch size: 32;
- learning rate: 3e-4;
- maximum epochs: 80;
- early stopping patience: 10.

Training objective:

```text
L = 1.00 * latent_alignment
  + 0.10 * command_cross_entropy
  + 0.25 * argument_cross_entropy
```

`latent_alignment` is MSE + cosine distance between text-predicted and
CAD-encoded latent vectors. Command/argument losses are computed after passing
the text latent through the frozen DeepCAD decoder. Thus the text encoder is
optimized for the actual engineering command history, not just latent distance.

Outputs:

```text
proj_log/Text2CAD/
  vocab.json
  metrics.jsonl
  latest.pth
  best.pth
```

`best.pth` is selected using validation loss. Learning rate is reduced on a
plateau and training stops early after sustained non-improvement.

Resume interrupted training:

```bash
docker compose -f docker-compose.gpu.yml --profile train run --rm trainer \
  /usr/local/bin/deepcad-python train_text2cad.py \
  --annotations /workspace/data/text2cad_annotations.jsonl \
  --data-root /workspace/data \
  --ae-proj-dir /app/proj_log \
  --ae-exp-name DeepCAD_Optimized \
  --ae-ckpt 500 \
  --output /workspace/proj_log/Text2CAD \
  --epochs 80 \
  --batch-size 32 \
  --resume /workspace/proj_log/Text2CAD/latest.pth
```

## 7. Held-out evaluation

Do not choose the model from the test split. `best.pth` is selected on
validation; the test split is used only after training.

```bash
bash text2cad_pipeline.sh eval
```

Report:

```text
proj_log/Text2CAD/evaluation.json
```

Metrics include:

- latent MSE;
- latent cosine similarity;
- command accuracy up to the first EOS (padding is excluded);
- argument accuracy only on arguments valid for each CAD command;
- exact command-sequence accuracy;
- valid CAD sequence rate;
- invalidity ratio;
- valid OpenCASCADE solid rate;
- inference time per sample.

For thesis comparison, keep the original DeepCAD COV/MMD/JSD results from the
previous work and report the new Text-to-CAD sequence/validity metrics alongside
them. Chamfer Distance can be added using the existing DeepCAD evaluation path
once generated test solids/point clouds are collected.

## 8. Start neural API

After `best.pth` exists:

```bash
bash text2cad_pipeline.sh serve-neural
```

Open:

```text
http://localhost:8001/docs
```

Endpoints:

- `GET /health` - CUDA/checkpoint/AE readiness;
- `POST /text_to_cad/neural/vector` - text -> neural DeepCAD command sequence;
- `POST /text_to_cad/neural` - text -> neural CAD -> STEP.

If weights are missing, the API still starts and `/health` explains exactly
which artifact is missing instead of crashing at import time.

## 9. Final comparison for the thesis

Use the same prompts/test objects to compare:

1. deterministic constrained Text-to-CAD baseline;
2. neural Text Transformer -> DeepCAD latent -> DeepCAD decoder;
3. optional direct text-to-command model as an ablation later.

Record model checkpoint, dataset source, split, seed and all metrics. Keep sample
STEP files/screenshots for qualitative examples, but do not use hand-picked
examples as the only evaluation.

## 10. Merge gate

Only merge PR #1 into `main.java` after all of the following are true:

- deterministic macOS API passes;
- `data/text2cad_annotations.jsonl` is reproducibly generated;
- neural training produces `best.pth` and `vocab.json`;
- held-out `evaluation.json` is saved and reviewed;
- `/text_to_cad/neural/vector` returns valid engineering command histories for
  representative prompts;
- `/text_to_cad/neural` creates STEP files that OpenCASCADE accepts;
- final thesis metrics and experiment configuration are recorded.
