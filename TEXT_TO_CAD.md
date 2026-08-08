# Text-to-CAD extension for Service_DeepCAD

## Goal

The extension introduces natural-language mapping while preserving the original
DeepCAD parametric representation. The result remains an editable engineering
history based on sketches, extrusion parameters and boolean operations instead
of a triangle mesh.

## Methods

### A. Deterministic constrained mapping (baseline)

```text
text -> engineering parser -> DeepCAD JSON -> CADSequence -> OpenCASCADE -> STEP
```

Supported baseline descriptions include cylinders, tubes, rectangular
blocks/plates and centered circular holes in Russian and English. This path is
CPU-safe and is the reproducible baseline for the thesis.

### B. Neural Text -> DeepCAD mapping

```text
text
  -> compositional text tokenizer
  -> Transformer Text Encoder
  -> predicted 256-D DeepCAD latent z
  -> frozen pretrained DeepCAD decoder
  -> Line / Arc / Circle / SOL / Ext / EOS
  -> CADSequence
  -> OpenCASCADE
  -> STEP
```

Only the Text Encoder is optimized. The previous DeepCAD decoder remains the
shared CAD generator, so the research variable is the input mapping rather than
a different CAD representation.

Numbers are tokenized compositionally. For example `-12.5` becomes a numeric
marker plus sign/digit/decimal tokens. Therefore dimensions do not have to occur
verbatim during training to be representable.

## Training objective

The text encoder is optimized with both latent-space and decoded CAD losses:

```text
L = 1.00 * L_latent + 0.10 * L_command + 0.25 * L_arguments

L_latent = MSE(z_text, z_cad)
         + 0.1 * (1 - cosine_similarity(z_text, z_cad))
```

`L_command` is cross entropy over DeepCAD command classes and `L_arguments` is
cross entropy only over arguments that are valid for the ground-truth command
according to `CMD_ARGS_MASK`. Gradients pass through the frozen DeepCAD decoder
to the Text Encoder, but decoder weights are not updated.

This is stronger than latent MSE alone because the learned representation is
explicitly pressured to decode into the correct engineering history.

## Data

### Generated reproducible captions

`build_text2cad_annotations.py` reads the original DeepCAD JSON histories and
keeps the original train/validation/test split. It can generate multiple wording
variants per CAD model without changing geometry or dimensions.

```bash
python3 build_text2cad_annotations.py \
  --data-root data \
  --output data/text2cad_annotations.jsonl \
  --language ru \
  --variants 3
```

### Official Text2CAD captions

`import_text2cad_annotations.py` supports the official Text2CAD CSV format using
its shared DeepCAD `uid` and annotation columns such as `abstract`, `beginner`,
`intermediate`, `expert` and `description`.

The external dataset is not downloaded automatically because its own license
must be accepted by the user. After downloading the CSV:

```bash
python3 import_text2cad_annotations.py \
  --csv /path/to/text2cad_v1.1.csv \
  --split-json data/train_val_test_split.json \
  --output data/text2cad_official.jsonl
```

The generated Russian captions and official captions can then be merged with
`merge_text2cad_annotations.py`.

## Training

Neural training uses the historical CUDA DeepCAD runtime and a frozen AE:

```bash
python train_text2cad.py \
  --annotations data/text2cad_annotations.jsonl \
  --data-root data \
  --ae-exp-name DeepCAD_Optimized \
  --ae-ckpt 500 \
  --output proj_log/Text2CAD
```

Saved artifacts:

- `vocab.json`;
- `metrics.jsonl`;
- `latest.pth`;
- `best.pth`.

`best.pth` is selected on validation loss. Training uses gradient clipping,
ReduceLROnPlateau and early stopping, and can resume from `latest.pth`.

## Evaluation

`evaluate_text2cad.py` evaluates held-out annotations and excludes EOS padding
from command accuracy. It reports:

- latent MSE and cosine similarity;
- command accuracy;
- valid argument accuracy;
- exact command sequence accuracy;
- CADSequence validity;
- invalidity ratio;
- optional OpenCASCADE solid validity;
- inference time.

Example:

```bash
python evaluate_text2cad.py \
  --annotations data/text2cad_annotations.jsonl \
  --data-root data \
  --checkpoint proj_log/Text2CAD/best.pth \
  --split test \
  --geometry
```

## APIs

### macOS / CPU baseline

`main_text_service.py`:

- `GET /health`;
- `POST /text_to_cad/json`;
- `POST /text_to_cad`.

Run with:

```bash
docker compose -f docker-compose.mac.yml up --build
```

Swagger: `http://localhost:8000/docs`.

### Neural CUDA API

`main_neural_text_service.py`:

- `GET /health`;
- `POST /text_to_cad/neural/vector`;
- `POST /text_to_cad/neural`.

The service loads `best.pth`, `vocab.json` and the existing DeepCAD AE
checkpoint. If an artifact is absent, the API starts in `not_ready` mode and
reports the exact loading error through `/health`.

Run on Linux + NVIDIA GPU:

```bash
docker compose -f docker-compose.gpu.yml up --build neural
```

Swagger: `http://localhost:8001/docs`.

## One-command workflow

```bash
bash text2cad_pipeline.sh diagnose
bash text2cad_pipeline.sh prepare
bash text2cad_pipeline.sh train
bash text2cad_pipeline.sh eval
bash text2cad_pipeline.sh serve-neural
```

For a full operational checklist see `RUN_TEXT2CAD.md`.

## Thesis experiment design

The main controlled comparison is:

1. deterministic constrained text mapping;
2. neural Text Transformer -> DeepCAD latent -> frozen DeepCAD decoder;
3. optional direct text-to-command decoder as a later ablation.

Keep the CAD decoder and geometry backend fixed across the first two methods.
Report quantitative results on the held-out split and separately show qualitative
STEP examples. The original DeepCAD COV/MMD/JSD values remain useful for
continuity with the previous work, while Text-to-CAD additionally requires
sequence accuracy, invalidity and text-conditioned geometric evaluation.
