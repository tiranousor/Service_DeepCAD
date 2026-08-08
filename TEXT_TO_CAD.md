# Text-to-CAD extension for Service_DeepCAD

## Goal

The extension adds a new input mapping to the existing DeepCAD service:

`natural-language description -> engineering CAD representation -> STEP`

The key requirement is to preserve the original DeepCAD representation instead
of generating a mesh. The final model therefore still consists of editable CAD
operations (sketch curves and extrusion/boolean operations).

## Architecture

Two complementary Text-to-CAD methods are included.

### 1. Constrained engineering baseline

`text_to_cad.py` parses explicit engineering descriptions and compiles them to
the same JSON schema already consumed by `CADSequence.from_dict`.

Current supported descriptions:

- cylinder / disk;
- hollow cylinder / tube;
- rectangular block / plate;
- centered circular through-hole;
- Russian and English engineering wording;
- dimensions written as named values or `60x40x5`.

Examples:

```text
Цилиндр диаметром 20 мм высотой 35 мм
Труба внешний диаметр 30 мм, внутренний диаметр 20 мм, длина 50 мм
Пластина 60x40x5 мм с отверстием диаметром 10 мм
Cylinder diameter 24 mm height 12 mm
```

This method is deterministic and useful as a reproducible baseline for the
thesis. Invalid or incomplete descriptions are rejected before OpenCASCADE is
called.

### 2. Trainable Text -> DeepCAD latent mapping

`text2cad_ml.py` implements a Transformer text encoder. It predicts a vector in
the existing 256-dimensional DeepCAD latent space. The pretrained DeepCAD
autoencoder remains frozen:

```text
text
  -> Text Transformer
  -> predicted DeepCAD latent z
  -> existing DeepCAD decoder
  -> command logits / argument logits
  -> logits2vec
  -> CADSequence
  -> OpenCASCADE
  -> STEP
```

This architecture isolates the research question: only the input mapping is
changed. The CAD decoder and command vocabulary are shared with the previous
DeepCAD implementation.

The training loss is:

```text
L_text = MSE(z_text, z_cad) + 0.1 * (1 - cosine_similarity(z_text, z_cad))
```

where `z_cad` is produced by the frozen pretrained DeepCAD encoder.

## Dataset generation

`build_text2cad_annotations.py` can create paired training data directly from
the original DeepCAD JSON histories:

```bash
python build_text2cad_annotations.py \
  --data-root data \
  --output data/text2cad_annotations.jsonl \
  --language ru
```

Each row has the form:

```json
{"id":"0000/00001234","text":"...","split":"train","source":"deepcad_history_template"}
```

For experiments with more natural semantic captions, the same JSONL format can
be populated with external/manual descriptions while keeping the CAD targets
unchanged.

## Training

```bash
python train_text2cad.py \
  --annotations data/text2cad_annotations.jsonl \
  --data-root data \
  --ae-exp-name DeepCAD_Optimized \
  --ae-ckpt 500 \
  --output proj_log/Text2CAD
```

The script saves:

- `vocab.json`;
- `latest.pth`;
- `best.pth`.

The current DeepCAD implementation is CUDA-only, therefore Text-to-CAD training
uses CUDA as well.

## API

### Inspect generated DeepCAD history

`POST /text_to_cad/json`

```json
{
  "description": "Пластина 60x40x5 мм с отверстием диаметром 10 мм"
}
```

The response contains parsed engineering parameters and the exact DeepCAD JSON.
This endpoint is useful for debugging and for demonstrating interpretability at
the thesis defense.

### Generate STEP

`POST /text_to_cad`

```json
{
  "description": "Цилиндр диаметром 20 мм высотой 35 мм"
}
```

The service compiles the description into a DeepCAD construction history,
creates the solid through the existing OpenCASCADE pipeline and returns a STEP
file.

## Tests

Parser/schema tests are located in `tests/test_text_to_cad.py`:

```bash
python -m unittest tests.test_text_to_cad
```

They do not require OpenCASCADE and verify parsing, dimensions, loops and
DeepCAD-compatible operation structure.

## Recommended thesis experiments

Compare at least three systems:

1. deterministic constrained mapping (baseline);
2. Text Transformer -> DeepCAD latent -> existing decoder;
3. optionally a direct text-to-command decoder as an ablation.

Report:

- syntactic validity of generated command sequences;
- percentage of sequences successfully converted to a solid;
- command accuracy and argument accuracy where paired ground truth exists;
- Chamfer Distance / MMD / JSD on sampled generated geometry for comparability
  with the previous DeepCAD work;
- dimensional error for explicit engineering prompts;
- inference time;
- ablation of latent loss components and text encoder depth.

This gives the project a clear research contribution: multiple mappings into a
single parametric CAD representation, with both deterministic and learned
methods evaluated under the same decoder and geometry backend.
