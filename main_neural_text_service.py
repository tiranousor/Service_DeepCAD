"""GPU Text-to-CAD API using trained TextLatentEncoder + frozen DeepCAD decoder.

The service starts even when weights are not mounted. In that case /health
reports the loading error and neural endpoints return HTTP 503. This makes the
Docker deployment easy to diagnose before a training run has produced best.pth.
"""

from __future__ import annotations

import os
import sys
import uuid

for candidate in (
    "/workspace",
    "/app",
    "/app/CadGen",
    "/app/DeepCAD",
    "/app/DeepCAD/CadGen",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cadlib.extrude import CADSequence
from cadlib.macro import ALL_COMMANDS, EOS_IDX
from cadlib.visualize import create_CAD
from config.configAE import ConfigAE
from trainer.trainerAE import TrainerAE
from text2cad_ml import TextVocabulary, TextLatentEncoder

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer


TEXT_CHECKPOINT = os.environ.get("TEXT2CAD_CKPT", "/workspace/proj_log/Text2CAD/best.pth")
TEXT_VOCAB = os.environ.get("TEXT2CAD_VOCAB", "")
DEEPCAD_PROJ_DIR = os.environ.get("DEEPCAD_PROJ_DIR", "/app/proj_log")
DEEPCAD_AE_EXP_NAME = os.environ.get("DEEPCAD_AE_EXP_NAME", "DeepCAD_Optimized")
DEEPCAD_AE_CKPT = os.environ.get("DEEPCAD_AE_CKPT", "500")
OUTPUT_DIR = os.environ.get("TEXT2CAD_OUTPUT_DIR", "/workspace/generated_steps")
os.makedirs(OUTPUT_DIR, exist_ok=True)


app = FastAPI(
    title="DeepCAD Neural Text-to-CAD Service",
    description=(
        "Trainable natural-language -> Text Transformer -> DeepCAD latent -> "
        "frozen DeepCAD decoder -> engineering CAD commands -> STEP."
    ),
)


class TextCADRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=3,
        examples=[
            "Создай цилиндр диаметром 20 мм и высотой 35 мм",
            "Построй прямоугольную пластину с круглым отверстием по центру",
        ],
    )


text_model = None
vocab = None
ae_trainer = None
model_checkpoint = None
model_load_error = None


def _load_autoencoder():
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], "-m", "dec"]
        cfg = ConfigAE("test")
    finally:
        sys.argv = saved_argv

    cfg.proj_dir = DEEPCAD_PROJ_DIR
    cfg.exp_name = DEEPCAD_AE_EXP_NAME
    cfg.exp_dir = os.path.join(cfg.proj_dir, cfg.exp_name)
    cfg.model_dir = os.path.join(cfg.exp_dir, "model")

    trainer = TrainerAE(cfg)
    trainer.load_ckpt(DEEPCAD_AE_CKPT)
    trainer.net.eval()
    for parameter in trainer.net.parameters():
        parameter.requires_grad = False
    return cfg, trainer


def load_models():
    global text_model, vocab, ae_trainer, model_checkpoint, model_load_error
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; neural Text-to-CAD requires NVIDIA GPU")
        if not os.path.exists(TEXT_CHECKPOINT):
            raise RuntimeError("Text2CAD checkpoint not found: {}".format(TEXT_CHECKPOINT))

        checkpoint = torch.load(TEXT_CHECKPOINT)
        vocab_path = TEXT_VOCAB or os.path.join(
            os.path.dirname(TEXT_CHECKPOINT), checkpoint.get("vocab_path", "vocab.json")
        )
        if not os.path.exists(vocab_path):
            raise RuntimeError("Text2CAD vocabulary not found: {}".format(vocab_path))

        loaded_vocab = TextVocabulary.load(vocab_path)
        model_config = checkpoint.get("model_config", {})
        loaded_model = TextLatentEncoder(
            vocab_size=len(loaded_vocab),
            dim_z=int(checkpoint.get("dim_z", 256)),
            d_model=int(model_config.get("d_model", 256)),
            nhead=int(model_config.get("nhead", 8)),
            num_layers=int(model_config.get("num_layers", 4)),
            dim_feedforward=int(model_config.get("dim_feedforward", 512)),
            max_len=int(checkpoint.get("max_text_len", 128)),
            dropout=float(model_config.get("dropout", 0.1)),
        ).cuda()
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.eval()

        _, loaded_ae = _load_autoencoder()

        vocab = loaded_vocab
        text_model = loaded_model
        ae_trainer = loaded_ae
        model_checkpoint = checkpoint
        model_load_error = None
        print(
            "Neural Text2CAD loaded: checkpoint={}, epoch={}, val_loss={}".format(
                TEXT_CHECKPOINT,
                checkpoint.get("epoch"),
                checkpoint.get("val_loss"),
            )
        )
    except Exception as exc:
        text_model = None
        vocab = None
        ae_trainer = None
        model_checkpoint = None
        model_load_error = str(exc)
        print("Neural Text2CAD is not ready: {}".format(model_load_error))


load_models()


def require_models():
    if text_model is None or vocab is None or ae_trainer is None:
        raise HTTPException(
            status_code=503,
            detail="Neural Text2CAD model is not ready: {}".format(model_load_error),
        )


def predict_vector(description):
    require_models()
    max_len = int(model_checkpoint.get("max_text_len", 128))
    token_ids = torch.tensor(
        [vocab.encode(description, max_len)], dtype=torch.long, device="cuda"
    )
    with torch.no_grad():
        z = text_model(token_ids)
        decoder_z = z.unsqueeze(1) if z.dim() == 2 else z
        outputs = ae_trainer.decode(decoder_z)
        vector = ae_trainer.logits2vec(outputs, refill_pad=True, to_numpy=True)[0]
    return vector, z.detach().cpu().numpy()[0]


def vector_summary(vector):
    result = []
    for index, row in enumerate(vector):
        command_index = int(row[0])
        if command_index < 0 or command_index >= len(ALL_COMMANDS):
            command_name = "INVALID"
        else:
            command_name = ALL_COMMANDS[command_index]
        result.append(
            {
                "index": index,
                "command": command_name,
                "command_index": command_index,
                "args": [int(x) for x in row[1:]],
            }
        )
        if command_index == EOS_IDX:
            break
    return result


def write_step(shape, output_path):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(output_path)
    if status != IFSelect_RetDone:
        raise RuntimeError("STEP write failed, status {}".format(status))
    return output_path


@app.get("/")
def root():
    return {
        "message": "DeepCAD neural Text-to-CAD service",
        "ready": model_load_error is None,
        "docs": "/docs",
        "health": "/health",
        "vector": "/text_to_cad/neural/vector",
        "step": "/text_to_cad/neural",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model_load_error is None else "not_ready",
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": TEXT_CHECKPOINT,
        "checkpoint_exists": os.path.exists(TEXT_CHECKPOINT),
        "ae_project_dir": DEEPCAD_PROJ_DIR,
        "ae_experiment": DEEPCAD_AE_EXP_NAME,
        "ae_checkpoint": DEEPCAD_AE_CKPT,
        "model_epoch": None if model_checkpoint is None else model_checkpoint.get("epoch"),
        "val_loss": None if model_checkpoint is None else model_checkpoint.get("val_loss"),
        "error": model_load_error,
    }


@app.post("/text_to_cad/neural/vector")
def neural_vector(request: TextCADRequest):
    try:
        vector, z = predict_vector(request.description)
        cad_seq = CADSequence.from_vector(vector, is_numerical=True, n=256)
        if not cad_seq.seq:
            raise ValueError("Predicted sequence contains no extrusion operations")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Neural CAD decoding error: {}".format(exc))

    return {
        "input": request.description,
        "latent_dim": len(z),
        "commands": vector_summary(vector),
    }


@app.post("/text_to_cad/neural", response_class=FileResponse)
def neural_step(request: TextCADRequest):
    try:
        vector, _ = predict_vector(request.description)
        cad_seq = CADSequence.from_vector(vector, is_numerical=True, n=256)
        if not cad_seq.seq:
            raise ValueError("Predicted sequence contains no extrusion operations")
        shape = create_CAD(cad_seq)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Neural CAD creation error: {}".format(exc))

    filename = "neural_textcad_{}.step".format(uuid.uuid4().hex[:8])
    output_path = os.path.join(OUTPUT_DIR, filename)
    try:
        write_step(shape, output_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="STEP export error: {}".format(exc))
    return FileResponse(output_path, filename=filename, media_type="application/x-step")
