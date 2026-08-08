import os
import sys
import uuid
import json
import torch

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD

from trainer.trainerAE import TrainerAE
from trainer.trainerLGAN import TrainerLatentWGAN
from config.configAE import ConfigAE
from config.configLGAN import ConfigLGAN
from text_to_cad import text_to_deepcad_json, parse_text_description, TextCADParseError

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone


app = FastAPI(
    title="DeepCAD Generative Service",
    description="DeepCAD service with JSON, random latent generation and Text-to-CAD mapping.",
)


tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'generated_steps'))
os.makedirs(tmp_dir, exist_ok=True)


class TextCADRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=3,
        examples=[
            "Цилиндр диаметром 20 мм высотой 35 мм",
            "Труба внешний диаметр 30 мм, внутренний диаметр 20 мм, длина 50 мм",
            "Пластина 60x40x5 мм с отверстием диаметром 10 мм",
        ],
    )


cfg_ae = ConfigAE(phase="test")
cfg_ae.exp_name = "DeepCAD_Optimized"
cfg_ae.exp_dir = os.path.join(cfg_ae.proj_dir, cfg_ae.exp_name)
cfg_ae.model_dir = os.path.join(cfg_ae.exp_dir, "model")
ae_trainer = TrainerAE(cfg_ae)
ae_trainer.load_ckpt("500")
ae_trainer.net.eval()

cfg_lgan = ConfigLGAN()
cfg_lgan.exp_name = "DeepCAD_Optimized"
cfg_lgan.ae_ckpt = "500"
lgan_trainer = TrainerLatentWGAN(cfg_lgan)
lgan_trainer.load_ckpt("200000")
lgan_trainer.netG.eval()


def write_step(shape, output_path: str) -> str:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(output_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed, status {status}")
    return output_path


def decode_latent(z: torch.Tensor) -> CADSequence:
    if z.dim() == 2:
        z = z.unsqueeze(1)
    with torch.no_grad():
        outputs = ae_trainer.decode(z)
        tokens = ae_trainer.logits2vec(outputs, refill_pad=True, to_numpy=True)[0]
    cad_seq = CADSequence.from_vector(tokens, is_numerical=True, n=256)
    return cad_seq


def deepcad_json_to_step(data, prefix: str) -> FileResponse:
    """Build a valid DeepCAD parametric solid and export it as STEP."""
    try:
        cad_seq = CADSequence.from_dict(data)
        # Keep physical dimensions for Text-to-CAD. Normalization is only needed
        # for neural-network vectorization and is not required by create_CAD().
        shape = create_CAD(cad_seq)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CAD creation error: {e}")

    filename = f"{prefix}_{uuid.uuid4().hex[:6]}.step"
    out_path = os.path.join(tmp_dir, filename)
    try:
        write_step(shape, out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STEP export error: {e}")

    return FileResponse(out_path, filename=filename, media_type="application/x-step")


@app.get("/")
def root():
    return {
        "message": "DeepCAD service is running.",
        "text_to_cad": "/text_to_cad",
        "text_to_cad_json": "/text_to_cad/json",
    }


@app.post("/text_to_cad/json")
def text_to_cad_json(request: TextCADRequest):
    """Return the generated engineering history in DeepCAD-compatible JSON."""
    try:
        spec = parse_text_description(request.description)
        data = text_to_deepcad_json(request.description)
    except TextCADParseError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "input": request.description,
        "parsed": {
            "primitive": spec.primitive,
            "length": spec.length,
            "width": spec.width,
            "height": spec.height,
            "radius": spec.radius,
            "inner_radius": spec.inner_radius,
            "hole_radius": spec.hole_radius,
        },
        "deepcad": data,
    }


@app.post("/text_to_cad", response_class=FileResponse)
def text_to_cad(request: TextCADRequest):
    """Natural-language engineering description -> DeepCAD history -> STEP."""
    try:
        data = text_to_deepcad_json(request.description)
    except TextCADParseError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return deepcad_json_to_step(data, "textcad")


@app.get("/generate_random")
def generate_random(checkpoint: str = Query("200000", description="GAN checkpoint id")):
    max_attempts = 10
    for _ in range(max_attempts):
        z = torch.randn(1, cfg_lgan.n_dim).cuda()
        with torch.no_grad():
            z_fake = lgan_trainer.netG(z)
        try:
            cad_seq = decode_latent(z_fake)
            if not cad_seq.seq:
                continue
            shape = create_CAD(cad_seq)
            filename = f"gen_{uuid.uuid4().hex[:6]}.step"
            out_path = os.path.join(tmp_dir, filename)
            write_step(shape, out_path)
            return FileResponse(out_path, filename=filename, media_type="application/x-step")
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Failed to generate a valid shape after multiple attempts.")


@app.post("/convert", response_class=FileResponse)
async def convert_cad(file: UploadFile = File(...)):
    try:
        data = json.loads((await file.read()).decode('utf-8'))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CAD creation error: {e}")

    filename = f"conv_{uuid.uuid4().hex[:6]}.step"
    out_path = os.path.join(tmp_dir, filename)

    try:
        write_step(shape, out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STEP export error: {e}")

    return FileResponse(out_path, filename=filename, media_type="application/x-step")


app.mount(
    "/steps",
    StaticFiles(directory=tmp_dir, html=False),
    name="steps",
)


@app.get("/generate_random_batch", response_class=FileResponse)
def generate_random_batch(
    min_size_kb: int = Query(80, ge=1),
):
    max_attempts = 20
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        z = torch.randn(1, cfg_lgan.n_dim).cuda()
        with torch.no_grad():
            latent = lgan_trainer.netG(z)

        try:
            cad_seq = decode_latent(latent)
            if not cad_seq.seq:
                continue
            shape = create_CAD(cad_seq)
            filename = f"batch_{uuid.uuid4().hex[:6]}.step"
            path = os.path.join(tmp_dir, filename)
            write_step(shape, path)

            size_kb = os.path.getsize(path) / 1024
            if size_kb >= min_size_kb:
                return FileResponse(path, filename=filename, media_type="application/x-step")
            os.remove(path)
        except Exception:
            continue

    raise HTTPException(
        status_code=500,
        detail=f"Failed to generate a file ≥ {min_size_kb}KB after {max_attempts} attempts.",
    )
