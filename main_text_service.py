"""CPU-safe Text-to-CAD API.

This service intentionally does not import or initialize the CUDA-only DeepCAD
AE/LGAN trainers. It reuses the existing cadlib/OpenCASCADE environment and is
therefore suitable for validating the deterministic Text-to-CAD mapping on
macOS through Docker Desktop.
"""

import os
import sys
import uuid

# The published DeepCAD image has historically kept project modules under /app.
# Add a few common roots so this overlay service can reuse cadlib regardless of
# whether the image exposes it as /app/cadlib or /app/DeepCAD/cadlib.
for candidate in (
    "/workspace",
    "/app",
    "/app/CadGen",
    "/app/DeepCAD",
    "/app/DeepCAD/CadGen",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD
from text_to_cad import (
    TextCADParseError,
    parse_text_description,
    text_to_deepcad_json,
)

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer


app = FastAPI(
    title="DeepCAD Text-to-CAD Service",
    description=(
        "CPU-safe deterministic Text-to-CAD mapping for validating natural "
        "language -> DeepCAD engineering history -> OpenCASCADE -> STEP."
    ),
)

OUTPUT_DIR = os.environ.get("TEXT2CAD_OUTPUT_DIR", "/workspace/generated_steps")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def write_step(shape, output_path: str) -> str:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(output_path)
    if status != IFSelect_RetDone:
        raise RuntimeError("STEP write failed, status {}".format(status))
    return output_path


@app.get("/")
def root():
    return {
        "message": "DeepCAD Text-to-CAD service is running.",
        "docs": "/docs",
        "json": "/text_to_cad/json",
        "step": "/text_to_cad",
        "neural_models_loaded": False,
    }


@app.get("/health")
def health():
    return {"status": "ok", "mode": "deterministic-text-to-cad", "cuda_required": False}


@app.post("/text_to_cad/json")
def text_to_cad_json(request: TextCADRequest):
    try:
        spec = parse_text_description(request.description)
        data = text_to_deepcad_json(request.description)
    except TextCADParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

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
    try:
        data = text_to_deepcad_json(request.description)
        cad_seq = CADSequence.from_dict(data)
        shape = create_CAD(cad_seq)
    except TextCADParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail="CAD creation error: {}".format(exc))

    filename = "textcad_{}.step".format(uuid.uuid4().hex[:8])
    output_path = os.path.join(OUTPUT_DIR, filename)
    try:
        write_step(shape, output_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="STEP export error: {}".format(exc))

    return FileResponse(output_path, filename=filename, media_type="application/x-step")
