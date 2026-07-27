"""FastAPI wrapper around tooth_seg.inference.pipeline for the React frontend.

Run: uvicorn web.backend.main:app --port 8000, then open :8000.
Frontend dev (hot-reload): add --reload-dir web/backend --reload-dir src,
then `npm run dev` in web/frontend (see its README).
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from tooth_seg.inference.pipeline import ToothSegPipeline, detections_to_table

BINARY_WEIGHTS = os.environ.get("TOOTH_SEG_BINARY_WEIGHTS", "models/binary_seg/best.pt")
MULTICLASS_WEIGHTS = os.environ.get("TOOTH_SEG_MULTICLASS_WEIGHTS", "models/multiclass_seg/best.pt")
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app = FastAPI(title="Tooth Segmentation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local/clinic deployment, not public internet
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: ToothSegPipeline | None = None


def get_pipeline() -> ToothSegPipeline:
    global _pipeline
    if _pipeline is None:
        bw = BINARY_WEIGHTS if os.path.exists(BINARY_WEIGHTS) else None
        mw = MULTICLASS_WEIGHTS if os.path.exists(MULTICLASS_WEIGHTS) else None
        _pipeline = ToothSegPipeline(binary_weights=bw, multiclass_weights=mw)
    return _pipeline


def _encode_jpeg(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class HealthResponse(BaseModel):
    binary_model_loaded: bool
    multiclass_model_loaded: bool


class Detection(BaseModel):
    FDI: str
    Tooth: str
    Group: str | None
    Confidence: float


class AnalyzeResponse(BaseModel):
    binary_overlay: str | None
    multiclass_overlay: str | None
    detections: list[Detection]
    binary_tooth_pixels: int | None


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline = get_pipeline()
    return HealthResponse(
        binary_model_loaded=pipeline.binary_model is not None,
        multiclass_model_loaded=pipeline.multiclass_model is not None,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), conf: float = 0.25) -> AnalyzeResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    raw = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    pipeline = get_pipeline()
    if pipeline.binary_model is None and pipeline.multiclass_model is None:
        raise HTTPException(status_code=503, detail="No trained model weights found on the server")

    result = pipeline.run(image, conf=conf)

    return AnalyzeResponse(
        binary_overlay=_encode_jpeg(result.binary_overlay) if result.binary_overlay is not None else None,
        multiclass_overlay=_encode_jpeg(result.multiclass_overlay) if result.multiclass_overlay is not None else None,
        detections=detections_to_table(result.detections),
        binary_tooth_pixels=int(result.binary_mask.sum()) if result.binary_mask is not None else None,
    )


# Mounted last so /api/* above takes precedence.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
