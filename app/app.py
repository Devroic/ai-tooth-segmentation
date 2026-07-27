"""Gradio dev tool for the trained models. Run: app/app.py"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from tooth_seg.inference.pipeline import ToothSegPipeline, detections_to_table

DEFAULT_BINARY_WEIGHTS = "models/binary_seg/best.pt"
DEFAULT_MULTICLASS_WEIGHTS = "models/multiclass_seg/best.pt"

_pipeline: ToothSegPipeline | None = None
_pipeline_key: tuple[str, str] | None = None


def get_pipeline(binary_weights: str, multiclass_weights: str) -> ToothSegPipeline:
    """Rebuilds only when the weight paths actually change."""
    global _pipeline, _pipeline_key
    key = (binary_weights, multiclass_weights)
    if _pipeline is None or _pipeline_key != key:
        bw = binary_weights if Path(binary_weights).exists() else None
        mw = multiclass_weights if Path(multiclass_weights).exists() else None
        _pipeline = ToothSegPipeline(binary_weights=bw, multiclass_weights=mw)
        _pipeline_key = key
    return _pipeline


def run_inference(image: np.ndarray, conf: float, binary_weights: str, multiclass_weights: str):
    if image is None:
        return None, None, pd.DataFrame(columns=["FDI", "Tooth", "Group", "Confidence"]), "Upload an image first."

    pipeline = get_pipeline(binary_weights, multiclass_weights)
    result = pipeline.run(image, conf=conf)

    binary_out = Image.fromarray(result.binary_overlay) if result.binary_overlay is not None else None
    multiclass_out = Image.fromarray(result.multiclass_overlay) if result.multiclass_overlay is not None else None
    table = pd.DataFrame(detections_to_table(result.detections))
    if table.empty:
        table = pd.DataFrame(columns=["FDI", "Tooth", "Group", "Confidence"])

    n_binary_px = int(result.binary_mask.sum()) if result.binary_mask is not None else 0
    status = f"Detected {len(result.detections)} teeth."
    if result.binary_mask is not None:
        status += f" Binary tooth-region pixels: {n_binary_px}."
    if pipeline.binary_model is None:
        status += " [binary model weights not found - train it first]"
    if pipeline.multiclass_model is None:
        status += " [multiclass model weights not found - train it first]"

    return binary_out, multiclass_out, table, status


def build_app(binary_weights: str, multiclass_weights: str) -> gr.Blocks:
    with gr.Blocks(title="Tooth Segmentation") as demo:
        gr.Markdown(
            "# Panoramic Radiograph Tooth Segmentation\n"
            "Upload a panoramic dental radiograph. The **binary model** highlights "
            "all tooth pixels; the **multi-class model** segments and numbers each "
            "individual tooth (FDI notation) and colors it by tooth group."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="numpy", label="Panoramic radiograph")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                with gr.Accordion("Model weights (advanced)", open=False):
                    binary_w = gr.Textbox(value=binary_weights, label="Binary model weights path")
                    multiclass_w = gr.Textbox(value=multiclass_weights, label="Multiclass model weights path")
                run_btn = gr.Button("Run segmentation", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    binary_out = gr.Image(label="Binary tooth mask")
                    multiclass_out = gr.Image(label="Per-tooth FDI segmentation")
                table_out = gr.Dataframe(headers=["FDI", "Tooth", "Group", "Confidence"], label="Detected teeth")

        run_btn.click(
            run_inference,
            inputs=[image_in, conf, binary_w, multiclass_w],
            outputs=[binary_out, multiclass_out, table_out, status],
        )
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary-weights", default=DEFAULT_BINARY_WEIGHTS)
    ap.add_argument("--multiclass-weights", default=DEFAULT_MULTICLASS_WEIGHTS)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    app = build_app(args.binary_weights, args.multiclass_weights)
    app.launch(share=args.share)
