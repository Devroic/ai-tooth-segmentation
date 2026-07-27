"""Train the multi-class per-tooth (32 FDI classes) segmentation model.

Tooth-group labels are derived from the predicted FDI code at inference time
(tooth_seg.taxonomy), not trained as a separate head.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = str((Path(__file__).resolve().parents[2] / "outputs" / "runs"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/multiclass/data.yaml")
    # CPU-only training; raise if run on a GPU.
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--name", default="multiclass_seg")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        patience=args.patience,
        project=PROJECT_DIR,
        name=args.name,
        resume=args.resume,
        workers=4,
        seed=0,
    )


if __name__ == "__main__":
    main()
