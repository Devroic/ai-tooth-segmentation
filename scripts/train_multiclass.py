"""Objective 3: train the multi-class per-tooth (32 FDI classes) instance
segmentation model. Tooth-group labels (incisor/canine/premolar/molar) are
derived from the predicted FDI code at inference time (tooth_seg.taxonomy),
not trained as a separate head.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# Binary timing test: ~34 min/epoch for 4373 images at imgsz=512. Multiclass
# has fewer images (2649 train) but a harder 32-way task and larger imgsz, so
# epoch budget is kept modest to fit a CPU-only training run in a practical
# amount of wall-clock time; raise --epochs if run on a GPU.
PROJECT_DIR = str((Path(__file__).resolve().parent.parent / "outputs" / "runs"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/multiclass/data.yaml")
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
