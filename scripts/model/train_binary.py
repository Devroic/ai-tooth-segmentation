"""Train the binary tooth / non-tooth segmentation model."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = str((Path(__file__).resolve().parents[2] / "outputs" / "runs"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/binary/data.yaml")
    # CPU-only training; ~34min/epoch at these defaults, mAP50=0.71 after epoch 1.
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--name", default="binary_seg")
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
        single_cls=True,
        workers=4,
        seed=0,
    )


if __name__ == "__main__":
    main()
