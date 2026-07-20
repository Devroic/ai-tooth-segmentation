"""Objective 2: train the binary tooth / non-tooth segmentation model.

Uses YOLOv8n-seg (smallest Ultralytics segmentation model) since training
runs on CPU only (no dedicated GPU on this machine) - defaults are chosen to
keep an epoch tractable; override via CLI flags on faster hardware.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# A 1-epoch timing test (4373 train images, imgsz=512, batch=8, CPU) took
# ~34 minutes and already reached mask mAP50=0.71 thanks to COCO-pretrained
# transfer learning on this comparatively easy "segment the blob" task, so
# the default epoch budget here is deliberately small.
PROJECT_DIR = str((Path(__file__).resolve().parents[2] / "outputs" / "runs"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/binary/data.yaml")
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
