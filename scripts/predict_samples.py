"""Run the trained binary + multiclass models over a handful of test-split
images and save side-by-side visualizations, for spot-checking / the thesis
report figures.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from tooth_seg.inference.pipeline import ToothSegPipeline, detections_to_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary-weights", default="outputs/runs/binary_seg/weights/best.pt")
    ap.add_argument("--multiclass-weights", default="outputs/runs/multiclass_seg/weights/best.pt")
    ap.add_argument("--test-dir", default="outputs/data/multiclass/test/images")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="outputs/reports/sample_predictions")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary_w = args.binary_weights if Path(args.binary_weights).exists() else None
    multiclass_w = args.multiclass_weights if Path(args.multiclass_weights).exists() else None
    pipeline = ToothSegPipeline(binary_weights=binary_w, multiclass_weights=multiclass_w)

    images = sorted(Path(args.test_dir).glob("*.jpg"))
    random.Random(args.seed).shuffle(images)
    images = images[:args.n]

    for img_path in images:
        image = np.array(Image.open(img_path).convert("RGB"))
        result = pipeline.run(image, conf=args.conf)

        panels = [image]
        if result.binary_overlay is not None:
            panels.append(result.binary_overlay)
        if result.multiclass_overlay is not None:
            panels.append(result.multiclass_overlay)

        h = max(p.shape[0] for p in panels)
        resized = []
        for p in panels:
            scale = h / p.shape[0]
            w = int(p.shape[1] * scale)
            resized.append(np.array(Image.fromarray(p).resize((w, h))))
        combined = np.concatenate(resized, axis=1)

        out_path = out_dir / f"{img_path.stem}.jpg"
        Image.fromarray(combined).save(out_path)
        table = detections_to_table(result.detections)
        print(f"{img_path.name}: {len(table)} teeth detected -> {out_path}")


if __name__ == "__main__":
    main()
