"""Finds a data-driven confidence threshold for the multiclass model instead
of a hand-picked one, using the mean F1-vs-confidence curve Ultralytics'
validator computes on the held-out test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_DIR = str((Path(__file__).resolve().parents[2] / "outputs" / "runs"))
REPORT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/multiclass_seg/best.pt")
    ap.add_argument("--data", default="outputs/data/multiclass/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--candidates", type=float, nargs="*", default=[0.15, 0.25])
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.val(data=args.data, split="test", imgsz=args.imgsz, device="cpu",
                         project=PROJECT_DIR, name="calibrate", plots=False, verbose=False)

    px = results.seg.f1_curve_x if hasattr(results.seg, "f1_curve_x") else np.linspace(0, 1, results.seg.f1_curve.shape[1])
    mean_f1 = results.seg.f1_curve.mean(axis=0)
    mean_p = results.seg.p_curve.mean(axis=0)
    mean_r = results.seg.r_curve.mean(axis=0)

    best_idx = int(np.argmax(mean_f1))
    best_conf = float(px[best_idx])

    def stats_at(conf: float) -> dict:
        idx = int(np.argmin(np.abs(px - conf)))
        return {"confidence": conf, "mean_precision": round(float(mean_p[idx]), 4),
                "mean_recall": round(float(mean_r[idx]), 4), "mean_f1": round(float(mean_f1[idx]), 4)}

    report = {
        "weights": args.weights,
        "best_f1_confidence": round(best_conf, 4),
        "best_f1_stats": stats_at(best_conf),
        "candidates": [stats_at(c) for c in args.candidates],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "confidence_calibration.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
