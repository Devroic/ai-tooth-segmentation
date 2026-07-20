"""Evaluate a trained binary or multi-class model on its held-out test split
and write a JSON metrics summary (mask precision/recall/mAP50/mAP50-95
overall + per-class, from Ultralytics' built-in validator). Ultralytics also
saves a confusion matrix plot into the run's save_dir automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from tooth_seg.taxonomy import FDI_CODES

PROJECT_DIR = str((Path(__file__).resolve().parents[2] / "outputs" / "runs"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None, help="output JSON report path")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device="cpu",
                         project=PROJECT_DIR, name="eval")

    summary = {k: float(v) for k, v in results.results_dict.items()}

    per_class = {}
    try:
        names = results.names  # {idx: class_name}
        ap50_95 = results.seg.maps if hasattr(results, "seg") else results.box.maps
        for idx, val in enumerate(ap50_95):
            per_class[names.get(idx, str(idx))] = float(val)
    except Exception as e:
        per_class = {"_error": str(e)}

    report = {"weights": args.weights, "data": args.data, "split": args.split,
              "overall": summary, "per_class_map50_95": per_class}

    out_path = Path(args.out) if args.out else Path(results.save_dir) / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"Confusion matrix / curves saved under: {results.save_dir}")


if __name__ == "__main__":
    main()
