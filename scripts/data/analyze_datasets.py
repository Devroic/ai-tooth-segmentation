"""Analyzes outputs/unified/*.json: class distribution, resolutions, box vs
mask counts, per dataset and overall. Writes a JSON report and plots.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tooth_seg.taxonomy import FDI_CODES

UNIFIED_DIR = Path("outputs/unified")
REPORT_DIR = Path("outputs/reports")


def load_all() -> dict[str, dict]:
    datasets = {}
    for path in sorted(UNIFIED_DIR.glob("*.json")):
        with open(path) as f:
            datasets[path.stem] = json.load(f)
    return datasets


def analyze() -> dict:
    datasets = load_all()
    report = {"per_dataset": {}, "overall": {}}

    overall_fdi_counter = Counter()
    overall_group_counter = Counter()
    overall_widths, overall_heights = [], []
    total_images = 0
    total_anns = 0
    total_box_only = 0
    total_with_fdi = 0

    for name, d in datasets.items():
        n_img = len(d["images"])
        n_ann = len(d["annotations"])
        fdi_counter = Counter(a["fdi"] for a in d["annotations"] if a["fdi"])
        group_counter = Counter(a["group"] for a in d["annotations"] if a["group"])
        box_only = sum(1 for a in d["annotations"] if a.get("box_only"))
        with_fdi = sum(1 for a in d["annotations"] if a["fdi"])
        widths = [im["width"] for im in d["images"]]
        heights = [im["height"] for im in d["images"]]

        report["per_dataset"][name] = {
            "images": n_img,
            "annotations": n_ann,
            "annotations_with_fdi": with_fdi,
            "annotations_box_only": box_only,
            "fdi_class_counts": dict(fdi_counter),
            "group_class_counts": dict(group_counter),
            "num_fdi_classes_present": len(fdi_counter),
            "avg_annotations_per_image": round(n_ann / n_img, 2) if n_img else 0,
            "resolution": {
                "width_min": min(widths) if widths else None, "width_max": max(widths) if widths else None,
                "height_min": min(heights) if heights else None, "height_max": max(heights) if heights else None,
            },
        }

        overall_fdi_counter.update(fdi_counter)
        overall_group_counter.update(group_counter)
        overall_widths.extend(widths)
        overall_heights.extend(heights)
        total_images += n_img
        total_anns += n_ann
        total_box_only += box_only
        total_with_fdi += with_fdi

    report["overall"] = {
        "total_images": total_images,
        "total_annotations": total_anns,
        "total_annotations_with_fdi": total_with_fdi,
        "total_annotations_box_only": total_box_only,
        "fdi_classes_present": len(overall_fdi_counter),
        "fdi_classes_missing": sorted(set(FDI_CODES) - set(overall_fdi_counter)),
        "fdi_class_counts": {fdi: overall_fdi_counter.get(fdi, 0) for fdi in FDI_CODES},
        "group_class_counts": dict(overall_group_counter),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "dataset_analysis.json", "w") as f:
        json.dump(report, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 4))
    counts = [overall_fdi_counter.get(fdi, 0) for fdi in FDI_CODES]
    ax.bar(FDI_CODES, counts, color="#3b6fb0")
    ax.set_title("Per-tooth (FDI) annotation counts across all source datasets")
    ax.set_xlabel("FDI tooth code")
    ax.set_ylabel("annotation count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "fdi_class_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(report["per_dataset"].keys())
    img_counts = [report["per_dataset"][n]["images"] for n in names]
    ax.bar(names, img_counts, color="#4caf82")
    ax.set_title("Image count per source dataset")
    ax.set_xlabel("dataset id")
    ax.set_ylabel("images")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "images_per_dataset.png", dpi=120)
    plt.close(fig)

    return report


if __name__ == "__main__":
    report = analyze()
    print(json.dumps(report["overall"], indent=2))
    print(f"\nFull report: {REPORT_DIR / 'dataset_analysis.json'}")
