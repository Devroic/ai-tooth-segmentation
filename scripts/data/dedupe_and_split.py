"""Cross-dataset image deduplication + train/val/test split assignment.

Several source datasets are suspiciously similarly sized (e.g. Datasets/11
and Datasets/20's "Panoramic radiography database" both have 598 images) and
likely derive from the same public panoramic-radiograph corpora re-exported
under different annotation tools. Training on one copy and testing on a
near-duplicate from another dataset would silently leak test data into
training. This script computes a difference-hash (dHash) per image, clusters
images with an identical hash (robust to re-compression/format changes),
and assigns every image in a cluster to the same split.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

UNIFIED_DIR = Path("outputs/unified")
REPORT_DIR = Path("outputs/reports")
SPLITS_PATH = Path("outputs/splits.json")

TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # remaining 0.1 -> test


def dhash(path: str, hash_size: int = 8) -> str:
    with Image.open(path) as im:
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join("1" if b else "0" for b in diff.flatten())
    return hashlib.md5(bits.encode()).hexdigest()


def split_for_cluster(cluster_key: str) -> str:
    """Deterministic pseudo-random split assignment from a stable hash."""
    h = int(hashlib.md5(cluster_key.encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000
    if frac < TRAIN_FRAC:
        return "train"
    if frac < TRAIN_FRAC + VAL_FRAC:
        return "val"
    return "test"


def main() -> None:
    all_images: list[tuple[str, str]] = []  # (dataset, file_name)
    for path in sorted(UNIFIED_DIR.glob("*.json")):
        with open(path) as f:
            d = json.load(f)
        for im in d["images"]:
            all_images.append((path.stem, im["file_name"]))

    print(f"hashing {len(all_images)} images...")
    hash_to_images: dict[str, list[tuple[str, str]]] = defaultdict(list)
    failed = 0
    for i, (dataset, file_name) in enumerate(all_images):
        try:
            h = dhash(file_name)
        except Exception:
            failed += 1
            continue
        hash_to_images[h].append((dataset, file_name))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(all_images)}")

    cross_dataset_dupe_clusters = {
        h: imgs for h, imgs in hash_to_images.items()
        if len({ds for ds, _ in imgs}) > 1
    }

    splits: dict[str, str] = {}
    for h, imgs in hash_to_images.items():
        split = split_for_cluster(h)
        for _, file_name in imgs:
            splits[file_name] = split

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f)

    split_counts = defaultdict(int)
    for s in splits.values():
        split_counts[s] += 1

    dupe_report = {
        "total_images": len(all_images),
        "hash_failures": failed,
        "unique_hash_clusters": len(hash_to_images),
        "cross_dataset_duplicate_clusters": len(cross_dataset_dupe_clusters),
        "cross_dataset_duplicate_examples": [
            {"hash": h, "members": [{"dataset": ds, "file": fn} for ds, fn in imgs]}
            for h, imgs in list(cross_dataset_dupe_clusters.items())[:20]
        ],
        "split_counts": dict(split_counts),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "dedup_report.json", "w") as f:
        json.dump(dupe_report, f, indent=2)

    print(f"images: {len(all_images)}  unique clusters: {len(hash_to_images)}  "
          f"cross-dataset duplicate clusters: {len(cross_dataset_dupe_clusters)}")
    print("split counts:", dict(split_counts))
    print(f"splits saved to {SPLITS_PATH}")
    print(f"dedup report saved to {REPORT_DIR / 'dedup_report.json'}")


if __name__ == "__main__":
    main()
