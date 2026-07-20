"""Build final YOLOv8-seg-ready dataset directories from the unified
per-dataset JSONs + the dedup-consistent split assignment.

Produces two datasets:
  outputs/data/binary/{train,val,test}/{images,labels} + data.yaml   (1 class: tooth)
  outputs/data/multiclass/{train,val,test}/{images,labels} + data.yaml (32 classes: FDI codes)

Images are re-encoded to JPEG and downscaled (longest side capped) to keep
CPU training tractable; polygon coordinates are rescaled to match and
simplified (cv2.approxPolyDP) to keep label files small. box_only
annotations (Dataset 30, bounding boxes only) are included in the
multiclass pool but excluded from the binary pool, since a box is not an
accurate pixel mask and would corrupt binary segmentation supervision.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tooth_seg.taxonomy import FDI_CODES

UNIFIED_DIR = Path("outputs/unified")
SPLITS_PATH = Path("outputs/splits.json")
OUT_ROOT = Path("outputs/data")

MULTICLASS_SOURCES = ["34", "11", "3", "30"]
BINARY_ONLY_SOURCES = ["42", "20", "18"]  # binary-mask-only datasets (no fdi)
FDI_CLASS_INDEX = {fdi: i for i, fdi in enumerate(FDI_CODES)}


def load_unified(name: str) -> dict:
    with open(UNIFIED_DIR / f"{name}.json") as f:
        return json.load(f)


def simplify_polygon(flat: list[float], epsilon: float = 1.5) -> list[float]:
    pts = np.array(flat, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3:
        return flat
    approx = cv2.approxPolyDP(pts, epsilon, closed=True)
    if len(approx) < 3:
        return flat
    return approx.reshape(-1, 2).flatten().tolist()


def largest_ring(segmentation: list[list[float]]) -> list[float]:
    """YOLO-seg label lines take one polygon; pick the largest-area ring if
    an instance produced multiple disjoint contours."""
    if len(segmentation) == 1:
        return segmentation[0]
    best, best_area = segmentation[0], -1
    for ring in segmentation:
        pts = np.array(ring, dtype=np.float32).reshape(-1, 2)
        area = cv2.contourArea(pts)
        if area > best_area:
            best, best_area = ring, area
    return best


def process_image(file_name: str, anns: list[dict], class_of, max_side: int) -> tuple[bytes, list[str]] | None:
    try:
        with Image.open(file_name) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            scale = min(1.0, max_side / max(w0, h0))
            if scale < 1.0:
                im = im.resize((max(1, round(w0 * scale)), max(1, round(h0 * scale))), Image.LANCZOS)
            w1, h1 = im.size
            import io
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
            jpeg_bytes = buf.getvalue()
    except Exception:
        return None

    lines = []
    for ann in anns:
        cls = class_of(ann)
        if cls is None:
            continue
        ring = largest_ring(ann["segmentation"])
        ring = simplify_polygon(ring)
        pts = np.array(ring, dtype=np.float64).reshape(-1, 2) * scale
        pts[:, 0] = np.clip(pts[:, 0], 0, w1 - 1) / w1
        pts[:, 1] = np.clip(pts[:, 1], 0, h1 - 1) / h1
        if len(pts) < 3:
            continue
        coord_str = " ".join(f"{v:.6f}" for v in pts.flatten())
        lines.append(f"{cls} {coord_str}")

    if not lines:
        return None
    return jpeg_bytes, lines


def write_yaml(path: Path, nc: int, names: list[str]) -> None:
    with open(path, "w") as f:
        f.write("train: train/images\nval: val/images\ntest: test/images\n")
        f.write(f"nc: {nc}\nnames: {names}\n")


def materialize(pool_name: str, images_by_id: dict, anns_by_image: dict, class_of, max_side: int,
                 splits: dict, cap_per_source: dict[str, int] | None, seed: int = 0) -> None:
    rng = random.Random(seed)
    out_dir = OUT_ROOT / pool_name
    for split in ("train", "val", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[int]] = {}
    for img_id, img in images_by_id.items():
        by_source.setdefault(img["source_dataset"], []).append(img_id)

    selected_ids = []
    for source, ids in by_source.items():
        cap = (cap_per_source or {}).get(source)
        if cap is not None and len(ids) > cap:
            ids = rng.sample(ids, cap)
        selected_ids.extend(ids)

    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    for i, img_id in enumerate(selected_ids):
        img = images_by_id[img_id]
        anns = anns_by_image.get(img_id, [])
        if not anns:
            counts["skipped"] += 1
            continue
        split = splits.get(img["file_name"], "train")
        result = process_image(img["file_name"], anns, class_of, max_side)
        if result is None:
            counts["skipped"] += 1
            continue
        jpeg_bytes, lines = result
        stem = f"{img['source_dataset']}_{Path(img['file_name']).stem}"
        (out_dir / split / "images" / f"{stem}.jpg").write_bytes(jpeg_bytes)
        (out_dir / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        counts[split] += 1
        if (i + 1) % 500 == 0:
            print(f"  [{pool_name}] {i + 1}/{len(selected_ids)}")

    print(f"[{pool_name}] done: {counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary-max-side", type=int, default=512)
    ap.add_argument("--multiclass-max-side", type=int, default=768)
    ap.add_argument("--binary-cap", type=int, default=1200, help="max images per binary-only source (42,20,18)")
    args = ap.parse_args()

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    # ---- multiclass pool ----
    mc_images, mc_anns = {}, {}
    for source in MULTICLASS_SOURCES:
        d = load_unified(source)
        id_offset = max(mc_images.keys(), default=-1) + 1
        local_to_global = {}
        for img in d["images"]:
            gid = id_offset + img["id"]
            local_to_global[img["id"]] = gid
            mc_images[gid] = img
        for ann in d["annotations"]:
            gid = local_to_global[ann["image_id"]]
            mc_anns.setdefault(gid, []).append(ann)

    def mc_class_of(ann):
        if not ann.get("fdi"):
            return None
        return FDI_CLASS_INDEX.get(ann["fdi"])

    materialize("multiclass", mc_images, mc_anns, mc_class_of, args.multiclass_max_side, splits, cap_per_source=None)
    write_yaml(OUT_ROOT / "multiclass" / "data.yaml", nc=len(FDI_CODES), names=FDI_CODES)

    # ---- binary pool: true-mask sources only (34, 11, 3 non-box_only + 42, 20, 18) ----
    bin_images, bin_anns = {}, {}
    for source in MULTICLASS_SOURCES + BINARY_ONLY_SOURCES:
        d = load_unified(source)
        id_offset = max(bin_images.keys(), default=-1) + 1
        local_to_global = {}
        for img in d["images"]:
            gid = id_offset + img["id"]
            local_to_global[img["id"]] = gid
            bin_images[gid] = img
        for ann in d["annotations"]:
            if ann.get("box_only"):
                continue
            gid = local_to_global[ann["image_id"]]
            bin_anns.setdefault(gid, []).append(ann)

    def bin_class_of(ann):
        return 0

    cap = {s: args.binary_cap for s in BINARY_ONLY_SOURCES}
    materialize("binary", bin_images, bin_anns, bin_class_of, args.binary_max_side, splits, cap_per_source=cap)
    write_yaml(OUT_ROOT / "binary" / "data.yaml", nc=1, names=["tooth"])


if __name__ == "__main__":
    main()
