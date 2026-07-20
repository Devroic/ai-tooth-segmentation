"""Generic converter for binary-mask datasets where each image has a single
paired mask file (foreground = tooth region, no per-tooth identity).
Used for Datasets 42, 20, and 18. Each connected component in the mask is
emitted as a separate instance annotation (fdi=None, group=None,
is_tooth=True) so these can supply extra training signal for the binary
stage while still being usable as instance segmentation targets.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tooth_seg.data.converters.common import UnifiedWriter, mask_to_polygons

MIN_COMPONENT_AREA = 30


def _mask_to_instance_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    """Binary mask -> list of instances, each a list of polygon rings."""
    mask_u8 = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    instances = []
    for label in range(1, n_labels):  # 0 = background
        if stats[label, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
            continue
        component = (labels == label).astype(np.uint8)
        polys = mask_to_polygons(component)
        if polys:
            instances.append(polys)
    return instances


def convert_paired_dir(dataset_id: str, image_paths: list[Path], mask_for_image, split_instances: bool = True) -> dict:
    """`mask_for_image(img_path) -> Path | None` locates the mask file for a
    given image. If split_instances is False, the whole mask is kept as one
    annotation (faster, used for very large/low-value corpora).
    """
    writer = UnifiedWriter(dataset_id)
    skipped = 0
    for img_path in image_paths:
        mask_path = mask_for_image(img_path)
        if mask_path is None or not mask_path.exists():
            skipped += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        mask = np.array(Image.open(mask_path).convert("L"))
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img_id = writer.add_image(str(img_path), w, h)

        if split_instances:
            instances = _mask_to_instance_polygons(mask)
            if not instances:
                skipped += 1
                continue
            for polys in instances:
                writer.add_annotation(image_id=img_id, segmentation=polys, fdi=None, group=None)
        else:
            polys = mask_to_polygons(mask)
            if not polys:
                skipped += 1
                continue
            writer.add_annotation(image_id=img_id, segmentation=polys, fdi=None, group=None)

    return {"images": writer.images, "annotations": writer.annotations, "_writer": writer, "_skipped": skipped}


def convert_42(datasets_root: Path) -> dict:
    """Datasets/42/{train,test}/{images,annotations}: 1:1 filename match,
    image X.jpg <-> mask X.png. 6225 pairs total, the largest binary corpus.
    """
    base = datasets_root / "42"
    image_paths = []
    for split in ("train", "test"):
        image_paths.extend(sorted((base / split / "images").glob("*.jpg")))

    def mask_for(img_path: Path) -> Path:
        split = img_path.parent.parent.name
        return base / split / "annotations" / (img_path.stem + ".png")

    return convert_paired_dir("42", image_paths, mask_for)


def convert_20(datasets_root: Path) -> dict:
    """Datasets/20/Dataset and code/{train,test}/{images,masks}: 1:1 filename
    match (.jpg <-> .bmp), masks/ covers the whole teeth region per image
    (not per-tooth), so instances are kept unsplit (split_instances=False)
    to avoid arbitrarily chopping a connected dental arch into fake instances.
    """
    base = datasets_root / "20" / "Dataset and code"
    image_paths = []
    for split in ("train", "test"):
        d = base / split / "images"
        if d.exists():
            image_paths.extend(sorted(d.glob("*.jpg")))

    def mask_for(img_path: Path) -> Path:
        split = img_path.parent.parent.name
        return base / split / "masks" / (img_path.stem + ".bmp")

    return convert_paired_dir("20", image_paths, mask_for, split_instances=False)


def convert_18(datasets_root: Path) -> dict:
    """Datasets/18/train/train/{image,mask}: 1:1 filename match, low-res
    (640x320) binary masks.
    """
    base = datasets_root / "18" / "train" / "train"
    image_paths = sorted((base / "image").glob("*.png"))

    def mask_for(img_path: Path) -> Path:
        return base / "mask" / img_path.name

    return convert_paired_dir("18", image_paths, mask_for, split_instances=False)


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Datasets")
    for name, fn in [("42", convert_42), ("20", convert_20), ("18", convert_18)]:
        out = fn(root)
        w = out.pop("_writer")
        print(f"[{name}] images={len(out['images'])} annotations={len(out['annotations'])} skipped={out['_skipped']}")
        path = w.save(Path("outputs/unified"))
        print("  saved to", path)
