"""Dataset 11: Supervisely export, ann/<name>.jpg.json + img/<name>.jpg.
Per-object bitmap masks (base64 + zlib-compressed PNG) positioned via
"origin" offset. classTitle is a Universal Numbering System code ("1".."32").
"""
from __future__ import annotations

import base64
import io
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from tooth_seg.data.converters.common import UnifiedWriter, mask_to_polygons
from tooth_seg.taxonomy import fdi_to_group, universal_to_fdi

DATASET_ID = "11"


def _decode_bitmap(obj: dict, full_h: int, full_w: int) -> np.ndarray:
    raw = base64.b64decode(obj["bitmap"]["data"])
    png_bytes = zlib.decompress(raw)
    patch = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
    patch = (patch > 0).astype(np.uint8)
    ox, oy = obj["bitmap"]["origin"]
    full = np.zeros((full_h, full_w), dtype=np.uint8)
    ph, pw = patch.shape
    full[oy:oy + ph, ox:ox + pw] = patch
    return full


def convert(datasets_root: Path) -> dict:
    base = datasets_root / "11"
    img_dir = base / "img"
    ann_dir = base / "ann"

    writer = UnifiedWriter(DATASET_ID)
    skipped = 0
    for ann_path in sorted(ann_dir.glob("*.json")):
        img_name = ann_path.name[:-len(".json")]  # strip trailing .json -> "<n>.jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        with open(ann_path) as f:
            data = json.load(f)
        h, w = data["size"]["height"], data["size"]["width"]
        img_id = writer.add_image(str(img_path), w, h)

        for obj in data.get("objects", []):
            if obj.get("geometryType") != "bitmap":
                continue
            fdi = universal_to_fdi(obj.get("classTitle"))
            if fdi is None:
                skipped += 1
                continue
            mask = _decode_bitmap(obj, h, w)
            polys = mask_to_polygons(mask)
            if not polys:
                skipped += 1
                continue
            writer.add_annotation(image_id=img_id, segmentation=polys, fdi=fdi, group=fdi_to_group(fdi))

    return {"images": writer.images, "annotations": writer.annotations, "_writer": writer, "_skipped": skipped}


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Datasets")
    out = convert(root)
    w = out.pop("_writer")
    print(f"images={len(out['images'])} annotations={len(out['annotations'])} skipped={out['_skipped']}")
    path = w.save(Path("outputs/unified"))
    print("saved to", path)
