"""Dataset 3: Segmentation/teeth_polygon.json, Labelbox-style export.
Each object's "polygons" field is a list of many small polygon fragments
(not one clean outline) - they must be rasterized and unioned into a mask,
then re-extracted as clean external contours. "title" is a Universal
Numbering System code ("1".."32").
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tooth_seg.data.converters.common import UnifiedWriter, mask_to_polygons, polygons_to_mask
from tooth_seg.taxonomy import fdi_to_group, universal_to_fdi

DATASET_ID = "3"


def convert(datasets_root: Path) -> dict:
    base = datasets_root / "3"
    img_dir = base / "Radiographs"
    poly_path = base / "Segmentation" / "teeth_polygon.json"

    with open(poly_path) as f:
        records = json.load(f)

    # Radiograph filenames are stored with ".JPG" (uppercase); build a
    # case-insensitive lookup since "External ID" casing may differ.
    img_files = {p.name.lower(): p for p in img_dir.glob("*")}

    writer = UnifiedWriter(DATASET_ID)
    skipped = 0
    for rec in records:
        ext_id = rec.get("External ID", "")
        img_path = img_files.get(ext_id.lower())
        if img_path is None:
            skipped += 1
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        img_id = writer.add_image(str(img_path), w, h)

        for obj in rec.get("Label", {}).get("objects", []):
            fdi = universal_to_fdi(obj.get("title"))
            if fdi is None:
                skipped += 1
                continue
            fragments = obj.get("polygons") or []
            if not fragments:
                continue
            flat_fragments = [
                [coord for pt in ring for coord in pt] for ring in fragments if len(ring) >= 3
            ]
            if not flat_fragments:
                continue
            mask = polygons_to_mask(flat_fragments, h, w)
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
