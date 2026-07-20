"""Dataset 34: CVAT COCO export, "segmentation and numbering/instances_default.json".
Categories 9-40 are literal FDI codes ("11".."48") - only these are kept as
individual-tooth ground truth (categories 1-8 are sparse/inconsistent
group-only labels mixed into the same export and are dropped to avoid noisy
duplicate annotations; categories 41-52 are anatomy/pathology, out of scope).
Images are referenced as "permanent/<n>.tif" in the JSON but stored flat as
"<n>.tif" on disk, and only ~772 of the 1000 referenced images are present -
annotations for missing images are skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

from tooth_seg.data.converters.common import UnifiedWriter
from tooth_seg.taxonomy import fdi_to_group

DATASET_ID = "34"
FDI_CATEGORY_IDS = {i: str(name) for i, name in [
    (9, "18"), (10, "17"), (11, "16"), (12, "15"), (13, "14"), (14, "13"), (15, "12"), (16, "11"),
    (17, "21"), (18, "22"), (19, "23"), (20, "24"), (21, "25"), (22, "26"), (23, "27"), (24, "28"),
    (25, "38"), (26, "37"), (27, "36"), (28, "35"), (29, "34"), (30, "33"), (31, "32"), (32, "31"),
    (33, "41"), (34, "42"), (35, "43"), (36, "44"), (37, "45"), (38, "46"), (39, "47"), (40, "48"),
]}


def convert(datasets_root: Path) -> dict:
    base = datasets_root / "34"
    ann_path = base / "segmentation and numbering" / "instances_default.json"
    img_dir = base / "images"

    with open(ann_path) as f:
        coco = json.load(f)

    # Verify categories match expectation (fail loudly if the source file changed shape).
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    for cid, expected in FDI_CATEGORY_IDS.items():
        assert cat_names.get(cid) == expected, f"category {cid} mismatch: {cat_names.get(cid)} != {expected}"

    writer = UnifiedWriter(DATASET_ID)
    img_id_map: dict[int, int] = {}
    for img in coco["images"]:
        file_name = Path(img["file_name"]).name  # strip "permanent/" prefix
        full_path = img_dir / file_name
        if not full_path.exists():
            continue
        new_id = writer.add_image(str(full_path), img["width"], img["height"])
        img_id_map[img["id"]] = new_id

    skipped = 0
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in FDI_CATEGORY_IDS:
            continue
        if ann["image_id"] not in img_id_map:
            continue
        fdi = FDI_CATEGORY_IDS[cat_id]
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, list):
            skipped += 1
            continue
        writer.add_annotation(
            image_id=img_id_map[ann["image_id"]],
            segmentation=seg,
            fdi=fdi,
            group=fdi_to_group(fdi),
            bbox=ann.get("bbox"),
        )

    return {"images": writer.images, "annotations": writer.annotations, "_writer": writer, "_skipped": skipped}


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Datasets")
    out = convert(root)
    w = out.pop("_writer")
    print(f"images={len(out['images'])} annotations={len(out['annotations'])} skipped={out['_skipped']}")
    path = w.save(Path("outputs/unified"))
    print("saved to", path)
