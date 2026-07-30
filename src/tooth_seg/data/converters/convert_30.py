"""Dataset 30: Roboflow YOLO export (yolo_train_dataset + yolo_test_dataset),
bounding boxes only (no polygon masks). The dataset's own data.yaml orders
classes by ascending FDI string ("11".."18", "21".."28", "31".."38",
"41".."48") - NOT the Universal-numbering order of tooth_seg.taxonomy.FDI_CODES
(which runs "18".."11", "38".."31", ...). Those two orders happen to coincide
for quadrants 2 and 4 but are exactly reversed for quadrants 1 and 3, so
class_id must be mapped via this dataset's own class list, not FDI_CODES.
Boxes are converted to degenerate 4-corner polygons and flagged box_only=True
so training/eval code can treat them as weak (coarse) mask supervision rather
than precise contours.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from tooth_seg.data.converters.common import UnifiedWriter, box_to_polygon
from tooth_seg.taxonomy import fdi_to_group

DATASET_ID = "30"

# This dataset's data.yaml `names` order (ascending FDI string), confirmed
# against Datasets/30/yolo_train_dataset/data.yaml and yolo_test_dataset/data.yaml.
_CLASS_ID_TO_FDI = [
    "11", "12", "13", "14", "15", "16", "17", "18",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "31", "32", "33", "34", "35", "36", "37", "38",
    "41", "42", "43", "44", "45", "46", "47", "48",
]


def _convert_split(writer: UnifiedWriter, images_dir: Path, labels_dir: Path, skipped: list[int]) -> None:
    for img_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        img_id = writer.add_image(str(img_path), w, h)

        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if not (0 <= cls_id < len(_CLASS_ID_TO_FDI)):
                skipped[0] += 1
                continue
            fdi = _CLASS_ID_TO_FDI[cls_id]
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            box_w = bw * w
            box_h = bh * h
            poly = box_to_polygon(x, y, box_w, box_h)
            writer.add_annotation(
                image_id=img_id, segmentation=poly, fdi=fdi, group=fdi_to_group(fdi),
                box_only=True, bbox=[x, y, box_w, box_h],
            )


def convert(datasets_root: Path) -> dict:
    base = datasets_root / "30"
    writer = UnifiedWriter(DATASET_ID)
    skipped = [0]

    for split_root, splits in [
        (base / "yolo_train_dataset", ["train", "valid", "test"]),
        (base / "yolo_test_dataset", ["test"]),
    ]:
        for split in splits:
            images_dir = split_root / split / "images"
            labels_dir = split_root / split / "labels"
            if images_dir.exists() and labels_dir.exists():
                _convert_split(writer, images_dir, labels_dir, skipped)

    return {"images": writer.images, "annotations": writer.annotations, "_writer": writer, "_skipped": skipped[0]}


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Datasets")
    out = convert(root)
    w = out.pop("_writer")
    print(f"images={len(out['images'])} annotations={len(out['annotations'])} skipped={out['_skipped']}")
    path = w.save(Path("outputs/unified"))
    print("saved to", path)
