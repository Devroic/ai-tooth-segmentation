"""Shared helpers for converting heterogeneous source-dataset annotation
formats into one unified schema.

Unified schema (per source dataset, written to outputs/unified/<id>.json):
{
  "images": [
    {"id": int, "file_name": str (absolute path), "width": int, "height": int,
     "source_dataset": str}
  ],
  "annotations": [
    {"image_id": int, "fdi": "36"|null, "group": "molar"|null,
     "is_tooth": true, "segmentation": [[x1,y1,x2,y2,...], ...],
     "bbox": [x,y,w,h], "box_only": bool}
  ]
}

`segmentation` is a list of polygon rings in absolute pixel coordinates
(COCO-style, exterior contours only). `box_only=True` marks annotations
where only a bounding box is available (no true mask) so downstream
consumers can decide whether to use them for mask training.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MIN_CONTOUR_AREA = 20  # px^2, drop tiny noise contours


def mask_to_polygons(mask: np.ndarray, min_area: float = MIN_CONTOUR_AREA) -> list[list[float]]:
    """Binary mask (HxW, any nonzero=foreground) -> list of flat polygon
    rings [x1,y1,x2,y2,...] in absolute pixel coords. External contours only.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        if len(c) < 3:
            continue
        flat = c.reshape(-1, 2).astype(float).flatten().tolist()
        polygons.append(flat)
    return polygons


def polygons_to_mask(polygons: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_bbox(polygons: list[list[float]]) -> list[float]:
    all_pts = np.concatenate([np.array(p).reshape(-1, 2) for p in polygons], axis=0)
    x0, y0 = all_pts.min(axis=0)
    x1, y1 = all_pts.max(axis=0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def box_to_polygon(x: float, y: float, w: float, h: float) -> list[list[float]]:
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


class UnifiedWriter:
    def __init__(self, source_dataset: str):
        self.source_dataset = source_dataset
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self._img_id = 0

    def add_image(self, file_name: str, width: int, height: int) -> int:
        img_id = self._img_id
        self._img_id += 1
        self.images.append({
            "id": img_id, "file_name": file_name, "width": width, "height": height,
            "source_dataset": self.source_dataset,
        })
        return img_id

    def add_annotation(self, image_id: int, segmentation: list[list[float]],
                        fdi: str | None, group: str | None, is_tooth: bool = True,
                        box_only: bool = False, bbox: list[float] | None = None) -> None:
        if not segmentation:
            return
        self.annotations.append({
            "image_id": image_id,
            "fdi": fdi,
            "group": group,
            "is_tooth": is_tooth,
            "segmentation": segmentation,
            "bbox": bbox if bbox is not None else polygon_bbox(segmentation),
            "box_only": box_only,
        })

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.source_dataset}.json"
        with open(out_path, "w") as f:
            json.dump({"images": self.images, "annotations": self.annotations}, f)
        return out_path
