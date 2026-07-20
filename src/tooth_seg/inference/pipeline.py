"""Combined inference pipeline: given a panoramic radiograph, produce
(1) a binary tooth/non-tooth mask overlay (objective 2 model) and
(2) a per-tooth instance segmentation with FDI + tooth-group labels
    (objective 3 model).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tooth_seg.taxonomy import FDI_CODES, fdi_to_group, tooth_display_name

_GROUP_COLORS = {
    "incisor": (66, 133, 244),   # blue
    "canine": (219, 68, 55),     # red
    "premolar": (244, 180, 0),   # amber
    "molar": (15, 157, 88),      # green
}
_DEFAULT_COLOR = (200, 200, 200)


@dataclass
class ToothDetection:
    fdi: str
    group: str | None
    confidence: float
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2
    mask: np.ndarray  # HxW bool, full image size


@dataclass
class PipelineResult:
    binary_mask: np.ndarray | None = None
    binary_overlay: np.ndarray | None = None
    detections: list[ToothDetection] = field(default_factory=list)
    multiclass_overlay: np.ndarray | None = None


class ToothSegPipeline:
    def __init__(self, binary_weights: str | Path | None = None,
                 multiclass_weights: str | Path | None = None):
        self.binary_model = YOLO(str(binary_weights)) if binary_weights else None
        self.multiclass_model = YOLO(str(multiclass_weights)) if multiclass_weights else None

    def run(self, image: np.ndarray, conf: float = 0.25, imgsz: int = 768) -> PipelineResult:
        """`image` is an RGB uint8 array (H,W,3)."""
        result = PipelineResult()

        if self.binary_model is not None:
            mask, overlay = self._run_binary(image, conf, imgsz)
            result.binary_mask, result.binary_overlay = mask, overlay

        if self.multiclass_model is not None:
            dets, overlay = self._run_multiclass(image, conf, imgsz)
            result.detections, result.multiclass_overlay = dets, overlay

        return result

    def _run_binary(self, image: np.ndarray, conf: float, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        preds = self.binary_model.predict(image, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
        mask = np.zeros((h, w), dtype=bool)
        if preds.masks is not None:
            for m in preds.masks.data.cpu().numpy():
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                mask |= m_resized

        overlay = image.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([15, 157, 88])).astype(np.uint8)
        return mask, overlay

    def _run_multiclass(self, image: np.ndarray, conf: float, imgsz: int) -> tuple[list[ToothDetection], np.ndarray]:
        h, w = image.shape[:2]
        preds = self.multiclass_model.predict(image, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
        detections: list[ToothDetection] = []
        overlay = image.copy()

        if preds.masks is None:
            return detections, overlay

        boxes = preds.boxes
        for i, m in enumerate(preds.masks.data.cpu().numpy()):
            cls_idx = int(boxes.cls[i].item())
            fdi = FDI_CODES[cls_idx] if cls_idx < len(FDI_CODES) else str(cls_idx)
            confv = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            group = fdi_to_group(fdi)
            detections.append(ToothDetection(fdi=fdi, group=group, confidence=confv,
                                              bbox=(x1, y1, x2, y2), mask=mask_resized))

        detections.sort(key=lambda d: d.fdi)

        for det in detections:
            color = _GROUP_COLORS.get(det.group, _DEFAULT_COLOR)
            overlay[det.mask] = (0.5 * overlay[det.mask] + 0.5 * np.array(color)).astype(np.uint8)
            contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            label = det.fdi
            cv2.putText(overlay, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return detections, overlay


def detections_to_table(detections: list[ToothDetection]) -> list[dict]:
    return [
        {
            "FDI": d.fdi,
            "Tooth": tooth_display_name(d.fdi),
            "Group": d.group,
            "Confidence": round(d.confidence, 3),
        }
        for d in detections
    ]
