from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tooth_seg.inference.pipeline import ToothSegPipeline
from tooth_seg.taxonomy import FDI_CODES

FIXTURE = Path(__file__).parent / "fixtures" / "sample_radiograph.jpg"
BINARY_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "binary_seg" / "best.pt"
MULTICLASS_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "multiclass_seg" / "best.pt"

pytestmark = pytest.mark.skipif(
    not (BINARY_WEIGHTS.exists() and MULTICLASS_WEIGHTS.exists()),
    reason="shipped model weights not found under models/",
)


@pytest.fixture(scope="module")
def pipeline():
    return ToothSegPipeline(binary_weights=BINARY_WEIGHTS, multiclass_weights=MULTICLASS_WEIGHTS)


@pytest.fixture(scope="module")
def image():
    return np.array(Image.open(FIXTURE).convert("RGB"))


def test_pipeline_runs_end_to_end(pipeline, image):
    result = pipeline.run(image, conf=0.15)

    assert result.binary_mask is not None
    assert result.binary_mask.shape == image.shape[:2]
    assert result.binary_mask.sum() > 0  # some tooth pixels found

    assert result.binary_overlay.shape == image.shape
    assert result.multiclass_overlay.shape == image.shape


def test_pipeline_detections_are_valid(pipeline, image):
    result = pipeline.run(image, conf=0.15)

    assert len(result.detections) > 0
    fdi_seen = [d.fdi for d in result.detections]

    assert len(fdi_seen) == len(set(fdi_seen)), "resolve_duplicate_fdi should leave no duplicate FDI codes"
    for d in result.detections:
        assert d.fdi in FDI_CODES
        assert 0.0 <= d.confidence <= 1.0
        assert d.mask.shape == image.shape[:2]
