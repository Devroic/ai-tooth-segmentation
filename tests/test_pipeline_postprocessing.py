import numpy as np

from tooth_seg.inference.pipeline import ToothDetection, mirror_fdi, resolve_duplicate_fdi
from tooth_seg.taxonomy import fdi_to_group

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 10


def make_detection(fdi, confidence, x1, x2):
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    mask[:, x1:x2] = True
    bbox = (float(x1), 0.0, float(x2), float(IMAGE_HEIGHT))
    return ToothDetection(fdi=fdi, group=fdi_to_group(fdi), confidence=confidence, bbox=bbox, mask=mask)


def test_mirror_fdi():
    assert mirror_fdi("11") == "21"
    assert mirror_fdi("21") == "11"
    assert mirror_fdi("36") == "46"
    assert mirror_fdi("48") == "38"


def test_resolve_duplicate_fdi_no_duplicates_unchanged():
    dets = [make_detection("11", 0.9, 10, 30), make_detection("36", 0.8, 60, 80)]
    resolved = resolve_duplicate_fdi(dets, IMAGE_WIDTH)
    assert {d.fdi for d in resolved} == {"11", "36"}
    assert len(resolved) == 2


def test_resolve_duplicate_fdi_overlapping_keeps_highest_confidence():
    # Same tooth flagged twice with near-identical, overlapping masks.
    dets = [make_detection("11", 0.9, 10, 30), make_detection("11", 0.5, 12, 32)]
    resolved = resolve_duplicate_fdi(dets, IMAGE_WIDTH)
    assert len(resolved) == 1
    assert resolved[0].fdi == "11"
    assert resolved[0].confidence == 0.9


def test_resolve_duplicate_fdi_mirror_mixup_relabels_wrong_side():
    # FDI "11" (quadrant 1) is expected on the left half of the image.
    left = make_detection("11", 0.9, 10, 30)   # correctly on the expected side
    right = make_detection("11", 0.6, 70, 90)  # actually the mirror tooth (21)
    resolved = resolve_duplicate_fdi([left, right], IMAGE_WIDTH)

    by_fdi = {d.fdi: d for d in resolved}
    assert set(by_fdi) == {"11", "21"}
    assert by_fdi["11"].bbox == left.bbox
    assert by_fdi["21"].bbox == right.bbox
    assert by_fdi["21"].group == fdi_to_group("21")


def test_resolve_duplicate_fdi_drops_rather_than_collide():
    left = make_detection("11", 0.9, 10, 30)
    right = make_detection("11", 0.3, 70, 90)   # would mirror to "21"
    existing_21 = make_detection("21", 0.7, 60, 65)  # "21" already taken elsewhere

    resolved = resolve_duplicate_fdi([left, right, existing_21], IMAGE_WIDTH)

    assert len(resolved) == 2
    by_fdi = {d.fdi: d for d in resolved}
    assert by_fdi["11"].bbox == left.bbox
    assert by_fdi["21"].bbox == existing_21.bbox
