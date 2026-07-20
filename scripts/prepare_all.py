"""Run every converter end-to-end: raw Datasets/ -> outputs/unified/*.json.
Single entry point so the whole conversion step can be reproduced with one
command: `python scripts/prepare_all.py`.
"""
from __future__ import annotations

from pathlib import Path

from tooth_seg.data.converters import (
    convert_3, convert_11, convert_30, convert_34, convert_binary_masks,
)

DATASETS_ROOT = Path("Datasets")
OUT_DIR = Path("outputs/unified")


def run(name: str, convert_fn) -> None:
    print(f"--- converting dataset {name} ---")
    out = convert_fn(DATASETS_ROOT)
    writer = out.pop("_writer")
    print(f"  images={len(out['images'])} annotations={len(out['annotations'])} skipped={out['_skipped']}")
    path = writer.save(OUT_DIR)
    print(f"  saved to {path}")


if __name__ == "__main__":
    run("34", convert_34.convert)
    run("11", convert_11.convert)
    run("3", convert_3.convert)
    run("30", convert_30.convert)
    run("42", convert_binary_masks.convert_42)
    run("20", convert_binary_masks.convert_20)
    run("18", convert_binary_masks.convert_18)
    print("\nAll conversions complete.")
