"""Unified tooth-labeling taxonomy.

Source datasets use one of two numbering conventions:
  - FDI / ISO-3950 two-digit notation (e.g. "36") - Datasets/34, /30
  - Universal Numbering System 1-32 (e.g. "3")    - Datasets/3, /11

This module is the single source of truth for converting between them, and
for deriving a tooth-group label (incisor/canine/premolar/molar) from an
FDI code.
"""
from __future__ import annotations

# --- Universal Numbering System (1-32) <-> FDI ------------------------------
# Standard adult permanent-dentition conversion table.
UNIVERSAL_TO_FDI: dict[int, str] = {
    1: "18", 2: "17", 3: "16", 4: "15", 5: "14", 6: "13", 7: "12", 8: "11",
    9: "21", 10: "22", 11: "23", 12: "24", 13: "25", 14: "26", 15: "27", 16: "28",
    17: "38", 18: "37", 19: "36", 20: "35", 21: "34", 22: "33", 23: "32", 24: "31",
    25: "41", 26: "42", 27: "43", 28: "44", 29: "45", 30: "46", 31: "47", 32: "48",
}
FDI_TO_UNIVERSAL: dict[str, int] = {v: k for k, v in UNIVERSAL_TO_FDI.items()}

# All 32 permanent-tooth FDI codes, in canonical (Universal 1-32) order.
FDI_CODES: list[str] = [UNIVERSAL_TO_FDI[i] for i in range(1, 33)]

# --- FDI -> tooth-group ------------------------------------------------------
# Second digit of FDI code = position from the midline within its quadrant.
_POSITION_TO_GROUP = {
    "1": "incisor", "2": "incisor",
    "3": "canine",
    "4": "premolar", "5": "premolar",
    "6": "molar", "7": "molar", "8": "molar",
}

# Quadrant (first digit of FDI, permanent dentition only: 1-4).
_QUADRANT_ARCH = {"1": "upper", "2": "upper", "3": "lower", "4": "lower"}
_QUADRANT_SIDE = {"1": "right", "2": "left", "3": "left", "4": "right"}


def fdi_to_group(fdi: str) -> str | None:
    """'36' -> 'molar'. Returns None for non-permanent / malformed codes."""
    if not fdi or len(fdi) != 2 or not fdi.isdigit():
        return None
    return _POSITION_TO_GROUP.get(fdi[1])


def fdi_to_arch_side(fdi: str) -> tuple[str, str] | None:
    """'36' -> ('lower', 'left')."""
    if not fdi or len(fdi) != 2 or not fdi.isdigit():
        return None
    q = fdi[0]
    if q not in _QUADRANT_ARCH:
        return None
    return _QUADRANT_ARCH[q], _QUADRANT_SIDE[q]


def universal_to_fdi(n: int | str) -> str | None:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    return UNIVERSAL_TO_FDI.get(n)


_ORDINAL_NAME = {
    "1": "central incisor", "2": "lateral incisor", "3": "canine",
    "4": "first premolar", "5": "second premolar",
    "6": "first molar", "7": "second molar", "8": "third molar",
}


def tooth_display_name(fdi: str) -> str:
    """'36' -> 'Lower Left First Molar (36)'."""
    arch_side = fdi_to_arch_side(fdi)
    ordinal = _ORDINAL_NAME.get(fdi[1]) if fdi and len(fdi) == 2 else None
    if not arch_side or not ordinal:
        return fdi
    arch, side = arch_side
    return f"{arch.capitalize()} {side.capitalize()} {ordinal.title()} ({fdi})"
