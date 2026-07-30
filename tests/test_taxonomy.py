from tooth_seg.taxonomy import (
    FDI_CODES,
    fdi_to_arch_side,
    fdi_to_group,
    tooth_display_name,
    universal_to_fdi,
)


def test_fdi_codes_cover_all_32_teeth():
    assert len(FDI_CODES) == 32
    assert len(set(FDI_CODES)) == 32


def test_universal_to_fdi_round_trip():
    assert universal_to_fdi(1) == "18"
    assert universal_to_fdi(8) == "11"
    assert universal_to_fdi(9) == "21"
    assert universal_to_fdi(32) == "48"
    assert universal_to_fdi("16") == "28"


def test_universal_to_fdi_invalid_input():
    assert universal_to_fdi(0) is None
    assert universal_to_fdi(33) is None
    assert universal_to_fdi("not a number") is None
    assert universal_to_fdi(None) is None


def test_fdi_to_group_by_position():
    assert fdi_to_group("11") == "incisor"
    assert fdi_to_group("12") == "incisor"
    assert fdi_to_group("13") == "canine"
    assert fdi_to_group("14") == "premolar"
    assert fdi_to_group("15") == "premolar"
    assert fdi_to_group("16") == "molar"
    assert fdi_to_group("17") == "molar"
    assert fdi_to_group("18") == "molar"


def test_fdi_to_group_invalid_input():
    assert fdi_to_group("") is None
    assert fdi_to_group("1") is None
    assert fdi_to_group("abc") is None
    assert fdi_to_group(None) is None


def test_fdi_to_arch_side():
    assert fdi_to_arch_side("11") == ("upper", "right")
    assert fdi_to_arch_side("21") == ("upper", "left")
    assert fdi_to_arch_side("31") == ("lower", "left")
    assert fdi_to_arch_side("41") == ("lower", "right")


def test_fdi_to_arch_side_invalid_input():
    assert fdi_to_arch_side("91") is None
    assert fdi_to_arch_side("") is None
    assert fdi_to_arch_side(None) is None


def test_tooth_display_name():
    assert tooth_display_name("36") == "Lower Left First Molar (36)"
    assert tooth_display_name("11") == "Upper Right Central Incisor (11)"


def test_tooth_display_name_falls_back_to_raw_code():
    assert tooth_display_name("99") == "99"
