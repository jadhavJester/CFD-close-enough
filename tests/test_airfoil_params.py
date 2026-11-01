"""Tests for airfoil_params.py — the NACA 4-digit label stepping helpers."""

from airfoil_params import build_naca, parse_naca, step_naca_param


def test_parse_naca_valid_label():
    assert parse_naca("NACA2412") == (2, 4, 12)
    assert parse_naca("NACA0012") == (0, 0, 12)


def test_parse_naca_rejects_non_naca_labels():
    assert parse_naca("KFM2") is None
    assert parse_naca("NACA012") is None       # not 4 digits
    assert parse_naca("NACA00122") is None
    assert parse_naca("NACA00a2") is None      # non-digit digits


def test_build_naca_roundtrip():
    label = "NACA2412"
    m, p, t = parse_naca(label)
    assert build_naca(m, p, t) == label


def test_step_naca_param_camber_keys():
    assert step_naca_param("NACA2412", "q") == "NACA3412"
    assert step_naca_param("NACA2412", "a") == "NACA1412"
    # clamps at the digit bounds
    assert step_naca_param("NACA9412", "q") == "NACA9412"
    assert step_naca_param("NACA0412", "a") == "NACA0412"


def test_step_naca_param_position_keys():
    assert step_naca_param("NACA2412", "w") == "NACA2512"
    # Regression: the S key used to be a no-op (params[1] - 0) and never
    # decreased the camber position.
    assert step_naca_param("NACA2412", "s") == "NACA2312"
    assert step_naca_param("NACA2912", "w") == "NACA2912"
    assert step_naca_param("NACA2012", "s") == "NACA2012"


def test_step_naca_param_thickness_keys():
    assert step_naca_param("NACA2412", "e") == "NACA2413"
    assert step_naca_param("NACA2412", "d") == "NACA2411"
    assert step_naca_param("NACA2440", "e") == "NACA2440"
    assert step_naca_param("NACA2401", "d") == "NACA2401"


def test_step_naca_param_leaves_non_naca_labels_untouched():
    for key in ("q", "a", "w", "s", "e", "d"):
        assert step_naca_param("KFM2", key) == "KFM2"


def test_step_naca_param_keys_are_case_insensitive():
    assert step_naca_param("NACA2412", "S") == "NACA2312"
    assert step_naca_param("NACA2412", "Q") == "NACA3412"
