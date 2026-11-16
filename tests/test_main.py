"""Tests for main.py — number-key airfoil presets."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

pytest.importorskip("pygame")

from main import preset_airfoil_for_key


def test_naca_presets_unchanged():
    assert preset_airfoil_for_key("1") == "NACA0012"
    assert preset_airfoil_for_key("2") == "NACA2412"
    assert preset_airfoil_for_key("3") == "NACA4415"


def test_kfm_presets_select_stepped_airfoils():
    # KFM-1..4 were advertised in the README and handled by apply_airfoil,
    # but no key could ever select them.
    assert preset_airfoil_for_key("4") == "KFM1"
    assert preset_airfoil_for_key("5") == "KFM2"
    assert preset_airfoil_for_key("6") == "KFM3"
    assert preset_airfoil_for_key("7") == "KFM4"


def test_other_keys_have_no_preset():
    assert preset_airfoil_for_key("q") is None
    assert preset_airfoil_for_key("0") is None
    assert preset_airfoil_for_key("8") is None
    assert preset_airfoil_for_key("") is None
