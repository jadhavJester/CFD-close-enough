"""Tests for airfoils.py — NACA 4-digit and KFM mask generators."""

import numpy as np
import pytest

from airfoils import generate_kfm, generate_naca4

SHAPE = (100, 200)
CX, CY, CHORD = 100, 50, 80


def test_naca_mask_shape_and_dtype():
    mask = generate_naca4(SHAPE, CX, CY, CHORD, "0012")
    assert mask.shape == SHAPE
    assert mask.dtype == bool


def test_naca_0012_nonempty_and_symmetric():
    mask = generate_naca4(SHAPE, CX, CY, CHORD, "0012", aoa_deg=0)
    assert mask.sum() > 100
    # Symmetric airfoil at zero AoA: mask mirrors about the centre row.
    top = mask[CY + 1: CY + 41, :]
    bottom = mask[CY - 40: CY, :][::-1, :]
    overlap = np.logical_and(top, bottom).sum()
    union = np.logical_or(top, bottom).sum()
    assert overlap / union > 0.9


def test_invalid_naca_string_returns_empty_mask():
    mask = generate_naca4(SHAPE, CX, CY, CHORD, "012")
    assert not mask.any()
    mask = generate_naca4(SHAPE, CX, CY, CHORD, "00123")
    assert not mask.any()
    # Non-digit 4-char strings used to crash in int(); they are invalid too.
    mask = generate_naca4(SHAPE, CX, CY, CHORD, "00a2")
    assert not mask.any()
    mask = generate_naca4(SHAPE, CX, CY, CHORD, "abcd")
    assert not mask.any()


def test_thicker_airfoil_covers_more_cells():
    thin = generate_naca4(SHAPE, CX, CY, CHORD, "0006")
    thick = generate_naca4(SHAPE, CX, CY, CHORD, "0024")
    assert thick.sum() > thin.sum() > 0


def test_cambered_airfoil_is_asymmetric():
    mask = generate_naca4(SHAPE, CX, CY, CHORD, "4412", aoa_deg=0)
    assert mask.sum() > 100
    rows = np.where(mask.any(axis=1))[0]
    centre_offset = (rows.min() + rows.max()) / 2 - CY
    assert centre_offset != 0  # camber shifts the body off the centreline


def test_kfm_types_nonempty():
    for kfm_type in (1, 2, 3, 4):
        mask = generate_kfm(SHAPE, CX, CY, CHORD, kfm_type)
        assert mask.shape == SHAPE
        assert mask.sum() > 50, f"KFM-{kfm_type} mask unexpectedly small"


def test_kfm_invalid_type_raises():
    # An unknown type used to fall through every step branch and silently
    # return a flat plate with no steps at all.
    for kfm_type in (0, 5, 99):
        with pytest.raises(ValueError, match="kfm_type must be 1, 2, 3 or 4"):
            generate_kfm(SHAPE, CX, CY, CHORD, kfm_type)


def test_kfm_nonpositive_thickness_raises():
    # thickness_pct=0 used to return a degenerate sliver from the
    # leading-edge rounding alone; a negative value, an empty mask.
    with pytest.raises(ValueError, match="thickness_pct must be positive"):
        generate_kfm(SHAPE, CX, CY, CHORD, 2, thickness_pct=0.0)
    with pytest.raises(ValueError, match="thickness_pct must be positive"):
        generate_kfm(SHAPE, CX, CY, CHORD, 2, thickness_pct=-0.05)


def test_nonpositive_chord_raises():
    # chord=0 used to short-circuit the bounding box and a negative chord
    # to invert it, both silently returning an empty mask — while the 3D
    # generator generate_naca_airfoil already rejects chord <= 0.
    for chord in (0, -80):
        with pytest.raises(ValueError, match="chord must be positive"):
            generate_naca4(SHAPE, CX, CY, chord, "2412")
        with pytest.raises(ValueError, match="chord must be positive"):
            generate_kfm(SHAPE, CX, CY, chord, 2)


def _top_edge_px(mask, chord_frac):
    """Height of the mask's top edge above CY at a chord fraction."""
    x = int(CX - CHORD / 2 + chord_frac * CHORD)
    rows = np.where(mask[:, x])[0]
    assert len(rows) > 0
    return CY - rows.min()


def test_kfm3_has_two_distinct_top_steps():
    # KFM-3 steps down twice on top: full thickness before 50% chord,
    # 0.70x thickness between 50% and 75%, thin base after 75%.
    # (Regression: the first step used to be overwritten by the second,
    # so the top was flat 0.70x thickness all the way to 75%.)
    mask = generate_kfm(SHAPE, CX, CY, CHORD, 3, thickness_pct=0.09)
    top_25 = _top_edge_px(mask, 0.25)
    top_60 = _top_edge_px(mask, 0.60)
    top_85 = _top_edge_px(mask, 0.85)
    assert top_25 > top_60 > top_85
    # Step heights track the design fractions of the chord (within a pixel).
    assert abs(top_25 - 0.09 * CHORD) <= 1
    assert abs(top_60 - 0.70 * 0.09 * CHORD) <= 1
