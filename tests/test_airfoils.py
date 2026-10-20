"""Tests for airfoils.py — NACA 4-digit and KFM mask generators."""

import numpy as np

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
