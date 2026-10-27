"""Tests for geometry.py — the 3D NACA airfoil mesh generator.

pyvista is an optional dependency (only geometry.py needs it), so these
tests are skipped when it is not installed, following the same pattern as
the visualization tests. CI installs pyvista so they run there.
"""

import numpy as np
import pytest

pytest.importorskip("pyvista")

from geometry import generate_naca_airfoil


def test_invalid_naca_length_raises():
    with pytest.raises(ValueError, match="exactly 4 digits"):
        generate_naca_airfoil("012", 0.0)
    with pytest.raises(ValueError, match="exactly 4 digits"):
        generate_naca_airfoil("00122", 0.0)


def test_symmetric_airfoil_bounds_match_chord_and_span():
    # NACA 0012, chord 1, span 1: centred on the quarter-chord point,
    # half-thickness 0.06 each side, span centred on z = 0.
    mesh = generate_naca_airfoil("0012", 0.0, num_points=60)
    assert mesh.n_points > 0
    assert mesh.volume > 0
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    assert xmin == pytest.approx(-0.25)
    assert xmax == pytest.approx(0.75)
    assert ymin == pytest.approx(-0.06, abs=1e-3)
    assert ymax == pytest.approx(0.06, abs=1e-3)
    assert zmin == pytest.approx(-0.5)
    assert zmax == pytest.approx(0.5)


def test_chord_and_span_scale_the_mesh():
    mesh = generate_naca_airfoil("0012", 0.0, chord=2.0, span=0.5, num_points=60)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    assert xmin == pytest.approx(-0.5)
    assert xmax == pytest.approx(1.5)
    assert ymin == pytest.approx(-0.12, abs=1e-3)
    assert ymax == pytest.approx(0.12, abs=1e-3)
    assert zmin == pytest.approx(-0.25)
    assert zmax == pytest.approx(0.25)


def test_angle_of_attack_preserves_volume():
    flat = generate_naca_airfoil("2412", 0.0, num_points=60)
    pitched = generate_naca_airfoil("2412", 10.0, num_points=60)
    assert pitched.volume == pytest.approx(flat.volume, rel=1e-2)
    # Pitching a cambered airfoil moves its y-extent asymmetrically.
    _, _, ymin, ymax, _, _ = pitched.bounds
    assert ymin < 0 < ymax
    assert abs(ymin) != pytest.approx(abs(ymax), rel=1e-2)
