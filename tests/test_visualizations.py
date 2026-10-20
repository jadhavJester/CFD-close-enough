"""Tests for visualizations.py — field-to-heatmap surface conversion."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from visualizations import add_obstacle_to_surface, field_to_surface

pytestmark = pytest.mark.skipif(
    not pygame.get_init() and False, reason="placeholder"
)


@pytest.fixture(scope="module", autouse=True)
def _pygame_init():
    pygame.init()
    pygame.display.set_mode((1, 1))
    yield
    pygame.quit()


def test_surface_size_matches_field():
    field = np.random.default_rng(0).random((20, 40))  # (ny, nx)
    surf = field_to_surface(field)
    assert surf is not None
    assert surf.get_size() == (40, 20)  # pygame surfaces are (width, height)


def test_constant_field_does_not_divide_by_zero():
    surf = field_to_surface(np.full((10, 10), 3.5))
    assert surf is not None
    assert surf.get_size() == (10, 10)


def test_grayscale_fallback_matches_normalised_values():
    field = np.array([[0.0, 1.0], [0.5, 0.25]])
    surf = field_to_surface(field, vmin=0.0, vmax=1.0, cmap="gray")
    px = pygame.surfarray.array3d(surf)  # (nx, ny, 3)
    assert px[1, 0, 0] == 255  # field max maps to white
    assert px[0, 0, 0] == 0    # field min maps to black


def test_add_obstacle_paints_masked_cells():
    surf = field_to_surface(np.zeros((10, 10)), cmap="gray")
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 5:7] = True
    out = add_obstacle_to_surface(surf, mask, color=(100, 100, 100))
    px = pygame.surfarray.array3d(out)
    assert (px[5, 2] == [100, 100, 100]).all()
    assert (px[0, 0] == [0, 0, 0]).all()


def test_add_obstacle_handles_none_surface():
    assert add_obstacle_to_surface(None, np.zeros((2, 2), dtype=bool)) is None
