"""Tests for lbm_solver.py — D2Q9 Lattice Boltzmann solver."""

import numpy as np

from lbm_solver import LBMSolver


def make_solver():
    return LBMSolver(nx=40, ny=20, tau=0.55)


def test_weights_sum_to_one_and_opposites_pair():
    s = make_solver()
    assert np.isclose(s.weights.sum(), 1.0)
    for i in range(s.NL):
        opp = s.opposite[i]
        assert s.opposite[opp] == i
        assert s.cxs[opp] == -s.cxs[i]
        assert s.cys[opp] == -s.cys[i]


def test_equilibrium_sums_to_rho():
    s = make_solver()
    rho = np.ones((s.ny, s.nx))
    ux = np.full((s.ny, s.nx), 0.1)
    uy = np.zeros((s.ny, s.nx))
    feq = s.equilibrium(rho, ux, uy)
    assert feq.shape == (s.ny, s.nx, s.NL)
    assert np.allclose(feq.sum(axis=2), rho, atol=1e-10)


def test_reset_state():
    s = make_solver()
    assert s.rho.shape == (s.ny, s.nx)
    assert np.allclose(s.rho, 1.0)
    assert not s.obstacle.any()
    speed = s.get_velocity_field()
    assert speed.shape == (s.ny, s.nx)
    assert np.allclose(speed, s.v_in)


def test_steps_stay_finite_and_positive():
    s = make_solver()
    for _ in range(10):
        s.step()
    assert np.isfinite(s.rho).all()
    assert (s.rho > 0).all()
    assert np.isfinite(s.get_velocity_field()).all()


def test_obstacle_velocity_stays_zero():
    s = make_solver()
    mask = np.zeros((s.ny, s.nx), dtype=bool)
    mask[8:12, 15:20] = True
    s.set_obstacle(mask)
    for _ in range(5):
        s.step()
    assert (s.ux[mask] == 0).all()
    assert (s.uy[mask] == 0).all()
