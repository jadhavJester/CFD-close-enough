"""Tests for lbm_solver.py — D2Q9 Lattice Boltzmann solver."""

import numpy as np
import pytest

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


def test_block_obstacle_has_positive_drag_and_no_lift():
    # Regression: calculate_forces used to return the force on the FLUID,
    # so drag came out negative and the HUD showed e.g. "Drag Force: -X N".
    # Drag on the obstacle must point downstream (+x); a symmetric block
    # in the middle of the flow has no lift by symmetry.
    s = make_solver()
    mask = np.zeros((s.ny, s.nx), dtype=bool)
    mask[8:12, 15:20] = True
    s.set_obstacle(mask)
    for _ in range(100):
        s.step()
    assert np.isfinite([s.lift, s.drag]).all()
    assert s.drag > 0
    assert s.lift == pytest.approx(0.0, abs=1e-9)


def test_lift_sign_follows_angle_of_attack():
    # NACA 2412 is cambered, so it lifts a little even at 0 degrees, and
    # lift grows with angle of attack (nose-up on screen = positive).
    from airfoils import generate_naca4

    def lift_at(aoa):
        s = LBMSolver(nx=150, ny=60, tau=0.55)
        mask = generate_naca4((60, 150), 150 // 4, 60 // 2, 40, "2412", aoa)
        s.set_obstacle(mask)
        for _ in range(300):
            s.step()
        return s.lift, s.drag

    lift_neg, drag_neg = lift_at(-10)
    lift_pos, drag_pos = lift_at(10)
    assert lift_pos > 0 > lift_neg
    # Drag stays positive (downstream) at either angle of attack.
    assert drag_pos > 0 and drag_neg > 0


def test_no_obstacle_means_no_forces():
    s = make_solver()
    for _ in range(50):
        s.step()
    assert s.lift == 0.0
    assert s.drag == 0.0


def test_set_obstacle_rejects_wrong_shape():
    # A wrong-shaped mask used to fail later with a cryptic boolean-index
    # error from inside step(); reject it at the call site instead.
    s = make_solver()
    with pytest.raises(ValueError, match="does not match solver grid"):
        s.set_obstacle(np.zeros((5, 5), dtype=bool))
    # The solver is still usable after the rejected call.
    s.set_obstacle(np.zeros((s.ny, s.nx), dtype=bool))
    s.step()
    assert np.isfinite(s.rho).all()
