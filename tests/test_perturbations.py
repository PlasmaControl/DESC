"""Tests for perturbation functions."""

import numpy as np
import pytest

import desc.examples
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZCurve
from desc.grid import ConcentricGrid, QuadratureGrid
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    ToroidalCurrent,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)
from desc.perturbations import optimal_perturb, perturb


@pytest.mark.regression
@pytest.mark.slow
def test_perturbation_orders():
    """Test that higher-order perturbations are more accurate."""
    eq = desc.examples.get("SOLOVEV")

    objective = get_equilibrium_objective(eq=eq)
    constraints = get_fixed_boundary_constraints(eq=eq)

    # perturb pressure
    tr_ratio = [0.01, 0.25, 0.25]
    dp = np.zeros_like(eq.p_l)
    dp[np.array([0, 2])] = 8e3 * np.array([1, -1])
    deltas = {"p_l": dp}
    eq0 = perturb(
        eq,
        objective,
        constraints,
        deltas,
        tr_ratio=tr_ratio,
        order=0,
        verbose=2,
        copy=True,
    )
    eq1 = perturb(
        eq,
        objective,
        constraints,
        deltas,
        tr_ratio=tr_ratio,
        order=1,
        verbose=2,
        copy=True,
    )
    eq2 = perturb(
        eq,
        objective,
        constraints,
        deltas,
        tr_ratio=tr_ratio,
        order=2,
        verbose=2,
        copy=True,
    )
    eq3 = perturb(
        eq,
        objective,
        constraints,
        deltas,
        tr_ratio=tr_ratio,
        order=3,
        verbose=2,
        copy=True,
    )

    # solve for "true" high-beta solution
    eqS = eq3.copy()
    eqS.solve(ftol=1e-2, verbose=3)

    # evaluate equilibrium force balance
    grid = ConcentricGrid(2 * eq.L, 2 * eq.M, 2 * eq.N, eq.NFP, node_pattern="jacobi")
    data0 = eq0.compute("|F|", grid=grid)
    data1 = eq1.compute("|F|", grid=grid)
    data2 = eq2.compute("|F|", grid=grid)
    data3 = eq3.compute("|F|", grid=grid)
    dataS = eqS.compute("|F|", grid=grid)

    # total error in Newtons throughout plasma volume
    f0 = np.sum(data0["|F|"] * np.abs(data0["sqrt(g)"]) * grid.weights)
    f1 = np.sum(data1["|F|"] * np.abs(data1["sqrt(g)"]) * grid.weights)
    f2 = np.sum(data2["|F|"] * np.abs(data2["sqrt(g)"]) * grid.weights)
    f3 = np.sum(data3["|F|"] * np.abs(data3["sqrt(g)"]) * grid.weights)
    fS = np.sum(dataS["|F|"] * np.abs(dataS["sqrt(g)"]) * grid.weights)

    assert f1 < f0
    assert f2 < f1
    assert f3 < f2
    assert fS < f3


@pytest.mark.unit
def test_perturb_with_float_without_error():
    """Test that perturb works without error if only a single float is passed."""
    # PR #
    # fixed bug where np.concatenate( [float] ) was called resulting in error that
    # np.concatenate cannot concatenate 0-D arrays. This test exercises the fix.
    eq = Equilibrium()
    objective = get_equilibrium_objective(eq=eq)
    constraints = get_fixed_boundary_constraints(eq=eq)

    # perturb Psi with a float
    deltas = {"Psi": float(eq.Psi)}
    eq = perturb(
        eq,
        objective,
        constraints,
        deltas,
        order=0,
        verbose=2,
        copy=True,
    )


@pytest.mark.unit
def test_optimal_perturb():
    """Test that a single step of optimal_perturb doesn't mess things up."""
    # as of v0.6.1, the recover operation from optimal_perturb would give
    # R_lmn etc. that are inconsistent with Rb_lmn due to recovering x with the wrong
    # particular solution. Here we do a simple test to ensure the interior and boundary
    # agree
    eq1 = desc.examples.get("DSHAPE")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq1.change_resolution(3, 3, 0, 6, 6, 0)
    eq1.change_resolution(N=1, N_grid=5)
    eq1.surface = eq1.get_surface_at(1.0)
    objective = ObjectiveFunction(
        ToroidalCurrent(
            eq=eq1, grid=QuadratureGrid(eq1.L, eq1.M, eq1.N), target=0, weight=1
        ),
        use_jit=False,
    )
    constraint = ObjectiveFunction(ForceBalance(eq=eq1, target=0), use_jit=False)

    objective.build()
    constraint.build()

    R_modes = np.zeros(eq1.surface.R_lmn.size).astype(bool)
    Z_modes = np.zeros(eq1.surface.Z_lmn.size).astype(bool)

    Rmask = np.logical_and(
        abs(eq1.surface.R_basis.modes[:, 1]) < 3,
        np.logical_and(
            abs(eq1.surface.R_basis.modes[:, 1]) > 0,
            abs(eq1.surface.R_basis.modes[:, 2]) > 0,
        ),
    )
    Zmask = np.logical_and(
        abs(eq1.surface.Z_basis.modes[:, 1]) < 3,
        np.logical_and(
            abs(eq1.surface.Z_basis.modes[:, 1]) > 0,
            abs(eq1.surface.Z_basis.modes[:, 2]) > 0,
        ),
    )
    R_modes[Rmask] = True
    Z_modes[Zmask] = True

    eq2 = optimal_perturb(
        eq1,
        constraint,
        objective,
        dRb=R_modes,
        dZb=Z_modes,
        order=1,
        tr_ratio=[0.03, 0.1],
        verbose=1,
        copy=True,
    )[0]

    assert eq2.is_nested()
    # recompute surface from R_lmn etc.
    surf1 = eq2.get_surface_at(1)
    # this is the surface from perturbed coefficients
    surf2 = eq2.surface

    np.testing.assert_allclose(surf1.R_lmn, surf2.R_lmn, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(surf1.Z_lmn, surf2.Z_lmn, atol=1e-12, rtol=1e-12)


@pytest.mark.unit
def test_perturb_axis():
    """Test that perturbing the axis gives correct soln and changes boundary."""
    ax1 = FourierRZCurve(10.0, 0.0, sym=False)
    eq = Equilibrium(L=2, M=2, N=0, sym=False, axis=ax1)
    ax2 = FourierRZCurve(10.25, 0.25, sym=False)

    from desc.perturbations import get_deltas

    deltas = get_deltas({"axis": eq.axis}, {"axis": ax2})

    eq_new = eq.perturb(deltas, copy=True)

    assert eq_new.is_nested()

    np.testing.assert_allclose(eq_new.axis.R_n, ax2.R_n)
    np.testing.assert_allclose(eq_new.axis.Z_n, ax2.Z_n)
    np.testing.assert_allclose(eq_new.get_axis().R_n, ax2.R_n)
    np.testing.assert_allclose(eq_new.get_axis().Z_n, ax2.Z_n)

    assert np.max(np.abs(eq.Rb_lmn - eq_new.Rb_lmn)) > 0.2
    assert np.max(np.abs(eq.Zb_lmn - eq_new.Zb_lmn)) > 0.2

    eq.change_resolution(N=2)
    assert eq.axis.N == 2
    assert ax2.N == 0
    eq.axis = ax2
    assert eq.axis.N == 2


@pytest.mark.unit
def test_perturb_to_lower_resolution_profile():
    """Test that perturbing to a lower-resolution profile works correctly."""
    from desc.perturbations import get_deltas

    # related to gh issue #1974
    eq = desc.equilibrium.Equilibrium(L=5, M=3)
    eq.current = None
    eq.iota = desc.profiles.PowerSeriesProfile(0.42)

    p1 = desc.profiles.PowerSeriesProfile(np.array([1e4, 0, -1e4]), modes=[0, 1, 2])
    p1.change_resolution(L=eq.L)
    eq.pressure = p1
    p2 = desc.profiles.PowerSeriesProfile(np.array([0.5e4, -0.5e4]), modes=[0, 2])

    # p2 has lower resolution than eq.L which is not possible. If
    # we wanted to assign it to eq.pressure, it would change automatically
    with pytest.warns(UserWarning, match="has lower resolution"):
        deltas = get_deltas({"pressure": eq.pressure}, {"pressure": p2})

    with pytest.raises(TypeError):
        _ = eq.perturb(deltas, order=0, copy=True)

    eq2 = eq.copy()
    # This assignment will fix the resolution issue
    eq2.pressure = p2
    deltas = get_deltas({"pressure": eq.pressure}, {"pressure": eq2.pressure})
    _ = eq.perturb(deltas, order=0, copy=True)
