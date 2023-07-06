"""Tests for perturbation functions."""

import numpy as np
import pytest

import desc.examples
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import ConcentricGrid, QuadratureGrid
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    ToroidalCurrent,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)
from desc.perturbations import optimal_perturb, perturb


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.solve
def test_perturbation_orders(SOLOVEV):
    """Test that higher-order perturbations are more accurate."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

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
    eqS.solve(objective=objective, ftol=1e-2, verbose=3)

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
    constraints = get_fixed_boundary_constraints(eq=eq, iota=False)

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
    # R_lmn etc that are inconsistent with Rb_lmn due to recovering x with the wrong
    # particular solution. Here we do a simple test to ensure the interior and boundary
    # agree
    eq1 = desc.examples.get("DSHAPE")
    eq1.change_resolution(N=1, N_grid=5)
    objective = ObjectiveFunction(
        ToroidalCurrent(
            eq=eq1, grid=QuadratureGrid(eq1.L, eq1.M, eq1.N), target=0, weight=1
        )
    )
    constraint = ObjectiveFunction(ForceBalance(eq=eq1, target=0))

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
    surf1 = eq1.get_surface_at(1)
    # this is the surface from perturbed coefficients
    surf2 = eq1.surface

    np.testing.assert_allclose(surf1.R_lmn, surf2.R_lmn, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(surf1.Z_lmn, surf2.Z_lmn, atol=1e-12, rtol=1e-12)
