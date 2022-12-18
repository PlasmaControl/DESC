"""Tests for perturbation functions."""

import numpy as np
import pytest

import desc.examples
from desc.equilibrium import EquilibriaFamily
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

    objective = get_equilibrium_objective()
    constraints = get_fixed_boundary_constraints()

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
def test_optimal_perturb():
    """Test that a single step of optimal_perturb doesn't mess things up."""
    eq0 = desc.examples.get("DSHAPE")
    eq0.change_resolution(N=1, N_grid=5)
    objective = ObjectiveFunction(
        ToroidalCurrent(grid=QuadratureGrid(eq0.L, eq0.M, eq0.N), target=0, weight=1)
    )
    constraint = ObjectiveFunction(ForceBalance(target=0))

    objective.build(eq0)
    constraint.build(eq0)

    idxR = np.nonzero(
        np.logical_and(
            (np.abs(eq0.surface.R_basis.modes[:, 1:]) <= 2).all(axis=1),
            (np.abs(eq0.surface.R_basis.modes[:, 1:]) > 0).all(axis=1),
        )
    )[0]
    idxZ = np.nonzero(
        np.logical_and(
            (np.abs(eq0.surface.Z_basis.modes[:, 1:]) <= 2).all(axis=1),
            (np.abs(eq0.surface.Z_basis.modes[:, 1:]) > 0).all(axis=1),
        )
    )[0]
    Rb_lmn = np.zeros(eq0.surface.R_lmn.size).astype(bool)
    Zb_lmn = np.zeros(eq0.surface.Z_lmn.size).astype(bool)
    Rb_lmn[idxR] = True
    Zb_lmn[idxZ] = True

    tr_ratio = [3e-2, 1e-1]

    # 1st order perturbation
    eq1 = optimal_perturb(
        eq0,
        constraint,
        objective,
        dRb=Rb_lmn,
        dZb=Zb_lmn,
        order=1,
        tr_ratio=tr_ratio,
        verbose=1,
        copy=True,
    )[0]

    # check that objective function was actually reduced
    assert objective.compute_scalar(objective.x(eq1)) < objective.compute_scalar(
        objective.x(eq0)
    )
    # check that new equilibrium still has nested surfaces
    assert eq1.is_nested()

    # 2nd order perturbation
    eq2 = optimal_perturb(
        eq0,
        constraint,
        objective,
        dRb=Rb_lmn,
        dZb=Zb_lmn,
        order=2,
        tr_ratio=tr_ratio,
        verbose=1,
        copy=True,
    )[0]

    # check that higher order reduces objective function by more
    assert objective.compute_scalar(objective.x(eq2)) < objective.compute_scalar(
        objective.x(eq1)
    )
    # check that new equilibrium still has nested surfaces
    assert eq2.is_nested()

    # check that R_lmn & Z_lmn coefficients are consistent with the boundary surface
    surf1 = eq2.get_surface_at(1)  # surface from R_lmn & Z_lmn
    surf2 = eq2.surface  # surface from perturbed Rb_lmn & Zb_lmn
    np.testing.assert_allclose(surf1.R_lmn, surf2.R_lmn, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(surf1.Z_lmn, surf2.Z_lmn, atol=1e-12, rtol=1e-12)
