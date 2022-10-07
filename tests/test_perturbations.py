import numpy as np
import pytest

from desc.equilibrium import EquilibriaFamily
from desc.grid import ConcentricGrid
from desc.objectives import get_equilibrium_objective, get_fixed_boundary_constraints
from desc.perturbations import perturb


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
    eq0 = perturb(
        eq,
        objective,
        constraints,
        dp=dp,
        tr_ratio=tr_ratio,
        order=0,
        verbose=2,
        copy=True,
    )
    eq1 = perturb(
        eq,
        objective,
        constraints,
        dp=dp,
        tr_ratio=tr_ratio,
        order=1,
        verbose=2,
        copy=True,
    )
    eq2 = perturb(
        eq,
        objective,
        constraints,
        dp=dp,
        tr_ratio=tr_ratio,
        order=2,
        verbose=2,
        copy=True,
    )
    eq3 = perturb(
        eq,
        objective,
        constraints,
        dp=dp,
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
    data0 = eq0.compute("|F|", grid)
    data1 = eq1.compute("|F|", grid)
    data2 = eq2.compute("|F|", grid)
    data3 = eq3.compute("|F|", grid)
    dataS = eqS.compute("|F|", grid)

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
