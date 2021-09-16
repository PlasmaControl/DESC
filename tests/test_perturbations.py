import numpy as np

from desc.equilibrium import EquilibriaFamily
from desc.grid import ConcentricGrid
from desc.objectives import (
    ObjectiveFunction,
    FixedBoundaryR,
    FixedBoundaryZ,
    FixedPressure,
    FixedIota,
    FixedPsi,
    LCFSBoundary,
    RadialForceBalance,
    HelicalForceBalance,
)
from desc.perturbations import perturb


def test_perturbation_orders(SOLOVEV):
    """Test that higher-order perturbations are more accurate."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

    objectives = (RadialForceBalance(), HelicalForceBalance())
    constraints = (
        FixedBoundaryR(),
        FixedBoundaryZ(),
        FixedPressure(),
        FixedIota(),
        FixedPsi(),
        LCFSBoundary(),
    )
    objective = ObjectiveFunction(objectives, constraints)

    # perturb pressure
    dp = np.zeros_like(eq.p_l)
    dp[np.array([0, 2])] = 8e3 * np.array([1, -1])
    eq0 = perturb(eq, objective, dp=dp, order=0, verbose=2, copy=True)
    eq1 = perturb(eq, objective, dp=dp, order=1, verbose=2, copy=True)
    eq2 = perturb(eq, objective, dp=dp, order=2, verbose=2, copy=True)
    eq3 = perturb(eq, objective, dp=dp, order=3, verbose=2, copy=True)

    # solve for "true" high-beta solution
    eqS = eq3.copy()
    eqS.solve(objective=objective, ftol=1e-2, verbose=3)

    # evaluate equilibrium force balance
    grid = ConcentricGrid(eq.L, eq.M, eq.N, eq.NFP, rotation="cos", node_pattern=None)
    data0 = eq0.compute("|F|", grid)
    data1 = eq1.compute("|F|", grid)
    data2 = eq2.compute("|F|", grid)
    data3 = eq3.compute("|F|", grid)
    dataS = eqS.compute("|F|", grid)

    # total error in Newtons throughout plasma volume
    f0 = np.sum(data0["|F|"] * np.abs(data0["sqrt(g)"]))
    f1 = np.sum(data1["|F|"] * np.abs(data1["sqrt(g)"]))
    f2 = np.sum(data2["|F|"] * np.abs(data2["sqrt(g)"]))
    f3 = np.sum(data3["|F|"] * np.abs(data3["sqrt(g)"]))
    fS = np.sum(dataS["|F|"] * np.abs(dataS["sqrt(g)"]))

    # error for each perturbation order
    err0 = np.abs(f0 - fS)
    err1 = np.abs(f1 - fS)
    err2 = np.abs(f2 - fS)
    err3 = np.abs(f3 - fS)

    # assert err1 < err0
    assert err2 < err1
    assert err3 < err2
