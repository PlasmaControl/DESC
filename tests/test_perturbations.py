import numpy as np

from desc.equilibrium import EquilibriaFamily
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

    eq0 = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]

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
    dp = np.zeros_like(eq0.p_l)
    dp[np.array([0, 2])] = 8e3 * np.array([1, -1])

    eq1 = perturb(eq0, objective, dp=dp, order=1, verbose=2, copy=True)
    eq2 = perturb(eq0, objective, dp=dp, order=2, verbose=2, copy=True)
    eq3 = perturb(eq0, objective, dp=dp, order=3, verbose=2, copy=True)

    # solve for "true" high-beta solution
    eq = eq3.copy()
    eq.solve(objective=objective, ftol=1e-2, verbose=3)

    R0 = eq0.R_lmn[np.where((eq0.R_basis.modes == [0, 0, 0]).all(axis=1))[0]][0]
    R1 = eq1.R_lmn[np.where((eq1.R_basis.modes == [0, 0, 0]).all(axis=1))[0]][0]
    R2 = eq2.R_lmn[np.where((eq2.R_basis.modes == [0, 0, 0]).all(axis=1))[0]][0]
    R3 = eq3.R_lmn[np.where((eq3.R_basis.modes == [0, 0, 0]).all(axis=1))[0]][0]
    R = eq.R_lmn[np.where((eq.R_basis.modes == [0, 0, 0]).all(axis=1))[0]][0]

    # error in Shafranov shift for each perturbation order
    err0 = np.abs(R0 - R)
    err1 = np.abs(R1 - R)
    err2 = np.abs(R2 - R)
    err3 = np.abs(R3 - R)

    assert err1 < err0
    assert err2 < err1
    assert err3 < err2
