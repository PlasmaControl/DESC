import numpy as np
from desc.equilibrium import EquilibriaFamily
from desc.objectives import (
    ObjectiveFunction,
    AspectRatio,
    get_fixed_boundary_constraints,
)
from desc.vmec import VMECIO
from desc.vmec_utils import vmec_boundary_subspace


# compare results to VMEC solution


def test_SOLOVEV_results(SOLOVEV):
    """Tests that the SOLOVEV example gives the same result as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    rho_err, theta_err = VMECIO.area_difference_vmec(eq, SOLOVEV["vmec_nc_path"])

    np.testing.assert_allclose(rho_err, 0, atol=1e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-5)


def test_DSHAPE_results(DSHAPE):
    """Tests that the DSHAPE example gives the same result as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    rho_err, theta_err = VMECIO.area_difference_vmec(eq, DSHAPE["vmec_nc_path"])

    np.testing.assert_allclose(rho_err, 0, atol=2e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-5)


def test_HELIOTRON_results(HELIOTRON):
    """Tests that the HELIOTRON example gives the same result as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(HELIOTRON["desc_h5_path"]))[-1]
    rho_err, theta_err = VMECIO.area_difference_vmec(eq, HELIOTRON["vmec_nc_path"])

    np.testing.assert_allclose(rho_err.mean(), 0, atol=1e-2)
    np.testing.assert_allclose(theta_err.mean(), 0, atol=2e-2)


# run optimization


def test_1d_optimization(SOLOVEV):
    """Tests 1D optimization for target aspect ratio."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    objective = ObjectiveFunction(
        AspectRatio(target=3),
    )
    constraints = get_fixed_boundary_constraints()
    perturb_options = {"dZb": True, "subspace": vmec_boundary_subspace(eq, ZBS=[0, 1])}
    eq = eq.optimize(objective, constraints, perturb_options=perturb_options)

    np.testing.assert_allclose(eq.compute("V")["R0/a"], 3)
