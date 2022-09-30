import numpy as np
import pytest

import desc.examples
from .utils import area_difference_vmec, area_difference_desc
from desc.io import load
from desc.grid import LinearGrid
from desc.profiles import PowerSeriesProfile
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    RadialForceBalance,
    HelicalForceBalance,
    QuasisymmetryTwoTerm,
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixCurrent,
    FixPsi,
)
from desc.optimize import Optimizer
from desc.plotting import plot_boozer_surface
from desc.vmec_utils import vmec_boundary_subspace


@pytest.mark.regression
@pytest.mark.solve
def test_SOLOVEV_vacuum(SOLOVEV_vac):
    """Tests that the SOLOVEV vacuum example gives no rotational transform."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV_vac["desc_h5_path"]))[-1]
    data = eq.compute("|J|")

    np.testing.assert_allclose(data["iota"], 0, atol=1e-16)
    np.testing.assert_allclose(data["|J|"], 0, atol=1e-3)


@pytest.mark.regression
@pytest.mark.solve
def test_SOLOVEV_results(SOLOVEV):
    """Tests that the SOLOVEV example gives the same result as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, SOLOVEV["vmec_nc_path"])

    np.testing.assert_allclose(rho_err, 0, atol=1e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-5)


@pytest.mark.regression
@pytest.mark.solve
def test_DSHAPE_results(DSHAPE):
    """Tests that the DSHAPE examples gives the same results as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, DSHAPE["vmec_nc_path"])
    np.testing.assert_allclose(rho_err, 0, atol=2e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-5)


@pytest.mark.regression
@pytest.mark.solve
def test_DSHAPE_current_results(DSHAPE_current):
    """Tests that the DSHAPE with fixed current gives the same results as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, DSHAPE_current["vmec_nc_path"])
    np.testing.assert_allclose(rho_err, 0, atol=2e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-5)


@pytest.mark.regression
@pytest.mark.solve
def test_HELIOTRON_results(HELIOTRON):
    """Tests that the HELIOTRON examples gives the same results as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(HELIOTRON["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, HELIOTRON["vmec_nc_path"])
    np.testing.assert_allclose(rho_err.mean(), 0, atol=1e-2)
    np.testing.assert_allclose(theta_err.mean(), 0, atol=2e-2)


@pytest.mark.regression
@pytest.mark.solve
def test_HELIOTRON_vac_results(HELIOTRON_vac):
    """Tests that the HELIOTRON examples gives the same results as VMEC."""

    eq = EquilibriaFamily.load(load_from=str(HELIOTRON_vac["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, HELIOTRON_vac["vmec_nc_path"])
    np.testing.assert_allclose(rho_err.mean(), 0, atol=1e-2)
    np.testing.assert_allclose(theta_err.mean(), 0, atol=2e-2)
    curr = eq.get_profile("current")
    np.testing.assert_allclose(curr(np.linspace(0, 1, 20)), atol=1e-8)


@pytest.mark.regression
@pytest.mark.solve
def test_precise_QH_results(precise_QH):
    """Tests that the precise QH initial solve gives the same results as a base case."""

    eq1 = EquilibriaFamily.load(load_from=str(precise_QH["desc_h5_path"]))[-1]
    eq2 = EquilibriaFamily.load(load_from=str(precise_QH["output_path"]))[-1]
    rho_err, theta_err = area_difference_desc(eq1, eq2)
    np.testing.assert_allclose(rho_err, 0, atol=1e-6)
    np.testing.assert_allclose(theta_err, 0, atol=1e-6)


@pytest.mark.regression
@pytest.mark.solve
def test_HELIOTRON_vac2_results(HELIOTRON_vac, HELIOTRON_vac2):
    """Tests that the 2 methods for solving vacuum give the same results."""

    eq1 = EquilibriaFamily.load(load_from=str(HELIOTRON_vac["desc_h5_path"]))[-1]
    eq2 = EquilibriaFamily.load(load_from=str(HELIOTRON_vac2["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_desc(eq1, eq2)
    np.testing.assert_allclose(rho_err[:, 3:], 0, atol=1e-2)
    np.testing.assert_allclose(theta_err, 0, atol=1e-5)
    curr1 = eq1.get_profile("current")
    curr2 = eq2.get_profile("current")
    iota1 = eq1.get_profile("iota")
    iota2 = eq2.get_profile("iota")
    np.testing.assert_allclose(curr1(np.linspace(0, 1, 20)), 0, atol=1e-8)
    np.testing.assert_allclose(curr2(np.linspace(0, 1, 20)), 0, atol=1e-8)
    np.testing.assert_allclose(iota1.params, iota2.params, rtol=1e-1, atol=1e-1)


@pytest.mark.regression
@pytest.mark.solve
def test_force_balance_grids():
    """Compares radial & helical force balance on same vs different grids."""
    # When ConcentricGrid had a rotation option,
    # Radial, HelicalForceBalance defaulted to cos, sin rotation, respectively
    # This test has been kept to increase code coverage.

    def test(iota=False):
        if iota:
            # pick quad here just to increase code coverage
            eq1 = Equilibrium(iota=PowerSeriesProfile(0), sym=True, node_pattern="quad")
            eq2 = Equilibrium(iota=PowerSeriesProfile(0), sym=True, node_pattern="quad")
        else:
            eq1 = Equilibrium(current=PowerSeriesProfile(0), sym=True)
            eq2 = Equilibrium(current=PowerSeriesProfile(0), sym=True)

        res = 3
        eq1.change_resolution(L=res, M=res)
        eq1.L_grid = res
        eq1.M_grid = res
        eq2.change_resolution(L=res, M=res)
        eq2.L_grid = res
        eq2.M_grid = res

        # force balances on the same grids
        obj1 = ObjectiveFunction(ForceBalance())
        eq1.solve(objective=obj1)

        # force balances on different grids
        obj2 = ObjectiveFunction((RadialForceBalance(), HelicalForceBalance()))
        eq2.solve(objective=obj2)

        np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn, atol=5e-4)
        np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn, atol=5e-4)
        np.testing.assert_allclose(eq1.L_lmn, eq2.L_lmn, atol=2e-3)

    test(iota=True)
    test(iota=False)


@pytest.mark.regression
@pytest.mark.solve
def test_1d_optimization(SOLOVEV):
    """Tests 1D optimization for target aspect ratio."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    objective = ObjectiveFunction(AspectRatio(target=2.5))
    constraints = (
        ForceBalance(),
        FixBoundaryR(),
        FixBoundaryZ(modes=eq.surface.Z_basis.modes[0:-1, :]),
        FixPressure(),
        FixIota(),
        FixPsi(),
    )
    options = {"perturb_options": {"order": 1}}
    with pytest.warns(UserWarning):
        eq.optimize(objective, constraints, options=options)

    np.testing.assert_allclose(eq.compute("V")["R0/a"], 2.5)


@pytest.mark.regression
@pytest.mark.solve
def test_1d_optimization_old(SOLOVEV):
    """Tests 1D optimization for target aspect ratio."""

    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    objective = ObjectiveFunction(AspectRatio(target=2.5))
    eq._optimize(
        objective,
        copy=False,
        solve_options={"verbose": 0},
        perturb_options={
            "dZb": True,
            "subspace": vmec_boundary_subspace(eq, ZBS=[0, 1]),
        },
    )

    np.testing.assert_allclose(eq.compute("V")["R0/a"], 2.5)


def run_qh_step(n, eq):
    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(helicity=(1, -eq.NFP), grid=grid),
            AspectRatio(target=8, weight=1e1),
        ),
        verbose=0,
    )
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > n + 1, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > n + 1, :
    ]
    constraints = (
        ForceBalance(),
        FixBoundaryR(modes=R_modes),
        FixBoundaryZ(modes=Z_modes),
        FixPressure(),
        FixCurrent(),
        FixPsi(),
    )
    optimizer = Optimizer("lsq-exact")
    eq1, history = eq.optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=50,
        verbose=3,
        copy=True,
        options={
            "initial_trust_radius": 0.5,
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0},
        },
    )

    return eq1


@pytest.mark.regression
@pytest.mark.solve
def test_qh_optimization1():
    """Tests precise QH optimization, step 1."""
    eq0 = load(".//tests//inputs//precise_QH_step0.h5")[-1]
    eq1 = load(".//tests//inputs//precise_QH_step1.h5")
    eq1a = run_qh_step(0, eq0)
    rho_err, theta_err = area_difference_desc(eq1, eq1a)
    np.testing.assert_allclose(rho_err, 0, atol=1e-6)
    np.testing.assert_allclose(theta_err, 0, atol=1e-6)


@pytest.mark.regression
@pytest.mark.solve
def test_qh_optimization2():
    """Tests precise QH optimization, step 2."""
    eq1 = load(".//tests//inputs//precise_QH_step1.h5")
    eq2 = load(".//tests//inputs//precise_QH_step2.h5")
    eq2a = run_qh_step(1, eq1)
    rho_err, theta_err = area_difference_desc(eq2, eq2a)
    np.testing.assert_allclose(rho_err, 0, atol=1e-6)
    np.testing.assert_allclose(theta_err, 0, atol=1e-6)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=15)
def test_qh_optimization3():
    """Tests precise QH optimization, step 3."""
    eq2 = load(".//tests//inputs//precise_QH_step2.h5")
    eq3 = load(".//tests//inputs//precise_QH_step3.h5")
    eq3a = run_qh_step(2, eq2)
    rho_err, theta_err = area_difference_desc(eq3, eq3a)
    np.testing.assert_allclose(rho_err, 0, atol=1e-6)
    np.testing.assert_allclose(theta_err, 0, atol=1e-6)

    grid = LinearGrid(M=eq3a.M_grid, N=eq3a.N_grid, NFP=eq3a.NFP, sym=False, rho=1.0)
    data = eq3a.compute("|B|_mn", grid, M_booz=eq3a.M, N_booz=eq3a.N)
    idx = np.where(np.abs(data["B modes"][:, 1] / data["B modes"][:, 2]) != 1)[0]
    B_asym = np.sort(np.abs(data["|B|_mn"][idx]))[:-1]
    np.testing.assert_array_less(B_asym, 2e-3)
    fig, ax = plot_boozer_surface(eq3a)
    return fig


class TestGetExample:
    """Tests for desc.examples.get."""

    @pytest.mark.unit
    def test_example_get_eq(self):
        """Test getting a single equilibrium."""
        eq = desc.examples.get("SOLOVEV")
        assert eq.Psi == 1

    @pytest.mark.unit
    def test_example_get_eqf(self):
        """Test getting full equilibria family."""
        eqf = desc.examples.get("DSHAPE", "all")
        np.testing.assert_allclose(eqf[0].pressure.params, 0)

    @pytest.mark.unit
    def test_example_get_boundary(self):
        """Test getting boundary surface."""
        surf = desc.examples.get("HELIOTRON", "boundary")
        np.testing.assert_allclose(surf.R_lmn[surf.R_basis.get_idx(0, 1, 1)], -0.3)

    @pytest.mark.unit
    def test_example_get_pressure(self):
        """Test getting pressure profile."""
        pres = desc.examples.get("ATF", "pressure")
        np.testing.assert_allclose(pres.params[:5], [5e5, -1e6, 5e5, 0, 0])

    @pytest.mark.unit
    def test_example_get_iota(self):
        """Test getting iota profile."""
        iota = desc.examples.get("NCSX", "iota")
        np.testing.assert_allclose(
            iota.params[:5],
            [
                3.49197642e-01,
                6.81105159e-01,
                -1.29781695e00,
                2.07888586e00,
                -1.15800135e00,
            ],
        )

    @pytest.mark.unit
    def test_example_get_current(self):
        """Test getting current profile."""
        current = desc.examples.get("QAS", "current")
        np.testing.assert_allclose(
            current.params[:11],
            [
                0.00000000e00,
                -5.30230329e03,
                -4.65196499e05,
                2.31960013e06,
                -1.20570566e07,
                4.17520547e07,
                -9.51373229e07,
                1.38268651e08,
                -1.23703891e08,
                6.24782996e07,
                -1.36284423e07,
            ],
        )
