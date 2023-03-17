"""Regression tests to verify that DESC agrees with VMEC and itself.

Computes several benchmark equilibria and compares the solutions by measuring the
difference in areas between constant theta and rho contours.
"""

import numpy as np
import pytest
from qic import Qic
from qsc import Qsc

import desc.examples
from desc.compute.utils import compress
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.io import load
from desc.objectives import (
    AspectRatio,
    CurrentDensity,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    HelicalForceBalance,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    RadialForceBalance,
    get_fixed_boundary_constraints,
)
from desc.objectives.utils import get_NAE_constraints
from desc.optimize import Optimizer
from desc.plotting import plot_boozer_surface
from desc.profiles import PowerSeriesProfile
from desc.vmec_utils import vmec_boundary_subspace

from .utils import area_difference_desc, area_difference_vmec


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
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)


@pytest.mark.regression
@pytest.mark.solve
def test_DSHAPE_results(DSHAPE):
    """Tests that the DSHAPE examples gives the same results as VMEC."""
    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, DSHAPE["vmec_nc_path"])
    np.testing.assert_allclose(rho_err, 0, atol=2e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)


@pytest.mark.regression
@pytest.mark.solve
def test_DSHAPE_current_results(DSHAPE_current):
    """Tests that the DSHAPE with fixed current gives the same results as VMEC."""
    eq = EquilibriaFamily.load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, DSHAPE_current["vmec_nc_path"])
    np.testing.assert_allclose(rho_err, 0, atol=3e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)


@pytest.mark.regression
@pytest.mark.solve
def test_HELIOTRON_results(HELIOTRON):
    """Tests that the HELIOTRON examples gives the same results as VMEC."""
    eq = EquilibriaFamily.load(load_from=str(HELIOTRON["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, HELIOTRON["vmec_nc_path"])
    np.testing.assert_allclose(rho_err.mean(), 0, atol=2e-2)
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
    np.testing.assert_allclose(curr(np.linspace(0, 1, 20)), 0, atol=1e-8)


@pytest.mark.regression
@pytest.mark.solve
def test_precise_QH_results(precise_QH):
    """Tests that the precise QH initial solve gives the same results as a base case."""
    eq1 = EquilibriaFamily.load(load_from=str(precise_QH["desc_h5_path"]))[-1]
    eq2 = EquilibriaFamily.load(load_from=str(precise_QH["output_path"]))[-1]
    rho_err, theta_err = area_difference_desc(eq1, eq2)
    np.testing.assert_allclose(rho_err, 0, atol=1e-2)
    np.testing.assert_allclose(theta_err, 0, atol=1e-2)


@pytest.mark.regression
@pytest.mark.solve
def test_HELIOTRON_vac2_results(HELIOTRON_vac, HELIOTRON_vac2):
    """Tests that the 2 methods for solving vacuum give the same results."""
    eq1 = EquilibriaFamily.load(load_from=str(HELIOTRON_vac["desc_h5_path"]))[-1]
    eq2 = EquilibriaFamily.load(load_from=str(HELIOTRON_vac2["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_desc(eq1, eq2)
    np.testing.assert_allclose(rho_err[:, 3:], 0, atol=1e-2)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)
    curr1 = eq1.get_profile("current")
    curr2 = eq2.get_profile("current")
    iota1 = eq1.get_profile("iota")
    iota2 = eq2.get_profile("iota")
    np.testing.assert_allclose(curr1(np.linspace(0, 1, 20)), 0, atol=1e-8)
    np.testing.assert_allclose(curr2(np.linspace(0, 1, 20)), 0, atol=1e-8)
    np.testing.assert_allclose(iota1.params, iota2.params, rtol=1e-1, atol=1e-1)


@pytest.mark.regression
def test_force_balance_grids():
    """Compares radial & helical force balance on same vs different grids."""
    # When ConcentricGrid had a rotation option, RadialForceBalance, HelicalForceBalance
    # defaulted to cos, sin rotation, respectively.
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
def test_solve_bounds():
    """Tests optimizing with bounds=(lower bound, upper bound)."""
    # decrease resolution and double pressure so no longer in force balance
    eq = desc.examples.get("DSHAPE")
    eq.change_resolution(L=eq.M, L_grid=eq.M_grid)
    eq.p_l *= 2

    # target force balance residuals with |F| <= 1e3 N
    obj = ObjectiveFunction(
        ForceBalance(normalize=False, normalize_target=False, bounds=(-1e3, 1e3)), eq=eq
    )
    eq.solve(objective=obj, ftol=1e-16, xtol=1e-16, maxiter=100, verbose=3)

    # check that all errors are nearly 0, since residual values are within target bounds
    f = obj.compute(obj.x(eq))
    np.testing.assert_allclose(f, 0, atol=1e-4)


@pytest.mark.regression
def test_1d_optimization(SOLOVEV):
    """Tests 1D optimization for target aspect ratio."""
    eq = desc.examples.get("SOLOVEV")
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
        eq.optimize(objective, constraints, optimizer="lsq-exact", options=options)

    np.testing.assert_allclose(eq.compute("R0/a")["R0/a"], 2.5, rtol=2e-4)


@pytest.mark.regression
def test_1d_optimization_old():
    """Tests 1D optimization for target aspect ratio."""
    eq = desc.examples.get("SOLOVEV")
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

    np.testing.assert_allclose(eq.compute("R0/a")["R0/a"], 2.5, rtol=2e-4)


def run_qh_step(n, eq):
    """Run 1 step of the precise QH optimization example from Landreman & Paul."""
    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(helicity=(1, eq.NFP), grid=grid),
            AspectRatio(target=8, weight=1e2),
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
    optimizer = Optimizer("proximal-lsq-exact")
    eq1, history = eq.optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=50,
        verbose=3,
        copy=True,
        options={
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
    rho_err, theta_err = area_difference_desc(eq1, eq1a, Nr=1, Nt=1)
    # only need crude tolerances here to make sure the boundaries are
    # similar, the main test is ensuring its not pathological and has good qs
    assert rho_err.mean() < 1

    grid = LinearGrid(M=eq1a.M_grid, N=eq1a.N_grid, NFP=eq1a.NFP, sym=False, rho=1.0)
    data = eq1a.compute(["|B|_mn", "B modes"], grid, M_booz=eq1a.M, N_booz=eq1a.N)
    idx = np.where(np.abs(data["B modes"][:, 1] / data["B modes"][:, 2]) != 1)[0]
    B_asym = np.sort(np.abs(data["|B|_mn"][idx]))[:-1]
    np.testing.assert_array_less(B_asym, 1e-1)


@pytest.mark.regression
@pytest.mark.solve
def test_qh_optimization2():
    """Tests precise QH optimization, step 2."""
    eq1 = load(".//tests//inputs//precise_QH_step1.h5")
    eq2 = load(".//tests//inputs//precise_QH_step2.h5")
    eq2a = run_qh_step(1, eq1)
    rho_err, theta_err = area_difference_desc(eq2, eq2a, Nr=1, Nt=1)
    # only need crude tolerances here to make sure the boundaries are
    # similar, the main test is ensuring its not pathological and has good qs
    assert rho_err.mean() < 1

    grid = LinearGrid(M=eq2a.M_grid, N=eq2a.N_grid, NFP=eq2a.NFP, sym=False, rho=1.0)
    data = eq2a.compute(["|B|_mn", "B modes"], grid, M_booz=eq2a.M, N_booz=eq2a.N)
    idx = np.where(np.abs(data["B modes"][:, 1] / data["B modes"][:, 2]) != 1)[0]
    B_asym = np.sort(np.abs(data["|B|_mn"][idx]))[:-1]
    np.testing.assert_array_less(B_asym, 1e-2)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=15)
def test_qh_optimization3():
    """Tests precise QH optimization, step 3."""
    eq2 = load(".//tests//inputs//precise_QH_step2.h5")
    eq3 = load(".//tests//inputs//precise_QH_step3.h5")
    eq3a = run_qh_step(2, eq2)
    rho_err, theta_err = area_difference_desc(eq3, eq3a, Nr=1, Nt=1)
    # only need crude tolerances here to make sure the boundaries are
    # similar, the main test is ensuring its not pathological and has good qs
    assert rho_err.mean() < 1

    grid = LinearGrid(M=eq3a.M_grid, N=eq3a.N_grid, NFP=eq3a.NFP, sym=False, rho=1.0)
    data = eq3a.compute(["|B|_mn", "B modes"], grid=grid, M_booz=eq3a.M, N_booz=eq3a.N)
    idx = np.where(np.abs(data["B modes"][:, 1] / data["B modes"][:, 2]) != 1)[0]
    B_asym = np.sort(np.abs(data["|B|_mn"][idx]))[:-1]
    np.testing.assert_array_less(B_asym, 2.5e-3)
    fig, ax = plot_boozer_surface(eq3a)
    return fig


@pytest.mark.regression
@pytest.mark.solve
def test_ATF_results(tmpdir_factory):
    """Test automatic continuation method with ATF."""
    output_dir = tmpdir_factory.mktemp("result")
    eq0 = desc.examples.get("ATF")
    eq = Equilibrium(
        Psi=eq0.Psi,
        NFP=eq0.NFP,
        L=eq0.L,
        M=eq0.M,
        N=eq0.N,
        L_grid=eq0.L_grid,
        M_grid=eq0.M_grid,
        N_grid=eq0.N_grid,
        node_pattern=eq0.node_pattern,
        pressure=eq0.pressure,
        iota=eq0.iota,
        surface=eq0.get_surface_at(rho=1),
        sym=eq0.sym,
        spectral_indexing=eq0.spectral_indexing,
    )
    eqf = EquilibriaFamily.solve_continuation_automatic(
        eq,
        verbose=2,
        checkpoint_path=output_dir.join("ATF.h5"),
    )
    eqf = load(output_dir.join("ATF.h5"))
    rho_err, theta_err = area_difference_desc(eq0, eqf[-1])
    np.testing.assert_allclose(rho_err[:, 4:], 0, atol=4e-2)
    np.testing.assert_allclose(theta_err, 0, atol=5e-4)


@pytest.mark.regression
@pytest.mark.solve
def test_ESTELL_results(tmpdir_factory):
    """Test automatic continuation method with ESTELL."""
    output_dir = tmpdir_factory.mktemp("result")
    eq0 = desc.examples.get("ESTELL")
    eq = Equilibrium(
        Psi=eq0.Psi,
        NFP=eq0.NFP,
        L=eq0.L,
        M=eq0.M,
        N=eq0.N,
        L_grid=eq0.L_grid,
        M_grid=eq0.M_grid,
        N_grid=eq0.N_grid,
        node_pattern=eq0.node_pattern,
        pressure=eq0.pressure,
        current=eq0.current,
        surface=eq0.get_surface_at(rho=1),
        sym=eq0.sym,
        spectral_indexing=eq0.spectral_indexing,
    )
    eqf = EquilibriaFamily.solve_continuation_automatic(
        eq,
        verbose=2,
        checkpoint_path=output_dir.join("ESTELL.h5"),
    )
    eqf = load(output_dir.join("ESTELL.h5"))
    rho_err, theta_err = area_difference_desc(eq0, eqf[-1])
    np.testing.assert_allclose(rho_err[:, 3:], 0, atol=5e-2)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)


@pytest.mark.regression
@pytest.mark.solve
def test_simsopt_QH_comparison():
    """Test case that previously stalled before getting to the solution.

    From Matt's comparison with SIMSOPT.
    """
    nfp = 4
    aspect_target = 8.0
    # Initial (m=0, n=nfp) mode of the axis:
    Delta = 0.2
    LMN_resolution = 6
    # Set shape of the initial condition.
    # R_lmn and Z_lmn are the amplitudes. modes_R and modes_Z are the (m,n) pairs.
    surface = FourierRZToroidalSurface(
        R_lmn=[1.0, 1.0 / aspect_target, Delta],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        Z_lmn=[0, 1.0 / aspect_target, Delta],
        modes_Z=[[0, 0], [-1, 0], [0, -1]],
        NFP=nfp,
    )
    # Set up a vacuum field:
    pressure = PowerSeriesProfile(params=[0], modes=[0])
    ndofs_current = 3
    current = PowerSeriesProfile(
        params=np.zeros(ndofs_current),
        modes=2 * np.arange(ndofs_current),
    )
    eq = Equilibrium(
        surface=surface,
        pressure=pressure,
        current=current,
        Psi=np.pi / (aspect_target**2),  # So |B| is ~ 1 T.
        NFP=nfp,
        L=LMN_resolution,
        M=LMN_resolution,
        N=LMN_resolution,
        L_grid=2 * LMN_resolution,
        M_grid=2 * LMN_resolution,
        N_grid=2 * LMN_resolution,
        sym=True,
    )
    # Fix the major radius, and all modes with |m| or |n| > 1:
    R_modes_to_fix = []
    for j in range(eq.surface.R_basis.modes.shape[0]):
        m = eq.surface.R_basis.modes[j, 1]
        n = eq.surface.R_basis.modes[j, 2]
        if (m * n < 0) or (m == 0 and n == 0) or (abs(m) > 1) or (abs(n) > 1):
            R_modes_to_fix.append([0, m, n])
        else:
            print(f"Freeing R mode: m={m}, n={n}")
    R_modes_to_fix = np.array(R_modes_to_fix)
    Z_modes_to_fix = []
    for j in range(eq.surface.Z_basis.modes.shape[0]):
        m = eq.surface.Z_basis.modes[j, 1]
        n = eq.surface.Z_basis.modes[j, 2]
        if (m * n > 0) or (m == 0 and n == 0) or (abs(m) > 1) or (abs(n) > 1):
            Z_modes_to_fix.append([0, m, n])
        else:
            print(f"Freeing Z mode: m={m}, n={n}")
    Z_modes_to_fix = np.array(Z_modes_to_fix)
    eq.solve(
        verbose=3,
        ftol=1e-8,
        constraints=get_fixed_boundary_constraints(profiles=False),
        objective=ObjectiveFunction(objectives=CurrentDensity()),
    )
    ##################################
    # Done creating initial condition.
    # Now define optimization problem.
    ##################################
    constraints = (
        CurrentDensity(),
        FixBoundaryR(modes=R_modes_to_fix),
        FixBoundaryZ(modes=Z_modes_to_fix),
        FixPressure(),
        FixCurrent(),
        FixPsi(),
    )
    # Objective function, for both desc and simsopt:
    # f = (aspect - 8)^2 + (2pi)^{-2} \int dtheta \int d\zeta [f_C(rho=1)]^2
    # grid for quasisymmetry objective:
    grid = LinearGrid(
        M=eq.M,
        N=eq.N,
        rho=[1.0],
        NFP=nfp,
    )
    # Cancel factor of 1/2 in desc objective which is not present in simsopt:
    aspect_weight = np.sqrt(2)
    # Also scale QS weight so objective is approximately independent of grid resolution:
    qs_weight = np.sqrt(len(grid.weights) / (8 * (np.pi**4)))
    objective = ObjectiveFunction(
        (
            AspectRatio(target=aspect_target, weight=aspect_weight, normalize=False),
            QuasisymmetryTwoTerm(
                helicity=(1, nfp), grid=grid, weight=qs_weight, normalize=False
            ),
        )
    )
    eq2, result = eq.optimize(
        verbose=3,
        objective=objective,
        constraints=constraints,
        optimizer=Optimizer("lsq-exact"),
    )
    aspect = eq2.compute("R0/a")["R0/a"]
    np.testing.assert_allclose(aspect, aspect_target, atol=1e-2, rtol=1e-3)
    np.testing.assert_array_less(objective.compute_scalar(objective.x(eq)), 1e-2)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_NAE_QSC_solve():
    """Test O(rho) NAE QSC constraints solve."""
    # get Qsc example
    qsc = Qsc.from_paper("precise QA")
    ntheta = 75
    r = 0.01
    N = 9
    eq = Equilibrium.from_near_axis(qsc, r=r, L=6, M=6, N=N, ntheta=ntheta)

    orig_Rax_val = eq.axis.R_n
    orig_Zax_val = eq.axis.Z_n

    eq_fit = eq.copy()

    # this has all the constraints we need,
    #  iota=False specifies we want to fix current instead of iota
    cs = get_NAE_constraints(eq, qsc, iota=False, order=1)

    objectives = ForceBalance()
    obj = ObjectiveFunction(objectives)

    eq.solve(verbose=3, ftol=1e-2, objective=obj, maxiter=50, xtol=1e-6, constraints=cs)

    # Make sure axis is same
    np.testing.assert_almost_equal(orig_Rax_val, eq.axis.R_n)
    np.testing.assert_almost_equal(orig_Zax_val, eq.axis.Z_n)

    # Make sure surfaces of solved equilibrium are similar near axis as QSC
    rho_err, theta_err = area_difference_desc(eq, eq_fit)

    np.testing.assert_allclose(rho_err[:, 0:-4], 0, atol=1e-2)
    np.testing.assert_allclose(theta_err[:, 0:-6], 0, atol=1e-3)

    # Make sure iota of solved equilibrium is same near axis as QSC
    grid = LinearGrid(L=10, M=20, N=20, sym=True, axis=False)
    iota = compress(grid, eq.compute("iota", grid=grid)["iota"], "rho")

    np.testing.assert_allclose(iota[1], qsc.iota, atol=1e-5)
    np.testing.assert_allclose(iota[1:10], qsc.iota, atol=1e-3)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_NAE_QIC_solve():
    """Test O(rho) NAE QIC constraints solve."""
    # get Qic example
    qsc = Qic.from_paper("r2 section 5.2", nphi=99)
    ntheta = 75
    r = 0.01
    N = 9
    eq = Equilibrium.from_near_axis(qsc, r=r, L=6, M=6, N=N, ntheta=ntheta)

    orig_Rax_val = eq.axis.R_n
    orig_Zax_val = eq.axis.Z_n

    eq_fit = eq.copy()

    # this has all the constraints we need,
    #  iota=False specifies we want to fix current instead of iota
    cs = get_NAE_constraints(eq, qsc, iota=False, order=1)

    objectives = ForceBalance()
    obj = ObjectiveFunction(objectives)

    eq.solve(verbose=3, ftol=1e-2, objective=obj, maxiter=50, xtol=1e-6, constraints=cs)

    # Make sure axis is same
    np.testing.assert_almost_equal(orig_Rax_val, eq.axis.R_n)
    np.testing.assert_almost_equal(orig_Zax_val, eq.axis.Z_n)

    # Make sure surfaces of solved equilibrium are similar near axis as QSC
    rho_err, theta_err = area_difference_desc(eq, eq_fit)

    np.testing.assert_allclose(rho_err[:, 0:-4], 0, atol=1e-2)
    np.testing.assert_allclose(theta_err[:, 0:-6], 0, atol=1e-3)

    # Make sure iota of solved equilibrium is same near axis as QSC
    grid = LinearGrid(L=10, M=20, N=20, sym=True, axis=False)
    iota = compress(grid, eq.compute("iota", grid=grid)["iota"], "rho")

    np.testing.assert_allclose(iota[1], qsc.iota, atol=2e-5)
    np.testing.assert_allclose(iota[1:10], qsc.iota, atol=1e-3)


class TestGetExample:
    """Tests for desc.examples.get."""

    @pytest.mark.unit
    def test_missing_example(self):
        """Test for correct error thrown when no example is found."""
        with pytest.raises(ValueError, match="example FOO not found"):
            desc.examples.get("FOO")

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
                -3.49197642e-01,
                -6.81105159e-01,
                1.29781695e00,
                -2.07888586e00,
                1.15800135e00,
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
