"""Regression tests to verify that DESC agrees with VMEC and itself.

Computes several benchmark equilibria and compares the solutions by measuring the
difference in areas between constant theta and rho contours.
"""

import numpy as np
import pytest
from qic import Qic
from qsc import Qsc
from scipy.constants import mu_0

import desc.examples
from desc.calc_BNORM_from_coilset import calc_BNORM_from_coilset
from desc.continuation import _solve_axisym, solve_continuation_automatic
from desc.cut_modular_coils import find_modular_coils
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.field_line_tracing_DESC_from_coilset import field_trace_from_coilset
from desc.field_line_tracing_DESC_with_current_potential_python_regcoil import (
    trace_from_curr_pot,
)
from desc.find_helical_contours_from_python_regcoil_equal_curr_line_integral import (
    find_helical_coils,
)
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import FourierCurrentPotentialField, ToroidalMagneticField
from desc.objectives import (
    AspectRatio,
    CurrentDensity,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixParameter,
    FixPressure,
    FixPsi,
    FixSumModesLambda,
    ForceBalance,
    ForceBalanceAnisotropic,
    HelicalForceBalance,
    MeanCurvature,
    ObjectiveFunction,
    PlasmaVesselDistance,
    PrincipalCurvature,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    RadialForceBalance,
    SurfaceCurrentRegularizedQuadraticFlux,
    Volume,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
)
from desc.optimize import Optimizer
from desc.profiles import FourierZernikeProfile, PowerSeriesProfile
from desc.regcoil import run_regcoil
from desc.vmec_utils import vmec_boundary_subspace

from .utils import area_difference_desc, area_difference_vmec


@pytest.mark.regression
@pytest.mark.solve
def test_SOLOVEV_vacuum(SOLOVEV_vac):
    """Tests that the SOLOVEV vacuum example gives no rotational transform."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV_vac["desc_h5_path"]))[-1]
    data = eq.compute("|J|")

    np.testing.assert_allclose(data["iota"], 0, atol=1e-16)
    np.testing.assert_allclose(data["|J|"], 0, atol=3e-3)

    # test that solving with the continuation method works correctly
    # when eq resolution is lower than the mres_step
    eq.change_resolution(L=3, M=3)
    eqf = _solve_axisym(eq, mres_step=6)
    assert len(eqf) == 1
    assert eqf[-1].L == eq.L
    assert eqf[-1].M == eq.M
    assert eqf[-1].N == eq.N


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
def test_SOLOVEV_anisotropic_results(SOLOVEV):
    """Tests that SOLOVEV with zero anisotropic pressure gives the same result."""
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    # reset to start
    eq.set_initial_guess()
    # give it a zero anisotropy profile
    anisotropy = FourierZernikeProfile()
    anisotropy.change_resolution(eq.L, eq.M, eq.N)
    eq.anisotropy = anisotropy

    obj = ObjectiveFunction(ForceBalanceAnisotropic(eq=eq))
    constraints = get_fixed_boundary_constraints(eq=eq)
    eq.solve(obj, constraints, verbose=3)
    rho_err, theta_err = area_difference_vmec(eq, SOLOVEV["vmec_nc_path"])

    np.testing.assert_allclose(rho_err, 0, atol=1e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)


@pytest.mark.unit
@pytest.mark.solve
def test_DSHAPE_results(DSHAPE):
    """Tests that the DSHAPE examples gives the same results as VMEC."""
    eq = EquilibriaFamily.load(load_from=str(DSHAPE["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_vmec(eq, DSHAPE["vmec_nc_path"])
    np.testing.assert_allclose(rho_err, 0, atol=2e-3)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)


@pytest.mark.unit
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


@pytest.mark.unit
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
def test_HELIOTRON_vac2_results(HELIOTRON_vac, HELIOTRON_vac2):
    """Tests that the 2 methods for solving vacuum give the same results."""
    eq1 = EquilibriaFamily.load(load_from=str(HELIOTRON_vac["desc_h5_path"]))[-1]
    eq2 = EquilibriaFamily.load(load_from=str(HELIOTRON_vac2["desc_h5_path"]))[-1]
    rho_err, theta_err = area_difference_desc(eq1, eq2)
    np.testing.assert_allclose(rho_err[:, 4:], 0, atol=1e-2)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)
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
    # When ConcentricGrid had a rotation option, RadialForceBalance, HelicalForceBalance
    # defaulted to cos, sin rotation, respectively.
    # This test has been kept to increase code coverage.

    def test(iota=False):
        if iota:
            eq1 = Equilibrium(iota=PowerSeriesProfile(0), sym=True)
            eq2 = Equilibrium(iota=PowerSeriesProfile(0), sym=True)
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
        obj1 = ObjectiveFunction(ForceBalance(eq=eq1))
        eq1.solve(objective=obj1)

        # force balances on different grids
        obj2 = ObjectiveFunction(
            (RadialForceBalance(eq=eq2), HelicalForceBalance(eq=eq2))
        )
        eq2.solve(objective=obj2)

        np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn, atol=5e-4)
        np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn, atol=5e-4)
        np.testing.assert_allclose(eq1.L_lmn, eq2.L_lmn, atol=2e-3)

    test(iota=True)
    test(iota=False)


@pytest.mark.regression
@pytest.mark.solve
def test_solve_bounds():
    """Tests optimizing with bounds=(lower bound, upper bound)."""
    # decrease resolution and double pressure so no longer in force balance
    eq = desc.examples.get("DSHAPE")
    eq.change_resolution(L=eq.M, L_grid=eq.M_grid)
    eq.p_l *= 2

    # target force balance residuals with |F| <= 1e3 N
    obj = ObjectiveFunction(
        ForceBalance(normalize=False, normalize_target=False, bounds=(-1e3, 1e3), eq=eq)
    )
    eq.solve(objective=obj, ftol=1e-16, xtol=1e-16, maxiter=200, verbose=3)

    # check that all errors are nearly 0, since residual values are within target bounds
    f = obj.compute_scaled_error(obj.x(eq))
    np.testing.assert_allclose(f, 0, atol=1e-4)


@pytest.mark.regression
@pytest.mark.optimize
def test_1d_optimization():
    """Tests 1D optimization for target aspect ratio."""
    eq = desc.examples.get("SOLOVEV")
    objective = ObjectiveFunction(AspectRatio(eq=eq, target=2.5))
    constraints = (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq),
        FixBoundaryZ(eq=eq, modes=eq.surface.Z_basis.modes[0:-1, :]),
        FixPressure(eq=eq),
        FixIota(eq=eq),
        FixPsi(eq=eq),
    )
    options = {"perturb_options": {"order": 1}}
    with pytest.warns((FutureWarning, UserWarning)):
        eq.optimize(objective, constraints, optimizer="lsq-exact", options=options)

    np.testing.assert_allclose(eq.compute("R0/a")["R0/a"], 2.5, rtol=2e-4)


@pytest.mark.regression
@pytest.mark.optimize
def test_1d_optimization_old():
    """Tests 1D optimization for target aspect ratio."""
    eq = desc.examples.get("SOLOVEV")
    objective = ObjectiveFunction(AspectRatio(eq=eq, target=2.5))
    eq._optimize(
        objective,
        copy=False,
        solve_options={"verbose": 0},
        perturb_options={
            "dZb": True,
            "subspace": vmec_boundary_subspace(eq, ZBS=[0, 1]),
        },
    )

    np.testing.assert_allclose(eq.compute("R0/a")["R0/a"], 2.5, rtol=1e-3)


def run_qh_step(n, eq):
    """Run 1 step of the precise QH optimization example from Landreman & Paul."""
    print(f"==========QH step {n+1}==========")
    grid = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid),
            AspectRatio(eq=eq, target=8, weight=1e2),
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
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
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
            "initial_trust_ratio": 1.0,  # for backwards consistency
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0},
        },
    )

    return eq1


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.optimize
def test_qh_optimization():
    """Tests first 3 steps of precise QH optimization."""
    # create initial equilibrium
    surf = FourierRZToroidalSurface(
        R_lmn=[1, 0.125, 0.1],
        Z_lmn=[-0.125, -0.1],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1]],
        NFP=4,
    )
    eq = Equilibrium(M=4, N=4, Psi=0.04, surface=surf)
    eq = solve_continuation_automatic(eq, objective="force", bdry_step=0.5, verbose=3)[
        -1
    ]

    eq1 = run_qh_step(0, eq)

    obj = QuasisymmetryBoozer(helicity=(1, eq1.NFP), eq=eq1)
    obj.build()
    B_asym = obj.compute(*obj.xs(eq1))

    np.testing.assert_array_less(np.abs(B_asym).max(), 1e-1)
    np.testing.assert_array_less(eq1.compute("a_major/a_minor")["a_major/a_minor"], 5)

    eq2 = run_qh_step(1, eq1)

    obj = QuasisymmetryBoozer(helicity=(1, eq2.NFP), eq=eq2)
    obj.build()
    B_asym = obj.compute(*obj.xs(eq2))
    np.testing.assert_array_less(np.abs(B_asym).max(), 1e-2)
    np.testing.assert_array_less(eq2.compute("a_major/a_minor")["a_major/a_minor"], 5)

    eq3 = run_qh_step(2, eq2)

    obj = QuasisymmetryBoozer(helicity=(1, eq3.NFP), eq=eq3)
    obj.build()
    B_asym = obj.compute(*obj.xs(eq3))
    np.testing.assert_array_less(np.abs(B_asym).max(), 2e-3)
    np.testing.assert_array_less(eq3.compute("a_major/a_minor")["a_major/a_minor"], 5)


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
    np.testing.assert_allclose(rho_err[:, 4:], 0, atol=5e-2)
    np.testing.assert_allclose(theta_err, 0, atol=1e-4)


@pytest.mark.regression
@pytest.mark.optimize
def test_simsopt_QH_comparison():
    """Test case that previously stalled before getting to the solution.

    From Matt's comparison with SIMSOPT.
    """
    nfp = 4
    aspect_target = 8.0
    # Initial (m=0, n=nfp) mode of the axis:
    torsion = 0.4
    LMN_resolution = 6
    # Set shape of the initial condition.
    # R_lmn and Z_lmn are the amplitudes. modes_R and modes_Z are the (m,n) pairs.
    surface = FourierRZToroidalSurface(
        R_lmn=[1.0, 1.0 / aspect_target, torsion],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        Z_lmn=[0, -1.0 / aspect_target, torsion],
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
        constraints=get_fixed_boundary_constraints(eq=eq, profiles=False),
        objective=ObjectiveFunction(objectives=CurrentDensity(eq=eq)),
    )
    ##################################
    # Done creating initial condition.
    # Now define optimization problem.
    ##################################
    constraints = (
        CurrentDensity(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes_to_fix),
        FixBoundaryZ(eq=eq, modes=Z_modes_to_fix),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
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
    # Also scale QS weight to match simsopt/VMEC
    qs_weight = np.sqrt(1 / (8 * (np.pi**4)))
    objective = ObjectiveFunction(
        (
            AspectRatio(
                eq=eq, target=aspect_target, weight=aspect_weight, normalize=False
            ),
            QuasisymmetryTwoTerm(
                eq=eq, helicity=(-1, nfp), grid=grid, weight=qs_weight, normalize=False
            ),
        )
    )
    eq2, result = eq.optimize(
        verbose=3,
        objective=objective,
        constraints=constraints,
        optimizer=Optimizer("proximal-lsq-exact"),
        ftol=1e-3,
    )
    aspect = eq2.compute("R0/a")["R0/a"]
    np.testing.assert_allclose(aspect, aspect_target, atol=1e-2, rtol=1e-2)
    np.testing.assert_array_less(objective.compute_scalar(objective.x(eq)), 0.075)
    np.testing.assert_array_less(eq2.compute("a_major/a_minor")["a_major/a_minor"], 5)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_NAE_QSC_solve():
    """Test O(rho) NAE QSC constraints solve."""
    qsc = Qsc.from_paper("precise QA")
    ntheta = 75
    r = 0.01
    N = 9
    eq = Equilibrium.from_near_axis(qsc, r=r, L=6, M=6, N=N, ntheta=ntheta)

    orig_Rax_val = eq.axis.R_n
    orig_Zax_val = eq.axis.Z_n

    eq_fit = eq.copy()
    eq_lambda_fixed_0th_order = eq.copy()
    eq_lambda_fixed_1st_order = eq.copy()

    # this has all the constraints we need,
    cs = get_NAE_constraints(eq, qsc, order=1, fix_lambda=False, N=eq.N)
    cs_lambda_fixed_0th_order = get_NAE_constraints(
        eq_lambda_fixed_0th_order, qsc, order=1, fix_lambda=0, N=eq.N
    )
    cs_lambda_fixed_1st_order = get_NAE_constraints(
        eq_lambda_fixed_1st_order, qsc, order=1, fix_lambda=True, N=eq.N
    )

    for c in cs:
        # should be no FixSumModeslambda in the fix_lambda=False constraint
        assert not isinstance(c, FixSumModesLambda)

    for eqq, constraints in zip(
        [eq, eq_lambda_fixed_0th_order, eq_lambda_fixed_1st_order],
        [cs, cs_lambda_fixed_0th_order, cs_lambda_fixed_1st_order],
    ):
        objectives = ForceBalance(eq=eqq)
        obj = ObjectiveFunction(objectives)
        print(constraints)

        eqq.solve(
            verbose=3,
            ftol=1e-2,
            objective=obj,
            maxiter=100,
            xtol=1e-6,
            constraints=constraints,
        )
    grid = LinearGrid(L=10, M=20, N=20, NFP=eq.NFP, sym=True, axis=False)
    grid_axis = LinearGrid(rho=0.0, theta=0.0, N=eq.N_grid, NFP=eq.NFP)
    # Make sure axis is same
    for eqq, string in zip(
        [eq, eq_lambda_fixed_0th_order, eq_lambda_fixed_1st_order],
        ["no lambda constraint", "lambda_fixed_0th_order", "lambda_fixed_1st_order"],
    ):
        np.testing.assert_array_almost_equal(orig_Rax_val, eqq.axis.R_n, err_msg=string)
        np.testing.assert_array_almost_equal(orig_Zax_val, eqq.axis.Z_n, err_msg=string)

        # Make sure surfaces of solved equilibrium are similar near axis as QSC
        rho_err, theta_err = area_difference_desc(eqq, eq_fit)

        np.testing.assert_allclose(rho_err[:, 0:-4], 0, atol=1e-2, err_msg=string)
        np.testing.assert_allclose(theta_err[:, 0:-6], 0, atol=1e-3, err_msg=string)

        # Make sure iota of solved equilibrium is same near axis as QSC

        iota = grid.compress(eqq.compute("iota", grid=grid)["iota"])

        np.testing.assert_allclose(iota[0], qsc.iota, atol=1e-5, err_msg=string)
        np.testing.assert_allclose(iota[1:10], qsc.iota, atol=1e-3, err_msg=string)

        ### check lambda to match near axis
        # Evaluate lambda near the axis
        data_nae = eqq.compute(["lambda", "|B|"], grid=grid_axis)
        lam_nae = data_nae["lambda"]
        # Reshape to form grids on theta and phi

        phi = np.squeeze(grid_axis.nodes[:, 2])
        np.testing.assert_allclose(
            lam_nae, -qsc.iota * qsc.nu_spline(phi), atol=2e-5, err_msg=string
        )

        # check |B| on axis
        np.testing.assert_allclose(
            data_nae["|B|"], np.ones(np.size(phi)) * qsc.B0, atol=1e-4, err_msg=string
        )


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_NAE_QIC_solve():
    """Test O(rho) NAE QIC constraints solve."""
    # get Qic example
    qic = Qic.from_paper("QI NFP2 r2", nphi=301, order="r1")
    qic.lasym = False  # don't need to consider stellarator asym for order 1 constraints
    ntheta = 75
    r = 0.01
    N = 11
    eq = Equilibrium.from_near_axis(qic, r=r, L=7, M=7, N=N, ntheta=ntheta)

    orig_Rax_val = eq.axis.R_n
    orig_Zax_val = eq.axis.Z_n

    eq_fit = eq.copy()

    # this has all the constraints we need,
    cs = get_NAE_constraints(eq, qic, order=1)

    objectives = ForceBalance(eq=eq)
    obj = ObjectiveFunction(objectives)

    eq.solve(
        verbose=3, ftol=1e-2, objective=obj, maxiter=100, xtol=1e-6, constraints=cs
    )

    # Make sure axis is same
    np.testing.assert_array_almost_equal(orig_Rax_val, eq.axis.R_n)
    np.testing.assert_array_almost_equal(orig_Zax_val, eq.axis.Z_n)

    # Make sure surfaces of solved equilibrium are similar near axis as QIC
    rho_err, theta_err = area_difference_desc(eq, eq_fit)

    np.testing.assert_allclose(rho_err[:, 0:3], 0, atol=5e-2)
    # theta error isn't really an indicator of near axis behavior
    # since it's computed over the full radius, but just indicates that
    # eq is similar to eq_fit
    np.testing.assert_allclose(theta_err, 0, atol=5e-2)

    # Make sure iota of solved equilibrium is same near axis as QIC
    grid = LinearGrid(L=10, M=20, N=20, NFP=eq.NFP, sym=True, axis=False)
    iota = grid.compress(eq.compute("iota", grid=grid)["iota"])

    np.testing.assert_allclose(iota[1], qic.iota, atol=2e-5)
    np.testing.assert_allclose(iota[1:10], qic.iota, atol=5e-4)

    # check lambda to match near axis
    grid_2d_05 = LinearGrid(rho=np.array(1e-6), M=50, N=50, NFP=eq.NFP, endpoint=True)

    # Evaluate lambda near the axis
    data_nae = eq.compute("lambda", grid=grid_2d_05)
    lam_nae = data_nae["lambda"]

    # Reshape to form grids on theta and phi
    zeta = (
        grid_2d_05.nodes[:, 2]
        .reshape(
            (grid_2d_05.num_theta, grid_2d_05.num_rho, grid_2d_05.num_zeta), order="F"
        )
        .squeeze()
    )

    lam_nae = lam_nae.reshape(
        (grid_2d_05.num_theta, grid_2d_05.num_rho, grid_2d_05.num_zeta), order="F"
    )

    phi = np.squeeze(zeta[0, :])
    lam_nae = np.squeeze(lam_nae[:, 0, :])

    lam_av_nae = np.mean(lam_nae, axis=0)
    np.testing.assert_allclose(
        lam_av_nae, -qic.iota * qic.nu_spline(phi), atol=1e-4, rtol=1e-2
    )

    # check |B| on axis

    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array(1e-6))
    # Evaluate B modes near the axis
    data_nae = eq.compute(["|B|_mn", "B modes"], grid=grid)
    modes = data_nae["B modes"]
    B_mn_nae = data_nae["|B|_mn"]
    # Evaluate B on an angular grid
    theta = np.linspace(0, 2 * np.pi, 150)
    phi = np.linspace(0, 2 * np.pi, qic.nphi)
    th, ph = np.meshgrid(theta, phi)
    B_nae = np.zeros((qic.nphi, 150))

    for i, (l, m, n) in enumerate(modes):
        if m >= 0 and n >= 0:
            B_nae += B_mn_nae[i] * np.cos(m * th) * np.cos(n * ph)
        elif m >= 0 > n:
            B_nae += -B_mn_nae[i] * np.cos(m * th) * np.sin(n * ph)
        elif m < 0 <= n:
            B_nae += -B_mn_nae[i] * np.sin(m * th) * np.cos(n * ph)
        elif m < 0 and n < 0:
            B_nae += B_mn_nae[i] * np.sin(m * th) * np.sin(n * ph)
    # Eliminate the poloidal angle to focus on the toroidal behavior
    B_av_nae = np.mean(B_nae, axis=1)
    np.testing.assert_allclose(B_av_nae, np.ones(np.size(phi)) * qic.B0, atol=2e-2)


@pytest.mark.unit
@pytest.mark.optimize
def test_multiobject_optimization():
    """Test for optimizing multiple objects at once."""
    eq = Equilibrium(L=4, M=4, N=0, iota=2)
    surf = FourierRZToroidalSurface(
        R_lmn=[10, 2.1],
        Z_lmn=[-2],
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
    )
    surf.change_resolution(M=4, N=0)
    constraints = (
        ForceBalance(eq=eq, bounds=(-1e-4, 1e-4), normalize_target=False),
        FixPressure(eq=eq),
        FixParameter(surf, ["Z_lmn", "R_lmn"], [[-1], [0]]),
        FixParameter(eq, ["Psi", "i_l"]),
        FixBoundaryR(eq, modes=[[0, 0, 0]]),
        PlasmaVesselDistance(surface=surf, eq=eq, target=1),
    )

    objective = ObjectiveFunction((Volume(eq=eq, target=eq.compute("V")["V"] * 2),))

    eq.solve(verbose=3)

    optimizer = Optimizer("fmin-auglag")
    (eq, surf), result = optimizer.optimize(
        (eq, surf), objective, constraints, verbose=3, maxiter=500
    )

    np.testing.assert_allclose(
        constraints[-1].compute(*constraints[-1].xs(eq, surf)), 1, rtol=1e-3
    )
    assert surf.R_lmn[0] == 10
    assert surf.Z_lmn[-1] == -2
    assert eq.Psi == 1.0
    np.testing.assert_allclose(eq.i_l, [2, 0, 0])


@pytest.mark.unit
@pytest.mark.optimize
def test_multiobject_optimization_prox():
    """Test for optimizing multiple objects at once using proximal projection."""
    eq = Equilibrium(L=4, M=4, N=0, iota=2)
    surf = FourierRZToroidalSurface(
        R_lmn=[10, 2.1],
        Z_lmn=[-2],
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
    )
    surf.change_resolution(M=4, N=0)
    constraints = (
        ForceBalance(eq=eq, bounds=(-1e-4, 1e-4), normalize_target=False),
        FixPressure(eq=eq),
        FixParameter(surf, ["Z_lmn", "R_lmn"], [[-1], [0]]),
        FixParameter(eq, ["Psi", "i_l"]),
        FixBoundaryR(eq, modes=[[0, 0, 0]]),
    )

    objective = ObjectiveFunction(
        (
            Volume(eq=eq, target=eq.compute("V")["V"] * 2),
            PlasmaVesselDistance(surface=surf, eq=eq, target=1),
        )
    )

    eq.solve(verbose=3)

    optimizer = Optimizer("proximal-lsq-exact")
    (eq, surf), result = optimizer.optimize(
        (eq, surf), objective, constraints, verbose=3, maxiter=100
    )

    np.testing.assert_allclose(
        objective.objectives[-1].compute(*objective.objectives[-1].xs(eq, surf)),
        1,
        rtol=1e-2,
    )
    assert surf.R_lmn[0] == 10
    assert surf.Z_lmn[-1] == -2
    assert eq.Psi == 1.0
    np.testing.assert_allclose(eq.i_l, [2, 0, 0])


@pytest.mark.unit
def test_non_eq_optimization():
    """Test for optimizing a non-eq object by fixing all eq parameters."""
    eq = desc.examples.get("DSHAPE")
    Rmax = 4
    Rmin = 2

    a = 2
    R0 = (Rmax + Rmin) / 2
    surf = FourierRZToroidalSurface(
        R_lmn=[R0, a],
        Z_lmn=[0.0, -a],
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
        sym=True,
        NFP=eq.NFP,
    )

    surf.change_resolution(M=eq.M, N=eq.N)
    constraints = (
        FixParameter(eq),
        MeanCurvature(surf, bounds=(-8, 8)),
        PrincipalCurvature(surf, bounds=(0, 15)),
    )

    grid = LinearGrid(M=18, N=0, NFP=eq.NFP)
    obj = PlasmaVesselDistance(
        surface=surf,
        eq=eq,
        target=0.5,
        use_softmin=True,
        surface_grid=grid,
        plasma_grid=grid,
        alpha=5000,
    )
    objective = ObjectiveFunction((obj,))
    optimizer = Optimizer("lsq-auglag")
    (eq, surf), result = optimizer.optimize(
        (eq, surf), objective, constraints, verbose=3, maxiter=100
    )

    np.testing.assert_allclose(obj.compute(*obj.xs(eq, surf)), 0.5, atol=1e-5)


@pytest.mark.unit
def test_only_non_eq_optimization():
    """Test for optimizing only a non-eq object."""
    eq = desc.examples.get("DSHAPE")
    surf = eq.surface

    surf.change_resolution(M=eq.M, N=eq.N)
    constraints = (
        FixParameter(surf, params="R_lmn", indices=surf.R_basis.get_idx(0, 0, 0)),
    )

    obj = PrincipalCurvature(surf, target=1)

    objective = ObjectiveFunction((obj,))
    optimizer = Optimizer("lsq-exact")
    (surf), result = optimizer.optimize(
        (surf), objective, constraints, verbose=3, maxiter=100
    )
    surf = surf[0]
    np.testing.assert_allclose(obj.compute(*obj.xs(surf)), 1, atol=1e-5)


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
        iota = desc.examples.get("W7-X", "iota")
        np.testing.assert_allclose(
            iota.params[:5],
            [
                -8.56047021e-01,
                -3.88095412e-02,
                -6.86795128e-02,
                -1.86970315e-02,
                1.90561179e-02,
            ],
        )

    @pytest.mark.unit
    def test_example_get_current(self):
        """Test getting current profile."""
        current = desc.examples.get("NCSX", "current")
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


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_axisymmetric():
    """Test axisymmetric regcoil solution."""
    surf_eq = FourierRZToroidalSurface(
        R_lmn=np.array([10, 1]),
        Z_lmn=np.array([-1]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
        sym=True,
    )
    surf_winding = FourierRZToroidalSurface(
        R_lmn=np.array([10, 2]),
        Z_lmn=np.array([-2]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
        sym=True,
        NFP=3,
    )

    # make a simple axisymmetric vacuum equilibrium
    eq = Equilibrium(surface=surf_eq, L=2, M=2, N=0)
    eq.solve()
    # no phi_SV is needed since it is axisymmetric,
    # so phi_mn should be zero when running REGCOIL
    # especially with a nonzero alpha

    surface_current_field = FourierCurrentPotentialField.from_surface(surf_winding)
    data = surface_current_field.run_regcoil(
        eq,
        M_Phi=1,
        N_Phi=1,
        alpha=1e-19,
        show_plots=False,
    )
    chi_B = data["chi^2_B"]
    phi_mn_opt = surface_current_field.Phi_mn
    G = surface_current_field.G
    np.testing.assert_allclose(phi_mn_opt, 0, atol=2e-9)
    np.testing.assert_allclose(chi_B, 0, atol=1e-14)
    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )
    np.testing.assert_allclose(B, B_from_surf, atol=1e-4)

    grid = LinearGrid(N=10, M=10)
    correct_phi = G * grid.nodes[:, 2] / 2 / np.pi
    np.testing.assert_allclose(
        surface_current_field.compute("Phi", grid=grid)["Phi"], correct_phi, atol=5e-9
    )

    # test with alpha large, should have no phi_mn
    surface_current_field, _, _, chi_B, _ = run_regcoil(
        basis_M=2,
        basis_N=2,
        eqname=eq,
        eval_grid_M=10,
        eval_grid_N=10,
        source_grid_M=40,
        source_grid_N=40,
        winding_surf=surf_winding,
        alpha=10,
    )
    phi_mn_opt = surface_current_field.Phi_mn
    np.testing.assert_allclose(phi_mn_opt, 0, atol=1e-16)
    np.testing.assert_allclose(chi_B, 0, atol=1e-16)
    np.testing.assert_allclose(
        surface_current_field.compute("Phi", grid=grid)["Phi"], correct_phi, atol=1e-16
    )
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )
    np.testing.assert_allclose(B, B_from_surf, atol=1e-4)

    # test with half the current given external to winding surface
    surface_current_field, _, _, chi_B, _ = run_regcoil(
        basis_M=2,
        basis_N=2,
        eqname=eq,
        eval_grid_M=10,
        eval_grid_N=10,
        source_grid_M=40,
        source_grid_N=40,
        alpha=10,
        winding_surf=surf_winding,
        # negate the B0 because a negative G corresponds to a positive B toroidal
        # and we want this to provide half the field the surface current's
        # G is providing, in the same direction
        external_field=ToroidalMagneticField(B0=-mu_0 * (G / 2) / 2 / np.pi, R0=1),
    )
    phi_mn_opt = surface_current_field.Phi_mn
    np.testing.assert_allclose(G / 2, surface_current_field.G, atol=1e-8)
    np.testing.assert_allclose(phi_mn_opt, 0, atol=1e-10)
    np.testing.assert_allclose(
        surface_current_field.compute("Phi", grid=grid)["Phi"],
        correct_phi / 2,
        atol=1e-9,
    )
    np.testing.assert_allclose(chi_B, 0, atol=1e-16)
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )
    np.testing.assert_allclose(B, B_from_surf * 2, atol=1e-4)

    # test with half the current given external to winding surface
    # using external TF argument
    surface_current_field, _, _, chi_B, _ = run_regcoil(
        basis_M=2,
        basis_N=2,
        eqname=eq,
        eval_grid_M=10,
        eval_grid_N=10,
        source_grid_M=40,
        source_grid_N=40,
        alpha=10,
        external_TF_fraction=0.5,
        winding_surf=surf_winding,
    )
    phi_mn_opt = surface_current_field.Phi_mn
    np.testing.assert_allclose(G / 2, surface_current_field.G, atol=1e-8)
    np.testing.assert_allclose(phi_mn_opt, 0, atol=1e-10)
    np.testing.assert_allclose(
        surface_current_field.compute("Phi", grid=grid)["Phi"],
        correct_phi / 2,
        atol=1e-9,
    )
    np.testing.assert_allclose(chi_B, 0, atol=1e-16)

    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )
    np.testing.assert_allclose(B, B_from_surf * 2, atol=1e-4)


# TODO: break this into a class so each test is separate (sequential somehow?)
@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_axisym_and_ellipse_surface():
    """Test regcoil for axisym eq and elliptical surface."""
    surf_eq = FourierRZToroidalSurface(
        R_lmn=np.array([10, 0.5]),
        Z_lmn=np.array([-0.5]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
        sym=True,
        NFP=3,
    )
    from desc.examples import get

    surf_winding = get("HELIOTRON").surface
    surf_winding.change_resolution(NFP=3)

    # make a simple axisymmetric vacuum equilibrium
    eq = Equilibrium(surface=surf_eq, L=2, M=2, N=0)
    eq.solve()

    surface_current_field, _, _, chi_B, _ = run_regcoil(
        basis_M=6,
        basis_N=6,
        eqname=eq,
        eval_grid_M=50,
        eval_grid_N=50,
        source_grid_M=50,
        source_grid_N=50,
        alpha=0,
        winding_surf=surf_winding,
        show_plots=False,
    )
    phi_mn_opt = surface_current_field.Phi_mn
    G = surface_current_field.G

    np.testing.assert_allclose(chi_B, 0, atol=1e-7)
    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )
    np.testing.assert_allclose(B, B_from_surf, atol=2e-4)

    # test with alpha large, should have no phi_mn
    surface_current_field, _, _, chi_B, _ = run_regcoil(
        basis_M=2,
        basis_N=2,
        eqname=eq,
        eval_grid_M=10,
        eval_grid_N=10,
        source_grid_M=40,
        source_grid_N=40,
        alpha=1e4,
        winding_surf=surf_winding,
    )
    phi_mn_opt = surface_current_field.Phi_mn
    np.testing.assert_allclose(phi_mn_opt, 0, atol=1e-9)

    # test with half the current given external to winding surface
    # using external_TF_fraction (putting current on the wind surf)
    surface_current_field, external_B, _, chi_B, _ = run_regcoil(
        basis_M=6,
        basis_N=6,
        eqname=eq,
        eval_grid_M=50,
        eval_grid_N=50,
        source_grid_M=50,
        source_grid_N=50,
        alpha=0,
        winding_surf=surf_winding,
        show_plots=False,
        external_TF_fraction=0.5,
    )
    phi_mn_opt = surface_current_field.Phi_mn

    np.testing.assert_allclose(G / 2, surface_current_field.G, atol=1e-8)
    np.testing.assert_allclose(chi_B, 0, atol=1e-7)
    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )
    B_ext = external_B.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )

    np.testing.assert_allclose(B, B_from_surf + B_ext, atol=2e-4)

    # test with half the current given external to winding surface
    # in the form of a TF field

    surface_current_field, external_field, _, chi_B, _ = run_regcoil(
        basis_M=6,
        basis_N=6,
        eqname=eq,
        eval_grid_M=50,
        eval_grid_N=50,
        source_grid_M=50,
        source_grid_N=50,
        alpha=0,
        winding_surf=surf_winding,
        external_field=ToroidalMagneticField(B0=-mu_0 * (G / 2) / 2 / np.pi, R0=1),
        verbose=2,
    )
    phi_mn_opt = surface_current_field.Phi_mn

    np.testing.assert_allclose(G / 2, surface_current_field.G, atol=1e-8)

    np.testing.assert_allclose(chi_B, 0, atol=1e-7)
    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )
    B_external = external_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP)
    )

    np.testing.assert_allclose(B, B_from_surf + B_external, atol=1e-4)


# TODO: break this into a class so each test is separate (sequential somehow?)
@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_ellipse_and_axisym_surface():
    """Test elliptical eq and circular winding surf regcoil solution."""
    eq = load("./tests/inputs/ellNFP4_init_smallish.h5")

    (
        all_phi_mns,
        alphas,
        surface_current_field,
        TF_B,
        chi_B,
        lowest_idx_without_saddles,
    ) = run_regcoil(
        basis_M=8,
        basis_N=8,
        eqname=eq,
        eval_grid_M=20,
        eval_grid_N=20,
        source_grid_M=40,
        source_grid_N=80,
        alpha=1e-15,
        scan=True,
        verbose=3,
    )

    assert np.all(np.asarray(chi_B[0:-10]) < 1e-8)
    surface_current_field.Phi_mn = all_phi_mns[12]

    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200)
    )
    np.testing.assert_allclose(B, B_from_surf, atol=1e-4)

    fieldR, fieldZ = trace_from_curr_pot(
        surface_current_field,
        eq,
        alpha=1e-15,
        M=50,
        N=160,
        ntransit=20,
        Rs=np.linspace(0.68, 0.72, 10),
        use_agg_backend=True,
    )

    assert np.max(fieldR) < 0.73
    assert np.min(fieldR) > 0.67

    assert np.max(fieldZ) < 0.02
    assert np.min(fieldZ) > -0.02
    # test with alpha large, should have very small phi_mn
    (surface_current_field, TF_B, mean_Bn, chi_B, Bn_tot,) = run_regcoil(
        basis_M=2,
        basis_N=2,
        eqname=eq,
        eval_grid_M=10,
        eval_grid_N=10,
        source_grid_M=40,
        source_grid_N=40,
        alpha=1e8,
    )
    # should be small
    np.testing.assert_allclose(surface_current_field.Phi_mn, 0, atol=1e-11)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_ellipse_helical_coils():
    """Test elliptical eq and circular winding surf helical coil regcoil solution."""
    eq = load("./tests/inputs/ellNFP4_init_smallish.h5")

    M_Phi = 8
    N_Phi = 8
    M_egrid = 20
    N_egrid = 20
    M_sgrid = 40
    N_sgrid = 80
    alpha = 1e-18

    (surface_current_field, TF_B, mean_Bn, chi_B, Bn_tot,) = run_regcoil(
        basis_M=M_Phi,
        basis_N=N_Phi,
        eqname=eq,
        eval_grid_M=M_egrid,
        eval_grid_N=N_egrid,
        source_grid_M=M_sgrid,
        source_grid_N=N_sgrid,
        alpha=alpha,
        helicity_ratio=-2,
    )
    assert np.all(chi_B < 1e-5)
    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200), basis="rpz"
    )
    np.testing.assert_allclose(B, B_from_surf, atol=1e-3)

    fieldR, fieldZ = trace_from_curr_pot(
        surface_current_field,
        eq,
        alpha=1e-15,
        M=50,
        N=160,
        ntransit=20,
        Rs=np.linspace(0.68, 0.72, 10),
    )

    assert np.max(fieldR) < 0.73
    assert np.min(fieldR) > 0.67

    assert np.max(fieldZ) < 0.02
    assert np.min(fieldZ) > -0.02

    # test finding coils

    numCoils = 15
    coilsFilename = "./coilsfile_15.txt"
    eqname = "./tests/inputs/ellNFP4_init_smallish.h5"

    coilset2 = find_helical_coils(
        surface_current_field,
        eqname,
        desirednumcoils=numCoils,
        coilsFilename=coilsFilename,
        step=6,
        save_figs=False,
    )
    coilset2 = coilset2.to_FourierXYZ(N=150)
    B_from_coils = coilset2.compute_magnetic_field(coords, basis="rpz")
    np.testing.assert_allclose(B, B_from_coils, atol=3e-3)

    fieldR, fieldZ = field_trace_from_coilset(
        coilset2, eq, 15, only_return_data=True, Rs=np.linspace(0.685, 0.715, 10)
    )

    assert np.max(fieldR) < 0.73
    assert np.min(fieldR) > 0.67

    assert np.max(fieldZ) < 0.02
    assert np.min(fieldZ) > -0.02

    B_ratio = calc_BNORM_from_coilset(coilset2, eqname, 0, 1, B0=None, save=False)
    np.testing.assert_allclose(B_ratio, 1.0, atol=1e-3)

    # check against the objective method of running REGCOIL

    surface_current_field2 = surface_current_field.copy()
    surface_current_field2.change_Phi_resolution(M=M_Phi, N=N_Phi, sym_Phi="sin")
    surface_current_field2.Phi_mn = np.zeros_like(surface_current_field2.Phi_mn)

    constraints = (  # now fix all but Phi_mn
        FixParameter(surface_current_field2, params=["I", "G", "R_lmn", "Z_lmn"]),
    )

    eval_grid = LinearGrid(M=M_egrid, N=N_egrid, NFP=eq.NFP, sym=True)
    sgrid = LinearGrid(
        M=M_sgrid,
        N=N_sgrid,
        NFP=eq.NFP,
    )
    obj = SurfaceCurrentRegularizedQuadraticFlux(
        surface_current_field=surface_current_field2,
        eq=eq,
        eval_grid=eval_grid,
        source_grid=sgrid,
        alpha=1e-18,
        eq_fixed=True,
    )
    optimizer = Optimizer("lsq-exact")

    objective = ObjectiveFunction(obj)
    (surface_current_field2,), result = optimizer.optimize(
        (surface_current_field2,),
        objective,
        constraints,
        verbose=1,
        maxiter=1,
        ftol=0,
        gtol=0,
        xtol=1e-16,
        options={"initial_trust_radius": np.inf},
    )

    np.testing.assert_allclose(
        surface_current_field2.Phi_mn,
        surface_current_field.Phi_mn,
        rtol=1e-5,
        atol=1e-13,
    )


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_ellipse_helical_coils_pos_helicity():
    """Test elliptical eq and circular winding surf helical coil regcoil solution."""
    # with positive helicity for the surface current
    eq = load("./tests/inputs/ellNFP4_init_smallish.h5")

    (surface_current_field, TF_B, mean_Bn, chi_B, Bn_tot,) = run_regcoil(
        basis_M=8,
        basis_N=8,
        eqname=eq,
        eval_grid_M=20,
        eval_grid_N=20,
        source_grid_M=40,
        source_grid_N=80,
        alpha=1e-18,
        helicity_ratio=2,
    )
    assert np.all(chi_B < 1e-5)
    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200), basis="rpz"
    )
    np.testing.assert_allclose(B, B_from_surf, atol=1e-3)

    fieldR, fieldZ = trace_from_curr_pot(
        surface_current_field,
        eq,
        alpha=1e-15,
        M=50,
        N=160,
        ntransit=20,
        Rs=np.linspace(0.68, 0.72, 10),
    )

    assert np.max(fieldR) < 0.73
    assert np.min(fieldR) > 0.67

    assert np.max(fieldZ) < 0.02
    assert np.min(fieldZ) > -0.02

    # test finding coils

    numCoils = 15
    coilsFilename = "./coilsfile_15.txt"
    eqname = "./tests/inputs/ellNFP4_init_smallish.h5"

    coilset2 = find_helical_coils(
        surface_current_field,
        eqname,
        desirednumcoils=numCoils,
        coilsFilename=coilsFilename,
        step=6,
        save_figs=False,
    )
    coilset2 = coilset2.to_FourierXYZ(N=150)
    B_from_coils = coilset2.compute_magnetic_field(coords, basis="rpz")
    np.testing.assert_allclose(B, B_from_coils, atol=3e-3)

    fieldR, fieldZ = field_trace_from_coilset(
        coilset2, eq, 15, only_return_data=True, Rs=np.linspace(0.685, 0.715, 10)
    )

    assert np.max(fieldR) < 0.73
    assert np.min(fieldR) > 0.67

    assert np.max(fieldZ) < 0.02
    assert np.min(fieldZ) > -0.02

    B_ratio = calc_BNORM_from_coilset(coilset2, eqname, 0, 1, B0=None, save=False)
    np.testing.assert_allclose(B_ratio, 1.0, atol=1e-3)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_ellipse_modular():
    """Test elliptical eq and circular winding surf modular coil solution."""
    eq = load("./tests/inputs/ellNFP4_init_smallish.h5")

    (surface_current_field, TF_B, mean_Bn, chi_B, Bn_tot,) = run_regcoil(
        basis_M=8,
        basis_N=8,
        eqname=eq,
        eval_grid_M=20,
        eval_grid_N=20,
        source_grid_M=40,
        source_grid_N=80,
        alpha=1e-18,
        helicity_ratio=0,
    )
    assert np.all(chi_B < 1e-5)
    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords, grid=LinearGrid(M=200, N=200)
    )
    np.testing.assert_allclose(B, B_from_surf, atol=1e-3)

    fieldR, fieldZ = trace_from_curr_pot(
        surface_current_field,
        eq,
        alpha=1e-15,
        M=50,
        N=160,
        ntransit=20,
        Rs=np.linspace(0.68, 0.72, 10),
    )

    assert np.max(fieldR) < 0.73
    assert np.min(fieldR) > 0.67

    assert np.max(fieldZ) < 0.02
    assert np.min(fieldZ) > -0.02

    # test finding coils

    numCoils = 240
    coilsFilename = "./coilsfile_240.txt"
    eqname = "./tests/inputs/ellNFP4_init_smallish.h5"

    coilset2 = find_modular_coils(
        surface_current_field,
        eqname,
        desirednumcoils=numCoils,
        coilsFilename=coilsFilename,
        step=6,
        save_figs=False,
    )

    B_from_coils = coilset2.compute_magnetic_field(coords)
    np.testing.assert_allclose(B, B_from_coils, atol=2.5e-3)

    B_ratio = calc_BNORM_from_coilset(coilset2, eqname, 0, 1, B0=None, save=False)
    np.testing.assert_allclose(B_ratio, 1.0, atol=2.5e-3)
