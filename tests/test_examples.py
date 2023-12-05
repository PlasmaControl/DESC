"""Regression tests to verify that DESC agrees with VMEC and itself.

Computes several benchmark equilibria and compares the solutions by measuring the
difference in areas between constant theta and rho contours.
"""

import numpy as np
import pytest
from qic import Qic
from qsc import Qsc

import desc.examples
from desc.continuation import _solve_axisym, solve_continuation_automatic
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
    FixParameter,
    FixPressure,
    FixPsi,
    FixSumModesLambda,
    ForceBalance,
    ForceBalanceAnisotropic,
    HelicalForceBalance,
    ObjectiveFunction,
    PlasmaVesselDistance,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    RadialForceBalance,
    Volume,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
)
from desc.optimize import Optimizer
from desc.profiles import FourierZernikeProfile, PowerSeriesProfile
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
