"""Regression tests to verify that DESC agrees with VMEC and itself.

Computes several benchmark equilibria and compares the solutions by measuring the
difference in areas between constant theta and rho contours.
"""

import numpy as np
import pytest
from qic import Qic
from qsc import Qsc

from desc.backend import jnp, tree_leaves
from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    FourierRZCoil,
    FourierXYZCoil,
    MixedCoilSet,
    _Coil,
)
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.equilibrium.coords import compute_theta_coords
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.io import load
from desc.magnetic_fields import (
    OmnigenousField,
    SplineMagneticField,
    ToroidalMagneticField,
    VerticalMagneticField,
)
from desc.objectives import (
    AspectRatio,
    BallooningStability,
    BoundaryError,
    CoilCurvature,
    CoilLength,
    CoilSetMinDistance,
    CoilTorsion,
    CurrentDensity,
    FixBoundaryR,
    FixBoundaryZ,
    FixCoilCurrent,
    FixCurrent,
    FixIota,
    FixOmniBmax,
    FixOmniMap,
    FixParameters,
    FixPressure,
    FixPsi,
    FixSumModesLambda,
    ForceBalance,
    ForceBalanceAnisotropic,
    GenericObjective,
    LinearObjectiveFromUser,
    MeanCurvature,
    ObjectiveFunction,
    Omnigenity,
    PlasmaCoilSetMinDistance,
    PlasmaVesselDistance,
    PrincipalCurvature,
    QuadraticFlux,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    VacuumBoundaryError,
    Volume,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
)
from desc.optimize import Optimizer
from desc.profiles import FourierZernikeProfile, PowerSeriesProfile

from .utils import area_difference_desc, area_difference_vmec


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
def test_solve_bounds():
    """Tests optimizing with bounds=(lower bound, upper bound)."""
    # decrease resolution and double pressure so no longer in force balance
    eq = get("DSHAPE")
    with pytest.warns(UserWarning, match="Reducing radial"):
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
    eq = get("SOLOVEV")
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
        options={},
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
    eq = Equilibrium(M=5, N=5, Psi=0.04, surface=surf)
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
    eq0 = get("ATF")
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
    eq2, _ = eq.optimize(
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
    eq = Equilibrium.from_near_axis(qsc, r=r, L=4, M=4, N=N, ntheta=ntheta)

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

        eqq.solve(
            verbose=3,
            ftol=1e-2,
            objective=obj,
            maxiter=50,
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

        np.testing.assert_allclose(rho_err[:, 0:-6], 0, atol=1.5e-2, err_msg=string)
        np.testing.assert_allclose(theta_err[:, 0:-6], 0, atol=1e-3, err_msg=string)

        # Make sure iota of solved equilibrium is same near axis as QSC

        iota = grid.compress(eqq.compute("iota", grid=grid)["iota"])

        np.testing.assert_allclose(iota[0], qsc.iota, atol=1e-5, err_msg=string)
        np.testing.assert_allclose(iota[1:10], qsc.iota, atol=1e-3, err_msg=string)

        # check lambda to match near axis
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
def test_NAE_QSC_solve_near_axis_based_off_eq():
    """Test O(rho) NAE QSC constraints solve when qsc eq is not given."""
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
    cs = get_NAE_constraints(eq, qsc_eq=None, order=1, fix_lambda=False, N=eq.N)
    cs_lambda_fixed_0th_order = get_NAE_constraints(
        eq_lambda_fixed_0th_order, qsc_eq=None, order=1, fix_lambda=0, N=eq.N
    )
    cs_lambda_fixed_1st_order = get_NAE_constraints(
        eq_lambda_fixed_1st_order, qsc_eq=None, order=1, fix_lambda=True, N=eq.N
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

        np.testing.assert_allclose(rho_err[:, 0:-6], 0, atol=1.5e-2, err_msg=string)
        np.testing.assert_allclose(theta_err[:, 0:-6], 0, atol=1e-3, err_msg=string)

        # Make sure iota of solved equilibrium is same near axis as QSC

        iota = grid.compress(eqq.compute("iota", grid=grid)["iota"])

        np.testing.assert_allclose(iota[0], qsc.iota, atol=1e-5, err_msg=string)
        np.testing.assert_allclose(iota[1:10], qsc.iota, rtol=5e-3, err_msg=string)

        ### check lambda to match on axis
        # Evaluate lambda on the axis
        data_nae = eqq.compute(["lambda", "|B|"], grid=grid_axis)
        lam_nae = data_nae["lambda"]

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

    np.testing.assert_allclose(iota[0], qic.iota, rtol=1e-5)
    np.testing.assert_allclose(iota[1:10], qic.iota, rtol=1e-3)

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
def test_multiobject_optimization_al():
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
        FixParameters(surf, {"R_lmn": np.array([0]), "Z_lmn": np.array([3])}),
        FixParameters(eq, {"Psi": True, "i_l": True}),
        FixBoundaryR(eq, modes=[[0, 0, 0]]),
        PlasmaVesselDistance(surface=surf, eq=eq, target=1),
    )

    objective = ObjectiveFunction((Volume(eq=eq, target=eq.compute("V")["V"] * 2),))

    eq.solve(verbose=3)

    optimizer = Optimizer("fmin-auglag")
    (eq, surf), _ = optimizer.optimize(
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
        ForceBalance(eq=eq),
        FixPressure(eq=eq),
        FixParameters(surf, {"R_lmn": np.array([0]), "Z_lmn": np.array([3])}),
        FixParameters(eq, {"Psi": True, "i_l": True}),
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
    (eq, surf), _ = optimizer.optimize(
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
@pytest.mark.optimize
def test_omnigenity_qa():
    """Test optimizing omnigenity parameters to match an axisymmetric equilibrium."""
    # Solov'ev examples has B_max contours at theta=pi, need to change to theta=0
    eq = get("SOLOVEV")
    rone = np.ones_like(eq.R_lmn)
    rone[eq.R_basis.modes[:, 1] != 0] *= -1
    eq.R_lmn *= rone
    zone = np.ones_like(eq.Z_lmn)
    zone[eq.Z_basis.modes[:, 1] != 0] *= -1
    eq.Z_lmn *= zone
    lone = np.ones_like(eq.L_lmn)
    lone[eq.L_basis.modes[:, 1] != 0] *= -1
    eq.L_lmn *= lone
    eq.axis = eq.get_axis()
    eq.surface = eq.get_surface_at(rho=1)
    eq.Psi *= 5  # B0 = 1 T
    eq.solve()

    field = OmnigenousField(
        L_B=1, M_B=4, L_x=1, M_x=1, N_x=1, NFP=eq.NFP, helicity=(1, 0)
    )

    eq_axis_grid = LinearGrid(rho=1e-2, M=4 * eq.M, N=4 * eq.N, NFP=eq.NFP, sym=False)
    eq_lcfs_grid = LinearGrid(rho=1.0, M=4 * eq.M, N=4 * eq.N, NFP=eq.NFP, sym=False)

    field_axis_grid = LinearGrid(
        rho=1e-2, theta=2 * field.M_B, N=2 * field.N_x, NFP=field.NFP, sym=False
    )
    field_lcfs_grid = LinearGrid(
        rho=1.0, theta=2 * field.M_B, N=2 * field.N_x, NFP=field.NFP, sym=False
    )

    objective = ObjectiveFunction(
        (
            Omnigenity(
                eq=eq,
                field=field,
                eq_grid=eq_axis_grid,
                field_grid=field_axis_grid,
                eq_fixed=True,
            ),
            Omnigenity(
                eq=eq,
                field=field,
                eq_grid=eq_lcfs_grid,
                field_grid=field_lcfs_grid,
                eq_fixed=True,
            ),
        )
    )

    optimizer = Optimizer("lsq-exact")
    (field,), _ = optimizer.optimize(
        field,
        objective,
        maxiter=100,
        ftol=1e-6,
        xtol=1e-6,
        verbose=3,
    )

    B_lm = field.B_lm.reshape((field.B_basis.L + 1, -1))
    B0 = field.B_basis.evaluate(np.array([[0, 0, 0]])) @ B_lm
    B1 = field.B_basis.evaluate(np.array([[1, 0, 0]])) @ B_lm

    # x_lmn=0 because the equilibrium is QS
    np.testing.assert_allclose(field.x_lmn, 0, atol=1e-12)

    # check that magnetic well parameters get |B| on axis correct
    grid = LinearGrid(N=eq.N_grid, NFP=eq.NFP, rho=0)
    data = eq.compute("|B|", grid=grid)
    np.testing.assert_allclose(B0, np.mean(data["|B|"]), rtol=1e-3)

    # check that magnetic well parameters get |B| min & max on LCFS correct
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=1)
    data = eq.compute(["min_tz |B|", "max_tz |B|"], grid=grid)
    np.testing.assert_allclose(np.min(B1), data["min_tz |B|"][0], rtol=2e-3)
    np.testing.assert_allclose(np.max(B1), data["max_tz |B|"][0], rtol=2e-3)


@pytest.mark.regression
@pytest.mark.optimize
@pytest.mark.slow
def test_omnigenity_optimization():
    """Test a realistic OP omnigenity optimization."""
    # this same example is used in docs/notebooks/tutorials/omnigenity

    # initial equilibrium from QP model
    surf = FourierRZToroidalSurface.from_qp_model(
        major_radius=1,
        aspect_ratio=10,
        elongation=3,
        mirror_ratio=0.2,
        torsion=0.1,
        NFP=2,
        sym=True,
    )
    eq = Equilibrium(Psi=3e-2, M=4, N=4, surface=surf)
    eq, _ = eq.solve(objective="force", verbose=3)

    # omnigenity optimization
    field = OmnigenousField(
        L_B=1,
        M_B=3,
        L_x=1,
        M_x=1,
        N_x=1,
        NFP=eq.NFP,
        helicity=(0, eq.NFP),
        B_lm=np.array([0.8, 1.0, 1.2, 0, 0, 0]),
    )

    def mirrorRatio(params):
        B_lm = params["B_lm"]
        f = jnp.array(
            [
                B_lm[0] - B_lm[field.M_B],  # B_min on axis
                B_lm[field.M_B - 1] - B_lm[-1],  # B_max on axis
            ]
        )
        return f

    eq_half_grid = LinearGrid(rho=0.5, M=4 * eq.M, N=4 * eq.N, NFP=eq.NFP, sym=False)
    eq_lcfs_grid = LinearGrid(rho=1.0, M=4 * eq.M, N=4 * eq.N, NFP=eq.NFP, sym=False)

    field_half_grid = LinearGrid(rho=0.5, theta=16, zeta=8, NFP=field.NFP, sym=False)
    field_lcfs_grid = LinearGrid(rho=1.0, theta=16, zeta=8, NFP=field.NFP, sym=False)

    objective = ObjectiveFunction(
        (
            GenericObjective("R0", thing=eq, target=1.0, name="major radius"),
            AspectRatio(eq=eq, bounds=(0, 10)),
            Omnigenity(
                eq=eq,
                field=field,
                eq_grid=eq_half_grid,
                field_grid=field_half_grid,
                eta_weight=1,
            ),
            Omnigenity(
                eq=eq,
                field=field,
                eq_grid=eq_lcfs_grid,
                field_grid=field_lcfs_grid,
                eta_weight=2,
            ),
        )
    )
    constraints = (
        CurrentDensity(eq=eq),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
        FixOmniBmax(field=field),
        FixOmniMap(field=field, indices=np.where(field.x_basis.modes[:, 1] == 0)[0]),
        LinearObjectiveFromUser(mirrorRatio, field, target=[0.8, 1.2]),
    )
    optimizer = Optimizer("lsq-auglag")
    (eq, field), _ = optimizer.optimize(
        (eq, field), objective, constraints, maxiter=100, verbose=3
    )
    eq, _ = eq.solve(objective="force", verbose=3)

    # check omnigenity error is low
    f = objective.compute_unscaled(objective.x(*(eq, field)))  # error in Tesla
    np.testing.assert_allclose(f[2:], 0, atol=1.2e-2)  # f[:2] is R0 and R0/a

    # check mirror ratio is correct
    grid = LinearGrid(N=eq.N_grid, NFP=eq.NFP, rho=np.array([0]))
    data = eq.compute("|B|", grid=grid)
    np.testing.assert_allclose(np.min(data["|B|"]), 0.8, atol=2e-2)
    np.testing.assert_allclose(np.max(data["|B|"]), 1.2, atol=2e-2)


@pytest.mark.unit
def test_omnigenity_proximal():
    """Test omnigenity optimization with proximal optimizer."""
    # this only tests that the optimization runs, not that it gives a good result

    # initial equilibrium and omnigenous field
    surf = FourierRZToroidalSurface.from_qp_model(
        major_radius=1,
        aspect_ratio=10,
        elongation=3,
        mirror_ratio=0.2,
        torsion=0.1,
        NFP=2,
        sym=True,
    )
    eq = Equilibrium(Psi=3e-2, M=4, N=4, surface=surf)
    eq, _ = eq.solve(objective="force", verbose=3)
    field = OmnigenousField(
        L_B=1,
        M_B=3,
        L_x=1,
        M_x=1,
        N_x=1,
        NFP=eq.NFP,
        helicity=(0, eq.NFP),
        B_lm=np.array([0.8, 1.0, 1.2, 0, 0, 0]),
    )

    # first, test optimizing the equilibrium with the field fixed
    objective = ObjectiveFunction(
        (
            GenericObjective("R0", thing=eq, target=1.0, name="major radius"),
            AspectRatio(eq=eq, bounds=(0, 10)),
            Omnigenity(eq=eq, field=field, field_fixed=True),  # field is fixed
        )
    )
    constraints = (
        CurrentDensity(eq=eq),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )
    optimizer = Optimizer("proximal-lsq-exact")
    [eq], _ = optimizer.optimize(eq, objective, constraints, maxiter=2, verbose=3)

    # second, test optimizing both the equilibrium and the field simultaneously
    objective = ObjectiveFunction(
        (
            GenericObjective("R0", thing=eq, target=1.0, name="major radius"),
            AspectRatio(eq=eq, bounds=(0, 10)),
            Omnigenity(eq=eq, field=field),  # field is not fixed
        )
    )
    constraints = (
        CurrentDensity(eq=eq),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )
    optimizer = Optimizer("proximal-lsq-exact")
    (eq, field), _ = optimizer.optimize(
        (eq, field), objective, constraints, maxiter=2, verbose=3
    )


@pytest.mark.unit
def test_non_eq_optimization():
    """Test for optimizing a non-eq object by fixing all eq parameters."""
    eq = get("DSHAPE")
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
        FixParameters(eq),
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
    (eq, surf), _ = optimizer.optimize(
        (eq, surf), objective, constraints, verbose=3, maxiter=100
    )

    np.testing.assert_allclose(obj.compute(*obj.xs(eq, surf)), 0.5, atol=1e-5)


@pytest.mark.unit
def test_only_non_eq_optimization():
    """Test for optimizing only a non-eq object."""
    eq = get("DSHAPE")
    surf = eq.surface
    surf.change_resolution(M=eq.M, N=eq.N)
    constraints = (
        FixParameters(surf, {"R_lmn": np.array(surf.R_basis.get_idx(0, 0, 0))}),
    )
    obj = PrincipalCurvature(surf, target=1)
    objective = ObjectiveFunction((obj,))
    optimizer = Optimizer("lsq-exact")
    (surf), _ = optimizer.optimize(
        (surf), objective, constraints, verbose=3, maxiter=100
    )
    surf = surf[0]
    np.testing.assert_allclose(obj.compute(*obj.xs(surf)), 1, atol=1e-5)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_freeb_vacuum():
    """Test for free boundary vacuum stellarator."""
    # currents from VMEC input this test is meant to reproduce
    extcur = [4700.0, 1000.0]
    ext_field = SplineMagneticField.from_mgrid(
        "tests/inputs/mgrid_test.nc", extcur=extcur
    )
    surf = FourierRZToroidalSurface(
        R_lmn=[0.70, 0.10],
        modes_R=[[0, 0], [1, 0]],
        Z_lmn=[-0.10],
        modes_Z=[[-1, 0]],
        NFP=5,
    )
    eq = Equilibrium(M=6, N=6, Psi=-0.035, surface=surf)
    eq.solve()

    constraints = (
        ForceBalance(eq=eq),
        FixCurrent(eq=eq),
        FixPressure(eq=eq),
        FixPsi(eq=eq),
    )
    objective = ObjectiveFunction(
        VacuumBoundaryError(eq=eq, field=ext_field, field_fixed=True)
    )
    eq, _ = eq.optimize(
        objective,
        constraints,
        optimizer="proximal-lsq-exact",
        verbose=3,
        options={},
    )

    rho_err, _ = area_difference_vmec(eq, "tests/inputs/wout_test_freeb.nc")
    np.testing.assert_allclose(rho_err[:, -1], 0, atol=4e-2)  # only check rho=1


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_freeb_axisym():
    """Test for free boundary finite beta tokamak."""
    # currents from VMEC input this test is meant to reproduce
    extcur = [
        3.884526409876309e06,
        -2.935577123737952e05,
        -1.734851853677043e04,
        6.002137016973160e04,
        6.002540940490887e04,
        -1.734993103183817e04,
        -2.935531536308510e05,
        -3.560639108717275e05,
        -6.588434719283084e04,
        -1.154387774712987e04,
        -1.153546510755219e04,
        -6.588300858364606e04,
        -3.560589388468855e05,
    ]
    ext_field = SplineMagneticField.from_mgrid(
        r"tests/inputs/mgrid_solovev.nc", extcur=extcur
    )

    pres = PowerSeriesProfile([1.25e-1, 0, -1.25e-1])
    iota = PowerSeriesProfile([-4.9e-1, 0, 3.0e-1])
    surf = FourierRZToroidalSurface(
        R_lmn=[4.0, 1.0],
        modes_R=[[0, 0], [1, 0]],
        Z_lmn=[-1.0],
        modes_Z=[[-1, 0]],
        NFP=1,
    )
    eq = Equilibrium(M=10, N=0, Psi=1.0, surface=surf, pressure=pres, iota=iota)
    eq.solve()

    constraints = (
        ForceBalance(eq=eq),
        FixIota(eq=eq),
        FixPressure(eq=eq),
        FixPsi(eq=eq),
    )
    objective = ObjectiveFunction(
        BoundaryError(eq=eq, field=ext_field, field_fixed=True)
    )

    # we know this is a pretty simple shape so we'll only use |m| <= 2
    R_modes = eq.surface.R_basis.modes[
        np.max(np.abs(eq.surface.R_basis.modes), 1) > 2, :
    ]
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > 2, :
    ]
    bdry_constraints = (
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
    )

    eq, _ = eq.optimize(
        objective,
        constraints + bdry_constraints,
        optimizer="proximal-lsq-exact",
        verbose=3,
        options={},
    )

    rho_err, _ = area_difference_vmec(eq, "tests/inputs/wout_solovev_freeb.nc")
    np.testing.assert_allclose(rho_err[:, -1], 0, atol=2e-2)  # only check rho=1


class TestGetExample:
    """Tests for desc.examples.get."""

    @pytest.mark.unit
    def test_missing_example(self):
        """Test for correct error thrown when no example is found."""
        with pytest.raises(ValueError, match="example FOO not found"):
            get("FOO")

    @pytest.mark.unit
    def test_example_get_eq(self):
        """Test getting a single equilibrium."""
        eq = get("SOLOVEV")
        assert eq.Psi == 1

    @pytest.mark.unit
    def test_example_get_eqf(self):
        """Test getting full equilibria family."""
        eqf = get("DSHAPE", "all")
        np.testing.assert_allclose(eqf[0].pressure.params, 0)

    @pytest.mark.unit
    def test_example_get_boundary(self):
        """Test getting boundary surface."""
        surf = get("HELIOTRON", "boundary")
        np.testing.assert_allclose(surf.R_lmn[surf.R_basis.get_idx(0, 1, 1)], -0.3)

    @pytest.mark.unit
    def test_example_get_pressure(self):
        """Test getting pressure profile."""
        pres = get("ATF", "pressure")
        np.testing.assert_allclose(pres.params[:5], [5e5, -1e6, 5e5, 0, 0])

    @pytest.mark.unit
    def test_example_get_iota(self):
        """Test getting iota profile."""
        iota = get("W7-X", "iota")
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
        current = get("NCSX", "current")
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


@pytest.mark.unit
def test_quadratic_flux_optimization_with_analytic_field():
    """Test analytic field optimization to reduce quadratic flux.

    Checks that B goes to zero for non-axisymmetric eq and axisymmetric field.
    """
    eq = get("precise_QA")
    field = ToroidalMagneticField(1, 1)
    eval_grid = LinearGrid(
        rho=np.array([1.0]),
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
        sym=False,
    )

    optimizer = Optimizer("lsq-exact")

    constraints = (FixParameters(field, {"R0": True}),)
    quadflux_obj = QuadraticFlux(
        eq=eq,
        field=field,
        eval_grid=eval_grid,
        vacuum=True,
    )
    objective = ObjectiveFunction(quadflux_obj)
    things, __ = optimizer.optimize(
        field,
        objective=objective,
        constraints=constraints,
        ftol=1e-14,
        gtol=1e-14,
        copy=True,
        verbose=3,
    )

    # optimizer should zero out field since that's the easiest way
    # to get to Bnorm = 0
    np.testing.assert_allclose(things[0].B0, 0, atol=1e-12)


@pytest.mark.unit
def test_second_stage_optimization():
    """Test optimizing magnetic field for a fixed axisymmetric equilibrium."""
    eq = get("DSHAPE")
    field = ToroidalMagneticField(B0=1, R0=3.5) + VerticalMagneticField(B0=1)
    objective = ObjectiveFunction(QuadraticFlux(eq=eq, field=field, vacuum=True))
    constraints = FixParameters(field, [{"R0": True}, {}])
    optimizer = Optimizer("scipy-trf")
    (field,), _ = optimizer.optimize(
        things=field,
        objective=objective,
        constraints=constraints,
        ftol=0,
        xtol=0,
        verbose=2,
    )
    np.testing.assert_allclose(field[0].R0, 3.5)  # this value was fixed
    np.testing.assert_allclose(field[0].B0, 1)  # toroidal field (no change)
    np.testing.assert_allclose(field[1].B0, 0, atol=1e-12)  # vertical field (vanishes)


@pytest.mark.unit
def test_second_stage_optimization_CoilSet():
    """Test optimizing CoilSet for a fixed axisymmetric equilibrium."""
    eq = get("SOLOVEV")
    R_coil = 3.5
    I = 100
    field = MixedCoilSet(
        FourierXYZCoil(
            current=I,
            X_n=[0, R_coil, 0],
            Y_n=[0, 0, R_coil],
            Z_n=[3, 0, 0],
            modes=[0, 1, -1],
        ),
        CoilSet(
            FourierPlanarCoil(
                current=I, center=[R_coil, 0, 0], normal=[0, 1, 0], r_n=2.5
            ),
            NFP=4,
            sym=True,
            check_intersection=False,
        ),
        check_intersection=False,
    )
    grid = LinearGrid(M=5)
    objective = ObjectiveFunction(
        QuadraticFlux(
            eq=eq, field=field, vacuum=True, eval_grid=grid, field_grid=LinearGrid(N=15)
        )
    )
    constraints = FixParameters(
        field,
        [
            {"X_n": True, "Y_n": True, "Z_n": True},
            {"r_n": True, "center": True, "normal": True, "current": True},
        ],
    )
    optimizer = Optimizer("lsq-exact")
    (field,), _ = optimizer.optimize(
        things=field,
        objective=objective,
        constraints=constraints,
        ftol=0,
        xtol=1e-7,
        gtol=0,
        verbose=2,
        maxiter=10,
    )

    # should be small current in the circular coil providing the vertical field
    np.testing.assert_allclose(field[0].current, 0, atol=1e-12)


@pytest.mark.slow
@pytest.mark.unit
def test_optimize_with_all_coil_types(DummyCoilSet, DummyMixedCoilSet):
    """Test optimizing for every type of coil and dummy coil sets."""
    sym_coils = load(load_from=str(DummyCoilSet["output_path_sym"]), file_format="hdf5")
    asym_coils = load(
        load_from=str(DummyCoilSet["output_path_asym"]), file_format="hdf5"
    )
    mixed_coils = load(
        load_from=str(DummyMixedCoilSet["output_path"]), file_format="hdf5"
    )
    nested_coils = MixedCoilSet(sym_coils, mixed_coils, check_intersection=False)
    eq = Equilibrium()
    # not attempting to accurately calc B for this test,
    # so make the grids very coarse
    quad_eval_grid = LinearGrid(M=2, sym=True)
    quad_field_grid = LinearGrid(N=2)

    def test(c, method):
        target = 11
        rtol = 1e-3
        # first just check that quad flux works for a couple iterations
        # as this is an expensive objective to compute
        obj = ObjectiveFunction(
            QuadraticFlux(
                eq=eq,
                field=c,
                vacuum=True,
                weight=1e-4,
                eval_grid=quad_eval_grid,
                field_grid=quad_field_grid,
            )
        )
        optimizer = Optimizer(method)
        (c,), _ = optimizer.optimize(c, obj, maxiter=2, ftol=0, xtol=1e-15)

        # now check with optimizing geometry and actually check result
        objs = [
            CoilLength(c, target=target),
        ]
        extra_msg = ""
        if isinstance(c, MixedCoilSet):
            # just to check they work without error
            objs.extend(
                [
                    CoilCurvature(c, target=0.5, weight=1e-2),
                    CoilTorsion(c, target=0, weight=1e-2),
                ]
            )
            rtol = 3e-2
            extra_msg = " with curvature and torsion obj"

        obj = ObjectiveFunction(objs)

        (c,), _ = optimizer.optimize(c, obj, maxiter=25, ftol=5e-3, xtol=1e-15)
        flattened_coils = tree_leaves(
            c, is_leaf=lambda x: isinstance(x, _Coil) and not isinstance(x, CoilSet)
        )
        lengths = [coil.compute("length")["length"] for coil in flattened_coils]
        np.testing.assert_allclose(
            lengths, target, rtol=rtol, err_msg=f"lengths {c}" + extra_msg
        )

    spline_coil = mixed_coils.coils[-1].copy()

    # single coil
    test(FourierPlanarCoil(), "fmintr")
    test(FourierRZCoil(), "fmintr")
    test(FourierXYZCoil(), "fmintr")
    test(spline_coil, "fmintr")

    # CoilSet
    test(sym_coils, "lsq-exact")
    test(asym_coils, "lsq-exact")

    # MixedCoilSet
    test(mixed_coils, "lsq-exact")
    test(nested_coils, "lsq-exact")


@pytest.mark.unit
def test_coilset_geometry_optimization():
    """Test optimizations with PlasmaCoilSetMinDistance and CoilSetMinDistance."""
    R0 = 5  # major radius of plasma
    a = 1.2  # minor radius of plasma
    phi0 = np.pi / 12  # initial angle of coil
    offset = 0.75  # target plasma-coil distance

    # circular tokamak
    surf = FourierRZToroidalSurface(
        R_lmn=np.array([R0, a]),
        Z_lmn=np.array([0, -a]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
    )
    eq = Equilibrium(Psi=3, surface=surf, NFP=1, M=2, N=0, sym=True)

    # symmetric coilset with 1 unique coil + 7 virtual coils
    # initial radius is too large for target plasma-coil distance
    # initial toroidal angle is too small for target coil-coil distance
    coil = FourierPlanarCoil(
        current=1e6,
        center=[R0, phi0, 0],
        normal=[0, 1, 0],
        r_n=[a + 2 * offset],
        basis="rpz",
    )
    coils = CoilSet(coil, NFP=4, sym=True)
    assert len(coils) == 1
    assert coils.num_coils == 8

    # grids
    plasma_grid = LinearGrid(M=8, zeta=64)
    coil_grid = LinearGrid(N=8)

    #### optimize coils with fixed equilibrium ####
    # optimizing for target coil-plasma distance and maximum coil-coil distance
    objective = ObjectiveFunction(
        (
            PlasmaCoilSetMinDistance(
                eq=eq,
                coil=coils,
                target=offset,
                weight=2,
                plasma_grid=plasma_grid,
                coil_grid=coil_grid,
                eq_fixed=True,
                coils_fixed=False,
            ),
            CoilSetMinDistance(
                coils,
                target=2 * np.pi * (R0 - offset) / coils.num_coils,
                grid=coil_grid,
            ),
        )
    )
    # only 2 free optimization variables are coil center position phi and radius r_n
    constraints = (
        FixCoilCurrent(coils),
        FixParameters(coils, {"center": np.array([0, 2]), "normal": True}),
    )
    optimizer = Optimizer("scipy-trf")
    [coils_opt], _ = optimizer.optimize(
        things=coils, objective=objective, constraints=constraints, verbose=2, copy=True
    )

    assert coils_opt[0].current == 1e6  # current was fixed
    np.testing.assert_allclose(  # check coils are equally spaced in toroidal angle phi
        coils_opt[0].center,
        [R0, np.pi / coils_opt.num_coils, 0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(coils_opt[0].normal, [0, 1, 0])  # normal was fixed
    np.testing.assert_allclose(
        coils_opt[0].r_n, a + offset, rtol=1e-2
    )  # check coil radius

    #### optimize coils with fixed surface ####
    # same optimization as above, but with a fixed surface instead of an equilibrium
    objective = ObjectiveFunction(
        (
            PlasmaCoilSetMinDistance(
                eq=surf,
                coil=coils,
                target=offset,
                weight=2,
                plasma_grid=plasma_grid,
                coil_grid=coil_grid,
                eq_fixed=True,
                coils_fixed=False,
            ),
            CoilSetMinDistance(
                coils,
                target=2 * np.pi * (R0 - offset) / coils.num_coils,
                grid=coil_grid,
            ),
        )
    )
    # only 2 free optimization variables are coil center position phi and radius r_n
    constraints = (
        FixCoilCurrent(coils),
        FixParameters(coils, {"center": np.array([0, 2]), "normal": True}),
    )
    optimizer = Optimizer("scipy-trf")
    [coils_opt], _ = optimizer.optimize(
        things=coils, objective=objective, constraints=constraints, verbose=2, copy=True
    )

    assert coils_opt[0].current == 1e6  # current was fixed
    np.testing.assert_allclose(  # check coils are equally spaced in toroidal angle phi
        coils_opt[0].center,
        [R0, np.pi / coils_opt.num_coils, 0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(coils_opt[0].normal, [0, 1, 0])  # normal was fixed
    np.testing.assert_allclose(
        coils_opt[0].r_n, a + offset, rtol=1e-2
    )  # check coil radius

    #### optimize surface with fixed coils ####
    # optimizing for target coil-plasma distance only

    def circle_constraint(params):
        """Constrain cross section of surface to be a circle."""
        return params["R_lmn"][1:] + jnp.flip(params["Z_lmn"])

    objective = ObjectiveFunction(
        (
            PlasmaCoilSetMinDistance(
                eq=surf,
                coil=coils,
                target=offset,
                plasma_grid=plasma_grid,
                coil_grid=coil_grid,
                eq_fixed=False,
                coils_fixed=True,
            ),
        )
    )
    # only 1 free optimization variable is surface minor radius
    constraints = (
        FixParameters(surf, {"R_lmn": np.array([0, 2])}),
        LinearObjectiveFromUser(circle_constraint, surf),
    )
    optimizer = Optimizer("scipy-trf")
    [surf_opt], _ = optimizer.optimize(
        things=[surf],
        objective=objective,
        constraints=constraints,
        verbose=2,
        copy=True,
    )

    # the R & Z boundary surface m=+/-1 coefficients should be equal magnitude and
    # have changed to match target offset
    np.testing.assert_allclose(
        coils[0].r_n,
        abs(surf_opt.R_lmn[surf_opt.R_basis.get_idx(M=1, N=0)]) + offset,
        rtol=2e-2,
    )
    np.testing.assert_allclose(
        coils[0].r_n,
        abs(surf_opt.Z_lmn[surf_opt.Z_basis.get_idx(M=-1, N=0)]) + offset,
        rtol=2e-2,
    )


@pytest.mark.unit
def test_ballooning_stability_opt():
    """Perform ballooning stability optimization with DESC."""
    try:
        eq = get("HELIOTRON")[-1]
    except TypeError:
        eq = get("HELIOTRON")

    # Flux surfaces on which to evaluate ballooning stability
    surfaces = [0.8]

    grid = LinearGrid(rho=jnp.array(surfaces), NFP=eq.NFP)
    eq_data_keys = ["iota"]

    data = eq.compute(eq_data_keys, grid=grid)
    iota = data["iota"]

    Nalpha = int(8)  # Number of field lines

    assert Nalpha == int(8), "Nalpha in the compute function hard-coded to 8!"

    # Field lines on which to evaluate ballooning stability
    alpha = jnp.linspace(0, np.pi, Nalpha + 1)[:Nalpha]

    # Number of toroidal transits of the field line
    ntor = int(3)

    # Number of point along a field line in ballooning space
    N0 = int(2.0 * ntor * eq.M_grid * eq.N_grid + 1)

    # range of the ballooning coordinate zeta
    zeta = np.linspace(-jnp.pi * ntor, jnp.pi * ntor, N0)

    lam2_initial = np.zeros(
        len(surfaces),
    )
    for i in range(len(surfaces)):
        rho = surfaces[i] * np.ones((N0 * Nalpha,))

        theta_PEST = alpha[:, None] + iota[0] * zeta
        zeta_full = jnp.tile(zeta, Nalpha)

        theta_PEST = theta_PEST.flatten()
        zeta_full = zeta_full.flatten()

        theta_coords = jnp.array([rho, theta_PEST, zeta_full]).T

        # Rootfinding theta for a given theta_PEST
        desc_coords = compute_theta_coords(
            eq, theta_coords, L_lmn=eq.L_lmn, tol=1e-8, maxiter=25
        )

        sfl_grid = Grid(desc_coords, sort=False)

        data_keys = ["ideal_ball_gamma2"]
        data = eq.compute(data_keys, grid=sfl_grid)

        lam2_initial[i] = data["ideal_ball_gamma2"]

    # Flux surfaces on which to evaluate ballooning stability
    surfaces_ball = surfaces

    # Determine which modes to unfix
    k = 2

    objs_ball = {}

    eq_ball_weight = 1.0e2

    for i, rho in enumerate(surfaces_ball):
        shift_arr = np.random.default_rng().uniform(-0.1, 0.1, Nalpha - 1)
        alpha = np.reshape(np.linspace(0, np.pi, Nalpha + 1)[:Nalpha], (-1, 1))
        alpha[1:, :] = alpha[1:, :] + np.reshape(shift_arr, (-1, 1))

        objs_ball[rho] = BallooningStability(
            eq=eq,
            rho=np.array([rho]),
            alpha=alpha,
            zetamax=ntor * jnp.pi,
            nzeta=N0,
            weight=eq_ball_weight,
        )

    modes_R = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    modes_Z = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]

    objective = ObjectiveFunction(tuple(objs_ball.values()))

    constraints = (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=modes_R),
        FixBoundaryZ(eq=eq, modes=modes_Z),
        FixPressure(eq=eq),
        FixIota(eq=eq),
        FixPsi(eq=eq),
    )

    optimizer = Optimizer("proximal-lsq-exact")
    (eq,), _ = optimizer.optimize(
        eq,
        objective,
        constraints,
        ftol=1e-4,
        xtol=1e-6,
        gtol=1e-6,
        maxiter=5,  # increase maxiter to 50 for a better result
        verbose=3,
        options={"initial_trust_ratio": 2e-3},
    )

    lam2_optimized = np.zeros(
        len(surfaces),
    )
    for i in range(len(surfaces)):
        rho = surfaces[i] * np.ones((N0 * Nalpha,))

        theta_PEST = alpha[:, None] + iota[0] * zeta
        zeta_full = jnp.tile(zeta, Nalpha)

        theta_PEST = theta_PEST.flatten()
        zeta_full = zeta_full.flatten()

        theta_coords = jnp.array([rho, theta_PEST, zeta_full]).T

        # Rootfinding theta for a given theta_PEST
        desc_coords = compute_theta_coords(
            eq, theta_coords, L_lmn=eq.L_lmn, tol=1e-8, maxiter=25
        )

        sfl_grid = Grid(desc_coords, sort=False)

        data_keys = ["ideal_ball_gamma2"]
        data = eq.compute(data_keys, grid=sfl_grid)

        lam2_optimized[i] = data["ideal_ball_gamma2"]

    assert lam2_optimized < lam2_initial
