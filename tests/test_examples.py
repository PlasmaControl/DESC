"""Regression tests to verify that DESC agrees with VMEC and itself.

Computes several benchmark equilibria and compares the solutions by measuring the
difference in areas between constant theta and rho contours.
"""

import numpy as np
import pytest
from netCDF4 import Dataset
from qic import Qic
from qsc import Qsc
from scipy.constants import mu_0

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
from desc.equilibrium.coords import get_rtz_grid
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import (
    FourierCurrentPotentialField,
    OmnigenousField,
    SplineMagneticField,
    ToroidalMagneticField,
    VerticalMagneticField,
    solve_regularized_surface_current,
)
from desc.objectives import (
    AspectRatio,
    BallooningStability,
    BoundaryError,
    CoilArclengthVariance,
    CoilCurvature,
    CoilLength,
    CoilSetMinDistance,
    CoilTorsion,
    CurrentDensity,
    ExternalObjective,
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
    SurfaceCurrentRegularization,
    SurfaceQuadraticFlux,
    ToroidalFlux,
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
        FixBoundaryR(eq=eq, modes=[0, 0, 0]),  # add a degenerate constraint to confirm
        # proximal-lsq-exact not affected by GH #1297
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
        eq, verbose=2, checkpoint_path=output_dir.join("ATF.h5"), jac_chunk_size=500
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
        constraints=get_fixed_boundary_constraints(eq=eq, profiles=True),
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
    grid_axis = LinearGrid(rho=0.0, theta=0.0, N=eq.N_grid, NFP=eq.NFP)
    # Make sure axis is same
    for eqq, string in zip(
        [eq, eq_lambda_fixed_0th_order, eq_lambda_fixed_1st_order],
        ["no lambda constraint", "lambda_fixed_0th_order", "lambda_fixed_1st_order"],
    ):
        np.testing.assert_array_almost_equal(orig_Rax_val, eqq.axis.R_n, err_msg=string)
        np.testing.assert_array_almost_equal(orig_Zax_val, eqq.axis.Z_n, err_msg=string)

        # Make sure iota of solved equilibrium is same on-axis as QSC

        iota = eqq.compute("iota", grid=LinearGrid(rho=0.0))["iota"]

        np.testing.assert_allclose(iota[0], qsc.iota, atol=1e-5, err_msg=string)

        # check lambda to match on-axis
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
def test_NAE_QSC_solve_near_axis_based_off_eq():
    """Test O(rho) NAE QSC constraints solve when qsc eq is not given."""
    qsc = Qsc.from_paper("precise QA")
    ntheta = 75
    r = 0.01
    N = 9
    eq = Equilibrium.from_near_axis(qsc, r=r, L=6, M=6, N=N, ntheta=ntheta)

    orig_Rax_val = eq.axis.R_n
    orig_Zax_val = eq.axis.Z_n

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
    grid_axis = LinearGrid(rho=0.0, theta=0.0, N=eq.N_grid, NFP=eq.NFP)
    # Make sure axis is same
    for eqq, string in zip(
        [eq, eq_lambda_fixed_0th_order, eq_lambda_fixed_1st_order],
        ["no lambda constraint", "lambda_fixed_0th_order", "lambda_fixed_1st_order"],
    ):
        np.testing.assert_array_almost_equal(orig_Rax_val, eqq.axis.R_n, err_msg=string)
        np.testing.assert_array_almost_equal(orig_Zax_val, eqq.axis.Z_n, err_msg=string)

        # Make sure iota of solved equilibrium is same on axis as QSC

        iota = eqq.compute("iota", grid=LinearGrid(rho=0.0))["iota"]

        np.testing.assert_allclose(iota[0], qsc.iota, atol=1e-5, err_msg=string)

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
    qic = Qic.from_paper("QI NFP2 r2", nphi=199, order="r1")
    qic.lasym = False  # don't need to consider stellarator asym for order 1 constraints
    ntheta = 75
    r = 0.01
    N = 9
    eq = Equilibrium.from_near_axis(qic, r=r, L=7, M=7, N=N, ntheta=ntheta)

    orig_Rax_val = eq.axis.R_n
    orig_Zax_val = eq.axis.Z_n

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

    # Make sure iota of solved equilibrium is same near axis as QIC
    iota = eq.compute("iota", grid=LinearGrid(rho=0.0))["iota"]

    np.testing.assert_allclose(iota[0], qic.iota, rtol=1e-5)

    grid_axis = LinearGrid(rho=0.0, theta=0.0, zeta=qic.phi, NFP=eq.NFP)
    phi = grid_axis.nodes[:, 2].squeeze()

    # check lambda to match on-axis
    lam_nae = eq.compute("lambda", grid=grid_axis)["lambda"]

    np.testing.assert_allclose(
        lam_nae, -qic.iota * qic.nu_spline(phi), atol=1e-4, rtol=1e-2
    )

    # check |B| on axis
    B_nae = eq.compute(["|B|"], grid=grid_axis)["|B|"]
    np.testing.assert_allclose(B_nae, qic.B0, atol=1e-3)


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
        FixBoundaryR(
            eq=eq, modes=[0, 0, 0]
        ),  # add a degenerate constraint to test fix of GH #1297 for lsq-auglag
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
        (eq, field), objective, constraints, maxiter=150, verbose=3
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
        softmin_alpha=5000,
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
        VacuumBoundaryError(eq=eq, field=ext_field, field_fixed=True),
        jac_chunk_size=1000,
        deriv_mode="batched",
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
    with pytest.warns(UserWarning, match="Vector potential"):
        # the mgrid file does not have the vector potential
        # saved so we will ignore the thrown warning
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


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_axisymmetric():
    """Test axisymmetric regcoil solution."""
    # make a simple axisymmetric vacuum equilibrium
    eq = load("./tests/inputs/vacuum_circular_tokamak.h5")
    # no phi_SV is needed since it is axisymmetric,
    # so phi_mn should be zero when running with simple regularization
    # especially with a nonzero lambda_regularization
    surf_winding = FourierRZToroidalSurface.constant_offset_surface(eq.surface, 2)

    surface_current_field = FourierCurrentPotentialField.from_surface(
        surf_winding, M_Phi=1, N_Phi=1, sym_Phi="sin"
    )
    surface_current_field, data = solve_regularized_surface_current(
        surface_current_field,
        eq,
        lambda_regularization=1e-30,  # little regularization to avoid singular matrix
        vacuum=True,
        verbose=2,
        regularization_type="regcoil",
        chunk_size=20,
    )
    # is a list of length one, index into it
    surface_current_field = surface_current_field[0]
    chi_B = data["chi^2_B"][0]
    phi_mn_opt = surface_current_field.Phi_mn
    G = surface_current_field.G
    np.testing.assert_allclose(phi_mn_opt, 0, atol=1e-6)
    np.testing.assert_allclose(chi_B, 0, atol=1e-10)
    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords,
        source_grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP),
        chunk_size=20,
    )
    np.testing.assert_allclose(B, B_from_surf, rtol=1e-4, atol=1e-8)

    grid = LinearGrid(N=10, M=10, NFP=surface_current_field.NFP)
    correct_phi = G * grid.nodes[:, 2] / 2 / np.pi
    np.testing.assert_allclose(
        surface_current_field.compute("Phi", grid=grid)["Phi"], correct_phi, atol=5e-9
    )
    surface_current_field.change_Phi_resolution(M=2, N=2)
    # test with lambda_regularization large, should have no phi_mn
    surface_current_field, data = solve_regularized_surface_current(
        surface_current_field,
        eq=eq,
        eval_grid=LinearGrid(M=10, N=10, NFP=eq.NFP, sym=eq.sym),
        source_grid=LinearGrid(M=40, N=40, NFP=eq.NFP),
        lambda_regularization=1e4,
        vacuum=True,
        regularization_type="simple",
        chunk_size=20,
    )
    surface_current_field = surface_current_field[0]
    phi_mn_opt = surface_current_field.Phi_mn
    np.testing.assert_allclose(phi_mn_opt, 0, atol=1e-16)
    np.testing.assert_allclose(data["chi^2_B"][0], 0, atol=1e-10)
    np.testing.assert_allclose(
        surface_current_field.compute("Phi", grid=grid)["Phi"], correct_phi, atol=1e-16
    )
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords,
        source_grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP),
        chunk_size=20,
    )
    np.testing.assert_allclose(B, B_from_surf, rtol=1e-4, atol=1e-8)

    # test with half the current given external to winding surface
    surface_current_field, data = solve_regularized_surface_current(
        surface_current_field,
        eq=eq,
        eval_grid=LinearGrid(M=10, N=10, NFP=eq.NFP, sym=eq.sym),
        source_grid=LinearGrid(M=40, N=80, NFP=eq.NFP),
        lambda_regularization=1e4,
        # negate the B0 because a negative G corresponds to a positive B toroidal
        # and we want this to provide half the field the surface current's
        # G is providing, in the same direction
        external_field=ToroidalMagneticField(B0=-mu_0 * (G / 2) / 2 / np.pi, R0=1),
        vacuum=True,
        chunk_size=20,
    )
    surface_current_field = surface_current_field[0]
    phi_mn_opt = surface_current_field.Phi_mn
    np.testing.assert_allclose(G / 2, surface_current_field.G, atol=1e-8)
    np.testing.assert_allclose(phi_mn_opt, 0, atol=1e-9)
    np.testing.assert_allclose(
        surface_current_field.compute("Phi", grid=grid)["Phi"],
        correct_phi / 2,
        atol=1e-9,
    )
    np.testing.assert_allclose(data["chi^2_B"][0], 0, atol=1e-11)
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords,
        source_grid=LinearGrid(M=200, N=200, NFP=surf_winding.NFP),
        chunk_size=20,
    )
    np.testing.assert_allclose(B, B_from_surf * 2, rtol=1e-4, atol=1e-8)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_modular_check_B(regcoil_modular_coils):
    """Test precise QA modular (helicity=(1,0)) regcoil solution."""
    (
        data,
        initial_surface_current_field,
        eq,
    ) = regcoil_modular_coils
    chi_B = data["chi^2_B"][0]
    surface_current_field = initial_surface_current_field.copy()

    np.testing.assert_array_less(chi_B, 1e-6)
    coords = eq.compute(["R", "phi", "Z", "B"], grid=LinearGrid(M=20, N=20, NFP=eq.NFP))
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords,
        source_grid=LinearGrid(M=60, N=60, NFP=surface_current_field.NFP),
        basis="rpz",
        chunk_size=20,
    )
    np.testing.assert_allclose(B, B_from_surf, rtol=5e-2, atol=5e-4)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_windowpane_check_B(regcoil_windowpane_coils):
    """Test precise QA windowpane (helicity=(0,0)) regcoil solution."""
    (
        data,
        surface_current_field,
        eq,
    ) = regcoil_windowpane_coils
    assert surface_current_field.I == 0
    assert surface_current_field.G == 0

    chi_B = data["chi^2_B"][0]
    np.testing.assert_array_less(chi_B, 1e-7)
    coords = eq.compute(["R", "phi", "Z", "B"], grid=data["eval_grid"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    field = surface_current_field + data["external_field"]
    B_from_surf = field.compute_magnetic_field(
        coords,
        source_grid=data["source_grid"],
        basis="rpz",
        chunk_size=20,
    )
    np.testing.assert_allclose(B, B_from_surf, rtol=1e-2, atol=5e-4)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_PF_check_B(regcoil_PF_coils):
    """Test precise QA PF (helicity=(0,2)) regcoil solution."""
    (data, surface_current_field, eq) = regcoil_PF_coils
    assert surface_current_field.G == 0
    assert abs(surface_current_field.I) > 0
    chi_B = data["chi^2_B"][0]
    np.testing.assert_array_less(chi_B, 1e-5)
    coords = eq.compute(["R", "phi", "Z", "B"], grid=data["eval_grid"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    field = surface_current_field + data["external_field"]
    B_from_surf = field.compute_magnetic_field(
        coords,
        source_grid=data["source_grid"],
        basis="rpz",
    )
    np.testing.assert_allclose(B, B_from_surf, rtol=4e-2, atol=5e-4)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_regcoil_helical_coils_check_objective_method(
    regcoil_helical_coils_scan,
):
    """Test precise QA helical coil regcoil solution."""
    (data, initial_surface_current_field, eq) = regcoil_helical_coils_scan
    lam_index = 1
    lam = data["lambda_regularization"][lam_index]
    initial_surface_current_field.Phi_mn = data["Phi_mn"][lam_index]
    surface_current_field = initial_surface_current_field.copy()

    # reset the Phi_mn
    surface_current_field.Phi_mn = surface_current_field.Phi_mn.at[:].set(0.0)
    constraints = (  # now fix all but Phi_mn
        FixParameters(
            surface_current_field,
            params={"I": True, "G": True, "R_lmn": True, "Z_lmn": True},
        ),
    )
    eval_grid = data["eval_grid"]
    sgrid = data["source_grid"]

    obj = QuadraticFlux(
        field=surface_current_field,
        eq=eq,
        eval_grid=eval_grid,
        field_grid=sgrid,
        vacuum=True,
    )

    objective = ObjectiveFunction(
        (
            obj,
            SurfaceCurrentRegularization(
                surface_current_field=surface_current_field,
                weight=np.sqrt(lam),
                source_grid=sgrid,
            ),
        ),
        use_jit=False,
    )

    optimizer = Optimizer("lsq-exact")

    (surface_current_field,), _ = optimizer.optimize(
        (surface_current_field,),
        objective,
        constraints,
        verbose=1,
        maxiter=1,
        ftol=0,
        gtol=0,
        xtol=1e-16,
        options={"initial_trust_radius": np.inf, "tr_method": "cho"},
    )

    coords = eq.compute(["R", "phi", "Z", "B"])
    B = coords["B"]
    coords = np.vstack([coords["R"], coords["phi"], coords["Z"]]).T
    B_from_surf = surface_current_field.compute_magnetic_field(
        coords,
        source_grid=sgrid,
        basis="rpz",
        chunk_size=20,
    )
    B_from_orig_surf = initial_surface_current_field.compute_magnetic_field(
        coords,
        source_grid=sgrid,
        basis="rpz",
        chunk_size=20,
    )
    np.testing.assert_allclose(B, B_from_surf, atol=6e-4, rtol=5e-2)
    np.testing.assert_allclose(B_from_orig_surf, B_from_surf, atol=1e-8, rtol=1e-8)


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
def test_qfm_optimization_with_analytic_field():
    """Test analytic field optimization to reduce quadratic flux.

    Checks that surface becomes axisymmetric with non-axisymmetric surface
    and axisymmetric field.
    """
    surface = get("HELIOTRON", data="boundary")
    surface.change_resolution(M=2, N=1)
    field = ToroidalMagneticField(1, 1)
    eval_grid = LinearGrid(
        rho=np.array([1.0]),
        M=10,
        N=4,
        NFP=surface.NFP,
        sym=True,
    )

    optimizer = Optimizer("lsq-exact")

    constraints = ()
    quadflux_obj = SurfaceQuadraticFlux(
        surface=surface,
        field=field,
        eval_grid=eval_grid,
        field_fixed=True,
    )
    torflux = ToroidalFlux(
        eq=surface,
        field=field,
        eq_fixed=False,
        field_fixed=True,
    )
    torflux.build()
    current_torflux = torflux.compute(surface.params_dict)
    torflux = ToroidalFlux(
        eq=surface,
        field=field,
        eq_fixed=False,
        field_fixed=True,
        target=current_torflux,
    )

    objective = ObjectiveFunction((quadflux_obj, torflux))
    (surface,), __ = optimizer.optimize(
        surface,
        objective=objective,
        constraints=constraints,
        ftol=1e-14,
        gtol=1e-14,
        xtol=1e-14,
        verbose=3,
    )

    # optimizer should make surface basically axisymmetric
    # to get to Bnorm = 0
    nonax_R = surface.R_lmn[np.where(surface.R_basis.modes[:, 2] != 0)]
    nonax_Z = surface.Z_lmn[np.where(surface.Z_basis.modes[:, 2] != 0)]
    np.testing.assert_allclose(nonax_R, 0, atol=1e-7)
    np.testing.assert_allclose(nonax_Z, 0, atol=1e-7)


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
        gtol=0,
        maxiter=100,
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
@pytest.mark.parametrize(
    "coil_type",
    [
        "FourierPlanarCoil",
        "FourierRZCoil",
        "FourierXYZCoil",
        "SplineXYZCoil",
        "CoilSet sym",
        "CoilSet asym",
        "MixedCoilSet",
        "nested CoilSet",
    ],
)
def test_optimize_with_all_coil_types(DummyCoilSet, DummyMixedCoilSet, coil_type):
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

    spline_coil = mixed_coils.coils[-1].copy()

    types = {
        "FourierPlanarCoil": (FourierPlanarCoil(), "fmintr"),
        "FourierRZCoil": (FourierRZCoil(), "fmintr"),
        "FourierXYZCoil": (FourierXYZCoil(), "fmintr"),
        "SplineXYZCoil": (spline_coil, "fmintr"),
        "CoilSet sym": (sym_coils, "lsq-exact"),
        "CoilSet asym": (asym_coils, "lsq-exact"),
        "MixedCoilSet": (mixed_coils, "lsq-exact"),
        "nested CoilSet": (nested_coils, "lsq-exact"),
    }
    c, method = types[coil_type]

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
    (cc,), _ = optimizer.optimize(c, obj, maxiter=2, ftol=0, xtol=1e-8, copy=True)

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

    (c,), _ = optimizer.optimize(c, obj, maxiter=25, ftol=5e-3, xtol=1e-8)
    flattened_coils = tree_leaves(
        c, is_leaf=lambda x: isinstance(x, _Coil) and not isinstance(x, CoilSet)
    )
    lengths = [coil.compute("length")["length"] for coil in flattened_coils]
    np.testing.assert_allclose(
        lengths, target, rtol=rtol, err_msg=f"lengths {c}" + extra_msg
    )


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
@pytest.mark.slow
def test_external_vs_generic_objectives(tmpdir_factory):
    """Test ExternalObjective compared to GenericObjective."""
    target = np.array([6.2e-3, 1.1e-1, 6.5e-3, 0])  # values at p_l = [2e2, -2e2]

    def data_from_vmec(eq, path="", surfs=8):
        # write data
        file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")
        NFP = eq.NFP
        M = eq.M
        N = eq.N
        M_nyq = M + 4
        N_nyq = N + 2 if N > 0 else 0
        s_full = np.linspace(0, 1, surfs)
        r_full = np.sqrt(s_full)
        file.createDimension("radius", surfs)
        grid_full = LinearGrid(M=M_nyq, N=N_nyq, NFP=NFP, rho=r_full)
        data_full = eq.compute(["p"], grid=grid_full)
        data_quad = eq.compute(["<beta>_vol", "<beta_pol>_vol", "<beta_tor>_vol"])
        betatotal = file.createVariable("betatotal", np.float64)
        betatotal[:] = data_quad["<beta>_vol"]
        betapol = file.createVariable("betapol", np.float64)
        betapol[:] = data_quad["<beta_pol>_vol"]
        betator = file.createVariable("betator", np.float64)
        betator[:] = data_quad["<beta_tor>_vol"]
        presf = file.createVariable("presf", np.float64, ("radius",))
        presf[:] = grid_full.compress(data_full["p"])
        file.close()
        # read data
        file = Dataset(path, mode="r")
        betatot = float(file.variables["betatotal"][0])
        betapol = float(file.variables["betapol"][0])
        betator = float(file.variables["betator"][0])
        presf1 = float(file.variables["presf"][-1])
        file.close()
        return np.atleast_1d([betatot, betapol, betator, presf1])

    eq0 = get("SOLOVEV")
    optimizer = Optimizer("lsq-exact")

    # generic
    objective = ObjectiveFunction(
        (
            GenericObjective("<beta>_vol", thing=eq0, target=target[0]),
            GenericObjective("<beta_pol>_vol", thing=eq0, target=target[1]),
            GenericObjective("<beta_tor>_vol", thing=eq0, target=target[2]),
            GenericObjective(
                "p", thing=eq0, target=0, grid=LinearGrid(rho=[1], M=0, N=0)
            ),
        )
    )
    constraints = FixParameters(
        eq0,
        {
            "R_lmn": True,
            "Z_lmn": True,
            "L_lmn": True,
            "p_l": np.arange(2, len(eq0.p_l)),
            "i_l": True,
            "Psi": True,
        },
    )
    [eq_generic], _ = optimizer.optimize(
        things=eq0,
        objective=objective,
        constraints=constraints,
        copy=True,
        ftol=0,
        verbose=2,
    )

    # external
    dir = tmpdir_factory.mktemp("results")
    path = dir.join("wout_result.nc")
    objective = ObjectiveFunction(
        ExternalObjective(
            eq=eq0,
            fun=data_from_vmec,
            dim_f=4,
            fun_kwargs={"path": path, "surfs": 8},
            vectorized=False,
            target=target,
        )
    )
    constraints = FixParameters(
        eq0,
        {
            "R_lmn": True,
            "Z_lmn": True,
            "L_lmn": True,
            "p_l": np.arange(2, len(eq0.p_l)),
            "i_l": True,
            "Psi": True,
        },
    )
    [eq_external], _ = optimizer.optimize(
        things=eq0,
        objective=objective,
        constraints=constraints,
        copy=True,
        ftol=0,
        verbose=2,
    )

    np.testing.assert_allclose(eq_generic.p_l, eq_external.p_l)
    np.testing.assert_allclose(eq_generic.p_l[:2], [2e2, -2e2], rtol=4e-2)
    np.testing.assert_allclose(eq_external.p_l[:2], [2e2, -2e2], rtol=4e-2)


@pytest.mark.unit
@pytest.mark.optimize
def test_coil_arclength_optimization():
    """Test coil arclength variance optimization."""
    c1 = FourierXYZCoil()
    c1.change_resolution(N=5)
    target_length = 2 * c1.compute("length")["length"]
    obj = ObjectiveFunction(
        (
            CoilLength(c1, target=target_length),
            CoilCurvature(c1, target=1, weight=1e-2),
        )
    )
    obj2 = ObjectiveFunction(
        (
            CoilLength(c1, target=target_length),
            CoilCurvature(c1, target=1, weight=1e-2),
            CoilArclengthVariance(c1, target=0, weight=100),
        )
    )
    opt = Optimizer("lsq-exact")
    (coil_opt_without_arc_obj,), _ = opt.optimize(
        c1, objective=obj, verbose=3, copy=True, ftol=1e-6
    )
    (coil_opt_with_arc_obj,), _ = opt.optimize(
        c1, objective=obj2, verbose=3, copy=True, ftol=1e-6, maxiter=200
    )
    xs1 = coil_opt_with_arc_obj.compute("x_s")["x_s"]
    xs2 = coil_opt_without_arc_obj.compute("x_s")["x_s"]
    np.testing.assert_allclose(
        coil_opt_without_arc_obj.compute("length")["length"], target_length, rtol=1e-4
    )
    np.testing.assert_allclose(
        coil_opt_with_arc_obj.compute("length")["length"], target_length, rtol=1e-4
    )
    np.testing.assert_allclose(np.var(np.linalg.norm(xs1, axis=1)), 0, atol=1e-5)
    assert np.var(np.linalg.norm(xs1, axis=1)) < np.var(np.linalg.norm(xs2, axis=1))


@pytest.mark.regression
def test_ballooning_stability_opt():
    """Perform ballooning stability optimization with DESC."""
    eq = get("HELIOTRON")

    # Flux surfaces on which to evaluate ballooning stability
    surfaces = [0.8]

    grid = LinearGrid(rho=jnp.array(surfaces), NFP=eq.NFP)
    eq_data_keys = ["iota"]

    data = eq.compute(eq_data_keys, grid=grid)

    Nalpha = 8  # Number of field lines

    # Field lines on which to evaluate ballooning stability
    alpha = jnp.linspace(0, np.pi, Nalpha)

    # Number of toroidal transits of the field line
    ntor = 2

    # Number of point along a field line in ballooning space
    N0 = 2 * ntor * eq.M_grid * eq.N_grid + 1

    # range of the ballooning coordinate zeta
    zeta = np.linspace(-jnp.pi * ntor, jnp.pi * ntor, N0)

    lam2_initial = np.zeros(
        len(surfaces),
    )
    for i in range(len(surfaces)):
        rho = surfaces[i]

        grid = get_rtz_grid(
            eq,
            rho,
            alpha,
            zeta,
            coordinates="raz",
            period=(np.inf, 2 * np.pi, np.inf),
        )

        data_keys = ["ideal ballooning lambda"]
        data = eq.compute(data_keys, grid=grid)

        lam2_initial[i] = np.max(data["ideal ballooning lambda"])

    # Flux surfaces on which to evaluate ballooning stability
    surfaces_ball = surfaces

    # Determine which modes to unfix
    k = 2

    objs_ball = {}

    eq_ball_weight = 1.0e2

    for i, rho in enumerate(surfaces_ball):
        alpha = np.linspace(0, np.pi, Nalpha)

        objs_ball[rho] = BallooningStability(
            eq=eq,
            rho=np.array([rho]),
            alpha=alpha,
            nturns=ntor,
            nzetaperturn=2 * (eq.M_grid * eq.N_grid),
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

    # aspect ratio of the original HELIOTRON is 10.48
    objective = ObjectiveFunction(
        (AspectRatio(eq=eq, bounds=(0, 12)),) + tuple(objs_ball.values())
    )

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
        maxiter=2,  # increase maxiter to 50 for a better result
        verbose=3,
        options={"initial_trust_ratio": 2e-3},
    )

    lam2_optimized = np.zeros(
        len(surfaces),
    )
    for i in range(len(surfaces)):
        rho = surfaces[i]

        grid = get_rtz_grid(
            eq,
            rho,
            alpha,
            zeta,
            coordinates="raz",
            period=(np.inf, 2 * np.pi, np.inf),
        )

        data_keys = ["ideal ballooning lambda"]
        data = eq.compute(data_keys, grid=grid)

        lam2_optimized[i] = np.max(data["ideal ballooning lambda"])

    assert lam2_initial - lam2_optimized >= 1.8e-2


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.optimize
def test_signed_PlasmaVesselDistance():
    """Tests that signed distance works with surface optimization."""
    eq = get("HELIOTRON")
    eq.change_resolution(M=2, N=2)

    surf = eq.surface.copy()
    surf.change_resolution(M=1, N=1)

    target_dist = -0.25

    grid = LinearGrid(M=10, N=4, NFP=eq.NFP)
    obj = PlasmaVesselDistance(
        surface=surf,
        eq=eq,
        target=target_dist,
        surface_grid=grid,
        plasma_grid=grid,
        use_signed_distance=True,
        eq_fixed=True,
    )
    objective = ObjectiveFunction((obj,))

    optimizer = Optimizer("lsq-exact")
    (surf,), _ = optimizer.optimize(
        (surf,), objective, verbose=3, maxiter=60, ftol=1e-8, xtol=1e-9
    )

    np.testing.assert_allclose(
        obj.compute(*obj.xs(surf)), target_dist, atol=1e-2, err_msg="Using hardmin"
    )

    # with softmin
    surf = eq.surface.copy()
    surf.change_resolution(M=1, N=1)
    obj = PlasmaVesselDistance(
        surface=surf,
        eq=eq,
        target=target_dist,
        surface_grid=grid,
        plasma_grid=grid,
        use_signed_distance=True,
        use_softmin=True,
        softmin_alpha=100,
        eq_fixed=True,
    )
    objective = ObjectiveFunction((obj,))

    optimizer = Optimizer("lsq-exact")
    (surf,), _ = optimizer.optimize(
        (surf,),
        objective,
        verbose=3,
        maxiter=60,
        ftol=1e-8,
        xtol=1e-9,
    )

    np.testing.assert_allclose(
        obj.compute(*obj.xs(surf)), target_dist, atol=1e-2, err_msg="Using softmin"
    )

    # with changing eq
    eq = Equilibrium(M=1, N=1)
    surf = eq.surface.copy()
    surf.change_resolution(M=1, N=1)
    grid = LinearGrid(M=20, N=8, NFP=eq.NFP)

    obj = PlasmaVesselDistance(
        surface=surf,
        eq=eq,
        target=target_dist,
        surface_grid=grid,
        plasma_grid=grid,
        use_signed_distance=True,
    )
    objective = ObjectiveFunction(obj)

    optimizer = Optimizer("lsq-exact")
    (eq, surf), _ = optimizer.optimize(
        (eq, surf),
        objective,
        constraints=(FixParameters(surf),),
        verbose=3,
        maxiter=60,
        ftol=1e-8,
        xtol=1e-9,
    )

    np.testing.assert_allclose(
        obj.compute(*obj.xs(eq, surf)),
        target_dist,
        atol=1e-2,
        err_msg="allowing eq to change",
    )


@pytest.mark.unit
def test_continuation_L_res():
    """Test for fix to gh issue 1346."""
    # previously this would throw a warning then an error
    eq = Equilibrium(L=8, M=6, N=0)
    eqf = solve_continuation_automatic(eq)
    assert len(eqf) == 2
    assert eqf[0].L == 6
    assert eqf[0].M == 6
    assert eqf[-1].L == 8
    assert eqf[-1].M == 6
