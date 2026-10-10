"""Tests for ProximalProjectionFreeBoundary."""

import numpy as np
import pytest

from desc.coils import initialize_modular_coils
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.magnetic_fields import (
    FourierCurrentPotentialField,
    SplineMagneticField,
    SumMagneticField,
    ToroidalMagneticField,
    VerticalMagneticField,
)
from desc.objectives import (
    AspectRatio,
    BoundaryError,
    CoilLength,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixParameters,
    FixPressure,
    FixPsi,
    ForceBalance,
    MercierStability,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    VacuumBoundaryError,
    Volume,
)
from desc.optimize import Optimizer, ProximalProjection, ProximalProjectionFreeBoundary
from desc.profiles import PowerSeriesProfile

from .utils import area_difference_desc


def _vacuum_stellarator(M):
    """Vacuum stellarator with mgrid coils, same as test_freeb_vacuum."""
    field = SplineMagneticField.from_mgrid(
        "tests/inputs/mgrid_test.nc", extcur=[4700.0, 1000.0]
    )
    surf = FourierRZToroidalSurface(
        R_lmn=[0.70, 0.10],
        modes_R=[[0, 0], [1, 0]],
        Z_lmn=[-0.10],
        modes_Z=[[-1, 0]],
        NFP=5,
    )
    eq = Equilibrium(M=M, N=M, Psi=-0.035, surface=surf)
    eq.solve(verbose=0)
    return eq, field


def _sheet_current_tokamak():
    """Finite beta tokamak with edge pressure, a sheet current and analytic fields."""
    surf = FourierRZToroidalSurface(
        R_lmn=[3.0, 1.0], modes_R=[[0, 0], [1, 0]], Z_lmn=[-1.0], modes_Z=[[-1, 0]]
    )
    eq = Equilibrium(
        L=3,
        M=3,
        N=0,
        Psi=3.0,
        surface=surf,
        pressure=PowerSeriesProfile([1e4, 0, -7e3]),
        current=PowerSeriesProfile([0, 0, 3e5, 0, -1.5e5]),
    )
    eq.solve(verbose=0)
    eq.surface = FourierCurrentPotentialField.from_surface(eq.surface, M_Phi=2, N_Phi=0)
    field = SumMagneticField(
        [ToroidalMagneticField(0.93, 3.0), VerticalMagneticField(-0.02)]
    )
    return eq, field


def _reference_jacobian(prox, gobjs, bobjs, eq):
    """Free boundary Jacobian assembled from a stock ProximalProjection.

    The stock ProximalProjection of the stacked objective [G; B] gives the fixed
    boundary total derivatives wrt boundary b and optimization variables q, from which
    the free boundary Jacobian is dG/dq - dG/db Z (dB/db Z)⁺ dB/dq.
    """
    ref = ProximalProjection(
        ObjectiveFunction(gobjs + bobjs),
        ObjectiveFunction(ForceBalance(eq)),
        eq,
        solve_options={"solve_during_proximal_build": False},
    )
    ref.build(verbose=0)
    # stock ProximalProjection maps Rb_lmn, Zb_lmn only to R_lmn, Z_lmn. Also keep the
    # direct dependence (used for the sheet current) to compare the same thing.
    dxdc = np.array(ref._dxdc)
    offset = 0
    for arg in ref._args:
        if arg in ["Rb_lmn", "Zb_lmn"]:
            dxdc[eq.x_idx[arg], np.arange(offset, offset + eq.dimensions[arg])] = 1
        offset += eq.dimensions[arg]
    ref._dxdc = dxdc
    J = np.asarray(ref.jac_scaled_error(ref.x(*ref.things)))

    offset, bcols, qcols = 0, [], []
    for t, dim in zip(ref.things, ref._dimc_per_thing):
        if t is eq:
            o = offset
            for arg in ref._args:
                n = eq.dimensions[arg]
                (bcols if arg in prox._fb_args else qcols).extend(range(o, o + n))
                o += n
        else:
            qcols.extend(range(offset, offset + dim))
        offset += dim
    nG = prox.dim_f
    Z = np.asarray(prox._fb_Z)
    JGq, JGb = J[:nG][:, qcols], J[:nG][:, bcols] @ Z
    JBq, JBb = J[nG:][:, qcols], J[nG:][:, bcols] @ Z
    rcond = np.finfo(JBb.dtype).eps * max(JBb.shape)
    return JGq - JGb @ np.linalg.pinv(JBb, rcond=rcond) @ JBq


@pytest.mark.unit
def test_free_boundary_jacobian_stellarator():
    """Compare jacobian to one assembled from the fixed boundary jacobians."""
    eq, field = _vacuum_stellarator(M=3)
    gobjs = (
        AspectRatio(eq),
        Volume(eq),
        QuasisymmetryTwoTerm(eq, helicity=(1, eq.NFP)),
    )
    bobjs = (VacuumBoundaryError(eq, field, field_fixed=False),)
    # only let the low order boundary modes change, to also test the constraints on
    # the free boundary unknowns
    R_modes = eq.surface.R_basis.modes[
        np.max(np.abs(eq.surface.R_basis.modes), 1) > 2, :
    ]
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > 2, :
    ]
    prox = ProximalProjectionFreeBoundary(
        ObjectiveFunction(gobjs),
        ObjectiveFunction((ForceBalance(eq),) + bobjs),
        eq,
        solve_options={"solve_during_proximal_build": False},
        free_boundary_options={
            "constraints": (
                FixBoundaryR(eq, modes=R_modes),
                FixBoundaryZ(eq, modes=Z_modes),
            )
        },
    )
    prox.build(verbose=0)
    # optimization variables are the eq profiles + Psi and the coil currents
    assert prox._args == [arg for arg in prox._args if arg not in prox._fb_args]
    assert prox.dim_x == (sum(eq.dimensions[arg] for arg in prox._args) + field.dim_x)
    assert prox._fb_Z.shape[1] < prox._x_idx_fb.size

    x = prox.x(*prox.things)
    J = np.asarray(prox.jac_scaled_error(x))
    J_ref = _reference_jacobian(prox, gobjs, bobjs, eq)
    np.testing.assert_allclose(J, J_ref, rtol=1e-10, atol=1e-10 * np.abs(J_ref).max())

    # jvp and grad should be consistent with the jacobian
    v = np.random.default_rng(0).normal(size=x.size)
    np.testing.assert_allclose(
        prox.jvp_scaled_error(v, x), J @ v, rtol=1e-10, atol=1e-12
    )
    f = prox.compute_scaled_error(x)
    np.testing.assert_allclose(prox.grad(x), f @ J, rtol=1e-10, atol=1e-12)

    # boundary prediction for a step, from the derivatives stored by the jacobian
    prox._x_old = x
    dx = 1e-3 * v * np.abs(x).max()
    b1 = np.asarray(prox._predict_boundary(x + dx))
    np.testing.assert_allclose(b1, prox._dbdv[2].T @ dx, rtol=1e-8, atol=1e-14)
    # predictions stay in the null space of the constraints on the boundary
    Z = np.asarray(prox._fb_Z)
    np.testing.assert_allclose(Z @ (Z.T @ b1), b1, atol=1e-12)
    prox._fb_predict_order = 2
    b2 = np.asarray(prox._predict_boundary(x + dx)) - b1
    assert np.all(np.isfinite(b2))
    assert np.linalg.norm(b2) <= 1.01 * prox._fb_predict_tr_ratio * np.linalg.norm(b1)
    np.testing.assert_allclose(Z @ (Z.T @ b2), b2, atol=1e-12)


@pytest.mark.unit
def test_free_boundary_jacobian_stellarator_coils():
    """Compare jacobian for single stage with filamentary coil shapes as variables."""
    eq, _ = _vacuum_stellarator(M=3)
    coils = initialize_modular_coils(eq, num_coils=2, r_over_a=3.0)
    # objective also depends on the coils directly, and use blocked derivatives
    gobjs = (
        AspectRatio(eq),
        QuasisymmetryTwoTerm(eq, helicity=(1, eq.NFP)),
        CoilLength(coils),
    )
    bobjs = (VacuumBoundaryError(eq, coils, field_fixed=False),)
    prox = ProximalProjectionFreeBoundary(
        ObjectiveFunction(gobjs, deriv_mode="blocked"),
        ObjectiveFunction((ForceBalance(eq),) + bobjs),
        eq,
        solve_options={"solve_during_proximal_build": False},
    )
    prox.build(verbose=0)
    assert prox.things[0] is eq and prox.things[1] is coils
    assert prox.dim_x == sum(eq.dimensions[arg] for arg in prox._args) + coils.dim_x

    x = prox.x(*prox.things)
    J = np.asarray(prox.jac_scaled_error(x))
    J_ref = _reference_jacobian(prox, gobjs, bobjs, eq)
    np.testing.assert_allclose(J, J_ref, rtol=1e-10, atol=1e-10 * np.abs(J_ref).max())
    # coil shapes change the boundary, so they should change the eq objectives
    coil_cols = slice(prox.dim_x - coils.dim_x, prox.dim_x)
    assert np.linalg.norm(J[:-2, coil_cols]) > 0


@pytest.mark.unit
def test_free_boundary_jacobian_sheet_current():
    """Compare jacobian for finite beta with sheet current and profiles as variables."""
    eq, field = _sheet_current_tokamak()
    gobjs = (AspectRatio(eq), Volume(eq), MercierStability(eq))
    bobjs = (BoundaryError(eq, field, field_fixed=False),)
    prox = ProximalProjectionFreeBoundary(
        ObjectiveFunction(gobjs),
        ObjectiveFunction((ForceBalance(eq),) + bobjs),
        eq,
        solve_options={"solve_during_proximal_build": False},
    )
    prox.build(verbose=0)
    for arg in ["Rb_lmn", "Zb_lmn", "I", "G", "Phi_mn"]:
        assert arg in prox._fb_args
        assert arg not in prox._args
    for arg in ["p_l", "c_l", "Psi"]:
        assert arg in prox._args

    x = prox.x(*prox.things)
    J = np.asarray(prox.jac_scaled_error(x))
    J_ref = _reference_jacobian(prox, gobjs, bobjs, eq)
    np.testing.assert_allclose(J, J_ref, rtol=1e-10, atol=1e-10 * np.abs(J_ref).max())


@pytest.mark.unit
def test_free_boundary_errors():
    """Test errors for things ProximalProjectionFreeBoundary can't handle."""
    eq = Equilibrium(M=2, N=0)
    field = SumMagneticField(
        [ToroidalMagneticField(1.0, 10.0), VerticalMagneticField(0.0)]
    )
    obj = ObjectiveFunction(AspectRatio(eq))
    with pytest.raises(ValueError, match="equilibrium constraint"):
        ProximalProjectionFreeBoundary(
            obj, ObjectiveFunction(VacuumBoundaryError(eq, field)), eq
        )
    with pytest.raises(ValueError, match="free boundary constraint"):
        ProximalProjectionFreeBoundary(obj, ObjectiveFunction(ForceBalance(eq)), eq)
    with pytest.raises(ValueError, match="general nonlinear constraints"):
        ProximalProjectionFreeBoundary(
            obj,
            ObjectiveFunction(
                (ForceBalance(eq), VacuumBoundaryError(eq, field), Volume(eq))
            ),
            eq,
        )
    # optimizing the field, but the boundary error doesn't know it can change
    with pytest.raises(ValueError, match="field_fixed=False"):
        prox = ProximalProjectionFreeBoundary(
            ObjectiveFunction((AspectRatio(eq), FixParameters(field))),
            ObjectiveFunction(
                (ForceBalance(eq), VacuumBoundaryError(eq, field, field_fixed=True))
            ),
            eq,
            solve_options={"solve_during_proximal_build": False},
        )
        prox.build(verbose=0)
    # constraint on both the boundary and the optimization variables
    prox = ProximalProjectionFreeBoundary(
        obj,
        ObjectiveFunction(
            (ForceBalance(eq), VacuumBoundaryError(eq, field, field_fixed=False))
        ),
        eq,
    )
    prox.build(verbose=0)
    with pytest.raises(ValueError, match="acts on both"):
        prox._separate_free_boundary_constraints(
            (FixParameters(eq, {"Rb_lmn": True, "Psi": True}),)
        )
    # constraints on only the boundary are taken, others are left
    fixR, fixpsi = FixBoundaryR(eq), FixPsi(eq)
    assert prox._separate_free_boundary_constraints((fixR, fixpsi)) == (fixpsi,)
    assert prox._fb_linear_constraints == (fixR,)


@pytest.mark.regression
@pytest.mark.slow
def test_free_boundary_single_stage_stellarator():
    """Single stage optimization of coil currents for a vacuum stellarator."""
    eq, field = _vacuum_stellarator(M=4)
    eq0 = eq.copy()
    V0 = eq.compute("V")["V"]
    objective = ObjectiveFunction(Volume(eq, target=1.1 * V0))
    constraints = (
        ForceBalance(eq),
        VacuumBoundaryError(eq, field, field_fixed=False),
        FixPsi(eq),
        FixPressure(eq),
        FixCurrent(eq),
    )
    # starts from a fixed boundary equilibrium, so the free boundary problem is
    # first solved when building
    with pytest.warns(UserWarning, match="not intended for scalar objective"):
        (eq, field), result = Optimizer("proximal-lsq-exact").optimize(
            [eq, field],
            objective,
            constraints,
            maxiter=20,
            ftol=1e-8,
            verbose=3,
            copy=False,
            options={"free_boundary_options": {"maxiter": 60}},
        )
    assert result["success"]
    np.testing.assert_allclose(eq.compute("V")["V"], 1.1 * V0, rtol=1e-4)
    # currents changed
    assert not np.allclose(field.currents, [4700.0, 1000.0])

    # solving the free boundary problem with the final currents the usual way
    # shouldn't change anything
    field_final = SplineMagneticField.from_mgrid(
        "tests/inputs/mgrid_test.nc", extcur=field.currents
    )
    eq_freeb = eq.copy()
    eq_freeb.optimize(
        ObjectiveFunction(VacuumBoundaryError(eq_freeb, field_final, field_fixed=True)),
        (
            ForceBalance(eq_freeb),
            FixPsi(eq_freeb),
            FixPressure(eq_freeb),
            FixCurrent(eq_freeb),
        ),
        optimizer="proximal-lsq-exact",
        maxiter=20,
        ftol=1e-8,
        verbose=0,
        copy=False,
    )
    rho_err, _ = area_difference_desc(eq, eq_freeb)
    np.testing.assert_allclose(rho_err[:, -1], 0, atol=2e-3)
    np.testing.assert_allclose(eq_freeb.compute("V")["V"], 1.1 * V0, rtol=1e-3)
    # and the boundary did change from the initial one
    rho_err, _ = area_difference_desc(eq, eq0)
    assert np.max(rho_err[:, -1]) > 1e-2
