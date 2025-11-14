"""Tests for linear constraints and objectives."""

import numpy as np
import pytest
from qsc import Qsc

import desc.examples
from desc.backend import jnp, put
from desc.coils import CoilSet, FourierXYZCoil
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.io import load
from desc.magnetic_fields import OmnigenousField
from desc.objectives import (
    AspectRatio,
    AxisRSelfConsistency,
    AxisZSelfConsistency,
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    FixAtomicNumber,
    FixAxisR,
    FixAxisZ,
    FixBoundaryR,
    FixBoundaryZ,
    FixCoilCurrent,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambdaGauge,
    FixModeLambda,
    FixModeR,
    FixModeZ,
    FixNearAxisLambda,
    FixNearAxisR,
    FixNearAxisZ,
    FixOmniBmax,
    FixOmniMap,
    FixOmniWell,
    FixParameters,
    FixPressure,
    FixPsi,
    FixSumCoilCurrent,
    FixSumModesLambda,
    FixSumModesR,
    FixSumModesZ,
    FixThetaSFL,
    ForceBalance,
    GenericObjective,
    LinearObjectiveFromUser,
    ObjectiveFunction,
    ShareParameters,
    get_equilibrium_objective,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import factorize_linear_constraints
from desc.optimize import LinearConstraintProjection
from desc.profiles import PowerSeriesProfile

# TODO (#1348): check for all bdryR things if work when False is passed in
# bc empty array indexed will lead to an error


@pytest.mark.unit
def test_LambdaGauge_sym(DummyStellarator):
    """Test that lambda is fixed correctly for symmetric equilibrium."""
    # symmetric cases automatically satisfy gauge freedom, no constraint needed.
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(L=2, M=1, N=1)
    lam_con = FixLambdaGauge(eq)
    lam_con.build()
    # should have no indices to fix
    assert lam_con._params["L_lmn"].size == 0


@pytest.mark.unit
def test_LambdaGauge_asym():
    """Test that lambda gauge is fixed correctly for asymmetric equilibrium."""
    # just testing the gauge condition
    inputs = {
        "sym": False,
        "NFP": 3,
        "Psi": 1.0,
        "L": 2,
        "M": 2,
        "N": 1,
        "pressure": np.array([[0, 1e4], [2, -2e4], [4, 1e4]]),
        "iota": np.array([[0, 0.5], [2, 0.5]]),
        "surface": np.array(
            [
                [0, 0, 0, 3, 0],
                [0, 1, 0, 1, 0],
                [0, -1, 0, 0, -1],
                [0, 1, 1, 0.3, 0],
                [0, -1, -1, 0.3, 0],
                [0, 1, -1, 0, -0.3],
                [0, -1, 1, 0, 0.3],
            ],
        ),
        "axis": np.array([[-1, 0, -0.2], [0, 3.4, 0], [1, 0.2, 0]]),
    }
    eq = Equilibrium(**inputs)
    lam_con = FixLambdaGauge(eq)
    lam_con.build()

    indices = np.where(
        np.logical_and(eq.L_basis.modes[:, 1] == 0, eq.L_basis.modes[:, 2] == 0)
    )[0]
    np.testing.assert_allclose(indices, lam_con._params["L_lmn"])


@pytest.mark.regression
@pytest.mark.solve
def test_bc_on_interior_surfaces():
    """Test applying boundary conditions on internal surface."""
    surf = FourierRZToroidalSurface(rho=0.5)
    iota = PowerSeriesProfile([1, 0, 0.5])
    eq = Equilibrium(L=4, M=4, N=0, surface=surf, iota=iota)
    eq.solve(verbose=0)
    surf5 = eq.get_surface_at(0.5)

    np.testing.assert_allclose(surf.R_lmn, surf5.R_lmn, atol=1e-12)
    np.testing.assert_allclose(surf.Z_lmn, surf5.Z_lmn, atol=1e-12)


@pytest.mark.unit
def test_constrain_bdry_with_only_one_mode():
    """Test Fixing boundary with a surface with only one mode in its basis."""
    eq = Equilibrium()
    FixZ = BoundaryZSelfConsistency(eq=eq)
    try:
        FixZ.build()
    except Exception:
        pytest.fail(
            "Error encountered when attempting to constrain surface with"
            + " only one mode in its basis"
        )


@pytest.mark.unit
def test_constrain_asserts():
    """Test error checking for incompatible constraints."""
    eqi = Equilibrium(iota=PowerSeriesProfile(0, 0), pressure=PowerSeriesProfile(0, 0))
    eqc = Equilibrium(current=PowerSeriesProfile([0, 0, 1]))
    obj_i = get_equilibrium_objective(eqi, "force")
    obj_c = get_equilibrium_objective(eqc, "force")
    obj_i.build()
    obj_c.build()
    # nonexistent toroidal current can't be constrained
    with pytest.raises(RuntimeError):
        con = FixCurrent(eq=eqi)
        con.build()
    # nonexistent rotational transform can't be constrained
    with pytest.raises(RuntimeError):
        con = FixIota(eq=eqc)
        con.build()
    # toroidal current and rotational transform can't be constrained simultaneously
    with pytest.raises(ValueError):
        con = (FixCurrent(eq=eqi), FixIota(eq=eqi))
        eqi.solve(constraints=con)
    # cannot use two incompatible constraints
    with pytest.raises(AssertionError):
        con1 = FixCurrent(target=eqc.c_l, eq=eqc)
        con2 = FixCurrent(target=eqc.c_l + 1, eq=eqc)
        con = ObjectiveFunction((con1, con2))
        con.build()
        _ = factorize_linear_constraints(obj_c, con)
    # if only slightly off, should raise only a warning
    with pytest.warns(UserWarning):
        con1 = FixCurrent(target=eqc.c_l, eq=eqc)
        con2 = FixCurrent(target=eqc.c_l * (1 + 1e-9), eq=eqc)
        con = ObjectiveFunction((con1, con2))
        con.build()
        _ = factorize_linear_constraints(obj_c, con)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_fixed_mode_solve():
    """Test solving an equilibrium with a fixed mode constraint."""
    # Reset DSHAPE to initial guess, fix a mode, and then resolve
    # and check that the mode stayed fix
    L = 1
    M = 1
    eq = desc.examples.get("DSHAPE")
    eq.set_initial_guess()
    fixR = FixModeR(
        eq=eq, modes=np.array([L, M, 0])
    )  # no target supplied, so defaults to the eq's current R_LMN coeff
    fixZ = FixModeZ(
        eq=eq, modes=np.array([L, -M, 0])
    )  # no target supplied, so defaults to the eq's current Z_LMN coeff
    orig_R_val = eq.R_lmn[eq.R_basis.get_idx(L=L, M=M, N=0)]
    orig_Z_val = eq.Z_lmn[eq.Z_basis.get_idx(L=L, M=-M, N=0)]

    constraints = (
        FixLambdaGauge(eq=eq),
        FixPressure(eq=eq),
        FixIota(eq=eq),
        FixPsi(eq=eq),
        FixBoundaryR(eq=eq),
        FixBoundaryR(
            eq=eq, modes=[0, 0, 0]
        ),  # add a degenerate constraint to test fix of GH #1297 for lsq-exact
        FixBoundaryZ(eq=eq),
        fixR,
        fixZ,
    )

    eq.solve(
        verbose=3,
        ftol=1e-2,
        objective="force",
        maxiter=10,
        xtol=1e-6,
        constraints=constraints,
    )
    np.testing.assert_almost_equal(
        orig_R_val, eq.R_lmn[eq.R_basis.get_idx(L=L, M=M, N=0)]
    )
    np.testing.assert_almost_equal(
        orig_Z_val, eq.Z_lmn[eq.Z_basis.get_idx(L=L, M=-M, N=0)]
    )


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_fixed_modes_solve():
    """Test solving an equilibrium with fixed sum modes constraint."""
    # Reset DSHAPE to initial guess, fix sum modes, and then resolve
    # and check that the mode sum stayed fix
    modes_R = np.array([[1, 1, 0], [2, 2, 0]])
    modes_Z = np.array([[1, -1, 0], [2, -2, 0]])

    eq = desc.examples.get("DSHAPE")
    eq.set_initial_guess()
    fixR = FixSumModesR(
        eq=eq, modes=modes_R, sum_weights=np.array([1, 2])
    )  # no target supplied, so defaults to the eq's current sum
    fixZ = FixSumModesZ(
        eq=eq, modes=modes_Z, sum_weights=np.array([1, 2])
    )  # no target supplied, so defaults to the eq's current sum
    fixLambda = FixSumModesLambda(
        eq=eq, modes=modes_Z, sum_weights=np.array([1, 2])
    )  # no target supplied, so defaults to the eq's current sum

    orig_R_val = (
        eq.R_lmn[eq.R_basis.get_idx(L=modes_R[0, 0], M=modes_R[0, 1], N=0)]
        + 2 * eq.R_lmn[eq.R_basis.get_idx(L=modes_R[1, 0], M=modes_R[1, 1], N=0)]
    )
    orig_Z_val = (
        eq.Z_lmn[eq.Z_basis.get_idx(L=modes_Z[0, 0], M=modes_Z[0, 1], N=0)]
        + 2 * eq.Z_lmn[eq.Z_basis.get_idx(L=modes_Z[1, 0], M=modes_Z[1, 1], N=0)]
    )
    orig_Lambda_val = (
        eq.L_lmn[eq.L_basis.get_idx(L=modes_Z[0, 0], M=modes_Z[0, 1], N=0)]
        + 2 * eq.L_lmn[eq.L_basis.get_idx(L=modes_Z[1, 0], M=modes_Z[1, 1], N=0)]
    )
    constraints = (
        FixLambdaGauge(eq=eq),
        FixPressure(eq=eq),
        FixIota(eq=eq),
        FixPsi(eq=eq),
        FixBoundaryR(eq=eq),
        FixBoundaryZ(eq=eq),
        fixR,
        fixZ,
        fixLambda,
    )

    eq.solve(
        verbose=3,
        ftol=1e-2,
        objective="force",
        maxiter=10,
        xtol=1e-6,
        constraints=constraints,
    )

    new_R_val = (
        eq.R_lmn[eq.R_basis.get_idx(L=modes_R[0, 0], M=modes_R[0, 1], N=0)]
        + 2 * eq.R_lmn[eq.R_basis.get_idx(L=modes_R[1, 0], M=modes_R[1, 1], N=0)]
    )
    new_Z_val = (
        eq.Z_lmn[eq.Z_basis.get_idx(L=modes_Z[0, 0], M=modes_Z[0, 1], N=0)]
        + 2 * eq.Z_lmn[eq.Z_basis.get_idx(L=modes_Z[1, 0], M=modes_Z[1, 1], N=0)]
    )
    new_Lambda_val = (
        eq.L_lmn[eq.L_basis.get_idx(L=modes_Z[0, 0], M=modes_Z[0, 1], N=0)]
        + 2 * eq.L_lmn[eq.L_basis.get_idx(L=modes_Z[1, 0], M=modes_Z[1, 1], N=0)]
    )

    np.testing.assert_almost_equal(orig_R_val, new_R_val)
    np.testing.assert_almost_equal(orig_Z_val, new_Z_val)
    np.testing.assert_almost_equal(orig_Lambda_val, new_Lambda_val)


@pytest.mark.regression
@pytest.mark.solve
@pytest.mark.slow
def test_fixed_axis_and_theta_SFL_solve():
    """Test solving an equilibrium with a fixed axis and theta SFL constraint."""
    # also tests zero lambda solve
    eq = Equilibrium()

    orig_R_val = eq.axis.R_n
    orig_Z_val = eq.axis.Z_n

    constraints = (FixThetaSFL(eq=eq),) + get_NAE_constraints(
        eq, None, order=0, fix_lambda=False
    )

    eq.solve(
        verbose=3,
        ftol=1e-2,
        objective="force",
        maxiter=10,
        xtol=1e-6,
        constraints=constraints,
    )

    np.testing.assert_allclose(orig_R_val, eq.axis.R_n, atol=1e-14)
    np.testing.assert_allclose(orig_Z_val, eq.axis.Z_n, atol=1e-14)
    np.testing.assert_allclose(np.zeros_like(eq.L_lmn), eq.L_lmn, atol=1e-14)


@pytest.mark.unit
def test_factorize_linear_constraints_asserts():
    """Test error checking for factorize_linear_constraints."""
    eq = Equilibrium()
    surf = eq.get_surface_at(rho=1)
    objective = get_equilibrium_objective(eq, "force")
    objective.build()

    # nonlinear constraint
    constraint = ObjectiveFunction(AspectRatio(eq=eq))
    constraint.build(verbose=0)
    with pytest.raises(ValueError):
        _ = factorize_linear_constraints(objective, constraint)

    # bounds instead of target
    constraint = ObjectiveFunction(get_fixed_boundary_constraints(eq=eq))
    constraint.build(verbose=0)
    constraint.objectives[3].bounds = (0, 1)
    with pytest.raises(ValueError):
        _ = factorize_linear_constraints(objective, constraint)

    # constraining a foreign thing
    constraint = ObjectiveFunction(FixParameters(surf))
    constraint.build(verbose=0)
    with pytest.raises(UserWarning):
        _ = factorize_linear_constraints(objective, constraint)


@pytest.mark.unit
def test_build_init():
    """Ensure that passing an equilibrium to init builds the objective correctly.

    Test for GH issue #378.
    """
    eq = Equilibrium(M=3, N=1)

    # initialize the constraints without building
    fbR1 = FixBoundaryR(eq=eq)
    fbZ1 = FixBoundaryZ(eq=eq)
    for obj in (fbR1, fbZ1):
        obj.build()

    xz = {key: np.zeros_like(val) for key, val in eq.params_dict.items()}
    arg = "Rb_lmn"
    A = fbR1.jac_scaled(xz)[0][arg]
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.R_basis.num_modes, eq.surface.R_basis.num_modes)

    arg = "Zb_lmn"
    A = fbZ1.jac_scaled(xz)[0][arg]
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.Z_basis.num_modes, eq.surface.Z_basis.num_modes)


@pytest.mark.unit
def test_kinetic_constraints():
    """Make sure errors are raised when trying to constrain nonexistent profiles."""
    eqp = Equilibrium(L=3, M=3, N=3, pressure=np.array([1, 0, -1]))
    eqk = Equilibrium(
        L=3,
        M=3,
        N=3,
        electron_temperature=np.array([1, 0, -1]),
        electron_density=np.array([2, 0, -2]),
    )
    pcon = (FixPressure(eq=eqk),)
    kcon = (
        FixAtomicNumber(eq=eqp),
        FixElectronDensity(eq=eqp),
        FixElectronTemperature(eq=eqp),
        FixIonTemperature(eq=eqp),
    )
    for con in pcon:
        with pytest.raises(RuntimeError):
            con.build()
    for con in kcon:
        with pytest.raises(RuntimeError):
            con.build()


@pytest.mark.unit
def test_correct_indexing_passed_modes():
    """Test indexing when passing in specified modes, related to gh issue #380."""
    n = 1
    eq = desc.examples.get("W7-X")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(3, 3, 3, 6, 6, 6)
    eq.surface = eq.get_surface_at(1.0)

    objective = ObjectiveFunction(
        (
            # just need dummy objective for factorizing constraints
            GenericObjective("0", thing=eq),
        ),
        use_jit=False,
    )
    objective.build()

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
        FixBoundaryR(eq=eq, modes=R_modes, normalize=False),
        FixBoundaryZ(eq=eq, modes=Z_modes, normalize=False),
        BoundaryRSelfConsistency(eq=eq),
        BoundaryZSelfConsistency(eq=eq),
        FixPressure(eq=eq),
    )
    constraint = ObjectiveFunction(constraints, use_jit=False)
    constraint.build()

    xp, A, b, Z, D, unfixed_idx, project, recover, *_ = factorize_linear_constraints(
        objective, constraint
    )

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    atol = 2e-15
    np.testing.assert_allclose(x1, x2, atol=atol)
    np.testing.assert_allclose(A @ xp[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ (x1[unfixed_idx] / D[unfixed_idx]), b, atol=atol)
    np.testing.assert_allclose(A @ (x2[unfixed_idx] / D[unfixed_idx]), b, atol=atol)
    np.testing.assert_allclose(A @ Z, 0, atol=atol)


@pytest.mark.unit
def test_correct_indexing_passed_modes_and_passed_target():
    """Test indexing when passing in specified modes, related to gh issue #380."""
    n = 1
    eq = desc.examples.get("W7-X")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(3, 3, 3, 6, 6, 6)
    eq.surface = eq.get_surface_at(1.0)

    objective = ObjectiveFunction(
        (GenericObjective("0", thing=eq),),
        use_jit=False,
    )
    objective.build()

    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > n + 1, :
            ],
        )
    )
    idxs = []
    for mode in R_modes:
        idxs.append(eq.surface.R_basis.get_idx(*mode))
    idxs = np.array(idxs)
    target_R = eq.surface.R_lmn[idxs]

    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > n + 1, :
    ]
    idxs = []
    for mode in Z_modes:
        idxs.append(eq.surface.Z_basis.get_idx(*mode))
    idxs = np.array(idxs)
    target_Z = eq.surface.Z_lmn[idxs]

    constraints = (
        FixBoundaryR(eq=eq, modes=R_modes, normalize=False, target=target_R),
        FixBoundaryZ(eq=eq, modes=Z_modes, normalize=False, target=target_Z),
        BoundaryRSelfConsistency(eq=eq),
        BoundaryZSelfConsistency(eq=eq),
        FixPressure(eq=eq),
    )
    constraint = ObjectiveFunction(constraints, use_jit=False)
    constraint.build()

    xp, A, b, Z, D, unfixed_idx, project, recover, *_ = factorize_linear_constraints(
        objective, constraint
    )

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    atol = 2e-15
    np.testing.assert_allclose(x1, x2, atol=atol)
    np.testing.assert_allclose(A @ xp[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ (x1[unfixed_idx] / D[unfixed_idx]), b, atol=atol)
    np.testing.assert_allclose(A @ (x2[unfixed_idx] / D[unfixed_idx]), b, atol=atol)
    np.testing.assert_allclose(A @ Z, 0, atol=atol)


@pytest.mark.unit
def test_correct_indexing_passed_modes_axis():
    """Test indexing when passing in specified axis modes, related to gh issue #380."""
    n = 1
    eq = desc.examples.get("W7-X")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(3, 3, 3, 6, 6, 6)
    eq.surface = eq.get_surface_at(1.0)
    eq.axis = eq.get_axis()

    objective = ObjectiveFunction(
        (GenericObjective("0", thing=eq),),
        use_jit=False,
    )
    objective.build()

    R_modes = np.vstack(
        (
            eq.axis.R_basis.modes[np.max(np.abs(eq.axis.R_basis.modes), 1) > n + 1, :],
            [0, 0, 0],
        )
    )
    R_modes = np.flip(R_modes, 0)
    Z_modes = eq.axis.Z_basis.modes[np.max(np.abs(eq.axis.Z_basis.modes), 1) > n + 1, :]
    Z_modes = np.flip(Z_modes, 0)
    constraints = (
        FixAxisR(eq=eq, modes=R_modes, normalize=False),
        FixAxisZ(eq=eq, modes=Z_modes, normalize=False),
        AxisRSelfConsistency(eq=eq),
        AxisZSelfConsistency(eq=eq),
        FixModeR(eq=eq, modes=np.array([[1, 1, 1], [2, 2, 2]]), normalize=False),
        FixModeZ(eq=eq, modes=np.array([[1, 1, -1], [2, 2, -2]]), normalize=False),
        FixModeLambda(eq=eq, modes=np.array([[1, 1, -1], [2, 2, -2]]), normalize=False),
        FixSumModesR(
            eq=eq,
            modes=np.array([[3, 3, 3], [4, 4, 4]]),
            normalize=False,
            sum_weights=np.ones(2),
        ),
        FixSumModesZ(eq=eq, modes=np.array([[3, 3, -3], [4, 4, -4]]), normalize=False),
        FixPressure(eq=eq),
    )
    constraint = ObjectiveFunction(constraints, use_jit=False)
    constraint.build()

    xp, A, b, Z, D, unfixed_idx, project, recover, *_ = factorize_linear_constraints(
        objective, constraint
    )

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    atol = 2e-15
    np.testing.assert_allclose(x1, x2, atol=atol)
    np.testing.assert_allclose(A @ xp[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ (x1[unfixed_idx] / D[unfixed_idx]), b, atol=atol)
    np.testing.assert_allclose(A @ (x2[unfixed_idx] / D[unfixed_idx]), b, atol=atol)
    np.testing.assert_allclose(A @ Z, 0, atol=atol)


@pytest.mark.unit
def test_correct_indexing_passed_modes_and_passed_target_axis():
    """Test indexing when passing in specified axis modes, related to gh issue #380."""
    n = 1

    eq = desc.examples.get("W7-X")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(4, 4, 4, 8, 8, 8)
    eq.surface = eq.get_surface_at(1.0)
    eq.axis = eq.get_axis()

    objective = ObjectiveFunction(
        (GenericObjective("0", thing=eq),),
        use_jit=False,
    )
    objective.build()

    R_modes = np.vstack(
        (
            eq.axis.R_basis.modes[np.max(np.abs(eq.axis.R_basis.modes), 1) > n + 1, :],
            [0, 0, 0],
        )
    )
    R_modes = np.flip(R_modes, 0)
    idxs = []
    for mode in R_modes:
        idxs.append(eq.axis.R_basis.get_idx(*mode))
    idxs = np.array(idxs)
    target_R = eq.axis.R_n[idxs]

    Z_modes = eq.axis.Z_basis.modes[np.max(np.abs(eq.axis.Z_basis.modes), 1) > n + 1, :]
    Z_modes = np.flip(Z_modes, 0)
    idxs = []
    for mode in Z_modes:
        idxs.append(eq.axis.Z_basis.get_idx(*mode))
    idxs = np.array(idxs)
    target_Z = eq.axis.Z_n[idxs]

    constraints = (
        AxisRSelfConsistency(eq=eq),
        AxisZSelfConsistency(eq=eq),
        FixAxisR(eq=eq, modes=R_modes, normalize=False, target=target_R),
        FixAxisZ(eq=eq, modes=Z_modes, normalize=False, target=target_Z),
        FixModeR(
            eq=eq,
            modes=np.array([[1, 1, 1], [2, 2, 2]]),
            target=np.array(
                [
                    eq.R_lmn[eq.R_basis.get_idx(*(1, 1, 1))],
                    eq.R_lmn[eq.R_basis.get_idx(*(2, 2, 2))],
                ]
            ),
            normalize=False,
        ),
        FixModeZ(
            eq=eq,
            modes=np.array([[1, 1, -1], [2, 2, -2]]),
            target=np.array(
                [
                    eq.Z_lmn[eq.Z_basis.get_idx(*(1, 1, -1))],
                    eq.Z_lmn[eq.Z_basis.get_idx(*(2, 2, -2))],
                ]
            ),
            normalize=False,
        ),
        FixModeLambda(
            eq=eq,
            modes=np.array([[1, 1, -1], [2, 2, -2]]),
            target=np.array(
                [
                    eq.L_lmn[eq.L_basis.get_idx(*(1, 1, -1))],
                    eq.L_lmn[eq.L_basis.get_idx(*(2, 2, -2))],
                ]
            ),
            normalize=False,
        ),
        FixSumModesR(
            eq=eq,
            modes=np.array([[3, 3, 3], [4, 4, 4]]),
            target=np.array(
                [
                    eq.R_lmn[eq.R_basis.get_idx(*(3, 3, 3))]
                    + eq.R_lmn[eq.R_basis.get_idx(*(4, 4, 4))]
                ]
            ),
            normalize=False,
        ),
        FixSumModesZ(
            eq=eq,
            modes=np.array([[3, 3, -3], [4, 4, -4]]),
            target=np.array(
                [
                    eq.Z_lmn[eq.Z_basis.get_idx(*(3, 3, -3))]
                    + eq.Z_lmn[eq.Z_basis.get_idx(*(4, 4, -4))]
                ]
            ),
            normalize=False,
        ),
        FixPressure(eq=eq),
        FixSumModesLambda(
            eq=eq,
            modes=np.array([[3, 3, -3], [4, 4, -4]]),
            target=np.array(
                [
                    eq.L_lmn[eq.L_basis.get_idx(*(3, 3, -3))]
                    + eq.L_lmn[eq.L_basis.get_idx(*(4, 4, -4))]
                ]
            ),
            normalize=False,
        ),
    )
    constraint = ObjectiveFunction(constraints, use_jit=False)
    constraint.build()

    xp, A, b, Z, D, unfixed_idx, project, recover, *_ = factorize_linear_constraints(
        objective, constraint
    )

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    atol = 2e-15
    np.testing.assert_allclose(x1, x2, atol=atol)
    np.testing.assert_allclose(A @ xp[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ (x1[unfixed_idx] / D[unfixed_idx]), b, atol=atol)
    np.testing.assert_allclose(A @ (x2[unfixed_idx] / D[unfixed_idx]), b, atol=atol)
    np.testing.assert_allclose(A @ Z, 0, atol=atol)


@pytest.mark.unit
def test_FixBoundary_with_single_weight():
    """Test Fixing boundary with only a single, passed weight."""
    eq = Equilibrium()
    w = 1.1
    FixZ = FixBoundaryZ(eq=eq, modes=np.array([[0, -1, 0]]), weight=w)
    FixZ.build()
    np.testing.assert_array_equal(FixZ.weight.size, 1)
    np.testing.assert_array_equal(FixZ.weight, w)
    FixR = FixBoundaryR(eq=eq, modes=np.array([[0, 1, 0]]), weight=w)
    FixR.build()
    np.testing.assert_array_equal(FixR.weight.size, 1)
    np.testing.assert_array_equal(FixR.weight, w)


@pytest.mark.unit
def test_FixBoundary_passed_target_no_passed_modes_error():
    """Test Fixing boundary with no passed-in modes."""
    eq = Equilibrium()
    FixZ = FixBoundaryZ(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixZ.build()
    FixZ = FixBoundaryZ(eq=eq, modes=False, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixZ.build()
    FixR = FixBoundaryR(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixR.build()
    FixR = FixBoundaryR(eq=eq, modes=False, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixR.build()


@pytest.mark.unit
def test_FixAxis_passed_target_no_passed_modes_error():
    """Test Fixing Axis with no passed-in modes."""
    eq = Equilibrium()
    FixR = FixAxisR(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixR.build()
    FixR = FixAxisR(eq=eq, modes=False, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixR.build()
    FixZ = FixAxisZ(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixZ.build()
    FixZ = FixAxisZ(eq=eq, modes=False, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixZ.build()


@pytest.mark.unit
def test_FixMode_passed_target_no_passed_modes_error():
    """Test Fixing Modes with no passed-in modes."""
    eq = Equilibrium()
    FixZ = FixModeZ(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixZ.build()
    FixR = FixModeR(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixR.build()
    FixL = FixModeLambda(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixL.build()


@pytest.mark.unit
def test_FixSumModes_passed_target_too_long():
    """Test Fixing Modes with more than a size-1 target."""
    # TODO (#1399): remove this test if FixSumModes is generalized
    # to accept multiple targets and sets of modes at a time
    eq = Equilibrium(L=3, M=4)
    with pytest.raises(ValueError):
        FixSumModesZ(
            eq, modes=np.array([[0, 0, 0], [1, 1, 1]]), target=np.array([[0, 1]])
        )
    with pytest.raises(ValueError):
        FixSumModesR(
            eq, modes=np.array([[0, 0, 0], [1, 1, 1]]), target=np.array([[0, 1]])
        )
    with pytest.raises(ValueError):
        FixSumModesLambda(
            eq, modes=np.array([[0, 0, 0], [1, 1, 1]]), target=np.array([[0, 1]])
        )


@pytest.mark.unit
def test_FixSumModes_False_or_None_modes():
    """Test Fixing Sum Modes without specifying modes or All modes."""
    eq = Equilibrium(L=3, M=4)
    with pytest.raises(ValueError):
        FixSumModesR(eq, modes=False)
    with pytest.raises(ValueError):
        FixSumModesR(eq, modes=None)
    with pytest.raises(ValueError):
        FixSumModesZ(eq, modes=False)
    with pytest.raises(ValueError):
        FixSumModesZ(eq, modes=None)
    with pytest.raises(ValueError):
        FixSumModesLambda(eq, modes=False)
    with pytest.raises(ValueError):
        FixSumModesLambda(eq, modes=None)


def _is_any_instance(things, cls):
    return any([isinstance(t, cls) for t in things])


@pytest.mark.unit
def test_FixAxis_util_correct_objectives():
    """Test util for fix axis constraints."""
    eq = Equilibrium()
    cs = get_fixed_axis_constraints(eq)
    assert _is_any_instance(cs, FixAxisR)
    assert _is_any_instance(cs, FixAxisZ)
    assert _is_any_instance(cs, FixPsi)
    assert _is_any_instance(cs, FixPressure)
    assert _is_any_instance(cs, FixCurrent)

    eq = Equilibrium(electron_temperature=1, electron_density=1, iota=1)
    cs = get_fixed_axis_constraints(eq)
    assert _is_any_instance(cs, FixAxisR)
    assert _is_any_instance(cs, FixAxisZ)
    assert _is_any_instance(cs, FixPsi)
    assert _is_any_instance(cs, FixElectronDensity)
    assert _is_any_instance(cs, FixElectronTemperature)
    assert _is_any_instance(cs, FixIonTemperature)
    assert _is_any_instance(cs, FixAtomicNumber)
    assert _is_any_instance(cs, FixIota)


@pytest.mark.unit
def test_FixNAE_util_correct_objectives():
    """Test util for fix NAE constraints."""
    eq = Equilibrium()
    qsc = Qsc.from_paper("precise QA")
    cs = get_NAE_constraints(eq, qsc)
    for c in cs:
        c.build()
    assert _is_any_instance(cs, FixPsi)
    assert _is_any_instance(cs, FixNearAxisR)
    assert _is_any_instance(cs, FixNearAxisZ)
    assert _is_any_instance(cs, FixPressure)
    assert _is_any_instance(cs, FixCurrent)

    cs = get_NAE_constraints(eq, qsc, fix_lambda=1)
    assert _is_any_instance(cs, FixPsi)
    assert _is_any_instance(cs, FixNearAxisR)
    assert _is_any_instance(cs, FixNearAxisZ)
    assert _is_any_instance(cs, FixNearAxisLambda)
    assert _is_any_instance(cs, FixPressure)
    assert _is_any_instance(cs, FixCurrent)

    eq = Equilibrium(electron_temperature=1, electron_density=1, iota=1)
    cs = get_NAE_constraints(eq, qsc)
    assert _is_any_instance(cs, FixPsi)
    assert _is_any_instance(cs, FixNearAxisR)
    assert _is_any_instance(cs, FixNearAxisZ)
    assert _is_any_instance(cs, FixElectronDensity)
    assert _is_any_instance(cs, FixElectronTemperature)
    assert _is_any_instance(cs, FixIonTemperature)
    assert _is_any_instance(cs, FixAtomicNumber)
    assert _is_any_instance(cs, FixIota)


@pytest.mark.unit
def test_fix_omni_indices():
    """Test that omnigenity parameters are constrained properly.

    Test for GH issue #768.
    """
    NFP = 3
    field = OmnigenousField(
        L_B=1, M_B=5, L_x=2, M_x=3, N_x=4, NFP=NFP, helicity=(1, NFP)
    )

    # no indices
    constraint = FixOmniWell(field=field, indices=False)
    constraint.build()
    assert constraint.dim_f == 0
    constraint = FixOmniMap(field=field, indices=False)
    constraint.build()
    assert constraint.dim_f == 0

    # all indices
    constraint = FixOmniWell(field=field, indices=True)
    constraint.build()
    assert constraint.dim_f == field.B_lm.size
    constraint = FixOmniMap(field=field, indices=True)
    constraint.build()
    assert constraint.dim_f == field.x_lmn.size

    # specified indices
    indices = np.arange(3, 8)
    constraint = FixOmniWell(field=field, indices=indices)
    constraint.build()
    assert constraint.dim_f == indices.size
    constraint = FixOmniMap(field=field, indices=indices)
    constraint.build()
    assert constraint.dim_f == indices.size


@pytest.mark.unit
def test_fix_omni_Bmax():
    """Test that omnigenity parameters are constrained for B_max to be a straight line.

    Test for GH issue #1266.
    """

    def _test(M_x, N_x, NFP, sum):
        field = OmnigenousField(L_x=2, M_x=M_x, N_x=N_x, NFP=NFP, helicity=(1, NFP))
        constraint = FixOmniBmax(field=field)
        constraint.build()
        assert constraint.dim_f == (2 * field.N_x + 1) * (field.L_x + 1)
        # 0 - 2 + 4 - 6 + 8 ...
        np.testing.assert_allclose(constraint._A @ field.x_basis.modes[:, 1], sum)

    _test(M_x=6, N_x=3, NFP=1, sum=-4)
    _test(M_x=9, N_x=4, NFP=2, sum=4)
    _test(M_x=12, N_x=5, NFP=3, sum=6)


@pytest.mark.unit
def test_fix_parameters_input_order(DummyStellarator):
    """Test that FixParameters preserves the input indices and target ordering."""
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")
    default_target = eq.Rb_lmn

    # default objective
    obj = FixBoundaryR(eq)
    obj.build()
    np.testing.assert_allclose(obj.target, default_target)

    # manually specify default
    obj = FixBoundaryR(eq, modes=eq.surface.R_basis.modes)
    obj.build()
    np.testing.assert_allclose(obj.target, default_target)

    # reverse order
    obj = FixBoundaryR(eq, modes=np.flipud(eq.surface.R_basis.modes))
    obj.build()
    np.testing.assert_allclose(obj.target, np.flipud(default_target))

    # custom order
    obj = ObjectiveFunction(
        FixBoundaryR(eq, modes=np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]))
    )
    obj.build()
    np.testing.assert_allclose(obj.target_scaled, np.array([3, 1, 0.3]))
    np.testing.assert_allclose(obj.compute_scaled_error(obj.x(eq)), np.zeros(obj.dim_f))

    # custom target
    obj = ObjectiveFunction(
        FixBoundaryR(
            eq,
            modes=np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]),
            target=np.array([0, -1, 0.5]),
        )
    )
    obj.build()
    np.testing.assert_allclose(
        obj.compute_scaled_error(obj.x(eq)), np.array([3, 2, -0.2])
    )


@pytest.mark.unit
def test_fix_subset_of_params_in_collection(DummyMixedCoilSet):
    """Tests FixParameters fixing a subset of things in the collection."""
    coilset = load(load_from=str(DummyMixedCoilSet["output_path"]), file_format="hdf5")

    params = [
        [
            {"current": True},
        ],
        {"shift": True, "rotmat": True},
        {"X_n": np.array([1, 2]), "Y_n": False, "Z_n": np.array([0])},
        {},
    ]
    target = np.concatenate(
        (
            np.array([3]),
            np.eye(3).flatten(),
            np.array([0, 0, 0]),
            np.eye(3).flatten(),
            np.array([0, 0, 1]),
            np.eye(3).flatten(),
            np.array([0, 0, 2]),
            np.array([10, 2, -2]),
        )
    )

    obj = FixParameters(coilset, params)
    obj.build()
    np.testing.assert_allclose(obj.target, target)


@pytest.mark.unit
def test_fix_coil_current(DummyMixedCoilSet):
    """Tests FixCoilCurrent."""
    coilset = load(load_from=str(DummyMixedCoilSet["output_path"]), file_format="hdf5")

    # fix a single coil current
    obj = FixCoilCurrent(coil=coilset.coils[1].coils[0])
    obj.build()
    assert obj.dim_f == 1
    np.testing.assert_allclose(obj.target, -1)

    # fix all coil currents
    obj = FixCoilCurrent(coil=coilset)
    obj.build()
    assert obj.dim_f == 6
    np.testing.assert_allclose(obj.target, [3, -1, -1, -1, 2, 1])

    # only fix currents of some coils in the coil set
    obj = FixCoilCurrent(
        coil=coilset, indices=[[True], [True, False, True], True, False]
    )
    obj.build()
    assert obj.dim_f == 4
    np.testing.assert_allclose(obj.target, [3, -1, -1, 2])


@pytest.mark.unit
def test_fix_sum_coil_current(DummyMixedCoilSet):
    """Tests FixSumCoilCurrent."""
    coilset = load(load_from=str(DummyMixedCoilSet["output_path"]), file_format="hdf5")

    # sum a single coil current
    obj = FixSumCoilCurrent(coil=coilset.coils[1].coils[0])
    obj.build()
    params = coilset.coils[1].coils[0].params_dict
    np.testing.assert_allclose(obj.compute(params), -1)

    # sum all coil currents
    obj = FixSumCoilCurrent(coil=coilset)
    obj.build()
    params = coilset.params_dict
    np.testing.assert_allclose(obj.compute(params), 3)
    # the default target should be the original sum
    np.testing.assert_allclose(obj.compute_scaled_error(params), 0)

    # only sum currents of some coils in the coil set
    obj = FixSumCoilCurrent(
        coil=coilset, indices=[[True], [True, False, True], True, False]
    )
    obj.build()
    np.testing.assert_allclose(obj.compute(params), 3)


@pytest.mark.unit
def test_linear_objective_from_user_on_collection(DummyCoilSet):
    """Test LinearObjectiveFromUser on an OptimizableCollection."""
    # test that LinearObjectiveFromUser can be used for the same functionality as
    # FixSumCoilCurrent to sum all currents in a CoilSet

    coilset = load(load_from=str(DummyCoilSet["output_path_asym"]), file_format="hdf5")
    params = coilset.params_dict

    obj1 = FixSumCoilCurrent(coil=coilset)
    obj1.build()

    obj2 = LinearObjectiveFromUser(
        lambda params: jnp.sum(jnp.array([param["current"] for param in params])),
        coilset,
    )
    obj2.build()

    np.testing.assert_allclose(obj1.compute(params), obj2.compute(params))


@pytest.mark.unit
def test_share_parameters_four_objects():
    """Tests ShareParameters with 4 objects."""
    eq1 = desc.examples.get("SOLOVEV")
    eq2 = eq1.copy()
    eq3 = eq1.copy()
    eq4 = eq1.copy()

    subobj = ShareParameters([eq1, eq2, eq3, eq4], {"p_l": True, "i_l": [1, 2]})
    subobj.build()
    obj = ObjectiveFunction(subobj)
    obj.build()

    # check dimensions
    # (len(things)-1) x p_l.size + (len(things)-1) x 2 for the 2 i_l indices
    assert subobj.dim_f == 3 * eq1.params_dict["p_l"].size + 3 * 2
    np.testing.assert_allclose(subobj.target, 0)

    # check compute
    np.testing.assert_allclose(obj.compute_unscaled(obj.x(eq1, eq2, eq3, eq4)), 0)

    # check the jacobian
    J = obj.jac_unscaled(obj.x(eq1, eq2, eq3, eq4))
    assert J.shape[0] == subobj.dim_f
    # make sure Jacobian is not trivial
    assert not np.allclose(J, 0)
    # now, check that each row sums to zero, and abs(J) rows sum to 2,
    # meaning each row has only 2 nonzero elements which are 1 and -1,
    J_row_sums = J.sum(axis=1)
    abs_J_row_sums = np.abs(J).sum(axis=1)
    np.testing.assert_allclose(abs_J_row_sums, 2)
    np.testing.assert_allclose(J_row_sums, 0)


@pytest.mark.unit
def test_share_parameters_two_optimizable_collections_CoilSet():
    """Tests ShareParameters with 2 CoilSets."""
    coils1 = CoilSet.linspaced_angular(FourierXYZCoil(), n=2)
    coils2 = coils1.copy()

    subobj = ShareParameters([coils1, coils2], {"X_n": True, "Y_n": True, "Z_n": [2]})
    subobj.build()
    obj = ObjectiveFunction(subobj)
    obj.build()

    # check dimensions
    # dim_f should be 2 (for the 2 subcoils in each coilset) x 2 (for X_n, Y_n)
    # x params["X_n"].size + 2 x 1 (Z_n) x 1 (bc only fixed idx=2)
    assert subobj.dim_f == 2 * 2 * coils1.params_dict[0]["X_n"].size + 2
    np.testing.assert_allclose(subobj.target, 0)

    # check compute
    np.testing.assert_allclose(obj.compute_unscaled(obj.x(coils1, coils2)), 0)

    # check the jacobian
    J = obj.jac_unscaled(obj.x(coils1, coils2))
    assert J.shape[0] == subobj.dim_f

    # make sure Jacobian is not trivial
    assert not np.allclose(J, 0)
    # now, check that each row sums to zero, and abs(J) rows sum to 2,
    # meaning each row has only 2 nonzero elements which are 1 and -1,
    J_row_sums = J.sum(axis=1)
    abs_J_row_sums = np.abs(J).sum(axis=1)
    np.testing.assert_allclose(abs_J_row_sums, 2)
    np.testing.assert_allclose(J_row_sums, 0)


@pytest.mark.unit
def test_linearconstraintprojection_update_target():
    """Test that LinearConstraintProjection updates the target properly."""
    eq = desc.examples.get("W7-X")
    obj = ObjectiveFunction(ForceBalance(eq))
    cons = get_fixed_boundary_constraints(eq)
    cons = maybe_add_self_consistency(eq, cons)
    con = ObjectiveFunction(cons)
    lc = LinearConstraintProjection(objective=obj, constraint=con)
    lc.build()

    eqp = eq.copy()
    # slighlty perturb the equilibrium
    eqp.Rb_lmn = put(eqp.Rb_lmn, 0, eqp.Rb_lmn[0] + 1e-3)
    eqp.Rb_lmn = put(eqp.Rb_lmn, 1, eqp.Rb_lmn[1] + 1e-3)
    eqp.Zb_lmn = put(eqp.Zb_lmn, 0, eqp.Zb_lmn[0] + 1e-3)
    eqp.p_l = put(eqp.p_l, 0, eqp.p_l[0] + 1e-3)

    # fresh constraints for perturbed equilibrium
    objp = ObjectiveFunction(ForceBalance(eqp))
    consp = get_fixed_boundary_constraints(eqp)
    consp = maybe_add_self_consistency(eqp, consp)
    conp = ObjectiveFunction(consp)
    lcp = LinearConstraintProjection(objective=objp, constraint=conp)
    lcp.build()

    # update the target without creating a new constraints as above
    lc.update_constraint_target(eqp)

    # perturb method is using this equivalently
    obj0 = ObjectiveFunction(ForceBalance(eq))
    cons0 = get_fixed_boundary_constraints(eq)
    cons0 = maybe_add_self_consistency(eq, cons0)
    con0 = ObjectiveFunction(cons0)
    con0.build()
    obj0.build()
    for con in con0.objectives:
        if hasattr(con, "update_target"):
            con.update_target(eqp)
    constraint = ObjectiveFunction(con0.objectives)
    constraint.build()
    (
        xp,
        A,
        b,
        Z,
        D,
        unfixed_idx,
        project,
        recover,
        ADinv,
        A_nondegenerate,
        degenerate_idx,
    ) = factorize_linear_constraints(obj0, constraint)

    # check that the target is updated properly
    np.testing.assert_allclose(lc._xp, lcp._xp)
    np.testing.assert_allclose(xp, lcp._xp)
    np.testing.assert_allclose(lc._ADinv, lcp._ADinv)
    np.testing.assert_allclose(ADinv, lcp._ADinv)
    np.testing.assert_allclose(lc._Z, lcp._Z)
    np.testing.assert_allclose(Z, lcp._Z)
    np.testing.assert_allclose(lc._D, lcp._D)
    np.testing.assert_allclose(D, lcp._D)
    np.testing.assert_allclose(lc._ZA, lcp._ZA)
    np.testing.assert_allclose(lc._Ainv, lcp._Ainv)
    np.testing.assert_allclose(lc._feasible_tangents, lcp._feasible_tangents)


@pytest.mark.unit
def test_NAE_asym_with_sym_axis():
    """Test that asym NAE constraints are correct when axis is sym."""
    qsc_eq = Qsc.from_paper("r2 section 5.5")
    # this NAE solution has a symmetric axis but an asymmetric B variation
    eq = Equilibrium(NFP=qsc_eq.nfp, N=4, sym=False)
    conR = FixNearAxisR(eq, target=qsc_eq)
    conZ = FixNearAxisZ(eq, target=qsc_eq)
    conR.build()
    conZ.build()
    assert conR._A.shape[0] == conR.dim_f
    assert conZ._A.shape[0] == conZ.dim_f
