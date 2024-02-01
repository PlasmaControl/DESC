"""Tests for linear constraints and objectives."""
import numpy as np
import pytest
import scipy.linalg
from qsc import Qsc

import desc.examples
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.io import load
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
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambdaGauge,
    FixModeLambda,
    FixModeR,
    FixModeZ,
    FixPressure,
    FixPsi,
    FixSumModesLambda,
    FixSumModesR,
    FixSumModesZ,
    FixThetaSFL,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    get_equilibrium_objective,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
)
from desc.objectives.utils import factorize_linear_constraints
from desc.profiles import PowerSeriesProfile

# TODO: check for all bdryR things if work when False is passed in
# bc empty array indexed will lead to an error


@pytest.mark.unit
def test_LambdaGauge_sym(DummyStellarator):
    """Test that lambda is fixed correctly for symmetric equilibrium."""
    # symmetric cases automatically satisfy gauge freedom, no constraint needed.
    eq = load(load_from=str(DummyStellarator["output_path"]), file_format="hdf5")
    eq.change_resolution(L=2, M=1, N=1)
    correct_constraint_matrix = np.zeros((0, 5))
    lam_con = FixLambdaGauge(eq)
    lam_con.build()
    np.testing.assert_array_equal(lam_con._A, correct_constraint_matrix)


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
        "objective": "force",
        "optimizer": "lsq-exact",
    }
    eq = Equilibrium(**inputs)
    lam_con = FixLambdaGauge(eq)
    lam_con.build()

    # make sure that any lambda in the null space gives lambda==0 at theta=zeta=0
    Z = scipy.linalg.null_space(lam_con._A)
    grid = LinearGrid(L=10, theta=[0], zeta=[0])
    for z in Z.T:
        eq.L_lmn = z
        lam = eq.compute("lambda", grid=grid)["lambda"]
        np.testing.assert_allclose(lam, 0, atol=1e-15)


@pytest.mark.unit
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
    eqc = Equilibrium(current=PowerSeriesProfile(0))
    # nonexistent toroidal current can't be constrained
    with pytest.raises(RuntimeError):
        eqi.solve(constraints=FixCurrent(eq=eqi))
    # nonexistent rotational transform can't be constrained
    with pytest.raises(RuntimeError):
        eqc.solve(constraints=FixIota(eq=eqc))
    # toroidal current and rotational transform can't be constrained simultaneously
    with pytest.raises(ValueError):
        eqi.solve(constraints=(FixCurrent(eq=eqi), FixIota(eq=eqi)))
    with pytest.raises(AssertionError):
        eqi.solve(
            constraints=(FixPressure(eq=eqi, target=2), FixPressure(eq=eqi, target=1))
        )
    # cannot use two incompatible constraints
    with pytest.raises(AssertionError):
        con1 = FixCurrent(target=eqc.c_l, eq=eqc)
        con2 = FixCurrent(target=eqc.c_l + 1, eq=eqc)
        eqc.solve(constraints=(con1, con2))


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

    constraints = (
        FixThetaSFL(eq=eq),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
        FixAxisR(eq=eq),
        FixAxisZ(eq=eq),
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
    constraints = get_fixed_boundary_constraints(eq=eq)
    for con in constraints:
        con.build(verbose=0)
    constraints[3].bounds = (0, 1)  # bounds on FixPsi

    objective = get_equilibrium_objective(eq, "force")
    objective.build()
    with pytest.raises(ValueError):
        xp, A, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
            constraints, objective
        )


@pytest.mark.unit
def test_build_init():
    """Ensure that passing an equilibrium to init builds the objective correctly.

    Related to gh issue #378
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
    """Test Indexing when passing in specified modes, related to gh issue #380."""
    n = 1

    eq = desc.examples.get("W7-X")

    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(eq=eq, weight=1e-2, helicity=(1, -eq.NFP), grid=grid),
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
        FixBoundaryR(eq=eq, modes=R_modes, normalize=False),
        FixBoundaryZ(eq=eq, modes=Z_modes, normalize=False),
        BoundaryRSelfConsistency(eq=eq),
        BoundaryZSelfConsistency(eq=eq),
        FixPressure(eq=eq),
    )
    for con in constraints:
        con.build(verbose=0)
    objective.build()

    xp, A, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints,
        objective,
    )

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    atol = 2e-15
    np.testing.assert_allclose(x1, x2, atol=atol)
    np.testing.assert_allclose(A @ xp[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ x1[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ x2[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ Z, 0, atol=atol)


@pytest.mark.unit
def test_correct_indexing_passed_modes_and_passed_target():
    """Test Indexing when passing in specified modes, related to gh issue #380."""
    n = 1

    eq = desc.examples.get("W7-X")

    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(eq=eq, weight=1e-2, helicity=(1, -eq.NFP), grid=grid),
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
    idxs = []
    for mode in R_modes:
        idxs.append(eq.surface.R_basis.get_idx(*mode))
    target_R = eq.surface.R_lmn[idxs]

    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > n + 1, :
    ]
    idxs = []
    for mode in Z_modes:
        idxs.append(eq.surface.Z_basis.get_idx(*mode))
    target_Z = eq.surface.Z_lmn[idxs]
    constraints = (
        FixBoundaryR(eq=eq, modes=R_modes, normalize=False, target=target_R),
        FixBoundaryZ(eq=eq, modes=Z_modes, normalize=False, target=target_Z),
        BoundaryRSelfConsistency(eq=eq),
        BoundaryZSelfConsistency(eq=eq),
        FixPressure(eq=eq),
    )
    for con in constraints:
        con.build(eq, verbose=0)
    objective.build(eq)

    xp, A, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints,
        objective,
    )

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    atol = 2e-15
    np.testing.assert_allclose(x1, x2, atol=atol)
    np.testing.assert_allclose(A @ xp[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ x1[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ x2[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ Z, 0, atol=atol)


@pytest.mark.unit
def test_correct_indexing_passed_modes_axis():
    """Test Indexing when passing in specified axis modes, related to gh issue #380."""
    n = 1

    eq = desc.examples.get("W7-X")

    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(eq=eq, weight=1e-2, helicity=(1, -eq.NFP), grid=grid),
            AspectRatio(eq=eq, target=8, weight=1e2),
        ),
        verbose=0,
    )
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
    for con in constraints:
        con.build(verbose=0)
    objective.build()

    xp, A, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints,
        objective,
    )

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    atol = 2e-15
    np.testing.assert_allclose(x1, x2, atol=atol)
    np.testing.assert_allclose(A @ xp[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ x1[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ x2[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ Z, 0, atol=atol)


@pytest.mark.unit
def test_correct_indexing_passed_modes_and_passed_target_axis():
    """Test Indexing when passing in specified axis modes, related to gh issue #380."""
    n = 1

    eq = desc.examples.get("W7-X")

    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(eq=eq, weight=1e-2, helicity=(1, -eq.NFP), grid=grid),
            AspectRatio(eq=eq, target=8, weight=1e2),
        ),
        verbose=0,
    )
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
    target_R = eq.axis.R_n[idxs]

    Z_modes = eq.axis.Z_basis.modes[np.max(np.abs(eq.axis.Z_basis.modes), 1) > n + 1, :]
    Z_modes = np.flip(Z_modes, 0)
    idxs = []
    for mode in Z_modes:
        idxs.append(eq.axis.Z_basis.get_idx(*mode))
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
    for con in constraints:
        con.build(verbose=0)
    objective.build()

    xp, A, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints,
        objective,
    )

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    atol = 2e-15
    np.testing.assert_allclose(x1, x2, atol=atol)
    np.testing.assert_allclose(A @ xp[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ x1[unfixed_idx], b, atol=atol)
    np.testing.assert_allclose(A @ x2[unfixed_idx], b, atol=atol)
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
    FixZ = FixAxisZ(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixZ.build()
    FixZ = FixAxisZ(eq=eq, modes=False, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixZ.build()
    FixR = FixAxisR(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixR.build()
    FixR = FixAxisR(eq=eq, modes=False, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixR.build()


@pytest.mark.unit
def test_FixMode_passed_target_no_passed_modes_error():
    """Test Fixing Modes with no passed-in modes."""
    eq = Equilibrium()
    FixZ = FixModeZ(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixZ.build()
    FixR = FixModeR(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixR.build(eq)
    FixL = FixModeLambda(eq=eq, modes=True, target=np.array([0, 0]))
    with pytest.raises(ValueError):
        FixL.build(eq)


@pytest.mark.unit
def test_FixSumModes_passed_target_too_long():
    """Test Fixing Modes with more than a size-1 target."""
    # TODO: remove this test if FixSumModes is generalized
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
def test_FixMode_False_or_None_modes():
    """Test Fixing Modes without specifying modes or All modes."""
    eq = Equilibrium(L=3, M=4)
    with pytest.raises(ValueError):
        FixModeR(eq, modes=False, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixModeR(eq, modes=None, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixModeZ(eq, modes=False, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixModeZ(eq, modes=None, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixModeLambda(eq, modes=False, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixModeLambda(eq, modes=None, target=np.array([[0, 1]]))


@pytest.mark.unit
def test_FixSumModes_False_or_None_modes():
    """Test Fixing Sum Modes without specifying modes or All modes."""
    eq = Equilibrium(L=3, M=4)
    with pytest.raises(ValueError):
        FixSumModesZ(eq, modes=False, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixSumModesZ(eq, modes=None, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixSumModesR(eq, modes=False, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixSumModesR(eq, modes=None, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixSumModesLambda(eq, modes=False, target=np.array([[0, 1]]))
    with pytest.raises(ValueError):
        FixSumModesLambda(eq, modes=None, target=np.array([[0, 1]]))


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
    assert _is_any_instance(cs, FixAxisR)
    assert _is_any_instance(cs, FixAxisZ)
    assert _is_any_instance(cs, FixPsi)
    assert _is_any_instance(cs, FixSumModesR)
    assert _is_any_instance(cs, FixSumModesZ)
    assert _is_any_instance(cs, FixPressure)
    assert _is_any_instance(cs, FixCurrent)

    eq = Equilibrium(electron_temperature=1, electron_density=1, iota=1)
    cs = get_NAE_constraints(eq, qsc)
    assert _is_any_instance(cs, FixAxisR)
    assert _is_any_instance(cs, FixAxisZ)
    assert _is_any_instance(cs, FixPsi)
    assert _is_any_instance(cs, FixSumModesR)
    assert _is_any_instance(cs, FixSumModesZ)
    assert _is_any_instance(cs, FixElectronDensity)
    assert _is_any_instance(cs, FixElectronTemperature)
    assert _is_any_instance(cs, FixIonTemperature)
    assert _is_any_instance(cs, FixAtomicNumber)
    assert _is_any_instance(cs, FixIota)
