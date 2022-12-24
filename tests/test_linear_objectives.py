"""Tests for linear constraints and objectives."""
import jax.numpy as jnp
import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.io import load
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixLambdaGauge,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
)
from desc.profiles import PowerSeriesProfile


@pytest.mark.unit
def test_LambdaGauge_axis_sym(DummyStellarator):
    """Test that lambda at axis is fixed correctly for symmetric equilibrium."""
    # symmetric cases only have the axis constraint
    eq = Equilibrium.load(
        load_from=str(DummyStellarator["output_path"]), file_format="hdf5"
    )
    eq.change_resolution(L=2, M=1, N=1)
    correct_constraint_matrix = np.zeros((1, 5))
    correct_constraint_matrix[0, 0] = 1
    correct_constraint_matrix[0, 2] = -1

    lam_con = FixLambdaGauge(eq)

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
        "M": 0,
        "N": 1,
        "pressure": np.array([[0, 1e4], [2, -2e4], [4, 1e4]]),
        "iota": np.array([[0, 0.5], [2, 0.5]]),
        "surface": np.array(
            [
                [0, 0, 0, 3, 0],
                [0, 1, 0, 1, 0],
                [0, -1, 0, 0, 1],
                [0, 1, 1, 0.3, 0],
                [0, -1, -1, -0.3, 0],
                [0, 1, -1, 0, -0.3],
                [0, -1, 1, 0, -0.3],
            ],
        ),
        "axis": np.array([[-1, 0, -0.2], [0, 3.4, 0], [1, 0.2, 0]]),
        "objective": "force",
        "optimizer": "lsq-exact",
    }
    eq = Equilibrium(**inputs)
    correct_constraint_matrix = np.zeros((3, eq.L_basis.num_modes))
    correct_constraint_matrix[0, 0] = 1.0
    correct_constraint_matrix[0, 1] = -1.0
    correct_constraint_matrix[2, 4] = 1.0
    correct_constraint_matrix[2, 5] = -1.0
    correct_constraint_matrix[1, 2] = 1.0
    correct_constraint_matrix[1, 3] = -1.0

    lam_con = FixLambdaGauge(eq)

    np.testing.assert_array_equal(
        lam_con._A[0:3, 0 : eq.L_basis.num_modes], correct_constraint_matrix
    )


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
    FixZ = FixBoundaryZ(fixed_boundary=True)
    try:
        FixZ.build(eq)
    except Exception:
        pytest.fail(
            "Error encountered when attempting to constrain surface with"
            + " only one mode in its basis"
        )


@pytest.mark.unit
def test_constrain_asserts():
    """Test error checking for incompatible constraints."""
    # nonexistent toroidal current can't be constrained
    eq = Equilibrium(iota=PowerSeriesProfile(0, 0))
    with pytest.raises(RuntimeError):
        eq.solve(constraints=FixCurrent())
    # nonexistent rotational transform can't be constrained
    eq = Equilibrium(current=PowerSeriesProfile(0))
    with pytest.raises(RuntimeError):
        eq.solve(constraints=FixIota())
    # toroidal current and rotational transform can't be constrained simultaneously
    eq = Equilibrium(current=PowerSeriesProfile(0))
    with pytest.raises(ValueError):
        eq.solve(constraints=(FixCurrent(), FixIota()))


@pytest.mark.unit
def test_build_init():
    """Ensure that passing an equilibrium to init builds the objective correctly.

    Related to gh issue #378
    """
    eq = Equilibrium(M=3, N=1)

    # initialize the constraints without building
    fbR1 = FixBoundaryR(fixed_boundary=False)
    fbZ1 = FixBoundaryZ(fixed_boundary=False)
    fbR2 = FixBoundaryR(fixed_boundary=True)
    fbZ2 = FixBoundaryZ(fixed_boundary=True)
    for obj in (fbR1, fbR2, fbZ1, fbZ2):
        obj.build(eq)

    arg = fbR1.args[0]
    A = fbR1.derivatives["jac"][arg](np.zeros(fbR1.dimensions[arg]))
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.R_basis.num_modes, eq.surface.R_basis.num_modes)

    arg = fbR2.args[0]
    A = fbR2.derivatives["jac"][arg](np.zeros(fbR2.dimensions[arg]))
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.R_basis.num_modes, eq.R_basis.num_modes)

    arg = fbZ1.args[0]
    A = fbZ1.derivatives["jac"][arg](np.zeros(fbZ1.dimensions[arg]))
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.Z_basis.num_modes, eq.surface.Z_basis.num_modes)

    arg = fbZ2.args[0]
    A = fbZ2.derivatives["jac"][arg](np.zeros(fbZ2.dimensions[arg]))
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.Z_basis.num_modes, eq.Z_basis.num_modes)

    fbR1 = FixBoundaryR(fixed_boundary=False, eq=eq)
    fbZ1 = FixBoundaryZ(fixed_boundary=False, eq=eq)
    fbR2 = FixBoundaryR(fixed_boundary=True, eq=eq)
    fbZ2 = FixBoundaryZ(fixed_boundary=True, eq=eq)

    arg = fbR1.args[0]
    A = fbR1.derivatives["jac"][arg](np.zeros(fbR1.dimensions[arg]))
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.R_basis.num_modes, eq.surface.R_basis.num_modes)

    arg = fbR2.args[0]
    A = fbR2.derivatives["jac"][arg](np.zeros(fbR2.dimensions[arg]))
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.R_basis.num_modes, eq.R_basis.num_modes)

    arg = fbZ1.args[0]
    A = fbZ1.derivatives["jac"][arg](np.zeros(fbZ1.dimensions[arg]))
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.Z_basis.num_modes, eq.surface.Z_basis.num_modes)

    arg = fbZ2.args[0]
    A = fbZ2.derivatives["jac"][arg](np.zeros(fbZ2.dimensions[arg]))
    assert np.max(np.abs(A)) == 1
    assert A.shape == (eq.surface.Z_basis.num_modes, eq.Z_basis.num_modes)


@pytest.mark.unit
def test_correct_indexing_passed_modes():
    """Test Indexing when passing in specified modes, related to gh issue #380."""
    n = 1

    eq = load(".//tests//inputs//precise_QH_step0.h5")[0]

    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(weight=1e-2, helicity=(1, -eq.NFP), grid=grid),
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
        FixBoundaryR(modes=R_modes, fixed_boundary=True, normalize=False),
        FixBoundaryZ(modes=Z_modes, fixed_boundary=True, normalize=False),
    )
    for con in constraints:
        con.build(eq, verbose=0)
    objective.build(eq)
    from desc.objectives.utils import factorize_linear_constraints

    xp, A, Ainv, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints,
        objective.args,
    )

    from scipy.linalg import block_diag

    from desc.compute import arg_order

    A_full = block_diag(*[A[arg] for arg in arg_order if arg in A.keys()])
    b_full = jnp.concatenate([b[arg] for arg in arg_order if arg in b.keys()])

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    assert np.isclose(np.max(np.abs(x1 - x2)), 0, atol=1e-15)
    assert np.isclose(np.max(np.abs(A_full @ xp - b_full)), 0, atol=1e-15)
    assert np.isclose(np.max(np.abs(A_full @ x1 - b_full)), 0, atol=1e-15)
    assert np.isclose(np.max(np.abs(A_full @ x2 - b_full)), 0, atol=1e-15)
    assert np.isclose(np.max(np.abs(A_full @ Z)), 0, atol=1e-15)


@pytest.mark.unit
def test_correct_indexing_passed_modes_and_passed_target():
    """Test Indexing when passing in specified modes, related to gh issue #380."""
    n = 1

    eq = load(".//tests//inputs//precise_QH_step0.h5")[0]

    grid = LinearGrid(
        M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(weight=1e-2, helicity=(1, -eq.NFP), grid=grid),
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
        FixBoundaryR(
            modes=R_modes, fixed_boundary=True, normalize=False, target=target_R
        ),
        FixBoundaryZ(
            modes=Z_modes, fixed_boundary=True, normalize=False, target=target_Z
        ),
    )
    for con in constraints:
        con.build(eq, verbose=0)
    objective.build(eq)
    from desc.objectives.utils import factorize_linear_constraints

    xp, A, Ainv, b, Z, unfixed_idx, project, recover = factorize_linear_constraints(
        constraints,
        objective.args,
    )

    from scipy.linalg import block_diag

    from desc.compute import arg_order

    A_full = block_diag(*[A[arg] for arg in arg_order if arg in A.keys()])
    b_full = jnp.concatenate([b[arg] for arg in arg_order if arg in b.keys()])

    x1 = objective.x(eq)
    x2 = recover(project(x1))

    assert np.isclose(np.max(np.abs(x1 - x2)), 0, atol=1e-15)
    assert np.isclose(np.max(np.abs(A_full @ xp - b_full)), 0, atol=1e-15)
    assert np.isclose(np.max(np.abs(A_full @ x1 - b_full)), 0, atol=1e-15)
    assert np.isclose(np.max(np.abs(A_full @ x2 - b_full)), 0, atol=1e-15)
    assert np.isclose(np.max(np.abs(A_full @ Z)), 0, atol=1e-15)


@pytest.mark.unit
def test_FixBoundary_with_single_weight():
    """Test Fixing boundary with only a single, passed weight."""
    eq = Equilibrium()
    w = 1.1
    FixZ = FixBoundaryZ(modes=np.array([[0, -1, 0]]), fixed_boundary=True, weight=w)
    FixZ.build(eq)
    print(FixZ._weight)
    np.testing.assert_array_equal(FixZ.weight.size, 1)
    np.testing.assert_array_equal(FixZ.weight, w)
    FixR = FixBoundaryR(modes=np.array([[0, 1, 0]]), fixed_boundary=True, weight=w)
    FixR.build(eq)
    np.testing.assert_array_equal(FixR.weight.size, 1)
    np.testing.assert_array_equal(FixR.weight, w)
