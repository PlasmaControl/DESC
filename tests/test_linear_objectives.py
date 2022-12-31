"""Tests for linear constraints and objectives."""

import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.objectives import (
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixLambdaGauge,
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
    eqi = Equilibrium(iota=PowerSeriesProfile(0, 0))
    eqc = Equilibrium(current=PowerSeriesProfile(0))
    # nonexistent toroidal current can't be constrained
    with pytest.raises(RuntimeError):
        eqi.solve(constraints=FixCurrent())
    # nonexistent rotational transform can't be constrained
    with pytest.raises(RuntimeError):
        eqc.solve(constraints=FixIota())
    # toroidal current and rotational transform can't be constrained simultaneously
    with pytest.raises(ValueError):
        eqi.solve(constraints=(FixCurrent(), FixIota()))


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
