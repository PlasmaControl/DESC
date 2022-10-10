"""Tests for linear constraints and objectives."""

import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.objectives import FixCurrent, FixIota, FixLambdaGauge
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
