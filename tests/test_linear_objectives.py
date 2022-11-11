"""Tests for linear constraints and objectives."""

import numpy as np
import pytest

import desc.examples
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.objectives import (
    FixAxisR,
    FixAxisZ,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixLambdaGauge,
    FixLambdaZero,
    FixModeR,
    FixModeZ,
    FixPressure,
    FixPsi,
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
    # cannot use two constraints on the same R mode
    with pytest.raises(RuntimeError):
        fixmode1 = FixModeR(modes=np.array([1, 1, 0]))
        fixmode2 = FixModeR(modes=np.array([[0, 0, 0], [1, 1, 0]]))
        eqc.solve(constraints=(fixmode1, fixmode2))
    # cannot use two constraints on the same Z mode
    with pytest.raises(RuntimeError):
        fixmode1 = FixModeZ(modes=np.array([1, -1, 0]))
        fixmode2 = FixModeZ(modes=np.array([1, -1, 0]))
        eqc.solve(constraints=(fixmode1, fixmode2))
    # cannot use two incompatible constraints
    with pytest.raises(ValueError):
        con1 = FixCurrent(target=eqc.c_l)
        con2 = FixCurrent(target=eqc.c_l + 1)
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
        modes=np.array([L, M, 0])
    )  # no target supplied, so defaults to the eq's current R_LMN coeff
    fixZ = FixModeZ(
        modes=np.array([L, -M, 0])
    )  # no target supplied, so defaults to the eq's current Z_LMN coeff
    orig_R_val = eq.R_lmn[eq.R_basis.get_idx(L=L, M=M, N=0)]
    orig_Z_val = eq.Z_lmn[eq.Z_basis.get_idx(L=L, M=-M, N=0)]

    constraints = (
        FixLambdaGauge(),
        FixPressure(),
        FixIota(),
        FixPsi(),
        FixBoundaryR(fixed_boundary=True),
        FixBoundaryZ(fixed_boundary=True),
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
def test_fixed_axis_solve():
    """Test solving an equilibrium with a fixed axis constraint."""
    # also tests zero lambda solve
    # Reset DSHAPE to initial guess, fix axis, and then resolve
    # and check that the axis stayed fix
    eq = desc.examples.get("DSHAPE")
    eq.axis.R_n[0] = 3.6
    eq.set_initial_guess()

    orig_R_val = eq.axis.R_n
    orig_Z_val = eq.axis.Z_n

    constraints = (
        FixLambdaZero(),
        FixPressure(),
        FixIota(),
        FixPsi(),
        FixAxisR(),
        FixAxisZ(),
    )

    eq.solve(
        verbose=3,
        ftol=1e-2,
        objective="force",
        maxiter=10,
        xtol=1e-6,
        constraints=constraints,
    )

    np.testing.assert_almost_equal(orig_R_val, eq.axis.R_n)
    np.testing.assert_almost_equal(orig_Z_val, eq.axis.Z_n)
    np.testing.assert_array_equal(np.zeros_like(eq.L_lmn), eq.L_lmn)
