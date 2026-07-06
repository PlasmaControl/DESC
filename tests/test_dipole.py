"""Tests for dipole magnetic fields."""

import numpy as np
import pytest
from scipy.constants import mu_0

from desc.backend import jnp
from desc.dipole import (
    DipoleSet,
    _Dipole,
    magnetic_dipole_field,
    magnetic_dipole_vector_field,
    import_dipoles
)
from desc.magnetic_fields._core import dipole_field
from desc.io import load
from pathlib import Path

TEST_DIR = (Path(__file__).parent / "inputs").resolve()
muse_dipoles_csv_filename = TEST_DIR / "muse_dipoles_desc.csv"
# MUSE plasma equilibrium, taken from Muse-Design-Paper https://github.com/tmqian/Muse-Design-Paper/tree/main#
muse_equilibrium_filename = TEST_DIR / "input.muse-fixedb_output.h5"

eq = load(muse_equilibrium_filename)[-1]
# imports one period of the MUSE dipoleset, which has NFP=2 and sym=True
MUSE_NFP1_symFalse = import_dipoles(NFP=1, sym=False, filename=muse_dipoles_csv_filename)
# imports one period of the MUSE dipoleset, which has NFP=2 and sym=True
MUSE_NFP1_symTrue = import_dipoles(NFP=1, sym=True, filename=muse_dipoles_csv_filename)
# imports one period of the MUSE dipoleset, which has NFP=2 and sym=False
MUSE_NFP2_symFalse = import_dipoles(NFP=2, sym=False, filename=muse_dipoles_csv_filename)
# imports one period of the MUSE dipoleset, which has NFP=2 and sym=True
MUSE_NFP2_symTrue = import_dipoles(NFP=2, sym=True, filename=muse_dipoles_csv_filename)


@pytest.mark.unit
@pytest.mark.parametrize(
    "theta, phi, expected_B, expected_A",
    [
        (
            0.0,
            0.0,
            [[0.0, 0.0, 2 * mu_0 / (4 * np.pi)]],
            [[0.0, 0.0, 0.0]],
        ),
        (
            np.pi / 2,
            0.0,
            [[-mu_0 / (4 * np.pi), 0.0, 0.0]],
            [[0.0, -mu_0 / (4 * np.pi), 0.0]],
        ),
    ],
)
def test_magnetic_dipole(theta, phi, expected_B, expected_A):
    """Test one dipole at the origin against analytic field values."""
    mag_points = jnp.array([[0.0, 0.0, 0.0]])
    eval_pts = jnp.array([[0.0, 0.0, 1.0]])
    m0 = 1.0

    B = magnetic_dipole_field(eval_pts, mag_points, phi, theta, m0)
    A = magnetic_dipole_vector_field(eval_pts, mag_points, phi, theta, m0)

    np.testing.assert_allclose(B, np.asarray(expected_B), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(A, np.asarray(expected_A), rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipoleset_at_multiple_evaluation_points():
    """Test the dipoleset magnetic field calculation at multiple points."""
    eval_pts = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.5], 
                          [1.5, -1.5, 0.0], [-2.0, 0.0, 1.0], [1.0, 2.0, 3.0],
                          [-1.5, 2.0, -2.0], [-1.5, -1.0, 2.0], [1.5, -2.0, -1.0], 
                          [1.0, 2.0, -1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    dipoles = DipoleSet(
        _Dipole(x=0.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0),
        _Dipole(x=1.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=-0.25),
        NFP=1,
        sym=False,
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")
    source_pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -0.25]])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)
    wrong_B = dipole_field(eval_pts, source_pts, np.array([[0.0, 0.0, 1.0]] * 2))

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(B, wrong_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipoleset_uses_each_dipole_strength():
    """Test DipoleSet field uses each dipole's own m0 and rho."""
    eval_pts = np.array([[0.0, 0.0, 2.0]])
    dipoles = DipoleSet(
        _Dipole(x=0.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0),
        _Dipole(x=1.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=-0.25),
        NFP=1,
        sym=False,
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")

    source_pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -0.25]])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipoleset_uses_each_dipole_direction():
    """Test DipoleSet field uses each dipole's own phi and theta."""
    eval_pts = np.array([[0.0, 0.0, 2.0]])
    dipoles = DipoleSet(
        _Dipole(x=0.0, y=0.0, z=0.0, phi=np.pi, theta=np.pi/2, m0=1.0, rho=1.0),
        _Dipole(x=1.0, y=0.0, z=0.0, phi=-np.pi/2, theta=0.0, m0=1.0, rho=1.0),
        NFP=1,
        sym=False,
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")

    source_pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipole_translation():
    """Test Dipole translate()."""
    eval_pts = np.array([[0.0, 0.0, 2.0]])
    dipole1 = _Dipole(x=0.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0)
    dipole2 = dipole1.translate(displacement=[0.0, 4.0, 0.0])
    dipoles = DipoleSet(
        dipole1, 
        dipole2, 
        NFP=1, 
        sym=False,
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")

    source_pts = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    m_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipole_rotate():
    """Test Dipole rotate()."""
    # Used in the context of rotated_dipoles.rotate(axis=[0, 0, 1], angle=2 * jnp.pi * k / NFP)
    eval_pts = np.array([[0.0, 0.0, 2.0]])
    dipole1 = _Dipole(x=0.0, y=1.0, z=0.0, phi=0.0, theta=np.pi/2, m0=1.0, rho=1.0)
    dipole2 = dipole1.rotate(axis=[0,0,1],angle=0)
    dipoles = DipoleSet(
        dipole1,
        dipole2,
        NFP=1,
        sym=False
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")

    source_pts = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    m_vectors = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipole_flip():
    """Test Dipole flip()."""
    eval_pts = np.array([[0.0, 0.0, 2.0]])
    dipole1 = _Dipole(x=0.0, y=0.0, z=1.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0)
    dipole2 = dipole1.flip(normal=[0,0,1])
    dipoles = DipoleSet(
        dipole1,
        dipole2,
        NFP=1,
        sym=False
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")

    source_pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    m_vectors = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipoleset_symmetry():
    """Test application of dipole symmetry when sym set to True in a DipoleSet object."""
    eval_pts = np.array([[0.0, 0.0, 2.0]])
    dipoles = DipoleSet(
        _Dipole(x=0.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0),
        _Dipole(x=1.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0),
        NFP=1,
        sym=True,
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")

    source_pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    m_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipoleset_NFP_without_symmetry():
    """Test application of dipole NFP when NFP>1 in a DipoleSet object, with sym=False."""
    eval_pts = np.array([[0.0, 0.0, 2.0]])
    dipoles = DipoleSet(
        _Dipole(x=0.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0),
        _Dipole(x=0.1, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0),
        NFP=2,
        sym=False,
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")

    source_pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    m_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_dipoleset_NFP_with_symmetry():
    """Test application of dipole NFP when NFP>1 in a DipoleSet object, with sym=True."""
    eval_pts = np.array([[0.0, 0.0, 2.0]])
    dipoles = DipoleSet(
        _Dipole(x=0.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0),
        _Dipole(x=1.0, y=0.0, z=0.0, phi=0.0, theta=0.0, m0=1.0, rho=1.0),
        NFP=2,
        sym=True,
    )

    B = dipoles.compute_magnetic_field(eval_pts, basis="xyz")

    # need to figure out last 4 coordinates for checking calculation
    source_pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [], [], [], []])
    m_vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [], [], [], []])
    expected_B = dipole_field(eval_pts, source_pts, m_vectors)

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)


@pytest.mark.unit
def test_MUSE_field_calculation_NFP1_symFalse():
    """Test the field calculation of the MUSE dipoles, with NFP=1 and sym=False."""
    eval_pts = np.array([[0.1, 0.0, 0.1]])
    B = MUSE_NFP1_symFalse.compute_magnetic_field(eval_pts, basis="xyz")
    # expected B taken from simsopt calculation
    expected_B = [[-0.00803107, -0.01251551,  0.00783013]]
    np.testing.assert_allclose(B, expected_B, rtol=1e-10, atol=1e-14)


@pytest.mark.unit
def test_MUSE_field_calculation_NFP1_symTrue():
    """Test the field calculation of the MUSE dipoles, with NFP=1 and sym=True."""
    eval_pts = np.array([[0.1, 0.0, 0.1]])
    B = MUSE_NFP1_symTrue.compute_magnetic_field(eval_pts, basis="xyz")
    # expected B taken from simsopt calculation
    expected_B = [[-0.01055722, -0.02013324, -0.00170434]]
    np.testing.assert_allclose(B, expected_B, rtol=1e-10, atol=1e-14)


@pytest.mark.unit
def test_MUSE_field_calculation_NFP2_symFalse():
    """Test the field calculation of the MUSE dipoles, with NFP=2 and sym=False."""
    eval_pts = np.array([[0.1, 0.0, 0.1]])
    B = MUSE_NFP2_symFalse.compute_magnetic_field(eval_pts, basis="xyz")
    # expected B taken from simsopt calculation
    expected_B = [[-0.0053455,  -0.01181211,  0.00882978]]
    np.testing.assert_allclose(B, expected_B, rtol=1e-10, atol=1e-14)


@pytest.mark.unit
def test_MUSE_field_calculation_NFP2_symTruee():
    """Test the field calculation of the MUSE dipoles, with NFP=2 and sym=True."""
    eval_pts = np.array([[0.1, 0.0, 0.1]])
    B = MUSE_NFP2_symTrue.compute_magnetic_field(eval_pts, basis="xyz")
    # expected B taken from simsopt calculation
    expected_B = [[-0.01015419, -0.01831055, -0.00213807]]
    np.testing.assert_allclose(B, expected_B, rtol=1e-10, atol=1e-14)

