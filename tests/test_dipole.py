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
)
from desc.magnetic_fields._core import dipole_field


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
    M0 = 1.0

    B = magnetic_dipole_field(eval_pts, mag_points, phi, theta, M0)
    A = magnetic_dipole_vector_field(eval_pts, mag_points, phi, theta, M0)

    np.testing.assert_allclose(B, np.asarray(expected_B), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(A, np.asarray(expected_A), rtol=1e-12, atol=1e-14)


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
    wrong_B = dipole_field(eval_pts, source_pts, np.array([[0.0, 0.0, 1.0]] * 2))

    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-14)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(B, wrong_B, rtol=1e-12, atol=1e-14)
