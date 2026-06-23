"""Tests for dipole magnetic fields."""

import numpy as np
import pytest
from scipy.constants import mu_0

from desc.backend import jnp
from desc.dipole import magnetic_dipole_field, magnetic_dipole_vector_field


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
    m_magnitude = 1.0

    B = magnetic_dipole_field(eval_pts, mag_points, phi, theta, m_magnitude)
    A = magnetic_dipole_vector_field(eval_pts, mag_points, phi, theta, m_magnitude)

    np.testing.assert_allclose(B, np.asarray(expected_B), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(A, np.asarray(expected_A), rtol=1e-12, atol=1e-14)
