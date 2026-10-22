"""Tests compute utilities."""

import jax
import numpy as np
import pytest

from desc.backend import jnp
from desc.utils import rotation_matrix_angle


@pytest.mark.unit
def test_rotation_matrix_angle():
    """Test that rotation_matrix_angle works with fwd & rev AD for axis=[0, 0, 0]."""
    dfdx_fwd = jax.jacfwd(rotation_matrix_angle)
    dfdx_rev = jax.jacrev(rotation_matrix_angle)
    x0 = jnp.array([0.0, 0.0, 0.0])

    np.testing.assert_allclose(rotation_matrix_angle(x0), np.eye(3))
    np.testing.assert_allclose(dfdx_fwd(x0), np.zeros((3, 3, 3)))
    np.testing.assert_allclose(dfdx_rev(x0), np.zeros((3, 3, 3)))
