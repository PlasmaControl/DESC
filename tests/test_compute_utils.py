"""Tests compute utilities."""

import jax
import numpy as np
import pytest

from desc.backend import jnp
from desc.basis import DoubleChebyshevFourierBasis
from desc.compute.nabla import curl_cylindrical, div_cylindrical
from desc.grid import CylindricalGrid
from desc.transform import Transform
from desc.utils import rotation_matrix


@pytest.mark.unit
def test_rotation_matrix():
    """Test that rotation_matrix works with fwd & rev AD for axis=[0, 0, 0]."""
    dfdx_fwd = jax.jacfwd(rotation_matrix)
    dfdx_rev = jax.jacrev(rotation_matrix)
    x0 = jnp.array([0.0, 0.0, 0.0])

    np.testing.assert_allclose(rotation_matrix(x0), np.eye(3))
    np.testing.assert_allclose(dfdx_fwd(x0), np.zeros((3, 3, 3)))
    np.testing.assert_allclose(dfdx_rev(x0), np.zeros((3, 3, 3)))


@pytest.mark.unit
def test_div_and_curl():
    """Test that divergence and curl utilities give the expected results."""
    grid = CylindricalGrid(3, 3, 3)
    basis = DoubleChebyshevFourierBasis(grid.L, grid.M, grid.N)
    transform = Transform(grid, basis, 1, build=True, build_pinv=True, method="rpz")

    # Shift to "real space"
    scales = np.ones(3)
    shifts = np.array([1, 0, 1])
    r, phi, z = (grid.nodes * scales + shifts).T

    # Define a test function to take the derivative of
    f = np.stack(
        [r * z * np.cos(phi), r * z**2 * np.cos(phi), r * np.sin(phi) + z], axis=-1
    )

    # Calculate the curl numerically and analytically
    curl_f_num = curl_cylindrical(f, r, r, transform, transform, scales)
    curl_f_an = np.stack(
        [
            np.cos(phi) - 2 * z * np.cos(phi) * r,
            r * np.cos(phi) - np.sin(phi),
            2 * z**2 * np.cos(phi) + z * np.sin(phi),
        ],
        axis=-1,
    )

    # Calculate the divergence numerically and analytically
    div_f_num = div_cylindrical(f, r, r, transform, transform, scales)
    div_f_an = 2 * z * np.cos(phi) - z**2 * np.sin(phi) + 1

    np.testing.assert_allclose(curl_f_an, curl_f_num)
    np.testing.assert_allclose(div_f_an, div_f_num)
