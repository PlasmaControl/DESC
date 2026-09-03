"""Tests for geometry util functions for converting coordinates."""

import numpy as np
import pytest
from jax.test_util import check_grads

from desc.backend import jnp
from desc.utils import (
    rotate_vector_to_vector,
    rotation_matrix,
    rpz2xyz,
    rpz2xyz_vec,
    safenormalize,
    xyz2rpz,
    xyz2rpz_vec,
)


@pytest.mark.unit
def test_rotation_matrix():
    """Test calculation of rotation matrices."""
    A = rotation_matrix([0, 0, np.pi / 2])
    At = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(A, At, atol=1e-10)


@pytest.mark.unit
def test_rotate_vector_to_vector():
    """Test calculation of rotation matrices given two vectors."""
    u = safenormalize(np.random.rand(3))
    v = safenormalize(np.random.rand(3) + np.array([0.1, 0.2, 0.3]))
    A = rotate_vector_to_vector(u, v)
    np.testing.assert_allclose(v, u @ A.T)
    # edge case: vectors parallel
    v = u
    A = rotate_vector_to_vector(u, v)
    np.testing.assert_allclose(v, u @ A.T)
    # edge case: vectors antiparallel
    v = -u
    A = rotate_vector_to_vector(u, v)
    np.testing.assert_allclose(v, u @ A.T)


@pytest.mark.unit
def test_rotate_vector_to_vector_gradients():
    """Test grad of rotate_vector_to_vector against finite diffs."""
    # Standard non-aligned case
    u = jnp.array([1.0, 2.0, 3.0])
    v = jnp.array([4.0, 1.0, -2.0])

    # Validates order 1 derivatives wrt positional args (0: u, 1: v)
    # using forward-mode ('fwd') and reverse-mode ('rev') AD against finite differences
    check_grads(
        rotate_vector_to_vector,
        args=(u, v),
        order=1,
        modes=["fwd", "rev"],
        eps=1e-4,
        atol=1e-3,
        rtol=1e-3,
    )

    # Parallel case
    u_par = jnp.array([1.0, 0.0, 0.0])
    v_par = jnp.array([2.0, 0.0, 0.0])
    check_grads(
        rotate_vector_to_vector,
        args=(u_par, v_par),
        order=1,
        modes=["fwd", "rev"],
        eps=1e-4,
        atol=1e-3,
    )
    # Nearly Parallel case
    u_par = jnp.array([1.0, 1e-5, 0.0])
    v_par = jnp.array([2.0, 0.0, 0.0])
    check_grads(
        rotate_vector_to_vector,
        args=(u_par, v_par),
        order=1,
        modes=["fwd", "rev"],
        eps=1e-4,
        atol=1e-3,
    )

    # Nearly Antiparallel case
    u_anti = jnp.array([1.0, 1e-2, 0.0])
    v_anti = jnp.array([-1.0, 0.0, 0.0])
    check_grads(
        rotate_vector_to_vector,
        args=(u_anti, v_anti),
        order=1,
        modes=["fwd", "rev"],
        eps=1e-6,
        atol=1e-6,
    )

    # Nearly Antiparallel case
    u_anti = jnp.array([1.0, 1e-5, 0.0])
    v_anti = jnp.array([-1.0, 0.0, 0.0])
    check_grads(
        rotate_vector_to_vector,
        args=(u_anti, v_anti),
        order=1,
        modes=["fwd", "rev"],
        eps=1e-6,
        atol=1e-6,
    )

    # Antiparallel case
    u_anti = jnp.array([1.0, 0.0, 0.0])
    v_anti = jnp.array([-1.0, 0.0, 0.0])
    check_grads(
        rotate_vector_to_vector,
        args=(u_anti, v_anti),
        order=1,
        modes=["fwd", "rev"],
        eps=1e-6,
        atol=1e-6,
    )


@pytest.mark.unit
def test_xyz2rpz():
    """Test converting between cartesian and polar coordinates."""
    xyz = np.array([1, 1, 1])
    rpz = xyz2rpz(xyz)
    np.testing.assert_allclose(rpz, [np.sqrt(2), np.pi / 4, 1], atol=1e-10)

    xyz = np.array([0, 1, 1])
    rpz = xyz2rpz_vec(xyz, x=0, y=1)
    np.testing.assert_allclose(rpz, np.array([1, 0, 1]), atol=1e-10)


@pytest.mark.unit
def test_rpz2xyz():
    """Test converting between polar and cartesian coordinates."""
    rpz = np.array([np.sqrt(2), np.pi / 4, 1])
    xyz = rpz2xyz(rpz)
    np.testing.assert_allclose(xyz, [1, 1, 1], atol=1e-10)

    rpz = np.array([[1, 0, 1]])
    xyz = rpz2xyz_vec(rpz, x=0, y=1)
    np.testing.assert_allclose(xyz, np.array([[0, 1, 1]]), atol=1e-10)
