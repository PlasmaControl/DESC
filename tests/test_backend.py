"""Tests for backend functions."""

import numpy as np
import pytest

from desc.backend import (
    _lstsq,
    fixed_point,
    jax,
    jnp,
    put,
    root,
    root_scalar,
    sign,
    vmap,
)


@pytest.mark.unit
def test_put():
    """Test put function as replacement for fancy array indexing."""
    a = np.array([0, 0, 0])
    b = np.array([1, 2, 3])
    a = put(a, np.array([0, 1, 2]), np.array([1, 2, 3]))
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_sign():
    """Test modified sign function to return +1 for x=0."""
    assert sign(4) == 1
    assert sign(0) == 1
    assert sign(-10.3) == -1


@pytest.mark.unit
def test_vmap():
    """Test lax numpy implementation of Python's map function."""
    a = np.arange(6)
    inputs = np.stack([a, a[::-1], -a])

    def f(x):
        return x[: x.size // 2] ** 3

    outputs = np.array([[0, 1, 8], [125, 64, 27], [0, -1, -8]])
    np.testing.assert_allclose(vmap(f)(inputs), outputs)
    np.testing.assert_allclose(vmap(f, out_axes=1)(inputs), outputs.T)


@pytest.mark.unit
def test_root():
    """Test root and its derivative works properly."""

    def fun(x, a):
        return a * x - 1

    def find_root(a):
        x0 = jnp.zeros_like(a)
        xk = root(fun, x0, args=(a,))
        return xk

    def find_root_full(a):
        x0 = jnp.zeros_like(a)
        xk, aux = root(fun, x0, args=(a,), full_output=True)
        return xk, aux

    a = 2 * jnp.ones(10)
    x = find_root(a)
    x_full, _ = find_root_full(a)

    exact = 1 / a
    np.testing.assert_allclose(x, exact)
    np.testing.assert_allclose(x_full, exact)

    J = jax.jit(jax.jacfwd(find_root))(a)
    J_rev = jax.jit(jax.jacrev(find_root))(a)
    J_full, _ = jax.jit(jax.jacfwd(find_root_full))(a)
    J_full_rev, _ = jax.jit(jax.jacrev(find_root_full))(a)
    J_exact = jnp.diag(-1 / a**2)

    np.testing.assert_allclose(J, J_exact)
    np.testing.assert_allclose(J_full, J_exact)
    np.testing.assert_allclose(J_rev, J_exact)
    np.testing.assert_allclose(J_full_rev, J_exact)


@pytest.mark.unit
def test_root_scalar():
    """Test root_scalar and its derivative works properly."""

    def fun(x, a):
        return a * x - 1

    def find_root(a):
        x0 = 0.0
        xk = root_scalar(fun, x0, args=(a,))
        return xk

    def find_root_full(a):
        x0 = 0.0
        xk, aux = root_scalar(fun, x0, args=(a,), full_output=True)
        return xk, aux

    a = 2.0
    x = find_root(a)
    x_full, _ = find_root_full(a)

    exact = 1 / a
    np.testing.assert_allclose(x, exact)
    np.testing.assert_allclose(x_full, exact)

    J = jax.jit(jax.jacfwd(find_root))(a)
    J_rev = jax.jit(jax.jacrev(find_root))(a)
    J_full, _ = jax.jit(jax.jacfwd(find_root_full))(a)
    J_full_rev, _ = jax.jit(jax.jacrev(find_root_full))(a)
    J_exact = -1 / a**2

    np.testing.assert_allclose(J, J_exact)
    np.testing.assert_allclose(J_full, J_exact)
    np.testing.assert_allclose(J_rev, J_exact)
    np.testing.assert_allclose(J_full_rev, J_exact)


@pytest.mark.unit
def test_lstsq():
    """Test cholesky factorization of least squares solution."""
    rng = np.random.default_rng(seed=0)

    # tall
    A = rng.standard_normal((10, 5))
    b = rng.standard_normal(10)
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )
    # wide
    A = rng.standard_normal((5, 10))
    b = rng.standard_normal(5)
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )
    # square
    A = rng.standard_normal((5, 5))
    b = rng.standard_normal(5)
    np.testing.assert_allclose(_lstsq(A, b), np.linalg.solve(A, b), rtol=1e-6)
    # scalar
    A = rng.standard_normal((1, 5))
    b = rng.standard_normal(1)
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )


@pytest.mark.unit
@pytest.mark.parametrize("method", ["del2", "iteration", "anderson"])
def test_fixed_point(method):
    """Test fixed point iteration."""

    def func(x, c1, c2):
        return jnp.sqrt(c1 / (x + c2))

    c1 = jnp.array([10.0, 12.0])
    c2 = jnp.array([3.0, 5.0])
    x0 = jnp.array([1.2, 1.3])
    p, (_, i, full_ps, full_fs) = fixed_point(
        func, x0, (c1, c2), xtol=1e-10, method=method, full_output=True, anderson_m=2
    )
    np.testing.assert_allclose(p, [1.4920333, 1.37228132])
    assert (
        (i == 4 and method == "del2")
        or (i == 14 and method == "iteration")
        or (i == 6 and method == "anderson")
    )
