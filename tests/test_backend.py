"""Tests for backend functions."""

import numpy as np
import pytest
from packaging.version import Version

from desc.backend import (
    _lstsq,
    fori_loop,
    jax,
    jnp,
    put,
    qr,
    qr_multiply,
    root,
    root_scalar,
    sign,
    solve_triangular,
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
@pytest.mark.skipif(
    Version(jax.__version__).release != (0, 9, 2),
    reason="DESC only backports this JAX scan-transpose fix to JAX 0.9.2",
)
def test_linear_transpose_scan():
    """Test transposing a scan with a closed-over linear operand."""
    x = jnp.arange(4.0)

    def fun(x):
        return fori_loop(
            0,
            3,
            lambda i, value: value + (i + 1) * x,
            jnp.zeros_like(x),
        )

    transpose = jax.linear_transpose(fun, x)(jnp.ones_like(x))[0]
    np.testing.assert_allclose(transpose, 6 * jnp.ones_like(x))


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
@pytest.mark.parametrize(
    "m, n",
    [
        (100, 20),  # tall
        (20, 100),  # wide
        (50, 50),  # square
        (600, 260),  # tall, and has more than nb columns
    ],
)
@pytest.mark.parametrize("cond", [None, 1e8])
def test_qr_multiply(m, n, cond):
    """Test qr_multiply matches forming Q explicitly."""

    def _qr_multiply_ref(a, c, mode="right"):
        """Reference qr_multiply that forms Q explicitly."""
        Q, R = qr(a, mode="economic")
        if mode == "right":
            cq = Q.T @ c if c.ndim == 1 else c @ Q
        else:
            cq = Q @ c
        return cq, R

    rng = np.random.default_rng(seed=0)
    k = min(m, n)

    if cond is None:
        # gaussian matrices are usually well conditioned
        A = rng.standard_normal((m, n))
    else:
        # create some ill conditioned matrix using reverse SVD
        # this is still full rank
        U = np.linalg.qr(rng.standard_normal((m, k)))[0]
        V = np.linalg.qr(rng.standard_normal((n, k)))[0]
        A = (U * np.logspace(0, -np.log10(cond), k)) @ V.T
    b = rng.standard_normal(m)
    print(
        f"Running {m=} {n=} {cond=}, actual condition number is {np.linalg.cond(A):.3e}"
    )

    # mode="right" with 1D c is Q.T@b
    Qtb, R = qr_multiply(A, b, mode="right")
    Qtb_ref, R_ref = _qr_multiply_ref(A, b, mode="right")
    assert R.shape == (k, n)
    np.testing.assert_allclose(R, R_ref, rtol=1e-12, atol=1e-12 * np.abs(A).max())
    np.testing.assert_allclose(Qtb, Qtb_ref, rtol=1e-10, atol=1e-10)

    # mode="right" with 2D c is c@Q
    C = rng.standard_normal((3, m))
    CQ, _ = qr_multiply(A, C, mode="right")
    np.testing.assert_allclose(CQ, _qr_multiply_ref(A, C, "right")[0], atol=1e-10)

    # mode="left" is Q@c, with c=I recovers Q
    Q, _ = qr_multiply(A, np.eye(k), mode="left")
    np.testing.assert_allclose(Q, _qr_multiply_ref(A, np.eye(k), "left")[0], atol=1e-10)
    np.testing.assert_allclose(Q.T @ Q, np.eye(k), atol=1e-10)
    np.testing.assert_allclose(Q @ R, A, atol=1e-10 * np.abs(A).max())
    y = rng.standard_normal(k)
    Qy, _ = qr_multiply(A, y, mode="left")
    np.testing.assert_allclose(Qy, _qr_multiply_ref(A, y, "left")[0], atol=1e-10)

    # solve A@x = b
    if m >= n:
        x = solve_triangular(R, Qtb)
        x_ref = solve_triangular(R_ref, Qtb_ref)
    else:
        # for wide A, use the QR of A.T
        Q1, R1 = qr_multiply(A.T, np.eye(k), mode="left")
        Q1_ref, R1_ref = _qr_multiply_ref(A.T, np.eye(k), mode="left")
        x = Q1 @ solve_triangular(R1.T, b, lower=True)
        x_ref = Q1_ref @ solve_triangular(R1_ref.T, b, lower=True)
    x_np = np.linalg.lstsq(A, b, rcond=None)[0]
    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-8 * np.abs(x_np).max())
    np.testing.assert_allclose(x, x_np, rtol=1e-8, atol=1e-8 * np.abs(x_np).max())
