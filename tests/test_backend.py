"""Tests for backend functions."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from desc.backend import _lstsq, jax, jnp, put, root, root_scalar, sign, vmap
from desc.batching import batch_map


def _run_forced_cpu_devices(code, num_devices=4):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env,
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
def test_batch_map_with_chunk_size():
    """Test batch_map with a chunk size."""
    x = jnp.arange(5.0)
    np.testing.assert_allclose(batch_map(lambda y: y + 1, x, batch_size=2), x + 1)


@pytest.mark.unit
def test_sharded_chunked_batching():
    """Test chunked batching with sharded input data."""
    _run_forced_cpu_devices("""
        import numpy as np

        from desc.backend import jax, jnp
        from desc.batching import batch_map, vmap_chunked

        assert jax.device_count() == 4
        x = jnp.arange(13.0)

        cases = [
            (
                lambda y: batch_map(
                    lambda z: z + 1,
                    y,
                    batch_size=2,
                    shard_input_data=True,
                ),
                x + 1,
            ),
            (
                lambda y: batch_map(lambda z: z + 1, y, shard_input_data=True),
                x + 1,
            ),
            (
                lambda y: batch_map(
                    lambda z: z + 1,
                    y,
                    batch_size=1,
                    strip_dim0=True,
                    shard_input_data=True,
                ),
                x + 1,
            ),
            (
                lambda y: vmap_chunked(
                    lambda z, scale: z * scale,
                    in_axes=(0, None),
                    chunk_size=2,
                    shard_input_data=True,
                )(y, 3.0),
                x * 3,
            ),
            (
                lambda y: vmap_chunked(
                    lambda z, scale: z * scale,
                    in_axes=(0, None),
                    shard_input_data=True,
                )(y, 3.0),
                x * 3,
            ),
            (
                lambda y: batch_map(
                    lambda z: z,
                    y,
                    batch_size=2,
                    reduction=jnp.add,
                    chunk_reduction=jnp.sum,
                    shard_input_data=True,
                ),
                jnp.sum(x),
            ),
        ]
        for fun, expected in cases:
            np.testing.assert_allclose(fun(x), expected)
            np.testing.assert_allclose(jax.jit(fun)(x), expected)

        two_inputs = lambda y, z: vmap_chunked(
            lambda a, b: a - b,
            in_axes=(0, 0),
            chunk_size=2,
            shard_input_data=True,
        )(y, z)
        np.testing.assert_allclose(two_inputs(x, x[::-1]), x - x[::-1])
        np.testing.assert_allclose(jax.jit(two_inputs)(x, x[::-1]), x - x[::-1])
        """)


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
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )
    # scalar
    A = rng.standard_normal((1, 5))
    b = rng.standard_normal(1)
    np.testing.assert_allclose(
        _lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0], rtol=1e-6
    )


@pytest.mark.unit
def test_make_shardable():
    """Test that sharding works."""
    _run_forced_cpu_devices("""
        import numpy as np

        from desc.backend import jax, jnp
        from desc.batching import make_shardable

        assert jax.device_count() == 4

        f = np.arange(21)
        sf, rf = make_shardable(f, num_devices=4)
        assert sf.size == 20
        assert rf.size == 1
        np.testing.assert_allclose(
            np.concatenate([np.asarray(jnp.sin(sf)), np.asarray(jnp.sin(rf))]),
            jnp.sin(f),
        )

        f = jnp.arange(20).reshape(2, 10)
        sf, rf = make_shardable(f, axis=1, num_devices=4)
        assert sf.shape == (2, 8)
        assert rf.shape == (2, 2)
        np.testing.assert_allclose(
            np.concatenate(
                [np.asarray(jnp.sin(sf)), np.asarray(jnp.sin(rf))], axis=1
            ),
            jnp.sin(f),
        )
        """)
