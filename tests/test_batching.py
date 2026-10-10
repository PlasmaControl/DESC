"""Tests for desc.batching, especially the chunked-Jacobian linearize fast path.

jacfwd_chunked builds a Jacobian by chunking a batch of tangent directions
through vmap+scan to bound memory. Since the function being differentiated
does not depend on the tangent direction, a naive implementation that chunks
fresh jax.jvp calls re-evaluates the (potentially expensive, nonlinear) primal
once per chunk instead of once overall -- chunking exists to bound memory, not
to multiply redundant work. These tests check both that the result is still
correct across chunk sizes, and (via monkeypatching jax.linearize) that the
fix actually avoids the redundant re-evaluation rather than just happening to
produce the same numbers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from desc.batching import jacfwd_chunked


def _nonlinear(x):
    """A function with real, non-trivial nonlinear structure to differentiate."""
    y = jnp.sin(x) * jnp.cos(x[::-1])
    for _ in range(5):
        y = jnp.tanh(y @ jnp.eye(x.size) + x)
    return y


class TestJacfwdChunked:
    """Tests for jacfwd_chunked."""

    @pytest.mark.unit
    @pytest.mark.parametrize("chunk_size", [None, 12, 5, 3])
    def test_matches_jax_jacfwd_single_arg(self, chunk_size):
        """jacfwd_chunked should exactly match jax.jacfwd regardless of chunking."""
        x = jnp.linspace(0.1, 1.0, 12)
        jac = jacfwd_chunked(_nonlinear, chunk_size=chunk_size)(x)
        jac_ref = jax.jacfwd(_nonlinear)(x)
        np.testing.assert_allclose(jac, jac_ref, atol=1e-10)

    @pytest.mark.unit
    @pytest.mark.parametrize("chunk_size", [None, 12, 4])
    def test_matches_jax_jacfwd_multi_arg_and_aux(self, chunk_size):
        """Multiple simultaneous argnums and has_aux should also match jax.jacfwd."""

        def fun(x, y):
            return jnp.sin(x) * y, x.sum()

        x = jnp.linspace(0.1, 1.0, 12)
        y = jnp.linspace(2.0, 3.0, 12)
        jac_ref, aux_ref = jax.jacfwd(fun, argnums=(0, 1), has_aux=True)(x, y)
        jac, aux = jacfwd_chunked(
            fun, argnums=(0, 1), has_aux=True, chunk_size=chunk_size
        )(x, y)
        for a, b in zip(jax.tree.leaves(jac), jax.tree.leaves(jac_ref)):
            np.testing.assert_allclose(a, b, atol=1e-10)
        np.testing.assert_allclose(aux, aux_ref)

    @pytest.mark.unit
    def test_linearizes_once_not_per_chunk(self, monkeypatch):
        """Regression test: chunking must not re-evaluate the primal per chunk.

        Before this fix, jacfwd_chunked chunked fresh jax.jvp calls, so the
        nonlinear primal was re-evaluated once per scan chunk instead of once
        overall -- the redundant work grew with the number of chunks, exactly
        backwards from what chunking (a memory bound) is supposed to buy you.
        jax.linearize should be called exactly once regardless of how many
        chunks the direction batch is split into, and not at all when
        everything fits in a single chunk (there vmap already shares the
        primal for free, so the fast path is skipped entirely).
        """
        import desc.batching as db

        calls = {"n": 0}
        real_linearize = jax.linearize

        def counting_linearize(*args, **kwargs):
            calls["n"] += 1
            return real_linearize(*args, **kwargs)

        monkeypatch.setattr(db.jax, "linearize", counting_linearize)

        x = jnp.linspace(0.1, 1.0, 12)

        calls["n"] = 0
        db.jacfwd_chunked(_nonlinear, chunk_size=3)(x)  # 12/3 = 4 scan chunks
        assert calls["n"] == 1, f"expected linearize called once, got {calls['n']}"

        calls["n"] = 0
        db.jacfwd_chunked(_nonlinear, chunk_size=12)(x)  # single chunk
        assert calls["n"] == 0, "single-chunk case should not need linearize"
