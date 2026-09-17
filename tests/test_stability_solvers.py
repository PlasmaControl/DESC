"""Unit tests for desc/compute/_stability_solvers.py.

These test the in-package solver machinery directly -- no equilibrium, no
subprocess, no scraped stdout. That is the point of having moved it into the
package: a test is now a driver that picks a size, calls in, and compares.

The properties checked here are the ones the deflated solve actually depends on:

* the reduced <-> physical maps round-trip, and their ``_h`` variants are true
  transposes (deflation needs ``P^T``, not ``P^-1``);
* the interpolation matrices are exact where exactness is claimed -- Fourier
  reproduces representable modes exactly, barycentric reproduces polynomials;
* the ring groups tile the DOFs exactly once, and the block preconditioner is the
  exact inverse of the block diagonal it was built from;
* the block Cholesky reproduces its blocks and flags a non-SPD block.

Every test here is self-contained: no equilibrium, no data files, and no code
outside this repository. Correctness is established against independently
computed ground truth (exact polynomial/Fourier interpolation, adjoint
identities, factorization residuals, dense linear solves), not by diffing
against another implementation.
"""

import numpy as np
import pytest

from desc.backend import jnp
from desc.compute._stability_solvers import (
    barycentric_matrix,
    factor_ring_blocks_traced,
    fourier_interp_matrix,
    from_phys,
    make_block_precond,
    make_transfer,
    ring_index_maps,
    to_phys,
)


def _meta(n_rho, n_theta, n_zeta, seed=0):
    """A synthetic level with random but invertible per-node transforms.

    The real ``linv_dt`` comes from a Cholesky factor of the mass blocks; here
    any well-conditioned 3x3 per node exercises the same code paths. ``keep``
    drops the xi^rho DOFs on the innermost and outermost rho shells, exactly as
    the Dirichlet condition does.
    """
    rng = np.random.default_rng(seed)
    n_total = n_rho * n_theta * n_zeta
    shell = n_theta * n_zeta

    drop = np.concatenate([np.arange(shell), n_total - shell + np.arange(shell)])
    keep = np.setdiff1d(np.arange(3 * n_total), drop)

    lin = rng.standard_normal((n_total, 3, 3)) * 0.3 + np.eye(3)
    diag = 1.0 + 0.1 * rng.random((n_total, 3))
    return {
        "n_rho": n_rho,
        "n_theta": n_theta,
        "n_zeta": n_zeta,
        "n_total": n_total,
        "keep": jnp.asarray(keep),
        "diag": jnp.asarray(diag),
        "linv_dt": jnp.asarray(lin),
        "inv_linv_dt": jnp.asarray(np.linalg.inv(lin)),
        "linv_dt_h": jnp.asarray(np.swapaxes(lin, -1, -2)),
        "inv_linv_dt_h": jnp.asarray(np.swapaxes(np.linalg.inv(lin), -1, -2)),
        "n_keep": keep.size,
    }


@pytest.mark.unit
def test_reduced_physical_roundtrip():
    """from_phys(to_phys(q)) == q on the kept DOFs."""
    m = _meta(4, 6, 4)
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal(m["n_keep"]))
    back = from_phys(m, to_phys(m, q))
    np.testing.assert_allclose(np.asarray(back), np.asarray(q), rtol=0, atol=1e-11)


@pytest.mark.unit
def test_fourier_interp_is_exact_on_representable_modes():
    """Fourier transfer reproduces any mode the coarse grid can represent.

    Exactness is the whole justification for using it on theta and zeta, so this
    asserts equality to machine precision rather than an interpolation-order
    bound. Modes are swept up to the coarse Nyquist; above it, aliasing is
    expected and not tested.
    """
    n_src, n_dst, period = 8, 20, 2.0 * np.pi
    P = fourier_interp_matrix(n_src, n_dst, period)
    x = np.arange(n_src) * (period / n_src)
    y = np.arange(n_dst) * (period / n_dst)

    for mode in range(0, n_src // 2):
        for f in (np.cos, np.sin):
            got = P @ f(mode * x)
            np.testing.assert_allclose(got, f(mode * y), atol=1e-12)


@pytest.mark.unit
def test_barycentric_is_exact_on_polynomials():
    """Barycentric transfer reproduces polynomials up to degree n_src-1.

    Also pins the coincident-node case: where a target node equals a source
    node, the row must be an exact delta, not a 0/0.
    """
    rng = np.random.default_rng(2)
    x_src = np.sort(rng.random(7))
    x_dst = np.sort(rng.random(13))
    x_dst[0] = x_src[0]  # force a coincident node

    P = barycentric_matrix(x_src, x_dst)
    np.testing.assert_allclose(P[0], np.eye(7)[0], atol=1e-14)

    for deg in range(7):
        np.testing.assert_allclose(P @ x_src**deg, x_dst**deg, atol=1e-10)

    # Interpolation is a partition of unity: rows sum to 1.
    np.testing.assert_allclose(P.sum(axis=1), np.ones(13), atol=1e-12)


@pytest.mark.unit
def test_ring_groups_tile_every_dof_exactly_once():
    """Every kept DOF appears in exactly one ring, and rings have 3*n_theta slots.

    A DOF appearing twice would be preconditioned twice (the scatter uses
    ``.add``); a DOF appearing zero times would never be preconditioned at all.
    """
    n_rho, n_theta, n_zeta = 4, 6, 4
    m = _meta(n_rho, n_theta, n_zeta)
    sel, pad, G = ring_index_maps(np.asarray(m["keep"]), (n_rho, n_theta, n_zeta))

    assert G.shape == (n_rho * n_zeta, 3 * n_theta)
    counts = np.bincount(G[G >= 0], minlength=m["n_keep"])
    assert counts.size == m["n_keep"], "ring indices exceed the reduced dimension"
    np.testing.assert_array_equal(counts, np.ones(m["n_keep"], dtype=counts.dtype))
    np.testing.assert_array_equal(np.asarray(pad) > 0, G >= 0)


@pytest.mark.unit
def test_prolongation_adjoint_is_exact():
    """PT is the true transpose of P.

    Checked on a synthetic level so the property is tested without an
    equilibrium.
    """
    m_c = _meta(3, 4, 2, seed=3)
    m_f = _meta(6, 8, 4, seed=4)
    rho_c = np.linspace(0.05, 0.95, 3)
    rho_f = np.linspace(0.05, 0.95, 6)

    pr = jnp.asarray(barycentric_matrix(rho_c, rho_f))
    pt = jnp.asarray(fourier_interp_matrix(4, 8, 2.0 * np.pi))
    pz = jnp.asarray(fourier_interp_matrix(2, 4, 2.0 * np.pi))
    P, PT = make_transfer(m_c, m_f, pr, pt, pz)

    rng = np.random.default_rng(0)
    defect = 0.0
    for _ in range(6):
        x = jnp.asarray(rng.standard_normal(m_c["n_keep"]))
        y = jnp.asarray(rng.standard_normal(m_f["n_keep"]))
        lhs = float(jnp.vdot(P(x), y).real)
        rhs = float(jnp.vdot(x, PT(y)).real)
        defect = max(defect, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300))
    assert defect < 1e-11, f"<Px,y> != <x,PTy>: worst relative defect {defect:.3e}"


@pytest.mark.unit
def test_block_precond_is_exact_inverse_of_block_diagonal():
    """``M^-1`` from the ring blocks inverts the block-diagonal matrix exactly.

    Built on the production ring groups (``ring_index_maps``), so a wrong group
    map, padding mask or Cholesky apply shows up as a residual, not as a slower
    solve.
    """
    n_rho, n_theta, n_zeta = 4, 6, 2
    m = _meta(n_rho, n_theta, n_zeta, seed=6)
    n = m["n_keep"]
    _, _, G = ring_index_maps(np.asarray(m["keep"]), (n_rho, n_theta, n_zeta))

    rng = np.random.default_rng(7)
    A = np.zeros((n, n))
    blocks = np.zeros((G.shape[0], G.shape[1], G.shape[1]))
    for gi, g in enumerate(G):
        pos = np.where(g >= 0)[0]
        live = g[pos]
        X = rng.standard_normal((live.size, live.size))
        blk = X @ X.T + live.size * np.eye(live.size)
        A[np.ix_(live, live)] = blk
        blocks[gi][np.ix_(pos, pos)] = blk
        pad_pos = np.where(g < 0)[0]
        blocks[gi][pad_pos, pad_pos] = 1.0  # inert identity on padding

    L, ok, _ = factor_ring_blocks_traced(jnp.asarray(blocks))
    assert bool(ok), "block Cholesky failed on an SPD block diagonal"
    M = make_block_precond(L, G, n)

    x = rng.standard_normal(n)
    got = np.asarray(M(jnp.asarray(A @ x)))
    np.testing.assert_allclose(got, x, rtol=0, atol=1e-10)


@pytest.mark.unit
def test_block_cholesky_factorization_residual():
    """L L^T reproduces the blocks; a non-SPD or non-finite block is flagged."""
    rng = np.random.default_rng(11)
    m, b = 10, 8
    X = rng.standard_normal((m, b, b))
    blocks_np = np.einsum("mij,mkj->mik", X, X) + b * np.eye(b)[None]
    blocks = jnp.asarray(blocks_np)

    L, ok, ridge = factor_ring_blocks_traced(blocks)
    assert bool(ok) and ridge == 0.0, "Cholesky failed on an SPD block diagonal"
    recon = np.einsum("mij,mkj->mik", np.asarray(L), np.asarray(L))
    rel = np.linalg.norm(recon - blocks_np) / np.linalg.norm(blocks_np)
    print(f"\n  ||M - L L^T||_F / ||M||_F = {rel:.3e}")
    assert rel < 1e-13, f"factorization residual {rel:.3e} is too large"

    # sigma above lambda_min makes a block indefinite: reported, not hidden
    indef = blocks_np.copy()
    indef[0] -= (np.linalg.eigvalsh(indef[0])[-1] + 1.0) * np.eye(b)
    _, ok_indef, _ = factor_ring_blocks_traced(jnp.asarray(indef))
    assert not bool(ok_indef)

    _, ok_bad, _ = factor_ring_blocks_traced(blocks.at[0, 0, 0].set(jnp.nan))
    assert not bool(ok_bad)


# ---------------------------------------------------------------------------
# Solver-option resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_kwargs_win_over_environment(monkeypatch):
    """An explicit keyword argument overrides the environment fallback.

    This pins the fix for a real defect. The old code resolved several numerical
    options as ``os.environ.get(VAR, str(kwargs.get(name, default)))``, which
    used the kwarg only as the ENVIRONMENT'S default -- so whenever the variable
    happened to be exported, an explicit argument was silently discarded. With
    a solver variable exported across the job scripts, passing the matching
    kwarg did nothing.

    These are numerical choices that change the answer, so a caller that passes
    one must get it. The environment stays as a fallback only.
    """
    from desc.compute._stability import _solver_flag, _solver_opt

    monkeypatch.setenv("AGNI_JD_OUTER", "999")
    monkeypatch.setenv("AGNI_EIGENSOLVER", "eigsh_callback")
    monkeypatch.setenv("AGNI_GPU_LU", "1")

    # kwarg present -> kwarg wins, environment ignored
    assert _solver_opt({"jd_outer": 64}, "jd_outer", "AGNI_JD_OUTER", 200, int) == 64
    assert (
        _solver_opt(
            {"eigensolver": "pcg_deflated"}, "eigensolver", "AGNI_EIGENSOLVER", "x"
        )
        == "pcg_deflated"
    )
    assert _solver_flag({"gpu_lu": False}, "gpu_lu", "AGNI_GPU_LU") is False

    # kwarg absent -> environment is the fallback
    assert _solver_opt({}, "jd_outer", "AGNI_JD_OUTER", 200, int) == 999
    assert _solver_flag({}, "gpu_lu", "AGNI_GPU_LU") is True

    # neither -> the declared default
    monkeypatch.delenv("AGNI_JD_OUTER")
    monkeypatch.delenv("AGNI_GPU_LU")
    assert _solver_opt({}, "jd_outer", "AGNI_JD_OUTER", 200, int) == 200
    assert _solver_flag({}, "gpu_lu", "AGNI_GPU_LU") is False

    # None is treated as "not supplied", so callers can pass through optionals
    assert _solver_opt({"jd_outer": None}, "jd_outer", "AGNI_JD_OUTER", 200, int) == 200


@pytest.mark.unit
def test_solver_flag_accepts_bools_and_strings():
    """The boolean resolver takes real bools as well as the shell spellings."""
    from desc.compute._stability import _solver_flag

    for truthy in (True, "1", "true", "TRUE", "yes", "on"):
        assert _solver_flag({"f": truthy}, "f", "NOPE") is True, truthy
    for falsy in (False, "0", "false", "no", "off", ""):
        assert _solver_flag({"f": falsy}, "f", "NOPE") is False, falsy
