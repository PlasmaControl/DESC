"""Unit tests for desc/compute/_stability_solvers.py.

These test the in-package solver machinery directly -- no equilibrium, no
subprocess, no scraped stdout. That is the point of having moved it into the
package: a test is now a driver that picks a size, calls in, and compares.

The properties checked here are the ones the deflated solve actually depends on:

* the reduced <-> physical maps round-trip, and their ``_h`` variants are true
  transposes (deflation needs ``P^T``, not ``P^-1``);
* the interpolation matrices are exact where exactness is claimed -- Fourier
  reproduces representable modes exactly, barycentric reproduces polynomials;
* the group partitions tile the DOFs exactly once, for every partition;
* PCG solves an SPD system, and deflation does not change the answer, only the
  work required to reach it.

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
    GROUP_PARTITIONS,
    adjoint_defect,
    barycentric_matrix,
    factor_ring_blocks,
    fourier_interp_matrix,
    from_phys,
    group_index_matrix,
    make_block_precond,
    make_transfer,
    pcg,
    pcg_deflated,
    to_phys,
    transfer_matrices,
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
@pytest.mark.parametrize("partition", GROUP_PARTITIONS)
def test_group_partitions_tile_every_dof_exactly_once(partition):
    """Every kept DOF appears in exactly one group, for every partition.

    A DOF appearing twice would be preconditioned twice (the scatter uses
    ``.add``); a DOF appearing zero times would never be preconditioned at all.
    Either silently degrades CG rather than failing, so it is worth asserting
    for all partitions and not just the production one.
    """
    n_rho, n_theta, n_zeta = 4, 6, 4
    m = _meta(n_rho, n_theta, n_zeta)
    Gs = group_index_matrix(np.asarray(m["keep"]), (n_rho, n_theta, n_zeta), partition)

    live = Gs[Gs >= 0]
    counts = np.bincount(live, minlength=m["n_keep"])
    assert counts.size == m["n_keep"], "group indices exceed the reduced dimension"
    np.testing.assert_array_equal(
        counts, np.ones(m["n_keep"], dtype=counts.dtype)
    ), f"{partition}: DOFs covered {sorted(set(counts.tolist()))} times, expected 1"


@pytest.mark.unit
def test_theta_line_and_shell_block_shapes():
    """theta_line gives 3*n_theta blocks per (rho, zeta); shell gives one per rho.

    These are the two groupings the solver offers. The shapes are what set the
    cost: shell blocks are n_zeta times wider, so their factorization is n_zeta^3
    times more expensive, which is the tradeoff a caller is making when choosing
    it.
    """
    n_rho, n_theta, n_zeta = 4, 6, 4
    keep = np.asarray(_meta(n_rho, n_theta, n_zeta)["keep"])
    res = (n_rho, n_theta, n_zeta)

    g_theta = group_index_matrix(keep, res, "theta_line")
    assert g_theta.shape == (n_rho * n_zeta, 3 * n_theta)

    g_shell = group_index_matrix(keep, res, "shell")
    assert g_shell.shape == (n_rho, 3 * n_theta * n_zeta)


@pytest.mark.unit
def test_prolongation_adjoint_is_exact():
    """PT is the true transpose of P.

    ``shakeout2`` records this at 1.414e-14 on the real operator; here it is
    checked on a synthetic level so the property is tested without an
    equilibrium. If this drifts, the deflated CG is projecting onto a space that
    is not what it thinks it is, and its symmetry argument fails.
    """
    m_c = _meta(3, 4, 2, seed=3)
    m_f = _meta(6, 8, 4, seed=4)
    rho_c = np.linspace(0.05, 0.95, 3)
    rho_f = np.linspace(0.05, 0.95, 6)

    pr, pt, pz = transfer_matrices(rho_c, rho_f, (3, 4, 2), (6, 8, 4), nfp=1)
    P, PT = make_transfer(m_c, m_f, pr, pt, pz)

    defect = adjoint_defect(P, PT, m_c["n_keep"], m_f["n_keep"], trials=6)
    assert defect < 1e-11, f"<Px,y> != <x,PTy>: worst relative defect {defect:.3e}"


@pytest.mark.unit
def test_pcg_solves_spd_system():
    """PCG reaches the direct solution of an SPD system."""
    rng = np.random.default_rng(5)
    n = 60
    B = rng.standard_normal((n, n))
    A = jnp.asarray(B @ B.T + n * np.eye(n))
    b = jnp.asarray(rng.standard_normal(n))

    x, iters, relres = pcg(lambda v: A @ v, b, lambda r: r, tol=1e-12, maxiter=500)
    np.testing.assert_allclose(
        np.asarray(x), np.linalg.solve(np.asarray(A), np.asarray(b)), atol=1e-8
    )
    assert int(iters) <= n + 1, f"CG took {int(iters)} iterations on n={n}"
    assert float(relres) < 1e-11


@pytest.mark.unit
def test_block_precond_reduces_iterations():
    """The block preconditioner cuts CG iterations on a block-dominant system.

    Not a claim about any particular speedup -- just that ``M^-1`` built from the
    exact block diagonal is doing what a preconditioner is supposed to do. If
    this ever regresses to "no better than identity", the group index map or the
    Cholesky apply is wrong.
    """
    n_rho, n_theta, n_zeta = 4, 6, 2
    m = _meta(n_rho, n_theta, n_zeta, seed=6)
    n = m["n_keep"]
    Gs = group_index_matrix(np.asarray(m["keep"]), (n_rho, n_theta, n_zeta))

    rng = np.random.default_rng(7)
    A = np.zeros((n, n))
    for g in Gs:
        live = g[g >= 0]
        blk = rng.standard_normal((live.size, live.size))
        A[np.ix_(live, live)] += blk @ blk.T + live.size * np.eye(live.size)
    A += 0.05 * np.eye(n)
    A = jnp.asarray(0.5 * (A + A.T))

    # Build each block AT ITS Gs POSITIONS rather than assuming a layout.
    # `group_index_matrix` packs live indices to the front and pads with -1 at
    # the end, and `b` is the longest LIVE group -- boundary groups are shorter
    # because the Dirichlet mask drops their xi^rho DOFs. Indexing by position
    # keeps this test correct regardless of where the padding lands. Padded
    # rows/cols get a unit diagonal to keep the Cholesky defined;
    # `make_block_precond` masks their contribution out either way.
    A_np = np.asarray(A)
    b_width = Gs.shape[1]
    blocks = np.zeros((Gs.shape[0], b_width, b_width))
    for gi, g in enumerate(Gs):
        live_pos = np.where(g >= 0)[0]
        live_idx = g[live_pos]
        blocks[gi][np.ix_(live_pos, live_pos)] = A_np[np.ix_(live_idx, live_idx)]
        pad_pos = np.where(g < 0)[0]
        blocks[gi][pad_pos, pad_pos] = 1.0
    blocks = jnp.asarray(blocks)
    L, ok, _ = factor_ring_blocks(blocks)
    assert ok, "block Cholesky failed on an SPD block diagonal"

    M = make_block_precond(L, Gs, n)
    b = jnp.asarray(rng.standard_normal(n))

    _, it_plain, _ = pcg(lambda v: A @ v, b, lambda r: r, tol=1e-10, maxiter=2000)
    x_pc, it_pc, relres = pcg(lambda v: A @ v, b, M, tol=1e-10, maxiter=2000)

    ref = np.linalg.solve(np.asarray(A), np.asarray(b))
    np.testing.assert_allclose(np.asarray(x_pc), ref, rtol=1e-5, atol=1e-7)
    assert int(it_pc) < int(it_plain), (
        f"block preconditioner did not help: {int(it_pc)} vs {int(it_plain)} "
        "unpreconditioned iterations"
    )


@pytest.mark.unit
def test_deflation_preserves_the_solution():
    """Deflating with an exact-invariant subspace changes work, not the answer.

    Deflation is only legitimate if it is a reformulation. Here ``Z`` is built
    from true eigenvectors of ``A``, which is the strongest form of the coarse
    space, and the deflated result must match the direct solve.
    """
    rng = np.random.default_rng(8)
    n, k = 80, 6
    B = rng.standard_normal((n, n))
    A_np = B @ B.T + 0.1 * np.eye(n)
    A = jnp.asarray(A_np)
    b = jnp.asarray(rng.standard_normal(n))

    w, V = np.linalg.eigh(A_np)
    Z = jnp.asarray(V[:, :k])  # the softest modes: what deflation targets

    ref = np.linalg.solve(A_np, np.asarray(b))

    x_d, it_d, relres_d = pcg_deflated(
        lambda v: A @ v, b, lambda r: r, tol=1e-11, maxiter=3000, Z=Z
    )
    np.testing.assert_allclose(np.asarray(x_d), ref, rtol=1e-4, atol=1e-6)

    _, it_plain, _ = pcg(lambda v: A @ v, b, lambda r: r, tol=1e-11, maxiter=3000)
    assert int(it_d) <= int(it_plain), (
        f"deflating the {k} softest modes made CG slower: "
        f"{int(it_d)} vs {int(it_plain)}"
    )


@pytest.mark.unit
def test_pcg_deflated_without_Z_matches_pcg():
    """Z=None is exactly plain PCG, so callers can leave deflation off."""
    rng = np.random.default_rng(9)
    n = 40
    B = rng.standard_normal((n, n))
    A = jnp.asarray(B @ B.T + n * np.eye(n))
    b = jnp.asarray(rng.standard_normal(n))

    x1, k1, r1 = pcg(lambda v: A @ v, b, lambda r: r, tol=1e-12, maxiter=500)
    x2, k2, r2 = pcg_deflated(
        lambda v: A @ v, b, lambda r: r, tol=1e-12, maxiter=500, Z=None
    )
    np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
    assert int(k1) == int(k2)


@pytest.mark.unit
def test_block_cholesky_factorization_residual():
    """L L^T reproduces the blocks it was factored from.

    Replaces the ``AGNI_FACTOR_DIAG >= 2`` diagnostic, which computed
    ``||M - L L^T||_F / ||M||_F`` and printed it. As a test it is strictly more
    useful: the number is checked rather than eyeballed, and it runs without
    anyone remembering to set an environment variable.

    Also pins the ridge behaviour. On an SPD block diagonal the factorization
    must succeed at ridge 0 -- a nonzero ridge means the shift has drifted into
    the spectrum, which is a real problem and not something to absorb silently.
    """
    rng = np.random.default_rng(11)
    m, b = 10, 8
    X = rng.standard_normal((m, b, b))
    blocks = jnp.asarray(np.einsum("mij,mkj->mik", X, X) + b * np.eye(b)[None])

    L, ok, ridge = factor_ring_blocks(blocks)
    assert ok, "Cholesky failed on an SPD block diagonal"
    assert ridge == 0.0, f"needed ridge {ridge:.3e} on SPD blocks; shift is wrong"

    recon = np.einsum("mij,mkj->mik", np.asarray(L), np.asarray(L))
    num = np.linalg.norm(recon - np.asarray(blocks))
    den = np.linalg.norm(np.asarray(blocks))
    rel = num / den
    print(f"\n  ||M - L L^T||_F / ||M||_F = {rel:.3e}")
    assert rel < 1e-13, f"factorization residual {rel:.3e} is too large"


@pytest.mark.unit
def test_indefinite_blocks_are_reported_not_hidden():
    """A block diagonal that is not SPD escalates the ridge and says so.

    The ridge is a MEASUREMENT, not a knob: it says how far the blocks are from
    positive definite, which is how far ``sigma`` has drifted into the spectrum.
    This pins that a non-SPD input does not silently sail through at ridge 0.
    """
    rng = np.random.default_rng(12)
    m, b = 6, 5
    X = rng.standard_normal((m, b, b))
    blocks = np.einsum("mij,mkj->mik", X, X) + b * np.eye(b)[None]
    blocks[0] -= (np.linalg.eigvalsh(blocks[0])[-1] + 1.0) * np.eye(
        b
    )  # force indefinite

    L, ok, ridge = factor_ring_blocks(jnp.asarray(blocks))
    assert ok, "ridge escalation failed to find any workable ridge"
    assert (
        ridge > 0.0
    ), "an indefinite block factored at ridge 0 -- escalation is broken"


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
    ``AGNI_NUM_MATVECS`` exported in 89 places across the job scripts, passing
    ``num_matvecs=`` did nothing.

    These are numerical choices that change the answer, so a caller that passes
    one must get it. The environment stays as a fallback only.
    """
    from desc.compute._stability import _solver_flag, _solver_opt

    monkeypatch.setenv("AGNI_NUM_MATVECS", "999")
    monkeypatch.setenv("AGNI_EIGENSOLVER", "eigsh_callback")
    monkeypatch.setenv("AGNI_RR_REFINE", "1")

    # kwarg present -> kwarg wins, environment ignored
    assert (
        _solver_opt({"num_matvecs": 64}, "num_matvecs", "AGNI_NUM_MATVECS", 50, int)
        == 64
    )
    assert (
        _solver_opt(
            {"eigensolver": "pcg_deflated"}, "eigensolver", "AGNI_EIGENSOLVER", "x"
        )
        == "pcg_deflated"
    )
    assert _solver_flag({"rr_refine": False}, "rr_refine", "AGNI_RR_REFINE") is False

    # kwarg absent -> environment is the fallback
    assert _solver_opt({}, "num_matvecs", "AGNI_NUM_MATVECS", 50, int) == 999
    assert _solver_flag({}, "rr_refine", "AGNI_RR_REFINE") is True

    # neither -> the declared default
    monkeypatch.delenv("AGNI_NUM_MATVECS")
    monkeypatch.delenv("AGNI_RR_REFINE")
    assert _solver_opt({}, "num_matvecs", "AGNI_NUM_MATVECS", 50, int) == 50
    assert _solver_flag({}, "rr_refine", "AGNI_RR_REFINE") is False

    # None is treated as "not supplied", so callers can pass through optionals
    assert (
        _solver_opt({"num_matvecs": None}, "num_matvecs", "AGNI_NUM_MATVECS", 50, int)
        == 50
    )


@pytest.mark.unit
def test_solver_flag_accepts_bools_and_strings():
    """The boolean resolver takes real bools as well as the shell spellings."""
    from desc.compute._stability import _solver_flag

    for truthy in (True, "1", "true", "TRUE", "yes", "on"):
        assert _solver_flag({"f": truthy}, "f", "NOPE") is True, truthy
    for falsy in (False, "0", "false", "no", "off", ""):
        assert _solver_flag({"f": falsy}, "f", "NOPE") is False, falsy
