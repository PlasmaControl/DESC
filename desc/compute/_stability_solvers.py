"""Numerical machinery for the finite-n (AGNI) stability eigensolver.

This module holds the *algorithms* that ``_stability.py``'s compute functions
call for the matrix-free Jacobi-Davidson eigensolve: the block ("ring")
preconditioner, the deflation term built from coarse modes, and the
coarse-to-fine prolongation used to seed and deflate the fine solve.

Nothing here knows about resolution, basis, equilibrium or optimizer settings.
Those are the caller's business: a driver (or a test) picks them, calls in, and
compares the number that comes back against a reference.

Conventions
-----------
``meta``
    Dict describing one discretization level. Required keys: ``n_rho``,
    ``n_theta``, ``n_zeta``, ``n_total``, ``keep`` (indices of retained DOFs in
    the length-``3*n_total`` component-major vector), ``diag``, ``linv_dt``,
    ``inv_linv_dt``, and for the adjoints ``linv_dt_h``, ``inv_linv_dt_h``.
``reduced`` vs ``physical``
    "Reduced" vectors carry only the kept DOFs (length ``n_keep``). "Physical"
    arrays are ``(n_rho, n_theta, n_zeta, 3)``. The two differ by the Dirichlet
    mask *and* by the Cholesky-transform scaling, so they are never interchanged
    implicitly.

Node ordering is rho-major throughout: the flat index of node ``(i, j, k)`` is
``(i * n_theta + j) * n_zeta + k``, and component ``c`` of that node lives at
``c * n_total + ...``. Every index map here assumes it.
"""

import numpy as np

from desc.backend import jax, jnp

__all__ = [
    "apply_space",
    "apply_space_t",
    "barycentric_matrix",
    "build_ring_blocks",
    "coarse_gen_modes",
    "coarse_seed_and_deflation",
    "deflation_Y",
    "finish_ring_block",
    "fourier_interp_matrix",
    "from_phys",
    "from_phys_h",
    "make_block_precond",
    "level_meta",
    "make_transfer",
    "ring_index_maps",
    "ring_nodes",
    "to_phys",
    "to_phys_h",
]


# ---------------------------------------------------------------------------
# Reduced <-> physical
# ---------------------------------------------------------------------------


def _scatter_red(meta, q_red):
    """Reduced vector -> (n_total, 3) node array, zeros on dropped DOFs."""
    full = jnp.zeros((3 * meta["n_total"],), dtype=q_red.dtype)
    full = full.at[meta["keep"]].set(q_red, unique_indices=True)
    return full.reshape(3, meta["n_total"]).T


def _gather_red(meta, qnodes):
    """(n_total, 3) node array -> reduced vector."""
    return qnodes.T.reshape(-1)[meta["keep"]]


def to_phys(meta, q_red):
    """Reduced solver coordinates -> physical ``(n_rho, n_theta, n_zeta, 3)``."""
    qnodes = _scatter_red(meta, q_red)
    u = meta["diag"] * jnp.einsum("nij,nj->ni", meta["linv_dt"], qnodes)
    return u.reshape(meta["n_rho"], meta["n_theta"], meta["n_zeta"], 3)


def from_phys(meta, u_full):
    """Physical field -> reduced solver coordinates. Inverse of `to_phys`."""
    unodes = u_full.reshape(meta["n_total"], 3)
    qnodes = jnp.einsum("nij,nj->ni", meta["inv_linv_dt"], unodes / meta["diag"])
    return _gather_red(meta, qnodes)


def to_phys_h(meta, u_full):
    """Transpose of `to_phys`.

    Needed because the prolongation's adjoint is not its inverse: ``PT`` in
    `make_transfer` must be the true transpose of ``P``.
    """
    unodes = u_full.reshape(meta["n_total"], 3)
    qnodes = jnp.einsum("nij,nj->ni", meta["linv_dt_h"], meta["diag"] * unodes)
    return _gather_red(meta, qnodes)


def from_phys_h(meta, q_red):
    """Transpose of `from_phys`."""
    qnodes = _scatter_red(meta, q_red)
    unodes = jnp.einsum("nij,nj->ni", meta["inv_linv_dt_h"], qnodes) / meta["diag"]
    return unodes.reshape(meta["n_rho"], meta["n_theta"], meta["n_zeta"], 3)


def level_meta(op):
    """Build a level ``meta`` dict from a matrix-free operator's output.

    ``op`` is what ``_agni3_matfree_operator`` returns. The inverses of the
    per-node Cholesky transform and both transposes are formed here once, so
    every transfer call downstream is a pure einsum.
    """
    linv_dt = jnp.asarray(op["Linv_DT"])
    inv_linv_dt = jnp.linalg.inv(linv_dt)
    return dict(
        linv_dt=linv_dt,
        linv_dt_h=jnp.swapaxes(linv_dt, -1, -2),
        inv_linv_dt=inv_linv_dt,
        inv_linv_dt_h=jnp.swapaxes(inv_linv_dt, -1, -2),
        diag=jnp.asarray(op["diagBsqinv"]),
        keep=jnp.asarray(op["keep"]),
        n_total=int(op["n_total"]),
        n_rho=int(op["n_rho"]),
        n_theta=int(op["n_theta"]),
        n_zeta=int(op["n_zeta"]),
        n_keep=int(op["n_keep"]),
    )


# ---------------------------------------------------------------------------
# Prolongation: coarse level -> fine level
# ---------------------------------------------------------------------------


def barycentric_matrix(x_src, x_dst):
    """Barycentric interpolation matrix from ``x_src`` nodes to ``x_dst``.

    Spectrally accurate on the Gauss-Jacobi/Lobatto radial nodes, which is why
    radial transfer uses this rather than linear interpolation: the coarse mode
    has to be represented well enough that it is a useful seed, and a low-order
    radial transfer would inject error exactly where the mode is sharpest.

    Rows where ``x_dst`` coincides with a source node are set to the exact
    delta, avoiding the 0/0 in the barycentric weights.
    """
    x_src = np.asarray(x_src, dtype=float)
    x_dst = np.asarray(x_dst, dtype=float)
    n = x_src.size
    w = np.ones(n)
    for j in range(n):
        w[j] = 1.0 / np.prod(x_src[j] - np.delete(x_src, j))
    mat = np.empty((x_dst.size, n), dtype=float)
    for i, x in enumerate(x_dst):
        hit = np.where(np.isclose(x, x_src, rtol=0.0, atol=1e-14))[0]
        if hit.size:
            mat[i, :] = 0.0
            mat[i, hit[0]] = 1.0
        else:
            tmp = w / (x - x_src)
            mat[i, :] = tmp / np.sum(tmp)
    return mat


def fourier_interp_matrix(n_src, n_dst, period):
    """Exact Fourier interpolation matrix on a uniform periodic grid.

    Used for theta (period ``2*pi``) and zeta (period ``2*pi/NFP``). Exact, not
    approximate: both grids are uniform and periodic, so the trigonometric
    interpolant through the coarse samples reproduces every mode the coarse grid
    can represent.
    """
    x = np.arange(n_src) * (period / n_src)
    y = np.arange(n_dst) * (period / n_dst)
    modes = np.fft.fftfreq(n_src) * n_src
    coeff = np.exp(-1j * np.outer(modes, x)) / n_src
    vals = np.exp(1j * np.outer(y, modes))
    return np.real_if_close(vals @ coeff, tol=1000).real


def apply_space(u, pr, pt, pz):
    """Separable tensor-product interpolation, coarse -> fine."""
    return jnp.einsum("ia,jb,kc,abcq->ijkq", pr, pt, pz, u)


def apply_space_t(u, pr, pt, pz, scale):
    """Transpose of `apply_space`, fine -> coarse."""
    return scale * jnp.einsum("ia,jb,kc,ijkq->abcq", pr, pt, pz, u)


def make_transfer(meta_c, meta_f, pr, pt, pz):
    """Return ``(P, PT)`` as callables on reduced-coordinate vectors.

    ``PT`` is the exact transpose of ``P``, not an inverse and not a
    re-derived restriction (checked by
    ``tests/test_stability_solvers.py::test_prolongation_adjoint_is_exact``).
    """

    def P(q_c):
        return from_phys(meta_f, apply_space(to_phys(meta_c, q_c), pr, pt, pz))

    def PT(q_f):
        return to_phys_h(
            meta_c, apply_space_t(from_phys_h(meta_f, q_f), pr, pt, pz, 1.0)
        )

    return P, PT


# ---------------------------------------------------------------------------
# Block ("ring") preconditioner
# ---------------------------------------------------------------------------


def make_block_precond(L, Gs, n):
    """Build ``M^-1`` from Cholesky factors and the group index map.

    ``M^-1 r`` gathers each group's entries out of ``r``, solves the group's
    Cholesky system, and scatters the result back with ``.add``. Padded slots
    (``Gs == -1``) are zeroed on both the gather and the scatter, so they
    contribute nothing; the gather index for them is clamped to 0 purely to keep
    it in bounds.

    The scatter uses ``.add`` rather than ``.set`` so that overlapping
    partitions would accumulate. For the ring groups of `ring_index_maps` the
    groups are disjoint, so add and set coincide -- but add is the correct
    operation for the additive Schwarz form this is.
    """
    from jax.scipy.linalg import solve_triangular

    Gs = jnp.asarray(Gs)
    mask = (Gs >= 0).astype(jnp.result_type(float))
    idx = jnp.where(Gs >= 0, Gs, 0)

    def M(r):
        y = r[idx] * mask.astype(r.dtype)  # (m, b)
        z = solve_triangular(L, y[..., None], lower=True)
        # RG: L factors H's own ring blocks, which ARE genuinely complex for
        # axisym=True (H carries D_zeta0=1j*n_mode) and Hermitian, not merely
        # symmetric -- measured ||H-H^H||/||H|| ~ 1e-17 vs ||H-H^T||/||H|| ~
        # 3e-3 on this operator. jnp.linalg.cholesky gives H = L L^H, so the
        # back-substitution needs L^H (conjugate transpose), not L^T. This was
        # a plain swapaxes before, silently wrong for every axisym run: for
        # real L (the 3D path) conj is a no-op, so it never showed up there.
        z = solve_triangular(jnp.conj(jnp.swapaxes(L, -1, -2)), z, lower=False)[..., 0]
        z = z * mask.astype(z.dtype)
        return jnp.zeros((n,), dtype=r.dtype).at[idx].add(z)

    return M


# ---------------------------------------------------------------------------
# Coarse generalized eigensolve and the deflation space it supplies
# ---------------------------------------------------------------------------


def coarse_gen_modes(Hc, blocks, Gs, k, num_matvecs, ridge=0.0, seed=3, chunk=2048):
    """Softest ``k`` generalized modes of ``(Hc, M_block)`` on the coarse level.

    Solves the pencil by congruence: with ``M_block = L L^T`` from the block
    Cholesky, ``A = L^-1 Hc L^-T`` is similar to ``M^-1 Hc``, so a standard
    symmetric eigensolve on ``A`` gives the generalized modes, back-transformed
    by ``x = L^-T y``. Shift-invert Lanczos (exact LU on ``A``) targets the
    SOFTEST end, which is the end that matters: those are the modes the fine
    solve struggles with and the ones worth deflating.

    Memory: ``A`` is formed in ONE working copy of ``Hc``, updated in place
    ``chunk`` columns (then rows) at a time inside a single ``lax.scan``, so the
    peak is ``Hc`` + the working copy + the LU factor, plus ``O(n * chunk)``
    per step. Forming it with whole-matrix gathers and scatters kept several
    ``n x n`` temporaries alive and ran out of memory at ``n = 34080``.

    Fully traceable -- safe inside jit, no host round-trips.

    NO RIDGE ESCALATION: choosing a ridge by reading a concrete bool off a
    traced array cannot be traced, so ``ridge`` is a static argument here. Both
    bases measured ridge=0 at 32x32x12. A non-SPD
    block therefore yields NaN rather than silently escalating -- visible in the
    result, which is the safer failure.

    Parameters
    ----------
    Hc : ndarray, (n_c, n_c)
        Symmetric, already shifted by ``-sigma``.
    blocks : ndarray, (m, b, b)
        Coarse block-diagonal of the mass/preconditioner operator.
    Gs : ndarray, (m, b)
        Group index map from `ring_index_maps`. Padding may be ``-1``.
    k, num_matvecs, seed : int
        Static. ``k`` modes retained, ``num_matvecs`` Lanczos steps.
    ridge : float
        Static Cholesky ridge.
    chunk : int
        Static. Columns (rows) transformed per in-place step.

    Returns
    -------
    lam : ndarray, (k,)
        Coarse generalized eigenvalues, ascending (softest first).
    X : ndarray, (n_c, k)
        Unit-norm modes.
    """
    from jax.scipy.linalg import solve_triangular
    from matfree import decomp, eig

    Gs = jnp.asarray(Gs)
    mask = (Gs >= 0).astype(blocks.dtype)
    idx = jnp.where(Gs >= 0, Gs, 0)

    b = Gs.shape[-1]
    eye = jnp.eye(b, dtype=blocks.dtype)[None]
    L = jnp.linalg.cholesky(blocks + ridge * eye)
    mask3 = mask[..., None]

    def blk_solve(Mat, lower):
        """``L^-1 Mat`` (lower) or ``L^-T Mat``, columns batched.

        The groups PARTITION the reduced indices, so ``Mat[idx]`` is a permuted
        copy -- ``(m, b, ncols)`` with ``m*b ~ n`` -- and one batched triangular
        solve covers every block. Works for any ``ncols``: a chunk of columns
        during the reduction, ``ncols=k`` for the back-transform.
        """
        Lu = L if lower else jnp.swapaxes(L, -1, -2)
        Y = Mat[idx] * mask3
        Zb = solve_triangular(Lu, Y, lower=lower) * mask3
        return jnp.zeros_like(Mat).at[idx].add(Zb)

    # A = L^-1 Hc L^-T in place, three passes over one working copy:
    #   0  columns:  A[:, j] <- L^-1 A[:, j]              gives L^-1 Hc
    #   1  rows:     A[i, :] <- (L^-1 A[i, :]^T)^T         gives (L^-1 Hc) L^-T
    #   2  Hermitian part: rows and columns <- 0.5 (A + A^H), chunk by chunk
    # L is the MASS matrix's Cholesky factor and stays real, so L^-H = L^-T and
    # the row pass is the same solve on transposed rows. Taking the Hermitian
    # part after the congruence equals taking it before (the map is linear and
    # L is real), so for axisym=True's complex Hc -- Hermitian to ~1e-17 but
    # not symmetric, ||Hc-Hc^T||/||Hc|| ~ 3e-3 -- the result is the same as
    # symmetrizing Hc first.
    #
    # Passes 0 and 1 must touch every column (row) exactly once. The last
    # step is clamped to start at n - c and masks out the columns an earlier
    # step already did. Pass 2 is idempotent (an entry already equal to the
    # conjugate of its mirror averages to itself), so its clamped step needs
    # no mask.
    n = Hc.shape[0]
    c = int(min(chunk, n))
    n_full = (n // c) * c
    starts = list(range(0, n_full, c))
    firsts = list(starts)
    if n_full < n:
        starts.append(n - c)
        firsts.append(n_full)
    n_steps = len(starts)
    xs = (
        jnp.repeat(jnp.arange(3), n_steps),
        jnp.asarray(starts * 3),
        jnp.asarray(firsts * 3),
    )
    offs = jnp.arange(c)

    def _cols(A, s, first):
        X = jax.lax.dynamic_slice(A, (0, s), (n, c))
        new = (s + offs >= first)[None, :]
        X = jnp.where(new, blk_solve(X, True), X)
        return jax.lax.dynamic_update_slice(A, X, (0, s))

    def _rows(A, s, first):
        X = jax.lax.dynamic_slice(A, (s, 0), (c, n)).T
        new = (s + offs >= first)[None, :]
        X = jnp.where(new, blk_solve(X, True), X)
        return jax.lax.dynamic_update_slice(A, X.T, (s, 0))

    def _herm(A, s, first):
        R = jax.lax.dynamic_slice(A, (s, 0), (c, n))
        C = jax.lax.dynamic_slice(A, (0, s), (n, c))
        M = 0.5 * (R + jnp.conj(C).T)
        A = jax.lax.dynamic_update_slice(A, M, (s, 0))
        return jax.lax.dynamic_update_slice(A, jnp.conj(M).T, (0, s))

    def _step(A, x):
        stage, s, first = x
        return jax.lax.switch(stage, (_cols, _rows, _herm), A, s, first), None

    A, _ = jax.lax.scan(_step, Hc, xs)
    lu = jax.scipy.linalg.lu_factor(A)
    del A
    tri = decomp.tridiag_sym(num_matvecs, reortho="full", materialize=True)
    alg = eig.eigh_partial(tri)
    v0 = jax.random.normal(jax.random.PRNGKey(seed), (n,), dtype=Hc.dtype)
    v0 = v0 / jnp.linalg.norm(v0)
    mu, vecs = alg(lambda rhs: jax.scipy.linalg.lu_solve(lu, rhs), v0)

    lam_all = 1.0 / mu
    order = jnp.argsort(lam_all)[:k]  # ascending: softest first
    lam = lam_all[order]
    X = blk_solve(jnp.swapaxes(vecs[order], 0, 1), False)  # x = L^-T y
    X = X / jnp.linalg.norm(X, axis=0, keepdims=True)
    return lam, X


def coarse_seed_and_deflation(
    Hc,
    blocks_c,
    Gs_c,
    meta_c,
    meta_f,
    pr,
    pt,
    pz,
    k,
    num_matvecs,
    ridge=0.0,
    seed=3,
):
    """Softest coarse generalized modes, prolonged to the fine grid.

    This is what makes the fine solve tractable: the coarse level is small
    enough to solve nearly exactly, and its softest modes -- prolonged -- are
    both a good starting vector and a deflation space that removes the fine
    operator's worst-conditioned directions.

    Returns
    -------
    v0 : ndarray, (n_f,)
        Unit-norm prolonged softest mode; the Jacobi-Davidson start vector.
    Z : ndarray, (n_f, k)
        Prolonged deflation basis. Column 0 is ``v0`` up to scaling.
    lam_c : ndarray, (k,)
        Coarse generalized eigenvalues, for reporting.
    X_c : ndarray, (n_c, k)
        The same modes BEFORE prolongation, unit-norm, on the coarse grid. `Z`
        cannot substitute: it lives in the fine space, so it cannot be paired
        with the coarse operator to form a Rayleigh quotient. Returned so the
        caller -- which is the only place `sigma` is known -- can report a
        coarse eigenvalue directly comparable to the fine level's lambda.
    """
    lam_c, X_c = coarse_gen_modes(
        Hc, blocks_c, Gs_c, k, num_matvecs, ridge=ridge, seed=seed
    )
    P, _ = make_transfer(meta_c, meta_f, pr, pt, pz)
    # X_c is (n_c, k): vmap P over the k columns, then put k back on axis 1.
    Z = jnp.swapaxes(jax.vmap(P)(jnp.swapaxes(X_c, 0, 1)), 0, 1)
    v0 = Z[:, 0]
    v0 = v0 / jnp.linalg.norm(v0)
    return v0, Z, lam_c, X_c


# ---------------------------------------------------------------------------
# Ring block assembly
# ---------------------------------------------------------------------------


def ring_nodes(n_rho, n_theta, n_zeta, i, k):
    """Node indices of the poloidal ring at ``(rho_i, zeta_k)``, rho-major."""
    return np.array(
        [(i * n_theta + j) * n_zeta + k for j in range(n_theta)], dtype=np.int64
    )


def ring_index_maps(keep, res):
    """Static index arrays for the ring build. Grid structure only.

    ``alive`` depends only on which reduced DOFs exist -- the keep mask drops
    ``xi^rho`` on the first and last radial shell -- so it is a property of the
    GRID and can be computed once on the host. That is what turns the per-ring
    masking, a variable-size gather, into a fixed-shape traced gather that vmap
    can batch over all rings at once.

    Returns
    -------
    sel : ndarray, (m, b) int
        Positions WITHIN the ``3*n_theta`` ring ordering that survive the keep
        mask, padded with 0.
    pad : ndarray, (m, b) float
        1.0 on real entries, 0.0 on padding.
    G : ndarray, (m, b) int
        The reduced indices, ``-1`` padded, compacted to the front of each row.
    """
    n_rho, n_theta, n_zeta = res
    n_total = n_rho * n_theta * n_zeta
    keep = np.asarray(keep)
    full_to_red = -np.ones(3 * n_total, dtype=np.int64)
    full_to_red[keep] = np.arange(keep.size)

    raw = []
    for i in range(n_rho):
        for k in range(n_zeta):
            nodes = ring_nodes(n_rho, n_theta, n_zeta, i, k)
            raw.append(
                np.concatenate([full_to_red[c * n_total + nodes] for c in range(3)])
            )
    raw = np.asarray(raw, dtype=np.int64)

    b = int(max((r >= 0).sum() for r in raw))
    m = raw.shape[0]
    sel = np.zeros((m, b), dtype=np.int64)
    pad = np.zeros((m, b))
    G = -np.ones((m, b), dtype=np.int64)
    for gi, r in enumerate(raw):
        pos = np.flatnonzero(r >= 0)
        sel[gi, : pos.size] = pos
        pad[gi, : pos.size] = 1.0
        G[gi, : pos.size] = r[pos]
    return jnp.asarray(sel), jnp.asarray(pad), G


def finish_ring_block(A_blk, Linv, au_diag_blk, n_nodes):
    """Reproduce the assembler tail on one ring: permute, whiten, shift, drive.

    The assembler has already applied the ``d`` symmetric scaling to ``A`` and
    built ``Linv``, so only the node-major permutation, the ``Linv A Linv^T``
    congruence, the 1e-14 shift and the drive diagonal remain. All of those are
    node-diagonal or a permutation, so they restrict to a ring exactly -- which
    is why the ring block equals the corresponding sub-block of the full matrix
    rather than merely approximating it.
    """
    N = n_nodes
    k = jnp.arange(N)
    p = jnp.zeros(3 * N, dtype=jnp.int64)
    p = p.at[3 * k + 0].set(k)
    p = p.at[3 * k + 1].set(N + k)
    p = p.at[3 * k + 2].set(2 * N + k)

    Ap = A_blk[p][:, p].reshape(N, 3, N, 3)
    Ap = jnp.einsum("ikl,iljq,jbq->ikjb", Linv, Ap, Linv)
    node = jnp.arange(N)
    Ap = Ap.at[node, :, node, :].add(1e-14 * jnp.eye(3))
    L0 = Linv[:, :, 0]
    Ap = Ap.at[node, :, node, :].add(
        au_diag_blk[:, None, None] * L0[:, :, None] * L0[:, None, :]
    )
    Ap = Ap.reshape(3 * N, 3 * N)
    pinv = jnp.zeros_like(p).at[p].set(jnp.arange(3 * N))
    return Ap[pinv][:, pinv]


def build_ring_blocks(
    assemble, params, transforms, profiles, data, kwargs, res, sel, pad, sigma, batch=64
):
    """Ring blocks of ``H = A - sigma I``, all rings at once under ``vmap``.

    ``assemble`` is ``_agni3_assemble``; passed in rather than imported to keep
    this module free of any dependency on the compute functions.

    The eager tail

        sub = blk[ix_(alive, alive)];  blocks[gi, :na, :na] = sub - sigma*I
        blocks[gi, t, t] = 1 for t >= na

    is written here, with ``w = pad_i * pad_j``, as

        blocks = sub*w - sigma*diag(pad) + diag(1 - pad)

    which is the same matrix: on real entries it is ``sub - sigma*I``; on padded
    rows ``w = 0`` kills ``sub``, ``diag(pad)`` kills the shift, and
    ``diag(1 - pad)`` leaves the inert identity the padding needs so the
    Cholesky stays defined.
    """
    n_rho, n_theta, n_zeta = res
    m, b = sel.shape
    nodes_all = jnp.asarray(
        np.stack(
            [
                ring_nodes(n_rho, n_theta, n_zeta, i, k)
                for i in range(n_rho)
                for k in range(n_zeta)
            ]
        )
    )

    def one_ring(nodes):
        out = assemble(params, transforms, profiles, data, ring_nodes=nodes, **kwargs)
        return finish_ring_block(out["A"], out["Linv"], out["au_diag"], n_theta)

    # Blocked, not one jax.vmap over all m rings: a full vmap needs
    # O(m*n_total) memory regardless of jit (test_AGNI.py's _Ax_block hit
    # the same wall). batch_size caps it at `batch` rings at a time; memory per
    # ring grows with n_total (48x48x48 at 64 rings: one 62.5 GB allocation).
    blk = jax.lax.map(one_ring, nodes_all, batch_size=min(batch, m))
    rows = sel[:, :, None]
    cols = sel[:, None, :]
    ar = jnp.arange(m)[:, None, None]
    sub = blk[ar, rows, cols]  # (m, b, b)
    # RG: these are H's own ring blocks -- Hermitian, not symmetric, for
    # axisym=True (see make_block_precond). conj() is a no-op for the real 3D
    # case, so this only changes behavior where the plain-transpose symmetrize
    # was actually wrong.
    sub = 0.5 * (sub + jnp.conj(jnp.swapaxes(sub, -1, -2)))
    w = pad[:, :, None] * pad[:, None, :]
    eye = jnp.eye(b, dtype=sub.dtype)[None]
    return sub * w - sigma * (pad[:, :, None] * eye) + (1.0 - pad)[:, :, None] * eye


def factor_ring_blocks_traced(blocks, ridge=0.0):
    """Cholesky at a FIXED ridge, safe under trace.

    Factors once at the given ridge and reports finiteness as a traced flag. A
    non-SPD block (sigma above lambda_min) yields NaN rather than escalating a
    ridge -- visible in the result, which is the safer failure inside a jitted
    solve.
    """
    b = blocks.shape[-1]
    eye = jnp.eye(b, dtype=blocks.dtype)[None]
    L = jnp.linalg.cholesky(blocks + ridge * eye)
    return L, jnp.all(jnp.isfinite(L)), ridge


def deflation_Y(Z, HZ, rcond=1e-12):
    """``Y`` for ``M^-1 = M_ring^-1 + Y Y^T``, fully traced, fixed shape ``(n, k)``.

    The obvious implementation selects surviving directions with BOOLEAN MASKS --
    ``Z[:, live] @ Q[:, keep] / sqrt(w[keep])`` -- which is a variable-size gather
    plus an ``int(keep.sum())`` Python branch. Neither can be traced, so that form
    cannot be used under jit.

    Same result at fixed shape: keep all ``k`` columns and ZERO the rejected ones.
    ``Y Y^T`` is unchanged, because a zero column contributes nothing to the outer
    product.

    Dead directions (``diag(Z^T H Z) <= 0``) are handled by zeroing those COLUMNS
    OF Z before the mixing, so whatever the eigenvectors do with them afterwards
    they multiply a zero column and cannot re-enter ``Y``.

    Returns
    -------
    Y : ndarray, (n, k)
    rank : int
        Number of directions that survived the ``rcond`` cut, as a traced scalar.
    """
    k = Z.shape[1]
    # RG: Z^H H Z, not Z^T H Z -- H is Hermitian, not symmetric, whenever Z/H
    # are complex (axisym=True); conj() is a no-op for real Z/H (3D). This also
    # matters for `dg > 0.0` below: only Z^H H Z has a guaranteed-real diagonal
    # for Hermitian H (z^H H z is real; z^T H z is not, for complex z), so the
    # unfixed form was comparing a not-necessarily-real quantity against 0.0.
    A2 = jnp.conj(jnp.swapaxes(Z, 0, 1)) @ HZ
    A2 = 0.5 * (A2 + jnp.conj(jnp.swapaxes(A2, 0, 1)))
    dg = jnp.diagonal(A2).real
    live = dg > 0.0
    d = jnp.where(live, jnp.sqrt(jnp.where(live, dg, 1.0)), 1.0)
    Hh = (A2 / d[:, None]) / d[None, :]
    eye = jnp.eye(k, dtype=A2.dtype)
    both = live[:, None] & live[None, :]
    # Dead rows/cols become identity so eigh stays well posed. Harmless: the
    # matching columns of Z are zeroed below.
    Hh = jnp.where(both, 0.5 * (Hh + jnp.conj(jnp.swapaxes(Hh, 0, 1))), eye)
    w, Q = jnp.linalg.eigh(Hh)
    keep = w > rcond * jnp.max(w)
    scale = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, w, 1.0)), 0.0)
    Zs = jnp.where(live[None, :], Z / d[None, :], 0.0)
    return (Zs @ Q) * scale[None, :], jnp.sum(keep)
