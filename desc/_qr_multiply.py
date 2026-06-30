"""Pure-JAX ``qr_multiply`` for JAX < 0.10.0.

Implements :func:`jax.scipy.linalg.qr_multiply` without the ``ormqr`` primitive
(which requires a jaxlib rebuild), so it runs on any installed jaxlib. The
Householder reflectors ``Q`` are applied to ``c`` via the blocked CWY/UT
transform of Puglisi (1992) and Joffrain & Low (2006) without ever forming
``Q`` -- on large tall systems this beats forming ``Q`` (``orgqr``) and, on GPU,
cuSOLVER's ``ormqr``.

Ported from https://github.com/jax-ml/jax/pull/36575. The ``DotAlgorithmPreset``
used there only mattered for float32 throughput, which DESC does not use, so
plain matmuls are used here. Block sizes are the upstream A100-tuned values.
"""

from functools import partial

from desc.backend import jax, jit, jnp, solve_triangular, vmap


def _householder_multiply(a, taus, c, *, transpose=False):
    """Apply the reflectors in ``a``/``taus`` to ``c`` from the left, one block.

    Forms ``Q = I - V T^{-1} V^H`` via the identity ``T^{-1} + T^{-H} = V^H V``,
    recovering the triangular ``T^{-1}`` by correcting the diagonal.
    """
    m, k = a.shape
    # V: unit lower-trapezoidal (reflectors below the diagonal, unit diagonal).
    V = jnp.where(jnp.tril(jnp.ones((m, k), bool), -1), a, jnp.eye(m, k, dtype=a.dtype))
    diag_correction = (1 / taus) if transpose else (1 / taus).conj()
    diag_correction = jnp.expand_dims(diag_correction, -1) * jnp.eye(k, dtype=a.dtype)
    Vh = V.conj().swapaxes(-1, -2)
    # solve_triangular reads only the relevant triangle, so passing the full
    # Gram matrix V^H V (minus the diagonal correction) recovers T^{-1}.
    T_inv = Vh @ V - diag_correction
    z = solve_triangular(T_inv, Vh @ c, lower=transpose)
    with jax.default_matmul_precision("highest"):
        return c - V @ z


def _blocked_householder_multiply(a, taus, c, *, left, transpose):
    """Apply ``Q`` (or ``Q^H``) to ``c`` in blocks (block sizes tuned on A100)."""
    if not left:  # c @ Q == (Q^H @ c^H)^H
        ct = c.conj().swapaxes(-1, -2)
        out = _blocked_householder_multiply(
            a, taus, ct, left=True, transpose=not transpose
        )
        return out.conj().swapaxes(-1, -2)

    if a.ndim > 2:  # batch dims via vmap, keep the core logic 2-D
        fn = partial(_blocked_householder_multiply, left=True, transpose=transpose)
        return vmap(fn)(a, taus, c)

    m = a.shape[0]
    k = taus.shape[0]
    if k == 0:  # no reflectors -> Q is the identity
        return c
    # Balances the per-block V^H V cost against the number of sequential blocks.
    esize = a.dtype.itemsize
    hi_limit = 4096 * max(1, 8 // esize)
    mid_limit = 4096 * esize
    nb = min(k, 256 if m <= hi_limit else 128 if m <= mid_limit else 64)

    blocks = range(0, k, nb)
    for j0 in blocks if transpose else reversed(blocks):
        c = c.at[j0:, :].set(
            _householder_multiply(
                a[j0:, j0 : j0 + nb], taus[j0 : j0 + nb], c[j0:, :], transpose=transpose
            )
        )
    return c


@partial(jit, static_argnames="mode")
def qr_multiply(a, c, mode="right"):
    """Pure-JAX drop-in for ``jax.scipy.linalg.qr_multiply``; returns ``(CQ, R)``.

    ``a = Q @ R`` is the (economic) QR factorization. For ``mode="right"``
    returns ``c @ Q`` (with 1-D ``c`` treated as a length-``M`` row vector); for
    ``mode="left"`` returns ``Q @ c``. ``R`` has shape ``(min(M, N), N)``.
    """
    m, n = a.shape
    k = min(m, n)
    # mode="raw" returns the packed reflectors (transposed) plus tau factors,
    # via the existing geqrf primitive -- no new primitive / jaxlib rebuild.
    h, taus = jnp.linalg.qr(a, mode="raw")
    packed = h.swapaxes(-1, -2)  # (M, N): lower triangle = reflectors, upper = R
    R = jnp.triu(packed)[:k, :]
    # When m <= n geqrf's last reflector (row m-1) is the identity (tau == 0).
    # Drop it statically -- the economic Q is unchanged and we avoid 1/0.
    n_refl = k - 1 if m <= n else k
    V = packed[:, :n_refl]
    taus = taus[:n_refl]
    c1d = c.ndim == 1

    if mode == "right":
        cq = _blocked_householder_multiply(
            V, taus, c[None, :] if c1d else c, left=False, transpose=False
        )
        cq = cq[:, :k]  # economic Q has min(M, N) columns
        return (cq[0] if c1d else cq), R

    C = c[:, None] if c1d else c
    pad = jnp.zeros((m - k, C.shape[1]), C.dtype)
    cq = _blocked_householder_multiply(
        V, taus, jnp.vstack([C, pad]), left=True, transpose=False
    )
    return (cq[:, 0] if c1d else cq), R
