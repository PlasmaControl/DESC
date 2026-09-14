"""Regression tests for finite-n lambda3 algebraic reconstructions.

These tests validate two lambda3 implementation optimizations:
1) reconstructing transformed Au from only the scaled rho-rho diagonal term,
2) reconstructing xi without materializing full LinvT_full.
"""

import numpy as np
import pytest


def _component_to_node_permutation(n_total):
    """Permutation from component-major to node-major ordering."""
    k = np.arange(n_total, dtype=np.int64)
    p = np.empty(3 * n_total, dtype=np.int64)
    p[3 * k + 0] = k
    p[3 * k + 1] = n_total + k
    p[3 * k + 2] = 2 * n_total + k
    return p


def _assemble_diagblocks_comp_major(blocks, rho_idx, theta_idx, zeta_idx, sym=False):
    """Assemble block-diagonal (component-major) matrix from (N,3,3) blocks."""
    n_total = blocks.shape[0]
    big = np.zeros((3 * n_total, 3 * n_total), dtype=blocks.dtype)

    big[rho_idx, rho_idx] = np.diag(blocks[:, 0, 0])
    big[theta_idx, theta_idx] = np.diag(blocks[:, 1, 1])
    big[zeta_idx, zeta_idx] = np.diag(blocks[:, 2, 2])

    big[theta_idx, rho_idx] = np.diag(blocks[:, 1, 0])
    big[zeta_idx, rho_idx] = np.diag(blocks[:, 2, 0])
    big[zeta_idx, theta_idx] = np.diag(blocks[:, 2, 1])

    if sym:
        big[rho_idx, theta_idx] = np.diag(blocks[:, 0, 1])
        big[rho_idx, zeta_idx] = np.diag(blocks[:, 0, 2])
        big[theta_idx, zeta_idx] = np.diag(blocks[:, 1, 2])

    return big


def _random_lower_tri_blocks(n_total, rng, dtype=float):
    """Generate random lower-triangular per-node 3x3 blocks with nonzero diagonals."""
    blocks = np.zeros((n_total, 3, 3), dtype=dtype)
    for i in range(n_total):
        m = rng.normal(size=(3, 3))
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            m = m + 1j * rng.normal(size=(3, 3))
        l = np.tril(m)
        diag = np.abs(np.diag(l)) + 0.5
        l[np.diag_indices(3)] = diag
        blocks[i] = l.astype(dtype)
    return blocks


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [float, complex])
def test_lambda3_au_diag_reconstruction_matches_full_path(dtype):
    """Old full-Au path and new diagonal-only path should produce identical Au."""
    rng = np.random.default_rng(20260409)
    n_total = 23

    rho_idx = slice(0, n_total)

    au_raw = rng.normal(size=n_total)
    d = 0.5 + rng.random(3 * n_total)
    linv = _random_lower_tri_blocks(n_total, rng, dtype=dtype)

    # old way: materialize full Au, scale, permute, transform, unpermute
    au_full = np.zeros((3 * n_total, 3 * n_total), dtype=dtype)
    au_full[rho_idx, rho_idx] = np.diag(au_raw)
    au_full = d[:, None] * au_full * d[None, :]

    p = _component_to_node_permutation(n_total)
    pinv = np.empty_like(p)
    pinv[p] = np.arange(3 * n_total)

    au_node = au_full[p][:, p].reshape(n_total, 3, n_total, 3)
    au_old = np.einsum("ikl,iljq,jbq->ikjb", linv, au_node, linv).reshape(
        3 * n_total, 3 * n_total
    )
    au_old = au_old[pinv][:, pinv]

    # new way: keep only scaled rho-rho diagonal and reconstruct transformed Au
    au_diag = d[rho_idx] ** 2 * au_raw
    l0 = linv[:, :, 0]
    au_new_node = np.zeros((n_total, 3, n_total, 3), dtype=dtype)
    idx = np.arange(n_total)
    au_new_node[idx, :, idx, :] = (
        au_diag[:, None, None] * l0[:, :, None] * l0[:, None, :]
    )
    au_new = au_new_node.reshape(3 * n_total, 3 * n_total)
    au_new = au_new[pinv][:, pinv]

    np.testing.assert_allclose(au_new, au_old, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_lambda3_xi_reconstruction_matches_linvt_full_path():
    """New xi reconstruction should match old LinvT_full-based reconstruction."""
    rng = np.random.default_rng(20260410)

    n_rho_max, n_theta_max, n_zeta_max = 6, 2, 3
    n_total = n_rho_max * n_theta_max * n_zeta_max

    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)

    linv = _random_lower_tri_blocks(n_total, rng, dtype=float)

    # Match lambda3 BC setup: remove rho-(theta,zeta) couplings on radial boundaries.
    n_per_shell = n_theta_max * n_zeta_max
    node_ids = np.arange(n_total)
    rho_shell = node_ids // n_per_shell
    boundary = (rho_shell == 0) | (rho_shell == (n_rho_max - 1))
    linv[boundary, 1, 0] = 0.0
    linv[boundary, 2, 0] = 0.0

    d = 0.5 + rng.random(3 * n_total)

    rho_start = n_per_shell
    rho_end = n_total - n_per_shell
    keep_1 = np.arange(rho_start, rho_end)
    keep_2 = np.arange(n_total, 3 * n_total)
    keep = np.concatenate([keep_1, keep_2])

    v_mode = rng.normal(size=keep.size)

    # old way: LinvT_full materialization
    linvt_full = _assemble_diagblocks_comp_major(linv, rho_idx, theta_idx, zeta_idx).T
    linvt_red = linvt_full[np.ix_(keep, keep)]
    d_red = d[keep]
    xi_red_old = d_red * (linvt_red @ v_mode)
    xi_full_old = np.zeros(3 * n_total)
    xi_full_old[keep] = xi_red_old

    # new way: direct nodewise reconstruction from Linv and full reduced eigenvector
    v_full = np.zeros(3 * n_total)
    v_full[keep] = v_mode
    vr, vv, vz = v_full[rho_idx], v_full[theta_idx], v_full[zeta_idx]

    xi_full_new = np.concatenate(
        [
            d[rho_idx] * (linv[:, 0, 0] * vr + linv[:, 1, 0] * vv + linv[:, 2, 0] * vz),
            d[theta_idx] * (linv[:, 1, 1] * vv + linv[:, 2, 1] * vz),
            d[zeta_idx] * (linv[:, 2, 2] * vz),
        ]
    )

    np.testing.assert_allclose(xi_full_new, xi_full_old, rtol=1e-12, atol=1e-12)
