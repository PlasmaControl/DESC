"""Compare reduced solve matrices from finite-n lambda3 and finite-n lambda31.

This script captures the actual matrix handed to `eigsh` by each compute function,
without modifying the compute-function internals.

Usage:
    source /pscratch/sd/r/rgaur/use_desc_env3.sh
    PYTHONPATH=/pscratch/sd/r/rgaur/DESC2/DESC:$PYTHONPATH \
    JAX_PLATFORM_NAME=cpu \
    python tests/lambda3_lambda31_matrix_compare.py
"""

import os

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.compute import _stability as stability
from desc.diffmat_utils import DiffMat, fourier_diffmat, legendre_diffmat
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import map_coordinates
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob


def _build_grid_and_diffmat(eq, n_rho=8, n_theta=8, n_zeta=6):
    x, _ = leggauss_lob(n_rho)
    rho = automorphism_staircase1(x, eps=1e-2, x_0=0.65, m_1=2.1, m_2=3.0)
    dx_f = jax.vmap(
        lambda x_val: jax.grad(automorphism_staircase1, argnums=0)(
            x_val, eps=1e-2, x_0=0.65, m_1=2.1, m_2=3.0
        )
    )
    scale_vector = 1 / (dx_f(x)[:, None])
    scale_vector_inv = dx_f(x)[:, None]

    d_rho, w_rho = legendre_diffmat(n_rho)
    d_rho = d_rho * scale_vector
    w_rho = w_rho * scale_vector_inv

    theta = jnp.linspace(0.0, 2 * jnp.pi, n_theta, endpoint=False)
    d_theta, w_theta = fourier_diffmat(n_theta)

    zeta = jnp.linspace(0.0, 2 * jnp.pi / eq.NFP, n_zeta, endpoint=False)
    d_zeta, w_zeta = fourier_diffmat(n_zeta)
    d_zeta = d_zeta * eq.NFP
    w_zeta = w_zeta / eq.NFP

    diffmat = DiffMat(
        D_rho=d_rho,
        W_rho=jnp.diagonal(w_rho),
        D_theta=d_theta,
        W_theta=jnp.diagonal(w_theta),
        D_zeta=d_zeta,
        W_zeta=jnp.diagonal(w_zeta),
    )

    grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)
    reshaped_nodes = jnp.reshape(
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_rho * n_theta * n_zeta, 3)
    )
    rtz_nodes = map_coordinates(
        eq,
        reshaped_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )
    grid = Grid(rtz_nodes)
    return grid, diffmat


def _capture_reduced_matrix(eq, name, grid, diffmat):
    original_eigsh = stability.eigsh
    captured = []

    def _fake_eigsh(A, *args, **kwargs):
        mat = np.asarray(A)
        captured.append(mat.copy())
        n = mat.shape[0]
        vec = np.ones((n, 1), dtype=mat.dtype)
        vec /= np.linalg.norm(vec)
        val = np.array([0.0], dtype=np.float64)
        return val, vec

    stability.eigsh = _fake_eigsh
    try:
        eq.compute(
            name,
            grid=grid,
            diffmat=diffmat,
            incompressible=False,
            gamma=5.0 / 3.0,
        )
    finally:
        stability.eigsh = original_eigsh

    if len(captured) != 1:
        raise RuntimeError(
            f"Expected exactly one eigsh call for {name}, got {len(captured)}"
        )
    return captured[0]


@pytest.mark.unit
@pytest.mark.slow
def test_lambda3_lambda31_matrix_compare():
    """Compare captured lambda3 and lambda31 reduced matrices."""
    eq_path = os.environ.get(
        "LAMBDA3_RUNTIME_EQ",
        "/pscratch/sd/r/rgaur/AGNI_var/matrix-free/"
        "qh_beta1.5_imin1.02_modprof_221410.h5",
    )
    if os.path.exists(eq_path):
        eq = Equilibrium.load(eq_path)
    else:
        eq = get("precise_QA")

    grid, diffmat = _build_grid_and_diffmat(eq)
    a3 = _capture_reduced_matrix(eq, "finite-n lambda3", grid, diffmat)
    a31 = _capture_reduced_matrix(eq, "finite-n lambda31", grid, diffmat)

    if a3.shape != a31.shape:
        raise AssertionError(
            f"Shape mismatch: lambda3={a3.shape}, lambda31={a31.shape}"
        )

    finite = np.isfinite(a3) & np.isfinite(a31)
    finite_frac = float(np.mean(finite))
    if np.any(finite):
        d = a3[finite] - a31[finite]
        rel = float(np.linalg.norm(d) / (np.linalg.norm(a3[finite]) + 1e-300))
        maxabs = float(np.max(np.abs(d)))
    else:
        rel = np.nan
        maxabs = np.nan

    allclose = bool(np.allclose(a3, a31, rtol=1e-8, atol=1e-10, equal_nan=True))

    print("lambda3 shape:", a3.shape)
    print("lambda31 shape:", a31.shape)
    print("finite overlap fraction:", finite_frac)
    print("relative diff (finite entries):", rel)
    print("max abs diff (finite entries):", maxabs)
    print("allclose(equal_nan=True):", allclose)

    # This is the actual correctness condition for this comparison test.
    assert allclose, (
        "finite-n lambda3 and finite-n lambda31 reduced matrices differ: "
        f"rel_diff={rel}, maxabs_diff={maxabs}, finite_overlap={finite_frac}"
    )
