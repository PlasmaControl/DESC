"""AGNI regression tests at moderate resolution for the lambda3 matrix-free path."""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import eigsh as scipy_eigsh

from desc.backend import jax, jnp
from desc.compute import _stability as stability
from desc.diffmat_utils import DiffMat, fourier_diffmat, legendre_diffmat
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid
from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob


def _load_old_equilibrium(path):
    """Load AGNI fixture equilibria saved before newer DESC attrs existed."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*The object attribute .* was not loaded from the file\.",
            category=RuntimeWarning,
        )
        return Equilibrium.load(path)


RES = os.environ.get("AGNI2_TEST_RES", "12,12,12")
N_RHO, N_THETA, N_ZETA = (int(v.strip()) for v in RES.split(","))

EQ_PATH = Path(
    os.environ.get(
        "AGNI_EQ_PATH",
        "/pscratch/sd/r/rgaur/AGNI_var/matrix-free/"
        "qh_beta1.5_imin1.02_modprof_221410.h5",
    )
)
AGNI_SKIP_REASON = f"AGNI equilibrium fixture not found: {EQ_PATH}"
pytestmark = pytest.mark.skipif(not EQ_PATH.is_file(), reason=AGNI_SKIP_REASON)

if EQ_PATH.is_file():
    EQ = _load_old_equilibrium(str(EQ_PATH))

    x, _ = leggauss_lob(N_RHO)
    rho = automorphism_staircase1(x, eps=1e-2, x_0=0.7, m_1=2.0, m_2=3.0)
    dx_f = jax.vmap(
        lambda x_val: jax.grad(automorphism_staircase1, argnums=0)(
            x_val, eps=1e-2, x_0=0.7, m_1=2.0, m_2=3.0
        )
    )
    scale_vector = 1.0 / (dx_f(x)[:, None])
    scale_vector_inv = dx_f(x)[:, None]

    d_rho, w_rho = legendre_diffmat(N_RHO)
    d_rho = d_rho * scale_vector
    w_rho = w_rho * scale_vector_inv

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, N_THETA, endpoint=False)
    d_theta, w_theta = fourier_diffmat(N_THETA)

    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / EQ.NFP, N_ZETA, endpoint=False)
    d_zeta, w_zeta = fourier_diffmat(N_ZETA)
    d_zeta = d_zeta * EQ.NFP
    w_zeta = w_zeta / EQ.NFP

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
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"),
        (N_RHO * N_THETA * N_ZETA, 3),
    )
    rtz_nodes = EQ.map_coordinates(
        reshaped_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )
    grid = Grid(rtz_nodes)

    N_TOTAL = N_RHO * N_THETA * N_ZETA
    N_SHELL = N_THETA * N_ZETA
    N_KEEP = 3 * N_TOTAL - 2 * N_SHELL
    V_GUESS = np.ones(N_KEEP)
else:
    EQ = grid = diffmat = None
    N_TOTAL = N_SHELL = N_KEEP = 0
    V_GUESS = None


def dense_eigsh(a, *args, **kwargs):
    """Use explicit ARPACK settings for dense matrix AGNI solves."""
    matrix = np.asarray(a)
    matrix = 0.5 * (matrix + matrix.T.conj())
    n = matrix.shape[0]
    v0 = kwargs.get("v0", None)
    if v0 is None or np.asarray(v0).size != n:
        v0 = np.ones(n)
    k = int(kwargs.get("k", 1))
    which = kwargs.get("which", "LM")
    tol = kwargs.get("tol", 1e-10)
    tol = 1e-10 if tol is None else float(tol)
    maxiter = kwargs.get("maxiter", 25000)
    maxiter = 25000 if maxiter is None else int(maxiter)
    sigma = kwargs.get("sigma", -1e-5)
    sigma = -1e-5 if sigma is None else float(sigma)
    ncv = min(n - 1, max(80, 8 * k + 1))
    return_eigenvectors = bool(kwargs.get("return_eigenvectors", True))

    try:
        return scipy_eigsh(
            matrix,
            k=k,
            sigma=sigma,
            which=which,
            v0=np.asarray(v0).reshape(-1),
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
            return_eigenvectors=return_eigenvectors,
        )
    except ArpackNoConvergence:
        return scipy_eigsh(
            matrix,
            k=k,
            which="SA",
            v0=np.asarray(v0).reshape(-1),
            tol=max(tol, 1e-8),
            maxiter=maxiter,
            ncv=ncv,
            return_eigenvectors=return_eigenvectors,
        )


@pytest.mark.unit
@pytest.mark.slow
def test_lambda3_matfree():
    """Dense finite-n lambda3 eigenpair satisfies matrix-free operator consistency."""
    old_solver = stability.eigsh
    stability.eigsh = dense_eigsh
    try:
        dense3 = EQ.compute(
            "finite-n lambda3",
            grid=grid,
            diffmat=diffmat,
            incompressible=False,
            gamma=5.0 / 3.0,
            v_guess=V_GUESS,
        )
    finally:
        stability.eigsh = old_solver

    matfree3 = EQ.compute(
        "finite-n lambda3 matfree",
        grid=grid,
        diffmat=diffmat,
        incompressible=False,
        gamma=5.0 / 3.0,
        v_guess=np.asarray(dense3["finite-n eigenfunction3"]),
        check_v_guess_only=True,
        lambda_guess=float(np.asarray(dense3["finite-n lambda3"])[0]),
    )

    np.testing.assert_allclose(
        np.asarray(matfree3["finite-n lambda3 matfree"]),
        np.asarray(dense3["finite-n lambda3"]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert float(matfree3["finite-n lambda3 matfree check relative_residual"]) < 5e-1
    assert float(matfree3["finite-n lambda3 matfree check rayleigh_residual"]) < 5e-1
