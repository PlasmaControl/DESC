"""Runtime validation script for finite-n lambda3 reconstruction equivalence.

This script runs a small real-equilibrium solve and compares old vs new
reconstruction paths inside `_AGNI3` using `debug_compare_reconstruction=True`.

Usage:
    source /pscratch/sd/r/rgaur/use_desc_env3.sh
    PYTHONPATH=/pscratch/sd/r/rgaur/DESC2/DESC:$PYTHONPATH \
    JAX_PLATFORM_NAME=cpu \
    python tests/lambda3_runtime_compare.py
"""

import os

import numpy as np

from desc.backend import jax, jnp
from desc.diffmat_utils import DiffMat, fourier_diffmat, legendre_diffmat
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import map_coordinates
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob


def main():
    """Run a small lambda3 reconstruction comparison."""
    # Prefer the same equilibrium used in the matrix-full lambda3 job.
    # Override with LAMBDA3_RUNTIME_EQ if desired.
    eq_path = os.environ.get(
        "LAMBDA3_RUNTIME_EQ",
        "/pscratch/sd/r/rgaur/AGNI_var/matrix-free/"
        "qh_beta1.5_imin1.02_modprof_221410.h5",
    )
    if os.path.exists(eq_path):
        eq = Equilibrium.load(eq_path)
    else:
        eq = get("precise_QA")

    n_rho, n_theta, n_zeta = 8, 8, 6

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

    data = eq.compute(
        "finite-n lambda3",
        grid=grid,
        diffmat=diffmat,
        incompressible=False,
        gamma=5.0 / 3.0,
        debug_compare_reconstruction=True,
    )

    au_rel = float(np.asarray(data["finite-n lambda3 debug au_relerr"]))
    au_max = float(np.asarray(data["finite-n lambda3 debug au_maxabs"]))
    xi_rel = float(np.asarray(data["finite-n lambda3 debug xi_relerr"]))
    xi_max = float(np.asarray(data["finite-n lambda3 debug xi_maxabs"]))

    print("lambda3:", np.asarray(data["finite-n lambda3"]))
    print("Au relerr:", au_rel)
    print("Au maxabs:", au_max)
    print("xi relerr:", xi_rel)
    print("xi maxabs:", xi_max)

    assert au_rel < 1e-10, f"Au relative error too large: {au_rel}"
    assert au_max < 1e-10, f"Au absolute error too large: {au_max}"
    assert xi_rel < 1e-10, f"xi relative error too large: {xi_rel}"
    assert xi_max < 1e-10, f"xi absolute error too large: {xi_max}"


if __name__ == "__main__":
    main()
