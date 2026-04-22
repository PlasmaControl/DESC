import pdb
import time
from desc.examples import get
from desc.grid import LinearGrid, Grid
from desc.diffmat_utils import *
from desc.equilibrium.coords import map_coordinates
from desc.equilibrium import Equilibrium
from desc.backend import jnp, jax
from matplotlib import pyplot as plt
from desc.utils import dot, apply
from desc.integrals.quad_utils import leggauss_lob, automorphism_staircase1
from desc.io import load
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# Make or load an ultra high aspect-ratio tokamak (essentially a screw pinch)
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.grid import QuadratureGrid
import os

from newcomb import *


# helper functions
def nodes_and_diffmats(n_rho, n_theta, n_zeta, NFP):
    x, w = leggauss_lob(n_rho)

    rho = automorphism_staircase1(x, eps=1e-2, x_0=0.5, m_1=2.0, m_2=2.0)
    dx_f = jax.vmap(
        lambda x_val: jax.grad(automorphism_staircase1, argnums=0)(
            x_val, eps=1e-2, x_0=0.5, m_1=2.0, m_2=2.0
        )
    )

    scale_vector = 1 / (dx_f(x)[:, None])
    scale_vector_inv = dx_f(x)[:, None]

    D0, W0 = legendre_diffmat(n_rho)

    # scaled D_rho
    D0 = D0 * scale_vector
    W0 = W0 * scale_vector_inv

    theta = jnp.linspace(0.0, 2 * jnp.pi, n_theta, endpoint=False)
    D1, W1 = fourier_diffmat(n_theta)

    zeta = jnp.linspace(0.0, 2 * jnp.pi / NFP, n_zeta, endpoint=False)
    D2, W2 = fourier_diffmat(n_zeta)

    W0 = jnp.diag(W0)
    W1 = jnp.diag(W1)
    W2 = jnp.diag(W2)

    diffmat = DiffMat(D_rho=D0, W_rho=W0, D_theta=D1, W_theta=W1, D_zeta=D2, W_zeta=W2)


    return diffmat, rho, theta, zeta

def mapping_and_grid(eq, rho, theta, zeta):
    n_rho = rho.shape[0]
    n_theta = theta.shape[0]
    n_zeta = zeta.shape[0]

    # rho, theta, zeta are in PEST coordinates
    grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)

    # reshaped according to rho
    reshaped_nodes = jnp.reshape(
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_rho * n_theta * n_zeta, 3)
    )

    # These nodes are in DESC coordinates
    rtz_nodes = map_coordinates(
        eq,
        reshaped_nodes,  # (ρ,θ_PEST,ζ)
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )

    # These nodes are in DESC coordinates
    grid = Grid(rtz_nodes)

    return grid, reshaped_nodes



def add_bc(X, n_rho, n_theta, n_zeta):
    # indices for slicing the input vector X into the three components
    idx0 = n_rho * n_theta * n_zeta
    idx1 = 2 * idx0

    xi_sup_rho = np.reshape(X[:idx0], (n_rho, n_theta, n_zeta))
    xi_sup_rho   = np.concatenate((xi_sup_rho,   xi_sup_rho[:, 0:1, :]),   axis=1)
    xi_sup_rho   = np.concatenate((xi_sup_rho,   xi_sup_rho[:, :, 0:1]),   axis=2)

    xi_sup_theta0 = np.reshape(X[idx0:idx1], (n_rho, n_theta, n_zeta))
    xi_sup_zeta0  = np.reshape(X[idx1:], (n_rho, n_theta, n_zeta))

    xi_sup_theta = np.concatenate((xi_sup_theta0, xi_sup_theta0[:, 0:1, :]), axis=1)
    xi_sup_theta = np.concatenate((xi_sup_theta,  xi_sup_theta[:, :, 0:1]), axis=2)

    xi_sup_zeta  = np.concatenate((xi_sup_zeta0,  xi_sup_zeta0[:, :, 0:1]), axis=2)
    xi_sup_zeta  = np.concatenate((xi_sup_zeta,   xi_sup_zeta[:, 0:1, :]),  axis=1)

    xi_rho_low   = np.asarray(xi_sup_rho)
    xi_theta_low = np.asarray(xi_sup_theta)
    xi_zeta_low  = np.asarray(xi_sup_zeta)

    return xi_rho_low, xi_theta_low, xi_zeta_low


# -------------------------
# UPSCALE: 3D interpolation on (rho,theta,zeta),
# periodic extension in theta/zeta, NO FFT.
# -------------------------
def interpolate_xi(xi_rho_low, xi_theta_low, xi_zeta_low, rho_low, theta_low, zeta_low, reshaped_nodes, n_rho, n_theta, n_zeta, NFP):
    # add boundaries for interpolation later
    theta_low = np.concatenate((theta_low, np.array([2 * np.pi])))
    zeta_low  = np.concatenate((zeta_low,  np.array([2 * np.pi / NFP])))

    def _interp3_periodic(f_ext, pts):

        i0 = RegularGridInterpolator(
            (rho_low, theta_low, zeta_low),
            f_ext,
            method="linear",
            #method="pchip",
            bounds_error=False,
            fill_value=None,
        )
        out = i0(pts)

        return out.reshape(n_rho, n_theta, n_zeta)

    xi_rho_hi = _interp3_periodic(xi_rho_low, reshaped_nodes)
    xi_theta_hi = _interp3_periodic(xi_theta_low, reshaped_nodes)
    xi_zeta_hi = _interp3_periodic(xi_zeta_low, reshaped_nodes)

    # Normalizing doesn't improves convergence.

    v_guess = jnp.concatenate(
        [
            (xi_rho_hi).flatten(),
            (xi_theta_hi).flatten(),
            (xi_zeta_hi).flatten(),
        ],
        axis=0,
    )
    v_guess = v_guess/jnp.linalg.norm(v_guess)

    # store indices needed to apply dirichlet BC to ξ^ρ
    n_total = n_rho * n_theta * n_zeta
    n_shell = n_theta * n_zeta
    rho_start = n_shell
    rho_end = n_total - n_shell
    keep_1 = jnp.arange(rho_start, rho_end)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    return v_guess[keep], xi_rho_hi, xi_theta_hi, xi_zeta_hi


# ── Plotting ───────────────────────────────────────────────────────────────────
rho_colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
labels      = [r"$\xi^\rho$", r"$\xi^\theta$", r"$\xi^\zeta$"]
comp_colors = ["steelblue", "darkorange", "seagreen"]


def save_eigenfunction_plots(xi_rho, xi_theta, xi_zeta, rho, theta, rho_iota1, title_base, tag):
    """Save 4 eigenfunction diagnostic plots tagged with `tag`.

    Works for both the raw-solve arrays (shape n_rho × (n_theta+1) × (n_zeta+1),
    from add_bc) and the interpolated arrays (shape n_rho × n_theta × n_zeta).
    Uses `len(theta)` as the authoritative theta length so the trailing boundary
    column is automatically ignored when present.
    """
    n_rho_loc = xi_rho.shape[0]
    n_theta_loc = len(theta)
    rho_plot_indices = np.linspace(n_rho_loc // 4, n_rho_loc - 1, 4, dtype=int)
    i_theta_half_pi = int(np.argmin(np.abs(theta - np.pi / 2)))

    def _plot_vs_rho(i_theta, theta_label, fname_suffix):
        fig, ax = plt.subplots(figsize=(7, 5))
        for xi, label, color in zip([xi_rho, xi_theta, xi_zeta], labels, comp_colors):
            ax.plot(rho, xi[:, i_theta, 0], color=color, lw=2, label=label)
        if rho_iota1 is not None and rho_iota1 <= 1.0:
            ax.axvline(rho_iota1, color="red", lw=1.2, ls=":", label=r"$\iota=1$")
        ax.set_xlabel(r"$\rho$", fontsize=14)
        ax.set_ylabel(r"$\xi$ (arb.)", fontsize=14)
        ax.set_title(title_base + f",  {theta_label}", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=13)
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        plt.tight_layout()
        fig.savefig(plot_path + f"{fname_suffix}_{tag}.png", dpi=150)
        plt.close(fig)

    def _plot_vs_theta(rho_indices, suptitle_extra, fname_suffix):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for ax, xi, label in zip(axes, [xi_rho, xi_theta, xi_zeta], labels):
            for i_rho, color in zip(rho_indices, rho_colors):
                ax.plot(
                    theta, xi[i_rho, :n_theta_loc, 0],
                    color=color, lw=2,
                    label=rf"$\rho = {rho[i_rho]:.2f}$",
                )
            ax.set_xlabel(r"$\theta$", fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
            ax.tick_params(labelsize=11)
            ax.axhline(0, color="gray", lw=0.7, ls="--")
            ax.legend(fontsize=10)
        fig.suptitle(title_base + suptitle_extra, fontsize=13)
        plt.tight_layout()
        fig.savefig(plot_path + f"{fname_suffix}_{tag}.png", dpi=150)
        plt.close(fig)

    _plot_vs_rho(0, r"$\theta=0$", "xi_vs_rho_theta0")
    _plot_vs_rho(
        i_theta_half_pi,
        rf"$\theta \approx \pi/2$  ($\theta={theta[i_theta_half_pi]:.3f}$)",
        "xi_vs_rho_theta_half_pi",
    )
    _plot_vs_theta(rho_plot_indices, "", "xi_vs_theta")
    _plot_vs_theta(
        [n_rho_loc - 1],
        rf"  —  $\rho = {rho[-1]:.3f}$ (boundary BC check)",
        "xi_vs_theta_bc",
    )
