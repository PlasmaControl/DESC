# A script to set up and solve a high aspect-ratio tokamak equilibrium using DESC,
# and then evaluate its stability using Newcomb's procedure.
#from desc import set_device
#set_device("gpu")

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

# Input parameters
a = 1  # Minor radius
aspect_ratio = 10  # Aspect ratio of the tokamak
R = aspect_ratio * a  # Major radius
NFP = 1
axisym = False  # Whether to enforce axisymmetry in the eigenvalue solve
n_mode_axisym = 1  # If axisym is True, the toroidal mode number to solve for


# Paths
save_path = "./eigenvalue_solve/"
plot_path   = save_path + "eigenfunction_plots/"
os.makedirs(save_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

# Quadratic iota profile: iota(rho) = iota_0 - 0.05*rho^2
# => d^2 iota / d rho^2 = -0.1 (decreasing, as requested)
iota_on_axis_values = np.linspace(0.8, 1.25, 10)

results_lambda_min = np.zeros_like(iota_on_axis_values)
stabilities = np.zeros_like(iota_on_axis_values, dtype=bool)


# helper functions
def nodes_and_diffmats(n_rho, n_theta, n_zeta):
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

    zeta = jnp.linspace(0.0, 2 * jnp.pi / eq.NFP, n_zeta, endpoint=False)
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
def interpolate_xi(xi_rho_low, xi_theta_low, xi_zeta_low, rho_low, theta_low, zeta_low, reshaped_nodes, n_rho, n_theta, n_zeta):
    # add boundaries for interpolation later
    theta_low = np.concatenate((theta_low, np.array([2 * np.pi])))
    zeta_low  = np.concatenate((zeta_low,  np.array([2 * np.pi / eq.NFP])))

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


def _save_eigenfunction_plots(xi_rho, xi_theta, xi_zeta, rho, theta, rho_iota1, title_base, tag):
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


for i, iota_0 in enumerate(iota_on_axis_values):
    iota_coeffs = np.array([iota_0, -0.5])
    iota_modes  = np.array([0, 2])
    iota_profile = PowerSeriesProfile(iota_coeffs, modes=iota_modes)
    I_profile = None

    p_coeffs = np.array([0.125, 0, 0, 0, -0.125])
    p_profile = PowerSeriesProfile(p_coeffs)

    # Save directory and filename
    save_tag = (
        f"axisym_{axisym}_ar_{aspect_ratio}_NFP_{NFP}"
        f"_p_{'_'.join(p_coeffs.astype(str))}"
        f"_iota0_{iota_0:.4f}_d2iota_{2*iota_coeffs[-1]:.4f}"
    )
    save_name = f"equilibrium_{save_tag}.h5"

    print(f"\n=== iota_0 = {iota_0:.4f} ===")
    print("solving equilibrium")

    eq = Equilibrium(
        L=12,
        M=12,
        N=0,
        surface=FourierRZToroidalSurface.from_shape_parameters(
            major_radius=R,
            aspect_ratio=aspect_ratio,
            elongation=1,
            triangularity=0,
            squareness=0,
            eccentricity=0,
            torsion=0,
            twist=0,
            NFP=NFP,
            sym=True,
        ),
        NFP=NFP,
        iota=iota_profile,
        current=I_profile,
        pressure=p_profile,
        Psi=1,
    )

    eq = solve_continuation_automatic(eq, ftol=1E-13, gtol=1E-13, xtol=1E-13, verbose=0)[-1]
    eq.save(save_path + save_name)
    print("equilibrium solved")

    # ι=1 surface location: ι(ρ) = ι₀ + 2·iota_coeffs[1]·ρ² = 1
    rho_iota1 = (
        np.sqrt((iota_0 - 1.0) / (-iota_coeffs[1])) if iota_0 > 1.0 else None
    )
    title_base = (
        rf"$\iota_0 = {iota_0:.3f}$,  "
        rf"$\iota(\rho) = \iota_0 + {2*iota_coeffs[-1]:.2f}\,\rho^2$"
    )

    print("making input grid and diffmats")
    # Low-res solve for eigenfunction guess
    n_rhos = np.array([14, 18, 36])
    n_thetas = np.array([14, 18, 36])
    if axisym:
        n_zetas = np.ones(3)
    else:
        n_zetas = np.array([9, 12, 14])

    v_guess = None
    X = None

    for n_rho, n_theta, n_zeta in zip(n_rhos, n_thetas, n_zetas):
        print(f"\n--- Solving at low res: n_rho={n_rho}, n_theta={n_theta}, n_zeta={n_zeta} ---")
        # paths for saving eigenfunction and related data
        save_tag_res = f"{save_tag}_nrho_{n_rho}_ntheta_{n_theta}_nzeta_{n_zeta}"
        X_path = save_path + f"low_res_eigenfunction_{save_tag_res}.npy"
        savez_path = save_path + f"{save_tag_res}.npz"

        # produce diffmats and grid nodes for the current resolution
        diffmat, rho, theta, zeta = nodes_and_diffmats(n_rho, n_theta, n_zeta)

        if os.path.exists(X_path):
            X = np.load(X_path)
            data = np.load(savez_path)
            lambda_min = data["lambda_min"]
            data.close()
            print("loaded low-res eigenfunction and lambda_min from previous run")
        else:
            grid, reshaped_nodes = mapping_and_grid(eq, rho, theta, zeta)
            if X is not None:
                v_guess, xi_rho_interp, xi_theta_interp, xi_zeta_interp = interpolate_xi(
                    xi_rho_low, xi_theta_low, xi_zeta_low, rho_low, theta_low, zeta_low,
                    reshaped_nodes, n_rho, n_theta, n_zeta,
                )
                print("saving after-interpolation plots")
                _save_eigenfunction_plots(
                    xi_rho_interp, xi_theta_interp, xi_zeta_interp,
                    rho, theta, rho_iota1, title_base,
                    f"interp_{save_tag_res}",
                )
            print("computing eigenmode at low res")

            tic = time.time()
            data = eq.compute(
                "finite-n lambda3", grid=grid, diffmat=diffmat,
                gamma=100, incompressible=False,
                axisym=axisym, n_mode_axisym=n_mode_axisym,
                v_guess=v_guess
            )
            toc = time.time()
            print(f"matrix full took {toc-tic:.1f} s.")
            print(data["finite-n lambda3"])

            X = data["finite-n eigenfunction3"]

            np.save(X_path, X)

            # save processed eigenfunction and related data for later analysis
            xi_full = data["finite-n xi"]
            deltaB = data["finite-n deltaB"]
            deltaB_r = data["finite-n deltaB_r"]
            deltaB_v = data["finite-n deltaB_v"]
            deltaB_z = data["finite-n deltaB_z"]
            lambda_min = data["finite-n lambda3"]

            np.savez(
                savez_path,
                xi=xi_full,
                deltaB=deltaB,
                deltaB_r=deltaB_r,
                deltaB_v=deltaB_v,
                deltaB_z=deltaB_z,
                lambda_min=lambda_min,
            )

        # add boundaries back to the low-res eigenfunction for interpolation later
        xi_rho_low, xi_theta_low, xi_zeta_low = add_bc(X, n_rho, n_theta, n_zeta)

        # Save nodes for interpolation
        rho_low = rho
        theta_low = theta
        zeta_low = zeta

        # save eigenfunction components
        np.save(save_path + f"xi_rho_{save_tag_res}.npy", xi_rho_low)
        np.save(save_path + f"xi_theta_{save_tag_res}.npy", xi_theta_low)
        np.save(save_path + f"xi_zeta_{save_tag_res}.npy", xi_zeta_low)

        # save plots of the solved eigenfunction at this resolution
        print("saving solved-eigenfunction plots")
        _save_eigenfunction_plots(
            xi_rho_low, xi_theta_low, xi_zeta_low,
            rho, theta, rho_iota1, title_base,
            f"solved_{save_tag_res}",
        )

    # add the final lambda_min to the results list for this iota_0
    results_lambda_min[i] = lambda_min
    print(f"iota_0={iota_0:.4f}: plots saved")


# ── Save summary ──────────────────────────────────────────────────────────────
results_lambda_min = np.array(results_lambda_min)

np.savez(
    save_path + "iota_scan_results.npz",
    iota_on_axis=iota_on_axis_values,
    lambda_min=results_lambda_min,
)

print("Done. Results saved to", save_path)
print("Run analyze_stability.py for Mercier + delta_W breakdown.")


# ── Lambda vs iota_0 summary plot ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.axvline(1, color="gray", lw=0.8, ls="--", label=r"$\iota_0 = 1$")
ax.plot(iota_on_axis_values, results_lambda_min, linestyle="-", marker=".",
        color="steelblue", lw=2, ms=7)
ax.set_xlabel(r"$\iota_0$", fontsize=14)
ax.set_ylabel(r"$\lambda_{\min}$", fontsize=14)
ax.set_title(
    r"Stability eigenvalue vs $\iota_0$" + "\n"
    f"$\\iota(\\rho) = \\iota_0 + {2*iota_coeffs[-1]}\\rho^2$",
    fontsize=12,
)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(plot_path + "lambda_vs_iota0.png", dpi=150)
plt.show()
print(f"Lambda plot saved to {plot_path}lambda_vs_iota0.png")
