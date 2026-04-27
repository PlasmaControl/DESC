from desc.basis import ChebyshevDoubleFourierBasis
from desc.grid import QuadratureGrid
from desc.transform import Transform

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
from desc.magnetic_fields import SourceFreeField

# Make or load an ultra high aspect-ratio tokamak (essentially a screw pinch)
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.grid import QuadratureGrid
import os

from stability_helpers import *

# Input parameters
from_scratch = True
if from_scratch:
    a = 1  # Minor radius
    aspect_ratio = 10  # Aspect ratio of the tokamak
    R_0 = aspect_ratio * a  # Major radius
    NFP = 1
    axisym = True  # Whether to enforce axisymmetry in the eigenvalue solve
    n_mode_axisym = 1  # If axisym is True, the toroidal mode number to solve for
else:
    eq_name = "NCSX"

# Paths
save_path = "./eigenvalue_solve/"
plot_path = save_path + "eigenfunction_plots/"
os.makedirs(save_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

# Quadratic iota profile: iota(rho) = iota_0 - 0.05*rho^2
# => d^2 iota / d rho^2 = -0.1 (decreasing, as requested)
power_series = True
fixed_boundary = False
if power_series:
    free_parameter_values = np.hstack(
        [np.linspace(0.8, 1.25, 20), np.linspace(0.8, 1.25, 46)]
    )
    free_parameter_values = np.unique(
        free_parameter_values, sorted=True
    )  # Remove duplicates
else:
    alpha_values = np.hstack(
        [   
            np.linspace(-2, 0.0, 20, endpoint=False),
            np.linspace(0.0, 1.0, 20, endpoint=False),
            np.linspace(0.95, 1.0, 10, endpoint=False),
        ]
    )
    alpha_values = np.unique(alpha_values, sorted=True)  # Remove duplicates
    free_parameter_values = alpha_values

results_lambda_min = np.zeros_like(free_parameter_values)
stabilities = np.zeros_like(free_parameter_values, dtype=bool)

# phi matrix resolution
M = 20
N = 0

# Real space grid resolution for the eigenvalue solve
n_rho = 24
if fixed_boundary:
    n_theta = 36
else:
    n_theta = 2 * M
if axisym:
    n_zeta = 1
else:
    n_zeta = 2 * N

M_basis = 3
modes = np.zeros((free_parameter_values.shape[0], M_basis + 1))

# ── Grid setup ────────────────────────────────────────────────────────────────
diffmat, rho, theta, zeta = nodes_and_diffmats(n_rho, n_theta, n_zeta, NFP)
pest_grid = LinearGrid(rho=rho, theta=theta, zeta=zeta, sym=False)

basis = ChebyshevDoubleFourierBasis(L=pest_grid.L - 1, M=M_basis, N=0)
transform = Transform(pest_grid, basis, build_pinv=True)

for i, free_param in enumerate(free_parameter_values):
    if power_series:
        iota_0 = free_param
        iota_coeffs = np.array([iota_0, -0.1])
        iota_modes = np.array([0, 2])
        iota_profile = PowerSeriesProfile(iota_coeffs, modes=iota_modes)
        print(f"\n=== iota_0 = {iota_0:.4f} ===")
        eq_name = f"ar_{aspect_ratio}_iota0_{iota_0:.4f}_d2iota_{2*iota_coeffs[-1]:.4f}"

    else:
        alpha = free_param
        iota_0 = 1.0  # Keep iota_0 fixed and vary alpha instead
        rho = np.linspace(0, 1, 100)
        if alpha == 0.0:
            # limit as alpha → 0: iota(rho) = iota_0 - 0.5 * rho^2
            iota_modes = np.array([0, 2])
            iota_coeffs = np.array([iota_0, -0.5])
            iota_profile = PowerSeriesProfile(iota_coeffs, modes=iota_modes)

        else:
            # iota(rho) = (iota_0 / (alpha^2 * rho^2)) * (alpha * rho^2 + (1 - alpha) * log(1 - alpha * rho^2))
            iota = (iota_0 / (alpha**2 * rho**2)) * (
                1 * alpha * rho**2 + (1 - alpha) * np.log(1 - alpha * rho**2)
            )
            iota[0] = iota_0
            iota_profile = SplineProfile(iota, rho)
        eq_name = f"ar_{aspect_ratio}_iota0_{iota_0:.4f}_alpha_{alpha:.4f}"

    I_profile = None
    p_coeffs = np.array([0.125, 0, 0, 0, -0.125])
    p_profile = PowerSeriesProfile(p_coeffs)

    # Save directory and filename
    save_tag = (
        f"axisym_{axisym}_NFP_{NFP}"
        f"_p_{'_'.join(p_coeffs.astype(str))}"
        f"_{eq_name}"
    )
    if not fixed_boundary:
        save_tag += f"_external_mode_n_{n_mode_axisym}"

    if from_scratch:
        save_name = f"equilibrium_{save_tag}.h5"
        override = False
        if os.path.exists(save_path + save_name) and not override:
            eq = load(save_path + save_name)
            print(f"Loaded equilibrium from {save_path + save_name}")
        else:
            print("solving equilibrium")
            surface = FourierRZToroidalSurface.from_shape_parameters(
                major_radius=R_0,
                aspect_ratio=aspect_ratio,
                elongation=1,
                triangularity=0,
                squareness=0,
                eccentricity=0,
                torsion=0,
                twist=0,
                NFP=NFP,
                sym=True,
            )
            surface = SourceFreeField(surface, M=M, N=0, NFP=NFP)
            eq = Equilibrium(
                L=12,
                M=12,
                N=0,
                surface=surface,
                NFP=NFP,
                iota=iota_profile,
                current=I_profile,
                pressure=p_profile,
                Psi=1,
            )

            eq = solve_continuation_automatic(
                eq, ftol=1e-13, gtol=1e-13, xtol=1e-13, verbose=0
            )[-1]
            eq.save(save_path + save_name)
            print("equilibrium solved")

    print("making input grid and diffmats")

    # paths for saving eigenfunction and related data
    save_tag_res = f"{save_tag}_nrho_{n_rho}_ntheta_{n_theta}_nzeta_{n_zeta}"
    X_path = save_path + f"low_res_eigenfunction_{save_tag_res}.npy"
    savez_path = save_path + f"{save_tag_res}.npz"

    # Map PEST angles (theta_PEST, zeta) → rtz native angles
    rho = np.array([1.0])
    theta = pest_grid.unique_theta
    zeta = pest_grid.unique_zeta
    grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=False)
    n_surf = n_theta * n_zeta
    reshaped_nodes = jnp.reshape(
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_surf, 3)
    )
    print("mapping coordinates …")
    rtz_nodes = map_coordinates(
        eq,
        reshaped_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )

    # produce diffmats and grid nodes for the current resolution
    diffmat, rho, theta, zeta = nodes_and_diffmats(n_rho, n_theta, n_zeta, NFP)

    X = np.load(X_path)
    data = np.load(savez_path)
    idx0 = n_rho * n_theta * n_zeta
    xi = data["xi"]
    xi_r = (
        xi[:idx0].reshape((n_rho, n_theta, n_zeta)).transpose(2, 0, 1).flatten()
    )  # reshape to (n_zeta, n_rho, n_theta)
    lambda_min = data["lambda_min"]
    data.close()
    print("loaded low-res eigenfunction and lambda_min from previous run")

    coeffs = transform.fit(xi_r)
    coeffs = np.abs(coeffs) ** 2
    coeffs = coeffs.reshape(basis.L + 1, 2 * basis.M + 1)
    coeffs = coeffs.sum(axis=0)
    coeffs_pos = coeffs[-basis.M :]
    coeffs_neg = coeffs[: basis.M][::-1]
    coeffs = np.hstack([coeffs[basis.M], coeffs_neg + coeffs_pos])
    coeffs = coeffs / coeffs.sum()  # Normalize mode coefficients
    coeffs = np.sqrt(coeffs)
    modes[i, :] = coeffs


# ── Save summary ──────────────────────────────────────────────────────────────
results_lambda_min = np.array(results_lambda_min)

fig, ax = plt.subplots(figsize=(10, 7))

mask = (modes > 1e-3).any(axis=0)

if power_series:
    if fixed_boundary:
        savez_name = "fixed_mode_results.npz"
    else:
        savez_name = "free_mode_results.npz"
    np.savez(
        save_path + savez_name,
        iota_0=free_parameter_values,
        modes=modes,
    )

    # ── Lambda vs iota_0 summary plot ────────────────────────────────────────────
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(1, color="gray", lw=0.8, ls="--", label=r"$\iota_0 = 1$")
    ax.plot(
        free_parameter_values,
        modes[:, mask],
        linestyle="-",
        marker=".",
        lw=2,
        ms=7,
        label=[f"m={m}" for m in np.arange(0, M_basis + 1)[mask]],
    )
    ax.set_xlabel(r"$\iota_0$", fontsize=14)

    name = "fixed boundary" if fixed_boundary else "free boundary"
    ax.set_title(
        rf"Fourier decomposition of {name} eigenmode vs $\iota_0$" + "\n"
        f"$\\iota(\\rho) = \\iota_0 - {np.abs(iota_coeffs[-1])}\\rho^2$",
        fontsize=18,
    )

else:
    alpha = free_parameter_values
    np.savez(
        save_path + "alpha_mode_results.npz",
        alpha=free_parameter_values,
        modes=modes,
    )

    # ── Lambda vs iota_a summary plot ────────────────────────────────────────────
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(1, color="gray", lw=0.8, ls="--")  # , label=r"$\iota_a = 1$")
    ax.axvline(2, color="gray", lw=0.8, ls="--")  # , label=r"$\iota_a = 1/2$")

    iota_a = (1 / (alpha**2)) * (alpha + (1 - alpha) * np.log(1 - alpha))
    
    ax.plot(
        1 / iota_a,
        modes[:, mask],
        linestyle="-",
        marker=".",
        lw=2,
        ms=7,
        label=[f"m={m}" for m in np.arange(0, M_basis + 1)[mask]],
    )
    ax.set_xlabel(r"1/$\iota_a$", fontsize=14)

    ax.set_title(
        r"Fourier decomposition of $\xi^\rho$ as a function of $1/\iota_a$ for" + "\n"
        r"$\iota(\rho) = (1 / (\alpha^2\rho^2))[\alpha \rho^2 + (1 - \alpha)\log(1 - \alpha \rho^2)]$",
        fontsize=18,
    )
ax.tick_params(labelsize=15)
ax.legend(fontsize=18)
ax.set_ylabel(
    r"Fourier coefficient for mode m, rms over radial coefficients", fontsize=14
)
fig.tight_layout()
fig.savefig(
    plot_path + f"power_series_{power_series}_external_modes_modenum_vs_iota0.png",
    dpi=150,
)
plt.show()
