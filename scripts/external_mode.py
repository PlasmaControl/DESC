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
from desc.profiles import PowerSeriesProfile
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
plot_path   = save_path + "eigenfunction_plots/"
os.makedirs(save_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

# Quadratic iota profile: iota(rho) = iota_0 - 0.05*rho^2
# => d^2 iota / d rho^2 = -0.1 (decreasing, as requested)
iota_on_axis_values = np.hstack([np.linspace(0.8, 1.25, 20),np.linspace(0.8, 1.25, 46)])
iota_on_axis_values = np.unique(iota_on_axis_values, sorted=True)  # Remove duplicates

results_lambda_min = np.zeros_like(iota_on_axis_values)
stabilities = np.zeros_like(iota_on_axis_values, dtype=bool)

# phi matrix resolution
M = 20
N = 0

# Real space grid resolution for the eigenvalue solve
n_rho = 24
n_theta = 2 * M
if axisym:
    n_zeta = 1
else:
    n_zeta = 2 * N

for i, iota_0 in enumerate(iota_on_axis_values):
    iota_coeffs = np.array([iota_0, -0.1])
    iota_modes  = np.array([0, 2])
    iota_profile = PowerSeriesProfile(iota_coeffs, modes=iota_modes)
    I_profile = None
    p_coeffs = np.array([0.125, 0, 0, 0, -0.125])
    p_profile = PowerSeriesProfile(p_coeffs)
    eq_name = f"ar_{aspect_ratio}_iota0_{iota_0:.4f}_d2iota_{2*iota_coeffs[-1]:.4f}"

    print(f"\n=== iota_0 = {iota_0:.4f} ===")
        
    # Save directory and filename
    save_tag = (
        f"axisym_{axisym}_NFP_{NFP}"
        f"_p_{'_'.join(p_coeffs.astype(str))}"
        f"_{eq_name}"
        f"_external_mode_n_{n_mode_axisym}"
    )

    if from_scratch:
        save_name = f"equilibrium_{save_tag}.h5"
        override = True
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

            eq = solve_continuation_automatic(eq, ftol=1E-13, gtol=1E-13, xtol=1E-13, verbose=0)[-1]
            eq.save(save_path + save_name)
            print("equilibrium solved")

    # ι=1 surface location: ι(ρ) = ι₀ + 2·iota_coeffs[1]·ρ² = 1
    rho_iota1 = (
        np.sqrt((iota_0 - 1.0) / (-iota_coeffs[1])) if iota_0 > 1.0 else None
    )
    title_base = (
        rf"$\iota_0 = {iota_0:.3f}$,  "
        rf"$\iota(\rho) = \iota_0 - {np.abs(iota_coeffs[-1]):.2f}\,\rho^2$"
        rf", free-boundary modes"
    )

    print("making input grid and diffmats")

    phi_save_name = f"{save_tag}_M_{M}_N_{N}_pseudospectral_phi_matrix.npy"

    
    print(f"\n--- Solving at res: n_rho={n_rho}, n_theta={n_theta}, n_zeta={n_zeta} ---")
    # paths for saving eigenfunction and related data
    save_tag_res = f"{save_tag}_nrho_{n_rho}_ntheta_{n_theta}_nzeta_{n_zeta}"
    X_path = save_path + f"low_res_eigenfunction_{save_tag_res}.npy"
    savez_path = save_path + f"{save_tag_res}.npz"

    # ── Grid setup ────────────────────────────────────────────────────────────────
    pest_grid = LinearGrid(rho=1.0, theta=n_theta, zeta=n_zeta, sym=False, NFP=128)

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
    # theta-outermost, zeta-fastest (AGNI C order)
    compute_grid = Grid(rtz_nodes, NFP=NFP)
    # BIEST surface grid: zeta-outermost, theta-fastest
    surf_nodes = (
        np.array(rtz_nodes)
        .reshape(n_theta, n_zeta, 3)
        .transpose(1, 0, 2)
        .reshape(n_surf, 3)
    )
    rtz_surface_grid = Grid(surf_nodes, NFP=NFP)

    # ── phi_matrix ────────────────────────────────────────────────────────────────
    if os.path.exists(save_path + phi_save_name):
        print(f"loading phi_matrix from {save_path + phi_save_name}")
        phi_matrix = np.load(save_path + phi_save_name)
    else:
        print("computing phi_matrix …")
        data_phi = eq.compute(
            ["phi_matrix_pest"],
            rtz_surface_grid,
            pest_grid=pest_grid,
            problem="exterior Neumann",
            chunk_size=1,
        )
        phi_matrix = np.array(data_phi["phi_matrix_pest"])
        phi_matrix = (
            phi_matrix.reshape(n_zeta, n_theta, n_zeta, n_theta)
            .transpose(1, 0, 3, 2)
            .reshape(n_surf, n_surf)
        )

    # produce diffmats and grid nodes for the current resolution
    diffmat, rho, theta, zeta = nodes_and_diffmats(n_rho, n_theta, n_zeta, NFP)

    override = False
    if os.path.exists(X_path) and not override:
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
                reshaped_nodes, n_rho, n_theta, n_zeta, NFP
            )
            print("saving after-interpolation plots")
            save_eigenfunction_plots(
                xi_rho_interp, xi_theta_interp, xi_zeta_interp,
                rho, theta, rho_iota1, title_base,
                f"interp_{save_tag_res}",
            )
        print("computing eigenmode at low res")

        tic = time.time()
        data = eq.compute(
            "finite-n lambda3", grid=grid, diffmat=diffmat,
            gamma=10, incompressible=False,
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
    save_eigenfunction_plots(
        plot_path,
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
    f"$\\iota(\\rho) = \\iota_0 - {np.abs(iota_coeffs[-1])}\\rho^2$",
    fontsize=12,
)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(plot_path + "lambda_vs_iota0.png", dpi=150)
plt.show()
print(f"Lambda plot saved to {plot_path}lambda_vs_iota0.png")

