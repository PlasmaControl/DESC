from desc import set_device
set_device("gpu")

import numpy as np
from desc.geometry import FourierRZToroidalSurface
from desc.utils import rpz2xyz
from desc.grid import LinearGrid
from desc.magnetic_fields import SourceFreeField
from desc.coils import FourierRZCoil, CoilSet
from desc.equilibrium import Equilibrium
from desc.profiles import PowerSeriesProfile
import pdb
import time
from desc.examples import get
from desc.grid import LinearGrid, Grid
from desc.diffmat_utils import *
from desc.equilibrium.coords import map_coordinates
from desc.equilibrium import Equilibrium
from desc.backend import jnp, jax
from matplotlib import pyplot as plt
from desc.utils import dot, safenorm
from desc.integrals.quad_utils import leggauss_lob, automorphism_staircase1
from desc.io import load
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from desc.transform import Transform
from desc.integrals.singularities import _G, _grad_G

# Make or load an ultra high aspect-ratio tokamak (essentially a screw pinch)
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.grid import QuadratureGrid
import os

from newcomb import *

from desc.integrals.singularities import _grad_G
import os

chunk_size = 50

fixed_point = False

resolutions = np.hstack([3 * np.logspace(1, 2, num=2, base=2, dtype=int), np.linspace(24, 36, 7, dtype=int)])

# Root mean square errors in B_t, B_z, and phi for each resolution, relative to the "true" values computed from the Green's function.
B_t_errs = np.zeros_like(resolutions, dtype=float)
B_z_errs = np.zeros_like(resolutions, dtype=float)
phi_errs = np.zeros_like(resolutions, dtype=float)

just_plot = False
path_exists = np.ones_like(resolutions, dtype=bool)

for i, res in enumerate(resolutions):
    # Misc inputs
    if fixed_point:
        pest = False
    else:
        pest = True
    coords = "($\\rho, \\vartheta, \\phi$)" if pest else "($\\rho, \\theta, \\zeta$)" 
    from_scratch = False

    # Equilibrium paremeteters
    if from_scratch:
        a = 1  # Minor radius
        aspect_ratio = 2  # Aspect ratio of the tokamak
        R0 = aspect_ratio * a  # Major radius
        axisym = False  # Whether to enforce axisymmetry in the eigenvalue solve
        n_mode_axisym = 0 # If axisym is True, the toroidal mode number to solve for
        NFP = 1  # Number of field periods

        name = "Low aspect ratio highly non-axisym"
        fixed_iota = True

        if fixed_iota:
            iota_coeffs = np.array([0.9, 0, 0.1, 0, 0.1])
            iota_profile = PowerSeriesProfile(iota_coeffs)
            I_coeffs = None
            I_profile = None
        else:
            #I_coeffs = np.array([0, 0, I, 0, - I/2])
            #I_profile = PowerSeriesProfile(I_coeffs)
            # This is the current profile that corresponds to the iota profile above
            I_coeffs = np.array([-3.15111573e-08,  7.16194786e+03,  7.95861633e+02,  7.95781352e+02,
            -1.19289606e-02, -3.19292792e-03, -4.34680863e-03])
            I_profile = PowerSeriesProfile(I_coeffs, sym=True)
            iota_coeffs = None
            iota_profile = None

        p_coeffs = np.array([0.125, 0, 0, 0, -0.125])
        p_profile = PowerSeriesProfile(p_coeffs)
    else:
        eq_tag = name = "NCSX"
        axisym = False
        n_mode_axisym = 0

    # Define resolution
    M = res
    N = res
    n_rho = 1#14
    n_theta = 2 * M
    if axisym:
        n_zeta = 1
    else:
        n_zeta = 2 * N


    # Save path
    save_path = "results/phi_matrix/"
    plot_path = save_path + "plots/"
    if from_scratch:
        profile_tag = f"iota_{"_".join(iota_coeffs.astype(str))}" if fixed_iota else f"I_{"_".join(I_coeffs.astype(str))}"
        name_lower = name.lower().replace(" ", "_").replace("-", "_")
        eq_tag = f"{name_lower}_p_{"_".join(p_coeffs.astype(str))}_{profile_tag}"
    save_tag = f"{eq_tag}_M_{M}_N_{N}_pest_{pest}_debug"#_pseudospectral"
    eq_save_name = f"equilibrium_{save_tag}.h5"
    if pest:
        phi_save_name = f"{save_tag}_phi_matrix.npy"
    elif fixed_point:
        phi_save_name = phi_save_name.replace("_matrix.npy", "_fixed_point.npz")
    else:
        phi_save_name = f"{save_tag}_phi_matrix_rtz.npy"
    phi_save_name = phi_save_name.replace(f"_pest_{pest}", "")
    rtz_save_name = f"{save_tag}_rtz.npy"
    pest_save_name = f"{save_tag}_rvp.h5"
    surf_save_name = f"{save_tag}_surf.npy"
    os.makedirs(plot_path, exist_ok=True)


    # Make surface
    if not from_scratch:
        eq = get(eq_tag)
        eq.change_resolution(NFP=1)
        surface = eq.surface
        NFP = eq.NFP


    # NOTE: equilibrium LCFS must be ForceFreeField object 
    if from_scratch:
        override = False
        if os.path.exists(save_path + eq_save_name) and (not override):
            eq = load(save_path + eq_save_name)
        else:
            surface = FourierRZToroidalSurface(
                R_lmn=[R0, 1, 0.2, 0.1],
                Z_lmn=[-2, -0.2, 1, 0.1],
                modes_R=[[0, 0], [1, 0], [0, 4], [1, 1]],
                modes_Z=[[-1, 0], [0, -1], [1, 4], [1, 1]],
            )
            field = SourceFreeField(surface, M, N)
            eq = Equilibrium(
                    L=12, 
                    M=12,
                    N=8,
                    surface=field,
                    NFP=field.NFP,
                    iota = iota_profile,
                    current=I_profile,
                    pressure=p_profile,
                    Psi=1,
                )
            eq = solve_continuation_automatic(eq, ftol=1E-13, gtol=1E-13, xtol=1E-13, verbose=0)[-1]
            eq.save(save_path + eq_save_name)
    else:
        field = SourceFreeField(eq.surface, M, N)
        eq.surface = field

    # PEST grid: uniform in (theta_PEST, zeta) at rho=1, required by BIEST interpolator
    pest_grid = LinearGrid(rho=1.0, theta=n_theta, zeta=n_zeta, NFP=NFP, sym=False)
    pest_grid.save(save_path + pest_save_name)

    # This will probably OOM with the matrix-full method
    print("making input grid and diffmats")
    """
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
    W0 = W0 * scale_vector_inv"""
    rho = np.array([1.0])

    theta = pest_grid.unique_theta
    D1, W1 = fourier_diffmat(n_theta)

    zeta = pest_grid.unique_zeta
    D2, W2 = fourier_diffmat(n_zeta)

    grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=NFP, sym=False)

    # reshaped according to rho
    reshaped_nodes = jnp.reshape(
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_rho * n_theta * n_zeta, 3)
    )
    print("mapping coordinates")

    rtz_nodes = map_coordinates(
        eq,
        reshaped_nodes,  # (ρ,θ_PEST,ζ)
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )

    print("coordinates mapped")

    print("making grid of mapped coordinates")
    grid = Grid(rtz_nodes, NFP=NFP)
    np.save(save_path + rtz_save_name, rtz_nodes)

    n_surf = n_theta * n_zeta

    # Surface nodes (rho=1) are the last n_surf entries of rtz_nodes, in AGNI C order
    # (flat index = theta_i * n_zeta + zeta_k).
    # FFT/BIEST requires DESC Fortran order (flat index = theta_i + n_theta * zeta_k),
    # so we permute before building the surface grid.
    surface_nodes_agni = np.array(rtz_nodes[-n_surf:])
    surf_nodes = surface_nodes_agni.reshape(n_theta, n_zeta, 3).transpose(1,0,2).reshape(n_surf,3)
    rtz_surface_grid = Grid(surf_nodes, NFP=NFP)
    np.save(save_path + surf_save_name, surf_nodes)

    """"""
    print("computing phi matrix")
    # Compute the matrix A such that Phi_periodic = A @ B0*n.
    override = False
    print(save_path + phi_save_name)
    print(os.path.exists(save_path + phi_save_name))

    # x0 for Green's function
    x0 = eq.axis.compute("x", grid=Grid(np.array([[0, 0, 0]])), basis="xyz")["x"].flatten()

    if os.path.exists(save_path + phi_save_name) and not override:
        if fixed_point:
            print("loading phi from fixed-point iteration")
            data = np.load(save_path + phi_save_name)
            phi = data["Phi"]
            B_theta = data["B_theta"]
            B_zeta = data["B_zeta"]
        else:
            print("loading phi matrix")
            phi_matrix = np.load(save_path + phi_save_name)
        
    elif just_plot:
        path_exists[i]=False
        continue
    else:
        # Equilibrium doesn't expose Phi_basis directly; get it from the SourceFreeField surface
        #phi_transform = Transform(eq.surface.Phi_basis, rtz_surface_grid)
        if pest:
            data_phi = eq.compute(
                ["phi_matrix_pest"],
                rtz_surface_grid,
                pest_grid=pest_grid,
                problem="exterior Neumann",
                chunk_size=chunk_size,
                #transforms={"Phi": phi_transform},
            )
            phi_matrix = np.array(data_phi["phi_matrix_pest"])

            # Reshape to align with surface nodes for AGNI grid
            phi_matrix = phi_matrix.reshape(n_zeta, n_theta, n_zeta, n_theta)
            phi_matrix = phi_matrix.transpose(1,0,3,2)
            phi_matrix = phi_matrix.reshape(n_surf, n_surf)
        elif fixed_point:
            maxiter=30
            chunk_size=1000
            data = surface.compute(["x", "n_rho", "e_theta", "e_zeta"], grid=pest_grid, basis="xyz")
            data = {"B0*n": -dot(_grad_G(data["x"] - x0), data["n_rho"])}
            data, RpZ_data = field.compute(
                ["∇φ", "Phi", "x", "n_rho"],
                pest_grid,
                data=data,
                problem="exterior Neumann",
                on_boundary=True,
                maxiter=maxiter,
                full_output=True,
                chunk_size=chunk_size,
                basis="xyz",
            )
            assert data is RpZ_data
            print("num iterations:", data["num iter"])
            print("Phi error     :", data["Phi error"])

            # phi from fixed-point iteration, and its derivatives along the basis vectors
            phi = data["Phi"]
            B_theta = dot(data["∇φ"], data["e_theta"])
            B_zeta = dot(data["∇φ"], data["e_zeta"])
        else:
            data_phi = field.compute(
                ["phi_matrix"],
                grid=pest_grid,
                problem="exterior Neumann",
                chunk_size=chunk_size,
                #transforms={"Phi": phi_transform},
            )
            phi_matrix = np.array(data_phi["phi_matrix"])

        # phi_matrix_pest is in DESC Fortran order (theta fastest).
        # Reorder to AGNI C order (zeta fastest) expected by the stability solver.
        if fixed_point:
            np.savez(save_path + phi_save_name, **data)
        else:
            np.save(save_path + phi_save_name, phi_matrix)

    n_surf = n_theta * n_zeta
    if pest:
        # rho, theta, zeta locations of surface nodes, ordered by (theta, zeta)
        compute_grid = Grid(rtz_nodes[-n_surf:])
    else:
        # rho, theta, zeta locations of surface nodes, ordered by (zeta, theta)
        compute_grid = pest_grid

    # Compute values at surface nodes
    data = eq.compute(["x", "n_rho", "e_theta_PEST", "e_phi|v,r"], grid=compute_grid, basis="xyz")
    
    # Toy magnetic field (grad(G) where G is Green's function for Laplace's equation in 3D)
    phi_true = _G(data["x"] - x0)
    B = _grad_G(data["x"] - x0)

    # Basis vectors at surface nodes
    if pest:
        e_theta = data["e_theta_PEST"]
        e_zeta = data["e_phi|v,r"]
    else:
        e_theta = data["e_theta"]
        e_zeta = data["e_zeta"]
        
    if not fixed_point:
        # Compute B dot n
        B_dot_n = dot(B, data["n_rho"])

        # Compute phi and compare to expected value
        phi = phi_matrix @ B_dot_n

        # Make differentiation matrices
        I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta))
        I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta))
        if pest:
            D_theta = jax.lax.stop_gradient(jnp.kron(D1, I_zeta0))
            D_zeta = jax.lax.stop_gradient(jnp.kron(I_theta0, D2))
        else:
            D_theta = jax.lax.stop_gradient(jnp.kron(I_zeta0, D1))
            D_zeta = jax.lax.stop_gradient(jnp.kron(D2, I_theta0))

        B_theta = D_theta @ phi
        B_zeta = D_zeta @ phi

    # title to display on plots
    eq_title = f"\n {name} equilibrium, M={M}, N={N} in {coords} coords"

    # Plot phi from matrix vs phi from Green's function, and save
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(_G(data["x"] - x0), phi, linestyle="None", marker=".")
    ax.plot(_G(data["x"] - x0),_G(data["x"] - x0), linestyle="dashed")
    ax.set_title("$\\Phi$ from matrix vs $G(\\mathbf{x}-\\mathbf{x}_0)$" + eq_title, fontsize=14)
    ax.set_xlabel("$G(\\mathbf{x}-\\mathbf{x}_0)$", fontsize=12)
    ax.set_ylabel("$\\Phi$ from matrix", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_path + f"phi_plot_{save_tag}.png", dpi=150)
    

    # Plot B dot e_theta vs D_theta @ phi, and B dot e_zeta vs D_zeta @ phi, and save
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(dot(B, e_zeta), B_zeta, linestyle="None", marker=".")
    ax[0].plot(dot(B, e_zeta), dot(B, e_zeta), linestyle="dashed")
    ax[1].plot(dot(B, e_theta), B_theta, linestyle="None", marker=".")
    ax[1].plot(dot(B, e_theta), dot(B, e_theta), linestyle="dashed")
    ax[0].set_title("$\\mathbf{B} \\cdot \\mathbf{e}_\\zeta$ vs $D_\\zeta \\Phi$" + eq_title, fontsize=14)
    ax[0].set_xlabel("$\\mathbf{B} \\cdot \\mathbf{e}_\\zeta$", fontsize=12)
    ax[0].set_ylabel("$D_\\zeta \\Phi$ from matrix", fontsize=12)
    ax[1].set_title("$\\mathbf{B} \\cdot \\mathbf{e}_\\theta$ vs $D_\\theta \\Phi$" + eq_title, fontsize=14)
    ax[1].set_xlabel("$\\mathbf{B} \\cdot \\mathbf{e}_\\theta$", fontsize=12)
    ax[1].set_ylabel("$D_\\theta \\Phi$ from matrix", fontsize=12)
    fig.suptitle("Checking that derivatives of $\\Phi$ from matrix match $\\nabla G(\\mathbf{x}-\\mathbf{x}_0)$", fontsize=16)
    fig.tight_layout()
    fig.savefig(plot_path + f"B_plot_{save_tag}.png", dpi=150)

    B_t_errs[i] = ((dot(B, e_theta) - B_theta)**2).mean()**0.5
    B_z_errs[i] = ((dot(B, e_zeta) - B_zeta)**2).mean()**0.5
    phi_errs[i] = ((phi - phi_true)**2).mean()**0.5

eq_title = f"\n {name} equilibrium in {coords} coords"

# Plot errors vs resolution
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].plot(resolutions[path_exists], B_t_errs[path_exists], marker="o")
#ax[0].set_xscale("log")
#ax[0].set_yscale("log")
ax[0].set_xlabel("Resolution (M=N)", fontsize=12)
ax[0].set_ylabel("RMS error in $\\mathbf{B} \\cdot \\mathbf{e}_\\theta$", fontsize=12)
ax[1].plot(resolutions[path_exists], B_z_errs[path_exists], marker="o")
#ax[1].set_xscale("log")
#ax[1].set_yscale("log")
ax[1].set_xlabel("Resolution (M=N)", fontsize=12)
ax[1].set_ylabel("RMS error in $\\mathbf{B} \\cdot \\mathbf{e}_\\zeta$", fontsize=12)
ax[2].plot(resolutions[path_exists], phi_errs[path_exists], marker="o")
#ax[2].set_xscale("log")
#ax[2].set_yscale("log")
ax[2].set_xlabel("Resolution (M=N)", fontsize=12)
ax[2].set_ylabel("RMS error in $\\Phi$ from matrix", fontsize=12)

fig.suptitle(f"Error in $\\Phi$ from matrix and its derivatives vs resolution" + eq_title, fontsize=16)

fig.tight_layout()
fig.savefig(plot_path + f"error_plot_{save_tag}.png", dpi=150)

"""
print("computing eigenmode at low res")
tic = time.time()
data = eq.compute("finite-n lambda", grid=grid, diffmat=diffmat, gamma=100, incompressible=False, axisym=axisym, n_mode_axisym=n_mode_axisym, phi_matrix=phi_matrix)
toc = time.time()
print(f"matrix full took {toc-tic} s.")

print(data["finite-n lambda"])
X = data["finite-n eigenfunction"]

np.save(save_path + f"low_res_eigenfunction_all_{save_tag}.npy", X)
np.save(save_path + f"low_res_eigenvalue_all_{save_tag}.npy", data["finite-n lambda"])
"""