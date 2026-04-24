# A script to set up and solve a high aspect-ratio tokamak equilibrium using DESC,
# and then evaluate its stability using Newcomb's procedure.
# from desc import set_device
# set_device("gpu")

import pdb
import time
from desc.examples import get
from desc.grid import LinearGrid, Grid
from desc.diffmat_utils import *
from desc.equilibrium.coords import map_coordinates
from desc.equilibrium import Equilibrium
from desc.backend import jnp, jax
from matplotlib import pyplot as plt
from desc.utils import dot, cross, safenorm
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

from newcomb import *
from desc.compute._stability import term_by_term_stability, energy_terms
from desc.compute.utils import get_params, get_transforms

from stability_helpers import *

# Input parameters
a = 1  # Minor radius
aspect_ratio = 200  # Aspect ratio of the tokamak
R = aspect_ratio * a  # Major radius
NFP = 1
n = 1  # toroidal mode number

# resolutions for phi matrix
M = 9
N = 9

# number of grid points in each direction for the eigenvalue solve
n_rho = 18
n_theta = 2 * M
n_zeta = 2 * N

# Quadratic iota profile: iota(rho) = iota_0 - 0.5*rho^2
iota_on_axis_values = np.linspace(0.8, 1.25, 10)

save_path = "./external_kink_mode/"
os.makedirs(save_path, exist_ok=True)

results_lambda_min = np.zeros_like(iota_on_axis_values)
results_term_by_term = {
    term: np.zeros_like(iota_on_axis_values) for term in energy_terms
}

iota_prime = -0.5
iota_a = 1.1  # iota on edge; iota_a > 1should be unstable
iota_0 = iota_a - iota_prime
iota_coeffs = np.array([iota_0, iota_prime])  # iota(rho) = iota_0 + (iota_prime) rho^2
iota_modes = np.array([0, 2])
iota_profile = PowerSeriesProfile(iota_coeffs, modes=iota_modes)

p_coeffs = np.array([0.125, 0, 0, 0, -0.125])
p_profile = PowerSeriesProfile(p_coeffs)

# Save directory and filename
save_tag = (
    f"ar_{aspect_ratio}_NFP_{NFP}"
    f"_p_{'_'.join(p_coeffs.astype(str))}"
    f"_iota0_{iota_0:.4f}_d2iota_{2*iota_coeffs[-1]:.4f}"
    f"_n_rho_{n_rho}_n_theta_{n_theta}_n_zeta_{n_zeta}"
    f"_external_kink_mode"
)
save_name = f"equilibrium_{save_tag}.h5"

print(f"\n=== iota_0 = {iota_0:.4f} ===")

if os.path.exists(save_path + save_name):
    print("loading existing equilibrium from", save_path + save_name)
    eq = load(save_path + save_name)
else:
    print("solving equilibrium")
    surface = FourierRZToroidalSurface.from_shape_parameters(
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
    )
    surface = SourceFreeField(surface, M, N)
    eq = Equilibrium(
        L=12,
        M=12,
        N=0,
        surface=surface,
        NFP=NFP,
        iota=iota_profile,
        pressure=p_profile,
        Psi=1,
    )

    eq = solve_continuation_automatic(
        eq, ftol=1e-13, gtol=1e-13, xtol=1e-13, verbose=0
    )[-1]
    eq.save(save_path + save_name)
    print("equilibrium solved")


print("making input grid and diffmats")

# get grid in PEST coordinates and corresponding diffmats
diffmat, rho, theta, zeta = nodes_and_diffmats(n_rho, n_theta, n_zeta, NFP)
grid, reshaped_nodes = mapping_and_grid(eq, rho, theta, zeta)
pest_grid = Grid(reshaped_nodes, NFP=NFP)
# pest_grid = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)

# get data for equilibrium quantites
data = eq.compute(
    [
        "b",
        "n_rho",
        "n_theta",
        "n_zeta",
        "iota",
        "e^vartheta",
        "grad(phi)",
        "e^rho",
        "R0",
        "<|B|>_vol",
        "iota",
        "sqrt(g)_PEST",
        "R",
        "psi_r",
        "Psi",
        "a",
        "g^rr",
        "|B|"
    ],
    grid=grid,  # ordered (rho, theta, zeta),
)

a_N = data["a"]
B_N = data["Psi"] / (jnp.pi * a_N**2)
psi_r = data["psi_r"] / (a_N**2 * B_N)

# evaluate quantities
rho, theta, zeta = reshaped_nodes.T  # pest_grid.nodes.T
r = rho * a
eps = 1 / aspect_ratio
normal_theta = data["e^vartheta"] / safenorm(data["e^vartheta"], axis=-1)[:, None]
normal_zeta = data["grad(phi)"] / safenorm(data["grad(phi)"], axis=-1)[:, None]
b_theta = dot(data["b"], normal_theta)
print(b_theta / eps)
b_z = dot(data["b"], normal_zeta)
print(b_z)  # should be ~ 1 + O(eps^2)
iota = data["iota"]
B_0 = data["<|B|>_vol"]
R = data["R"]
R_0 = data["R0"]

# get analytic eigenfunction
k = -n / R
k0_sq = k**2 + (1 / r**2)
F = k * b_z + b_theta / r  # F/B
G = -k * b_theta + b_z / r  # G/B


delta = 1e-4  # small shift to avoid singularity at rational surface
xi_0 = 1
xi_normal = xi_0 * np.cos(theta - n * zeta)
xi_eta = -1 / (r * k0_sq) * (2 * k * b_theta + G) * xi_0 * np.sin(theta - n * zeta)
"""
xi_eta = (
    -xi_0
    * b_z
    * ((1 - iota * n * (eps**2) * (rho**2)) / (1 + (n**2) * (eps**2) * (rho**2)))
    * np.sin(theta - n * zeta)
)"""

"""xi_parallel = (
    (1 / (rho * eps))
    * ((n - iota) / ((n - iota)**2 + delta**2))
    * (1 + n * iota * eps**2 * rho**2)
    * xi_eta
)"""
# xi_parallel = - (F/(F**2 + delta**2)) * G * xi_eta
xi_parallel = -(F / (F**2 + delta**2)) * (
    xi_0 * np.sin(theta - n * zeta) / r + G * xi_eta
)


# reconstruct 3d eigenfunction arrays
# \hat{\eta} = \hat{b} \times \hat{r}
b_hat = data["b"]
r_hat = data["n_rho"]
eta_hat = cross(b_hat, r_hat)
div_fig, div_ax = plt.subplots(figsize=(7, 4))

for xi, label in zip(
    [
        xi_normal[:, None] * r_hat,
        xi_eta[:, None] * eta_hat,
        xi_parallel[:, None] * b_hat,
        xi_normal[:, None] * r_hat
        + xi_eta[:, None] * eta_hat
        + xi_parallel[:, None] * b_hat,
    ],
    ["normal", "eta", "parallel", "total"],
):

    # convert to pest coordinates
    xi_r = dot(xi, data["e^rho"])  # xi^rho
    xi_theta = dot(xi, data["e^vartheta"])  # xi^theta
    xi_z = dot(xi, data["grad(phi)"])  # xi^z
    xi = np.concatenate((xi_r / (psi_r + 1e-5), xi_theta, xi_z), axis=0)

    # ─── Debugging: eigenfunction plots ──────────────────────────────────────────
    # Reshape all 1D (n_total,) arrays to (n_rho, n_theta, n_zeta)
    """xi_normal_3d = pest_grid.meshgrid_reshape(xi_normal,   order="rtz")
    xi_eta_3d    = pest_grid.meshgrid_reshape(xi_eta,      order="rtz")
    xi_par_3d    = pest_grid.meshgrid_reshape(xi_parallel, order="rtz")
    xi_r_3d      = pest_grid.meshgrid_reshape(xi_r,        order="rtz")
    xi_t_3d      = pest_grid.meshgrid_reshape(xi_theta,    order="rtz")
    xi_z_3d      = pest_grid.meshgrid_reshape(xi_z,        order="rtz")"""

    # rho_3d   = pest_grid.meshgrid_reshape(rho,   order="rtz")
    # theta_3d = pest_grid.meshgrid_reshape(theta, order="rtz")
    # zeta_3d  = pest_grid.meshgrid_reshape(zeta,  order="rtz")
    rho_3d = rho.reshape(n_rho, n_theta, n_zeta)
    theta_3d = theta.reshape(n_rho, n_theta, n_zeta)
    zeta_3d = zeta.reshape(n_rho, n_theta, n_zeta)
    xi_normal_3d = xi_normal.reshape(n_rho, n_theta, n_zeta)
    xi_eta_3d = xi_eta.reshape(n_rho, n_theta, n_zeta)
    xi_par_3d = xi_parallel.reshape(n_rho, n_theta, n_zeta)
    xi_r_3d = xi_r.reshape(n_rho, n_theta, n_zeta)
    xi_t_3d = xi_theta.reshape(n_rho, n_theta, n_zeta)
    xi_z_3d = xi_z.reshape(n_rho, n_theta, n_zeta)

    rho_1d = np.array(rho_3d[:, 0, 0])
    theta_1d = np.array(theta_3d[0, :, 0])
    zeta_1d = np.array(zeta_3d[0, 0, :])

    comp_list = [xi_normal_3d, xi_eta_3d, xi_par_3d, xi_r_3d, xi_t_3d, xi_z_3d]
    comp_labels = [
        r"$\xi_{\rm normal}$",
        r"$\xi_\eta$",
        r"$\xi_\parallel$",
        r"$\xi^r$",
        r"$\xi^\theta$",
        r"$\xi^\zeta$",
    ]

    for xvals, xlabel, fname, slice_fn in [
        (rho_1d, r"$\rho$", "xi_vs_rho.png", lambda c: c[:, 0, 0]),
        (theta_1d, r"$\theta$", "xi_vs_theta.png", lambda c: c[-1, :, 0]),
        (zeta_1d, r"$\zeta$", "xi_vs_zeta.png", lambda c: c[-1, 0, :]),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, comp, lbl in zip(axes.flat, comp_list, comp_labels):
            ax.plot(xvals, slice_fn(comp))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(lbl)
            ax.set_title(lbl)
            ax.axhline(0, color="gray", lw=0.7, ls="--")
        fig.suptitle(f"Eigenfunction vs {xlabel}")
        plt.tight_layout()
        fig.savefig(save_path + fname, dpi=150)
        plt.close(fig)

    print("Saved eigenfunction plots to", save_path)

    # ─── Divergence check: ∇·ξ = (1/√g)[∂ρ(√g ξ^ρ) + ∂ϑ(√g ξ^ϑ) + ∂ζ(√g ξ^ζ)] ──
    sqrtg_3d = data["sqrt(g)_PEST"].reshape(
        n_rho, n_theta, n_zeta
    )  # pest_grid.meshgrid_reshape(data["sqrt(g)_PEST"], order="rtz")
    D_rho_mat = np.array(diffmat.D_rho)
    D_theta_mat = np.array(diffmat.D_theta)
    D_zeta_mat = np.array(diffmat.D_zeta)

    dr = np.einsum("ij,jkl->ikl", D_rho_mat, sqrtg_3d * xi_r_3d)
    dt = np.einsum("ij,kjl->kil", D_theta_mat, sqrtg_3d * xi_t_3d)
    dz = np.einsum("ij,klj->kli", D_zeta_mat, sqrtg_3d * xi_z_3d)

    print(
        f"dr: {np.max(np.abs(dr)):.4e}, dt: {np.max(np.abs(dt)):.4e}, dz: {np.max(np.abs(dz)):.4e}"
    )
    div_xi = (dr + dt + dz) / sqrtg_3d

    print(f"Divergence check: max |∇·ξ| = {np.max(np.abs(div_xi)):.4e}")
    print(f"Divergence check: RMS |∇·ξ| = {np.sqrt(np.mean(div_xi**2)):.4e}")

    div_ax.plot(rho_1d, np.max(np.abs(div_xi), axis=(1, 2)) + 1e-30, label=label)
print("total divergence:")

div_ax.legend()
div_ax.set_xlabel(r"$\rho$")
div_ax.set_ylabel(r"$\max_{\theta,\zeta}|\nabla\cdot\xi|$")
div_ax.set_title("Divergence of analytic eigenfunction")
plt.tight_layout()
fig.savefig(save_path + "divergence_check.png", dpi=150)
plt.close(fig)

# get phi matrix
n_surf = n_theta * n_zeta
rtz_nodes = grid.nodes  # grid nodes are in (rho, theta, zeta)
surface_nodes_agni = np.array(rtz_nodes[-n_surf:])  # last n_surf nodes
surf_nodes = (
    surface_nodes_agni.reshape(n_theta, n_zeta, 3).transpose(1, 0, 2).reshape(n_surf, 3)
)
rtz_surface_grid = Grid(surf_nodes, NFP=NFP)
pest_grid_surf = LinearGrid(rho=1.0, theta=n_theta, zeta=n_zeta, NFP=NFP, sym=False)

phi_path = save_path + f"phi_matrix_{save_tag}.npy"
if os.path.exists(phi_path):
    print(f"Loading phi matrix from {phi_path}")
    phi_matrix = np.load(phi_path)
else:
    data_phi = eq.compute(
        ["phi_matrix_pest"],
        rtz_surface_grid,
        pest_grid=pest_grid_surf,
        problem="exterior Neumann",
        chunk_size=1,
        # transforms={"Phi": phi_transform},
    )
    phi_matrix = np.array(data_phi["phi_matrix_pest"])

    # Reshape to align with surface nodes for AGNI grid
    phi_matrix = phi_matrix.reshape(n_zeta, n_theta, n_zeta, n_theta)
    phi_matrix = phi_matrix.transpose(1, 0, 3, 2)
    phi_matrix = phi_matrix.reshape(n_surf, n_surf)

    np.save(phi_path, phi_matrix)
"""data = eq.compute(
    "finite-n lambda3",
    xi=xi,
    grid=grid,
    diffmat=diffmat,
    gamma=100,
    incompressible=False,
    phi_matrix=phi_matrix,
)

print("minimum eigenvalue:", data["energy"])
"""
n_total = n_rho * n_theta * n_zeta
b_idx = slice(n_total - n_surf, n_total)

# surface quantities
psi_r_s = 1  # psi_r[b_idx, :] = 1 on the boundary
sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3
g_sup_rr = data["g^rr"][:, None] * a_N**2
sqrtg_grad_rho = sqrtg[b_idx, :] * np.sqrt(g_sup_rr[b_idx, :])
print("sqrt(g) |grad(rho)| on boundary:", sqrtg_grad_rho.flatten())

surface_jacobian = eq.surface.compute("|e_theta x e_zeta|", grid=rtz_surface_grid)[
    "|e_theta x e_zeta|"
]

iota_s = iota[b_idx, None]
xi_b = xi[b_idx]

I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta))
I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta))
D_theta = jax.lax.stop_gradient(jnp.kron(D_theta_mat, I_zeta0))
D_zeta = jax.lax.stop_gradient(jnp.kron(I_theta0, D_zeta_mat))
B_dot_n = (
    psi_r_s / sqrtg_grad_rho * (iota_s * D_theta + D_zeta)
) @ xi_b  # B dot n = (1/sqrt(g)|grad(rho)|) * (iota ∂_θ + ∂_ζ) xi^ρ on the boundary
W_theta = diffmat.W_theta
W_zeta = diffmat.W_zeta
W = jnp.kron(jnp.diag(W_theta), jnp.diag(W_zeta))


def _cT(x):
    return jnp.conjugate(jnp.transpose(x))


B_dot_n = jnp.ones_like(B_dot_n)
W_V = - dot(B_dot_n, (W * sqrtg_grad_rho * phi_matrix) @ B_dot_n)

analytic_W_V = (np.mean(r[b_idx])**2 * np.mean(F[b_idx])**2 * np.mean(data["|B|"])[b_idx]**2) * xi_0**2
analytic_W_V = analytic_W_V * 2 * np.pi**2 * R_0 / (mu_0)

print("W_V from AGNI:", W_V)
print("analytic W_V:", np.sum(analytic_W_V))
"""
# ─── Term-by-term energy contributions ────────────────────────────────────────
from desc.compute._stability import energy_terms

print("\nTerm-by-term energy contributions:")
for term_name in energy_terms:
    data_term = eq.compute(
        "finite-n lambda3",
        xi=xi,
        grid=grid,
        diffmat=diffmat,
        gamma=100,
        incompressible=False,
        phi_matrix=phi_matrix,
        term=term_name,
    )
    print(f"  {term_name}: {data_term['energy']}")

W_0 = (2 * np.pi**2 * R_0 * B_0**2) / (mu_0 * a**2)
delta_W_hat_analytic = 2 * a**2 * xi_0**2 * iota_a * n * (n / iota_a - 1)
delta_W_analytic = delta_W_hat_analytic * W_0 * eps**2
print("analytic expectation:", delta_W_analytic)
"""
