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
from desc.utils import dot, cross
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
M = 18
N = 18

# number of grid points in each direction for the eigenvalue solve
n_rho = 36
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
pest_grid = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)

# get data for equilibrium quantites
data = eq.compute(
    ["b", "n_rho", "n_theta", "n_zeta", "iota", "e^vartheta", "grad(phi)", "e^rho", "R0", "<|B|>_vol", "iota"],
    grid=pest_grid,
)

# evaluate quantities
rho, theta, zeta = pest_grid.nodes.T
eps = 1 / aspect_ratio
b_theta = dot(data["b"], data["n_theta"])
b_z = dot(data["b"], data["n_zeta"])
iota = data["iota"]
B_0 = data["<|B|>_vol"]
R_0 = data["R0"]

# get analytic eigenfunction
delta = 1e-2  # small shift to avoid singularity at rational surface
xi_0 = 1
rho, theta, zeta = pest_grid.nodes.T
xi = xi_0 * np.cos(theta + n * zeta)
xi_eta = (
    -xi_0
    * b_z
    * ((1 - iota * n * (eps**2) * (rho**2)) / (1 + (n**2) * (eps**2) * (rho**2)))
    * np.sin(theta + n * zeta)
)

xi_parallel = (
    (1 / (rho * eps))
    * ((n - iota) / ((n - iota**2) + delta**2))
    * (1 + n * iota * eps**2 * rho**2)
    * xi_eta
)


# reconstruct 3d eigenfunction arrays
# \hat{\eta} = \hat{b} \times \hat{r}
b_hat = data["b"]
r_hat = data["n_rho"]
eta_hat = cross(b_hat, r_hat)
xi = xi[:, None] * r_hat + xi_eta[:, None] * eta_hat + xi_parallel[:, None] * b_hat

# convert to pest coordinates
xi_r = dot(xi, data["e^rho"])  # xi^rho
xi_theta = dot(xi, data["e^vartheta"])  # xi^theta
xi_z = dot(xi, data["grad(phi)"])  # xi^z
xi = np.concatenate((xi_r, xi_theta, xi_z), axis=0)

# get phi matrix
n_surf = n_theta * n_zeta
rtz_nodes = grid.nodes # grid nodes are in (rho, theta, zeta)
surface_nodes_agni = np.array(rtz_nodes[-n_surf:]) # last n_surf nodes
surf_nodes = surface_nodes_agni.reshape(n_theta, n_zeta, 3).transpose(1,0,2).reshape(n_surf,3)
rtz_surface_grid = Grid(surf_nodes, NFP=NFP)
pest_grid_surf = LinearGrid(rho=1.0, theta=n_theta, zeta=n_zeta, NFP=NFP, sym=False)
data_phi = eq.compute(
            ["phi_matrix_pest"],
            rtz_surface_grid,
            pest_grid=pest_grid_surf,
            problem="exterior Neumann",
            chunk_size=50,
            #transforms={"Phi": phi_transform},
        )
phi_matrix = np.array(data_phi["phi_matrix_pest"])

# Reshape to align with surface nodes for AGNI grid
phi_matrix = phi_matrix.reshape(n_zeta, n_theta, n_zeta, n_theta)
phi_matrix = phi_matrix.transpose(1,0,3,2)
phi_matrix = phi_matrix.reshape(n_surf, n_surf)

data = eq.compute(
    "finite-n lambda3",
    xi=xi,
    grid=grid,
    diffmat=diffmat,
    gamma=100,
    incompressible=False,
    phi_matrix=phi_matrix,
)

print("minimum eigenvalue:", data["energy"])

W_0 = 2* np.pi**2 * R_0 * B_0 **2/(mu_0 * a**2)
delta_W_hat_analytic = 2 * a**2 * xi_0**2 * iota_a * n * (n/iota_a - 1)
delta_W_analytic = delta_W_hat_analytic * W_0 * eps**2
print("analytic expectation:", delta_W_analytic)