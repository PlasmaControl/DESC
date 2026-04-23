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
from desc.utils import dot
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
from desc.compute._stability import term_by_term_stability, energy_terms
from desc.compute.utils import get_params, get_transforms

from stability_helpers import *

# Input parameters
a = 1  # Minor radius
aspect_ratio = 200  # Aspect ratio of the tokamak
R = aspect_ratio * a  # Major radius
NFP = 1
n = 1             # toroidal mode number
m = 1             # poloidal mode number

# Low-res solve for eigenfunction guess
n_rho = 36
n_theta = 36
n_zeta = 14

# Quadratic iota profile: iota(rho) = iota_0 - 0.5*rho^2
iota_on_axis_values = np.linspace(0.8, 1.25, 10)

save_path = "./external_kink_mode/"
os.makedirs(save_path, exist_ok=True)

results_lambda_min = np.zeros_like(iota_on_axis_values)
results_term_by_term = {term: np.zeros_like(iota_on_axis_values) for term in energy_terms}

iota_prime = - 0.5 
iota_a = 1.1 # iota on edge; iota_a > 1should be unstable
iota_0 = iota_a - iota_prime
iota_coeffs = np.array([iota_0, iota_prime])  # iota(rho) = iota_0 + (iota_prime) rho^2
iota_modes  = np.array([0, 2])
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
        pressure=p_profile,
        Psi=1,
    )

    eq = solve_continuation_automatic(eq, ftol=1E-13, gtol=1E-13, xtol=1E-13, verbose=0)[-1]
    eq.save(save_path + save_name)
    print("equilibrium solved")



print("making input grid and diffmats")

# get grid in PEST coordinates and corresponding diffmats
diffmat, rho, theta, zeta = nodes_and_diffmats(n_rho, n_theta, n_zeta, NFP)
pest_grid = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)
iota = eq.compute("iota", grid=pest_grid)["iota"]

# get analytic eigenfunction
delta = 1e-2  # small shift to avoid singularity at rational surface
xi_0 = 1
rho, theta, zeta = pest_grid.nodes.T
xi = xi_0 * np.cos(m * theta + n * zeta)
xi_eta = - np.sin(m * theta + n * zeta)
xi_parallel = xi_0 * aspect_ratio/rho * (-n + iota)/((-n+iota)**2 + delta**2) * np.sin(m * theta + n * zeta)


# reconstruct 3d eigenfunction arrays
data = eq.compute(["b", "n_rho", ])
xi_3d = 

# get pest transform
from desc.basis import ChebyshevDoubleFourierBasis


grid, reshaped_nodes = mapping_and_grid(eq, rho, theta, zeta)
tbt_data_keys = [
    "g_rr|PEST", "g_rv|PEST", "g_rp|PEST",
    "g_vv|PEST", "g_vp|PEST", "g_pp|PEST",
    "g^rr", "g^rv", "g^rz",
    "J^theta_PEST", "J^zeta", "|J|",
    "sqrt(g)_PEST",
    "(sqrt(g)_PEST_r)|PEST",
    "(sqrt(g)_PEST_v)|PEST",
    "(sqrt(g)_PEST_p)|PEST",
    "finite-n instability drive",
    "iota", "psi_r", "psi_rr", "p", "a",
]
tbt_data   = eq.compute(tbt_data_keys, grid=grid)
params     = get_params("finite-n lambda", eq)
transforms = get_transforms("finite-n lambda", eq, grid, diffmat=diffmat)

energy = term_by_term_stability(
    v, params, transforms, tbt_data,
    diffmat=diffmat,
    gamma=100,
    incompressible=False,
    sigma=0,
)
toc = time.time()
total_energy = np.sum([value for key, value in energy.items()])
print(energy)
print(f"  term_by_term_stability took {toc-tic:.1f} s,  Rayleigh quotient = {total_energy:.6e}")
for key, value in energy.items():
    print(f"    {key}: {value:.6e}")
    results_term_by_term[key][i] = np.complex128(value)


results_lambda_min[i] = total_energy


# ── Save summary ──────────────────────────────────────────────────────────────
np.savez(
    save_path + "iota_scan_results_analytic.npz",
    iota_on_axis=iota_on_axis_values,
    lambda_min=results_lambda_min,
)

print("Done. Results saved to", save_path)
print("Run analyze_stability.py for Mercier + delta_W breakdown.")
