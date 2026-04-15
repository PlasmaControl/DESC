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

# Input parameters
a = 1  # Minor radius
aspect_ratio = 200  # Aspect ratio of the tokamak
R = aspect_ratio * a  # Major radius
NFP = 1
axisym = True  # Whether to enforce axisymmetry in the eigenvalue solve
n_mode_axisym = 1  # If axisym is True, the toroidal mode number to solve for

# Low-res solve for eigenfunction guess
n_rho = 24
n_theta = 24
if axisym:
    n_zeta = 1
else:
    n_zeta = 14

# Quadratic iota profile: iota(rho) = iota_0 - 0.05*rho^2
# => d^2 iota / d rho^2 = -0.1 (decreasing, as requested)
iota_on_axis_values = np.linspace(0.6, 1.5, 10)

save_path = "./high_aspect_ratio_tokamak/"
os.makedirs(save_path, exist_ok=True)

results_lambda_min = []

stabilities = []

for iota_0 in iota_on_axis_values:
    iota_coeffs = np.array([iota_0, -0.05])
    iota_modes  = np.array([0, 2])
    iota_profile = PowerSeriesProfile(iota_coeffs, modes=iota_modes)
    I_profile = None

    p_coeffs = np.array([0.125, 0, 0, 0, -0.125])
    p_profile = PowerSeriesProfile(p_coeffs)

    # Save directory and filename
    save_tag = (
        f"axisym_{axisym}_ar_{aspect_ratio}_NFP_{NFP}"
        f"_p_{'_'.join(p_coeffs.astype(str))}"
        f"_iota0_{iota_0:.4f}_d2iota_{-0.1:.4f}"
        f"n_rho_{n_rho}_n_theta_{n_theta}_n_zeta_{n_zeta}"
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

    
    
    print("making input grid and diffmats")
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
    D0 = D0 * scale_vector
    W0 = W0 * scale_vector_inv

    theta = jnp.linspace(0.0, 2 * jnp.pi, n_theta, endpoint=False)
    D1, W1 = fourier_diffmat(n_theta)

    zeta = jnp.linspace(0.0, 2 * jnp.pi / eq.NFP, n_zeta, endpoint=False)
    D2, W2 = fourier_diffmat(n_zeta)

    grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)
    diffmat = DiffMat(D_rho=D0, W_rho=W0, D_theta=D1, W_theta=W1, D_zeta=D2, W_zeta=W2)

    reshaped_nodes = jnp.reshape(
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_rho * n_theta * n_zeta, 3)
    )
    print("mapping coordinates")
    rtz_nodes = map_coordinates(
        eq,
        reshaped_nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )
    print("coordinates mapped")

    grid = Grid(rtz_nodes)

    print("computing eigenmode at low res")
    tic = time.time()
    data = eq.compute(
        "finite-n lambda", grid=grid, diffmat=diffmat,
        gamma=100, incompressible=False,
        axisym=axisym, n_mode_axisym=n_mode_axisym,
    )
    toc = time.time()
    print(f"matrix full took {toc-tic:.1f} s.")
    print(data["finite-n lambda"])

    X = data["finite-n eigenfunction"]
    np.save(save_path + f"low_res_eigenfunction_all_{save_tag}.npy", X)

    idx0 = (n_rho - 2) * n_theta * n_zeta
    idx1 = idx0 + (n_rho) * n_theta * n_zeta

    xi_sup_rho0 = np.reshape(X[:idx0, 0], (n_rho - 2, n_theta, n_zeta))
    xi_sup_rho = np.concatenate(
        (np.zeros((1, n_theta, n_zeta), dtype=xi_sup_rho0.dtype),
         xi_sup_rho0,
         np.zeros((1, n_theta, n_zeta), dtype=xi_sup_rho0.dtype)),
        axis=0,
    )
    xi_sup_rho   = np.concatenate((xi_sup_rho,   xi_sup_rho[:, 0:1, :]),   axis=1)
    xi_sup_rho   = np.concatenate((xi_sup_rho,   xi_sup_rho[:, :, 0:1]),   axis=2)

    xi_sup_theta0 = np.reshape(X[idx0:idx1, 0], (n_rho, n_theta, n_zeta))
    xi_sup_zeta0  = np.reshape(X[idx1:,     0], (n_rho, n_theta, n_zeta))

    xi_sup_theta = np.concatenate((xi_sup_theta0, xi_sup_theta0[:, 0:1, :]), axis=1)
    xi_sup_theta = np.concatenate((xi_sup_theta,  xi_sup_theta[:, :, 0:1]), axis=2)

    xi_sup_zeta  = np.concatenate((xi_sup_zeta0,  xi_sup_zeta0[:, :, 0:1]), axis=2)
    xi_sup_zeta  = np.concatenate((xi_sup_zeta,   xi_sup_zeta[:, 0:1, :]),  axis=1)

    rho_low   = rho
    theta_low = np.concatenate((theta, np.array([2 * np.pi])))
    zeta_low  = np.concatenate((zeta,  np.array([2 * np.pi / eq.NFP])))

    xi_rho_low   = np.asarray(xi_sup_rho)
    xi_theta_low = np.asarray(xi_sup_theta)
    xi_zeta_low  = np.asarray(xi_sup_zeta)
    np.save(save_path + f"xi_rho_low_{save_tag}.npy", xi_rho_low)
    np.save(save_path + f"xi_theta_low_{save_tag}.npy", xi_theta_low)
    np.save(save_path + f"xi_zeta_low_{save_tag}.npy", xi_zeta_low)

    results_lambda_min.append(data["finite-n lambda"][0])


# ── Save summary ──────────────────────────────────────────────────────────────
results_lambda_min = np.array(results_lambda_min)

np.savez(
    save_path + "iota_scan_results.npz",
    iota_on_axis=iota_on_axis_values,
    lambda_min=results_lambda_min,
)

print("Done. Results saved to", save_path)
print("Run analyze_stability.py for Mercier + delta_W breakdown.")
