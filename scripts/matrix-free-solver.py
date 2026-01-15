#!/usr/bin/env python3
from desc import set_device
set_device("gpu")

import pdb
import time
from desc.examples import get
from desc.grid import LinearGrid, Grid
from desc.diffmat_utils import *
from desc.equilibrium.coords import map_coordinates
from desc.integrals.quad_utils import leggauss_lob
from desc.equilibrium import Equilibrium
from desc.backend import jnp, jax
from matplotlib import pyplot as plt
from desc.utils import dot
from desc.integrals.quad_utils import leggauss_lob, automorphism_staircase1
from desc.io import load
import numpy as np
from scipy.interpolate import RegularGridInterpolator


# make or load an ultra high aspect-ratio tokamak (essentially a screw pinch)
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.grid import QuadratureGrid
import os

# Input parameters
I = 1e5  # Toroidal plasma current
R = 4  # Major radius
aspect_ratio = 25  # Aspect ratio of the tokamak
a = R / aspect_ratio  # Minor radius
save_path = "./high_aspect_ratio_tokamak/"
save_name = f"tokamak_AR{aspect_ratio}_I{I}_R{R}.h5"
os.makedirs(save_path, exist_ok=True)

solovev = get("solovev")
override = True
if os.path.exists(save_path + save_name) and not override:
    eq = load(save_path + save_name)
else:
    p_coeffs = np.array([1.8e4, 0, -3.6e4, 0, 1.8e4])  # Pressure profile coefficients
    # Create a very high aspect ratio tokamak
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
            NFP=solovev.NFP,
            sym=solovev.sym,
        ),
        NFP=1,
        #current=PowerSeriesProfile([0, 0, I]),
        iota = solovev.iota.copy(),#PowerSeriesProfile([0.3, 0.0, 0.2]),
        pressure=solovev.pressure.copy(),#PowerSeriesProfile(p_coeffs),
        Psi=solovev.Psi,
    )

    # Solve equilbrium
    eq = solve_continuation_automatic(eq, ftol=1E-13, gtol=1E-13, xtol=1E-13)[-1]
    eq.save(save_path + save_name)

#eq = get("SOLOVEV")

# resolution for low-res solve
n_rho = 26
n_theta = 32
n_zeta = 9

# This will probably OOM with the matrix-full method
#n_rho = 48
#n_theta = 32
#n_zeta = 10

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

grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)

diffmat = DiffMat(D_rho=D0, W_rho=W0, D_theta=D1, W_theta=W1, D_zeta=D2, W_zeta=W2)

# reshaped according to rho
reshaped_nodes = jnp.reshape(
    grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_rho * n_theta * n_zeta, 3)
)
rtz_nodes = map_coordinates(
    eq,
    reshaped_nodes,  # (ρ,θ_PEST,ζ)
    inbasis=("rho", "theta_PEST", "zeta"),
    outbasis=("rho", "theta", "zeta"),
    period=(jnp.inf, 2 * jnp.pi, jnp.inf),
    tol=1e-12,
    maxiter=50,
)
grid = Grid(rtz_nodes)

data = eq.compute("finite-n lambda", grid=grid, diffmat=diffmat, incompressible=False, gamma=100)

print(data["finite-n lambda"])
X = data["finite-n eigenfunction"]

idx0 = (n_rho - 2) * n_theta * n_zeta
idx1 = idx0 + (n_rho) * n_theta * n_zeta

# xi^rho
xi_sup_rho0 = np.reshape(X[:idx0, 0], (n_rho - 2, n_theta, n_zeta))
xi_sup_rho = np.concatenate(
    (np.zeros((1, n_theta, n_zeta), dtype=xi_sup_rho0.dtype), xi_sup_rho0, np.zeros((1, n_theta, n_zeta), dtype=xi_sup_rho0.dtype)),
    axis=0,
)

xi_sup_rho = np.concatenate((xi_sup_rho, xi_sup_rho[:, 0:1, :]), axis=1)
xi_sup_rho = np.concatenate((xi_sup_rho, xi_sup_rho[:, :, 0:1]), axis=2)

xi_sup_theta0 = np.reshape(X[idx0:idx1, 0], (n_rho, n_theta, n_zeta))
xi_sup_zeta0 = np.reshape(X[idx1:, 0], (n_rho, n_theta, n_zeta))

xi_sup_theta = np.concatenate((xi_sup_theta0, xi_sup_theta0[:, 0:1, :]), axis=1)
xi_sup_theta = np.concatenate((xi_sup_theta, xi_sup_theta[:, :, 0:1]), axis=2)

xi_sup_zeta = np.concatenate((xi_sup_zeta0, xi_sup_zeta0[:, :, 0:1]), axis=2)
xi_sup_zeta = np.concatenate((xi_sup_zeta, xi_sup_zeta[:, 0:1, :]), axis=1)

psi_r = np.reshape(data["psi_r"], (n_rho, n_theta, n_zeta))

#rtz_nodes
#plt.plot(rho, xi_sup_rho[:, :, 0] * psi_r[:, :, 0], "-or")
##plt.plot(rho, xi_sup_rho[:, :, 0] , '-or');
#plt.figure()
#plt.plot(theta, xi_sup_rho[:, :, 0].T * psi_r[:, :, 0].T, "-og")
##plt.plot(theta, xi_sup_rho[:, :, 0].T, '-og');
#plt.show()

# SAVE low-res grids & eigenfunction components (to upscale later)
rho_low = rho
theta_low = np.concatenate((theta, np.array([2*np.pi])))
zeta_low = np.concatenate((zeta, np.array([2*np.pi/eq.NFP])))

xi_rho_low = np.asarray(xi_sup_rho)       # (n_rho, n_theta, n_zeta)
xi_theta_low = np.asarray(xi_sup_theta)   # (n_rho, n_theta, n_zeta)
xi_zeta_low = np.asarray(xi_sup_zeta)     # (n_rho, n_theta, n_zeta)


#######################################################
###----Interpolate and upscale the eigenfunction----###
#######################################################

n_rho = 64
n_theta = 64
n_zeta = 12

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

# rho, theta, zeta are in PEST coordinates
grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)

diffmat = DiffMat(D_rho=D0, W_rho=W0, D_theta=D1, W_theta=W1, D_zeta=D2, W_zeta=W2)

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


# -------------------------
# UPSCALE: 3D interpolation on (rho,theta,zeta),
# periodic extension in theta/zeta, NO FFT.
# -------------------------
Ltheta = 2.0 * np.pi
Lzeta = 2.0 * np.pi / eq.NFP

rho_hi = np.asarray(rho)
theta_hi = np.asarray(theta)
zeta_hi = np.asarray(zeta)

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

tic = time.time()
data = eq.compute(
    "finite-n lambda matfree",
    grid=grid,
    diffmat=diffmat,
    incompressible=False,
    gamma=100,
    v_guess=v_guess,
)

print(data["finite-n lambda matfree"])
v = data["finite-n eigenfunction matfree"]

n_total = n_rho * n_theta * n_zeta
xi_sup_rho_final = np.reshape(v[:n_total], (n_rho, n_theta, n_zeta))
xi_sup_theta_final = np.reshape(v[n_total:2*n_total], (n_rho, n_theta, n_zeta))
xi_sup_zeta_final = np.reshape(v[2*n_total:], (n_rho, n_theta, n_zeta))

toc = time.time()
print(f"matrix free took {toc-tic} s.")

print(v.min())

