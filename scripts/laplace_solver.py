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
from desc.utils import dot
from desc.integrals.quad_utils import leggauss_lob, automorphism_staircase1
from desc.io import load
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from desc.transform import Transform


# Make or load an ultra high aspect-ratio tokamak (essentially a screw pinch)
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.grid import QuadratureGrid
import os

from newcomb import *

from desc import set_device
set_device("gpu")

from desc.integrals.singularities import _grad_G
import os

chunk_size = 30

# Equilibrium paremeteters
a = 1  # Minor radius
aspect_ratio = 2  # Aspect ratio of the tokamak
R0 = aspect_ratio * a  # Major radius
NFP = 1  # Number of field periods
axisym = False  # Whether to enforce axisymmetry in the eigenvalue solve
n_mode_axisym = 0 # If axisym is True, the toroidal mode number to solve for

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

# Define resolution
M = 7
N = 5
n_rho = 14
n_theta = 2 * M
if axisym:
    n_zeta = 1
else:
    n_zeta = 2 * N

# Save path
save_path = "phi_matrix/"
profile_tag = f"iota_{"_".join(iota_coeffs.astype(str))}" if fixed_iota else f"I_{"_".join(I_coeffs.astype(str))}"
save_tag = f"axisym_{axisym}_ar_{aspect_ratio}_NFP_{NFP}_p_{"_".join(p_coeffs.astype(str))}_{profile_tag}"
eq_save_name = f"equilibrium_{save_tag}.h5"
phi_save_name = f"{save_tag}_M_{M}_N_{N}_phi_matrix.npy"
os.makedirs(save_path, exist_ok=True)


# Make surface
surface = FourierRZToroidalSurface.from_shape_parameters(
    major_radius=R0,
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
assert surface.NFP == 1

# Precompute interpolator and surface values
field = SourceFreeField(surface, M, N)

# NOTE: equilibrium LCFS must be ForceFreeField object 
if os.path.exists(save_path + eq_save_name):
    eq = load(save_path + eq_save_name)
else:
    eq = Equilibrium(
            L=12, 
            M=12,
            N=0,
            surface=field,
            NFP=field.NFP,
            iota = iota_profile,
            current=I_profile,
            pressure=p_profile,
            Psi=1,
        )
    eq.save(save_path + eq_save_name)

# Evaluate stability using Rahul's method
# The rest of the script is basically unchanged from what Rahul sent me
# resolution for low-res solve

# This will probably OOM with the matrix-full method
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
grid = Grid(rtz_nodes)

n_surf = n_theta * n_zeta

# Surface nodes (rho=1) are the last n_surf entries of rtz_nodes, in AGNI C order
# (flat index = theta_i * n_zeta + zeta_k).
# FFT/BIEST requires DESC Fortran order (flat index = theta_i + n_theta * zeta_k),
# so we permute before building the surface grid.
surface_nodes_agni = np.array(rtz_nodes[-n_surf:])
desc_flat = np.arange(n_surf)
perm_to_desc = (desc_flat % n_theta) * n_zeta + (desc_flat // n_theta)
rtz_surface_grid = Grid(surface_nodes_agni[perm_to_desc], NFP=NFP)

# PEST grid: uniform in (theta_PEST, zeta) at rho=1, required by BIEST interpolator
pest_grid = LinearGrid(rho=1.0, theta=n_theta, zeta=n_zeta, NFP=NFP, sym=False)

print("computing phi matrix")
# Compute the matrix A such that Phi_periodic = A @ B0*n.
print(eq.surface)
if os.path.exists(save_path + phi_save_name):
    phi_matrix = np.load(save_path + phi_save_name)
else:
    # Equilibrium doesn't expose Phi_basis directly; get it from the SourceFreeField surface
    #phi_transform = Transform(eq.surface.Phi_basis, rtz_surface_grid)
    data_phi = eq.surface.compute(
        ["phi_matrix_pest"],
        rtz_surface_grid,
        pest_grid=pest_grid,
        problem="exterior Neumann",
        chunk_size=chunk_size,
        #transforms={"Phi": phi_transform},
    )
    # phi_matrix_pest is in DESC Fortran order (theta fastest).
    # Reorder to AGNI C order (zeta fastest) expected by the stability solver.
    phi_matrix_desc = np.array(data_phi["phi_matrix_pest"])
    phi_matrix = phi_matrix_desc.reshape(
        n_theta, n_zeta, n_theta, n_zeta, order="F"
    ).reshape(n_surf, n_surf, order="C")

    np.save(save_path + phi_save_name, phi_matrix)



print("computing eigenmode at low res")
tic = time.time()
data = eq.compute("finite-n lambda", grid=grid, diffmat=diffmat, gamma=100, incompressible=False, axisym=axisym, n_mode_axisym=n_mode_axisym, phi_matrix=phi_matrix)
toc = time.time()
print(f"matrix full took {toc-tic} s.")

print(data["finite-n lambda"])
X = data["finite-n eigenfunction"]

np.save(save_path + f"low_res_eigenfunction_all_{save_tag}.npy", X)
np.save(save_path + f"low_res_eigenvalue_all_{save_tag}.npy", data["finite-n lambda"])