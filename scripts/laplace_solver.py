from desc import set_device
set_device("gpu")

import numpy as np
from desc.geometry import FourierRZToroidalSurface
from desc.utils import rpz2xyz
from desc.grid import LinearGrid
from desc.magnetic_fields import SourceFreeField
from desc.coils import FourierRZCoil, CoilSet

from desc.integrals.singularities import _grad_G
import os

chunk_size = 100

# Define surface parameters
R0 = 2
aspect_ratio = 2

# Define poloidal and toroidal resolution
grid = LinearGrid(M=32, N=32)

# Save path
save_path = "phi_matrix/"
fname = f"M_{grid.M}_N_{grid.N}_phi_matrix.npy"
os.makedirs(save_path, exist_ok=True)

# Make surface
a = R0/aspect_ratio
surface = FourierRZToroidalSurface.from_shape_parameters(
    major_radius=R0,
    aspect_ratio=aspect_ratio,
    elongation=1,
    triangularity=0,
    squareness=0,
    eccentricity=0,
    torsion=0,
    twist=0,
    NFP=1,
    sym=True,
)
assert surface.NFP == 1

# Compute on-surface data
data_keys = ["x", "n_rho", "a"]
data = surface.compute(data_keys, grid=grid, basis="xyz")

# Precompute interpolator and surface values
field = SourceFreeField(surface, grid.M, grid.N)
data, RpZ_data = field.compute(["interpolator", "x", "n_rho", "potential data"], grid)

# Compute the matrix A such that Phi_periodic = A @ B0*n.
# The old convention was phi_func(B_n) with B0*n = -B_n, so phi_matrix = -A.
data, _ = field.compute(
    ["phi_matrix"],
    grid,
    data=data,
    problem="exterior Neumann",
    chunk_size=chunk_size,
)
phi_matrix = -data["phi_matrix"]


np.save(save_path + fname, phi_matrix)
