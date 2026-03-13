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

# Define field type
field_type = "coil"

# Save path
save_path = "phi_matrix/"
fname = f"{field_type}_M_{grid.M}_N_{grid.N}_phi_matrix.npy"
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


if field_type == "greens_function":
    # This field type corresponds to a magnetic monopole
    # It's the field used in Kaya's original test
    # But it's not physical
    x0 = rpz2xyz(np.array([R0, 0, 0]))
    B = _grad_G(data["x"] - x0)
elif field_type == "coil":
    # The field type should be a quadrupole,
    # since it is two currents with opposing sign
    # This also means there should be no net current in ~\mathcal{X}
    # So there are no secular terms in \Phi
    I = 1e6
    coil1 = FourierRZCoil(current=I, R_n=[R0], Z_n=[data["a"] / 2], NFP=1)
    coil2 = FourierRZCoil(current=-I, R_n=[R0], Z_n=[-data["a"] / 2], NFP=1)
    coil = CoilSet([coil1, coil2])

    B = coil.compute_magnetic_field(data["x"], source_grid=LinearGrid(N=grid.N), basis="xyz")

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
