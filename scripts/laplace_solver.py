import numpy as np
from desc.geometry import FourierRZToroidalSurface
from desc.utils import rpz2xyz, dot
from desc.grid import LinearGrid
from desc.magnetic_fields import SourceFreeField
from desc.coils import FourierRZCoil, CoilSet
from desc.backend import jnp
from desc.batching import vmap_chunked

from desc.integrals.singularities import _grad_G
import os

maxiter = 30
chunk_size = 1000

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
data, RpZ_data  = field.compute(["interpolator", "x", "n_rho", "potential data"], grid)

def phi_func(B_n):
    data_cp = data.copy()
    data_cp["B0*n"] = -B_n
    data_cp, RpZ_data = field.compute(
        ["Phi"],
        grid,
        data=data_cp,
        problem="exterior Neumann ",
        on_boundary=True,
        maxiter=maxiter,
        full_output=True,
        chunk_size=chunk_size,
        basis="xyz",
    )
    return data_cp["Phi"]


phi_matrix = vmap_chunked(phi_func, chunk_size=50)(jnp.eye(grid.num_nodes))


np.save(save_path + fname, phi_matrix)