import numpy as np
from desc.geometry import FourierRZToroidalSurface
from desc.utils import rpz2xyz, dot
from desc.grid import LinearGrid
from desc.magnetic_fields import SourceFreeField
from desc.coils import FourierRZCoil

maxiter = 50
chunk_size = 1000
# Fourier spectrum of G(x) becomes very wide at large R0 (e.g. 10 is large).
R0 = 2
aspect_ratio = 3
NFP = 2
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

x0 = rpz2xyz(np.array([R0, 0, 0]))

grid = LinearGrid(M=128, N=128)
data = surface.compute(["x", "n_rho", "a"], grid=grid, basis="xyz")


coil = FourierRZCoil(current=1E2, R_n=[R0], Z_n=[0], NFP=1)
B = coil.compute_magnetic_field(data["x"])

data = {"B0*n": -dot(B, data["n_rho"])}

field = SourceFreeField(surface, grid.M, grid.N)
data, RpZ_data = field.compute(
    ["∇φ", "Phi", "x", "n_rho"],
    grid,
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

np.savez("coil_phi.npz", **data)

np.testing.assert_allclose(
    data["∇φ"],
    B,
    atol=1e-6,
)

