# from desc import set_device
# set_device("gpu")
from desc.grid import Grid
from desc.plotting import plot_surfaces, plot_3d
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
import jax.numpy as jnp

surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.125, -0.1], #alterar 0.1
    Z_lmn=[-0.125, -0.1],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=3,
)
eq = Equilibrium(M=1, N=1, Psi=1, surface=surf)
eq = solve_continuation_automatic(eq, objective="force", bdry_step=0.5, verbose=3)[-1]
#eq.Psi = eq.Psi/19

grid = Grid(jnp.array([jnp.sqrt(0.2), 0, 0]).T, jitable=True, sort=False)
data = eq.compute(["|B|", "R0"], grid=grid)

print(f"Magnetic Field (abs): {data['|B|']}")

eq.save("eq_M1_N1.h5")