# from desc import set_device
# set_device("gpu")
from desc.grid import Grid
from desc.plotting import plot_surfaces, plot_3d
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
import jax.numpy as jnp

surf = FourierRZToroidalSurface(
    R_lmn=jnp.array([1, -0.1, 0.06, 0.07]),  # boundary coefficients # conditions to eq_2411_M1_N1.h5
    Z_lmn=jnp.array([0.1, -0.03, -0.03]),
    modes_R=jnp.array(
        [[0, 0], [1, 0], [1, 1], [-1, -1]]
    ),  # [M, N] boundary Fourier modes
    modes_Z=jnp.array([[-1, 0], [-1, 1], [1, -1]]),
    NFP=5,  # number of (toroidal) field periods
)
eq = Equilibrium(M=1, N=1, Psi=1, surface=surf)
eq = solve_continuation_automatic(eq, objective="force", bdry_step=0.5, verbose=3)[-1]

eq.save("eq_0108_M1_N1.h5")