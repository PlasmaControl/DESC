# %%
import numpy as np
from scipy.constants import mu_0

# import pytest
# from scipy.io import netcdf_file
# from scipy.signal import convolve2d

# from desc.compute import data_index, rpz2xyz_vec
from desc.equilibrium import EquilibriaFamily, Equilibrium

# from desc.examples import get
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierRZToroidalSurface,
    FourierXYZCurve,
    ZernikeRZToroidalSection,
)
from desc.grid import LinearGrid, QuadratureGrid
from desc.profiles import PowerSeriesProfile

# %% test1
eq = Equilibrium()

B = eq.compute("B")["B"]

# %% test2

eq = Equilibrium(L=3, M=3)
grid = LinearGrid(L=3, M=3)
rho = grid.nodes[:, 0]
theta = grid.nodes[:, 1]
a0 = 4 / 5
a1 = 1 / 5
R = 10 + (a0 * rho + a1 * (3 * rho**3 - 2 * rho)) * np.cos(theta)
Z = -(a0 * rho + a1 * (3 * rho**3 - 2 * rho)) * np.sin(theta)
eq.set_initial_guess(grid, R, Z)
B = eq.compute("B", grid=grid)["B"]


def B_ana_cal(rho, Psi0, a0, a1):
    return (
        Psi0
        / np.pi
        / (a0 + a1 * (3 * rho**2 - 2))
        / (a0 - 2 * a1 + 9 * a1 * rho**2)
    )


B_ana = B_ana_cal(rho, eq.Psi, a0, a1)

# %%

Psi = 1
a = 1
R = 10
A = ((2 * Psi) / (np.pi * a**2 * (np.sqrt(2) + 1))) ** 2
p_coeff = A / 2 / mu_0

p = PowerSeriesProfile(params=[p_coeff, -p_coeff], modes=[0, 2])

iota = PowerSeriesProfile(params=[0, 0])

surf = FourierRZToroidalSurface(
    R_lmn=[R, a],
    modes_R=[[0, 0], [1, 0]],
    Z_lmn=[0, -a],
    modes_Z=[[0, 0], [-1, 0]],
)
# %%

eq = Equilibrium(
    surface=surf,
    pressure=p,
    iota=iota,
    Psi=Psi,
    NFP=1,
    L=6,
    M=3,
    N=0,
    L_grid=12,
    M_grid=6,
    N_grid=0,
)

# %%
from desc.continuation import solve_continuation_automatic

solve_continuation_automatic(eq)

# %%

grid = LinearGrid(rho=np.linspace(0, 1, 10), theta=[np.pi * 0], zeta=[0])

data = eq.compute(["R", "Z"], grid=grid)
r = np.linalg.norm(np.stack((data["R"] - 10, data["Z"]), axis=1), axis=1)

#%%
def r_test(rho, a):
    return np.sqrt(np.sqrt(2) + 1) * a * np.sqrt(np.sqrt(1 + rho**2) - 1)
#%%
r_test(grid.nodes[:,0],a)