from desc.grid import LinearGrid
import matplotlib.pyplot as plt

from desc.integrals.singularities import _kernel_biot_savart
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
from sklearn.linear_model import LinearRegression

# Input parameters
from desc.equilibrium import Equilibrium
from desc.continuation import solve_continuation_automatic
from desc.coils import FourierRZCoil
from desc.profiles import PowerSeriesProfile
from desc.geometry import FourierRZToroidalSurface

I = 1e5  # Toroidal plasma current
R = 6  # Major radius
aspect_ratio = 4
NFP = 3
# Create a circular tokamak
eq = Equilibrium(
    L=3,
    M=3,
    N=3,
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
    current=PowerSeriesProfile([0, 0, I]),
    pressure=PowerSeriesProfile([1.8e4, 0, -3.6e4, 0, 1.8e4]),
    Psi=1.0,
)

# Biot-Savart method won't work without the equilibrium being solved
eq = solve_continuation_automatic(eq)[-1]

M = 24
N = 24
n_theta = n_zeta = N * 2 + 1


from desc.backend import jnp
import jax
from desc.integrals import singular_integral, FFTInterpolator
from desc.integrals.singularities import _kernel_1_over_r, _kernel_nr_over_r3, best_params, best_ratio
from desc.utils import dot
from desc.examples import get
from desc.grid import LinearGrid
import numpy as np

# Singularities matrix method

def _H(eval_data,source_data,diag=False):
    # 1/|r-r'| \hat{n} \cdot \nabla \phi
    G = _kernel_1_over_r(eval_data,source_data,diag)
    f = source_data["B_n"] 
    return f * G
_H.ndim = 1
_H.keys = ["B_n"] + _kernel_1_over_r.keys

def _G(eval_data,source_data, diag=False):
    # \phi \hat{n} \cdot \nabla 1/|r-r'|
    f = source_data["scalar_potential"]
    G = _kernel_nr_over_r3(eval_data,source_data,diag)
    return f * G
_G.ndim = 1
_G.keys = ["scalar_potential"] + _kernel_nr_over_r3.keys


source_grid = LinearGrid(rho=1, theta=n_theta, zeta=n_zeta, NFP=NFP)
data_keys = _kernel_nr_over_r3.keys + _kernel_1_over_r.keys + ["e_theta", "e_zeta", "|e_theta x e_zeta|"]
source_data = eq.compute(data_keys, grid=source_grid)
eval_grid = LinearGrid(rho=1, theta=n_theta, zeta=n_zeta, NFP=NFP)
eval_data = eq.compute(data_keys, grid=eval_grid)
eval_data["B_n"] = 0

st, sz, q = best_params(source_grid, best_ratio(source_data))
H_interpolator = FFTInterpolator(eval_grid, source_grid, st, sz, q)
G_interpolator = FFTInterpolator(eval_grid, eval_grid, st, sz, q)

def H_func(B_n):
    src = source_data.copy()
    src["B_n"] = B_n
    return - singular_integral(eval_data, src, _H, H_interpolator).flatten()

def G_func(phi):
    # takes in a vector of phi values, and returns integral of phi * \hat{n} \cdot \nabla 1/|r-r'|
    # + phi itself, to include diagonal terms
    src = eval_data.copy()
    src["scalar_potential"] = phi
    return 2 * jnp.pi * phi - singular_integral(src, src, _G, G_interpolator).flatten()
H_matrix = jax.vmap(H_func)(jnp.eye(source_grid.num_nodes)).T
G_matrix = jax.vmap(G_func)(jnp.eye(eval_grid.num_nodes)).T

#2 \pi \phi + \int_S \phi \hat{n} \cdot \nabla 1/|r-r'| = \int_S 1/|r-r'| \hat{n} \cdot \nabla \phi

from desc.utils import dot
from desc.coils import FourierRZCoil


coil = FourierRZCoil(current=I, R_n=[R], Z_n=[0], NFP=eq.NFP)
eval_B = coil.compute_magnetic_field(coords=eval_data["x"], basis="rpz")
source_B = coil.compute_magnetic_field(coords=source_data["x"], basis="rpz")

grid = LinearGrid(rho=1.0, theta=n_theta, zeta=n_zeta, NFP=eq.NFP)
source_data = eq.compute(
    ["x", "n_rho", "e^theta", "e^zeta", "e_theta", "e_zeta"],
    grid=source_grid,
    basis="rpz",
)
eval_data = eq.compute(
    ["x", "n_rho", "e^theta", "e^zeta", "e_theta", "e_zeta"],
    grid=eval_grid,
    basis="rpz",
)
n = source_data["n_rho"]
B_n = dot(source_B, n)

phi = jnp.linalg.solve(G_matrix, H_matrix @ B_n.flatten())


basis = DoubleFourierSeries(M=N, N=N, NFP=eq.NFP)
transform = Transform(eval_grid, basis, build_pinv=True, derivs=1)

phi_c = transform.fit(phi)
B_theta = transform.transform(phi_c, dt=1)
B_zeta = transform.transform(phi_c, dz=1)

B_theta_true = dot(eval_B, eval_data["e_theta"])
B_zeta_true = dot(eval_B, eval_data["e_zeta"])


eval_B = coil.compute_magnetic_field(coords=eval_data["x"], basis="rpz")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(B_theta_true, B_theta, linestyle="", marker=".")
model = LinearRegression(fit_intercept=True).fit(B_theta_true[:, None], B_theta)
print(model.coef_, model.intercept_)
ax[0].plot(B_theta_true, B_theta_true * model.coef_[0] + model.intercept_, linestyle="dashed", label="$B^\\theta$")
ax[0].set_xlabel("True $B_\\theta$ (from coil)")
ax[0].set_ylabel("Computed $B_\\theta$ ($\\partial\\phi/\\partial\\theta$)")

ax[1].plot(B_zeta_true,B_zeta, linestyle="", marker=".", label ="$B^\\zeta$")
model = LinearRegression(fit_intercept=False).fit(B_zeta_true[:, None],B_zeta)
print(model.coef_)
ax[1].plot(B_zeta_true, B_zeta_true * model.coef_[0], linestyle="dashed")
ax[1].set_xlabel("True $B_\\zeta$ (from coil)")
ax[1].set_ylabel("Computed $B_\\zeta$ ($\\partial\\phi/\\partial\\zeta$)")
fig.suptitle("singularities.py matrix method to obtain $\\phi$, then $\\nabla \\phi$")

fig.suptitle("singularities.py matrix method to obtain $\\phi$, then $\\nabla \\phi$")

plt.tight_layout()
plt.savefig("singularities_matrix_method.png")
