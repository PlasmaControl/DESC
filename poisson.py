"""
Poisson equation spectral solver in polar coordinates.

Delta u(r,t) = F(r,t)
F(x,y) = F0*e^(-((x-x0)^2+(y-y0)^2)/(2*sigma^2))
on the unit disc r in [0,1], t in [0,2*pi)
with the Dirichlet boundary condition u(1,t) = 0

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from desc.backend import jnp
from desc.utils import copy_coeffs
from desc.derivatives import Derivative
from desc.basis import FourierZernikeBasis
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.transform import Transform
from desc.optimize import Optimizer


class ObjectiveFun:
    """Objective function to solve."""

    def __init__(self, M, index):
        # parameters
        self.F0 = -1
        self.r0 = 1 / np.e
        self.t0 = np.pi / np.e
        self.sigma = 0.1

        # grid, basis, transform
        basis = FourierZernikeBasis(M=M, N=0, index=index)
        grid = QuadratureGrid(L=np.ceil((basis.L + 1) / 2), M=(2 * basis.M + 1), N=1)
        self.transform = Transform(grid, basis, derivs=2)
        if index == "ansi":
            self.idx = np.squeeze(np.argwhere(np.abs(basis.modes[:, 1]) < basis.M))
        else:
            self.idx = np.squeeze(np.argwhere(
                basis.modes[:, 0] + np.abs(basis.modes[:, 1]) < basis.L))

        # Dirichlet boundary condition
        A = np.zeros((2 * basis.M + 1, basis.num_modes))
        b = np.zeros((2 * basis.M + 1,))
        for l, m, n in basis.modes:
            j = np.argwhere(basis.modes[:, 1] == m)
            A[m + basis.M, j] = 1
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        K = min(M, N)
        rcond = np.finfo(A.dtype).eps * max(M, N)
        tol = np.amax(s) * rcond
        large = s > tol
        num = np.sum(large, dtype=int)
        uk = u[:, :K]
        vhk = vh[:K, :]
        s = np.divide(1, s, where=large, out=s)
        s[~large] = 0
        Ainv = np.matmul(
            np.transpose(vhk), np.multiply(s[..., np.newaxis], np.transpose(uk))
        )
        self.x0 = jnp.dot(Ainv, b)
        self.Z = vh[num:, :].T.conj()

        self.grad = Derivative(self.compute_scalar, mode="grad")
        self.hess = Derivative(self.compute_scalar, mode="hess")
        self.jac = Derivative(self.compute, mode="fwd")

    def recover(self, x):
        """Recover full state vector from optimization variables."""
        return self.x0 + jnp.dot(self.Z, x)

    def project(self, x):
        """Project full state vector onto optimization space."""
        dx = x - self.x0
        return jnp.squeeze(jnp.dot(self.Z.T, dx))

    def residual(self, x):
        """Retrun f(x) at nodes."""
        xx = self.recover(x)

        r = self.transform.grid.nodes[:, 0]
        t = self.transform.grid.nodes[:, 1]

        u_rr = self.transform.transform(xx, 2, 0)
        u_r = self.transform.transform(xx, 1, 0)
        u_tt = self.transform.transform(xx, 0, 2)

        Lu = u_rr + u_r / r + u_tt / r ** 2
        F = self.F0 * jnp.exp((-r ** 2 - self.r0 ** 2 + 2 * r * self.r0 * (
           jnp.cos(t) * jnp.cos(self.t0) + jnp.sin(t) * jnp.sin(self.t0))) / (
           2 * self.sigma ** 2))
        return Lu - F

    def compute(self, x):
        """Return f(x) coefficients."""
        f = self.residual(x)
        w = self.transform.grid.weights / (2*jnp.pi)
        A = self.transform._matrices[0][0][0]
        r = jnp.sum(jnp.atleast_2d(f * w).T * A, axis=0)
        return r[self.idx]

    def compute_scalar(self, x):
        """Return the integral of f(x) over the unit disc."""
        f = self.residual(x)
        return jnp.sum(jnp.abs(f)*self.transform.grid.weights) / (2*jnp.pi)

    def grad_x(self, x):
        """Return gradient of scalar objective."""
        return self.grad.compute(x)

    def hess_x(self, x):
        """Return Hessian of scalar objective."""
        return self.hess.compute(x)

    def jac_x(self, x):
        """Return Jacobian of objective function."""
        return self.jac.compute(x)

    @property
    def scalar(self):
        """Not a scalar objective function."""
        return False


M = 16

# ANSI index
results_ansi = {M: np.arange(2, M + 1)}
for m in np.arange(2, M + 1):
    print("\n\nM = {}".format(m))

    objective = ObjectiveFun(m, "ansi")
    optimizer = Optimizer("scipy-trf", objective)

    if m == 2:
        x0 = jnp.zeros((objective.transform.basis.num_modes,))
    else:
        x0 = copy_coeffs(
            results_ansi[m - 1]["x"], old_modes, objective.transform.basis.modes
        )
    old_modes = objective.transform.basis.modes
    x0 = jnp.atleast_1d(objective.project(x0))

    result = optimizer.optimize(
        x_init=x0,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        verbose=3,
        maxiter=100,
        # options={"return_all": True},
    )
    result["allx"] = (result["x"],)
    result["x"] = objective.recover(result["x"])
    result["allf"] = np.zeros((len(result["allx"]),))
    grid = Grid(np.array([[objective.r0, objective.t0, 0]]))
    objective.transform.grid = grid
    for k in range(len(result["allx"])):
        result["allf"][k] = np.mean(np.abs(objective.residual(result["allx"][k])))
    results_ansi[m] = result

# plot ANSI result
grid = LinearGrid(L=50, M=91, endpoint=True)
transform = Transform(grid, objective.transform.basis)
x = (grid.nodes[:, 0] * np.cos(grid.nodes[:, 1])).reshape((grid.L, grid.M))
y = (grid.nodes[:, 0] * np.sin(grid.nodes[:, 1])).reshape((grid.L, grid.M))
u = transform.transform(results_ansi[M]["x"]).reshape((grid.L, grid.M))
fig, ax = plt.subplots()
norm = matplotlib.colors.Normalize()
levels = np.linspace(u.min(), u.max(), 100)
cax_kwargs = {"size": "5%", "pad": 0.05}
divider = make_axes_locatable(ax)
cntr = ax.contourf(x, y, u, levels=levels, cmap="jet", norm=norm)
cax = divider.append_axes("right", **cax_kwargs)
cbar = fig.colorbar(cntr, cax=cax)
cbar.update_ticks()
ax.axis("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("$u(\\rho,\\theta)$  M={}  ANSI".format(M))
fig.set_tight_layout(True)

# fringe index
results_fringe = {M: np.arange(1, M + 1)}
for m in np.arange(1, M + 1):
    print("\n\nM = {}".format(m))

    objective = ObjectiveFun(m, "fringe")
    optimizer = Optimizer("scipy-trf", objective)

    if m == 1:
        x0 = jnp.zeros((objective.transform.basis.num_modes,))
    else:
        x0 = copy_coeffs(
            results_fringe[m - 1]["x"], old_modes, objective.transform.basis.modes
        )
    old_modes = objective.transform.basis.modes
    x0 = jnp.atleast_1d(objective.project(x0))

    result = optimizer.optimize(
        x_init=x0,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        verbose=3,
        maxiter=100,
        # options={"return_all": True},
    )
    result["allx"] = (result["x"],)
    result["x"] = objective.recover(result["x"])
    result["allf"] = np.zeros((len(result["allx"]),))
    grid = Grid(np.array([[objective.r0, objective.t0, 0]]))
    objective.transform.grid = grid
    for k in range(len(result["allx"])):
        result["allf"][k] = np.mean(np.abs(objective.residual(result["allx"][k])))
    results_fringe[m] = result

# plot fringe result
grid = LinearGrid(L=50, M=91, endpoint=True)
transform = Transform(grid, objective.transform.basis)
x = (grid.nodes[:, 0] * np.cos(grid.nodes[:, 1])).reshape((grid.L, grid.M))
y = (grid.nodes[:, 0] * np.sin(grid.nodes[:, 1])).reshape((grid.L, grid.M))
u = transform.transform(results_fringe[M]["x"]).reshape((grid.L, grid.M))
fig, ax = plt.subplots()
norm = matplotlib.colors.Normalize()
levels = np.linspace(u.min(), u.max(), 100)
cax_kwargs = {"size": "5%", "pad": 0.05}
divider = make_axes_locatable(ax)
cntr = ax.contourf(x, y, u, levels=levels, cmap="jet", norm=norm)
cax = divider.append_axes("right", **cax_kwargs)
cbar = fig.colorbar(cntr, cax=cax)
cbar.update_ticks()
ax.axis("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("$u(\\rho,\\theta)$  M={}  Fringe".format(M))
fig.set_tight_layout(True)

# plot coefficients
basis_ansi = FourierZernikeBasis(M=M, N=0, index="ansi")
basis_fringe = FourierZernikeBasis(M=M, N=0, index="fringe")
fig, ax = plt.subplots()
ax.semilogy(np.sum(np.abs(basis_ansi.modes), axis=1), np.abs(results_ansi[M]["x"]),
            "bo", label="ANSI")
ax.semilogy(np.sum(np.abs(basis_fringe.modes), axis=1), np.abs(results_fringe[M]["x"]),
            "ro", label="Fringe")
ax.set_ylim(1e-10, 1e-2)
ax.legend()
ax.set_xlabel("$l + |m|$")
ax.set_ylabel("$|a^m_l|$")
fig.set_tight_layout(True)

m_vec_ansi = np.zeros((M - 1,))
m_vec_fringe = np.zeros((M,))
f_vec_ansi = np.zeros((M - 1,))
f_vec_fringe = np.zeros((M,))
for m in np.arange(2, M + 1):
    m_vec_ansi[m - 2] = (m + 1) * (m + 2) / 2
    f_vec_ansi[m - 2] = results_ansi[m]["allf"][-1]
for m in np.arange(1, M + 1):
    m_vec_fringe[m - 1] = (m + 1) ** 2
    f_vec_fringe[m - 1] = results_fringe[m]["allf"][-1]

# plot convergence vs M
fig, ax = plt.subplots()
ax.semilogy(np.arange(2, M + 1), f_vec_ansi, "bo", label="ANSI")
ax.semilogy(np.arange(1, M + 1), f_vec_fringe, "ro", label="Fringe")
ax.legend()
ax.set_xlabel("M")
ax.set_ylabel("$|f(\\rho_0,\\theta_0)|$")
fig.set_tight_layout(True)

# plot convergence vs dimension
fig, ax = plt.subplots()
ax.semilogy(m_vec_ansi, f_vec_ansi, "bo", label="ANSI")
ax.semilogy(m_vec_fringe, f_vec_fringe, "ro", label="Fringe")
ax.legend()
ax.set_xlabel("number of coefficients")
ax.set_ylabel("$|f(\\rho_0,\\theta_0)|$")
fig.set_tight_layout(True)

# plot convergence vs iteration
# fig, ax = plt.subplots()
# for m in np.arange(M0, M + 1):
#     ax.semilogy(
#         np.arange(len(results[m]["allf"])),
#         results[m]["allf"],
#         "o",
#         label="M = " + str(m),
#     )
# ax.legend()
# ax.set_xlabel("iteration")
# ax.set_ylabel("$|f(\\rho_0,\\theta_0)|$")
# fig.set_tight_layout(True)
