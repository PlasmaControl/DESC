"""
Poisson equation spectral solver in polar coordinates.

Delta u(r,t) = F(r,t)
where F(r,t) = F0*e^(-(r-r0)^2/(2*sigma^2))
with the Dirichlet boundary condition u(1,t) = 0
on the unit disc r in [0,1], t in [0,2*pi)

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from desc.backend import jnp
from desc.utils import copy_coeffs
from desc.derivatives import Derivative
from desc.basis import FourierZernikeBasis
from desc.grid import LinearGrid, ConcentricGrid, QuadratureGrid
from desc.transform import Transform
from desc.optimize import Optimizer


class ObjectiveFun:
    """Objective function to solve."""

    def __init__(self, M, index, nodes):
        # parameters
        self.F0 = 1
        self.r0 = 1 / np.e
        self.t0 = np.pi / np.e
        self.sigma = 0.1

        # grid, basis, transform
        basis = FourierZernikeBasis(M=M, N=0, index=index)
        if nodes == "quad":
            grid = QuadratureGrid(L=(basis.L + 1) / 2, M=2 * basis.M + 1, N=1)
        else:
            grid = ConcentricGrid(M=M, N=0, surfs="jacobi")
        self.transform = Transform(grid, basis, derivs=2)

        # Dirichlet boundary condition
        A = np.zeros((2 * basis.M + 1, basis.num_modes))
        for l, m, n in basis.modes:
            j = np.argwhere(basis.modes[:, 1] == m)
            A[m + basis.M, j] = 1
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        rcond = np.finfo(A.dtype).eps * max(M, N)
        tol = np.amax(s) * rcond
        large = s > tol
        num = np.sum(large, dtype=int)
        self.Z = vh[num:, :].T.conj()

        self.jac = Derivative(self.compute, mode="fwd")
        self.grad = Derivative(self.compute_scalar, mode="grad")

    def compute(self, x):
        """Return f(x)."""
        xx = jnp.squeeze(jnp.dot(self.Z, x))
        r = self.transform.grid.nodes[:, 0]

        u_rr = self.transform.transform(xx, 2, 0)
        u_r = self.transform.transform(xx, 1, 0)
        u_tt = self.transform.transform(xx, 0, 2)

        Lu = u_rr + u_r / r + u_tt / r ** 2
        F = self.F0 * jnp.exp(-((r - self.r0) ** 2) / (2 * self.sigma ** 2))
        return Lu - F

    def compute_scalar(self, x):
        """Return 1/2 sum( f(x)^2 )."""
        residual = self.compute(x)
        return 1 / 2 * jnp.sum(residual ** 2)

    def jac_x(self, x):
        """Return df/dx."""
        return self.jac.compute(x)

    def grad_x(self, x):
        """Return df/dx."""
        return self.grad.compute(x)

    @property
    def scalar(self):
        """Not a scalar objective function."""
        return False


M = 3
index = "fringe"
nodes = "jacobi"

results = {M: np.arange(1, M + 1)}
for m in np.arange(1, M + 1):

    objective = ObjectiveFun(m, index, nodes)
    optimizer = Optimizer("lsq-exact", objective)

    if m == 1:
        xx0 = jnp.zeros((objective.transform.basis.num_modes,))
    else:
        xx0 = copy_coeffs(
            results[m - 1]["x"], old_modes, objective.transform.basis.modes
        )
    old_modes = objective.transform.basis.modes
    x0 = jnp.atleast_1d(jnp.squeeze(jnp.dot(objective.Z.T, xx0)))

    result = optimizer.optimize(
        x_init=x0,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=3,
        maxiter=None,
        options={"return_all": True},
    )
    result["x"] = np.squeeze(np.dot(objective.Z, result["x"]))
    result["allf"] = np.zeros((len(result["allx"]),))
    for k in range(len(result["allx"])):
        result["allf"][k] = objective.compute_scalar(result["allx"][k])
    results[m] = result


# plot convergence
fig, ax = plt.subplots()
for m in np.arange(1, M + 1):
    ax.semilogy(
        np.arange(len(results[m]["allf"])),
        results[m]["allf"],
        "o",
        label="M = " + str(m),
    )
ax.legend()
ax.set_xlabel("iteration")
ax.set_ylabel("$\\Sigma f(x)^2 / 2$")
fig.set_tight_layout(True)

# plot final result
grid = LinearGrid(L=50, M=91, endpoint=True)
transform = Transform(grid, objective.transform.basis)
x = (grid.nodes[:, 0] * np.cos(grid.nodes[:, 1])).reshape((grid.L, grid.M))
y = (grid.nodes[:, 0] * np.sin(grid.nodes[:, 1])).reshape((grid.L, grid.M))
u = transform.transform(result["x"]).reshape((grid.L, grid.M))
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
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("u")
fig.set_tight_layout(True)
