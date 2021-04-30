"""Testing QS perturbations."""

import numpy as np
from desc.equilibrium import EquilibriaFamily
from desc.objective_funs import QuasisymmetryTripleProduct
from desc.transform import Transform
from desc.grid import LinearGrid
from desc.plotting import plot_surfaces, plot_2d


iters = 1
levels = np.logspace(-4, -1, num=7)

fam = EquilibriaFamily.load("examples/DESC/ITER_iota-98_output.h5")
eq = fam[-1]
plot_surfaces(eq, nzeta=4)
eq.change_resolution(N=1, N_grid=3)

qs_grid = LinearGrid(M=2 * eq.M_grid + 1, N=2 * eq.N + 1, NFP=eq.NFP, rho=0.75)
plot_grid = LinearGrid(M=100, N=100, NFP=eq.NFP, endpoint=True, rho=0.75)

R_transform = Transform(qs_grid, eq.R_basis)
Z_transform = Transform(qs_grid, eq.Z_basis)
L_transform = Transform(qs_grid, eq.L_basis)
Rb_transform = Transform(qs_grid, eq.Rb_basis)
Zb_transform = Transform(qs_grid, eq.Zb_basis)
p_transform = Transform(qs_grid, eq.p_basis)
i_transform = Transform(qs_grid, eq.i_basis)
qs_fun = QuasisymmetryTripleProduct(
    eq.transforms["R"],
    eq.transforms["Z"],
    eq.transforms["L"],
    eq.transforms["Rb"],
    eq.transforms["Zb"],
    eq.transforms["p"],
    eq.transforms["i"],
    eq.constraint,
)
dRb = np.invert((eq.Rb_basis.modes == [0, 0, 0]).all(axis=1))

eq_opt = eq.perturb(
    objective=qs_fun,
    dRb=eq.Rb_basis.modes[:, 2] != 0,
    dZb=eq.Zb_basis.modes[:, 2] != 0,
    order=2,
    verbose=2,
    copy=True,
)
eq_opt.solve(ftol=1e-4, xtol=1e-6, gtol=1e-6, maxiter=500, verbose=2)


# for i in range(iters):
  #  eq = eq.copy()
  #  fam.insert(len(fam), eq)
  #  eq.perturb(objective=qs_fun, dRb=dRb, dZb=True, order=1, verbose=2, copy=False)
  #  eq.solve(ftol=1e-3, xtol=1e-3, gtol=1e-3, maxiter=100, verbose=2)
    # plot_2d(eq, "QS_TP", grid=plot_grid, log=True, levels=levels)

# plot_surfaces(eq, nzeta=4)


# import matplotlib.pyplot as plt

# plt.rcParams.update({"font.size": 20})
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# x = np.array([0, 1, 2, 3])
# y = np.array(
#     [
#         0.016760120039801756,
#         0.009529658046763602,
#         0.0032032574422849167,
#         0.0014210420367508677,
#     ]
# )
# ax.semilogy(x, y, "bo-")
# ax.set_xticks(x)
# ax.set_ylim(1e-3, 2e-2)
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Flux surface average $g(x,c)$ $(T^4/m^2)$")
# ax.set_title("Quasisymmetry error at $\\rho=0.9$")
# fig.set_tight_layout(True)
