"""Testing QS perturbations."""

import numpy as np
from desc.equilibrium import EquilibriaFamily
from desc.objective_funs import QuasisymmetryTripleProduct
from desc.transform import Transform
from desc.grid import LinearGrid
# from desc.plotting import plot_surfaces, plot_2d


rho = 0.9  # surface to optimize
order = 2  # optimization order
iters = 3  # optimization iterations

fam = EquilibriaFamily.load("examples/DESC/HELIOTRON_QS2_r90.h5")
eq = fam[-1]
qs_grid = LinearGrid(M=2 * eq.M_grid + 1, N=2 * eq.N_grid + 1, NFP=eq.NFP, rho=rho)
# plot_grid = LinearGrid(M=100, N=100, NFP=eq.NFP, endpoint=True, rho=rho)

# levels = np.logspace(-4, -1, num=7)
# plot_surfaces(eq, nzeta=4)
# plot_2d(eq, "QS_TP", grid=plot_grid, log=True, levels=levels)

R_transform = Transform(qs_grid, eq.R_basis)
Z_transform = Transform(qs_grid, eq.Z_basis)
L_transform = Transform(qs_grid, eq.L_basis)
Rb_transform = Transform(qs_grid, eq.Rb_basis)
Zb_transform = Transform(qs_grid, eq.Zb_basis)
p_transform = Transform(qs_grid, eq.p_basis)
i_transform = Transform(qs_grid, eq.i_basis)
dRb = np.invert((eq.Rb_basis.modes == [0, 0, 0]).all(axis=1))

for i in range(iters):
    qs_fun = QuasisymmetryTripleProduct(
        R_transform,
        Z_transform,
        L_transform,
        Rb_transform,
        Zb_transform,
        p_transform,
        i_transform,
        eq.constraint,
    )
    eq = eq.perturb(
        objective=qs_fun,
        dRb=dRb,
        dZb=True,
        order=order,
        verbose=2,
        copy=True,
    )
    fam.insert(len(fam), eq)
    eq.solve(ftol=1e-2, xtol=1e-6, gtol=1e-6, maxiter=50, verbose=3)
#     plot_surfaces(eq, nzeta=4)
#     plot_2d(eq, "QS_TP", grid=plot_grid, log=True, levels=levels)

fam.save("examples/DESC/HELIOTRON_QS2_r90.h5")

"""
fam1 = EquilibriaFamily.load("examples/DESC/HELIOTRON_vacuum_QS1.h5")
fam2 = EquilibriaFamily.load("examples/DESC/HELIOTRON_vacuum_QS2.h5")
eq = fam1[-1]

ii = range(iters + 1)
err1 = np.zeros_like(ii)
err2 = np.zeros_like(ii)
qs_grid = LinearGrid(M=90, N=90, NFP=eq.NFP, rho=rho)

R_transform = Transform(qs_grid, eq.R_basis)
Z_transform = Transform(qs_grid, eq.Z_basis)
L_transform = Transform(qs_grid, eq.L_basis)
Rb_transform = Transform(qs_grid, eq.Rb_basis)
Zb_transform = Transform(qs_grid, eq.Zb_basis)
p_transform = Transform(qs_grid, eq.p_basis)
i_transform = Transform(qs_grid, eq.i_basis)
dRb = np.invert((eq.Rb_basis.modes == [0, 0, 0]).all(axis=1))

for i in ii:
    print("\nIteration = {}".format(i))
    print("-------------")
    eq1 = fam1[2 + i]
    eq2 = fam2[2 + i]
    fun1 = QuasisymmetryTripleProduct(
        R_transform,
        Z_transform,
        L_transform,
        Rb_transform,
        Zb_transform,
        p_transform,
        i_transform,
        eq1.constraint,
    )
    fun2 = QuasisymmetryTripleProduct(
        R_transform,
        Z_transform,
        L_transform,
        Rb_transform,
        Zb_transform,
        p_transform,
        i_transform,
        eq2.constraint,
    )
    args1 = (eq1.x, eq1.Rb_lmn, eq1.Zb_lmn, eq1.p_l, eq1.i_l, eq1.Psi)
    args2 = (eq2.x, eq2.Rb_lmn, eq2.Zb_lmn, eq2.p_l, eq2.i_l, eq2.Psi)
    err1[i] = np.mean(np.abs(fun1.compute(*args1)))
    err2[i] = np.mean(np.abs(fun2.compute(*args2)))
    print("1st order error: {}".format(err1[i]))
    print("2nd order error: {}".format(err2[i]))

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.semilogy(ii, err1/err1[0], "bo-", label="1st order")
ax.semilogy(ii, err2/err1[0], "ro-", label="2nd order")
ax.legend()
ax.set_xticks(ii)
ax.set_ylim(1e-3, 2e0)
ax.set_xlabel("Iteration")
ax.set_ylabel("$\\langle g(x,c) \\rangle / \\langle g(x_0,c_0) \\rangle$")
ax.set_title("Quasisymmetry error at $\\rho=0.9$")
fig.set_tight_layout(True)
"""
