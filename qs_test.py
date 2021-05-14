"""Testing QS perturbations."""

import numpy as np
import matplotlib.pyplot as plt
from desc.equilibrium import EquilibriaFamily
from desc.objective_funs import QuasisymmetryTripleProduct, QuasisymmetryFluxFunction
from desc.transform import Transform
from desc.grid import LinearGrid
from desc.plotting import plot_surfaces, plot_2d, plot_section


# validating functions

fam = EquilibriaFamily.load("examples/DESC/outputs/HSXT_ansi.h5")
eq = fam[-1]
args = (eq.x, eq.Rb_lmn, eq.Zb_lmn, eq.p_l, eq.i_l, eq.Psi)

res = 30
rho = np.logspace(-1, 0, 11)
TP_err = np.zeros_like(rho)
FF_err = np.zeros_like(rho)

for i in range(rho.size):
    print("rho = {}".format(rho[i]))
    grid = LinearGrid(M=res, N=res, NFP=eq.NFP, rho=rho[i])
    R_transform = Transform(grid, eq.R_basis)
    Z_transform = Transform(grid, eq.Z_basis)
    L_transform = Transform(grid, eq.L_basis)
    Rb_transform = Transform(grid, eq.Rb_basis)
    Zb_transform = Transform(grid, eq.Zb_basis)
    p_transform = Transform(grid, eq.p_basis)
    i_transform = Transform(grid, eq.i_basis)
    TP_fun = QuasisymmetryTripleProduct(
        R_transform,
        Z_transform,
        L_transform,
        Rb_transform,
        Zb_transform,
        p_transform,
        i_transform,
        eq.constraint,
    )
    FF_fun = QuasisymmetryFluxFunction(
        R_transform,
        Z_transform,
        L_transform,
        Rb_transform,
        Zb_transform,
        p_transform,
        i_transform,
        eq.constraint,
    )
    TP_err[i] = TP_fun.compute(*args)
    FF_err[i] = FF_fun.compute(*args)

plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.loglog(rho, TP_err, "bo-", label="$f_T$")
ax.loglog(rho, FF_err, "ro-", label="$f_P$")
ax.legend()
ax.set_xlabel("$\\rho = \\sqrt{\\psi_N}$")
ax.set_title("HSX")
fig.set_tight_layout(True)

"""
rho = 0.9  # surface to optimize
order = 2  # optimization order
iters = 5  # optimization iterations

# optimization

fam = EquilibriaFamily.load("examples/DESC/HELIOTRON_output.h5")
eq = fam[-1]
qs_grid = LinearGrid(M=2 * eq.M_grid + 1, N=2 * eq.N_grid + 1, NFP=eq.NFP, rho=rho)

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
        objective=qs_fun, dRb=dRb, dZb=True, order=order, verbose=2, copy=True,
    )
    fam.insert(len(fam), eq)
    eq.solve(ftol=1e-2, xtol=1e-6, gtol=1e-6, maxiter=50, verbose=3)

fam.save("examples/DESC/HELIOTRON_QS2_r90.h5")

# plotting

fam1 = EquilibriaFamily.load("examples/DESC/HELIOTRON_QS1_nosolve.h5")
fam2 = EquilibriaFamily.load("examples/DESC/HELIOTRON_QS2_nosolve.h5")
eq = fam1[-1]

ii = range(iters + 1)
err1 = np.zeros_like(ii, dtype="float")
err2 = np.zeros_like(ii, dtype="float")
qs_grid = LinearGrid(M=180, N=180, NFP=eq.NFP, rho=rho)

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

levels = np.logspace(-4, -1, num=7)
plot_surfaces(fam2[2], nzeta=4)
plot_surfaces(fam2[-1], nzeta=4)
plot_2d(fam2[2], "QS_TP", grid=qs_grid, log=True, levels=levels)
plot_2d(fam2[-1], "QS_TP", grid=qs_grid, log=True, levels=levels)

plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.semilogy(ii, err1, "bo-", label="1st order")
ax.semilogy(ii, err2, "ro-", label="2nd order")
ax.legend()
ax.set_xticks(ii)
ax.set_xlabel("Iteration")
ax.set_ylabel("$\\langle g(x,c) \\rangle$")
ax.set_title("Quasi-symmetry error at $\\rho={}$".format(rho))
fig.set_tight_layout(True)
"""
