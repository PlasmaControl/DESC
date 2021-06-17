"""Testing QS optimization."""

import numpy as np

from desc.utils import Timer
from desc.equilibrium import EquilibriaFamily
from desc.objective_funs import QuasisymmetryTripleProduct, QuasisymmetryFluxFunction
from desc.transform import Transform
from desc.grid import LinearGrid

fname = "QHS_FF_h11_r70_o1_aR0_m4_i10"

rho = 0.7  # surface to optimize
order = 1  # optimization order
iters = 10  # number of iterations
mn_lim = np.array([1, 2, 3, 4])  # boundary mode optimization limit

fam = EquilibriaFamily.load("examples/DESC/qs/QHS_M8N8.h5")
eq = fam[-1]

timer = Timer()
timer.start("total")

grid = LinearGrid(M=2 * eq.M_grid + 1, N=2 * eq.N_grid + 1, NFP=eq.NFP, rho=rho)
R_transform = Transform(grid, eq.R_basis)
Z_transform = Transform(grid, eq.Z_basis)
L_transform = Transform(grid, eq.L_basis)
Rb_transform = Transform(grid, eq.Rb_basis)
Zb_transform = Transform(grid, eq.Zb_basis)
p_transform = Transform(grid, eq.p_basis)
i_transform = Transform(grid, eq.i_basis)

for k, mn in enumerate(mn_lim):
    print("\nOptimization step {}. Optimizing boundary modes m,n <= {}".format(k, mn))
    timer.start("opt step {}".format(k))

    # optimization variables: boundary modes <= MN
    dRb = np.logical_and(
        np.abs(eq.Rb_basis.modes[:, 1]) <= mn, np.abs(eq.Rb_basis.modes[:, 2]) <= mn
    )
    dZb = np.logical_and(
        np.abs(eq.Zb_basis.modes[:, 1]) <= mn, np.abs(eq.Zb_basis.modes[:, 2]) <= mn
    )
    # fix major radius
    dRb[np.where((eq.Rb_basis.modes == [0, 0, 0]).all(axis=1))[0]] = False

    for i in range(iters):
        print("\nIteration {}".format(i))
        timer.start("iteration {}".format(i))

        # QS objective functions
        fun = QuasisymmetryFluxFunction(
            R_transform,
            Z_transform,
            L_transform,
            Rb_transform,
            Zb_transform,
            p_transform,
            i_transform,
            eq.constraint,
        )
        args = (eq.x, eq.Rb_lmn, eq.Zb_lmn, eq.p_l, eq.i_l, eq.Psi)
        err0 = fun.compute_scalar(*args)
        err = err0 + 1
        print("error = {}".format(err))

        tr_ratio = 0.1
        while err > err0:
            print("\ntrust-region ratio = {}".format(tr_ratio))
            eq_p = eq.perturb(
                objective=fun,
                dRb=dRb,
                dZb=dZb,
                order=order,
                tr_ratio=tr_ratio,
                verbose=2,
                copy=True,
            )
            args = (eq_p.x, eq_p.Rb_lmn, eq_p.Zb_lmn, eq_p.p_l, eq_p.i_l, eq_p.Psi)
            err = fun.compute_scalar(*args)
            tr_ratio /= 2

        print("error = {}".format(err))
        eq = eq_p
        fam.insert(len(fam), eq)
        eq.solve(ftol=1e-2, xtol=1e-6, gtol=1e-6, maxiter=50, verbose=3)
        fam.save("examples/DESC/qs/" + fname + ".h5")
        timer.stop("iteration {}".format(i))
        timer.disp("iteration {}".format(i))

    timer.stop("opt step {}".format(k))
    timer.disp("opt step {}".format(k))

timer.stop("total")
timer.disp("total")
