"""Testing QS optimization."""

import numpy as np

from desc.utils import Timer
from desc.equilibrium import EquilibriaFamily
from desc.objective_funs import QuasisymmetryTripleProduct, QuasisymmetryFluxFunction
from desc.transform import Transform
from desc.grid import LinearGrid

fname = "QHS_TP_r70_aR0_m2_i2"

rho = 0.7  # surface to optimize
order = 2  # optimization order
iters = 2  # number of iterations
mn_lim = np.array([1, 2])  # boundary mode optimization limit

fam = EquilibriaFamily.load("examples/DESC/qs/QHS_init.h5")
eq = fam[-1]

err = 1e6
err0 = 0

timer = Timer()
timer.start("total")

# objective function transforms
grid_obj = LinearGrid(M=2 * eq.M_grid + 1, N=2 * eq.N_grid + 1, NFP=eq.NFP, rho=rho)
R_tform_obj = Transform(grid_obj, eq.R_basis)
Z_tform_obj = Transform(grid_obj, eq.Z_basis)
L_tform_obj = Transform(grid_obj, eq.L_basis)
Rb_tform_obj = Transform(grid_obj, eq.Rb_basis)
Zb_tform_obj = Transform(grid_obj, eq.Zb_basis)
p_tform_obj = Transform(grid_obj, eq.p_basis)
i_tform_obj = Transform(grid_obj, eq.i_basis)

# error evaluation transforms
grid_err = LinearGrid(M=180, N=180, NFP=eq.NFP, rho=rho)
R_tform_err = Transform(grid_err, eq.R_basis)
Z_tform_err = Transform(grid_err, eq.Z_basis)
L_tform_err = Transform(grid_err, eq.L_basis)
Rb_tform_err = Transform(grid_err, eq.Rb_basis)
Zb_tform_err = Transform(grid_err, eq.Zb_basis)
p_tform_err = Transform(grid_err, eq.p_basis)
i_tform_err = Transform(grid_err, eq.i_basis)

for k, mn in enumerate(mn_lim):
    print("Optimization step {}. Optimizing boundary modes m,n<={}".format(k, mn))
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
        print("Iteration {}".format(i))

        # QS objective functions
        fun_obj = QuasisymmetryTripleProduct(
            R_tform_obj,
            Z_tform_obj,
            L_tform_obj,
            Rb_tform_obj,
            Zb_tform_obj,
            p_tform_obj,
            i_tform_obj,
            eq.constraint,
        )
        fun_err = QuasisymmetryTripleProduct(
            R_tform_err,
            Z_tform_err,
            L_tform_err,
            Rb_tform_err,
            Zb_tform_err,
            p_tform_err,
            i_tform_err,
            eq.constraint,
        )

        tr_ratio = 0.1
        while err > err0:
            print("trust-region ratio = {}".format(tr_ratio))
            eq_p = eq.perturb(
                objective=fun_obj,
                dRb=dRb,
                dZb=dZb,
                order=order,
                tr_ratio=tr_ratio,
                verbose=2,
                copy=True,
            )
            args = (eq_p.x, eq_p.Rb_lmn, eq_p.Zb_lmn, eq_p.p_l, eq_p.i_l, eq_p.Psi)
            err = fun_err.compute(*args)
            tr_ratio /= 2

        err0 = err
        eq = eq_p
        fam.insert(len(fam), eq)
        eq.solve(ftol=1e-2, xtol=1e-6, gtol=1e-6, maxiter=50, verbose=3)
        timer.stop("opt step {}".format(k))
        timer.disp("opt step {}".format(k))

fam.save("examples/DESC/qs/" + fname + ".h5")
timer.stop("total")
timer.disp("total")
