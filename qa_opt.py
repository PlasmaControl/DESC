import numpy as np
import matplotlib.pyplot as plt

from desc import set_device

set_device("gpu")

from desc.utils import Timer
from desc.grid import ConcentricGrid
from desc.equilibrium import EquilibriaFamily
from desc.optimize import Optimizer
from desc.objectives import (
    ObjectiveFunction,
    FixedBoundaryR,
    FixedBoundaryZ,
    FixedPressure,
    FixedIota,
    FixedPsi,
    LCFSBoundary,
    RadialForceBalance,
    HelicalForceBalance,
    QuasisymmetryFluxFunction,
    QuasisymmetryTripleProduct,
)
from desc.plotting import (
    plot_surfaces,
    plot_section,
    plot_current,
    plot_qs_error,
)

n = 3
path = "/projects/EKOLEMEN/QS/QA/SIMSOPT_QA_fC1_n{}/".format(n)

# optimization parameters
ftol = 1e-2  # tolerance on objective function value
xtol = 1e-6  # tolerance on state vector
gtol = 1e-6  # tolerance on gradient vector
maxiter = 50  # maximum number of Gauss-Newton steps
minbres = n  # minimum boundary mode for optimization
maxbres = 4  # maximum boundary mode for optimization
tr_ratio = 10 ** -n  # trust region ratio

fam = EquilibriaFamily.load(
    "/home/ddudt/DESC/examples/DESC/SIMSOPT_QA_n{}.h5".format(n - 1)
)
eq = fam[-1]

# save figures of initial equilibrium
ax = plot_surfaces(eq)
plt.show()
plt.savefig(path + "surfaces_inital.png")
ax = plot_section(eq, "|F|", log=True, norm_F=True)
plt.show()
plt.savefig(path + "equilibrium_initial.png")
ax = plot_qs_error(eq, L=20, fB=False)
plt.show()
plt.savefig(path + "qs_initial.png")
ax = plot_current(eq, L=20, log=True)
plt.show()
plt.savefig(path + "current_initial.png")

timer = Timer()
timer.start("Total")

grid = ConcentricGrid(
    L=eq.L_grid,
    M=eq.M_grid,
    N=eq.N_grid,
    NFP=eq.NFP,
    sym=eq.sym,
    axis=False,
    rotation="cos",
    node_pattern=eq.node_pattern,
)

optimizer = Optimizer("lsq-exact")

# equilibrium force balance objective function
objectives_eq = (
    RadialForceBalance(),  # minimize JxB - grad(p)
    HelicalForceBalance(),  # minimize J^rho
)
constraints_eq = (
    FixedBoundaryR(),  # R boundary modes are fixed
    FixedBoundaryZ(),  # Z boundary modes are fixed
    FixedPressure(),  # pressure profile is fixed
    FixedIota(),  # rotational transform profile is fixed
    FixedPsi(),  # total toroidal magnetic flux is fixed
    LCFSBoundary(),  # fixed-boundary constraint
)
fun_eq = ObjectiveFunction(objectives_eq, constraints_eq, eq)

# quasi-symmetry objective function
# objectives_qs = QuasisymmetryTripleProduct(grid=grid)
objectives_qs = QuasisymmetryFluxFunction()
constraints_qs = (  # constraints are same as fun_eq, but new instance is required
    FixedBoundaryR(),
    FixedBoundaryZ(),
    FixedPressure(),
    FixedIota(),
    FixedPsi(),
    LCFSBoundary(),
)
fun_qs = ObjectiveFunction(objectives_qs, constraints_qs, eq)

# indicies of variables to include in optimization
Rb_modes = eq.surface.R_basis.modes
Zb_modes = eq.surface.Z_basis.modes
dRb = np.zeros((Rb_modes.shape[0],), dtype=bool)
dZb = np.zeros((Zb_modes.shape[0],), dtype=bool)
idxRb = np.where(
    np.logical_and(
        (np.abs(Rb_modes) >= [0, minbres, minbres]).all(axis=1),
        (np.abs(Rb_modes) <= [0, maxbres, maxbres]).all(axis=1),
    )
)[0]
idxZb = np.where(
    np.logical_and(
        (np.abs(Zb_modes) >= [0, minbres, minbres]).all(axis=1),
        (np.abs(Zb_modes) <= [0, maxbres, maxbres]).all(axis=1),
    )
)[0]
dRb[idxRb] = True  # free variables: R boundary modes with (m <= M, n <= N)
dZb[idxZb] = True  # free variables: Z boundary modes with (m <= M, n <= N)

f0 = fun_qs.compute_scalar(fun_qs.y(eq))  # initial cost
for i in range(1, maxiter):
    fun_qs.callback(fun_qs.y(eq))

    # perturb stellarator boundary to optimize objective
    eq, red_ratio = eq.perturb(
        fun_eq,  # constraint function (equilibrium)
        fun_qs,  # objective function (quasi-symmetry)
        dRb=dRb,
        dZb=dZb,
        order=2,
        tr_ratio=tr_ratio,
        verbose=2,
        copy=True,
    )
    fun_qs.callback(fun_qs.y(eq))

    # re-solve equilibrium to satisfy force balance
    eq.solve(
        optimizer,
        fun_eq,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        maxiter=maxiter,
        verbose=3,
    )

    # stop optimization if cost reduction is small
    f = fun_qs.compute_scalar(fun_qs.y(eq))
    print("Total (sum of squares): {:10.3e}, ".format(f))
    if f > f0:
        print("Objective function increased!")
        break
    if (f0 - f) / f0 < ftol:
        print("Optimization tolerance satisfied.")
        break
    f0 = f

eq.save(path + "DESC_QA.h5")

# save figures of optimized equilibrium
ax = plot_surfaces(eq)
plt.show()
plt.savefig(path + "surfaces_final.png")
ax = plot_section(eq, "|F|", log=True, norm_F=True)
plt.show()
plt.savefig(path + "equilibrium_final.png")
ax = plot_qs_error(eq, L=20, fB=False)
plt.show()
plt.savefig(path + "qs_final.png")
ax = plot_current(eq, L=20, log=True)
plt.show()
plt.savefig(path + "current_final.png")

timer.stop("Total")
timer.disp("Total")
