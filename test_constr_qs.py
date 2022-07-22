#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:39:53 2022

@author: pk123
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12

import desc.io
from desc.grid import LinearGrid, ConcentricGrid
from desc.objectives import (
    ObjectiveFunction,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixPsi,
    AspectRatio,
    ForceBalance,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    QuasisymmetryTripleProduct,
    FixLambdaGauge
)
from desc.optimize import Optimizer
from desc.plotting import plot_grid, plot_boozer_modes, plot_boozer_surface, plot_qs_error

#%%
eq_init = desc.io.load("/home/pk123/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
optimizer = Optimizer("lsq-exact")
idx_Rcc = eq_init.surface.R_basis.get_idx(M=1, N=2)
idx_Rss = eq_init.surface.R_basis.get_idx(M=-1, N=-2)
idx_Zsc = eq_init.surface.Z_basis.get_idx(M=-1, N=2)
idx_Zcs = eq_init.surface.Z_basis.get_idx(M=1, N=-2)

# boundary modes to constrain
R_modes = np.delete(eq_init.surface.R_basis.modes, [idx_Rcc, idx_Rss], axis=0)
Z_modes = np.delete(eq_init.surface.Z_basis.modes, [idx_Zsc, idx_Zcs], axis=0)

# constraints
constraints = (
    ForceBalance(),  # enforce JxB-grad(p)=0 during optimization
    FixBoundaryR(modes=R_modes),  # fix specified R boundary modes
    FixBoundaryZ(modes=Z_modes),  # fix specified Z boundary modes
    FixPressure(),  # fix pressure profile
    FixIota(),  # fix rotational transform profile
    FixPsi(),  # fix total toroidal magnetic flux
)

grid_vol = ConcentricGrid(L=eq_init.L_grid, M=eq_init.M_grid, N=eq_init.N_grid, NFP=eq_init.NFP, sym=eq_init.sym)
#plot_grid(grid_vol);
objective_fT = ObjectiveFunction(QuasisymmetryTripleProduct(grid=grid_vol), verbose=0)

eq_qs_T_unc, result_T_unc = eq_init.optimize(
    objective=objective_fT,
    constraints=constraints,
    optimizer=optimizer,
    ftol=1e-2,  # stopping tolerance on the function value
    xtol=1e-6,  # stopping tolerance on the step size
    gtol=1e-6,  # stopping tolerance on the gradient
    maxiter=50,  # maximum number of iterations
    options={
        "perturb_options": {"order": 2, "verbose": 0},  # use 2nd-order perturbations
        "solve_options": {"ftol": 1e-2, "xtol": 1e-6, "gtol": 1e-6, "verbose": 0}, # for equilibrium subproblem
    },
    copy=True,  # return a new Equilibrium object (copy=False will overwrite the original)
    verbose=3,
)
#%%
eq_init = desc.io.load("/home/pk123/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
optimizer = Optimizer("lsq-auglag")
idx_Rcc = eq_init.surface.R_basis.get_idx(M=1, N=2)
idx_Rss = eq_init.surface.R_basis.get_idx(M=-1, N=-2)
idx_Zsc = eq_init.surface.Z_basis.get_idx(M=-1, N=2)
idx_Zcs = eq_init.surface.Z_basis.get_idx(M=1, N=-2)

# boundary modes to constrain
R_modes = np.delete(eq_init.surface.R_basis.modes, [idx_Rcc, idx_Rss], axis=0)
Z_modes = np.delete(eq_init.surface.Z_basis.modes, [idx_Zsc, idx_Zcs], axis=0)

constraints = (
    ForceBalance(),  # enforce JxB-grad(p)=0 during optimization
    FixBoundaryR(modes=R_modes,fixed_boundary=True),  # fix specified R boundary modes
    FixBoundaryZ(modes=Z_modes,fixed_boundary=True),  # fix specified Z boundary modes
    FixPressure(),  # fix pressure profile
    FixIota(),  # fix rotational transform profile
    FixPsi(),  # fix total toroidal magnetic flux
    FixLambdaGauge()
)

grid_vol = ConcentricGrid(L=eq_init.L_grid, M=eq_init.M_grid, N=eq_init.N_grid, NFP=eq_init.NFP, sym=eq_init.sym)
#plot_grid(grid_vol);
objective_fT = ObjectiveFunction(QuasisymmetryTripleProduct(grid=grid_vol), verbose=0)

eq_qs_T, result_T = eq_init.optimize(
    objective=objective_fT,
    constraints=constraints,
    optimizer=optimizer,
    ftol=1e-2,  # stopping tolerance on the function value
    xtol=1e-6,  # stopping tolerance on the step size
    gtol=1e-6,  # stopping tolerance on the gradient
    maxiter=50,  # maximum number of iterations
    options={
        "perturb_options": {"order": 2, "verbose": 0},  # use 2nd-order perturbations
        "solve_options": {"ftol": 1e-2, "xtol": 1e-6, "gtol": 1e-6, "verbose": 0}, # for equilibrium subproblem
    },
    copy=True,  # return a new Equilibrium object (copy=False will overwrite the original)
    verbose=3,
)
#%% QS Optimization with Aspect Ratio Constraint
eq_init = desc.io.load("/home/pk123/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
optimizer = Optimizer("lsq-auglag")
idx_Rcc = eq_init.surface.R_basis.get_idx(M=1, N=2)
idx_Rss = eq_init.surface.R_basis.get_idx(M=-1, N=-2)
idx_Zsc = eq_init.surface.Z_basis.get_idx(M=-1, N=2)
idx_Zcs = eq_init.surface.Z_basis.get_idx(M=1, N=-2)

# boundary modes to constrain
R_modes = np.delete(eq_init.surface.R_basis.modes, [idx_Rcc, idx_Rss], axis=0)
Z_modes = np.delete(eq_init.surface.Z_basis.modes, [idx_Zsc, idx_Zcs], axis=0)

constraints = (
    ForceBalance(),  # enforce JxB-grad(p)=0 during optimization
    FixBoundaryR(modes=R_modes,fixed_boundary=True),  # fix specified R boundary modes
    FixBoundaryZ(modes=Z_modes,fixed_boundary=True),  # fix specified Z boundary modes
    FixPressure(),  # fix pressure profile
    FixIota(),  # fix rotational transform profile
    FixPsi(),  # fix total toroidal magnetic flux
    FixLambdaGauge(),
    AspectRatio(target=5,equality=False)
)

grid_vol = ConcentricGrid(L=eq_init.L_grid, M=eq_init.M_grid, N=eq_init.N_grid, NFP=eq_init.NFP, sym=eq_init.sym)
#plot_grid(grid_vol);
objective_fT = ObjectiveFunction(QuasisymmetryTripleProduct(grid=grid_vol), verbose=0)

eq_qs_T, result_T = eq_init.optimize(
    objective=objective_fT,
    constraints=constraints,
    optimizer=optimizer,
    ftol=1e-2,  # stopping tolerance on the function value
    xtol=1e-6,  # stopping tolerance on the step size
    gtol=1e-6,  # stopping tolerance on the gradient
    maxiter=50,  # maximum number of iterations
    options={
        "perturb_options": {"order": 2, "verbose": 0},  # use 2nd-order perturbations
        "solve_options": {"ftol": 1e-2, "xtol": 1e-6, "gtol": 1e-6, "verbose": 0}, # for equilibrium subproblem
    },
    copy=True,  # return a new Equilibrium object (copy=False will overwrite the original)
    verbose=3,
)
#%% QS optimization solovev unconstrained
path = '/home/pk123/DESC/examples/DESC/SOLOVEV_output.h5'
eq_init = desc.io.load(path)[-1]

optimizer = Optimizer("lsq-exact")
# idx_Rcc = eq_init.surface.R_basis.get_idx(M=1, N=2)
# idx_Rss = eq_init.surface.R_basis.get_idx(M=-1, N=-2)
# idx_Zsc = eq_init.surface.Z_basis.get_idx(M=-1, N=2)
# idx_Zcs = eq_init.surface.Z_basis.get_idx(M=1, N=-2)

# boundary modes to constrain
# R_modes = np.delete(eq_init.surface.R_basis.modes, [idx_Rcc, idx_Rss], axis=0)
# Z_modes = np.delete(eq_init.surface.Z_basis.modes, [idx_Zsc, idx_Zcs], axis=0)

grid_vol = ConcentricGrid(L=eq_init.L_grid, M=eq_init.M_grid, N=eq_init.N_grid, NFP=eq_init.NFP, sym=eq_init.sym)
#plot_grid(grid_vol);
objective_fT = ObjectiveFunction(QuasisymmetryTripleProduct(grid=grid_vol), verbose=0)

#objective = ObjectiveFunction(AspectRatio(target=2.5))
constraints = (
    ForceBalance(),
    FixBoundaryR(),
    FixBoundaryZ(modes=eq_init.surface.Z_basis.modes[0:-1, :]),
    FixPressure(),
    FixIota(),
    FixPsi(),
)
options = {"perturb_options": {"order": 2}}
result_unc = eq_init.optimize(objective_fT, constraints, options=options)

#np.testing.assert_allclose(eq.compute("V")["R0/a"], 2.5)
#%%QS optimization unconstrained, 1 mode
eq_init = desc.io.load("/home/pk123/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
optimizer = Optimizer("lsq-exact")
# idx_Rcc = eq_init.surface.R_basis.get_idx(M=1, N=2)
# idx_Rss = eq_init.surface.R_basis.get_idx(M=-1, N=-2)
# idx_Zsc = eq_init.surface.Z_basis.get_idx(M=-1, N=2)
# idx_Zcs = eq_init.surface.Z_basis.get_idx(M=1, N=-2)

# # boundary modes to constrain
# R_modes = np.delete(eq_init.surface.R_basis.modes, [idx_Rcc, idx_Rss], axis=0)
# Z_modes = np.delete(eq_init.surface.Z_basis.modes, [idx_Zsc, idx_Zcs], axis=0)

# constraints
constraints = (
    ForceBalance(),  # enforce JxB-grad(p)=0 during optimization
    FixBoundaryR(),  # fix specified R boundary modes
    FixBoundaryZ(modes=eq_init.surface.Z_basis.modes[0:-1, :]),  # fix specified Z boundary modes
    FixPressure(),  # fix pressure profile
    FixIota(),  # fix rotational transform profile
    FixPsi(),  # fix total toroidal magnetic flux
)

grid_vol = ConcentricGrid(L=eq_init.L_grid, M=eq_init.M_grid, N=eq_init.N_grid, NFP=eq_init.NFP, sym=eq_init.sym)
#plot_grid(grid_vol);
objective_fT = ObjectiveFunction(QuasisymmetryTripleProduct(grid=grid_vol), verbose=0)

eq_qs_T, result_T = eq_init.optimize(
    objective=objective_fT,
    constraints=constraints,
    optimizer=optimizer,
    ftol=1e-2,  # stopping tolerance on the function value
    xtol=1e-6,  # stopping tolerance on the step size
    gtol=1e-6,  # stopping tolerance on the gradient
    maxiter=50,  # maximum number of iterations
    options={
        "perturb_options": {"order": 2, "verbose": 0},  # use 2nd-order perturbations
        "solve_options": {"ftol": 1e-2, "xtol": 1e-6, "gtol": 1e-6, "verbose": 0}, # for equilibrium subproblem
    },
    copy=True,  # return a new Equilibrium object (copy=False will overwrite the original)
    verbose=3,
)

#%% Aspect ratio constrained qs equil
eq_init = desc.io.load("/home/pk123/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
optimizer = Optimizer("lsq-auglag")
# idx_Rcc = eq_init.surface.R_basis.get_idx(M=1, N=2)
# idx_Rss = eq_init.surface.R_basis.get_idx(M=-1, N=-2)
# idx_Zsc = eq_init.surface.Z_basis.get_idx(M=-1, N=2)
# idx_Zcs = eq_init.surface.Z_basis.get_idx(M=1, N=-2)

# # boundary modes to constrain
# R_modes = np.delete(eq_init.surface.R_basis.modes, [idx_Rcc, idx_Rss], axis=0)
# Z_modes = np.delete(eq_init.surface.Z_basis.modes, [idx_Zsc, idx_Zcs], axis=0)

constraints = (
    ForceBalance(),  # enforce JxB-grad(p)=0 during optimization
    FixBoundaryR(fixed_boundary=True),  # fix specified R boundary modes
    FixBoundaryZ(modes=eq_init.surface.Z_basis.modes[0:-1, :],fixed_boundary=True),  # fix specified Z boundary modes
    FixPressure(),  # fix pressure profile
    FixIota(),  # fix rotational transform profile
    FixPsi(),  # fix total toroidal magnetic flux
)

# grid_vol = ConcentricGrid(L=eq_init.L_grid, M=eq_init.M_grid, N=eq_init.N_grid, NFP=eq_init.NFP, sym=eq_init.sym)
# #plot_grid(grid_vol);
# objective_fT = ObjectiveFunction(QuasisymmetryTripleProduct(grid=grid_vol), verbose=0)
objective = ObjectiveFunction(AspectRatio(target=5.0))

eq_qs_T, result_T = eq_init.optimize(
    objective=objective,
    constraints=constraints,
    optimizer=optimizer,
    ftol=1e-2,  # stopping tolerance on the function value
    xtol=1e-6,  # stopping tolerance on the step size
    gtol=1e-6,  # stopping tolerance on the gradient
    maxiter=50,  # maximum number of iterations
    # options={
    #     "perturb_options": {"order": 2, "verbose": 0},  # use 2nd-order perturbations
    #     "solve_options": {"ftol": 1e-2, "xtol": 1e-6, "gtol": 1e-6, "verbose": 0}, # for equilibrium subproblem
    # },
    # copy=True,  # return a new Equilibrium object (copy=False will overwrite the original)
    # verbose=3,
)