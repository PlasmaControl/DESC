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
    FixLambdaGauge,
    ForceBalanceGalerkin
)
from desc.optimize import Optimizer
from desc.plotting import plot_grid, plot_boozer_modes, plot_boozer_surface, plot_qs_error

#%%
eq_init = desc.io.load("/scratch/gpfs/pk2354/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
#eq_init = desc.io.load("/home/pk123/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
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

eq_qs_T.save('/scratch/gpfs/pk2354/DESC/test_equilibria/constrained_qs_fb.h5')

