"""Quasi-symmetry with poloidal contours."""

from desc import set_device

set_device("gpu")

import numpy as np
from qic import Qic

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import (
    CurrentDensity,
    ObjectiveFunction,
    Omnigenity,
    StraightBmaxContour,
)
from desc.objectives.utils import get_fixed_boundary_constraints, get_NAE_constraints

fname = "poloidal_qs"
sym = True
NFP = 1
helicity = (0, NFP)
LM = [8, 10, 12]
N = 12
L_well = 4
M_well = 8
L_omni = 4
M_omni = 4
N_omni = 0
well_weight = 2
eq_weights = [1e2, 2e2, 4e2]
aspect_ratio = 20
surfaces = [0.2, 0.4, 0.6, 0.8, 1.0]

assert len(LM) == len(eq_weights)


def eq_error(eq):
    grid = QuadratureGrid(L=32, M=32, N=32, NFP=NFP)
    data = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=grid)
    return data["<|F|>_vol"] / data["<|grad(|B|^2)|/2mu0>_vol"]


fam = EquilibriaFamily()

# initial NAE solution
qic = Qic(  # "QI NFP1 r1 Jorge"
    nfp=NFP,
    rc=[
        1.0,
        0.0,
        -0.4056622889934463,
        0.0,
        0.07747378220100756,
        0.0,
        -0.007803860877024245,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    zs=[
        0.0,
        0.0,
        0.24769666390049602,
        0.0,
        -0.06767352436978152,
        0.0,
        0.006980621303449165,
        0.0,
        0.0006816270917189934,
        0.0,
        1.4512784317099981e-05,
        0.0,
        2.839050532138523e-06,
    ],
    B0_vals=[1.0, 0.16915531046156507],
    k_buffer=3,
    k_second_order_SS=0.0,
    d_over_curvature=0.5183783762725197,
    d_svals=[
        0.0,
        -0.003563114185517955,
        -0.0002015921485566435,
        0.0012178616509882368,
        0.00011629450296628697,
        8.255825435616736e-07,
        -3.2011540526397e-06,
    ],
    delta=0.1,
    nphi=201,
    omn=True,
    omn_method="non-zone",
)
eq = Equilibrium.from_near_axis(
    qic,
    r=1 / aspect_ratio,
    L=LM[0],
    M=LM[0],
    N=N,
    L_well=L_well,
    M_well=M_well,
    L_omni=L_omni,
    M_omni=M_omni,
    N_omni=N_omni,
)
fam.append(eq)
fam.save(fname + ".h5")
print("equlibrium error: {:.2e}".format(eq_error(eq)))

# re-solve with NAE constraints
constraints = get_NAE_constraints(eq, qic, order=1)
eq, result = eq.solve(
    objective="vacuum",
    constraints=constraints,
    ftol=1e-2,
    xtol=1e-6,
    gtol=1e-6,
    maxiter=200,
    verbose=3,
    copy=True,
)
fam.append(eq)
fam.save(fname + ".h5")
print("equlibrium error: {:.2e}".format(eq_error(eq)))

# optimize with increasing resolution
for i in range(len(LM)):
    eq.change_resolution(L=LM[i], M=LM[i], L_grid=2 * LM[i], M_grid=2 * LM[i], sym=sym)
    M_booz = min(int(np.ceil(1.5 * LM[i])), 16)
    N_booz = min(int(np.ceil(1.5 * N)), 16)
    M_grid = int(np.ceil(1.5 * M_booz))
    N_grid = int(np.ceil(1.5 * N_booz))

    grids = {}
    objs = {}
    for rho in surfaces:
        grids[rho] = LinearGrid(M=M_grid, N=N_grid, NFP=eq.NFP, sym=False, rho=rho)
        objs[rho] = Omnigenity(
            grid=grids[rho],
            helicity=helicity,
            M_booz=M_booz,
            N_booz=N_booz,
            well_weight=well_weight,
        )

    objective = ObjectiveFunction(
        (CurrentDensity(weight=eq_weights[i]),) + tuple(objs.values())
    )
    constraints = get_NAE_constraints(eq, qic, order=1) + (StraightBmaxContour(),)
    eq, result = eq.solve(
        objective=objective,
        constraints=constraints,
        optimizer="lsq-exact",
        ftol=1e-3,
        xtol=1e-6,
        gtol=1e-6,
        maxiter=50,
        verbose=3,
        copy=True,
    )
    fam.append(eq)
    fam.save(fname + ".h5")
    print("equlibrium error: {:.2e}".format(eq_error(eq)))

# re-solve with fixed boundary constraints
constraints = get_fixed_boundary_constraints(iota=False)
eq, result = eq.solve(
    objective="vacuum",
    constraints=constraints,
    ftol=1e-2,
    xtol=1e-6,
    gtol=1e-6,
    maxiter=200,
    verbose=3,
    copy=True,
)
fam.append(eq)
fam.save(fname + ".h5")
print("equlibrium error: {:.2e}".format(eq_error(eq)))
