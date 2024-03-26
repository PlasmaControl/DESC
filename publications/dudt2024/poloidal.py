"""Omnigenity with poloidal contours."""

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

fname = "poloidal"
sym = True
NFP = 2
helicity = (0, NFP)
LM = [8, 10, 12]
N = 12
L_well = 4
M_well = 8
L_omni = 4
M_omni = 4
N_omni = 4
well_weight = 2
eq_weights = [2.5e2, 5e2, 1e3]
aspect_ratio = 20
surfaces = [0.2, 0.4, 0.6, 0.8, 1.0]

assert len(LM) == len(eq_weights)


def eq_error(eq):
    grid = QuadratureGrid(L=32, M=32, N=32, NFP=NFP)
    data = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=grid)
    return data["<|F|>_vol"] / data["<|grad(|B|^2)|/2mu0>_vol"]


fam = EquilibriaFamily()

# initial NAE solution
qic = Qic(  # "QI NFP2 r2"
    nfp=NFP,
    rc=[
        1.0,
        0.0,
        -0.07764451554933544,
        0.0,
        0.005284971468552636,
        0.0,
        -0.00016252676632564814,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
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
        -0.06525233925323416,
        0.0,
        0.005858113288916291,
        0.0,
        -0.0001930489465183875,
        0.0,
        -1.21045713465733e-06,
        0.0,
        -6.6162738585035e-08,
        0.0,
        -1.8633251242689778e-07,
        0.0,
        1.4688345268925702e-07,
        0.0,
        -8.600467886165271e-08,
        0.0,
        4.172537468496238e-08,
        0.0,
        -1.099753830863863e-08,
    ],
    B0_vals=[1.0, 0.12735237900304514],
    k_buffer=1,
    p_buffer=2,
    k_second_order_SS=-25.137439389881692,
    d_over_curvature=-0.14601620836497467,
    d_svals=[
        0.0,
        -5.067489975338647,
        0.2759212337742016,
        -0.1407115065170644,
        0.00180521570352059,
        -0.03135134464554904,
        0.009582569807320895,
        -0.004780243312143034,
        0.002241790407060276,
        -0.0006738437017134619,
        0.00031559081192998053,
    ],
    delta=0.8,
    nphi=201,
    omn=True,
    omn_method="non-zone-fourier",
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
        maxiter=200,
        verbose=3,
        copy=True,
    )
    fam.append(eq)
    fam.save(fname + ".h5")
    print("equlibrium error: {:.2e}".format(eq_error(eq)))

# make iota positive
rone = np.ones_like(eq.R_lmn)
rone[eq.R_basis.modes[:, 2] < 0] *= -1
eq.R_lmn *= rone
zone = np.ones_like(eq.Z_lmn)
zone[eq.Z_basis.modes[:, 2] < 0] *= -1
eq.Z_lmn *= zone
lone = np.ones_like(eq.L_lmn)
lone[eq.L_basis.modes[:, 2] < 0] *= -1
eq.L_lmn *= lone
eq.surface = eq.get_surface_at(rho=1)

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
