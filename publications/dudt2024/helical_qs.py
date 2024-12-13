"""Quasi-symmetry with helical contours."""

from desc import set_device

set_device("gpu")

import numpy as np
from qsc import Qsc

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import LinearGrid, QuadratureGrid
from desc.objectives import (
    CurrentDensity,
    FixOmni,
    ObjectiveFunction,
    Omnigenity,
    StraightBmaxContour,
)
from desc.objectives.utils import get_fixed_boundary_constraints, get_NAE_constraints

fname = "helical_qs"
sym = True
NFP = 5
helicity = (1, NFP)
LM = [8, 10, 12]
N = 12
L_well = 4
M_well = 8
L_omni = 0
M_omni = 0
N_omni = 0
well_weight = 2
eq_weights = [1e0, 2e0, 4e0]
aspect_ratio = 20
surfaces = [0.2, 0.4, 0.6, 0.8, 1.0]

assert len(LM) == len(eq_weights)


def eq_error(eq):
    grid = QuadratureGrid(L=32, M=32, N=32, NFP=NFP)
    data = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=grid)
    return data["<|F|>_vol"] / data["<|grad(|B|^2)|/2mu0>_vol"]


fam = EquilibriaFamily()

# initial NAE solution
qsc = Qsc(  # custom?
    nfp=NFP,
    rc=[
        1.00000000e00,
        1.36094189e-01,
        1.15569807e-02,
        5.77324841e-04,
        -2.13812436e-05,
        -6.86819127e-06,
        -3.25014052e-07,
        7.33393963e-08,
        1.42375011e-08,
        7.98521016e-10,
    ],
    zs=[
        0.00000000e00,
        -1.25243961e-01,
        -1.10551096e-02,
        -5.87380185e-04,
        1.59769355e-05,
        6.41471931e-06,
        3.47327323e-07,
        -6.49967945e-08,
        -1.39333315e-08,
        -8.47322874e-10,
    ],
    B0=1.0,
    B2c=-0.38028563,
    etabar=2.179209340685954,
    I2=0.0,
    p2=0.0,
    order="r1",
)
eq = Equilibrium.from_near_axis(
    qsc,
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
constraints = get_NAE_constraints(eq, qsc, order=1)
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
    constraints = get_NAE_constraints(eq, qsc, order=1) + (
        FixOmni(),
        StraightBmaxContour(),
    )
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
