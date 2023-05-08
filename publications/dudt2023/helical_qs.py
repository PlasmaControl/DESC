"""Quasi-symmetry with helical contours."""

from desc import set_device

set_device("gpu")

import numpy as np
from qsc import Qsc

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import (
    CurrentDensity,
    FixOmni,
    ObjectiveFunction,
    Omnigenity,
    StraightBmaxContour,
)
from desc.objectives.utils import get_NAE_constraints
from desc.vmec import VMECIO


sym = True
NFP = 5
helicity = (1, NFP)
L = 8
M = 8
N = 8
L_well = 4
M_well = 8
L_omni = 0
M_omni = 0
N_omni = 0
well_weight = 2
aspect_ratio = 20
surfaces = [0.2, 0.4, 0.6, 0.8, 1.0]

M_booz = int(np.ceil(1.5 * M))
N_booz = int(np.ceil(1.5 * N))
M_grid = int(np.ceil(1.5 * M_booz))
N_grid = int(np.ceil(1.5 * N_booz))

qsc = Qsc(
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
    L=L,
    M=M,
    N=N,
    L_well=L_well,
    M_well=M_well,
    L_omni=L_omni,
    M_omni=M_omni,
    N_omni=N_omni,
)
constraints = get_NAE_constraints(eq, qsc, order=1) + (
    FixOmni(),
    StraightBmaxContour(),
)
eq.solve(
    objective="vacuum",
    constraints=constraints,
    ftol=1e-2,
    xtol=1e-6,
    gtol=1e-6,
    maxiter=100,
    verbose=3,
)
eq.save("helical_qs.h5")

grids = {}
objs = {}
for rho in surfaces:
    grids[rho] = LinearGrid(M=M_grid, N=N_grid, NFP=eq.NFP, sym=False, rho=rho)
    objs[rho] = Omnigenity(
        bounds=(-1e-3, 1e-3),
        grid=grids[rho],
        helicity=helicity,
        M_booz=M_booz,
        N_booz=N_booz,
        well_weight=well_weight,
    )

objective = ObjectiveFunction(
    (CurrentDensity(bounds=(-1e-6, 1e-6), weight=1e0),) + tuple(objs.values())
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
eq.save("helical_qs.h5")

VMECIO.save(eq, "wout_helical_qs.nc", surfs=256)
