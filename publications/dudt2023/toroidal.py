"""Omnigenity with toroidal contours."""

from desc import set_device

set_device("gpu")

import numpy as np
import os
from qsc import Qsc

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import (
    FixOmni,
    ForceBalance,
    ObjectiveFunction,
    Omnigenity,
    StraightBmaxContour,
)
from desc.objectives.utils import get_NAE_constraints
from desc.vmec import VMECIO


sym = True
NFP = 1
helicity = (1, 0)
L = 8
M = 8
N = 8
L_well = 4
M_well = 8
L_omni = 0
M_omni = 1
N_omni = 1
well_weight = 2
aspect_ratio = 20
target_mode = [0, 1, -1]
target_amplitude = np.pi / 6
surfaces = [0.2, 0.4, 0.6, 0.8, 1.0]

M_booz = int(np.ceil(1.5 * M))
N_booz = int(np.ceil(1.5 * N))
M_grid = int(np.ceil(1.5 * M_booz))
N_grid = int(np.ceil(1.5 * N_booz))

qsc = Qsc(
    nfp=NFP, rc=[1, 0.3], zs=[0, -0.3], B0=1.0, etabar=1.0, I2=1.0, p2=-4e6, order="r1"
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
idx = np.nonzero((eq.omni_basis.modes == target_mode).all(axis=1))[0]
omni_lmn = np.zeros(eq.omni_basis.num_modes)
omni_lmn[idx] = target_amplitude
eq._omni_lmn = omni_lmn
constraints = get_NAE_constraints(eq, qsc, order=1) + (
    StraightBmaxContour(),
    FixOmni(),
)
eq.solve(
    objective="force",
    constraints=constraints,
    ftol=1e-2,
    xtol=1e-6,
    gtol=1e-6,
    maxiter=100,
    verbose=3,
)
eq.save("toroidal.h5")

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
    (ForceBalance(bounds=(-1e-6, 1e-6), weight=2e2),) + tuple(objs.values())
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
eq.save("toroidal.h5")

VMECIO.save(eq, "wout_toroidal.nc", surfs=256)
