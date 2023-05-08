"""Quasi-symmetry with poloidal contours."""

from desc import set_device

set_device("gpu")

import numpy as np

from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.objectives import (
    CurrentDensity,
    FixAxisR,
    FixAxisZ,
    FixCurrent,
    FixLambdaGauge,
    FixPsi,
    FixWell,
    ObjectiveFunction,
    Omnigenity,
    StraightBmaxContour,
)
from desc.vmec import VMECIO


sym = True
NFP = 1
helicity = (0, NFP)
L = 8
M = 8
N = 8
L_well = 1
M_well = 2
L_omni = 4
M_omni = 4
N_omni = 0
well_weight = 2
aspect_ratio = 20
mirror_ratio = 0.2
elongation = 1.0
torsion = 0.3
surfaces = [0.2, 0.4, 0.6, 0.8, 1.0]

M_booz = int(np.ceil(1.5 * M))
N_booz = int(np.ceil(1.5 * N))
M_grid = int(np.ceil(1.5 * M_booz))
N_grid = int(np.ceil(1.5 * N_booz))
minor_radius = 1 / aspect_ratio
flux = np.pi * minor_radius**2

surface = FourierRZToroidalSurface.from_qp_model(
    major_radius=1,
    aspect_ratio=aspect_ratio,
    mirror_ratio=mirror_ratio,
    elongation=elongation,
    torsion=torsion,
    NFP=NFP,
    positive_iota=True,
    sym=True,
)
eq = Equilibrium(
    Psi=flux,
    NFP=NFP,
    L=L,
    M=M,
    N=N,
    sym=True,
    surface=surface,
    L_well=L_well,
    M_well=M_well,
    L_omni=L_omni,
    M_omni=M_omni,
    N_omni=N_omni,
    well_l=np.concatenate(
        (
            np.linspace(1 - mirror_ratio, 1 + mirror_ratio, M_well),
            np.zeros((L_well * M_well,)),
        )
    ),
)
eq = solve_continuation_automatic(
    eq, objective="vacuum", ftol=1e-2, xtol=1e-6, gtol=1e-6, nfev=50, verbose=3
)[-1]
if not sym:
    eq.change_resolution(sym=False)
eq.save("poloidal_qs.h5")

constraints = (
    FixAxisR(),
    FixAxisZ(),
    FixCurrent(),
    FixLambdaGauge(),
    FixPsi(),
    FixWell(
        target=np.array([1 - mirror_ratio, 1 + mirror_ratio]),
        indices=np.array([0, M_well - 1]),
    ),
    StraightBmaxContour(),
)

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
    (CurrentDensity(bounds=(-1e-6, 1e-6), weight=3e2),) + tuple(objs.values())
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
eq.save("poloidal_qs.h5")

VMECIO.save(eq, "wout_poloidal_qs.nc", surfs=256)
