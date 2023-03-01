# from desc import set_device
# set_device("gpu")

import desc.io
import numpy as np
import pickle

from desc.compute.utils import compress
from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio,
    CurrentDensity,
    Elongation,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixLambda,
    FixPressure,
    FixPsi,
    FixQIShape,
    FixQIShift,
    FixR,
    FixZ,
    MirrorRatio,
    ObjectiveFunction,
    QuasiIsodynamic,
)


L_QI = 8
M_QI = 6
N_QI = 6
surfs = 5

# load Goodman solution
eq_opt = desc.io.load("Goodman_QI_nfp1_vacuum_M7_N7_output.h5")[-1]

# create intial guess
eq_init = desc.io.load("Goodman_QI_nfp1_vacuum_M2_N2_output.h5")[-1]
eq = eq_init.copy()

M_booz = int(np.ceil(eq.M * 1.5))
N_booz = int(np.ceil(eq.N * 1.5))
M_grid = int(eq.M * 2)
N_grid = int(eq.N * 2)

rho = 1 - np.linspace(0, 1, num=surfs, endpoint=False)
grid0 = LinearGrid(M=M_grid, N=N_grid, NFP=eq.NFP, sym=False, rho=1e-2)
grid_all = LinearGrid(M=M_grid, N=N_grid, NFP=eq.NFP, sym=False, rho=rho)
grids = {}
objs = {}
for r in rho:
    name = "{:2.3f}".format(r)
    grids[name] = LinearGrid(M=M_grid, N=N_grid, NFP=eq_opt.NFP, sym=False, rho=r)
    objs[name] = QuasiIsodynamic(
        grid=grids[name],
        L_QI=L_QI,
        M_QI=M_QI,
        N_QI=N_QI,
        M_booz=M_booz,
        N_booz=N_booz,
        name=name,
    )

data = eq_opt.compute(["R0/a", "a_major/a_minor"])
aspect_ratio = data["R0/a"]
elongation = data["a_major/a_minor"]
data = eq_opt.compute("mirror ratio", grid=grid0)
mirror_ratio = compress(grid0, data["mirror ratio"])[0]

print("\n======================")
print("Aspect Ratio limit: {:.2f}".format(aspect_ratio))
print("Elongation limit: {:.2f}".format(elongation))
print("Mirror Ratio limit: {:.2f}".format(mirror_ratio))
print("======================\n")

# find QI parameters
print("\n====================")
print("Optimizing QI params")
print("====================\n")
objective = ObjectiveFunction(tuple(objs.values()))
eq_opt, result = eq_opt.optimize(
    objective=objective,
    constraints=(
        FixR(),
        FixZ(),
        FixLambda(),
        FixPressure(),
        FixCurrent(),
        FixPsi(),
    ),
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    maxiter=200,
    verbose=3,
)

# save QI parameters
QI_params = {}
for name in grids.keys():
    QI_params[name] = {
        "QI_l": np.array(result["history"]["QI_l {}".format(name)][-1]),
        "QI_mn": np.array(result["history"]["QI_mn {}".format(name)][-1]),
    }

with open("QI_params.pkl", "wb") as handle:
    pickle.dump(QI_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

weight = 1e-2

# optimize the boundary to improve QI
for m in range(2, 8):
    n = m  # 2 * m - 2
    print("\n====================================")
    print("Optimizing boundary modes m,n <= {},{}".format(m, n))
    print("====================================\n")

    # objectives & constraints
    objs = {}
    cons_l = {}
    cons_mn = {}
    for name in grids.keys():
        objs[name] = QuasiIsodynamic(
            grid=grids[name],
            L_QI=L_QI,
            M_QI=M_QI,
            N_QI=N_QI,
            M_booz=M_booz,
            N_booz=N_booz,
            QI_l=QI_params[name]["QI_l"],
            QI_mn=QI_params[name]["QI_mn"],
            name=name,
        )
        cons_l[name] = FixQIShape(target=QI_params[name]["QI_l"], name=name)
        cons_mn[name] = FixQIShift(target=QI_params[name]["QI_mn"], name=name)
    geo_objs = (
        AspectRatio(bounds=(0, aspect_ratio), weight=weight),
        Elongation(bounds=(0, elongation), weight=weight),
        MirrorRatio(bounds=(0, mirror_ratio), grid=grid0, weight=weight),
    )
    objective = ObjectiveFunction(tuple(objs.values()) + geo_objs)

    idxR = np.where((np.abs(eq.surface.R_basis.modes) > [0, m, n]).any(axis=1))[0]
    idxZ = np.where((np.abs(eq.surface.Z_basis.modes) > [0, m, n]).any(axis=1))[0]
    R_modes = np.vstack(([0, 0, 0], eq.surface.R_basis.modes[idxR, :]))
    Z_modes = eq.surface.Z_basis.modes[idxZ, :]
    constraints = (
        CurrentDensity(),
        FixBoundaryR(modes=R_modes),
        FixBoundaryZ(modes=Z_modes),
        FixPressure(),
        FixCurrent(),
        FixPsi(),
    ) + tuple(cons_l.values()) + tuple(cons_mn.values())

    # optimize
    eq, result = eq.optimize(
        objective=objective,
        constraints=constraints,
        verbose=3,
        options={
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0},
        },
    )

    weight *= 2

eq.save("Goodman_QI_nfp1_vacuum_DESC.h5")
