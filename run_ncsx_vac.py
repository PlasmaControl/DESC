import numpy as np
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
from desc import set_device

set_device("gpu")

import desc.examples
from desc.equilibrium import Equilibrium
from desc.io import load
from desc.objectives import (
    BoundaryErrorNESTOR,
    FixIota,
    FixLambdaGauge,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)

os.getcwd()

ext_field = load("./ext_field_ncsx.h5")
eqf = desc.examples.get("NCSX", "all")
veq = eqf[-1]

veq.change_resolution(L=20, M=10, N=10, L_grid=30, M_grid=15, N_grid=15)
veq.resolution_summary()

surf = veq.get_surface_at(1)
# # ensure positive jacobian
# one = np.ones_like(surf.R_lmn)
# one[surf.R_basis.modes[:,1] < 0] *= -1
# surf.R_lmn *= one
# one = np.ones_like(surf.Z_lmn)
# one[surf.Z_basis.modes[:,1] < 0] *= -1
# surf.Z_lmn *= one
# surf.change_resolution(M=1, N=1)
iota = veq.iota.copy()
# iota.params *= -1

pressure = veq.pressure.copy()
pressure.params *= 0

eq = Equilibrium(
    Psi=veq.Psi,
    surface=surf,
    pressure=pressure,
    iota=iota,
    spectral_indexing=veq.spectral_indexing,
    sym=veq.sym,
    NFP=int(veq.NFP),
)

eq.change_resolution(
    veq.L // 3,
    veq.M // 3,
    3,
    veq.L_grid // 3,
    veq.M_grid // 3,
    6,
)
print("==========SOLVING EQ1==========")
eq.solve(ftol=1e-2, verbose=3)


bc_objective = BoundaryErrorNESTOR(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
    FixLambdaGauge(),
    FixIota(),
    FixPsi(),
)

fb_objective.build(eq)
bc_objective.build(eq)


eq1 = eq.copy()
print("==========OPTIMIZING EQ1==========")
out = eq1.optimize(
    objective,
    constraints,
    maxiter=50,
    verbose=3,
    options={"initial_trust_ratio": 0.01}
)

eq1.save("run_ncsx_vac_out1.h5")
with open("run_ncsx_vac_out1.pkl", "wb+") as f:
    pickle.dump(out, f)


eq2 = eq1.copy()

eq2.change_resolution(
    veq.L //3*2,
    veq.M//3*2,
    6,
    veq.L_grid//3*2,
    veq.M_grid//3*2,
    9,
)
print("==========SOLVING EQ2==========")
eq2.solve(ftol=1e-2, verbose=3)


bc_objective = BoundaryErrorNESTOR(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
    FixLambdaGauge(),
    FixIota(),
    FixPsi(),
)

fb_objective.build(eq2)
bc_objective.build(eq2)

print("==========OPTIMIZING EQ2==========")
out = eq2.optimize(
    objective,
    constraints,
    maxiter=50,
    verbose=3,
    options={"initial_trust_ratio": 0.01}
)


eq2.save("run_ncsx_vac_out2.h5")
with open("run_ncsx_vac_out2.pkl", "wb+") as f:
    pickle.dump(out, f)


eq3 = eq2.copy()

eq3.change_resolution(
    veq.L,
    veq.M,
    veq.N,
    veq.L_grid,
    veq.M_grid,
    veq.N_grid,
)
print("==========SOLVING EQ3==========")
eq3.solve(ftol=1e-2, verbose=3)


bc_objective = BoundaryErrorNESTOR(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
    FixLambdaGauge(),
    FixIota(),
    FixPsi(),
)

fb_objective.build(eq3)
bc_objective.build(eq3)

print("==========OPTIMIZING EQ3==========")
out = eq3.optimize(
    objective,
    constraints,
    maxiter=50,
    verbose=3,
    options={"initial_trust_ratio": 0.01}
)


eq2.save("run_ncsx_vac_out3.h5")
with open("run_ncsx_vac_out3.pkl", "wb+") as f:
    pickle.dump(out, f)



print("==========OPTIMIZING VEQ==========")
out = veq.optimize(
    objective,
    constraints,
    maxiter=50,
    verbose=3,
    options={"initial_trust_ratio": 0.01}
)

veq.save("run_ncsx_vac_outv.h5")
with open("run_ncsx_vac_outv.pkl", "wb+") as f:
    pickle.dump(out, f)
