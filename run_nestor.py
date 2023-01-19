import numpy as np
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
from desc import set_device

set_device("gpu")


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
from desc.compat import ensure_positive_jacobian

os.getcwd()

ext_field = load("./ext_field.h5")
ext_field._axisym = False
eqf = load("eqfb_freeb_test_beta.h5")
veq = eqf[-1]
veq = ensure_positive_jacobian(veq)
veq.resolution_summary()
surf = veq.get_surface_at(1)
iota = veq.iota.copy()


eq = Equilibrium(
    surface=surf,
    Psi=veq.Psi,
    pressure=veq.pressure,
    iota=iota,
    spectral_indexing=veq.spectral_indexing,
    sym=veq.sym,
    NFP=veq.NFP,
)


eq.change_resolution(
    veq.L // 2,
    veq.M // 2,
    veq.N // 2,
    veq.L_grid // 2,
    veq.M_grid // 2,
    veq.N_grid // 2,
)
print("==========SOLVING EQ1==========")
eq.solve(ftol=1e-2, verbose=3)

from desc.objectives import (
    FixIota,
    FixLambdaGauge,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)

bc_objective = BoundaryErrorNESTOR(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
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
    maxiter=60,
    verbose=3,
    options={
        "perturb_options": {"order": 2},
        "ga_tr_ratio": 0.05,
    },
)

eq1.save("run_nestor_out1.h5")
with open("run_nestor_out1.pkl", "wb+") as f:
    pickle.dump(out, f)


eq2 = eq1.copy()

eq2.change_resolution(veq.L, veq.M, veq.N, veq.L_grid, veq.M_grid, veq.N_grid)
print("==========SOLVING EQ2==========")
eq2.solve(ftol=1e-2, verbose=3)


bc_objective = BoundaryErrorNESTOR(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
    FixIota(),
    FixPsi(),
)

fb_objective.build(eq2)
bc_objective.build(eq2)

print("==========OPTIMIZING EQ2==========")
out = eq2.optimize(
    objective,
    constraints,
    maxiter=60,
    verbose=3,
    options={
        "perturb_options": {"order": 2},
        "ga_tr_ratio": 0.05,
    },
)


eq2.save("run_nestor_out2.h5")
with open("run_nestor_out2.pkl", "wb+") as f:
    pickle.dump(out, f)

# print("==========OPTIMIZING VEQ==========")
# out = veq.optimize(
#     objective,
#     constraints,
#     maxiter=60,
#     verbose=3,
#     options={
#         "perturb_options": {"order": 2},
#         "ga_tr_ratio":  0.05,
#     },
# )

# veq.save("run_nestor_outv.h5")
# with open("run_nestor_outv.pkl", "wb+") as f:
#     pickle.dump(out, f)
