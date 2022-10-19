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

os.getcwd()

ext_field = load("./ext_field.h5")
eq = load("run__nestor_out3.h5")

eq.change_resolution(
    24,
    12,
    12,
    30,
    15,
    15,
)
print("==========SOLVING EQ4==========")
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


print("==========OPTIMIZING EQ4==========")
out = eq._optimize(
    ObjectiveFunction(bc_objective),
    ObjectiveFunction(fb_objective),
    maxiter=30,
    verbose=3,
    perturb_options={"order": 2, "dZb": True, "dRb": True, "tr_ratio": [0.01, 0.01]},
)

eq.save("run__nestor_out4.h5")
with open("run__nestor_out4.pkl", "wb+") as f:
    pickle.dump(out, f)
