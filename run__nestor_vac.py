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
eqf = load("eqfv_freeb_test_iota.h5")
veq = eqf[-1]


veq.resolution_summary()
print("==========SOLVING VEQ==========")
veq.solve(ftol=1e-2, xtol=1e-6, gtol=1e-6, maxiter=100, verbose=3)


surf = veq.get_surface_at(1)
# ensure positive jacobian
one = np.ones_like(surf.R_lmn)
one[surf.R_basis.modes[:,1] < 0] *= -1
surf.R_lmn *= one
one = np.ones_like(surf.Z_lmn)
one[surf.Z_basis.modes[:,1] < 0] *= -1
surf.Z_lmn *= one
surf.change_resolution(M=1, N=1)

eq = Equilibrium(
    surface=surf,
    Psi=veq.Psi,
    pressure=veq.pressure,
    iota=-veq.iota,
    spectral_indexing=veq.spectral_indexing,
    sym=veq.sym,
    NFP=veq.NFP,
)

eq.change_resolution(
    veq.L // 3,
    veq.M // 3,
    veq.N // 3,
    veq.L_grid // 3,
    veq.M_grid // 3,
    veq.N_grid // 3,
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
out = eq1._optimize(
    ObjectiveFunction(bc_objective),
    ObjectiveFunction(fb_objective),
    maxiter=30,
    verbose=3,
    perturb_options={"order": 2, "dZb": True, "dRb": True, "tr_ratio": [0.01, 0.01]},
)

eq1.save("run__nestor_vac_out1.h5")
with open("run__nestor_vac_out1.pkl", "wb+") as f:
    pickle.dump(out, f)


eq2 = eq1.copy()

eq2.change_resolution(
    veq.L // 3 * 2,
    veq.M // 3 * 2,
    veq.N // 3 * 2,
    veq.L_grid // 3 * 2,
    veq.M_grid // 3 * 2,
    veq.N_grid // 3 * 2,
)
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
out = eq2._optimize(
    ObjectiveFunction(bc_objective),
    ObjectiveFunction(fb_objective),
    maxiter=30,
    verbose=3,
    perturb_options={"order": 2, "dZb": True, "dRb": True, "tr_ratio": [0.01, 0.01]},
)


eq2.save("run__nestor_vac_out2.h5")
with open("run__nestor_vac_out2.pkl", "wb+") as f:
    pickle.dump(out, f)


eq3 = eq2.copy()

eq3.change_resolution(veq.L, veq.M, veq.N, veq.L_grid, veq.M_grid, veq.N_grid)
print("==========SOLVING EQ3==========")
eq3.solve(ftol=1e-2, verbose=3)


bc_objective = BoundaryErrorNESTOR(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
    FixIota(),
    FixPsi(),
)

fb_objective.build(eq3)
bc_objective.build(eq3)

print("==========OPTIMIZING EQ3==========")
out = eq3._optimize(
    ObjectiveFunction(bc_objective),
    ObjectiveFunction(fb_objective),
    maxiter=30,
    verbose=3,
    perturb_options={"order": 2, "dZb": True, "dRb": True, "tr_ratio": [0.01, 0.01]},
)


eq3.save("run__nestor_vac_out3.h5")
with open("run__nestor_vac_out3.pkl", "wb+") as f:
    pickle.dump(out, f)

print("==========OPTIMIZING VEQ==========")
out = veq._optimize(
    ObjectiveFunction(bc_objective),
    ObjectiveFunction(fb_objective),
    maxiter=30,
    verbose=3,
    perturb_options={"order": 2, "dZb": True, "dRb": True, "tr_ratio": [0.01, 0.01]},
)

veq.save("run__nestor_vac_outv.h5")
with open("run__nestor_vac_outv.pkl", "wb+") as f:
    pickle.dump(out, f)
