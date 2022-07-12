import numpy as np
import desc.examples
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    RadialForceBalance,
    HelicalForceBalance,
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixPsi,
    Zero,
    GradientForceBalance
)
from desc.vmec import VMECIO
from desc.vmec_utils import vmec_boundary_subspace
import desc.io
from desc.optimize import Optimizer
from desc.grid import ConcentricGrid

#%% Original unconstrained optimization
path = '/home/pk123/DESC/examples/DESC/SOLOVEV_output.h5'
eq = desc.io.load(path)[-1]
objective = ObjectiveFunction(AspectRatio(target=2.5))
constraints = (
    ForceBalance(),
    FixBoundaryR(),
    FixBoundaryZ(modes=eq.surface.Z_basis.modes[0:-1, :]),
    FixPressure(),
    FixIota(),
    FixPsi(),
)
options = {"perturb_options": {"order": 2}}
result_unc = eq.optimize(objective, constraints, options=options)

#np.testing.assert_allclose(eq.compute("V")["R0/a"], 2.5)

#%% Constrained Optimization
path = '/home/pk123/DESC/examples/DESC/SOLOVEV_output.h5'
eq = desc.io.load(path)[-1]
objective = ObjectiveFunction(AspectRatio(target=2.5))
constraints = (
    #ForceBalance(grid=ConcentricGrid(eq.L, eq.M, eq.N, eq.NFP, eq.sym)),
    ForceBalance(),
    FixBoundaryR(fixed_boundary = True),
    FixBoundaryZ(modes=eq.surface.Z_basis.modes[0:-1, :],fixed_boundary = True),
    FixPressure(),
    FixIota(),
    FixPsi(),
)
# #constraints[0].build(eq,grid=ConcentricGrid(eq.L, eq.M, eq.N, eq.NFP, eq.sym))
options = {"perturb_options": {"order": 1}}
result = eq.optimize(objective, constraints, optimizer = Optimizer("auglag"), options=options)

#objective = ObjectiveFunction(Zero())

#%%
path = '/home/pk123/DESC/examples/DESC/SOLOVEV_output.h5'
eq1 = desc.io.load(path)[-1]
result1 = eq1.solve(objective = 'force',ftol = 0, gtol = 1e-04, maxiter = 200,verbose=3)

eq2 = desc.io.load(path)[-1]
result2 = eq2.solve(objective = 'gradient force',ftol = 0, xtol = 1e-12, gtol = 1e-06, maxiter = 200,verbose=3)