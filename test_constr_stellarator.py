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
    Zero
)
from desc.vmec import VMECIO
from desc.vmec_utils import vmec_boundary_subspace
import desc.io
from desc.optimize import Optimizer


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
options = {"perturb_options": {"order": 1}}
eq.optimize(objective, constraints, options=options)

#np.testing.assert_allclose(eq.compute("V")["R0/a"], 2.5)

#%% Constrained Optimization
path = '/home/pk123/DESC/examples/DESC/SOLOVEV_output.h5'
eq = desc.io.load(path)[-1]
objective = ObjectiveFunction(Zero())
constraints = (
    ForceBalance(),
    FixBoundaryR(fixed_boundary = True),
    FixBoundaryZ(modes=eq.surface.Z_basis.modes[0:-1, :],fixed_boundary = True),
    FixPressure(),
    FixIota(),
    FixPsi(),
)
options = {"perturb_options": {"order": 1}}
result = eq.optimize(objective, constraints, optimizer = Optimizer("auglag"), options=options)
