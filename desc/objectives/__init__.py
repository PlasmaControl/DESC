"""Classes defining objectives for equilibrium and optimization."""

from ._bootstrap import BootstrapRedlConsistency
from ._equilibrium import (
    CurrentDensity,
    Energy,
    ForceBalance,
    HelicalForceBalance,
    RadialForceBalance,
)
from ._generic import (
    Generic1DObjective,
    Generic0DObjective,
    RotationalTransform,
    ToroidalCurrent,
)
from ._geometry import AspectRatio, Volume
from ._qs import QuasisymmetryBoozer, QuasisymmetryTripleProduct, QuasisymmetryTwoTerm
from ._stability import MagneticWell, MercierStability
from ._wrappers import WrappedEquilibriumObjective
from .linear_objectives import (
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixLambdaGauge,
    FixPressure,
    FixPsi,
)
from .objective_funs import ObjectiveFunction
from .utils import get_equilibrium_objective, get_fixed_boundary_constraints
