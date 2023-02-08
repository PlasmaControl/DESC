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
    GenericObjective,
    MirrorRatio,
    RotationalTransform,
    ToroidalCurrent,
)
from ._geometry import (
    AspectRatio,
    Elongation,
    MeanCurvature,
    PlasmaVesselDistance,
    PrincipalCurvature,
    Volume,
)
from ._qs import (
    QuasiIsodynamic,
    QuasisymmetryBoozer,
    QuasisymmetryTwoTerm,
    QuasisymmetryTripleProduct,
)
from ._stability import MagneticWell, MercierStability
from ._wrappers import WrappedEquilibriumObjective
from .linear_objectives import (
    FixAtomicNumber,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambda,
    FixLambdaGauge,
    FixPressure,
    FixPsi,
    FixQIShape,
    FixQIShift,
    FixR,
    FixZ,
)
from .objective_funs import ObjectiveFunction
from .utils import get_equilibrium_objective, get_fixed_boundary_constraints
