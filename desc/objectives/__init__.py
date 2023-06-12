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
    ObjectiveFromUser,
    RotationalTransform,
    ToroidalCurrent,
)
from ._geometry import (
    AspectRatio,
    BScaleLength,
    Elongation,
    MeanCurvature,
    PlasmaVesselDistance,
    PrincipalCurvature,
    Volume,
)
from ._qs import (
    Isodynamicity,
    QuasisymmetryBoozer,
    QuasisymmetryTripleProduct,
    QuasisymmetryTwoTerm,
)
from ._stability import MagneticWell, MercierStability
from .linear_objectives import (
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    FixAtomicNumber,
    FixAxisR,
    FixAxisZ,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambdaGauge,
    FixModeR,
    FixModeZ,
    FixPressure,
    FixPsi,
    FixSumModesR,
    FixSumModesZ,
    FixThetaSFL,
)
from .objective_funs import ObjectiveFunction
from .utils import (
    get_equilibrium_objective,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
)
