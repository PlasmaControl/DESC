"""Classes defining objectives for equilibrium and optimization."""

from ._auglagrangian import AugLagrangian, AugLagrangianLS
from ._bootstrap import BootstrapRedlConsistency
from ._equilibrium import (
    CurrentDensity,
    Energy,
    ForceBalance,
    HelicalForceBalance,
    RadialForceBalance,
)
from ._generic import GenericObjective, RotationalTransform, ToroidalCurrent
from ._geometry import (
    AspectRatio,
    Elongation,
    MeanCurvature,
    PlasmaVesselDistance,
    PrincipalCurvature,
    Volume,
)
from ._iota_utils import IotaAt, MeanIota
from ._qs import QuasisymmetryBoozer, QuasisymmetryTripleProduct, QuasisymmetryTwoTerm
from ._stability import MagneticWell, MercierStability
from .linear_objectives import (
    FixAtomicNumber,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambdaGauge,
    FixPressure,
    FixPsi,
)
from .objective_funs import ObjectiveFunction
from .utils import get_equilibrium_objective, get_fixed_boundary_constraints
