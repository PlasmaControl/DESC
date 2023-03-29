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
    FluxGradient,
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
from .linear_objectives import (
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
    FixLambda,
    FixLambdaGauge,
    FixModeR,
    FixModeZ,
    FixPressure,
    FixPsi,
    FixQIShape,
    FixQIShift,
    FixR,
    FixSumModesR,
    FixSumModesZ,
    FixThetaSFL,
    FixZ,
)
from .objective_funs import ObjectiveFunction
from .utils import (
    get_equilibrium_objective,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
)
