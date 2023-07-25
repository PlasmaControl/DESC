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
    ObjectiveFromUser,
    RotationalTransform,
    ToroidalCurrent,
)
from ._geometry import (
    AspectRatio,
    BScaleLength,
    Elongation,
    FluxGradient,
    MeanCurvature,
    PlasmaVesselDistance,
    PrincipalCurvature,
    Volume,
)
from ._omnigenity import (
    Isodynamicity,
    Omnigenity,
    QuasisymmetryBoozer,
    QuasisymmetryTripleProduct,
    QuasisymmetryTwoTerm,
)
from ._stability import MagneticWell, MercierStability
from .linear_objectives import (
    AxisRSelfConsistency,
    AxisZSelfConsistency,
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
    FixLambda,
    FixLambdaGauge,
    FixModeR,
    FixModeZ,
    FixOmni,
    FixPressure,
    FixPsi,
    FixR,
    FixSumModesR,
    FixSumModesZ,
    FixThetaSFL,
    FixWell,
    FixZ,
    StraightBmaxContour,
)
from .objective_funs import ObjectiveFunction
from .utils import (
    get_equilibrium_objective,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
)
