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
from ._omnigenity import (
    Isodynamicity,
    Omnigenity,
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
    FixLambda,
    FixLambdaGauge,
    FixModeR,
    FixModeZ,
    FixOmni_l,
    FixOmni_lmn,
    FixPressure,
    FixPsi,
    FixR,
    FixSumModesR,
    FixSumModesZ,
    FixThetaSFL,
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
