"""Classes defining objectives for equilibrium and optimization."""

from ._bootstrap import BootstrapRedlConsistency
from ._equilibrium import (
    CurrentDensity,
    Energy,
    ForceBalance,
    ForceBalanceAnisotropic,
    HelicalForceBalance,
    RadialForceBalance,
)
from ._free_boundary import BoundaryError, VacuumBoundaryError
from ._generic import GenericObjective, LinearObjectiveFromUser, ObjectiveFromUser
from ._geometry import (
    AspectRatio,
    BScaleLength,
    Elongation,
    GoodCoordinates,
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
from ._profiles import Pressure, RotationalTransform, Shear, ToroidalCurrent
from ._stability import MagneticWell, MercierStability
from .getters import (
    get_equilibrium_objective,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
    get_NAE_constraints,
    maybe_add_self_consistency,
)
from .linear_objectives import (
    AxisRSelfConsistency,
    AxisZSelfConsistency,
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    FixAnisotropy,
    FixAtomicNumber,
    FixAxisR,
    FixAxisZ,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixCurveRotation,
    FixCurveShift,
    FixElectronDensity,
    FixElectronTemperature,
    FixIonTemperature,
    FixIota,
    FixLambdaGauge,
    FixModeLambda,
    FixModeR,
    FixModeZ,
    FixOmniBmax,
    FixOmniMap,
    FixOmniWell,
    FixParameter,
    FixPressure,
    FixPsi,
    FixSumModesLambda,
    FixSumModesR,
    FixSumModesZ,
    FixThetaSFL,
)
from .objective_funs import ObjectiveFunction
