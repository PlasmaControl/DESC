from .objective_funs import ObjectiveFunction
from .linear_objectives import (
    FixedBoundaryR,
    FixedBoundaryZ,
    FixedPressure,
    FixedIota,
    FixedPsi,
    LCFSBoundary,
    TargetIota,
)
from .nonlinear_objectives import (
    Volume,
    Energy,
    RadialForceBalance,
    HelicalForceBalance,
    RadialCurrent,
    PoloidalCurrent,
    ToroidalCurrent,
    QuasisymmetryBoozer,
    QuasisymmetryFluxFunction,
    QuasisymmetryTripleProduct,
)


__all__ = [
    "ObjectiveFunction",
    "FixedBoundaryR",
    "FixedBoundaryZ",
    "FixedPressure",
    "FixedIota",
    "FixedPsi",
    "LCFSBoundary",
    "TargetIota",
    "Volume",
    "Energy",
    "RadialForceBalance",
    "HelicalForceBalance",
    "RadialCurrent",
    "PoloidalCurrent",
    "ToroidalCurrent",
    "QuasisymmetryBoozer",
    "QuasisymmetryFluxFunction",
    "QuasisymmetryTripleProduct",
]

ForceBalanceObjective = ObjectiveFunction(
    (RadialForceBalance(), HelicalForceBalance()),
    (
        FixedBoundaryR(),
        FixedBoundaryZ(),
        FixedPressure(),
        FixedIota(),
        FixedPsi(),
        LCFSBoundary(),
    ),
)
EnergyObjective = ObjectiveFunction(
    (RadialForceBalance(), HelicalForceBalance()),
    (
        FixedBoundaryR(),
        FixedBoundaryZ(),
        FixedPressure(),
        FixedIota(),
        FixedPsi(),
        LCFSBoundary(),
    ),
)
