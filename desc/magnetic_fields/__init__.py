"""Classes for Magnetic Fields."""

from ._core import (
    MagneticFieldFromUser,
    OmnigenousField,
    PoloidalMagneticField,
    ScalarPotentialField,
    ScaledMagneticField,
    SplineMagneticField,
    SumMagneticField,
    ToroidalMagneticField,
    VectorPotentialField,
    VerticalMagneticField,
    _MagneticField,
    field_line_integrate,
    read_BNORM_file,
)
from ._current_potential import (
    CurrentPotentialField,
    FourierCurrentPotentialField,
    solve_regularized_surface_current,
)
from ._dommaschk import DommaschkPotentialField, dommaschk_potential

__all__ = [
    "CurrentPotentialField",
    "DommaschkPotentialField",
    "FourierCurrentPotentialField",
    "MagneticFieldFromUser",
    "OmnigenousField",
    "PoloidalMagneticField",
    "ScalarPotentialField",
    "ScaledMagneticField",
    "SplineMagneticField",
    "SumMagneticField",
    "ToroidalMagneticField",
    "VectorPotentialField",
    "VerticalMagneticField",
    "field_line_integrate",
    "read_BNORM_file",
    "solve_regularized_surface_current",
]
