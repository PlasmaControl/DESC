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
from ._current_potential import CurrentPotentialField, FourierCurrentPotentialField
from ._dommaschk import DommaschkPotentialField, dommaschk_potential
