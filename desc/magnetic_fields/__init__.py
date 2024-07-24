"""Classes for Magnetic Fields."""

from ._core import (
    MagneticFieldFromUser,
    OmnigenousField,
    PiecewiseOmnigenousField,
    PoloidalMagneticField,
    ScalarPotentialField,
    ScaledMagneticField,
    SplineMagneticField,
    SumMagneticField,
    ToroidalMagneticField,
    VerticalMagneticField,
    _MagneticField,
    field_line_integrate,
    read_BNORM_file,
)
from ._current_potential import CurrentPotentialField, FourierCurrentPotentialField
from ._dommaschk import DommaschkPotentialField, dommaschk_potential
