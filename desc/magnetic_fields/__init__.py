"""Classes for Magnetic Fields."""

from ._core import (
    OmnigenousField,
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
from ._current_potential import (
    CurrentPotentialField,
    FourierCurrentPotentialField,
    run_regcoil,
)
from ._dommaschk import DommaschkPotentialField, dommaschk_potential
