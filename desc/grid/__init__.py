"""Classes for representing collocation grids of coordinates."""

import warnings

from .core import AbstractGrid
from .curve import AbstractGridCurve, CustomGridCurve, LinearGridCurve
from .flux import (
    AbstractGridFlux,
    ConcentricGridFlux,
    CustomGridFlux,
    LinearGridFlux,
    QuadratureGridFlux,
)
from .surface import (
    AbstractGridToroidalSurface,
    CustomGridToroidalSurface,
    LinearGridToroidalSurface,
)
from .utils import (
    cf_to_dec,
    dec_to_cf,
    find_least_rational_surfaces,
    find_most_distant,
    find_most_rational_surfaces,
    midpoint_spacing,
    most_rational,
    n_most_rational,
    periodic_spacing,
)


def __getattr__(name):
    """Get new classes for deprecated names."""
    if name == "Grid":
        warnings.warn(FutureWarning("Grid is deprecated, use CustomGridFlux instead."))
        return CustomGridFlux
    elif name == "LinearGrid":
        warnings.warn(
            FutureWarning("LinearGrid is deprecated, use LinearGridFlux instead.")
        )
        return LinearGridFlux
    elif name == "ConcentricGrid":
        warnings.warn(
            FutureWarning(
                "ConcentricGrid is deprecated, use ConcentricGridFlux instead."
            )
        )
        return ConcentricGridFlux
    elif name == "QuadratureGrid":
        warnings.warn(
            FutureWarning(
                "QuadratureGrid is deprecated, use QuadratureGridFlux instead."
            )
        )
        return QuadratureGridFlux
