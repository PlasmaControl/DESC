"""Classes for function integration."""

from .bounce_integral import Bounce1D, Bounce2D
from .singularities import (
    DFTInterpolator,
    FFTInterpolator,
    compute_B_plasma,
    singular_integral,
    virtual_casing_biot_savart,
)
from .surface_integral import (
    line_integrals,
    surface_averages,
    surface_averages_map,
    surface_integrals,
    surface_integrals_map,
    surface_integrals_transform,
    surface_max,
    surface_min,
    surface_variance,
)
