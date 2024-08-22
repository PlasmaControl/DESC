"""Classes for function integration."""

from .singularities import (
    DFTInterpolator,
    FFTInterpolator,
    compute_B_plasma,
    singular_integral,
    virtual_casing_biot_savart,
)
from .surface_integral import (
    surface_averages,
    surface_averages_map,
    surface_integrals,
    surface_integrals_map,
    surface_integrals_transform,
    surface_max,
    surface_min,
    surface_variance,
)
