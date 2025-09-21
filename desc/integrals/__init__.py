"""Classes for function integration."""

from ._bounce_utils import fast_chebyshev, fast_cubic_spline, fourier_chebyshev
from ._interp_utils import nufft1d2r, nufft2d2r
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
