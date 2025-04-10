"""Classes for function integration."""

from ._free_boundary import (
    FreeBoundarySolver,
    compute_B_plasma,
    virtual_casing_biot_savart,
)
from ._vacuum import VacuumSolver
from .bounce_integral import Bounce1D, Bounce2D
from .singularities import DFTInterpolator, FFTInterpolator, singular_integral
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
