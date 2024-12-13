"""Classes for representing geometric objects like curves and surfaces."""

from .core import Curve, Surface
from .curve import FourierPlanarCurve, FourierRZCurve, FourierXYZCurve, SplineXYZCurve
from .surface import FourierRZToroidalSurface, ZernikeRZToroidalSection
