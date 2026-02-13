"""Classes for Synthetic Diagnostics."""

from ._core import DiagnosticSet
from ._magnetics import (
    PointBMeasurements,
    RogowskiCoilFourierPlanar,
    RogowskiCoilFourierRZ,
    RogowskiCoilFourierXY,
    RogowskiCoilFourierXYZ,
    RogowskiCoilSplineXYZ,
)

__all__ = [
    "DiagnosticSet",
    "PointBMeasurements",
    "RogowskiCoilFourierRZ",
    "RogowskiCoilFourierXYZ",
    "RogowskiCoilFourierPlanar",
    "RogowskiCoilFourierXY",
    "RogowskiCoilSplineXYZ",
]
