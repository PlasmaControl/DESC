import unittest
import numpy as np
import pytest
from desc.grid import LinearGrid, ConcentricGrid
from desc.equilibrium import Equilibrium
from desc.objectives import (
    ObjectiveFunction,
    FixedBoundaryR,
    FixedBoundaryZ,
    FixedPressure,
    FixedIota,
    FixedPsi,
    LCFSBoundary,
    RadialForceBalance,
    HelicalForceBalance,
    Energy,
)


class TestObjectiveFunction(unittest.TestCase):
    """Test ObjectiveFunction class."""

    def test_something(self):
        """Test something."""
        pass
