import numpy as np
from abc import ABC, abstractmethod
import copy

from desc.backend import jnp
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.io import IOAble
from desc.basis import DoubleFourierSeries, ZernikePolynomial


class Surface(ABC, IOAble):
    """Abstract base class for 2d surfaces in 3d space,
    such as flux surfaces, plasma boundaries, poincare sections
    """

    _io_attrs_ = ["_name", "_grid"]
    _object_lib_ = {
        "Grid": Grid,
        "LinearGrid": LinearGrid,
        "ConcentricGrid": ConcentricGrid,
        "QuadratureGrid": QuadratureGrid,
    }

    @property
    def name(self):
        """Name of the profile"""
        return self._name

    @name.setter
    def name(self, new):
        self._name = new

    @property
    @abstractmethod
    def grid(self):
        """Default grid for computation"""

    @abstractmethod
    def transform(self, params, dt=0, dz=0):
        """Compute real space coordinates on predefined grid"""

    @abstractmethod
    def compute_normal(self, params):
        """Compute normal vectors to the surface on predefined grid"""

    @abstractmethod
    def compute_coordinates(self, nodes, params, dt=0, dz=0):
        """Compute coordinate values at specified nodes"""

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this surface."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new
