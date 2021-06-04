from abc import ABC, abstractmethod
import copy
from desc.backend import jnp
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.io import IOAble


def cart2pol(xyz):
    x, y, z = xyz.T
    r = jnp.sqrt(x ** 2 + y ** 2)
    phi = jnp.arctan2(y, x)
    return jnp.array([r, phi, z]).T


def pol2cart(rpz):
    r, p, z = rpz.T
    x = r * jnp.cos(p)
    y = r * jnp.sin(p)
    return jnp.array([x, y, z]).T


class Curve(ABC, IOAble):
    """Abstract base class for 1D curves in 3D space"""

    _io_attrs_ = ["_name", "_grid"]

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
    def transform(self, params, dt=0):
        """Compute real space coordinates on predefined grid"""

    @abstractmethod
    def compute_frenet_frame(self, params):
        """Compute frenet frame on predefined grid"""

    @abstractmethod
    def compute_curvature(self, params):
        """Compute curvature on predefined grid"""

    @abstractmethod
    def compute_torsion(self, params):
        """Compute torsion on predefined grid"""

    @abstractmethod
    def compute_length(self, nodes=None):
        """Compute the length of the curve using specified nodes for quadrature"""

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this curve."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new


class Surface(ABC, IOAble):
    """Abstract base class for 2d surfaces in 3d space,
    such as flux surfaces, plasma boundaries, poincare sections
    """

    _io_attrs_ = ["_name", "_grid", "_sym"]

    @property
    def name(self):
        """Name of the surface"""
        return self._name

    @name.setter
    def name(self, new):
        self._name = new

    @property
    def sym(self):
        """stellarator symmetry"""
        return self._sym

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

    @abstractmethod
    def compute_surface_area(self, nodes=None, params=None):
        """Compute surface area via quadrature"""

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this surface."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new
