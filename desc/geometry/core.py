from abc import ABC, abstractmethod

from desc.backend import jnp
from desc.io import IOAble
from .utils import rotation_matrix, reflection_matrix


class Curve(IOAble, ABC):
    """Abstract base class for 1D curves in 3D space."""

    _io_attrs_ = ["_name", "_grid", "shift", "rotmat"]

    def __init__(self, name=""):
        self.shift = jnp.array([0, 0, 0])
        self.rotmat = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.name = name

    @property
    def name(self):
        """Name of the curve."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = new

    @property
    @abstractmethod
    def grid(self):
        """Grid for computation."""

    @abstractmethod
    def compute_coordinates(self, params=None, grid=None, dt=0):
        """Compute real space coordinates on predefined grid."""

    @abstractmethod
    def compute_frenet_frame(self, params=None, grid=None):
        """Compute Frenet frame on predefined grid."""

    @abstractmethod
    def compute_curvature(self, params=None, grid=None):
        """Compute curvature on predefined grid."""

    @abstractmethod
    def compute_torsion(self, params=None, grid=None):
        """Compute torsion on predefined grid."""

    @abstractmethod
    def compute_length(self, params=None, grid=None):
        """Compute the length of the curve using specified nodes for quadrature."""

    def translate(self, displacement=[0, 0, 0]):
        """Translate the curve by a rigid displacement in x, y, z"""
        self.shift += jnp.asarray(displacement)

    def rotate(self, axis=[0, 0, 1], angle=0):
        """Rotate the curve by a fixed angle about axis in xyz coordinates"""
        R = rotation_matrix(axis, angle)
        self.rotmat = R @ self.rotmat
        self.shift = self.shift @ R.T

    def flip(self, normal):
        """Flip the curve about the plane with specified normal"""
        F = reflection_matrix(normal)
        self.rotmat = F @ self.rotmat
        self.shift = self.shift @ F.T

    def __repr__(self):
        """string form of the object"""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, grid={})".format(self.name, self.grid)
        )


class Surface(IOAble, ABC):
    """Abstract base class for 2d surfaces in 3d space."""

    _io_attrs_ = ["_name", "_grid", "_sym", "_L", "_M", "_N"]

    @property
    def name(self):
        """Name of the surface."""
        return self._name

    @name.setter
    def name(self, new):
        self._name = new

    @property
    def L(self):
        """maximum radial mode number"""
        return self._L

    @property
    def M(self):
        """maximum poloidal mode number"""
        return self._M

    @property
    def N(self):
        """maximum toroidal mode number"""
        return self._N

    @property
    def sym(self):
        """Stellarator symmetry."""
        return self._sym

    @property
    @abstractmethod
    def grid(self):
        """Grid for computation."""

    @abstractmethod
    def change_resolution(self, *args, **kwargs):
        """Change the maximum resolution."""

    @abstractmethod
    def compute_coordinates(self, params=None, grid=None, dt=0, dz=0):
        """Compute coordinate values at specified nodes."""

    @abstractmethod
    def compute_normal(self, params=None, grid=None):
        """Compute normal vectors to the surface on predefined grid."""

    @abstractmethod
    def compute_surface_area(self, params=None, grids=None):
        """Compute surface area via quadrature."""

    @abstractmethod
    def compute_curvature(self, params=None, grid=None):
        """Compute gaussian and mean curvature."""

    def __repr__(self):
        """string form of the object"""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (name={}, grid={})".format(self.name, self.grid)
        )
