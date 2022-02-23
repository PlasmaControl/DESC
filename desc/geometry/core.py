from abc import ABC, abstractmethod
from desc.backend import jnp
from desc.grid import Grid, LinearGrid, ConcentricGrid, QuadratureGrid
from desc.io import IOAble


def reflection_matrix(normal):
    """Matrix to reflect points across plane through origin with specified normal"""
    normal = jnp.asarray(normal)
    R = jnp.eye(3) - 2 * jnp.outer(normal, normal) / jnp.inner(normal, normal)
    return R


def rotation_matrix(axis, angle=None):
    """Matrix to rotate points about axis by given angle"""
    if angle is None:
        angle = jnp.linalg.norm(axis)
    axis = jnp.asarray(axis) / jnp.linalg.norm(axis)
    R1 = jnp.cos(angle) * jnp.eye(3)
    R2 = jnp.sin(angle) * jnp.cross(axis, jnp.identity(axis.shape[0]) * -1)
    R3 = (1 - jnp.cos(angle)) * jnp.outer(axis, axis)
    return R1 + R2 + R3


def xyz2rpz(pts):
    """Transform points from cartesian to polar form

    Parameters
    ----------
    pts : ndarray, shape(n,3)
        points in xyz coordinates

    Returns
    -------
    pts : ndarray, shape(n,3)
        points in rpz coordinates
    """
    x, y, z = pts.T
    r = jnp.sqrt(x ** 2 + y ** 2)
    p = jnp.arctan2(y, x)
    return jnp.array([r, p, z]).T


def rpz2xyz(pts):
    """Transform points from polar to cartesian form

    Parameters
    ----------
    pts : ndarray, shape(n,3)
        points in rpz coordinates

    Returns
    -------
    pts : ndarray, shape(n,3)
        points in xyz coordinates
    """
    r, p, z = pts.T
    x = r * jnp.cos(p)
    y = r * jnp.sin(p)
    return jnp.array([x, y, z]).T


def xyz2rpz_vec(vec, x=None, y=None, phi=None):
    """Transform vectors from cartesian to polar form.

    Parameters
    ----------
    vec : ndarray, shape(n,3)
        vectors, in cartesian (xyz) form
    x, y, phi : ndarray, shape(n,)
        anchor points for vectors. Either x and y, or phi must be supplied

    Returns
    -------
    vec : ndarray, shape(n,3)
        vectors, in polar (rpz) form
    """
    if x is not None and y is not None:
        phi = jnp.arctan2(y, x)
    rot = jnp.array(
        [
            [jnp.cos(phi), jnp.sin(phi), jnp.zeros_like(phi)],
            [-jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
            [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
        ]
    )
    rot = jnp.moveaxis(rot, -1, 0)
    polar = jnp.matmul(rot, vec.reshape((-1, 3, 1)))
    return polar.reshape((-1, 3))


def rpz2xyz_vec(vec, x=None, y=None, phi=None):
    """Transform vectors from polar to cartesian form.

    Parameters
    ----------
    vec : ndarray, shape(n,3)
        vectors, in polar (rpz) form
    x, y, phi : ndarray, shape(n,)
        anchor points for vectors. Either x and y, or phi must be supplied

    Returns
    -------
    vec : ndarray, shape(n,3)
        vectors, in cartesian (xyz) form
    """
    if x is not None and y is not None:
        phi = jnp.arctan2(y, x)
    rot = jnp.array(
        [
            [jnp.cos(phi), -jnp.sin(phi), jnp.zeros_like(phi)],
            [jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
            [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
        ]
    )
    rot = jnp.moveaxis(rot, -1, 0)
    cart = jnp.matmul(rot, vec.reshape((-1, 3, 1)))
    return cart.reshape((-1, 3))


class Curve(IOAble, ABC):
    """Abstract base class for 1D curves in 3D space"""

    _io_attrs_ = ["_name", "_grid", "shift", "rotmat"]

    def __init__(self, name):
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
        """Default grid for computation"""

    @abstractmethod
    def compute_coordinates(self, params=None, grid=None, dt=0):
        """Compute real space coordinates on predefined grid"""

    @abstractmethod
    def compute_frenet_frame(self, params=None, grid=None):
        """Compute frenet frame on predefined grid"""

    @abstractmethod
    def compute_curvature(self, params=None, grid=None):
        """Compute curvature on predefined grid"""

    @abstractmethod
    def compute_torsion(self, params=None, grid=None):
        """Compute torsion on predefined grid"""

    @abstractmethod
    def compute_length(self, params=None, grid=None):
        """Compute the length of the curve using specified nodes for quadrature"""

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


class Surface(IOAble, ABC):
    """Abstract base class for 2d surfaces in 3d space,
    such as flux surfaces, plasma boundaries, poincare sections
    """

    _io_attrs_ = ["_name", "_grid", "_sym", "_L", "_M", "_N"]

    @property
    def name(self):
        """Name of the surface"""
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
        """stellarator symmetry"""
        return self._sym

    @property
    @abstractmethod
    def grid(self):
        """Default grid for computation"""

    @abstractmethod
    def compute_coordinates(self, params=None, grid=None, dt=0, dz=0):
        """Compute coordinate values at specified nodes"""

    @abstractmethod
    def compute_normal(self, params=None, grid=None):
        """Compute normal vectors to the surface on predefined grid"""

    @abstractmethod
    def compute_surface_area(self, params=None, grids=None):
        """Compute surface area via quadrature"""

    @abstractmethod
    def compute_curvature(self, params=None, grid=None):
        """Compute gaussian and mean curvature"""
