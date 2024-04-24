"""Functions for converting between coordinate systems."""

import functools

from desc.backend import jnp

from .utils import safenorm, safenormalize


def reflection_matrix(normal):
    """Matrix to reflect points across plane through origin with specified normal.

    Parameters
    ----------
    normal : array-like, shape(3,)
        Vector normal to plane of reflection, in cartesian (X,Y,Z) coordinates

    Returns
    -------
    flip : ndarray, shape(3,3)
        Matrix to flip points in cartesian (X,Y,Z) coordinates
    """
    normal = jnp.asarray(normal)
    R = jnp.eye(3) - 2 * jnp.outer(normal, normal) / jnp.inner(normal, normal)
    return R


def rotation_matrix(axis, angle=None):
    """Matrix to rotate points about axis by given angle.

    Parameters
    ----------
    axis : array-like, shape(3,)
        Axis of rotation, in cartesian (X,Y,Z) coordinates
    angle : float or None
        Angle to rotate by, in radians. If None, use norm of axis vector.

    Returns
    -------
    rotmat : ndarray, shape(3,3)
        Matrix to rotate points in cartesian (X,Y,Z) coordinates.

    """
    axis = jnp.asarray(axis)
    norm = safenorm(axis)
    axis = safenormalize(axis)
    if angle is None:
        angle = norm
    eps = 1e2 * jnp.finfo(axis.dtype).eps
    R1 = jnp.cos(angle) * jnp.eye(3)
    R2 = jnp.sin(angle) * jnp.cross(axis, jnp.identity(axis.shape[0]) * -1)
    R3 = (1 - jnp.cos(angle)) * jnp.outer(axis, axis)
    return jnp.where(norm < eps, jnp.eye(3), R1 + R2 + R3)  # if axis=0, no rotation


def xyz2rpz(pts):
    """Transform points from cartesian (X,Y,Z) to polar (R,phi,Z) form.

    Parameters
    ----------
    pts : ndarray, shape(...,3)
        points in cartesian (X,Y,Z) coordinates

    Returns
    -------
    pts : ndarray, shape(...,3)
        points in polar (R,phi,Z) coordinates

    """
    x, y, z = pts.T
    r = jnp.sqrt(x**2 + y**2)
    p = jnp.arctan2(y, x)
    return jnp.array([r, p, z]).T


def rpz2xyz(pts):
    """Transform points from polar (R,phi,Z) to cartesian (X,Y,Z) form.

    Parameters
    ----------
    pts : ndarray, shape(...,3)
        points in polar (R,phi,Z) coordinates

    Returns
    -------
    pts : ndarray, shape(...,3)
        points in cartesian (X,Y,Z) coordinates

    """
    r, p, z = pts.T
    x = r * jnp.cos(p)
    y = r * jnp.sin(p)
    return jnp.array([x, y, z]).T


def xyz2rpz_vec(vec, x=None, y=None, phi=None):
    """Transform vectors from cartesian (X,Y,Z) to polar (R,phi,Z) form.

    Parameters
    ----------
    vec : ndarray, shape(...,3)
        vectors, in cartesian (X,Y,Z) form
    x, y, phi : ndarray, shape(...,)
        anchor points for vectors. Either x and y, or phi must be supplied

    Returns
    -------
    vec : ndarray, shape(...,3)
        vectors, in polar (R,phi,Z) form

    """
    if x is not None and y is not None:
        phi = jnp.arctan2(y, x)

    @functools.partial(jnp.vectorize, signature="(3),()->(3)")
    def inner(vec, phi):
        rot = jnp.array(
            [
                [jnp.cos(phi), -jnp.sin(phi), jnp.zeros_like(phi)],
                [jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
                [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
            ]
        )
        rot = rot.T
        polar = jnp.matmul(rot, vec)
        return polar

    return inner(vec, phi)


def rpz2xyz_vec(vec, x=None, y=None, phi=None):
    """Transform vectors from polar (R,phi,Z) to cartesian (X,Y,Z) form.

    Parameters
    ----------
    vec : ndarray, shape(n,3)
        vectors, in polar (R,phi,Z) form
    x, y, phi : ndarray, shape(n,)
        anchor points for vectors. Either x and y, or phi must be supplied

    Returns
    -------
    vec : ndarray, shape(n,3)
        vectors, in cartesian (X,Y,Z) form

    """
    if x is not None and y is not None:
        phi = jnp.arctan2(y, x)

    @functools.partial(jnp.vectorize, signature="(3),()->(3)")
    def inner(vec, phi):
        rot = jnp.array(
            [
                [jnp.cos(phi), jnp.sin(phi), jnp.zeros_like(phi)],
                [-jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
                [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
            ]
        )
        rot = rot.T
        cart = jnp.matmul(rot, vec)
        return cart

    return inner(vec, phi)
