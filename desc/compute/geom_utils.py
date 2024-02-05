"""Functions for converting between coordinate systems."""

import functools

from desc.backend import cond, jnp


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


def rotation_matrix(axis):
    """Matrix to rotate points about axis by given angle.

    Parameters
    ----------
    axis : array-like, shape(3,)
        Axis of rotation, in cartesian (X,Y,Z) coordinates.
        The norm of the vector is the angle of rotation, in radians.

    Returns
    -------
    rot : ndarray, shape(3,3)
        Matrix to rotate points in cartesian (X,Y,Z) coordinates.

    """
    axis = jnp.asarray(axis)
    eps = 1e2 * jnp.finfo(axis.dtype).eps
    return cond(
        jnp.all(jnp.abs(axis) < eps),
        lambda axis: jnp.eye(3),
        lambda axis: jnp.cos(jnp.linalg.norm(axis)) * jnp.eye(3)  # R1
        + jnp.sin(jnp.linalg.norm(axis))  # R2
        * jnp.cross(axis / jnp.linalg.norm(axis), jnp.identity(axis.shape[0]) * -1)
        + (1 - jnp.cos(jnp.linalg.norm(axis)))  # R3
        * jnp.outer(axis / jnp.linalg.norm(axis), axis / jnp.linalg.norm(axis)),
        axis,
    )


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
