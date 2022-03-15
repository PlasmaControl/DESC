from desc.backend import jnp


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
            [jnp.cos(phi), -jnp.sin(phi), jnp.zeros_like(phi)],
            [jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
            [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
        ]
    )
    rot = rot.T  # jnp.moveaxis(rot, -1, 0)
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
            [jnp.cos(phi), jnp.sin(phi), jnp.zeros_like(phi)],
            [-jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)],
            [jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)],
        ]
    )
    rot = jnp.moveaxis(rot, -1, 0)
    cart = jnp.matmul(rot, vec.reshape((-1, 3, 1)))
    return cart.reshape((-1, 3))
