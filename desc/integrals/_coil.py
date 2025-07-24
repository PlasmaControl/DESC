"""Integrals for magnetic scalar potential of coil field."""

from desc.backend import jnp
from desc.utils import rpz2xyz


def scalar_potential(eval_pts, source_pts, I):  # noqa: E741
    """Compute equation C in [1].

    Parameters
    ----------
    eval_pts : jnp.ndarray
        Shape (num eval, 3)
        Evaluation points of cylindrical basis.
    source_pts : jnp.ndarray
        Shape (num source, )
        Samples of coil loop position uniformly spaced under
        parameter spanning [0, 2π).
    I : float
        Net current along coil in Tesla-meters.

    Returns
    -------
    Phi : jnp.ndarray
        Shape (num eval, )
        Scalar potential.

    """
    dx = jnp.linalg.norm(
        rpz2xyz(source_pts) - rpz2xyz(eval_pts)[:, jnp.newaxis],
        axis=-1,
    )
    R_c, phi_c, Z_c = source_pts.T
    R, phi, Z = eval_pts.T

    D = dx * (dx + Z_c - Z[:, jnp.newaxis])
    dt = jnp.linspace(0, 2 * jnp.pi, R_c.size) - phi[:, jnp.newaxis]
    N = R_c * (R_c + R[:, jnp.newaxis] * (jnp.sin(dt) - jnp.cos(dt)))

    Phi = I / (2 * jnp.pi) * (jnp.mean(N / D, axis=-1) - phi)
    assert Phi.shape == R.shape
    return Phi
