"""Fast transformable basis."""

from functools import partial

from desc.backend import flatnonzero, jnp, put
from desc.utils import setdefault


@partial(jnp.vectorize, signature="(m),(m)->(m)")
def _in_epigraph_and(is_intersect, df_dy_sign, /):
    """Set and epigraph of function f with the given set of points.

    Used to return only intersects where the straight line path between
    adjacent intersects resides in the epigraph of a continuous map ``f``.

    Parameters
    ----------
    is_intersect : jnp.ndarray
        Boolean array indicating whether index corresponds to an intersect.
    df_dy_sign : jnp.ndarray
        Shape ``is_intersect.shape``.
        Sign of ∂f/∂y (yᵢ) for f(yᵢ) = 0.

    Returns
    -------
    is_intersect : jnp.ndarray
        Boolean array indicating whether element is an intersect
        and satisfies the stated condition.

    Examples
    --------
    See ``desc/integrals/bounce_utils.py::bounce_points``.
    This is used there to ensure the domains of integration are magnetic wells.

    """
    # The pairs ``y1`` and ``y2`` are boundaries of an integral only if ``y1 <= y2``.
    # For the integrals to be over wells, it is required that the first intersect
    # has a non-positive derivative. Now, by continuity,
    # ``df_dy_sign[...,k]<=0`` implies ``df_dy_sign[...,k+1]>=0``,
    # so there can be at most one inversion, and if it exists, the inversion
    # must be at the first pair. To correct the inversion, it suffices to disqualify the
    # first intersect as a right boundary, except under an edge case of a series of
    # inflection points.
    idx = flatnonzero(is_intersect, size=2, fill_value=-1)
    edge_case = (
        (df_dy_sign[idx[0]] == 0)
        & (df_dy_sign[idx[1]] < 0)
        & is_intersect[idx[0]]
        & is_intersect[idx[1]]
        # In theory, we need to keep propagating this edge case, e.g.
        # (df_dy_sign[..., 1] < 0) | (
        #     (df_dy_sign[..., 1] == 0) & (df_dy_sign[..., 2] < 0)...
        # ).
        # At each step, the likelihood that an intersection has already been lost
        # due to floating point errors grows, so the real solution is to pick a less
        # degenerate pitch value - one that does not ride the global extrema of f.
    )
    return put(is_intersect, idx[0], edge_case)


def _add2legend(legend, lines):
    """Add lines to legend if it's not already in it."""
    for line in setdefault(lines, [lines], hasattr(lines, "__iter__")):
        label = line.get_label()
        if label not in legend:
            legend[label] = line


def _plot_intersect(ax, legend, z1, z2, k, k_transparency, klabel):
    """Plot intersects on ``ax``."""
    if k is None:
        return

    k = jnp.atleast_1d(jnp.squeeze(k))
    assert k.ndim == 1
    z1, z2 = jnp.atleast_2d(z1, z2)
    assert z1.ndim == z2.ndim >= 2
    assert k.shape[0] == z1.shape[0] == z2.shape[0]
    for p in k:
        _add2legend(
            legend,
            ax.axhline(p, color="tab:purple", alpha=k_transparency, label=klabel),
        )
    for i in range(k.size):
        _z1, _z2 = z1[i], z2[i]
        if _z1.size == _z2.size:
            mask = (_z1 - _z2) != 0.0
            _z1 = _z1[mask]
            _z2 = _z2[mask]
        _add2legend(
            legend,
            ax.scatter(
                _z1,
                jnp.full_like(_z1, k[i]),
                marker="v",
                color="tab:red",
                label=r"$z_1$",
            ),
        )
        _add2legend(
            legend,
            ax.scatter(
                _z2,
                jnp.full_like(_z2, k[i]),
                marker="^",
                color="tab:green",
                label=r"$z_2$",
            ),
        )
