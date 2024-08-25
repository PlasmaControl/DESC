from functools import partial

from orthax.chebyshev import chebroots

from desc.backend import flatnonzero, jnp, put
from desc.integrals.quad_utils import composite_linspace
from desc.utils import setdefault

# TODO: Boyd's method 𝒪(N²) instead of Chebyshev companion matrix 𝒪(N³).
#  John P. Boyd, Computing real roots of a polynomial in Chebyshev series
#  form through subdivision. https://doi.org/10.1016/j.apnum.2005.09.007.
chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")


def flatten_matrix(y):
    """Flatten batch of matrix to batch of vector."""
    return y.reshape(*y.shape[:-2], -1)


def subtract(c, k):
    """Subtract ``k`` from last axis of ``c``, obeying numpy broadcasting."""
    c_0 = c[..., 0] - k
    c = jnp.concatenate(
        [
            c_0[..., jnp.newaxis],
            jnp.broadcast_to(c[..., 1:], (*c_0.shape, c.shape[-1] - 1)),
        ],
        axis=-1,
    )
    return c


def filter_bounce_points(bp1, bp2):
    """Return only bounce points such that ``bp2-bp1`` ≠ 0."""
    mask = (bp2 - bp1) != 0.0
    return bp1[mask], bp2[mask]


def add2legend(legend, lines):
    """Add lines to legend if it's not already in it."""
    for line in setdefault(lines, [lines], hasattr(lines, "__iter__")):
        label = line.get_label()
        if label not in legend:
            legend[label] = line


def plot_intersect(ax, legend, z1, z2, k, k_transparency):
    """Plot intersects on ``ax``."""
    if k is None:
        return

    k = jnp.atleast_1d(jnp.squeeze(k))
    assert k.ndim == 1
    z1, z2 = jnp.atleast_2d(z1, z2)
    assert z1.ndim == z2.ndim == 2
    assert k.shape[0] == z1.shape[0] == z2.shape[0]
    for p in k:
        add2legend(
            legend,
            ax.axhline(p, color="tab:purple", alpha=k_transparency),
        )
    for i in range(k.size):
        _z1, _z2 = z1[i], z2[i]
        if _z1.size == _z2.size:
            _z1, _z2 = filter_bounce_points(_z1, _z2)
        add2legend(
            legend,
            ax.scatter(_z1, jnp.full(z1.shape[1], k[i]), marker="v", color="tab:red"),
        )
        add2legend(
            legend,
            ax.scatter(_z2, jnp.full(z2.shape[1], k[i]), marker="^", color="tab:green"),
        )


@partial(jnp.vectorize, signature="(m),(m)->(m)")
def fix_inversion(is_intersect, df_dy_sign):
    """Disqualify first intersect except under an edge case.

    The pairs ``y1`` and ``y2`` are boundaries of an integral only if
    ``y1 <= y2``. It is required that the first intersect satisfies
    non-positive derivative. Now, because
    ``df_dy_sign[...,k]<=0`` implies ``df_dy_sign[...,k+1]>=0``
    by continuity, there can be at most one inversion, and if it exists,
    the inversion must be at the first pair. To correct the inversion,
    it suffices to disqualify the first intersect as a right boundary,
    except under an edge case.

    Parameters
    ----------
    is_intersect : jnp.ndarray
        Boolean array into ``y`` indicating whether element is an intersect.
    df_dy_sign : jnp.ndarray
        Shape ``is_intersect.shape``.
        Sign of ∂f/∂y (x, yᵢ).

    Returns
    -------
    is_intersect : jnp.ndarray

    """
    # idx of first two intersects
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
        # degenerate pitch value - one that does not ride the global extrema of |B|.
    )
    return put(is_intersect, idx[0], edge_case)


def get_pitch(min_B, max_B, num, relative_shift=1e-6):
    """Return uniformly spaced pitch values between ``1/max_B`` and ``1/min_B``.

    Parameters
    ----------
    min_B : jnp.ndarray
        Minimum |B| value.
    max_B : jnp.ndarray
        Maximum |B| value.
    num : int
        Number of values, not including endpoints.
    relative_shift : float
        Relative amount to shift maxima down and minima up to avoid floating point
        errors in downstream routines.

    Returns
    -------
    pitch : jnp.ndarray
        Shape (num + 2, *min_B.shape).

    """
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift values slightly to resolve this issue.
    min_B = (1 + relative_shift) * min_B
    max_B = (1 - relative_shift) * max_B
    pitch = composite_linspace(1 / jnp.stack([max_B, min_B]), num)
    assert pitch.shape == (num + 2, *min_B.shape)
    return pitch


# TODO: Generalize this beyond ζ = ϕ or just map to Clebsch with ϕ
def get_alpha(alpha_0, iota, num_transit, period):
    """Get sequence of poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) of field line.

    Parameters
    ----------
    alpha_0 : float
        Starting field line poloidal label.
    iota : jnp.ndarray
        Shape (iota.size, ).
        Rotational transform normalized by 2π.
    num_transit : float
        Number of ``period``s to follow field line.
    period : float
        Toroidal period after which to update label.

    Returns
    -------
    alpha : jnp.ndarray
        Shape (iota.size, num_transit).
        Sequence of poloidal coordinates A = (α₀, α₁, …, αₘ₋₁) that specify field line.

    """
    # Δϕ (∂α/∂ϕ) = Δϕ ι̅ = Δϕ ι/2π = Δϕ data["iota"]
    alpha = alpha_0 + period * iota[:, jnp.newaxis] * jnp.arange(num_transit)
    return alpha
