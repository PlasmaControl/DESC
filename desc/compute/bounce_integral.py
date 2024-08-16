"""Methods for computing bounce integrals."""

from functools import partial

import numpy as np
from interpax import CubicHermiteSpline, PPoly, interp1d
from jax.nn import softmax
from matplotlib import pyplot as plt
from numpy.polynomial.legendre import leggauss

from desc.backend import flatnonzero, imap, jnp, put
from desc.compute._interp_utils import poly_root, polyder_vec, polyval_vec
from desc.compute._quad_utils import (
    automorphism_sin,
    bijection_from_disc,
    grad_automorphism_sin,
    grad_bijection_from_disc,
)
from desc.compute.utils import take_mask
from desc.utils import errorif, setdefault, warnif


# use for debugging and testing
def _filter_not_nan(a, check=False):
    """Filter out nan from ``a`` while asserting nan is padded at right."""
    is_nan = np.isnan(a)
    if check:
        assert np.array_equal(is_nan, np.sort(is_nan, axis=-1))
    return a[~is_nan]


# use for debugging and testing
def _filter_nonzero_measure(bp1, bp2):
    """Return only bounce points such that |bp2 - bp1| > 0."""
    mask = (bp2 - bp1) != 0
    return bp1[mask], bp2[mask]


def plot_field_line(
    B,
    pitch=None,
    bp1=np.array([]),
    bp2=np.array([]),
    start=None,
    stop=None,
    num=1000,
    title=r"Computed bounce points for $\vert B \vert$ and pitch $\lambda$",
    title_id=None,
    include_knots=True,
    alpha_knot=0.1,
    alpha_pitch=0.3,
    show=True,
):
    """Plot the field line given spline of |B|.

    Parameters
    ----------
    B : PPoly
        Spline of |B| over given field line.
    pitch : np.ndarray
        λ value.
    bp1 : np.ndarray
        Bounce points with (∂|B|/∂ζ)|ρ,α <= 0.
    bp2 : np.ndarray
        Bounce points with (∂|B|/∂ζ)|ρ,α >= 0.
    start : float
        Minimum ζ on plot.
    stop : float
        Maximum ζ on plot.
    num : int
        Number of ζ points to plot. Pick a big number.
    title : str
        Plot title.
    title_id : str
        Identifier string to append to plot title.
    include_knots : bool
        Whether to plot vertical lines at the knots.
    alpha_knot : float
        Transparency of knot lines.
    alpha_pitch : float
        Transparency of pitch lines.
    show : bool
        Whether to show the plot. Default is true.

    Returns
    -------
    fig, ax : matplotlib figure and axes.

    """
    legend = {}

    def add(lines):
        if not hasattr(lines, "__iter__"):
            lines = [lines]
        for line in lines:
            label = line.get_label()
            if label not in legend:
                legend[label] = line

    fig, ax = plt.subplots()
    if include_knots:
        for knot in B.x:
            add(ax.axvline(x=knot, color="tab:blue", alpha=alpha_knot, label="knot"))
    z = np.linspace(
        start=setdefault(start, B.x[0]),
        stop=setdefault(stop, B.x[-1]),
        num=num,
    )
    add(ax.plot(z, B(z), label=r"$\vert B \vert (\zeta)$"))

    if pitch is not None:
        b = 1 / np.atleast_1d(pitch)
        for val in b:
            add(
                ax.axhline(
                    val, color="tab:purple", alpha=alpha_pitch, label=r"$1 / \lambda$"
                )
            )
        bp1, bp2 = np.atleast_2d(bp1, bp2)
        for i in range(bp1.shape[0]):
            if bp1.shape == bp2.shape:
                bp1_i, bp2_i = _filter_nonzero_measure(bp1[i], bp2[i])
            else:
                bp1_i, bp2_i = bp1[i], bp2[i]
            bp1_i, bp2_i = map(_filter_not_nan, (bp1_i, bp2_i))
            add(
                ax.scatter(
                    bp1_i,
                    np.full_like(bp1_i, b[i]),
                    marker="v",
                    color="tab:red",
                    label="bp1",
                )
            )
            add(
                ax.scatter(
                    bp2_i,
                    np.full_like(bp2_i, b[i]),
                    marker="^",
                    color="tab:green",
                    label="bp2",
                )
            )

    ax.set_xlabel(r"Field line $\zeta$")
    ax.set_ylabel(r"$\vert B \vert \sim 1 / \lambda$")
    ax.legend(legend.values(), legend.keys(), loc="lower right")
    if title_id is not None:
        title = f"{title}. id = {title_id}."
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
        plt.close()
    return fig, ax


def _check_bounce_points(bp1, bp2, sentinel, pitch, knots, B_c, plot, **kwargs):
    """Check that bounce points are computed correctly."""
    bp1 = jnp.where(bp1 > sentinel, bp1, jnp.nan)
    bp2 = jnp.where(bp2 > sentinel, bp2, jnp.nan)

    eps = jnp.finfo(jnp.array(1.0).dtype).eps * 10
    P, S = bp1.shape[:-1]
    msg_1 = "Bounce points have an inversion."
    err_1 = jnp.any(bp1 > bp2, axis=-1)
    msg_2 = "Discontinuity detected."
    err_2 = jnp.any(bp1[..., 1:] < bp2[..., :-1], axis=-1)

    for s in range(S):
        B = PPoly(B_c[:, s], knots)
        for p in range(P):
            B_mid = B((bp1[p, s] + bp2[p, s]) / 2)
            err_3 = jnp.any(B_mid > 1 / pitch[p, s] + eps)
            if err_1[p, s] or err_2[p, s] or err_3:
                bp1_p = _filter_not_nan(bp1[p, s], check=True)
                bp2_p = _filter_not_nan(bp2[p, s], check=True)
                B_mid = _filter_not_nan(B_mid, check=True)
                if plot:
                    plot_field_line(
                        B, pitch[p, s], bp1_p, bp2_p, title_id=f"{p},{s}", **kwargs
                    )
                print("bp1:", bp1_p)
                print("bp2:", bp2_p)
                assert not err_1[p, s], msg_1
                assert not err_2[p, s], msg_2
                msg_3 = (
                    f"Detected B midpoint = {B_mid}>{1 / pitch[p, s] + eps} = 1/pitch. "
                    "You need to use more knots or, if that is infeasible, switch to a "
                    "monotonic spline method.\n"
                )
                assert not err_3, msg_3
        if plot:
            plot_field_line(
                B, pitch[:, s], bp1[:, s], bp2[:, s], title_id=str(s), **kwargs
            )


def _check_shape(knots, B_c, B_z_ra_c, pitch=None):
    """Ensure inputs have compatible shape, and return them with full dimension.

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line-following ζ coordinates of spline knots.

    Returns
    -------
    B_c : jnp.ndarray
        Shape (B_c.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
    B_z_ra_c : jnp.ndarray
        Shape (B_c.shape[0] - 1, *B_c.shape[1:]).
        Polynomial coefficients of the spline of (∂|B|/∂ζ)|ρ,α in local power basis.
    pitch : jnp.ndarray
        Shape (P, S).
        λ values to evaluate the bounce integral at each field line.

    """
    errorif(knots.ndim != 1, msg=f"knots should be 1d; got shape {knots.shape}.")
    if B_c.ndim == 2 and B_z_ra_c.ndim == 2:
        # Add axis which enumerates field lines.
        B_c = B_c[:, jnp.newaxis]
        B_z_ra_c = B_z_ra_c[:, jnp.newaxis]
    msg = (
        "Invalid shape for spline arrays. "
        f"B_c.shape={B_c.shape}. B_z_ra_c.shape={B_z_ra_c.shape}."
    )
    errorif(not (B_c.ndim == B_z_ra_c.ndim == 3), msg=msg)
    errorif(B_c.shape[0] - 1 != B_z_ra_c.shape[0], msg=msg)
    errorif(B_c.shape[1:] != B_z_ra_c.shape[1:], msg=msg)
    errorif(
        B_c.shape[-1] != knots.size - 1,
        msg=(
            "Last axis does not enumerate polynomials of spline. "
            f"B_c.shape={B_c.shape}. knots.shape={knots.shape}."
        ),
    )
    if pitch is not None:
        pitch = jnp.atleast_2d(pitch)
        msg = f"Invalid shape {pitch.shape} for pitch angles."
        errorif(pitch.ndim != 2, msg=msg)
        errorif(pitch.shape[-1] != 1 and pitch.shape[-1] != B_c.shape[1], msg=msg)
    return B_c, B_z_ra_c, pitch


@partial(jnp.vectorize, signature="(m),(m)->(m)")
def _fix_inversion(is_intersect, B_z_ra):
    # idx of first two intersects
    idx = flatnonzero(is_intersect, size=2, fill_value=-1)
    edge_case = (
        (B_z_ra[idx[0]] == 0)
        & (B_z_ra[idx[1]] < 0)
        & is_intersect[idx[0]]
        & is_intersect[idx[1]]
        # In theory, we need to keep propagating this edge case,
        # e.g. (B_z_ra[..., 1] < 0) | ((B_z_ra[..., 1] == 0) & (B_z_ra[..., 2] < 0)...).
        # At each step, the likelihood that an intersection has already been lost
        # due to floating point errors grows, so the real solution is to pick a less
        # degenerate pitch value - one that does not ride the global extrema of |B|.
    )
    # The pairs bp1[i, j, k] and bp2[i, j, k] are boundaries of an integral only
    # if bp1[i, j, k] <= bp2[i, j, k]. For correctness of the algorithm, it is
    # required that the first intersect satisfies non-positive derivative. Now,
    # because B_z_ra[i, j, k] <= 0 implies B_z_ra[i, j, k + 1] >= 0 by continuity,
    # there can be at most one inversion, and if it exists, the inversion must be
    # at the first pair. To correct the inversion, it suffices to disqualify the
    # first intersect as a right boundary, except under the above edge case.
    return put(is_intersect, idx[0], edge_case)


def bounce_points(
    pitch, knots, B_c, B_z_ra_c, num_well=None, check=False, plot=True, **kwargs
):
    """Compute the bounce points given spline of |B| and pitch λ.

    Parameters
    ----------
    pitch : jnp.ndarray
        Shape (P, S).
        λ values to evaluate the bounce integral at each field line. λ(ρ,α) is
        specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
        interpreted as the index into the last axis that corresponds to that field
        line. If two-dimensional, the first axis is the batch axis.
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line-following ζ coordinates of spline knots. Must be strictly increasing.
    B_c : jnp.ndarray
        Shape (B_c.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    B_z_ra_c : jnp.ndarray
        Shape (B_c.shape[0] - 1, *B_c.shape[1:]).
        Polynomial coefficients of the spline of (∂|B|/∂ζ)|ρ,α in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    num_well : int or None
        If not specified, then all bounce points are returned in an array whose
        last axis has size ``(knots.size - 1) * (B_c.shape[0] - 1)``. If there
        were less than that many wells detected along a field line, then the last
        axis of the returned arrays, which enumerates bounce points for a particular
        field line and pitch, is padded with zero.

        Specify to return the first ``num_well`` pairs of bounce points for each
        pitch along each field line. This is useful if ``num_well`` tightly
        bounds the actual number of wells. To obtain a good choice for ``num_well``,
        plot the field line with all the bounce points identified by calling this
        function with ``check=True``. As a reference, there are typically <= 5 wells
        per toroidal transit.
    check : bool
        Flag for debugging.
    plot : bool
        Whether to plot some things if check is true. Default is true.

    Returns
    -------
    bp1, bp2 : (jnp.ndarray, jnp.ndarray)
        Shape (P, S, num_well).
        The field line-following coordinates of bounce points for a given pitch along
        a field line. The pairs ``bp1`` and ``bp2`` form left and right integration
        boundaries, respectively, for the bounce integrals.

    """
    B_c, B_z_ra_c, pitch = _check_shape(knots, B_c, B_z_ra_c, pitch)
    P, S, N, degree = pitch.shape[0], B_c.shape[1], knots.size - 1, B_c.shape[0] - 1
    # Intersection points in local power basis.
    intersect = poly_root(
        c=B_c,
        k=(1 / pitch)[..., jnp.newaxis],
        a_min=jnp.array([0.0]),
        a_max=jnp.diff(knots),
        sort=True,
        sentinel=-1,
        distinct=True,
    )
    assert intersect.shape == (P, S, N, degree)

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    B_z_ra = polyval_vec(x=intersect, c=B_z_ra_c[..., jnp.newaxis]).reshape(P, S, -1)
    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = intersect.reshape(P, S, -1) >= 0
    # Following discussion on page 3 and 5 of https://doi.org/10.1063/1.873749,
    # we ignore the bounce points of particles only assigned to a class that are
    # trapped outside this snapshot of the field line.
    is_bp1 = (B_z_ra <= 0) & is_intersect
    is_bp2 = (B_z_ra >= 0) & _fix_inversion(is_intersect, B_z_ra)

    # Transform out of local power basis expansion.
    intersect = (intersect + knots[:-1, jnp.newaxis]).reshape(P, S, -1)
    # New versions of jax only like static sentinels.
    sentinel = -10000000.0  # knots[0] - 1
    bp1 = take_mask(intersect, is_bp1, size=num_well, fill_value=sentinel)
    bp2 = take_mask(intersect, is_bp2, size=num_well, fill_value=sentinel)

    if check:
        _check_bounce_points(bp1, bp2, sentinel, pitch, knots, B_c, plot, **kwargs)

    mask = (bp1 > sentinel) & (bp2 > sentinel)
    # Set outside mask to same value so integration is over set of measure zero.
    bp1 = jnp.where(mask, bp1, 0)
    bp2 = jnp.where(mask, bp2, 0)
    return bp1, bp2


def _composite_linspace(x, num):
    """Returns linearly spaced points between every pair of points ``x``.

    Parameters
    ----------
    x : jnp.ndarray
        First axis has values to return linearly spaced values between. The remaining
        axes are batch axes. Assumes input is sorted along first axis.
    num : int
        Number of points between every pair of points in ``x``.

    Returns
    -------
    pts : jnp.ndarray
        Shape ((x.shape[0] - 1) * num + x.shape[0], *x.shape[1:]).
        Linearly spaced points between ``x``.

    """
    x = jnp.atleast_1d(x)
    pts = jnp.linspace(x[:-1], x[1:], num + 1, endpoint=False)
    pts = jnp.swapaxes(pts, 0, 1).reshape(-1, *x.shape[1:])
    pts = jnp.append(pts, x[jnp.newaxis, -1], axis=0)
    assert pts.shape == ((x.shape[0] - 1) * num + x.shape[0], *x.shape[1:])
    return pts


def get_pitch(min_B, max_B, num, relative_shift=1e-6):
    """Return uniformly spaced pitch values between 1 / max B and 1 / min B.

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
    pitch = _composite_linspace(1 / jnp.stack([max_B, min_B]), num)
    assert pitch.shape == (num + 2, *pitch.shape[1:])
    return pitch


def _get_extrema(knots, B_c, B_z_ra_c, sentinel=jnp.nan):
    """Return extrema of |B| along field line. Sort order is arbitrary.

    Parameters
    ----------
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line-following ζ coordinates of spline knots. Must be strictly increasing.
    B_c : jnp.ndarray
        Shape (B_c.shape[0], S, knots.size - 1).
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    B_z_ra_c : jnp.ndarray
        Shape (B_c.shape[0] - 1, *B_c.shape[1:]).
        Polynomial coefficients of the spline of (∂|B|/∂ζ)|ρ,α in local power basis.
        First axis enumerates the coefficients of power series. Second axis
        enumerates the splines along the field lines. Last axis enumerates the
        polynomials that compose the spline along a particular field line.
    sentinel : float
        Value with which to pad array to return fixed shape.

    Returns
    -------
    extrema, B_extrema : jnp.ndarray
        Shape (S, N * (degree - 1)).

    """
    B_c, B_z_ra_c, _ = _check_shape(knots, B_c, B_z_ra_c)
    S, N, degree = B_c.shape[1], knots.size - 1, B_c.shape[0] - 1
    extrema = poly_root(
        c=B_z_ra_c, a_min=jnp.array([0.0]), a_max=jnp.diff(knots), sentinel=sentinel
    )
    assert extrema.shape == (S, N, degree - 1)
    B_extrema = polyval_vec(x=extrema, c=B_c[..., jnp.newaxis]).reshape(S, -1)
    # Transform out of local power basis expansion.
    extrema = (extrema + knots[:-1, jnp.newaxis]).reshape(S, -1)
    return extrema, B_extrema


def _plot(Z, V, title_id=""):
    """Plot V[λ, (ρ, α), (ζ₁, ζ₂)](Z)."""
    for p in range(Z.shape[0]):
        for s in range(Z.shape[1]):
            marked = jnp.nonzero(jnp.any(Z != 0, axis=-1))[0]
            if marked.size == 0:
                continue
            fig, ax = plt.subplots()
            ax.set_xlabel(r"Field line $\zeta$")
            ax.set_ylabel(title_id)
            ax.set_title(
                f"Interpolation of {title_id} to quadrature points. Index {p},{s}."
            )
            for i in marked:
                ax.plot(Z[p, s, i], V[p, s, i], marker="o")
            fig.text(
                0.01,
                0.01,
                f"Each color specifies the set of points and values (ζ, {title_id}(ζ)) "
                "used to evaluate an integral.",
            )
            plt.tight_layout()
            plt.show()


def _check_interp(Z, f, b_sup_z, B, B_z_ra, result, plot):
    """Check for floating point errors.

    Parameters
    ----------
    Z : jnp.ndarray
        Quadrature points at field line-following ζ coordinates.
    f : list of jnp.ndarray
        Arguments to the integrand interpolated to Z.
    b_sup_z : jnp.ndarray
        Contravariant field-line following toroidal component of magnetic field,
        interpolated to Z.
    B : jnp.ndarray
        Norm of magnetic field, interpolated to Z.
    B_z_ra : jnp.ndarray
        Norm of magnetic field, derivative with respect to field-line following
        coordinate, interpolated to Z.
    result : jnp.ndarray
        Output of ``_interpolate_and_integrate``.
    plot : bool
        Whether to plot stuff.

    """
    assert jnp.isfinite(Z).all(), "NaN interpolation point."
    # Integrals that we should be computing.
    marked = jnp.any(Z != 0, axis=-1)
    goal = jnp.sum(marked)

    msg = "Interpolation failed."
    assert jnp.isfinite(B_z_ra).all(), msg
    assert goal == jnp.sum(marked & jnp.isfinite(jnp.sum(b_sup_z, axis=-1))), msg
    assert goal == jnp.sum(marked & jnp.isfinite(jnp.sum(B, axis=-1))), msg
    for f_i in f:
        assert goal == jnp.sum(marked & jnp.isfinite(jnp.sum(f_i, axis=-1))), msg

    msg = "|B| has vanished, violating the hairy ball theorem."
    assert not jnp.isclose(B, 0).any(), msg
    assert not jnp.isclose(b_sup_z, 0).any(), msg

    # Number of those integrals that were computed.
    actual = jnp.sum(marked & jnp.isfinite(result))
    assert goal == actual, (
        f"Lost {goal - actual} integrals from NaN generation in the integrand. This "
        "can be caused by floating point error or a poor choice of quadrature nodes."
    )
    if plot:
        _plot(Z, B, title_id=r"$\vert B \vert$")
        _plot(Z, b_sup_z, title_id=r"$ (B/\vert B \vert) \cdot e^{\zeta}$")


_interp1d_vec = jnp.vectorize(
    interp1d, signature="(m),(n),(n)->(m)", excluded={"method"}
)


@partial(jnp.vectorize, signature="(m),(n),(n),(n)->(m)")
def _interp1d_vec_with_df(xq, x, f, fx):
    return interp1d(xq, x, f, method="cubic", fx=fx)


def _interpolate_and_integrate(
    Q,
    w,
    integrand,
    f,
    B_sup_z,
    B_sup_z_ra,
    B,
    B_z_ra,
    pitch,
    knots,
    method,
    check=False,
    plot=False,
):
    """Interpolate given functions to points ``Q`` and perform quadrature.

    Parameters
    ----------
    Q : jnp.ndarray
        Shape (P, S, Q.shape[2], w.size).
        Quadrature points at field line-following ζ coordinates.

    Returns
    -------
    result : jnp.ndarray
        Shape Q.shape[:-1].
        Quadrature for every pitch along every field line.

    """
    assert pitch.ndim == 2
    assert w.ndim == knots.ndim == 1
    assert 3 <= Q.ndim <= 4 and Q.shape[:2] == (pitch.shape[0], B.shape[0])
    assert Q.shape[-1] == w.size
    assert knots.size == B.shape[-1]
    assert B_sup_z.shape == B_sup_z_ra.shape == B.shape == B_z_ra.shape

    pitch = jnp.expand_dims(pitch, axis=(2, 3) if (Q.ndim == 4) else 2)
    shape = Q.shape
    Q = Q.reshape(Q.shape[0], Q.shape[1], -1)
    b_sup_z = _interp1d_vec_with_df(
        Q, knots, B_sup_z / B, B_sup_z_ra / B - B_sup_z * B_z_ra / B**2
    ).reshape(shape)
    B = _interp1d_vec_with_df(Q, knots, B, B_z_ra).reshape(shape)
    # Spline the integrand so that we can evaluate it at quadrature points without
    # expensive coordinate mappings and root finding. Spline each function separately so
    # that the singularity near the bounce points can be captured more accurately than
    # can be by any polynomial.
    f = [_interp1d_vec(Q, knots, f_i, method=method).reshape(shape) for f_i in f]
    result = jnp.dot(integrand(*f, B=B, pitch=pitch) / b_sup_z, w)

    if check:
        _check_interp(Q.reshape(shape), f, b_sup_z, B, B_z_ra, result, plot)

    return result


def _bounce_quadrature(
    bp1,
    bp2,
    x,
    w,
    integrand,
    f,
    B_sup_z,
    B_sup_z_ra,
    B,
    B_z_ra,
    pitch,
    knots,
    method="akima",
    batch=True,
    check=False,
):
    """Bounce integrate ∫ f(ℓ) dℓ.

    Parameters
    ----------
    bp1 : jnp.ndarray
        Shape (P, S, num_well).
        The field line-following ζ coordinates of bounce points for a given pitch along
        a field line. The pairs ``bp1[i,j,k]`` and ``bp2[i,j,k]`` form left and right
        integration boundaries, respectively, for the bounce integrals.
    bp2 : jnp.ndarray
        Shape (P, S, num_well).
        The field line-following ζ coordinates of bounce points for a given pitch along
        a field line. The pairs ``bp1[i,j,k]`` and ``bp2[i,j,k]`` form left and right
        integration boundaries, respectively, for the bounce integrals.
    x : jnp.ndarray
        Shape (w.size, ).
        Quadrature points in [-1, 1].
    w : jnp.ndarray
        Shape (w.size, ).
        Quadrature weights.
    integrand : callable
        The composition operator on the set of functions in ``f`` that maps the
        functions in ``f`` to the integrand f(ℓ) in ∫ f(ℓ) dℓ. It should accept the
        arrays in ``f`` as arguments as well as the additional keyword arguments:
        ``B`` and ``pitch``. A quadrature will be performed to approximate the
        bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
    f : list of jnp.ndarray
        Shape (S, knots.size) or (S * knots.size).
        Arguments to the callable ``integrand``. These should be the scalar-valued
        functions in the bounce integrand evaluated on the DESC grid.
    B_sup_z : jnp.ndarray
        Shape (S, knots.size) or (S * knots.size).
        Contravariant field-line following toroidal component of magnetic field.
    B_sup_z_ra : jnp.ndarray
        Shape (S, knots.size) or (S * knots.size).
        Contravariant field-line following toroidal component of magnetic field,
        derivative with respect to field-line following coordinate.
    B : jnp.ndarray
        Shape (S, knots.size).
        Norm of magnetic field.
    B_z_ra : jnp.ndarray
        Shape (S, knots.size).
        Norm of magnetic field, derivative with respect to field-line following
        coordinate.
    pitch : jnp.ndarray
        Shape (P, S).
        λ values to evaluate the bounce integral at each field line.
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line following coordinate values where ``B_sup_z``, ``B``, ``B_z_ra``, and
        those in ``f`` supplied to the returned method were evaluated. Must be strictly
        increasing.
    method : str
        Method of interpolation for functions contained in ``f``.
        See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
        Default is akima spline.
    batch : bool
        Whether to perform computation in a batched manner. Default is true.
    check : bool
        Flag for debugging.

    Returns
    -------
    result : jnp.ndarray
        Shape (P, S, bp1.shape[-1]).
        First axis enumerates pitch values. Second axis enumerates the field lines.
        Last axis enumerates the bounce integrals.

    """
    errorif(bp1.ndim != 3 or bp1.shape != bp2.shape)
    errorif(x.ndim != 1 or x.shape != w.shape)
    pitch = jnp.atleast_2d(pitch)
    S = B.shape[0]
    if not isinstance(f, (list, tuple)):
        f = [f]
    # group data by field line
    f = map(lambda f_i: f_i.reshape(-1, knots.size), f)

    # Integrate and complete the change of variable.
    if batch:
        result = _interpolate_and_integrate(
            bijection_from_disc(x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis]),
            w,
            integrand,
            f,
            B_sup_z,
            B_sup_z_ra,
            B,
            B_z_ra,
            pitch,
            knots,
            method,
            check,
            # Only developers doing debugging want to see these plots.
            plot=False,
        )
    else:
        f = list(f)

        # TODO: Use batched vmap.
        def loop(bp):
            bp1, bp2 = bp
            return None, _interpolate_and_integrate(
                bijection_from_disc(x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis]),
                w,
                integrand,
                f,
                B_sup_z,
                B_sup_z_ra,
                B,
                B_z_ra,
                pitch,
                knots,
                method,
                check=False,
                plot=False,
            )

        result = jnp.moveaxis(
            imap(loop, (jnp.moveaxis(bp1, -1, 0), jnp.moveaxis(bp2, -1, 0)))[1],
            source=0,
            destination=-1,
        )

    result = result * grad_bijection_from_disc(bp1, bp2)
    assert result.shape == (pitch.shape[0], S, bp1.shape[-1])
    return result


def required_names():
    """Return names in ``data_index`` required to compute bounce integrals."""
    return ["B^zeta", "B^zeta_z|r,a", "|B|", "|B|_z|r,a"]


def bounce_integral(
    data,
    knots,
    quad=leggauss(21),
    automorphism=(automorphism_sin, grad_automorphism_sin),
    B_ref=1.0,
    L_ref=1.0,
    check=False,
    plot=False,
    **kwargs,
):
    """Returns a method to compute bounce integrals.

    The bounce integral is defined as ∫ f(ℓ) dℓ, where
        dℓ parameterizes the distance along the field line in meters,
        λ is a constant proportional to the magnetic moment over energy,
        |B| is the norm of the magnetic field,
        f(ℓ) is the quantity to integrate along the field line,
        and the boundaries of the integral are bounce points ζ₁, ζ₂ s.t. λ|B|(ζᵢ) = 1.

    For a particle with fixed λ, bounce points are defined to be the location on the
    field line such that the particle's velocity parallel to the magnetic field is zero.
    The bounce integral is defined up to a sign. We choose the sign that corresponds to
    the particle's guiding center trajectory traveling in the direction of increasing
    field-line-following coordinate ζ.

    Notes
    -----
    The quantities in ``data`` and those in ``f`` supplied to the returned method
    must be separable into data evaluated along particular field lines
    via ``.reshape(S,knots.size)``. One way to satisfy this is to compute stuff on the
    grid returned from the method ``desc.equilibrium.coords.get_rtz_grid``. See
    ``tests.test_bounce_integral.test_bounce_integral_checks`` for example use.

    Parameters
    ----------
    data : dict of jnp.ndarray
        Data evaluated on grid.
        Shape (S * knots.size, ) or (S, knots.size).
        Should contain all names in ``required_names()``.
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line following coordinate values where arrays in ``data`` and ``f``
        supplied to the returned method were evaluated. Must be strictly
        increasing. These knots are used to compute a spline of |B| and interpolate the
        integrand. A good reference density is 100 knots per toroidal transit.
    quad : (jnp.ndarray, jnp.ndarray)
        Quadrature points xₖ and weights wₖ for the approximate evaluation of an
        integral ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ). Default is 21 points.
    automorphism : (Callable, Callable) or None
        The first callable should be an automorphism of the real interval [-1, 1].
        The second callable should be the derivative of the first. This map defines a
        change of variable for the bounce integral. The choice made for the automorphism
        can affect the performance of the quadrature method.
    B_ref : float
        Optional. Reference magnetic field strength for normalization.
    L_ref : float
        Optional. Reference length scale for normalization.
    check : bool
        Flag for debugging. Must be false for jax transformations.
    plot : bool
        Whether to plot stuff if ``check`` is true. Default is false.

    Returns
    -------
    bounce_integrate : callable
        This callable method computes the bounce integral ∫ f(ℓ) dℓ for every
        specified field line for every λ value in ``pitch``.
    spline : dict of jnp.ndarray
        knots : jnp.ndarray
            Shape (knots.size, ).
            Field line-following ζ coordinates of spline knots.
        B_c : jnp.ndarray
            Shape (4, S, knots.size - 1).
            Polynomial coefficients of the spline of |B| in local power basis.
            First axis enumerates the coefficients of power series. Second axis
            enumerates the splines along the field lines. Last axis enumerates the
            polynomials that compose the spline along a particular field line.
        B_z_ra_c : jnp.ndarray
            Shape (3, S, knots.size - 1).
            Polynomial coefficients of the spline of (∂|B|/∂ζ)|ρ,α in local power basis.
            First axis enumerates the coefficients of power series. Second axis
            enumerates the splines along the field lines. Last axis enumerates the
            polynomials that compose the spline along a particular field line.

    """
    warnif(
        check and kwargs.pop("warn", True) and jnp.any(data["B^zeta"] <= 0),
        msg="(∂ℓ/∂ζ)|ρ,a > 0 is required. Enforcing positive B^ζ.",
    )
    # Strictly increasing zeta knots enforces dζ > 0.
    # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require B^ζ = B⋅∇ζ > 0.
    # This is equivalent to changing the sign of ∇ζ (or [∂ℓ/∂ζ]|ρ,a).
    # Recall dζ = ∇ζ⋅dR, implying 1 = ∇ζ⋅(e_ζ|ρ,a). Hence, a sign change in ∇ζ
    # requires the same sign change in e_ζ|ρ,a to retain the metric identity.
    B_sup_z = jnp.abs(data["B^zeta"]).reshape(-1, knots.size) * L_ref / B_ref
    B_sup_z_ra = (
        (data["B^zeta_z|r,a"] * jnp.sign(data["B^zeta"])).reshape(-1, knots.size)
        * L_ref
        / B_ref
    )
    B = data["|B|"].reshape(-1, knots.size) / B_ref
    # This is already the correct sign.
    B_z_ra = data["|B|_z|r,a"].reshape(-1, knots.size) / B_ref

    # Compute local splines.
    B_c = CubicHermiteSpline(knots, B, B_z_ra, axis=-1, check=check).c
    B_c = jnp.moveaxis(B_c, source=1, destination=-1)
    B_z_ra_c = polyder_vec(B_c)
    degree = 3
    assert B_c.shape[0] == degree + 1
    assert B_z_ra_c.shape[0] == degree
    assert B_c.shape[-1] == B_z_ra_c.shape[-1] == knots.size - 1
    spline = {"knots": knots, "B_c": B_c, "B_z_ra_c": B_z_ra_c}

    x, w = quad
    assert x.ndim == w.ndim == 1
    if automorphism is not None:
        auto, grad_auto = automorphism
        w = w * grad_auto(x)
        # Recall affine_bijection(auto(x), ζ_b₁, ζ_b₂) = ζ.
        x = auto(x)

    def bounce_integrate(
        integrand,
        f,
        pitch,
        weight=None,
        num_well=None,
        method="akima",
        batch=True,
    ):
        """Bounce integrate ∫ f(ℓ) dℓ.

        Parameters
        ----------
        integrand : callable
            The composition operator on the set of functions in ``f`` that maps the
            functions in ``f`` to the integrand f(ℓ) in ∫ f(ℓ) dℓ. It should accept the
            arrays in ``f`` as arguments as well as the additional keyword arguments:
            ``B`` and ``pitch``. A quadrature will be performed to approximate the
            bounce integral of ``integrand(*f,B=B,pitch=pitch)``.
        f : list of jnp.ndarray
            Shape (S, knots.size) or (S * knots.size).
            Arguments to the callable ``integrand``. These should be real scalar-valued
            functions in the bounce integrand evaluated on the DESC grid.
        pitch : jnp.ndarray
            Shape (P, S).
            λ values to evaluate the bounce integral at each field line. λ(ρ,α) is
            specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        weight : jnp.ndarray
            Shape (S, knots.size) or (S * knots.size).
            If supplied, the bounce integral labeled by well j is weighted such that
            the returned value is w(j) ∫ f(ℓ) dℓ, where w(j) is ``weight``
            interpolated to the deepest point in the magnetic well.
        num_well : int or None
            If not specified, then all bounce integrals are returned in an array whose
            last axis has size ``(knots.size-1)*degree``. If there
            were less than that many wells detected along a field line, then the last
            axis of the returned array, which enumerates bounce integrals for a
            particular field line and pitch, is padded with zero.

            Specify to return the bounce integrals between the first ``num_well``
            wells for each pitch along each field line. This is useful if ``num_well``
            tightly bounds the actual number of wells. To obtain a good
            choice for ``num_well``, plot the field line with all the bounce points
            identified. This will be done automatically if the ``bounce_integral``
            function is called with ``check=True`` and ``plot=True``. As a reference,
            there are typically <= 5 wells per toroidal transit.
        method : str
            Method of interpolation for functions contained in ``f``.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
            Default is akima spline.
        batch : bool
            Whether to perform computation in a batched manner. Default is true.

        Returns
        -------
        result : jnp.ndarray
            Shape (P, S, num_well).
            First axis enumerates pitch values. Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.

        """
        bp1, bp2 = bounce_points(pitch, knots, B_c, B_z_ra_c, num_well, check, plot)
        result = _bounce_quadrature(
            bp1,
            bp2,
            x,
            w,
            integrand,
            f,
            B_sup_z,
            B_sup_z_ra,
            B,
            B_z_ra,
            pitch,
            knots,
            method,
            batch,
            check,
        )
        if weight is not None:
            result *= _interp_to_argmin_B_soft(
                weight, bp1, bp2, knots, B_c, B_z_ra_c, method
            )
        assert result.shape[-1] == setdefault(num_well, (knots.size - 1) * degree)
        return result

    return bounce_integrate, spline


def _interp_to_argmin_B_soft(f, bp1, bp2, knots, B_c, B_z_ra_c, method, beta=-50):
    """Compute ``f`` at deepest point in the magnetic well.

    Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A = argmin_E |B|(ζ). Returns mean_A f(ζ).

    Parameters
    ----------
    beta : float
        More negative gives exponentially better approximation at the
        expense of noisier gradients.

    """
    ext, B = _get_extrema(knots, B_c, B_z_ra_c, sentinel=0)
    assert ext.shape[0] == B.shape[0] == bp1.shape[1] == bp2.shape[1]
    argmin = softmax(
        beta
        * jnp.where(
            (bp1[..., jnp.newaxis] < ext[:, jnp.newaxis])
            & (ext[:, jnp.newaxis] < bp2[..., jnp.newaxis]),
            jnp.expand_dims(B / jnp.mean(B, axis=-1, keepdims=True), axis=1),
            1e2,  # >> max(|B|) / mean(|B|)
        ),
        axis=-1,
    )
    f = jnp.linalg.vecdot(
        argmin,
        _interp1d_vec(ext, knots, f.reshape(-1, knots.size), method=method)[
            :, jnp.newaxis
        ],
    )
    assert f.shape == bp1.shape == bp2.shape
    return f


# Less efficient than above if P >> 1.
def _interp_to_argmin_B_hard(f, bp1, bp2, knots, B_c, B_z_ra_c, method):
    """Compute ``f`` at deepest point in the magnetic well.

    Let E = {ζ ∣ ζ₁ < ζ < ζ₂} and A ∈ argmin_E |B|(ζ). Returns f(A).
    """
    ext, B = _get_extrema(knots, B_c, B_z_ra_c, sentinel=0)
    assert ext.shape[0] == B.shape[0] == bp1.shape[1] == bp2.shape[1]
    argmin = jnp.argmin(
        jnp.where(
            (bp1[..., jnp.newaxis] < ext[:, jnp.newaxis])
            & (ext[:, jnp.newaxis] < bp2[..., jnp.newaxis]),
            B[:, jnp.newaxis],
            1e2 + jnp.max(B),
        ),
        axis=-1,
    )
    A = jnp.take_along_axis(ext[jnp.newaxis], argmin, axis=-1)
    f = _interp1d_vec(A, knots, f.reshape(-1, knots.size), method=method)
    assert f.shape == bp1.shape == bp2.shape
    return f
