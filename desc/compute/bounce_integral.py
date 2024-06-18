"""Methods for computing bounce integrals."""

from functools import partial

from interpax import CubicHermiteSpline, PchipInterpolator, PPoly, interp1d
from matplotlib import pyplot as plt
from orthax.legendre import leggauss

from desc.backend import flatnonzero, imap, jnp, put_along_axis, take
from desc.compute.utils import safediv
from desc.utils import errorif, warnif


@partial(jnp.vectorize, signature="(m),(m)->(n)", excluded={2, 3})
def take_mask(a, mask, size=None, fill_value=None):
    """JIT compilable method to return ``a[mask][:size]`` padded by ``fill_value``.

    Parameters
    ----------
    a : jnp.ndarray
        The source array.
    mask : jnp.ndarray
        Boolean mask to index into ``a``. Should have same shape as ``a``.
    size : int
        Elements of ``a`` at the first size True indices of ``mask`` will be returned.
        If there are fewer elements than size indicates, the returned array will be
        padded with fill_value. Defaults to ``mask.size``.
    fill_value : Any
        When there are fewer than the indicated number of elements, the remaining
        elements will be filled with ``fill_value``. Defaults to NaN for inexact types,
        the largest negative value for signed types, the largest positive value for
        unsigned types, and True for booleans.

    Returns
    -------
    result : jnp.ndarray
        Shape (size, ).

    """
    assert a.shape == mask.shape
    idx = flatnonzero(
        mask, size=mask.size if size is None else size, fill_value=mask.size
    )
    return take(
        a,
        idx,
        mode="fill",
        fill_value=fill_value,
        unique_indices=True,
        indices_are_sorted=True,
    )


# only use for debugging
def _filter_not_nan(a):
    """Filter out nan from ``a`` while asserting nan is padded at right."""
    is_nan = jnp.isnan(a)
    assert jnp.array_equal(is_nan, jnp.sort(is_nan, axis=-1))
    return a[~is_nan]


def _filter_real(a, a_min=-jnp.inf, a_max=jnp.inf):
    """Keep real values inside [``a_min``, ``a_max``] and set others to nan.

    Parameters
    ----------
    a : jnp.ndarray
    a_min, a_max : jnp.ndarray or float, jnp.ndarray or float
        Minimum and maximum value to keep real values between.
        Should broadcast with ``a``.

    Returns
    -------
    result : jnp.ndarray
        The real values of ``a`` in [``a_min``, ``a_max``]; others set to nan.

    """
    if a_min is None:
        a_min = -jnp.inf
    if a_max is None:
        a_max = jnp.inf
    return jnp.where(
        jnp.isclose(jnp.imag(a), 0) & (a_min <= a) & (a <= a_max),
        jnp.real(a),
        jnp.nan,
    )


def _nan_concat(r, num=1):
    # Concat nan num times to r on last axis.
    nan = jnp.broadcast_to(jnp.nan, (*r.shape[:-1], num))
    return jnp.concatenate([r, nan], axis=-1)


def _root_linear(a, b, distinct=False):
    """Return r such that a r + b = 0."""
    return safediv(-b, a, fill=jnp.where(jnp.isclose(b, 0), 0, jnp.nan))


def _root_quadratic(a, b, c, distinct=False):
    """Return r such that a r² + b r + c = 0, assuming real coefficients."""
    # numerical.recipes/book.html, page 227
    discriminant = b**2 - 4 * a * c
    q = -0.5 * (b + jnp.sign(b) * jnp.sqrt(discriminant))
    r1 = safediv(q, a, _root_linear(b, c))
    # more robust to remove repeated roots with discriminant
    r2 = jnp.where(
        distinct & jnp.isclose(discriminant, 0), jnp.nan, safediv(c, q, jnp.nan)
    )
    return jnp.stack([r1, r2], axis=-1)


def _root_cubic(a, b, c, d, distinct=False):
    """Return r such that a r³ + b r² + c r + d = 0, assuming real coefficients."""
    # numerical.recipes/book.html, page 228

    def irreducible(Q, R, b):
        # Three irrational real roots.
        theta = jnp.arccos(R / jnp.sqrt(Q**3))
        j = -2 * jnp.sqrt(Q)
        r1 = j * jnp.cos(theta / 3) - b / 3
        r2 = j * jnp.cos((theta + 2 * jnp.pi) / 3) - b / 3
        r3 = j * jnp.cos((theta - 2 * jnp.pi) / 3) - b / 3
        return jnp.stack([r1, r2, r3], axis=-1)

    def reducible(Q, R, b):
        # One real and two complex roots.
        A = -jnp.sign(R) * (jnp.abs(R) + jnp.sqrt(R**2 - Q**3)) ** (1 / 3)
        B = safediv(Q, A)
        r1 = (A + B) - b / 3
        return _nan_concat(r1[..., jnp.newaxis], 2)

    def root(b, c, d):
        b = safediv(b, a)
        c = safediv(c, a)
        d = safediv(d, a)
        Q = (b**2 - 3 * c) / 9
        R = (2 * b**3 - 9 * b * c + 27 * d) / 54
        return jnp.where(
            jnp.expand_dims(R**2 < Q**3, axis=-1),
            irreducible(Q, R, b),
            reducible(Q, R, b),
        )

    return jnp.where(
        jnp.isclose(a, 0)[..., jnp.newaxis],
        _nan_concat(_root_quadratic(b, c, d, distinct)),
        root(b, c, d),
    )


_roots = jnp.vectorize(partial(jnp.roots, strip_zeros=False), signature="(m)->(n)")


def _poly_root(
    c, k=0, a_min=None, a_max=None, sort=False, distinct=False, poly_is_real=True
):
    """Roots of polynomial with given coefficients.

    Parameters
    ----------
    c : jnp.ndarray
        First axis should store coefficients of a polynomial. For a polynomial given by
        ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0]-1``, coefficient cᵢ should be stored at
        ``c[n-i]``.
    k : Array
        Specify to find solutions to ∑ᵢⁿ cᵢ xⁱ = ``k``. Should broadcast with arrays of
        shape c.shape[1:].
    a_min, a_max : jnp.ndarray, jnp.ndarray
        Minimum and maximum value to return roots between. If specified only real roots
        are returned. If None, returns all complex roots. Should broadcast with arrays
        of shape c.shape[1:].
    sort : bool
        Whether to sort the roots.
    distinct : bool
        Whether to only return the distinct roots. If true, when the multiplicity is
        greater than one, the repeated roots are set to nan.
    poly_is_real : bool
        Whether the coefficients ``c`` and ``k`` are real. Default is true.

    Returns
    -------
    r : jnp.ndarray
        Shape (..., c.shape[1:], c.shape[0] - 1).
        The roots of the polynomial, iterated over the last axis.

    """
    get_only_real_roots = not (a_min is None and a_max is None)
    func = {2: _root_linear, 3: _root_quadratic, 4: _root_cubic}
    if c.shape[0] in func and poly_is_real and get_only_real_roots:
        # Compute from analytic formula to avoid the issue of complex roots with small
        # imaginary parts.
        r = func[c.shape[0]](*c[:-1], c[-1] - k, distinct)
        distinct = distinct and c.shape[0] > 3
    else:
        # Compute from eigenvalues of polynomial companion matrix.
        c_n = c[-1] - k
        c = [jnp.broadcast_to(c_i, c_n.shape) for c_i in c[:-1]]
        c.append(c_n)
        c = jnp.stack(c, axis=-1)
        r = _roots(c)
    if get_only_real_roots:
        if a_min is not None:
            a_min = a_min[..., jnp.newaxis]
        if a_max is not None:
            a_max = a_max[..., jnp.newaxis]
        r = _filter_real(r, a_min, a_max)

    if sort or distinct:
        r = jnp.sort(r, axis=-1)
    if distinct:
        # Atol needs to be low enough that distinct roots which are close do not
        # get removed, otherwise algorithms that rely on continuity of the spline
        # such as bounce_points() will fail. The current atol was chosen so that
        # test_bounce_points() passes.
        mask = jnp.isclose(jnp.diff(r, axis=-1, prepend=jnp.nan), 0, atol=1e-15)
        r = jnp.where(mask, jnp.nan, r)
    return r


def _poly_der(c):
    """Coefficients for the derivatives of the given set of polynomials.

    Parameters
    ----------
    c : jnp.ndarray
        First axis should store coefficients of a polynomial. For a polynomial given by
        ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0]-1``, coefficient cᵢ should be stored at
        ``c[n-i]``.

    Returns
    -------
    poly : jnp.ndarray
        Coefficients of polynomial derivative, ignoring the arbitrary constant. That is,
        ``poly[i]`` stores the coefficient of the monomial xⁿ⁻ⁱ⁻¹,  where n is
        ``c.shape[0]-1``.

    """
    poly = (c[:-1].T * jnp.arange(c.shape[0] - 1, 0, -1)).T
    return poly


def _poly_val(x, c):
    """Evaluate the set of polynomials ``c`` at the points ``x``.

    Note this function is not the same as ``np.polynomial.polynomial.polyval(x,c)``.

    Parameters
    ----------
    x : jnp.ndarray
        Coordinates at which to evaluate the set of polynomials.
    c : jnp.ndarray
        First axis should store coefficients of a polynomial. For a polynomial given by
        ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0]-1``, coefficient cᵢ should be stored at
        ``c[n-i]``.

    Returns
    -------
    val : jnp.ndarray
        Polynomial with given coefficients evaluated at given points.

    Examples
    --------
    .. code-block:: python

        val = _poly_val(x, c)
        if val.ndim != max(x.ndim, c.ndim - 1):
            raise ValueError(f"Incompatible shapes {x.shape} and {c.shape}.")
        for index in np.ndindex(c.shape[1:]):
            idx = (..., *index)
            np.testing.assert_allclose(
                actual=val[idx],
                desired=np.poly1d(c[idx])(x[idx]),
                err_msg=f"Failed with shapes {x.shape} and {c.shape}.",
            )

    """
    # Fine instead of Horner's method as we expect to evaluate cubic polynomials.
    X = x[..., jnp.newaxis] ** jnp.arange(c.shape[0] - 1, -1, -1)
    val = jnp.einsum("...i,i...->...", X, c)
    return val


def plot_field_line(
    B,
    pitch=None,
    bp1=jnp.array([]),
    bp2=jnp.array([]),
    start=None,
    stop=None,
    num=1000,
    title=r"Computed bounce points for $\vert B \vert$ and pitch $\lambda$",
    title_id=None,
    include_knots=True,
    alpha_knot=0.1,
    alpha_pitch=0.25,
    show=True,
):
    """Plot the field line given spline of |B|.

    Parameters
    ----------
    B : PPoly
        Spline of |B| over given field line.
    pitch : jnp.ndarray
        λ value.
    bp1 : jnp.ndarray
        Bounce points with (∂|B|/∂ζ)|ρ,α <= 0.
    bp2 : jnp.ndarray
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
    z = jnp.linspace(
        start=B.x[0] if start is None else start,
        stop=B.x[-1] if stop is None else stop,
        num=num,
    )
    add(ax.plot(z, B(z), label=r"$\vert B \vert (\zeta)$"))

    if pitch is not None:
        b = jnp.reciprocal(pitch)
        for val in b:
            add(
                ax.axhline(
                    val, color="tab:purple", alpha=alpha_pitch, label=r"$1 / \lambda$"
                )
            )
        bp1, bp2 = jnp.atleast_2d(bp1, bp2)
        for i in range(bp1.shape[0]):
            bp1_i, bp2_i = map(_filter_not_nan, (bp1[i], bp2[i]))
            add(
                ax.scatter(
                    bp1_i,
                    jnp.full_like(bp1_i, b[i]),
                    marker="v",
                    color="tab:red",
                    label="bp1",
                )
            )
            add(
                ax.scatter(
                    bp2_i,
                    jnp.full_like(bp2_i, b[i]),
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


def _check_bounce_points(bp1, bp2, pitch, knots, B_c, plot, **kwargs):
    """Check that bounce points are computed correctly."""
    eps = 10 * jnp.finfo(jnp.array(1.0).dtype).eps
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
                bp1_p, bp2_p, B_mid = map(
                    _filter_not_nan, (bp1[p, s], bp2[p, s], B_mid)
                )
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
    errorif(knots.ndim != 1)
    if B_c.ndim == 2 and B_z_ra_c.ndim == 2:
        # Add axis which enumerates field lines.
        B_c = B_c[:, jnp.newaxis]
        B_z_ra_c = B_z_ra_c[:, jnp.newaxis]
    msg = "Supplied invalid shape for splines."
    errorif(not (B_c.ndim == B_z_ra_c.ndim == 3), msg=msg)
    errorif(B_c.shape[0] - 1 != B_z_ra_c.shape[0], msg=msg)
    errorif(B_c.shape[1:] != B_z_ra_c.shape[1:], msg=msg)
    msg = "Last axis fails to enumerate spline polynomials."
    errorif(B_c.shape[-1] != knots.size - 1, msg=msg)
    if pitch is not None:
        pitch = jnp.atleast_2d(pitch)
        msg = "Supplied invalid shape for pitch angles."
        errorif(pitch.ndim != 2, msg=msg)
        errorif(pitch.shape[-1] != 1 and pitch.shape[-1] != B_c.shape[1], msg=msg)
    return B_c, B_z_ra_c, pitch


def bounce_points(pitch, knots, B_c, B_z_ra_c, check=False, plot=False, **kwargs):
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
    check : bool
        Flag for debugging.
    plot : bool
        Whether to plot some things if check is true.

    Returns
    -------
    bp1, bp2 : (jnp.ndarray, jnp.ndarray)
        Shape (P, S, N * degree).
        The field line-following ζ coordinates of bounce points for a given pitch along
        a field line. The pairs ``bp1[i,j,k]`` and ``bp2[i,j,k]`` form left and right
        integration boundaries, respectively, for the bounce integrals.

        For the shaping notation, the ``degree`` of the spline of |B| matches
        ``B_c.shape[0]-1``, the number of polynomials per spline ``N`` matches
        ``knots.size-1``, and the number of field lines is denoted by ``S``.
        If there were less than ``N*degree`` bounce points detected along a field line,
        then the last axis, which enumerates the bounce points for a particular field
        line, is padded with nan.

    """
    B_c, B_z_ra_c, pitch = _check_shape(knots, B_c, B_z_ra_c, pitch)
    P, S, N, degree = pitch.shape[0], B_c.shape[1], knots.size - 1, B_c.shape[0] - 1
    intersect = _poly_root(
        c=B_c,
        k=jnp.reciprocal(pitch)[..., jnp.newaxis],
        a_min=jnp.array([0]),
        a_max=jnp.diff(knots),
        sort=True,
        distinct=True,
    )
    assert intersect.shape == (P, S, N, degree)

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    B_z_ra = _poly_val(x=intersect, c=B_z_ra_c[..., jnp.newaxis]).reshape(P, S, -1)
    # Transform out of local power basis expansion.
    intersect = (intersect + knots[:-1, jnp.newaxis]).reshape(P, S, -1)

    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = ~jnp.isnan(intersect)
    # Reorder so that all intersects along a field line are contiguous.
    intersect = take_mask(intersect, is_intersect)
    B_z_ra = take_mask(B_z_ra, is_intersect)
    assert intersect.shape == B_z_ra.shape == (P, S, N * degree)
    is_bp1 = B_z_ra <= 0
    is_bp2 = B_z_ra >= 0
    # The pairs bp1[i, j, k] and bp2[i, j, k] are boundaries of an integral only
    # if bp1[i, j, k] <= bp2[i, j, k]. For correctness of the algorithm, it is
    # required that the first intersect satisfies non-positive derivative. Now,
    # because B_z_ra[i, j, k] <= 0 implies B_z_ra[i, j, k + 1] >= 0 by continuity,
    # there can be at most one inversion, and if it exists, the inversion must be
    # at the first pair. To correct the inversion, it suffices to disqualify the
    # first intersect as a right boundary, except under the following edge case.
    edge_case = (B_z_ra[..., 0] == 0) & (B_z_ra[..., 1] < 0)
    # In theory, we need to keep propagating this edge case,
    # e.g (B_z_ra[..., 1] < 0) | ((B_z_ra[..., 1] == 0) & (B_z_ra[..., 2] < 0)...).
    # At each step, the likelihood that an intersection has already been lost
    # due to floating point errors grows, so the real solution is to pick a less
    # degenerate pitch value - one that does not ride the global extrema of |B|.
    is_bp2 = put_along_axis(is_bp2, jnp.array(0), edge_case, axis=-1)
    # Get ζ values of bounce points from the masks.
    bp1 = take_mask(intersect, is_bp1)
    bp2 = take_mask(intersect, is_bp2)

    # Following discussion on page 3 and 5 of https://doi.org/10.1063/1.873749,
    # we ignore the bounce points of particles assigned to a class that are
    # trapped outside this snapshot of the field line.
    # TODO: Better to always consider boundary as bounce points.
    if check:
        _check_bounce_points(bp1, bp2, pitch, knots, B_c, plot, **kwargs)
    return bp1, bp2


def composite_linspace(x, num):
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
    pts = jnp.linspace(x[:-1, ...], x[1:, ...], num + 1, endpoint=False)
    pts = jnp.moveaxis(pts, source=0, destination=1).reshape(-1, *x.shape[1:])
    pts = jnp.append(pts, x[jnp.newaxis, -1, ...], axis=0)
    assert pts.shape == ((x.shape[0] - 1) * num + x.shape[0], *x.shape[1:])
    return pts


def get_pitch(min_B, max_B, num, relative_shift=1e-6):
    """Return uniformly spaced pitch values between 1 / max B and 1 / min B.

    Parameters
    ----------
    min_B, max_B : jnp.ndarray, jnp.ndarray
        Minimum and maximum |B| values.
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
    pitch = composite_linspace(jnp.reciprocal(jnp.stack([max_B, min_B])), num)
    assert pitch.shape == (num + 2, *pitch.shape[1:])
    return pitch


def get_extrema(knots, B_c, B_z_ra_c, relative_shift=1e-6):
    """Return |B| values at extrema.

    The quantity 1 / √(1 − λ |B|) common to bounce integrals is singular with
    strength ~ |ζ_b₂ - ζ_b₁| / |(∂|B|/∂ζ)|ρ,α|. Therefore, an integral over the pitch
    angle λ may have mass concentrated near λ = 1 / |B|(ζ*) where |B|(ζ*) is a
    local maximum. Depending on the quantity to integrate, it may be beneficial
    to place quadrature points at these regions.

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
    relative_shift : float
        Relative amount to shift maxima down and minima up to avoid floating point
        errors in downstream routines.

    Returns
    -------
    B_extrema : jnp.ndarray
        Shape (N * (degree - 1), S).
        For the shaping notation, the ``degree`` of the spline of |B| matches
        ``B_c.shape[0]-1``, the number of polynomials per spline ``N`` matches
        ``knots.size-1``, and the number of field lines is denoted by ``S``.
        If there were less than ``N*degree`` bounce points detected along a field line,
        then the last axis, which enumerates the bounce points for a particular field
        line, is padded with nan.

    """
    B_c, B_z_ra_c, _ = _check_shape(knots, B_c, B_z_ra_c)
    S, N, degree = B_c.shape[1], knots.size - 1, B_c.shape[0] - 1
    extrema = _poly_root(c=B_z_ra_c, a_min=jnp.array([0]), a_max=jnp.diff(knots))
    assert extrema.shape == (S, N, degree - 1)
    B_extrema = _poly_val(x=extrema, c=B_c[..., jnp.newaxis])
    B_zz_ra_extrema = _poly_val(x=extrema, c=_poly_der(B_z_ra_c)[..., jnp.newaxis])
    # Floating point error impedes consistent detection of bounce points riding
    # extrema. Shift pitch values slightly to resolve this issue.
    B_extrema = (
        jnp.where(
            # Higher priority to shift down maxima than shift up minima, so identify
            # near equality with zero as maxima.
            B_zz_ra_extrema <= 0,
            (1 - relative_shift) * B_extrema,
            (1 + relative_shift) * B_extrema,
        )
        .reshape(S, -1)
        .T
    )
    assert B_extrema.shape == (N * (degree - 1), S)
    return B_extrema


def affine_bijection(x, a, b):
    """[−1, 1] ∋ x ↦ y ∈ [a, b]."""
    y = (x + 1) / 2 * (b - a) + a
    return y


def grad_affine_bijection(a, b):
    """Gradient of affine bijection."""
    dy_dx = (b - a) / 2
    return dy_dx


def automorphism_arcsin(x):
    """[-1, 1] ∋ x ↦ y ∈ [−1, 1].

    The arcsin transformation introduces a singularity that augments the singularity
    in the bounce integral, so the quadrature scheme used to evaluate the integral must
    work well on functions with large derivative near the boundary.

    Parameters
    ----------
    x : jnp.ndarray
        Points to transform.

    Returns
    -------
    y : jnp.ndarray
        Transformed points.

    """
    y = 2 * jnp.arcsin(x) / jnp.pi
    return y


def grad_automorphism_arcsin(x):
    """Gradient of arcsin automorphism."""
    dy_dx = 2 / (jnp.sqrt(1 - x**2) * jnp.pi)
    return dy_dx


grad_automorphism_arcsin.__doc__ += "\n" + automorphism_arcsin.__doc__


def automorphism_sin(x, s=0, m=10):
    """[-1, 1] ∋ x ↦ y ∈ [−1, 1].

    When used as the change of variable map for the bounce integral, the Lipschitzness
    of the sin transformation prevents generation of new singularities. Furthermore,
    its derivative vanishes to zero slowly near the boundary, which will suppress the
    large derivatives near the boundary of singular integrals.

    In effect, this map pulls the mass of the integral away from the singularities,
    which should improve convergence if the quadrature performs better on less singular
    integrands. Pairs well with Gauss-Legendre quadrature.

    Parameters
    ----------
    x : jnp.ndarray
        Points to transform.
    s : float
        Strength of derivative suppression, s ∈ [0, 1].
    m : int
        Number of machine epsilons used for floating point error buffer.

    Returns
    -------
    y : jnp.ndarray
        Transformed points.

    """
    errorif(not (0 <= s <= 1))
    # s = 0 -> derivative vanishes like cosine.
    # s = 1 -> derivative vanishes like cosine^k.
    y0 = jnp.sin(jnp.pi * x / 2)
    y1 = x + jnp.sin(jnp.pi * x) / jnp.pi  # k = 2
    y = (1 - s) * y0 + s * y1
    # y is an expansion, so y(x) > x near x ∈ {−1, 1} and there is a tendency
    # for floating point error to overshoot the true value.
    eps = m * jnp.finfo(jnp.array(1.0).dtype).eps
    return jnp.clip(y, -1 + eps, 1 - eps)


def grad_automorphism_sin(x, s=0):
    """Gradient of sin automorphism."""
    dy0_dx = jnp.pi * jnp.cos(jnp.pi * x / 2) / 2
    dy1_dx = 1 + jnp.cos(jnp.pi * x)
    dy_dx = (1 - s) * dy0_dx + s * dy1_dx
    return dy_dx


grad_automorphism_sin.__doc__ += "\n" + automorphism_sin.__doc__


def tanh_sinh(deg, m=10):
    """Tanh-Sinh quadrature.

    Returns quadrature points xₖ and weights wₖ for the approximate evaluation of the
    integral ∫₋₁¹ f(x) dx ≈ ∑ₖ wₖ f(xₖ).

    Parameters
    ----------
    deg: int
        Number of quadrature points.
    m : int
        Number of machine epsilons used for floating point error buffer. Larger implies
        less floating point error, but increases the minimum achievable error.

    Returns
    -------
    x, w : (jnp.ndarray, jnp.ndarray)
        Quadrature points and weights.

    """
    # buffer to avoid numerical instability
    x_max = jnp.array(1.0)
    x_max = x_max - m * jnp.finfo(x_max.dtype).eps
    t_max = jnp.arcsinh(2 * jnp.arctanh(x_max) / jnp.pi)
    # maximal-spacing scheme, doi.org/10.48550/arXiv.2007.15057
    t = jnp.linspace(-t_max, t_max, deg)
    dt = 2 * t_max / (deg - 1)
    arg = 0.5 * jnp.pi * jnp.sinh(t)
    x = jnp.tanh(arg)  # x = g(t)
    w = 0.5 * jnp.pi * jnp.cosh(t) / jnp.cosh(arg) ** 2 * dt  # w = (dg/dt) dt
    return x, w


def _plot(Z, V, title_id=""):
    """Plot V[λ, (ρ, α), (ζ₁, ζ₂)](Z)."""
    for p in range(Z.shape[0]):
        for s in range(Z.shape[1]):
            is_quad_point_set = jnp.nonzero(~jnp.any(jnp.isnan(Z[p, s]), axis=-1))[0]
            if not is_quad_point_set.size:
                continue
            fig, ax = plt.subplots()
            ax.set_xlabel(r"Field line $\zeta$")
            ax.set_ylabel(title_id)
            ax.set_title(
                f"Interpolation of {title_id} to quadrature points. Index {p},{s}."
            )
            for i in is_quad_point_set:
                ax.plot(Z[p, s, i], V[p, s, i], marker="o")
            fig.text(
                0.01,
                0.01,
                f"Each color specifies the set of points and values (ζ, {title_id}(ζ)) "
                "used to evaluate an integral.",
            )
            plt.tight_layout()
            plt.show()


def _check_interpolation(Z, f, B_sup_z, B, B_z_ra, inner_product, plot):
    """Check for floating point errors.

    Parameters
    ----------
    Z : jnp.ndarray
        Quadrature points at field line-following ζ coordinates.
    f : list of jnp.ndarray
        Arguments to the integrand interpolated to Z.
    B_sup_z : jnp.ndarray
        Contravariant field-line following toroidal component of magnetic field,
        interpolated to Z.
    B : jnp.ndarray
        Norm of magnetic field, interpolated to Z.
    B_z_ra : jnp.ndarray
        Norm of magnetic field, derivative with respect to field-line following
        coordinate, interpolated to Z.
    inner_product : jnp.ndarray
        Output of ``_interpolatory_quadrature``.
    plot : bool
        Whether to plot stuff.

    """
    is_not_quad_point = jnp.isnan(Z)
    # We want quantities to evaluate as finite only at quadrature points
    # for the integrals with boundaries at valid bounce points.
    msg = "Interpolation failed."
    assert jnp.all(jnp.isfinite(B_sup_z) != is_not_quad_point), msg
    assert jnp.all(jnp.isfinite(B) != is_not_quad_point), msg
    assert jnp.all(jnp.isfinite(B_z_ra)), msg
    for f_i in f:
        assert jnp.all(jnp.isfinite(f_i) != is_not_quad_point), msg

    msg = "|B| has vanished, violating the hairy ball theorem."
    assert not jnp.isclose(B, 0).any(), msg
    assert not jnp.isclose(B_sup_z, 0).any(), msg

    quad_resolution = Z.shape[-1]
    # Number of integrals that we should be computing.
    goal = jnp.sum(1 - is_not_quad_point) // quad_resolution
    # Number of integrals that were actually computed.
    actual = jnp.isfinite(inner_product).sum()
    assert goal == actual, (
        f"Lost {goal - actual} integrals "
        "from floating point or spline approximation error."
    )
    if plot:
        _plot(Z, B, title_id=r"$\vert B \vert$")
        _plot(Z, B_sup_z, title_id=r"$ (B/\vert B \vert) \cdot e^{\zeta}$")


_interp1d_vec = jnp.vectorize(
    interp1d, signature="(m),(n),(n)->(m)", excluded={"method"}
)


@partial(jnp.vectorize, signature="(m),(n),(n),(n)->(m)", excluded={"method"})
def _interp1d_vec_with_df(xq, x, f, fx, method):
    return interp1d(xq, x, f, method, fx=fx)


def _interpolate_and_integrate(
    Z,
    w,
    integrand,
    f,
    B_sup_z,
    B,
    B_z_ra,
    pitch,
    knots,
    method,
    method_B="cubic",
    check=False,
    plot=False,
):
    """Interpolate given functions to points ``Z`` and perform quadrature.

    Parameters
    ----------
    Z : jnp.ndarray
        Shape (P, S, Z.shape[2], w.size).
        Quadrature points at field line-following ζ coordinates.

    Returns
    -------
    inner_product : jnp.ndarray
        Shape Z.shape[:-1].
        Quadrature for every pitch along every field line.

    """
    assert pitch.ndim == 2
    assert w.ndim == knots.ndim == 1
    assert 3 <= Z.ndim <= 4 and Z.shape[:2] == (pitch.shape[0], B.shape[0])
    assert Z.shape[-1] == w.size
    assert knots.size == B.shape[-1]
    assert B_sup_z.shape == B.shape == B_z_ra.shape
    # Spline the integrand so that we can evaluate it at quadrature points without
    # expensive coordinate mappings and root finding. Spline each function separately so
    # that the singularity near the bounce points can be captured more accurately than
    # can be by any polynomial.
    shape = Z.shape
    Z = Z.reshape(Z.shape[0], Z.shape[1], -1)
    f = [_interp1d_vec(Z, knots, f_i, method=method).reshape(shape) for f_i in f]
    # TODO: Pass in derivative and use method_B.
    b_sup_z = _interp1d_vec(Z, knots, B_sup_z / B, method=method).reshape(shape)
    B = _interp1d_vec_with_df(Z, knots, B, B_z_ra, method=method_B).reshape(shape)
    pitch = jnp.expand_dims(pitch, axis=(2, 3) if len(shape) == 4 else 2)
    # Assuming that the integrand is a well-behaved function of some interpolation
    # points Z, it should evaluate as NaN only if Z is NaN. This condition needs to be
    # enforced explicitly due to floating point and interpolation error. In the context
    # of bounce integrals, the √(1 − λ |B|) terms necessitate this as interpolation
    # error in |B| may yield λ|B| > 1 at quadrature points between bounce points. Don't
    # suppress inf as that indicates catastrophic floating point error.
    inner_product = jnp.dot(
        jnp.nan_to_num(integrand(*f, B=B, pitch=pitch), posinf=jnp.inf, neginf=-jnp.inf)
        / b_sup_z,
        w,
    )
    if check:
        _check_interpolation(
            Z.reshape(shape), f, b_sup_z, B, B_z_ra, inner_product, plot
        )
    return inner_product


def _bounce_quadrature(
    bp1,
    bp2,
    x,
    w,
    integrand,
    f,
    B_sup_z,
    B,
    B_z_ra,
    pitch,
    knots,
    method="akima",
    method_B="cubic",
    batch=True,
    check=False,
):
    """Bounce integrate ∫ f(ℓ) dℓ.

    Parameters
    ----------
    bp1, bp2 : jnp.ndarray, jnp.ndarray
        Shape (P, S, bp1.shape[-1]).
        The field line-following ζ coordinates of bounce points for a given pitch along
        a field line. The pairs ``bp1[i,j,k]`` and ``bp2[i,j,k]`` form left and right
        integration boundaries, respectively, for the bounce integrals.
    x, w : jnp.ndarray, jnp.ndarray
        Shape (w.size, ).
        Quadrature points in [-1, 1] and weights.
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
    B : jnp.ndarray
        Shape (S, knots.size) or (S * knots.size).
        Norm of magnetic field.
    B_z_ra : jnp.ndarray
        Shape (S, knots.size) or (S * knots.size).
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
    method_B : str
        Method of interpolation for |B|. Default is C1 cubic Hermite spline.
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
            affine_bijection(x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis]),
            w,
            integrand,
            f,
            B_sup_z,
            B,
            B_z_ra,
            pitch,
            knots,
            method,
            method_B,
            check,
            # Only developers doing debugging want to see these plots.
            plot=False,
        )
    else:
        f = list(f)

        def loop(bp):
            bp1, bp2 = bp
            return None, _interpolate_and_integrate(
                affine_bijection(x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis]),
                w,
                integrand,
                f,
                B_sup_z,
                B,
                B_z_ra,
                pitch,
                knots,
                method,
                method_B,
                check=False,
                plot=False,
            )

        result = jnp.moveaxis(
            imap(loop, (jnp.moveaxis(bp1, -1, 0), jnp.moveaxis(bp2, -1, 0)))[1],
            source=0,
            destination=-1,
        )

    result = result * grad_affine_bijection(bp1, bp2)
    assert result.shape == (pitch.shape[0], S, bp1.shape[-1])
    return result


def _fix_sign_and_normalize(B_sup_z, B, B_z_ra, B_ref=1, L_ref=1, check=False):
    """Correct signs for consistency with strictly increasing zeta requirement.

    Parameters
    ----------
    B_sup_z : jnp.ndarray
        Contravariant field-line following toroidal component of magnetic field.
    B : jnp.ndarray
        Norm of magnetic field.
    B_z_ra : jnp.ndarray
        Norm of magnetic field, derivative with respect to field-line following
        coordinate.
    B_ref : float
        Optional. Reference magnetic field strength for normalization.
    L_ref : float
        Optional. Reference length scale for normalization.
    check : bool
        Flag for debugging. Must be false for jax transformations.

    Returns
    -------
    B_sup_z, B, B_z_ra : (jnp.ndarray, jnp.ndarray, jnp.ndarray)
        Same as inputs but with corrected sign and normalized by length scales.

    """
    warnif(
        check and jnp.any(jnp.sign(B_sup_z) <= 0),
        msg="(∂ℓ/∂ζ)|ρ,a > 0 is required. Correcting signs of B^ζ and (∂|B|/∂ζ)|ρ,α.",
    )
    # Strictly increasing zeta knots enforces dζ > 0.
    # To retain dℓ = (|B|/B^ζ) dζ > 0 after fixing dζ > 0, we require B^ζ = B⋅∇ζ > 0.
    # This is equivalent to changing the sign of ∇ζ (or [∂/∂ζ]|ρ,a).
    # Recall dζ = ∇ζ⋅dR, implying 1 = ∇ζ⋅(e_ζ|ρ,a). Hence, a sign change in ∇ζ
    # induces the same sign change in e_ζ|ρ,a to retain the metric identity. For any
    # quantity f, we may write df = ∇f⋅dR, implying ∂f/∂ζ|ρ,α = ∇f ⋅ e_ζ|ρ,a. Therefore,
    # a sign change in e_ζ|ρ,a induces the same sign change in ∂f/∂ζ|ρ,α.
    B_z_ra = B_z_ra / B_ref * jnp.sign(B_sup_z)
    B_sup_z = jnp.abs(B_sup_z) * L_ref / B_ref
    B = B / B_ref
    return B_sup_z, B, B_z_ra


def bounce_integral(
    B_sup_z,
    B,
    B_z_ra,
    knots,
    quad=leggauss(21),
    automorphism=(automorphism_sin, grad_automorphism_sin),
    B_ref=1,
    L_ref=1,
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
    The quantities ``B_sup_z``, ``B``, ``B_z_ra``, and those in ``f`` supplied to the
    returned method must be separable into data evaluated along particular field lines
    via ``.reshape(S,knots.size)``. One way to satisfy this is to compute stuff on the
    grid returned from the method ``desc.equilibrium.coords.rtz_grid``. See
    ``tests.test_bounce_integral.test_bounce_integral_checks`` for example use.

    The strictly increasing knots requirement enforces dζ > 0, which constrains the
    signs of B^ζ and ∂/∂ζ. The signs of B^ζ and (∂|B|/∂ζ)|ρ,α will automatically be
    corrected to match this requirement. Pass in ``check=True`` to be notified if the
    signs for B^ζ and (∂|B|/∂ζ)|ρ,α required correction.

    Parameters
    ----------
    B_sup_z : jnp.ndarray
        Shape (S, knots.size) or (S * knots.size).
        Contravariant field-line following toroidal component of magnetic field.
        B^ζ(ρ, α, ζ) is specified by ``B_sup_z[(ρ,α),ζ]``, where in the latter the
        labels (ρ,α) are interpreted as the index into the first axis that corresponds
        to that field line.
    B : jnp.ndarray
        Shape (S, knots.size) or (S * knots.size).
        Norm of magnetic field. |B|(ρ, α, ζ) is specified by ``B[(ρ,α),ζ]``, where in
        the latter the labels (ρ,α) are interpreted as the index into the first axis
        that corresponds to that field line.
    B_z_ra : jnp.ndarray
        Shape (S, knots.size) or (S * knots.size).
        Norm of magnetic field, derivative with respect to field-line following
        coordinate. (∂|B|/∂ζ)|ρ,α(ρ, α, ζ) is specified by ``B_z_ra[(ρ,α),ζ]``, where in
        the latter the labels (ρ,α) are interpreted as the index into the first axis
        that corresponds to that field line.
    knots : jnp.ndarray
        Shape (knots.size, ).
        Field line following coordinate values where ``B_sup_z``, ``B``, ``B_z_ra``, and
        those in ``f`` supplied to the returned method were evaluated. Must be strictly
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
        Whether to plot stuff if ``check`` is true.

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
    B_sup_z, B, B_z_ra = (
        f.reshape(-1, knots.size)  # group data by field line
        for f in _fix_sign_and_normalize(B_sup_z, B, B_z_ra, B_ref, L_ref, check)
    )

    # Compute splines.
    monotonic = kwargs.pop("monotonic", False)
    # Interpax interpolation requires strictly increasing knots.
    B_c = (
        PchipInterpolator(knots, B, axis=-1, check=check).c
        if monotonic
        else CubicHermiteSpline(knots, B, B_z_ra, axis=-1, check=check).c
    )
    B_c = jnp.moveaxis(B_c, source=1, destination=-1)
    B_z_ra_c = _poly_der(B_c)
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

    def bounce_integrate(integrand, f, pitch, method="akima", batch=True):
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
            Arguments to the callable ``integrand``. These should be the scalar-valued
            functions in the bounce integrand evaluated on the DESC grid.
        pitch : jnp.ndarray
            Shape (P, S).
            λ values to evaluate the bounce integral at each field line. λ(ρ,α) is
            specified by ``pitch[...,(ρ,α)]`` where in the latter the labels (ρ,α) are
            interpreted as the index into the last axis that corresponds to that field
            line. If two-dimensional, the first axis is the batch axis.
        method : str
            Method of interpolation for functions contained in ``f``.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.
            Default is akima spline.
        batch : bool
            Whether to perform computation in a batched manner. Default is true.

        Returns
        -------
        result : jnp.ndarray
            Shape (P, S, (knots.size - 1) * degree).
            First axis enumerates pitch values. Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.

        """
        bp1, bp2 = bounce_points(pitch, knots, B_c, B_z_ra_c, check, plot)
        result = _bounce_quadrature(
            bp1,
            bp2,
            x,
            w,
            integrand,
            f,
            B_sup_z,
            B,
            B_z_ra,
            pitch,
            knots,
            method,
            method_B="monotonic" if monotonic else "cubic",
            batch=batch,
            check=check,
        )
        assert result.shape[-1] == (knots.size - 1) * degree
        return result

    return bounce_integrate, spline
