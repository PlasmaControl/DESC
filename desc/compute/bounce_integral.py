"""Methods for computing bounce integrals."""

from functools import partial

from interpax import CubicHermiteSpline, interp1d

from desc.backend import complex_sqrt, flatnonzero, jnp, put_along_axis, take
from desc.compute.utils import safediv
from desc.equilibrium.coords import desc_grid_from_field_line_coords


@partial(jnp.vectorize, signature="(m),(m)->(n)", excluded={2, 3})
def take_mask(a, mask, size=None, fill_value=None):
    """JIT compilable method to return ``a[mask][:size]`` padded by ``fill_value``.

    Parameters
    ----------
    a : Array
        The source array.
    mask : Array
        Boolean mask to index into ``a``. Should have same shape as ``a``.
    size : int
        Elements of ``a`` at the first size True indices of ``mask`` will be returned.
        If there are fewer elements than size indicates, the returned array will be
        padded with fill_value. Defaults to ``mask.size``.
    fill_value :
        When there are fewer than the indicated number of elements,
        the remaining elements will be filled with ``fill_value``.
        Defaults to NaN for inexact types,
        the largest negative value for signed types,
        the largest positive value for unsigned types,
        and True for booleans.

    Returns
    -------
    a[mask][:size] : Array, shape(size, )
        Output array.

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


def _filter_real(a, a_min=-jnp.inf, a_max=jnp.inf):
    """Keep real values inside [a_min, a_max] and set others to nan.

    Parameters
    ----------
    a : Array
        Complex-valued array.
    a_min, a_max : Array, Array
        Minimum and maximum value to keep real values between.
        Should broadcast with ``a``.

    Returns
    -------
    roots : Array
        The real values of ``a`` in [``a_min``, ``a_max``]; others set to nan.
        The returned array preserves the order of ``a``.

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


def _root_linear(a, b, distinct=False):
    """Return r such that a * r + b = 0."""
    return safediv(-b, a, fill=jnp.where(jnp.isclose(b, 0), 0, jnp.nan))


def _root_quadratic(a, b, c, distinct=False):
    """Return r such that a * r**2 + b * r + c = 0."""
    discriminant = b**2 - 4 * a * c
    C = complex_sqrt(discriminant)

    def root(xi):
        return (-b + xi * C) / (2 * a)

    is_linear = jnp.isclose(a, 0)
    suppress_root = distinct & jnp.isclose(discriminant, 0)
    r1 = jnp.where(is_linear, _root_linear(b, c), root(-1))
    r2 = jnp.where(is_linear | suppress_root, jnp.nan, root(1))
    return r1, r2


def _root_cubic(a, b, c, d, distinct=False):
    """Return r such that a * r**3 + b * r**2 + c * r + d = 0."""
    # https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
    t_0 = b**2 - 3 * a * c
    t_1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d
    discriminant = t_1**2 - 4 * t_0**3
    C = ((t_1 + complex_sqrt(discriminant)) / 2) ** (1 / 3)
    C_is_zero = jnp.isclose(t_0, 0) & jnp.isclose(t_1, 0)

    def root(xi):
        return (b + xi * C + jnp.where(C_is_zero, 0, t_0 / (xi * C))) / (-3 * a)

    xi0 = 1
    xi1 = (-1 + (-3) ** 0.5) / 2
    xi2 = xi1**2
    is_quadratic = jnp.isclose(a, 0)
    # C = 0 is equivalent to existence of triple root.
    # Assuming the coefficients are real, it is also equivalent to
    # existence of any real roots with multiplicity > 1.
    suppress_root = distinct & C_is_zero
    q1, q2 = _root_quadratic(b, c, d, distinct)
    r1 = jnp.where(is_quadratic, q1, root(xi0))
    r2 = jnp.where(is_quadratic, q2, jnp.where(suppress_root, jnp.nan, root(xi1)))
    r3 = jnp.where(is_quadratic | suppress_root, jnp.nan, root(xi2))
    return r1, r2, r3


_roots = jnp.vectorize(partial(jnp.roots, strip_zeros=False), signature="(m)->(n)")


def poly_root(c, k=0, a_min=None, a_max=None, sort=False, distinct=False):
    """Roots of polynomial with given coefficients.

    Parameters
    ----------
    c : Array
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.
    k : Array
        Specify to find solutions to ∑ᵢⁿ cᵢ xⁱ = ``k``.
        Should broadcast with arrays of shape(*c.shape[1:]).
    a_min, a_max : Array, Array
        Minimum and maximum value to return roots between.
        If specified only real roots are returned.
        If None, returns all complex roots.
        Should broadcast with arrays of shape(*c.shape[1:]).
    sort : bool
        Whether to sort the roots.
    distinct : bool
        Whether to only return the distinct roots. If true, when the
        multiplicity is greater than one, the repeated roots are set to nan.

    Returns
    -------
    r : Array, shape(..., c.shape[1:], c.shape[0] - 1)
        The roots of the polynomial, iterated over the last axis.

    """
    keep_only_real = not (a_min is None and a_max is None)
    func = {2: _root_linear, 3: _root_quadratic, 4: _root_cubic}
    if c.shape[0] in func:
        # Compute from analytic formula.
        r = func[c.shape[0]](*c[:-1], c[-1] - k, distinct)
        if keep_only_real:
            r = [_filter_real(rr, a_min, a_max) for rr in r]
        r = jnp.stack(r, axis=-1)
        # We had ignored the case of double complex roots.
        distinct = distinct and c.shape[0] > 3 and not keep_only_real
    else:
        # Compute from eigenvalues of polynomial companion matrix.
        # This method can fail to detect roots near extrema, which is often
        # where we want to detect roots for bounce integrals.
        c_n = c[-1] - k
        c = [jnp.broadcast_to(c_i, c_n.shape) for c_i in c[:-1]]
        c.append(c_n)
        c = jnp.stack(c, axis=-1)
        r = _roots(c)
        if keep_only_real:
            if a_min is not None:
                a_min = a_min[..., jnp.newaxis]
            if a_max is not None:
                a_max = a_max[..., jnp.newaxis]
            r = _filter_real(r, a_min, a_max)
    if sort or distinct:
        r = jnp.sort(r, axis=-1)
    if distinct:
        mask = jnp.isclose(jnp.diff(r, axis=-1, prepend=jnp.nan), 0)
        r = jnp.where(mask, jnp.nan, r)
    return r


def poly_int(c, k=None):
    """Coefficients for the primitives of the given set of polynomials.

    Parameters
    ----------
    c : Array
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.
    k : Array
        Integration constants.
        Should broadcast with arrays of shape(*coef.shape[1:]).

    Returns
    -------
    poly : Array
        Coefficients of polynomial primitive.
        That is, ``poly[i]`` stores the coefficient of the monomial xⁿ⁻ⁱ⁺¹,
        where n is ``c.shape[0] - 1``.

    """
    if k is None:
        k = jnp.broadcast_to(0.0, c.shape[1:])
    poly = (c.T / jnp.arange(c.shape[0], 0, -1)).T
    poly = jnp.append(poly, k[jnp.newaxis], axis=0)
    return poly


def poly_der(c):
    """Coefficients for the derivatives of the given set of polynomials.

    Parameters
    ----------
    c : Array
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.

    Returns
    -------
    poly : Array
        Coefficients of polynomial derivative, ignoring the arbitrary constant.
        That is, ``poly[i]`` stores the coefficient of the monomial xⁿ⁻ⁱ⁻¹,
        where n is ``c.shape[0] - 1``.

    """
    poly = (c[:-1].T * jnp.arange(c.shape[0] - 1, 0, -1)).T
    return poly


def poly_val(x, c):
    """Evaluate the set of polynomials c at the points x.

    Note that this function does not perform the same operation as
    ``np.polynomial.polynomial.polyval(x, c)``.

    Parameters
    ----------
    x : Array
        Coordinates at which to evaluate the set of polynomials.
    c : Array
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.

    Returns
    -------
    val : Array
        Polynomial with given coefficients evaluated at given points.

    Examples
    --------
    .. code-block:: python

        val = polyval(x, c)
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
    # Should be fine to do this instead of Horner's method
    # because we expect to usually integrate up to quartic polynomials.
    X = x[..., jnp.newaxis] ** jnp.arange(c.shape[0] - 1, -1, -1)
    val = jnp.einsum("...i,i...->...", X, c)
    assert val.ndim == max(x.ndim, c.ndim - 1)
    return val


def _check_shape(knots, B_c, B_z_ra_c, pitch=None):
    """Ensure inputs have compatible shape, and return them with full dimension.

    Parameters
    ----------
    knots : Array, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.

    Returns
    -------
    B_c : Array, shape(B_c.shape[0], S, knots.size - 1)
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.
    B_z_ra_c : Array, shape(B_c.shape[0] - 1, *B_c.shape[1:])
        Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.
    pitch : Array, shape(P, S)
        λ values.
        Last axis enumerates the λ value for a particular field line
        parameterized by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., (ρ, α)]``
        where in the latter the labels (ρ, α) are interpreted as index into the
        last axis that corresponds to that field line.
        If two-dimensional, the first axis is the batch axis as usual.

    """
    if B_c.ndim == 2 and B_z_ra_c.ndim == 2:
        # Add axis which enumerates field lines.
        B_c = B_c[:, jnp.newaxis]
        B_z_ra_c = B_z_ra_c[:, jnp.newaxis]
    err_msg = "Supplied invalid shape for splines."
    assert B_c.ndim == B_z_ra_c.ndim == 3, err_msg
    assert (
        B_c.shape[0] - 1 == B_z_ra_c.shape[0] and B_c.shape[1:] == B_z_ra_c.shape[1:]
    ), err_msg
    assert (
        B_c.shape[-1] == knots.size - 1
    ), "Last axis fails to enumerate spline polynomials."
    if pitch is not None:
        pitch = jnp.atleast_2d(pitch)
        err_msg = "Supplied invalid shape for pitch angles."
        assert pitch.ndim == 2, err_msg
        assert pitch.shape[-1] == 1 or pitch.shape[-1] == B_c.shape[1], err_msg
    return B_c, B_z_ra_c, pitch


def pitch_of_extrema(knots, B_c, B_z_ra_c):
    """Return pitch values that will capture fat banana orbits.

    These pitch values are 1/|B|(ζ*) where |B|(ζ*) are local maxima.
    The local minima are returned as well.

    Parameters
    ----------
    knots : Array, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    B_c : Array, shape(B_c.shape[0], S, knots.size - 1)
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.
    B_z_ra_c : Array, shape(B_c.shape[0] - 1, *B_c.shape[1:])
        Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.

    Returns
    -------
    pitch : Array, shape(N * (degree - 1), S)
        For the shaping notation, the ``degree`` of the spline of |B| matches
        ``B_c.shape[0] - 1``, the number of polynomials per spline ``N`` matches
        ``knots.size - 1``, and the number of field lines is denoted by ``S``.

        If there were less than ``N * (degree - 1)`` extrema detected along a
        field line, then the first axis, which enumerates the pitch values for
        a particular field line, is padded with nan.

    """
    B_c, B_z_ra_c, _ = _check_shape(knots, B_c, B_z_ra_c)
    S, N, degree = B_c.shape[1], knots.size - 1, B_c.shape[0] - 1
    extrema = poly_root(
        c=B_z_ra_c,
        a_min=jnp.array([0]),
        a_max=jnp.diff(knots),
        # False to double weight orbits with |B|_z_ra = |B|_zz_ra = 0 at bounce points.
        distinct=True,
    )
    # Can detect at most degree of |B|_z_ra spline extrema between each knot.
    assert extrema.shape == (S, N, degree - 1)
    # Reshape so that last axis enumerates (unsorted) extrema along a field line.
    B_extrema = poly_val(x=extrema, c=B_c[..., jnp.newaxis]).reshape(S, -1)
    # Might be useful to pad all the nan at the end rather than interspersed.
    B_extrema = take_mask(B_extrema, ~jnp.isnan(B_extrema))
    pitch = 1 / B_extrema.T
    assert pitch.shape == (N * (degree - 1), S)
    return pitch


def bounce_points(knots, B_c, B_z_ra_c, pitch, check=False):
    """Compute the bounce points given spline of |B| and pitch λ.

    Parameters
    ----------
    knots : Array, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    B_c : Array, shape(B_c.shape[0], S, knots.size - 1)
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.
    B_z_ra_c : Array, shape(B_c.shape[0] - 1, *B_c.shape[1:])
        Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.
    pitch : Array, shape(P, S)
        λ values.
        Last axis enumerates the λ value for a particular field line
        parameterized by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., (ρ, α)]``
        where in the latter the labels (ρ, α) are interpreted as index into the
        last axis that corresponds to that field line.
        If two-dimensional, the first axis is the batch axis as usual.
    check : bool
        Flag for debugging.

    Returns
    -------
    bp1, bp2 : Array, Array, shape(P, S, N * degree)
        For the shaping notation, the ``degree`` of the spline of |B| matches
        ``B_c.shape[0] - 1``, the number of polynomials per spline ``N`` matches
        ``knots.size - 1``, and the number of field lines is denoted by ``S``.

        The returned arrays are the field line-following ζ coordinates of bounce
        points for a given pitch along a field line. The pairs bp1[i, j, k] and
        bp2[i, j, k] form left and right integration boundaries, respectively,
        for the bounce integrals. If there were less than ``N * degree`` bounce
        points detected along a field line, then the last axis, which enumerates
        the bounce points for a particular field line, is padded with nan.

    """
    B_c, B_z_ra_c, pitch = _check_shape(knots, B_c, B_z_ra_c, pitch)
    P, S, N, degree = pitch.shape[0], B_c.shape[1], knots.size - 1, B_c.shape[0] - 1
    # The polynomials' intersection points with 1 / λ is given by ``intersect``.
    # In order to be JIT compilable, this must have a shape that accommodates the
    # case where each polynomial intersects 1 / λ degree times.
    # nan values in ``intersect`` denote a polynomial has less than degree intersects.
    intersect = poly_root(
        c=B_c,
        # Expand to use same pitches across polynomials of a particular spline.
        k=jnp.expand_dims(1 / pitch, axis=-1),
        a_min=jnp.array([0]),
        a_max=jnp.diff(knots),
        sort=True,
        distinct=True,  # Required for correctness of ``edge_case``.
    )
    assert intersect.shape == (P, S, N, degree)

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    B_z_ra = poly_val(x=intersect, c=B_z_ra_c[..., jnp.newaxis]).reshape(P, S, -1)
    # Transform out of local power basis expansion.
    intersect = intersect + knots[:-1, jnp.newaxis]
    intersect = intersect.reshape(P, S, -1)

    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = ~jnp.isnan(intersect)
    # Reorder so that all intersects along a field line are contiguous.
    intersect = take_mask(intersect, is_intersect)
    B_z_ra = take_mask(B_z_ra, is_intersect)
    assert intersect.shape == B_z_ra.shape == (P, S, N * degree)
    # Sign of derivative determines whether an intersect is a valid bounce point.
    # Need to include zero derivative intersects to compute the WFB
    # (world's fattest banana) orbit bounce integrals.
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
    is_bp2 = put_along_axis(is_bp2, jnp.array(0), edge_case, axis=-1)
    # Get ζ values of bounce points from the masks.
    bp1 = take_mask(intersect, is_bp1)
    bp2 = take_mask(intersect, is_bp2)

    if check:
        if jnp.any(bp1 > bp2):
            raise AssertionError("Bounce points have an inversion.")
        if jnp.any(bp1[..., 1:] < bp2[..., :-1]):
            raise AssertionError(
                "Discontinuity detected. Is B_z_ra the derivative of the spline of B?"
            )

    return bp1, bp2
    # This is no longer implemented at the moment.
    #   If the first intersect is at a non-negative derivative, that particle
    #   may be trapped in a well outside this snapshot of the field line. If, in
    #   addition, the last intersect is at a non-positive derivative, then we
    #   have information to compute a bounce integral between these points.
    #   This single bounce integral is somewhat undefined since the field typically
    #   does not close on itself, but in some cases it can make sense to include it.
    #   To make this integral well-defined, an approximation is made that the field
    #   line is periodic such that ζ = knots[-1] can be interpreted as ζ = 0 so
    #   that the distance between these bounce points is well-defined. This is fine
    #   as long as after a transit the field line begins physically close to where
    #   it began on the previous transit, for then continuity of |B| implies
    #   |B|(knots[-1] < ζ < knots[-1] + knots[0]) is close to |B|(0 < ζ < knots[0]).
    #   We don't need to check conditions for the latter, because if they are not
    #   satisfied, the quadrature will evaluate √(1 − λ |B|) as nan automatically.


def _compute_bp_if_given_pitch(
    knots, B_c, B_z_ra_c, pitch, check, *original, err=False
):
    """Conditionally return the ingredients needed to compute bounce integrals.

    Parameters
    ----------
    original : tuple
        Whatever this method returned earlier.
    err : bool
        Whether to raise an error if ``pitch`` is None and ``original`` is empty.

    """
    if pitch is None:
        if err and not original:
            raise ValueError("No pitch values were given.")
        return original
    else:
        pitch = jnp.atleast_2d(pitch)
        return *bounce_points(knots, B_c, B_z_ra_c, pitch, check), pitch


def tanh_sinh_quad(resolution, w=lambda x: 1):
    """Tanh-Sinh quadrature.

    Returns quadrature points xₖ and weights Wₖ for the approximate evaluation
    of the integral ∫₋₁¹ w(x) f(x) dx ≈ ∑ₖ Wₖ f(xₖ).

    Parameters
    ----------
    resolution: int
        Number of quadrature points.
    w : callable
        Weight function defined, positive, and continuous on (-1, 1).

    Returns
    -------
    x : Array
        Quadrature points.
    W : Array
        Quadrature weights.

    """
    # boundary of integral
    x_max = jnp.array(1.0)
    # subtract machine epsilon with buffer for floating point error
    x_max = x_max - 10 * jnp.finfo(x_max).eps
    # inverse of tanh-sinh transformation
    t_max = jnp.arcsinh(2 * jnp.arctanh(x_max) / jnp.pi)
    kh = jnp.linspace(-t_max, t_max, resolution)
    h = 2 * t_max / (resolution - 1)
    arg = 0.5 * jnp.pi * jnp.sinh(kh)
    x = jnp.tanh(arg)
    # weights for Tanh-Sinh quadrature ∫₋₁¹ f(x) dx ≈ ∑ₖ ωₖ f(xₖ)
    W = 0.5 * jnp.pi * h * jnp.cosh(kh) / jnp.cosh(arg) ** 2
    W = W * w(x)
    return x, W


_interp1d_vec = jnp.vectorize(
    interp1d,
    signature="(m),(n),(n)->(m)",
    excluded={"method", "derivative", "extrap", "period"},
)


@partial(
    jnp.vectorize,
    signature="(m),(n),(n),(n)->(m)",
    excluded={"method", "derivative", "extrap", "period"},
)
def _interp1d_vec_with_df(
    xq,
    x,
    f,
    fx,
    method="cubic",
    derivative=0,
    extrap=False,
    period=None,
):
    return interp1d(xq, x, f, method, derivative, extrap, period, fx=fx)


def _bounce_quad(Z, w, knots, B_sup_z, B, B_z_ra, integrand, f, pitch, method):
    """Compute bounce quadrature for every pitch along every field line.

    Parameters
    ----------
    Z : Array, shape(P, S, Z.shape[2], w.size)
        Quadrature points at field line-following ζ coordinates.
    w : Array, shape(w.size, )
        Quadrature weights.
    knots : Array, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    B_sup_z : Array, shape(S, knots.size, )
        Contravariant field-line following toroidal component of magnetic field.
    B : Array, shape(S, knots.size, )
        Norm of magnetic field.
    B_z_ra : Array, shape(S, knots.size, )
        Norm of magnetic field derivative with respect to field-line following label.
    integrand : callable
        This callable is the composition operator on the set of functions in ``f``
        that maps the functions in ``f`` to the integrand f(ℓ) in ∫ f(ℓ) dℓ.
        It should accept the items in ``f`` as arguments as well as two additional
        keyword arguments: ``B``, and ``pitch``. A quadrature will be performed to
        approximate the bounce integral of ``integrand(*f, B=B, pitch=pitch)``.
        Note that any arrays baked into the callable method should broadcast
        with arrays of shape(P, S, 1, 1).
    f : iterable of Array, shape(P, S, knots.size, )
        Arguments to the callable ``integrand``.
        These should be the functions in the integrand of the bounce integral
        evaluated (or interpolated to) the nodes of the returned desc
        coordinate grid.
        All items in the list should be two-dimensional. The first axis of
        that item is interpreted as the batch axis, which enumerates the
        evaluation of the function at particular pitch values.
    pitch : Array, shape(P, S)
        λ values.
    method : str
        Method of interpolation for functions contained in ``f``.
        See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.

    Returns
    -------
    inner_product : Array, shape(Z.shape[:-1])
        Bounce quadrature for every pitch along every field line.

    """
    assert pitch.ndim == 2
    assert w.ndim == knots.ndim == 1
    assert Z.shape == (pitch.shape[0], B.shape[0], Z.shape[2], w.size)
    assert knots.size == B.shape[-1]
    assert B_sup_z.shape == B.shape == B_z_ra.shape
    # Spline the integrand so that we can evaluate it at quadrature points
    # without expensive coordinate mappings and root finding.
    # Spline each function separately so that the singularity near the bounce
    # points can be captured more accurately than can be by any polynomial.
    shape = Z.shape
    Z = Z.reshape(Z.shape[0], Z.shape[1], -1)
    f = [_interp1d_vec(Z, knots, ff, method=method).reshape(shape) for ff in f]
    B_sup_z = _interp1d_vec(Z, knots, B_sup_z, method=method).reshape(shape)
    # Specify derivative at knots for ≈ cubic hermite interpolation.
    B = _interp1d_vec_with_df(Z, knots, B, B_z_ra, method="cubic").reshape(shape)
    pitch = pitch[..., jnp.newaxis, jnp.newaxis]
    inner_product = jnp.dot(integrand(*f, B=B, pitch=pitch) / B_sup_z, w)
    return inner_product


def _affine_bijection_forward(x, a, b):
    """[a, b] ∋ x ↦ y ∈ [−1, 1]."""
    y = 2 * (x - a) / (b - a) - 1
    return y


def _affine_bijection_reverse(x, a, b):
    """[−1, 1] ∋ x ↦ y ∈ [a, b]."""
    y = (x + 1) / 2 * (b - a) + a
    return y


def _grad_affine_bijection_reverse(a, b):
    """Gradient of reverse affine bijection."""
    dy_dx = (b - a) / 2
    return dy_dx


def automorphism_arcsin(x):
    """[-1, 1] ∋ x ↦ y ∈ [−1, 1].

    The arcsin automorphism is an expansion, so it pushes the evaluation points
    of the bounce integrand toward the singular region, which may induce
    floating point error.

    The gradient of the arcsin automorphism introduces a singularity that augments
    the singularity in the bounce integral. Therefore, the quadrature scheme
    used to evaluate the integral must work well on hypersingular integrals.

    Parameters
    ----------
    x : Array

    Returns
    -------
    y : Array

    """
    y = 2 * jnp.arcsin(x) / jnp.pi
    return y


def grad_automorphism_arcsin(x):
    """Gradient of arcsin automorphism.

    The arcsin automorphism is an expansion, so it pushes the evaluation points
    of the bounce integrand toward the singular region, which may induce
    floating point error.

    The gradient of the arcsin automorphism introduces a singularity that augments
    the singularity in the bounce integral. Therefore, the quadrature scheme
    used to evaluate the integral must work well on hypersingular integrals.

    Parameters
    ----------
    x : Array

    Returns
    -------
    dy_dx : Array

    """
    dy_dx = 2 / (jnp.sqrt(1 - x**2) * jnp.pi)
    return dy_dx


def automorphism_sin(x):
    """[-1, 1] ∋ x ↦ y ∈ [−1, 1].

    The sin automorphism is a contraction, so it pulls the evaluation points
    of the bounce integrand away from the singular region, inducing less
    floating point error.

    The derivative of the sin automorphism is Lipschitz.
    When this automorphism is used as the change of variable map for the bounce
    integral, the Lipschitzness prevents generation of new singularities.
    Furthermore, its derivative vanishes like the integrand of the elliptic
    integral the second kind E(φ | 1), suppressing the singularity in the
    bounce integrand.

    Therefore, this automorphism pulls the mass of the bounce integral away
    from the singularities, which should improve convergence of the quadrature
    to the principal value of the true integral, so long as the quadrature
    performs better on less singular integrands. If the integral was
    hypersingular to begin with, Tanh-Sinh quadrature will still work well.
    Otherwise, Gauss-Legendre quadrature can outperform Tanh-Sinh.

    Parameters
    ----------
    x : Array

    Returns
    -------
    y : Array

    """
    y = jnp.sin(jnp.pi * x / 2)
    return y


def grad_automorphism_sin(x):
    """Gradient of sin automorphism.

    The sin automorphism is a contraction, so it will pull the evaluation points
    away from the singular region, inducing less floating point error.

    The sin automorphism is a contraction, so it pulls the evaluation points
    of the bounce integrand away from the singular region, inducing less
    floating point error.

    The derivative of the sin automorphism is Lipschitz.
    When this automorphism is used as the change of variable map for the bounce
    integral, the Lipschitzness prevents generation of new singularities.
    Furthermore, its derivative vanishes like the integrand of the elliptic
    integral the second kind E(φ | 1), suppressing the singularity in the
    bounce integrand.

    Therefore, this automorphism pulls the mass of the bounce integral away
    from the singularities, which should improve convergence of the quadrature
    to the principal value of the true integral, so long as the quadrature
    performs better on less singular integrands. If the integral was
    hypersingular to begin with, Tanh-Sinh quadrature will still work well.
    Otherwise, Gauss-Legendre quadrature can outperform Tanh-Sinh.

    Parameters
    ----------
    x : Array

    Returns
    -------
    dy_dx : Array

    """
    dy_dx = jnp.pi * jnp.cos(jnp.pi * x / 2) / 2
    return dy_dx


def bounce_integral_map(
    eq,
    rho=jnp.linspace(1e-12, 1, 10),
    alpha=None,
    knots=jnp.linspace(0, 6 * jnp.pi, 20),
    quad=tanh_sinh_quad,
    automorphism=automorphism_sin,
    grad_automorphism=grad_automorphism_sin,
    pitch=None,
    return_items=True,
    **kwargs,
):
    """Returns a method to compute the bounce integral of any quantity.

    The bounce integral is defined as the principal value of ∫ f(ℓ) dℓ, where
        dℓ parameterizes the distance along the field line,
        λ is a constant proportional to the magnetic moment over energy,
        |B| is the norm of the magnetic field,
        f(ℓ) is the quantity to integrate along the field line,
        and the boundaries of the integral are bounce points, ζ₁, ζ₂, such that
        (λ |B|)(ζᵢ) = 1.
    Physically, the pitch angle λ is the magnetic moment over the energy
    of particle. For a particle with fixed λ, bounce points are defined to be
    the location on the field line such that the particle's velocity parallel
    to the magnetic field is zero.

    The bounce integral is defined up to a sign.
    We choose the sign that corresponds the particle's guiding center trajectory
    traveling in the direction of increasing field-line-following label.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium on which the bounce integral is computed.
    rho : Array
        Unique flux surface label coordinates.
    alpha : Array
        Unique field line label coordinates over a constant rho surface.
    knots : Array
        Field line following coordinate values at which to compute a spline
        of the integrand, for every field line in the meshgrid formed from
        rho and alpha specified above.
        The number of knots specifies a grid resolution as increasing the
        number of knots increases the accuracy of representing the integrand
        and the accuracy of the locations of the bounce points.
    quad : callable
        The quadrature scheme used to evaluate the integral.
        The returned quadrature points xₖ and weights wₖ
        should approximate ∫₋₁¹ g(x) dx = ∑ₖ wₖ g(xₖ).
        For the default choice of the automorphism below,
        Tanh-Sinh quadrature works well if the integrand is hypersingular.
        Otherwise, Gauss-Legendre quadrature can be more competitive.
    automorphism : callable
        The reverse automorphism of the real interval [-1, 1] defined below.
        The forward automorphism is composed with the affine bijection
        that maps the bounce points to [-1, 1]. The resulting forward map defines
        a change of variable for the bounce integral. The choice made for
        the automorphism can augment of suppress singularities.
        Keep this in mind when choosing the quadrature method.
    grad_automorphism : callable
        Derivative of the reverse automorphism, i.e. the derivative of the map
        ``automorphism``. (Or 1 / derivative of the forward automorphism).
        May be useful to use automatic differentiation.
    pitch : Array, shape(P, S)
        λ values to evaluate the bounce integral at each field line.
        May be specified later.
        Last axis enumerates the λ value for a particular field line parameterized
        by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., (ρ, α)]``
        where in the latter the labels (ρ, α) are interpreted as index into the
        last axis that corresponds to that field line.
        If two-dimensional, the first axis is the batch axis as usual.
    return_items : bool
        Whether to return ``items`` as described below.
    kwargs
        Can specify additional arguments to the quadrature function with kwargs.

    Returns
    -------
    bounce_integral : callable
        This callable method computes the bounce integral ∫ f(ℓ) dℓ for every
        specified field line ℓ (constant rho and alpha), for every λ value in ``pitch``.
    items : dict
        grid_fl : Grid
            Clebsch-Type field-line coordinate grid.
        grid_desc : Grid
            DESC coordinate grid for the given field line coordinates.
        data : dict
            Dictionary of Arrays of stuff evaluated on ``grid``.
        B.c : Array, shape(4, S, zeta.size - 1)
            Polynomial coefficients of the spline of |B| in local power basis.
            First axis enumerates the coefficients of power series.
            Second axis enumerates the splines along the field lines.
            Last axis enumerates the polynomials of the spline along a particular
            field line.
        B_z_ra.c : Array, shape(3, S, zeta.size - 1)
            Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
            First axis enumerates the coefficients of power series.
            Second axis enumerates the splines along the field lines.
            Last axis enumerates the polynomials of the spline along a particular
            field line.

    Examples
    --------
    Suppose we want to compute a bounce average of the function
    f(ℓ) = (1 − λ |B|) * g_zz, where g_zz is the squared norm of the
    toroidal basis vector on some set of field lines specified by (ρ, α)
    coordinates. This is defined as
        [∫ f(ℓ) / √(1 − λ |B|) dℓ] / [∫ 1 / √(1 − λ |B|) dℓ]


    .. code-block:: python

        def integrand_num(g_zz, B, pitch):
            # Integrand in integral in numerator of bounce average.
            f = (1 - pitch * B) * g_zz
            return f / jnp.sqrt(1 - pitch * B)

        def integrand_den(B, pitch):
            # Integrand in integral in denominator of bounce average.
            return 1 / jnp.sqrt(1 - pitch * B)

        eq = get("HELIOTRON")
        rho = jnp.linspace(1e-12, 1, 6)
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 5)
        knots = jnp.linspace(0, 6 * jnp.pi, 20)

        bounce_integral, items = bounce_integral_map(eq, rho, alpha, knots)

        g_zz = eq.compute("g_zz", grid=items["grid_desc"], data=items["data"])["g_zz"]
        pitch = pitch_of_extrema(knots, items["B.c"], items["B_z_ra.c"])
        num = bounce_integral(integrand_num, g_zz, pitch)
        den = bounce_integral(integrand_den, [], pitch)
        average = num / den
        assert jnp.isfinite(average).any()

        # Now we can group the data by field line.
        average = average.reshape(pitch.shape[0], rho.size, alpha.size, -1)
        # The bounce averages stored at index i, j
        i, j = 0, 0
        print(average[:, i, j])
        # are the bounce averages along the field line with nodes
        # given in Clebsch-Type field-line coordinates ρ, α, ζ
        nodes = items["grid_fl"].nodes.reshape(rho.size, alpha.size, -1, 3)
        print(nodes[i, j])
        # for the pitch values stored in
        pitch = pitch.reshape(pitch.shape[0], rho.size, alpha.size)
        print(pitch[:, i, j])
        # Some of these bounce averages will evaluate as nan.
        # You should filter out these nan values when computing stuff.
        average_sum_over_field_line = jnp.nansum(average, axis=-1)
        print(average_sum_over_field_line)
        assert not jnp.allclose(average_sum_over_field_line, 0)

    """
    check = kwargs.pop("check", False)
    normalize = kwargs.pop("normalize", 1)
    if quad == tanh_sinh_quad:
        kwargs.setdefault("resolution", 19)
    x, w = quad(**kwargs)
    # The gradient of the reverse transformation is the weight function w(x) of
    # the quadrature. Apply weight function for the automorphism.
    w = w * grad_automorphism(x)
    # Apply reverse automorphism change of variable to quadrature points.
    # Recall x = forward(_affine_bijection_forward(ζ, ζ_b₁, ζ_b₂)).
    x = automorphism(x)

    if alpha is None:
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 10)
    rho = jnp.atleast_1d(rho)
    alpha = jnp.atleast_1d(alpha)
    knots = jnp.atleast_1d(knots)
    # number of field lines or splines
    S = rho.size * alpha.size

    grid_fl, grid_desc, data = desc_grid_from_field_line_coords(eq, rho, alpha, knots)
    data = eq.compute(["B^zeta", "|B|", "|B|_z|r,a"], grid=grid_desc, data=data)
    B_sup_z = data["B^zeta"].reshape(S, knots.size)
    B = data["|B|"].reshape(S, knots.size) / normalize
    B_z_ra = data["|B|_z|r,a"].reshape(S, knots.size) / normalize
    B_c = jnp.moveaxis(
        CubicHermiteSpline(knots, B, B_z_ra, axis=-1, check=check).c,
        source=1,
        destination=-1,
    )
    assert B_c.shape == (4, S, knots.size - 1)
    B_z_ra_c = poly_der(B_c)
    assert B_z_ra_c.shape == (3, S, knots.size - 1)
    original = _compute_bp_if_given_pitch(knots, B_c, B_z_ra_c, pitch, check, err=False)

    def _group_grid_data_by_field_line(f):
        assert f.ndim <= 2, "See the docstring below."
        return f.reshape(-1, S, knots.size)

    def bounce_integral(integrand, f, pitch=None, method="akima"):
        """Bounce integrate ∫ f(ℓ) dℓ.

        Parameters
        ----------
        integrand : callable
            This callable is the composition operator on the set of functions in ``f``
            that maps the functions in ``f`` to the integrand f(ℓ) in ∫ f(ℓ) dℓ.
            It should accept the items in ``f`` as arguments as well as two additional
            keyword arguments: ``B``, and ``pitch``. A quadrature will be performed to
            approximate the bounce integral of ``integrand(*f, B=B, pitch=pitch)``.
            Note that any arrays baked into the callable method should broadcast
            with arrays of shape(P, S, 1, 1) where
                P is the batch axis size of pitch,
                S is the number of field lines given by rho.size * alpha.size.
        f : list of Array, shape(P, items["grid_desc"].num_nodes, )
            Arguments to the callable ``integrand``.
            These should be the functions in the integrand of the bounce integral
            evaluated (or interpolated to) the nodes of the returned desc
            coordinate grid.
            If an item in the list is two-dimensional, the first axis of that
            item is interpreted as the batch axis, which enumerates the
            evaluation of the function at particular pitch values.
        pitch : Array, shape(P, S)
            λ values to evaluate the bounce integral at each field line.
            If None, uses the values given to the parent function.
            Last axis enumerates the λ value for a particular field line parameterized
            by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., (ρ, α)]``
            where in the latter the labels (ρ, α) are interpreted as index into the
            last axis that corresponds to that field line.
            If two-dimensional, the first axis is the batch axis as usual.
        method : str
            Method of interpolation for functions contained in ``f``.
            Defaults to akima spline to suppress oscillation.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html.

        Returns
        -------
        result : Array, shape(P, S, (zeta.size - 1) * 3)
            First axis enumerates pitch values.
            Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.

        """
        bp1, bp2, pitch = _compute_bp_if_given_pitch(
            knots, B_c, B_z_ra_c, pitch, check, *original, err=True
        )
        # # Apply affine change of variable to quadrature points.
        Z = _affine_bijection_reverse(x, bp1[..., jnp.newaxis], bp2[..., jnp.newaxis])
        if not isinstance(f, (list, tuple)):
            f = [f]
        f = map(_group_grid_data_by_field_line, f)
        # Integrate and complete the change of variable.
        result = _bounce_quad(
            Z, w, knots, B_sup_z, B, B_z_ra, integrand, f, pitch, method
        ) * _grad_affine_bijection_reverse(bp1, bp2)
        assert result.shape == (pitch.shape[0], S, (knots.size - 1) * 3)
        return result

    if return_items:
        items = {
            "grid_fl": grid_fl,
            "grid_desc": grid_desc,
            "data": data,
            "knots": knots,
            "B.c": B_c,
            "B_z_ra.c": B_z_ra_c,
        }
        return bounce_integral, items
    else:
        return bounce_integral
