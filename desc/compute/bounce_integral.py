"""Methods for computing bounce integrals."""

from functools import partial

from interpax import CubicHermiteSpline, interp1d

from desc.backend import complex_sqrt, flatnonzero, jnp, put_along_axis, take, vmap
from desc.compute.utils import safediv
from desc.equilibrium.coords import desc_grid_from_field_line_coords

roots = jnp.vectorize(partial(jnp.roots, strip_zeros=False), signature="(m)->(n)")


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
            r = tuple(map(partial(_filter_real, a_min=a_min, a_max=a_max), r))
        r = jnp.stack(r, axis=-1)
        if sort:
            r = jnp.sort(r, axis=-1)
    else:
        # Compute from eigenvalues of polynomial companion matrix.
        # This method can fail to detect roots near extrema, which is often
        # where we want to detect roots for bounce integrals.
        c_n = c[-1] - k
        c = [jnp.broadcast_to(c_i, c_n.shape) for c_i in c[:-1]]
        c.append(c_n)
        c = jnp.stack(c)
        r = roots(c.reshape(c.shape[0], -1).T).reshape(*c.shape[1:], -1)
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


def _check_shape(knots, B, B_z_ra, pitch=None):
    """Ensure inputs have correct shape and return labels for those shapes."""
    if B.ndim == 2 and B_z_ra.ndim == 2:
        # Add axis which enumerates field lines.
        B = B[:, jnp.newaxis]
        B_z_ra = B_z_ra[:, jnp.newaxis]
    err_msg = "Supplied invalid shape for splines."
    assert B.ndim == B_z_ra.ndim == 3, err_msg
    S = B.shape[1]
    N = knots.size - 1
    degree = B.shape[0] - 1
    assert degree == B_z_ra.shape[0] and B.shape[1:] == B_z_ra.shape[1:], err_msg
    assert N == B.shape[-1], "Last axis fails to enumerate spline polynomials."

    if pitch is None:
        return S, N, degree
    pitch = jnp.atleast_2d(pitch)
    err_msg = "Supplied invalid shape for pitch angles."
    assert pitch.ndim == 2, err_msg
    assert pitch.shape[-1] == 1 or pitch.shape[-1] == B.shape[1], err_msg
    P = pitch.shape[0]
    return P, S, N, degree


def pitch_of_extrema(knots, B, B_z_ra):
    """Return pitch values that will capture fat banana orbits.

    These pitch values are 1/|B|(ζ*) where |B|(ζ*) are local maxima.
    The local minima are returned as well.

    Parameters
    ----------
    knots : Array, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    B : Array, shape(B.shape[0], S, knots.size - 1)
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.
    B_z_ra : Array, shape(B.shape[0] - 1, *B.shape[1:])
        Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.

    Returns
    -------
    pitch : Array, shape(N * (degree - 1), S)
        For the shaping notation, the ``degree`` of the spline of |B| matches
        ``B.shape[0] - 1``, the number of polynomials per spline ``N`` matches
        ``knots.size - 1``, and the number of field lines is denoted by ``S``.

        If there were less than ``N * (degree - 1)`` extrema detected along a
        field line, then the first axis, which enumerates the pitch values for
        a particular field line, is padded with nan.

    """
    S, N, degree = _check_shape(knots, B, B_z_ra)
    extrema = poly_root(
        c=B_z_ra,
        a_min=jnp.array([0]),
        a_max=jnp.diff(knots),
        sort=False,  # don't need to sort
        # False will double weight orbits with B_z = B_zz = 0 at bounce points.
        distinct=True,
    )
    # Can detect at most degree of |B|_z spline extrema between each knot.
    assert extrema.shape == (S, N, degree - 1)
    # Reshape so that last axis enumerates (unsorted) extrema along a field line.
    B_extrema = poly_val(x=extrema, c=B[..., jnp.newaxis]).reshape(S, -1)
    # Might be useful to pad all the nan at the end rather than interspersed.
    B_extrema = take_mask(B_extrema, ~jnp.isnan(B_extrema))
    pitch = 1 / B_extrema.T
    assert pitch.shape == (N * (degree - 1), S)
    return pitch


def bounce_points(knots, B, B_z_ra, pitch, check=False):
    """Compute the bounce points given spline of |B| and pitch λ.

    Parameters
    ----------
    knots : Array, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    B : Array, shape(B.shape[0], S, knots.size - 1)
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis enumerates the coefficients of power series.
        Second axis enumerates the splines along the field lines.
        Last axis enumerates the polynomials of the spline along a particular
        field line.
    B_z_ra : Array, shape(B.shape[0] - 1, *B.shape[1:])
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
        ``B.shape[0] - 1``, the number of polynomials per spline ``N`` matches
        ``knots.size - 1``, and the number of field lines is denoted by ``S``.

        The returned arrays are the field line-following ζ coordinates of bounce
        points for a given pitch along a field line. The pairs bp1[i, j] and
        bp2[i, j] form left and right integration boundaries, respectively,
        for the bounce integrals. If there were less than ``N * degree`` bounce
        points detected along a field line, then the last axis, which enumerates
        the bounce points for a particular field line, is padded with nan.

    """
    P, S, N, degree = _check_shape(knots, B, B_z_ra, pitch)
    # The polynomials' intersection points with 1 / λ is given by ``intersect``.
    # In order to be JIT compilable, this must have a shape that accommodates the
    # case where each polynomial intersects 1 / λ degree times.
    # nan values in ``intersect`` denote a polynomial has less than degree intersects.
    intersect = poly_root(
        c=B,
        # Expand to use same pitches across polynomials of a particular spline.
        k=jnp.expand_dims(1 / pitch, axis=-1),
        a_min=jnp.array([0]),
        a_max=jnp.diff(knots),
        sort=True,
        distinct=True,  # Required for correctness of ``edge_case``.
    )

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    # Condense remaining axes to vectorize over them.
    B_z_ra = poly_val(x=intersect, c=B_z_ra[..., jnp.newaxis]).reshape(P * S, -1)
    # Transform out of local power basis expansion.
    intersect = intersect + knots[:-1, jnp.newaxis]
    intersect = intersect.reshape(P * S, -1)

    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = ~jnp.isnan(intersect)
    # Reorder so that all intersects along a field line are contiguous.
    intersect = take_mask(intersect, is_intersect)
    B_z_ra = take_mask(B_z_ra, is_intersect)
    assert intersect.shape == B_z_ra.shape == (P * S, N * degree)
    # Sign of derivative determines whether an intersect is a valid bounce point.
    # Need to include zero derivative intersects to compute the WFB
    # (world's fattest banana) orbit bounce integrals.
    is_bp1 = B_z_ra <= 0
    is_bp2 = B_z_ra >= 0
    # The pairs bp1[i, j] and bp2[i, j] are boundaries of an integral only if
    # bp1[i, j] <= bp2[i, j]. For correctness of the algorithm, it is necessary
    # that the first intersect satisfies non-positive derivative. Now, because
    # B_z_ra[i, j] <= 0 implies B_z_ra[i, j + 1] >= 0 by continuity, there can
    # be at most one inversion, and if it exists, the inversion must be at the
    # first pair. To correct the inversion, it suffices to disqualify the first
    # intersect as an ending bounce point, except under the following edge case.
    edge_case = (B_z_ra[:, 0] == 0) & (B_z_ra[:, 1] < 0)
    is_bp2 = put_along_axis(is_bp2, jnp.array(0), edge_case, axis=-1)
    # Get ζ values of bounce points from the masks.
    bp1 = take_mask(intersect, is_bp1).reshape(P, S, -1)
    bp2 = take_mask(intersect, is_bp2).reshape(P, S, -1)

    if check:
        if jnp.any(bp1 > bp2):
            raise AssertionError("Bounce points have an inversion.")
        if jnp.any(bp1[:, 1:] < bp2[:, :-1]):
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
    knots, B, B_z_ra, pitch, *original, err=False, check=False
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
        return *bounce_points(knots, B, B_z_ra, pitch, check), pitch


def tanh_sinh_quad(resolution=7):
    """Tanh-Sinh quadrature.

    This function outputs the quadrature points xₖ and weights wₖ
    for a tanh-sinh quadrature.

    ∫₋₁¹ f(x) dx = ∑ₖ wₖ f(xₖ)

    Parameters
    ----------
    resolution: int
        Number of quadrature points, preferably odd.

    Returns
    -------
    x : numpy array
        Quadrature points
    w : numpy array
        Quadrature weights

    """
    # boundary of integral
    x_max = jnp.array(1.0)
    # subtract machine epsilon with buffer for floating point error
    x_max = x_max - 10 * jnp.finfo(x_max).eps
    # inverse of tanh-sinh transformation
    t_max = jnp.arcsinh(2 * jnp.arctanh(x_max) / jnp.pi)
    kh = jnp.linspace(-t_max, t_max, resolution)
    h = 2 * t_max / (resolution - 1)
    x = jnp.tanh(0.5 * jnp.pi * jnp.sinh(kh))
    w = 0.5 * jnp.pi * h * jnp.cosh(kh) / jnp.cosh(0.5 * jnp.pi * jnp.sinh(kh)) ** 2
    return x, w


# Vectorize to compute a bounce integral for every pitch along every field line.
@partial(vmap, in_axes=(1, 1, None, None, 0, 0, 0, 0, None), out_axes=1)
def _bounce_quad(pitch, X, w, knots, f, B_sup_z, B, B_z_ra, f_method):
    """Compute a bounce integral for every pitch along a particular field line.

    Parameters
    ----------
    pitch : Array, shape(pitch.size, )
        λ values.
    X : Array, shape(pitch.size, X.shape[1], w.size)
        Quadrature points.
    w : Array, shape(w.size, )
        Quadrature weights.
    knots : Array, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    f : Array, shape(knots.size, )
        Function to compute bounce integral of, evaluated at knots.
    B_sup_z : Array, shape(knots.size, )
        Contravariant field-line following toroidal component of magnetic field.
    B : Array, shape(knots.size, )
        Norm of magnetic field.
    B_z_ra : Array, shape(knots.size, )
        Norm of magnetic field derivative with respect to field-line following label.
    f_method : str
        Method of interpolation for f.

    Returns
    -------
    inner_product : Array, shape(pitch.size, X.shape[1])
        Bounce integrals for every pitch along a particular field line.

    """
    assert pitch.ndim == 1 == w.ndim
    assert X.shape == (pitch.size, X.shape[1], w.size)
    assert knots.shape == f.shape == B_sup_z.shape == B.shape == B_z_ra.shape
    # Spline the integrand so that we can evaluate it at quadrature points
    # without expensive coordinate mappings and root finding.
    # Spline each function separately so that the singularity near the bounce
    # points can be captured more accurately than can be by any polynomial.
    shape = X.shape
    X = X.ravel()
    if f_method == "constant":
        f = f[0]
    else:
        f = interp1d(X, knots, f, method=f_method).reshape(shape)
    # Use akima spline to suppress oscillation.
    B_sup_z = interp1d(X, knots, B_sup_z, method="akima").reshape(shape)
    # Specify derivative at knots with fx=B_z_ra for ≈ cubic hermite interpolation.
    B = interp1d(X, knots, B, fx=B_z_ra, method="cubic").reshape(shape)
    pitch = pitch[:, jnp.newaxis, jnp.newaxis]
    inner_product = jnp.dot(f / (B_sup_z * jnp.sqrt(1 - pitch * B)), w)
    return inner_product


def bounce_integral(
    eq,
    pitch=None,
    rho=jnp.linspace(1e-12, 1, 10),
    alpha=None,
    zeta=jnp.linspace(0, 6 * jnp.pi, 20),
    quad=tanh_sinh_quad,
    **kwargs,
):
    """Returns a method to compute the bounce integral of any quantity.

    The bounce integral is defined as F_ℓ(λ) = ∫ f(ℓ) / √(1 − λ |B|) dℓ, where
        dℓ parameterizes the distance along the field line,
        λ is a constant proportional to the magnetic moment over energy,
        |B| is the norm of the magnetic field,
        f(ℓ) is the quantity to integrate along the field line,
        and the endpoints of the integration are at the bounce points.
    Physically, the pitch angle λ is the magnetic moment over the energy
    of particle. For a particle with fixed λ, bounce points are defined to be
    the location on the field line such that the particle's velocity parallel
    to the magnetic field is zero, i.e. λ |B| = 1.

    The bounce integral is defined up to a sign.
    We choose the sign that corresponds the particle's guiding center trajectory
    traveling in the direction of increasing field-line-following label.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium on which the bounce integral is computed.
    pitch : Array, shape(P, S)
        λ values to evaluate the bounce integral at each field line.
        May be specified later.
        Last axis enumerates the λ value for a particular field line parameterized
        by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., (ρ, α)]``
        where in the latter the labels (ρ, α) are interpreted as index into the
        last axis that corresponds to that field line.
        If two-dimensional, the first axis is the batch axis as usual.
    rho : Array
        Unique flux surface label coordinates.
    alpha : Array
        Unique field line label coordinates over a constant rho surface.
    zeta : Array
        A spline of the integrand is computed at these values of the field
        line following coordinate, for every field line in the meshgrid formed from
        rho and alpha specified above.
        The number of knots specifies the grid resolution as increasing the
        number of knots increases the accuracy of representing the integrand
        and the accuracy of the locations of the bounce points.
    quad : callable
        The quadrature scheme used to evaluate the integral.
        Should return quadrature points and weights when called.
        The returned points should be within the domain [-1, 1].
    kwargs : dict
        Can specify additional arguments to the quadrature function with kwargs.
        Can also specify whether to return items with ``return_items=True``.

    Returns
    -------
    bi : callable
        This callable method computes the bounce integral F_ℓ(λ) for every
        specified field line ℓ (constant rho and alpha), for every λ value in ``pitch``.
    items : dict
        Dictionary of useful intermediate quantities.
            grid : Grid
                DESC coordinate grid for the given field line coordinates.
            data : dict
                Dictionary of Arrays of stuff evaluated on ``grid``.
            poly_B : Array, shape(4, S, zeta.size - 1)
                Polynomial coefficients of the spline of |B| in local power basis.
                First axis enumerates the coefficients of power series.
                Second axis enumerates the splines along the field lines.
                Last axis enumerates the polynomials of the spline along a particular
                field line.
            poly_B_z : Array, shape(3, S, zeta.size - 1)
                Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
                First axis enumerates the coefficients of power series.
                Second axis enumerates the splines along the field lines.
                Last axis enumerates the polynomials of the spline along a particular
                field line.

    Examples
    --------
    .. code-block:: python

        rho = jnp.linspace(1e-12, 1, 6)
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 5)
        bi, items = bounce_integral(eq, rho=rho, alpha=alpha, return_items=True)
        name = "g_zz"
        f = eq.compute(name, grid=items["grid"], data=items["data"])[name]
        B = items["data"]["B"].reshape(rho.size * alpha.size, -1)
        pitch_res = 30
        pitch = jnp.linspace(1 / B.max(axis=-1), 1 / B.min(axis=-1), pitch_res)
        result = bi(f, pitch).reshape(pitch_res, rho.size, alpha.size, -1)

    """
    check = kwargs.pop("check", False)
    return_items = kwargs.pop("return_items", False)

    if alpha is None:
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 10)
    rho = jnp.atleast_1d(rho)
    alpha = jnp.atleast_1d(alpha)
    zeta = jnp.atleast_1d(zeta)
    S = rho.size * alpha.size

    grid, data = desc_grid_from_field_line_coords(eq, rho, alpha, zeta)
    data = eq.compute(["B^zeta", "|B|", "|B|_z|r,a"], grid=grid, data=data)
    B_sup_z = data["B^zeta"].reshape(S, -1)
    B = data["|B|"].reshape(S, -1)
    B_z_ra = data["|B|_z|r,a"].reshape(S, -1)
    poly_B = jnp.moveaxis(
        CubicHermiteSpline(zeta, B, B_z_ra, axis=-1, check=check).c, 1, -1
    )
    poly_B_z = poly_der(poly_B)
    assert poly_B.shape == (4, S, zeta.size - 1)
    assert poly_B_z.shape == (3, S, zeta.size - 1)

    x, w = quad(**kwargs)
    # change of variable, x = sin([0.5 + (ζ − ζ_b₂)/(ζ_b₂−ζ_b₁)] π)
    x = jnp.arcsin(x) / jnp.pi - 0.5
    original = _compute_bp_if_given_pitch(
        zeta, poly_B, poly_B_z, pitch, err=False, check=check
    )

    def _bounce_integral(f, pitch=None, f_method="akima"):
        """Compute the bounce integral of ``f``.

        Parameters
        ----------
        f : Array, shape(items["grid"].num_nodes, )
            Quantity to compute the bounce integral of.
        pitch : Array, shape(P, S)
            λ values to evaluate the bounce integral at each field line.
            If None, uses the values given to the parent function.
            Last axis enumerates the λ value for a particular field line parameterized
            by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., (ρ, α)]``
            where in the latter the labels (ρ, α) are interpreted as index into the
            last axis that corresponds to that field line.
            If two-dimensional, the first axis is the batch axis as usual.
        f_method : str, optional
            Method of interpolation for f.

        Returns
        -------
        result : Array, shape(P, S, (zeta.size - 1) * 3)
            First axis enumerates pitch values.
            Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.

        """
        bp1, bp2, pitch = _compute_bp_if_given_pitch(
            zeta, poly_B, poly_B_z, pitch, *original, err=True, check=check
        )
        X = x * (bp2 - bp1)[..., jnp.newaxis] + bp2[..., jnp.newaxis]
        pitch = jnp.broadcast_to(pitch, shape=(pitch.shape[0], S))
        f = f.reshape(S, -1)
        result = (
            _bounce_quad(pitch, X, w, zeta, f, B_sup_z, B, B_z_ra, f_method)
            / (bp2 - bp1)
            * jnp.pi
        )
        assert result.shape == (pitch.shape[0], S, (zeta.size - 1) * 3)
        return result

    if return_items:
        items = {"grid": grid, "data": data, "poly_B": poly_B, "poly_B_z": poly_B_z}
        return _bounce_integral, items
    else:
        return _bounce_integral


def bounce_average(
    eq,
    pitch=None,
    rho=jnp.linspace(1e-12, 1, 10),
    alpha=None,
    zeta=jnp.linspace(0, 6 * jnp.pi, 20),
    quad=tanh_sinh_quad,
    **kwargs,
):
    """Returns a method to compute the bounce average of any quantity.

    The bounce average is defined as
    F_ℓ(λ) = (∫ f(ℓ) / √(1 − λ |B|) dℓ) / (∫ 1 / √(1 − λ |B|) dℓ), where
        dℓ parameterizes the distance along the field line,
        λ is a constant proportional to the magnetic moment over energy,
        |B| is the norm of the magnetic field,
        f(ℓ) is the quantity to integrate along the field line,
        and the endpoints of the integration are at the bounce points.
    Physically, the pitch angle λ is the magnetic moment over the energy
    of particle. For a particle with fixed λ, bounce points are defined to be
    the location on the field line such that the particle's velocity parallel
    to the magnetic field is zero, i.e. λ |B| = 1.

    The bounce integral is defined up to a sign.
    We choose the sign that corresponds the particle's guiding center trajectory
    traveling in the direction of increasing field-line-following label.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium on which the bounce average is computed.
    pitch : Array, shape(P, S)
        λ values to evaluate the bounce integral at each field line.
        May be specified later.
        Last axis enumerates the λ value for a particular field line parameterized
        by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., (ρ, α)]``
        where in the latter the labels (ρ, α) are interpreted as index into the
        last axis that corresponds to that field line.
        If two-dimensional, the first axis is the batch axis as usual.
    rho : Array
        Unique flux surface label coordinates.
    alpha : Array
        Unique field line label coordinates over a constant rho surface.
    zeta : Array
        A spline of the integrand is computed at these values of the field
        line following coordinate, for every field line in the meshgrid formed from
        rho and alpha specified above.
        The number of knots specifies the grid resolution as increasing the
        number of knots increases the accuracy of representing the integrand
        and the accuracy of the locations of the bounce points.
    quad : callable
        The quadrature scheme used to evaluate the integral.
        Should return quadrature points and weights when called.
        The returned points should be within the domain [-1, 1].
    kwargs : dict
        Can specify additional arguments to the quadrature function with kwargs.
        Can also specify whether to return items with ``return_items=True``.

    Returns
    -------
    ba : callable
        This callable method computes the bounce average F_ℓ(λ) for every
        specified field line ℓ (constant rho and alpha), for every λ value in ``pitch``.
    items : dict
        Dictionary of useful intermediate quantities.
            grid : Grid
                DESC coordinate grid for the given field line coordinates.
            data : dict
                Dictionary of Arrays of stuff evaluated on ``grid``.
            poly_B : Array, shape(4, S, zeta.size - 1)
                Polynomial coefficients of the spline of |B| in local power basis.
                First axis enumerates the coefficients of power series.
                Second axis enumerates the splines along the field lines.
                Last axis enumerates the polynomials of the spline along a particular
                field line.
            poly_B_z : Array, shape(3, S, zeta.size - 1)
                Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
                First axis enumerates the coefficients of power series.
                Second axis enumerates the splines along the field lines.
                Last axis enumerates the polynomials of the spline along a particular
                field line.

    Examples
    --------
    .. code-block:: python

        rho = jnp.linspace(1e-12, 1, 6)
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 5)
        ba, items = bounce_average(eq, rho=rho, alpha=alpha, return_items=True)
        name = "g_zz"
        f = eq.compute(name, grid=items["grid"], data=items["data"])[name]
        B = items["data"]["B"].reshape(rho.size * alpha.size, -1)
        pitch_res = 30
        pitch = jnp.linspace(1 / B.max(axis=-1), 1 / B.min(axis=-1), pitch_res)
        result = ba(f, pitch).reshape(pitch_res, rho.size, alpha.size, -1)

    """

    def _bounce_average(f, pitch=None, f_method="akima"):
        """Compute the bounce average of ``f``.

        Parameters
        ----------
        f : Array, shape(items["grid"].num_nodes, )
            Quantity to compute the bounce average of.
        pitch : Array, shape(P, S)
            λ values to evaluate the bounce average at each field line.
            If None, uses the values given to the parent function.
            Last axis enumerates the λ value for a particular field line parameterized
            by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., (ρ, α)]``
            where in the latter the labels (ρ, α) are interpreted as index into the
            last axis that corresponds to that field line.
            If two-dimensional, the first axis is the batch axis as usual.
        f_method : str, optional
            Method of interpolation for f.


        Returns
        -------
        result : Array, shape(P, S, (zeta.size - 1) * 3)
            First axis enumerates pitch values.
            Second axis enumerates the field lines.
            Last axis enumerates the bounce integrals.

        """
        return bi(f, pitch, f_method) / bi(jnp.ones_like(f), pitch, "constant")

    bi = bounce_integral(eq, pitch, rho, alpha, zeta, quad, **kwargs)
    if kwargs.get("return_items"):
        bi, items = bi
        return _bounce_average, items
    else:
        return _bounce_average
