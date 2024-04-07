"""Methods for computing bounce integrals."""

from functools import partial

from interpax import CubicHermiteSpline, interp1d

from desc.backend import complex_sqrt, flatnonzero, jnp, put_along_axis, take, vmap
from desc.compute.utils import safediv
from desc.equilibrium.coords import desc_grid_from_field_line_coords

roots = jnp.vectorize(partial(jnp.roots, strip_zeros=False), signature="(n)->(m)")


# vmap to compute a bounce integral for every pitch along every field line.
@partial(vmap, in_axes=(1, 1, None, None, 0, 0, 0, 0), out_axes=1)
def bounce_quadrature(pitch, X, w, knots, f, B_sup_z, B, B_z_ra):
    """Compute a bounce integral for every pitch along a particular field line.

    Parameters
    ----------
    pitch : ndarray, shape(pitch.size, )
        λ values.
    X : ndarray, shape(pitch.size, (knots.size - 1) * degree, w.size)
        Quadrature points.
    w : ndarray, shape(w.size, )
        Quadrature weights.
    knots : ndarray, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    f : ndarray, shape(knots.size, )
        Function to compute bounce integral of, evaluated at knots.
    B_sup_z : ndarray, shape(knots.size, )
        Contravariant field-line following toroidal component of magnetic field.
    B : ndarray, shape(knots.size, )
        Norm of magnetic field.
    B_z_ra : ndarray, shape(knots.size, )
        Norm of magnetic field derivative with respect to field-line following label.

    Returns
    -------
    inner_product : ndarray, shape(P, (knots.size - 1) * degree)
        Bounce integrals for every pitch along a particular field line.

    """
    assert pitch.ndim == 1 == w.ndim
    assert X.shape == (pitch.size, (knots.size - 1) * 3, w.size)
    assert knots.shape == f.shape == B_sup_z.shape == B.shape == B_z_ra.shape
    # Spline the integrand so that we can evaluate it at quadrature points
    # without expensive coordinate mappings and root finding.
    # Spline each function separately so that the singularity near the bounce
    # points can be captured more accurately than can be by any polynomial.
    shape = X.shape
    X = X.ravel()
    # Use akima spline to suppress oscillation.
    f = interp1d(X, knots, f, method="akima").reshape(shape)
    B_sup_z = interp1d(X, knots, B_sup_z, method="akima").reshape(shape)
    # Specify derivative at knots with fx=B_z_ra for ≈ cubic hermite interpolation.
    B = interp1d(X, knots, B, fx=B_z_ra, method="cubic").reshape(shape)
    pitch = pitch[:, jnp.newaxis, jnp.newaxis]
    inner_product = jnp.dot(f / (B_sup_z * jnp.sqrt(1 - pitch * B)), w)
    return inner_product


def tanh_sinh_quadrature(resolution=7):
    """
    tanh_sinh quadrature.

    This function outputs the quadrature points and weights
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
    # https://github.com/f0uriest/quadax/blob/main/quadax/utils.py#L166
    # Compute boundary of quadrature.
    # x_max = 1 - eps with some buffer
    x_max = jnp.array(1.0) - 10 * jnp.finfo(jnp.array(1.0)).eps
    tanhinv = lambda x: 1 / 2 * jnp.log((1 + x) / (1 - x))
    sinhinv = lambda x: jnp.log(x + jnp.sqrt(x**2 + 1))
    # inverse of tanh-sinh transformation for x_max
    t_max = sinhinv(2 / jnp.pi * tanhinv(x_max))

    points = jnp.linspace(-t_max, t_max, resolution)
    h = 2 * t_max / (resolution - 1)
    sinh_points = jnp.sinh(points)
    x = jnp.tanh(0.5 * jnp.pi * sinh_points)
    w = 0.5 * jnp.pi * h * jnp.cosh(points) / jnp.cosh(0.5 * jnp.pi * sinh_points) ** 2
    return x, w


@vmap
def take_mask(a, mask, size=None, fill_value=None):
    """JIT compilable method to return ``a[mask][:size]`` padded by ``fill_value``.

    Parameters
    ----------
    a : ndarray
        The source array.
    mask : ndarray
        Boolean mask to index into ``a``. Should have same size as ``a``.
    size : int
        Elements of ``a`` at the first size True indices of ``mask`` will be returned.
        If there are fewer elements than size indicates, the returned array will be
        padded with fill_value. Defaults to ``a.size``.
    fill_value :
        When there are fewer than the indicated number of elements,
        the remaining elements will be filled with ``fill_value``.
        Defaults to NaN for inexact types,
        the largest negative value for signed types,
        the largest positive value for unsigned types,
        and True for booleans.

    Returns
    -------
    a_mask : ndarray, shape(size, )
        Output array.

    """
    assert a.size == mask.size
    if size is None:
        size = mask.size
    idx = flatnonzero(mask, size=size, fill_value=mask.size)
    a_mask = take(
        a,
        idx,
        axis=0,
        mode="fill",
        fill_value=fill_value,
        unique_indices=True,
        indices_are_sorted=True,
    )
    return a_mask


def polyint(c, k=None):
    """Coefficients for the primitives of the given set of polynomials.

    Parameters
    ----------
    c : ndarray
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.
    k : ndarray
        Integration constants.
        Should broadcast with arrays of shape(*coef.shape[1:]).

    Returns
    -------
    poly : ndarray
        Coefficients of polynomial primitive.
        That is, ``poly[i]`` stores the coefficient of the monomial xⁿ⁻ⁱ⁺¹,
        where n is ``c.shape[0] - 1``.

    """
    if k is None:
        k = jnp.broadcast_to(0.0, c.shape[1:])
    poly = (c.T / jnp.arange(c.shape[0], 0, -1)).T
    poly = jnp.append(poly, k[jnp.newaxis], axis=0)
    return poly


def polyder(c):
    """Coefficients for the derivatives of the given set of polynomials.

    Parameters
    ----------
    c : ndarray
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.

    Returns
    -------
    poly : ndarray
        Coefficients of polynomial derivative, ignoring the arbitrary constant.
        That is, ``poly[i]`` stores the coefficient of the monomial xⁿ⁻ⁱ⁻¹,
        where n is ``c.shape[0] - 1``.

    """
    poly = (c[:-1].T * jnp.arange(c.shape[0] - 1, 0, -1)).T
    return poly


def polyval(x, c):
    """Evaluate the set of polynomials c at the points x.

    Note that this function does not perform the same operation as
    ``np.polynomial.polynomial.polyval(x, c)``.

    Parameters
    ----------
    x : ndarray
        Coordinates at which to evaluate the set of polynomials.
    c : ndarray
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.

    Returns
    -------
    val : ndarray
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


def _complex_to_nan(root, a_min=-jnp.inf, a_max=jnp.inf):
    """Set complex-valued roots and real roots outside [a_min, a_max] to nan.

    Parameters
    ----------
    root : ndarray
        Complex-valued roots.
    a_min, a_max : ndarray
        Minimum and maximum value to return roots between.
        Should broadcast with ``root`` array.

    Returns
    -------
    roots : ndarray
        The real roots in [a_min, a_max], others transformed to nan.

    """
    if a_min is None:
        a_min = -jnp.inf
    if a_max is None:
        a_max = jnp.inf
    return jnp.where(
        jnp.isclose(jnp.imag(root), 0) & (a_min <= root) & (root <= a_max),
        jnp.real(root),
        jnp.nan,
    )


def _root_linear(a, b):
    sentinel = -1  # 0 is minimum value for valid root in local power basis
    return safediv(-b, a, fill=sentinel)


def _root_quadratic(a, b, c):
    t = complex_sqrt(b**2 - 4 * a * c)
    root = lambda xi: safediv(-b + xi * t, 2 * a)
    is_linear = jnp.isclose(a, 0)
    r1 = jnp.where(is_linear, _root_linear(b, c), root(-1))
    r2 = jnp.where(is_linear, jnp.nan, root(1))
    return r1, r2


def _root_cubic(a, b, c, d):
    # https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
    t_0 = b**2 - 3 * a * c
    t_1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d
    C = ((t_1 + complex_sqrt(t_1**2 - 4 * t_0**3)) / 2) ** (1 / 3)
    C_is_zero = jnp.isclose(C, 0)

    def root(xi):
        return safediv(b + xi * C + jnp.where(C_is_zero, 0, t_0 / (xi * C)), -3 * a)

    xi1 = (-1 + (-3) ** 0.5) / 2
    xi2 = xi1**2
    xi3 = 1
    is_quadratic = jnp.isclose(a, 0)
    q1, q2 = _root_quadratic(b, c, d)
    r1 = jnp.where(is_quadratic, q1, root(xi1))
    r2 = jnp.where(is_quadratic, q2, root(xi2))
    r3 = jnp.where(is_quadratic, jnp.nan, root(xi3))
    return r1, r2, r3


def poly_roots(coef, k=0, a_min=None, a_max=None, sort=False):
    """Roots of polynomial with given real coefficients.

    Parameters
    ----------
    coef : ndarray
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.
    k : ndarray
        Specify to find solutions to ∑ᵢⁿ cᵢ xⁱ = ``k``.
        Should broadcast with arrays of shape(*coef.shape[1:]).
    a_min, a_max : ndarray
        Minimum and maximum value to return roots between.
        If specified only real roots are returned.
        If None, returns all complex roots.
        Should broadcast with arrays of shape(*coef.shape[1:]).
    sort : bool
        Whether to sort the roots.

    Returns
    -------
    r : ndarray
        The roots of the polynomial, iterated over the last axis.

    """
    # TODO: need to add option to filter double/triple roots into single roots
    if 2 <= coef.shape[0] <= 4:
        # compute from analytic formula
        func = {4: _root_cubic, 3: _root_quadratic, 2: _root_linear}[coef.shape[0]]
        r = func(*coef[:-1], coef[-1] - k)
        if not (a_min is None and a_max is None):
            r = tuple(map(partial(_complex_to_nan, a_min=a_min, a_max=a_max), r))
        r = jnp.stack(r, axis=-1)
    else:
        # compute from eigenvalues of polynomial companion matrix
        d = coef[-1] - k
        c = [jnp.broadcast_to(c, d.shape) for c in coef[:-1]]
        c.append(d)
        coef = jnp.stack(c)
        r = roots(coef.reshape(coef.shape[0], -1).T).reshape(*coef.shape[1:], -1)
        if not (a_min is None and a_max is None):
            if a_min is not None:
                a_min = a_min[..., jnp.newaxis]
            if a_max is not None:
                a_max = a_max[..., jnp.newaxis]
            r = _complex_to_nan(r, a_min, a_max)
    if sort:
        r = jnp.sort(r, axis=-1)
    return r


def pitch_extrema(knots, poly_B, poly_B_z):
    """Returns pitch that will capture fat banana orbits.

    These pitch values are 1/|B|(ζ*) where |B|(ζ*) is a local maximum.
    The local minimum are returned as well.

    Parameters
    ----------
    knots : ndarray, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray, shape(poly_B.shape[0], R * A, knots.size - 1)
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.
    poly_B_z : ndarray, shape(poly_B_z.shape[0], R * A, knots.size - 1)
        Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.

    Returns
    -------
    pitch : ndarray, shape((knots.size - 1) * (poly_B_z.shape[0] - 1), R * A)
        Returns at most pitch.shape[0] many pitch values for every field line.
        If less extrema were found, then the array has nan padded on the right.
        You will likely need to reshape the output as follows:
        pitch = pitch.reshape(pitch.shape[0], rho.size, alpha.size).

    """
    RA = poly_B.shape[1]  # rho.size * alpha.size
    N = knots.size - 1  # number of piecewise cubic polynomials per field line
    assert poly_B.shape[1:] == poly_B_z.shape[1:]
    assert poly_B.shape[-1] == N
    degree = poly_B_z.shape[0] - 1
    extrema = poly_roots(
        coef=poly_B_z,
        a_min=jnp.array([0]),
        a_max=jnp.diff(knots),
        sort=False,  # don't need to sort
    )
    assert extrema.shape == (RA, N, degree)
    B_extrema = polyval(x=extrema, c=poly_B[..., jnp.newaxis]).reshape(RA, -1)
    B_extrema = take_mask(B_extrema, ~jnp.isnan(B_extrema))
    pitch = 1 / B_extrema.T
    assert pitch.shape == (N * degree, RA)
    return pitch


def compute_bounce_points(pitch, knots, poly_B, poly_B_z):
    """Compute the bounce points given |B| and pitch λ.

    Parameters
    ----------
    pitch : ndarray, shape(P, R * A)
        λ values.
        Last two axes should specify the λ value for a particular field line
        parameterized by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., ρ, α]``
        where in the latter the labels are interpreted as indices that correspond
        to that field line.
        If an additional axis exists on the left, it is the batch axis as usual.
    knots : ndarray, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray, shape(poly_B.shape[0], R * A, knots.size - 1)
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.
    poly_B_z : ndarray, shape(poly_B_z.shape[0], R * A, knots.size - 1)
        Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.

    Returns
    -------
    bp1, bp2 : ndarray, ndarray
        Field line-following ζ coordinates of bounce points for a given pitch
        along a field line. Has shape (P, R * A, (knots.size - 1) * 3).
        If there were less than (knots.size - 1) * 3 bounce points along a
        field line, then the last axis is padded with nan.
        The pairs bp1[..., i] and bp2[..., i] form integration boundaries
        for bounce integrals.

    """
    P = pitch.shape[0]  # batch size
    RA = poly_B.shape[1]  # rho.size * alpha.size
    N = knots.size - 1  # number of piecewise cubic polynomials per field line
    assert poly_B.shape[1:] == poly_B_z.shape[1:]
    assert poly_B.shape[-1] == N
    degree = poly_B.shape[0] - 1

    # The polynomials' intersection points with 1 / λ is given by ``intersect``.
    # In order to be JIT compilable, this must have a shape that accommodates the
    # case where each cubic polynomial intersects 1 / λ thrice.
    # nan values in ``intersect`` denote a polynomial has less than three intersects.
    intersect = poly_roots(
        coef=poly_B,
        k=jnp.expand_dims(1 / pitch, axis=-1),
        a_min=jnp.array([0]),
        a_max=jnp.diff(knots),
        sort=True,
    )
    assert intersect.shape == (P, RA, N, degree)

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    # Condense remaining axes to vmap over them.
    B_z = polyval(x=intersect, c=poly_B_z[..., jnp.newaxis]).reshape(P * RA, -1)
    # Transform from local power basis expansion to real space.
    intersect = intersect + knots[:-1, jnp.newaxis]
    intersect = intersect.reshape(P * RA, -1)
    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = ~jnp.isnan(intersect)

    # Rearrange so that all intersects along a field line are contiguous.
    intersect = take_mask(intersect, is_intersect)
    B_z = take_mask(B_z, is_intersect)
    assert intersect.shape == is_intersect.shape == B_z.shape == (P * RA, N * degree)
    # The boolean masks is_bp1 and is_bp2 will encode whether a given entry in
    # intersect is a valid starting and ending bounce point, respectively.
    # Sign of derivative determines whether an intersect is a valid bounce point.
    # Need to include zero derivative intersects to compute the WFB
    # (world's fattest banana) orbit bounce integrals.
    is_bp1 = B_z <= 0
    is_bp2 = B_z >= 0
    # For correctness, it is necessary that the first intersect satisfies B_z <= 0.
    # That is, the pairs bp1[:, i] and bp2[:, i] are the boundaries of an
    # integral only if bp1[:, i] <= bp2[:, i].
    # Now, because B_z[:, i] <= 0 implies B_z[:, i + 1] >= 0 by continuity,
    # there can be at most one inversion, and if it exists, the inversion must be
    # at the first pair. To correct the inversion, it suffices to disqualify
    # the first intersect as a right bounce point.
    edge_case = (B_z[:, 0] == 0) & (B_z[:, 1] < 0)
    is_bp2 = put_along_axis(is_bp2, jnp.array([0]), edge_case[:, jnp.newaxis], axis=-1)
    # Get ζ values of bounce points from the masks.
    bp1 = take_mask(intersect, is_bp1).reshape(P, RA, -1)
    bp2 = take_mask(intersect, is_bp2).reshape(P, RA, -1)
    return bp1, bp2
    # This is no longer implemented at the moment, but can be simply.
    #   If the first intersect satisfies B_z >= 0, that particle may be
    #   trapped in a well outside this snapshot of the field line.
    #   If, in addition, the last intersect satisfies B_z <= 0, then we have the
    #   required information to compute a bounce integral between these points.
    #   This single bounce integral is somewhat undefined since the field typically
    #   does not close on itself, but in some cases it can make sense to include it.
    #   (To make this integral well-defined, an approximation is made that the field
    #   line is periodic such that ζ = knots[-1] can be interpreted as ζ = 0 so
    #   that the distance between these bounce points is well-defined. This is fine
    #   as long as after a transit the field line begins physically close to where
    #   it began on the previous transit, for then continuity of |B| implies
    #   |B|(knots[-1] < ζ < knots[-1] + knots[0]) is close to |B|(0 < ζ < knots[0])).
    #   We don't need to check the conditions for the latter, because if they are
    #   not satisfied, the quadrature will evaluate √(1 − λ |B|) as nan, as desired.


def _compute_bp_if_given_pitch(pitch, knots, poly_B, poly_B_z, *original, err=False):
    """Return the ingredients needed by the ``bounce_integral`` function.

    Parameters
    ----------
    pitch : ndarray, shape(P, R, A)
        λ values.
        Last two axes should specify the λ value for a particular field line
        parameterized by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., ρ, α]``
        where in the latter the labels are interpreted as indices that correspond
        to that field line.
        If an additional axis exists on the left, it is the batch axis as usual.
    knots : ndarray, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray, shape(poly_B.shape[0], R * A, knots.size - 1)
        Polynomial coefficients of the spline of |B| in local power basis.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.
    poly_B_z : ndarray, shape(poly_B_z.shape[0], R * A, knots.size - 1)
        Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power basis.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.
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
        # ensure pitch has shape (batch size, rho.size, alpha.size)
        pitch = jnp.atleast_2d(pitch)
        if pitch.ndim == 2:
            # Can't use atleast_3d; see https://github.com/numpy/numpy/issues/25805.
            pitch = pitch[jnp.newaxis]
        err_msg = "Supplied invalid shape for pitch angles."
        assert pitch.ndim == 3, err_msg
        pitch = pitch.reshape(pitch.shape[0], -1)
        assert pitch.shape[-1] == 1 or pitch.shape[-1] == poly_B.shape[1], err_msg
        return pitch, *compute_bounce_points(pitch, knots, poly_B, poly_B_z)


def bounce_integral(
    eq,
    pitch=None,
    rho=jnp.linspace(1e-12, 1, 10),
    alpha=None,
    zeta=jnp.linspace(0, 6 * jnp.pi, 20),
    quadrature=tanh_sinh_quadrature,
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
        Equilibrium on which the bounce integral is defined.
    pitch : ndarray
        λ values to evaluate the bounce integral at each field line.
        May be specified later.
        Last two axes should specify the λ value for a particular field line
        parameterized by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., ρ, α]``
        where in the latter the labels are interpreted as indices that correspond
        to that field line.
        If an additional axis exists on the left, it is the batch axis as usual.
    rho : ndarray
        Unique flux surface label coordinates.
    alpha : ndarray
        Unique field line label coordinates over a constant rho surface.
    zeta : ndarray
        A spline of the integrand is computed at these values of the field
        line following coordinate, for every field line in the meshgrid formed from
        rho and alpha specified above.
        The number of knots specifies the grid resolution as increasing the
        number of knots increases the accuracy of representing the integrand
        and the accuracy of the locations of the bounce points.
    quadrature : callable
        The quadrature scheme used to evaluate the integral.
        Should return quadrature points and weights when called.
        The returned points should be within the domain [-1, 1].
    kwargs : dict
        Can specify arguments to the quadrature function with kwargs if convenient.
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
                Dictionary of ndarrays of stuff evaluated on ``grid``.
            poly_B : ndarray, shape(4, R * A, zeta.size - 1)
                Polynomial coefficients of the spline of |B| in local power basis.
                First axis should iterate through coefficients of power series,
                and the last axis should iterate through the piecewise
                polynomials of a particular spline of |B| along field line.
            poly_B_z : ndarray, shape(3, R * A, zeta.size - 1)
                Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power
                basis. First axis should iterate through coefficients of power series,
                and the last axis should iterate through the piecewise
                polynomials of a particular spline of |B| along field line.

    Examples
    --------
    .. code-block:: python

        rho = jnp.linspace(1e-12, 1, 6)
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 5)
        bi, items = bounce_integral(eq, rho=rho, alpha=alpha, return_items=True)
        name = "g_zz"
        f = eq.compute(name, grid=items["grid"], data=items["data"])[name]
        B = items["data"]["B"].reshape(rho.size * alpha.size, -1)
        pitch = jnp.linspace(1 / B.max(axis=-1), 1 / B.min(axis=-1), 30).reshape(
            -1, rho.size, alpha.size
        )
        result = bi(f, pitch)

    """
    if alpha is None:
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 10)
    rho = jnp.atleast_1d(rho)
    alpha = jnp.atleast_1d(alpha)
    zeta = jnp.atleast_1d(zeta)
    R = rho.size
    A = alpha.size

    grid, data = desc_grid_from_field_line_coords(eq, rho, alpha, zeta)
    data = eq.compute(["B^zeta", "|B|", "|B|_z|r,a"], grid=grid, data=data)
    B_sup_z = data["B^zeta"].reshape(R * A, -1)
    B = data["|B|"].reshape(R * A, -1)
    B_z_ra = data["|B|_z|r,a"].reshape(R * A, -1)
    poly_B = CubicHermiteSpline(zeta, B, B_z_ra, axis=-1, check=False).c
    poly_B = jnp.moveaxis(poly_B, 1, -1)
    poly_B_z = polyder(poly_B)
    assert poly_B.shape == (4, R * A, zeta.size - 1)
    assert poly_B_z.shape == (3, R * A, zeta.size - 1)

    return_items = kwargs.pop("return_items", False)
    x, w = quadrature(**kwargs)
    # change of variable, x = sin([0.5 + (ζ − ζ_b₂)/(ζ_b₂−ζ_b₁)] π)
    x = jnp.arcsin(x) / jnp.pi - 0.5
    original = _compute_bp_if_given_pitch(pitch, zeta, poly_B, poly_B_z, err=False)

    def _bounce_integral(f, pitch=None):
        """Compute the bounce integral of ``f``.

        Parameters
        ----------
        f : ndarray
            Quantity to compute the bounce integral of.
        pitch : ndarray
            λ values to evaluate the bounce integral at each field line.
            If None, uses the values given to the parent function.
            Last two axes should specify the λ value for a particular field line
            parameterized by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., ρ, α]``
            where in the latter the labels are interpreted as indices that correspond
            to that field line.
            If an additional axis exists on the left, it is the batch axis as usual.

        Returns
        -------
        result : ndarray, shape(P, rho.size, alpha.size, (zeta.size - 1) * 3)
            The last axis iterates through every bounce integral performed
            along that field line padded by nan.

        """
        pitch, bp1, bp2 = _compute_bp_if_given_pitch(
            pitch, zeta, poly_B, poly_B_z, *original, err=True
        )
        P = pitch.shape[0]
        pitch = jnp.broadcast_to(pitch, shape=(P, R * A))
        X = x * (bp2 - bp1)[..., jnp.newaxis] + bp2[..., jnp.newaxis]
        f = f.reshape(R * A, zeta.size)
        result = jnp.reshape(
            bounce_quadrature(pitch, X, w, zeta, f, B_sup_z, B, B_z_ra)
            # complete the change of variable
            / (bp2 - bp1) * jnp.pi,
            newshape=(P, R, A, -1),
        )
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
    quadrature=tanh_sinh_quadrature,
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
        Equilibrium on which the bounce average is defined.
    pitch : ndarray
        λ values to evaluate the bounce average at each field line.
        May be specified later.
        Last two axes should specify the λ value for a particular field line
        parameterized by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., ρ, α]``
        where in the latter the labels are interpreted as indices that correspond
        to that field line.
        If an additional axis exists on the left, it is the batch axis as usual.
    rho : ndarray
        Unique flux surface label coordinates.
    alpha : ndarray
        Unique field line label coordinates over a constant rho surface.
    zeta : ndarray
        A spline of the integrand is computed at these values of the field
        line following coordinate, for every field line in the meshgrid formed from
        rho and alpha specified above.
        The number of knots specifies the grid resolution as increasing the
        number of knots increases the accuracy of representing the integrand
        and the accuracy of the locations of the bounce points.
        If an integer is given, that many knots are linearly spaced from 0 to 10 pi.
    quadrature : callable
        The quadrature scheme used to evaluate the integral.
        Should return quadrature points and weights when called.
        The returned points should be within the domain [-1, 1].
        Can specify arguments to this callable with kwargs if convenient.
    kwargs : dict
        Can specify arguments to the quadrature function with kwargs if convenient.
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
                Dictionary of ndarrays of stuff evaluated on ``grid``.
            poly_B : ndarray, shape(4, R * A, zeta.size - 1)
                Polynomial coefficients of the spline of |B| in local power basis.
                First axis should iterate through coefficients of power series,
                and the last axis should iterate through the piecewise
                polynomials of a particular spline of |B| along field line.
            poly_B_z : ndarray, shape(3, R * A, zeta.size - 1)
                Polynomial coefficients of the spline of ∂|B|/∂_ζ in local power
                basis. First axis should iterate through coefficients of power series,
                and the last axis should iterate through the piecewise
                polynomials of a particular spline of |B| along field line.

    Examples
    --------
    .. code-block:: python

        rho = jnp.linspace(1e-12, 1, 6)
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 5)
        ba, items = bounce_average(eq, rho=rho, alpha=alpha, return_items=True)
        name = "g_zz"
        f = eq.compute(name, grid=items["grid"], data=items["data"])[name]
        B = items["data"]["B"].reshape(rho.size * alpha.size, -1)
        pitch = jnp.linspace(1 / B.max(axis=-1), 1 / B.min(axis=-1), 30).reshape(
            -1, rho.size, alpha.size
        )
        result = ba(f, pitch)

    """

    def _bounce_average(f, pitch=None):
        """Compute the bounce average of ``f``.

        Parameters
        ----------
        f : ndarray
            Quantity to compute the bounce average of.
        pitch : ndarray
            λ values to evaluate the bounce average at each field line.
            If None, uses the values given to the parent function.
            Last two axes should specify the λ value for a particular field line
            parameterized by ρ, α. That is, λ(ρ, α) is specified by ``pitch[..., ρ, α]``
            where in the latter the labels are interpreted as indices that correspond
            to that field line.
            If an additional axis exists on the left, it is the batch axis as usual.

        Returns
        -------
        result : ndarray, shape(P, rho.size, alpha.size, (zeta.size - 1) * 3)
            The last axis iterates through every bounce average performed
            along that field line padded by nan.

        """
        # Should be fine to fit akima spline to constant function 1 since
        # akima suppresses oscillation of the spline.
        return bi(f, pitch) / bi(jnp.ones_like(f), pitch)

    bi = bounce_integral(eq, pitch, rho, alpha, zeta, quadrature, **kwargs)
    if kwargs.get("return_items"):
        bi, items = bi
        return _bounce_average, items
    else:
        return _bounce_average
