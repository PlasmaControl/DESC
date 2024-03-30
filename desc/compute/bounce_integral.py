"""Methods for computing bounce integrals."""

from functools import partial

from interpax import CubicHermiteSpline, interp1d

from desc.backend import complex_sqrt, flatnonzero, jnp, put, put_along_axis, take, vmap
from desc.grid import Grid, LinearGrid, _meshgrid_expand

from .data_index import data_index


# vmap to compute a bounce integral for every pitch along every field line.
@partial(vmap, in_axes=(1, 1, None, None, 0, 0, 0, 0), out_axes=1)
def bounce_quadrature(pitch, X, w, knots, f, B_sup_z, B, B_z_ra):
    """Compute a bounce integral for every pitch along a particular field line.

    Parameters
    ----------
    pitch : ndarray, shape(pitch.size, )
        λ values.
    X : ndarray, shape(pitch.size, (knots.size - 1) * 3, w.size)
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
    inner_product : ndarray, shape(P, (knots.size - 1) * 3)
        Bounce integrals for every pitch along a particular field line.

    """
    assert pitch.ndim == 1
    assert X.shape == (pitch.size, (knots.size - 1) * 3, w.size)
    assert knots.shape == f.shape == B_sup_z.shape == B.shape == B_z_ra.shape
    # Cubic spline the integrand so that we can evaluate it at quadrature points
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
def take_mask(a, mask, size=None, fill_value=jnp.nan):
    """JIT compilable method to return ``a[mask][:size]`` padded by ``fill_value``.

    Parameters
    ----------
    a : ndarray
        The source array.
    mask : ndarray
        Boolean mask to index into ``a``. Should have same size as ``a``.
    size :
        Elements of ``a`` at the first size True indices of ``mask`` will be returned.
        If there are fewer elements than size indicates, the returned array will be
        padded with fill_value. Defaults to ``a.size``.
    fill_value :
        When there are fewer than the indicated number of elements,
        the remaining elements will be filled with ``fill_value``.

    Returns
    -------
    a_mask : ndarray, shape(size, )
        Output array.

    """
    if size is None:
        size = a.size
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


@vmap
def _last_value(a):
    """Assuming a is padded with nan at the right, return the last non-nan value."""
    assert a.ndim == 1
    a = a[::-1]
    idx = flatnonzero(~jnp.isnan(a), size=1, fill_value=0)
    return a[idx]


@vmap
def _roll_and_replace(a, shift, replacement):
    assert a.ndim == 1
    assert shift.size == 1 and shift.dtype == bool
    assert replacement.size == 1
    # maybe jax will prefer this to an if statement
    replacement = replacement * shift + a[0] * (~shift)
    a = put(jnp.roll(a, shift), jnp.array([0]), replacement)
    return a


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


def cubic_poly_roots(coef, k=0, a_min=None, a_max=None, sort=False):
    """Roots of cubic polynomial with given coefficients.

    Parameters
    ----------
    coef : ndarray
        First axis should store coefficients of a polynomial. For a polynomial
        given by c₁ x³ + c₂ x² + c₃ x + c₄, ``coef[i]`` should store cᵢ.
        It is assumed that c₁ is nonzero.
    k : ndarray
        Specify to find solutions to c₁ x³ + c₂ x² + c₃ x + c₄ = ``k``.
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
    roots : ndarray
        The roots of the cubic polynomial.
        The three roots are iterated over the last axis.

    """
    # https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
    # The common libraries use root-finding which isn't JIT compilable.
    clip = not (a_min is None and a_max is None)
    if a_min is None:
        a_min = -jnp.inf
    if a_max is None:
        a_max = jnp.inf

    a, b, c, d = coef
    d = d - k
    t_0 = b**2 - 3 * a * c
    t_1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d
    C = ((t_1 + complex_sqrt(t_1**2 - 4 * t_0**3)) / 2) ** (1 / 3)
    is_zero = jnp.isclose(C, 0)

    def compute_root(xi):
        t_2 = jnp.where(is_zero, 0, t_0 / (xi * C))
        return -(b + xi * C + t_2) / (3 * a)

    def clip_to_nan(root):
        return jnp.where(
            jnp.isreal(root) & (a_min <= root) & (root <= a_max),
            jnp.real(root),
            jnp.nan,
        )

    xi_1 = (-1 + (-3) ** 0.5) / 2
    xi_2 = xi_1**2
    xi_3 = 1
    roots = tuple(map(compute_root, (xi_1, xi_2, xi_3)))
    if clip:
        roots = tuple(map(clip_to_nan, roots))
    roots = jnp.stack(roots, axis=-1)
    if sort:
        roots = jnp.sort(roots, axis=-1)
    return roots


def compute_bounce_points(pitch, knots, poly_B, poly_B_z):
    """Compute the bounce points given |B| and pitch λ.

    Parameters
    ----------
    pitch : ndarray, shape(P, A * R)
        λ values.
        Last two axes should specify the λ value for a particular field line
        parameterized by α, ρ. That is, λ(α, ρ) is specified by ``pitch[..., α, ρ]``
        where in the latter the labels are interpreted as indices that correspond
        to that field line.
        If an additional axis exists on the left, it is the batch axis as usual.
    knots : ndarray, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray, shape(4, A * R, knots.size - 1)
        Polynomial coefficients of the cubic spline of |B|.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.
    poly_B_z : ndarray, shape(3, A * R, knots.size - 1)
        Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.

    Returns
    -------
    bp1, bp2 : ndarray, ndarray
        Field line-following ζ coordinates of bounce points for a given pitch
        along a field line. Has shape (P, A * R, (knots.size - 1) * 3).
        If there were less than (knots.size - 1) * 3 bounce points along a
        field line, then the last axis is padded with nan.

    """
    P = pitch.shape[0]  # batch size
    AR = poly_B.shape[1]  # alpha.size * rho.size
    N = knots.size - 1  # number of piecewise cubic polynomials per field line
    assert poly_B.shape[-1] == poly_B_z.shape[-1] == N

    # The polynomials' intersection points with 1 / λ is given by ``intersect``.
    # In order to be JIT compilable, this must have a shape that accommodates the
    # case where each cubic polynomial intersects 1 / λ thrice.
    # nan values in ``intersect`` denote a polynomial has less than three intersects.
    intersect = cubic_poly_roots(
        coef=poly_B,
        k=jnp.expand_dims(1 / pitch, axis=-1),
        a_min=knots[:-1],
        a_max=knots[1:],
        sort=True,
    )
    assert intersect.shape == (P, AR, N, 3)

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    # Condense remaining axes to vmap over them.
    B_z = polyval(x=intersect, c=poly_B_z[..., jnp.newaxis]).reshape(P * AR, -1)
    intersect = intersect.reshape(P * AR, -1)
    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = ~jnp.isnan(intersect)

    # Rearrange so that all intersects along a field line are contiguous.
    intersect = take_mask(intersect, is_intersect)
    B_z = take_mask(B_z, is_intersect)
    assert intersect.shape == B_z.shape == is_intersect.shape == (P * AR, N * 3)
    # The boolean masks is_bp1 and is_bp2 will encode whether a given entry in
    # intersect is a valid starting and ending bounce point, respectively.
    # Sign of derivative determines whether an intersect is a valid bounce point.
    is_bp1 = B_z < 0
    is_bp2 = B_z >= 0
    # Get ζ values of bounce points from the masks.
    bp1 = take_mask(intersect, is_bp1)
    bp2 = take_mask(intersect, is_bp2)
    # For correctness, it is necessary that the first intersect satisfies B_z <= 0.
    # That is, the pairs bp1[:, i] and bp2[:, i] are the boundaries of an
    # integral only if bp1[:, i] <= bp2[:, i].
    # Now, because B_z[:, i] <= 0 implies B_z[:, i + 1] >= 0 by continuity,
    # there can be at most one inversion, and if it exists, the inversion must be
    # at the first pair. To correct the inversion, it suffices to roll forward bp1.
    # Then the pairs bp1[:, i] and bp2[:, i] for i > 0 form integration boundaries.
    # Moreover, if the first intersect satisfies B_z >= 0, that particle may be
    # trapped in a well outside this snapshot of the field line.
    # If, in addition, the last intersect satisfies B_z < 0, then we have the
    # required information to compute a bounce integral between these points.
    # The below logic handles both tasks.
    last_intersect = jnp.squeeze(_last_value(intersect))
    bp1 = _roll_and_replace(bp1, bp1[:, 0] > bp2[:, 0], last_intersect - knots[-1])
    # Notice that for the latter, an "approximation" is made that the field line is
    # periodic such that ζ = knots[-1] can be interpreted as ζ = 0 so that the
    # distance between these bounce points is well-defined. This may worry the
    # reader if they recall that it is not desirable to have field lines close
    # on themselves. However, for any irrational value for the rotational
    # transform, there exists an arbitrarily close rational value (I'm just saying
    # the basic result that rational numbers are dense in the real numbers).
    # After such a rational amount of transits, the points corresponding to this
    # distance along the field line and the start of the field line will be
    # physically close. By continuity, the value of |B| at ζ = 0 is then close
    # to the value of |B| of at ζ = knots[-1]. In general, continuity implies
    # |B|(knots[-1] < ζ < knots[-1] + knots[0]) will approximately equal
    # |B|(0 < ζ < knots[0]) as long as ζ = knots[-1] is large enough.
    bp1 = bp1.reshape(P, AR, -1)
    bp2 = bp2.reshape(P, AR, -1)
    return bp1, bp2


def _compute_bp_if_given_pitch(pitch, knots, poly_B, poly_B_z, *original, err=False):
    """Return the ingredients needed by the ``bounce_integral`` function.

    Parameters
    ----------
    pitch : ndarray, shape(P, A, R)
        λ values.
        Last two axes should specify the λ value for a particular field line
        parameterized by α, ρ. That is, λ(α, ρ) is specified by ``pitch[..., α, ρ]``
        where in the latter the labels are interpreted as indices that correspond
        to that field line.
        If an additional axis exists on the left, it is the batch axis as usual.
    knots : ndarray, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray, shape(4, A * R, knots.size - 1)
        Polynomial coefficients of the cubic spline of |B|.
        First axis should iterate through coefficients of power series,
        and the last axis should iterate through the piecewise
        polynomials of a particular spline of |B| along field line.
    poly_B_z : ndarray, shape(3, A * R, knots.size - 1)
        Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.
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
        # ensure pitch has shape (batch size, alpha.size, rho.size)
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
    rho=None,
    alpha=None,
    zeta=20,
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
        parameterized by α, ρ. That is, λ(α, ρ) is specified by ``pitch[..., α, ρ]``
        where in the latter the labels are interpreted as indices that correspond
        to that field line.
        If an additional axis exists on the left, it is the batch axis as usual.
    rho : ndarray or float
        Unique flux surface label coordinates.
    alpha : ndarray or float
        Unique field line label coordinates over a constant rho surface.
    zeta : ndarray or int
        A cubic spline of the integrand is computed at these values of the field
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
            poly_B : ndarray, shape(4, A * R, zeta.size - 1)
                Polynomial coefficients of the cubic spline of |B|.
                First axis should iterate through coefficients of power series,
                and the last axis should iterate through the piecewise
                polynomials of a particular spline of |B| along field line.
            poly_B_z : ndarray, shape(3, A * R, zeta.size - 1)
                Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.
                First axis should iterate through coefficients of power series,
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
        B = items["data"]["B"].reshape(alpha.size * rho.size, -1)
        pitch = jnp.linspace(1 / B.max(axis=-1), 1 / B.min(axis=-1), 30).reshape(
            -1, alpha.size, rho.size
        )
        result = bi(f, pitch)

    """
    if rho is None:
        rho = jnp.linspace(1e-12, 1, 10)
    if alpha is None:
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 10)
    rho = jnp.atleast_1d(rho)
    alpha = jnp.atleast_1d(alpha)
    zeta = jnp.atleast_1d(zeta)
    if zeta.size == 1:
        zeta = jnp.linspace(0, 10 * jnp.pi, zeta.item())
    R = rho.size
    A = alpha.size

    grid, data = field_line_to_desc_coords(eq, rho, alpha, zeta)
    data = eq.compute(["B^zeta", "|B|", "|B|_z|r,a"], grid=grid, data=data)
    B_sup_z = data["B^zeta"].reshape(A * R, -1)
    B = data["|B|"].reshape(A * R, -1)
    B_z_ra = data["|B|_z|r,a"].reshape(A * R, -1)
    poly_B = CubicHermiteSpline(zeta, B, B_z_ra, axis=-1, check=False).c
    poly_B = jnp.moveaxis(poly_B, 1, -1)
    poly_B_z = polyder(poly_B)
    assert poly_B.shape == (4, A * R, zeta.size - 1)
    assert poly_B_z.shape == (3, A * R, zeta.size - 1)

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
            parameterized by α, ρ. That is, λ(α, ρ) is specified by ``pitch[..., α, ρ]``
            where in the latter the labels are interpreted as indices that correspond
            to that field line.
            If an additional axis exists on the left, it is the batch axis as usual.

        Returns
        -------
        result : ndarray, shape(P, alpha.size, rho.size, (zeta.size - 1) * 3)
            The last axis iterates through every bounce integral performed
            along that field line padded by nan.

        """
        pitch, bp1, bp2 = _compute_bp_if_given_pitch(
            pitch, zeta, poly_B, poly_B_z, *original, err=True
        )
        P = pitch.shape[0]
        pitch = jnp.broadcast_to(pitch, shape=(P, A * R))
        X = x * (bp2 - bp1)[..., jnp.newaxis] + bp2[..., jnp.newaxis]
        f = f.reshape(A * R, -1)
        result = jnp.reshape(
            bounce_quadrature(pitch, X, w, zeta, f, B_sup_z, B, B_z_ra)
            # complete the change of variable
            / (bp2 - bp1) * jnp.pi,
            newshape=(P, A, R, -1),
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
    rho=None,
    alpha=None,
    zeta=20,
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
        parameterized by α, ρ. That is, λ(α, ρ) is specified by ``pitch[..., α, ρ]``
        where in the latter the labels are interpreted as indices that correspond
        to that field line.
        If an additional axis exists on the left, it is the batch axis as usual.
    rho : ndarray or float
        Unique flux surface label coordinates.
    alpha : ndarray or float
        Unique field line label coordinates over a constant rho surface.
    zeta : ndarray or int
        A cubic spline of the integrand is computed at these values of the field
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
            poly_B : ndarray, shape(4, A * R, zeta.size - 1)
                Polynomial coefficients of the cubic spline of |B|.
                First axis should iterate through coefficients of power series,
                and the last axis should iterate through the piecewise
                polynomials of a particular spline of |B| along field line.
            poly_B_z : ndarray, shape(3, A * R, zeta.size - 1)
                Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.
                First axis should iterate through coefficients of power series,
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
        B = items["data"]["B"].reshape(alpha.size * rho.size, -1)
        pitch = jnp.linspace(1 / B.max(axis=-1), 1 / B.min(axis=-1), 30).reshape(
            -1, alpha.size, rho.size
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
            parameterized by α, ρ. That is, λ(α, ρ) is specified by ``pitch[..., α, ρ]``
            where in the latter the labels are interpreted as indices that correspond
            to that field line.
            If an additional axis exists on the left, it is the batch axis as usual.

        Returns
        -------
        result : ndarray, shape(P, alpha.size, rho.size, (zeta.size - 1) * 3)
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


def field_line_to_desc_coords(eq, rho, alpha, zeta, jitable=True):
    """Get DESC grid from unique field line coordinates."""
    r, a, z = jnp.meshgrid(rho, alpha, zeta, indexing="ij")
    r, a, z = r.ravel(), a.ravel(), z.ravel()
    # Map these Clebsch-Type field-line coordinates to DESC coordinates.
    # Note that the rotational transform can be computed apriori because it is a single
    # variable function of rho, and the coordinate mapping does not change rho. Once
    # this is known, it is simple to compute theta_PEST from alpha. Then we transform
    # from straight field-line coordinates to DESC coordinates with the method
    # compute_theta_coords. This is preferred over transforming from Clebsch-Type
    # coordinates to DESC coordinates directly with the more general method
    # map_coordinates. That method requires an initial guess to be compatible with JIT,
    # and generating a reasonable initial guess requires computing the rotational
    # transform to approximate theta_PEST and the poloidal stream function anyway.
    # TODO: map coords recently updated, so maybe just switch to that
    lg = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
    lg_data = eq.compute(["iota", "iota_r", "iota_rr"], grid=lg)
    data = {
        d: _meshgrid_expand(lg.compress(lg_data[d]), rho.size, alpha.size, zeta.size)
        for d in lg_data
        if data_index["desc.equilibrium.equilibrium.Equilibrium"][d]["coordinates"]
        == "r"
    }
    sfl_coords = jnp.column_stack([r, a + data["iota"] * z, z])
    desc_coords = eq.compute_theta_coords(sfl_coords)
    grid = Grid(desc_coords, jitable=jitable)
    return grid, data


# Current algorithm used for bounce integrals no longer requires these
# two functions. TODO: Delete before merge.
def diff_mask(a, mask, n=1, axis=-1, prepend=None):
    """Calculate the n-th discrete difference along the given axis of ``a[mask]``.

    The first difference is given by ``out[i] = a[i+1] - a[i]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively. This method is JIT compatible.

    Parameters
    ----------
    a : array_like
        Input array
    mask : array_like
        Boolean mask to index like ``a[mask]`` prior to computing difference.
        Should have same size as ``a``.
    n : int, optional
        The number of times values are differenced.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend : array_like, optional
        Values to prepend to `a` along axis prior to performing the difference.
        Scalar values are expanded to arrays with length 1 in the direction of
        axis and the shape of the input array in along all other axes.
        Otherwise, the dimension and shape must match `a` except along axis.

    Returns
    -------
    diff : ndarray
        The n-th differences. The shape of the output is the same as ``a``
        except along ``axis`` where the dimension is smaller by ``n``. The
        type of the output is the same as the type of the difference
        between any two elements of ``a``.

    Notes
    -----
    The result is padded with nan at the end to be jit compilable.

    """
    prepend = () if prepend is None else (prepend,)
    return jnp.diff(take_mask(a, mask, fill_value=jnp.nan), n, axis, *prepend)


def stretch_batches(in_arr, in_batch_size, out_batch_size, fill):
    """Stretch batches of ``in_arr``.

    Given that ``in_arr`` is composed of N batches of ``in_batch_size``
    along its last axis, stretch the last axis so that it is composed of
    N batches of ``out_batch_size``. The ``out_batch_size - in_batch_size``
    missing elements in each batch are populated with ``fill``.
    By default, these elements are populated evenly surrounding the input batches.

    Parameters
    ----------
    in_arr : ndarray, shape(..., in_batch_size * N)
        Input array
    in_batch_size : int
        Length of batches along last axis of input array.
    out_batch_size : int
        Length of batches along last axis of output array.
    fill : bool or int or float
        Value to fill at missing indices of each batch.

    Returns
    -------
    out_arr : ndarray, shape(..., out_batch_size * N)
        Output array

    """
    assert out_batch_size >= in_batch_size
    N = in_arr.shape[-1] // in_batch_size
    out_shape = in_arr.shape[:-1] + (N * out_batch_size,)
    offset = (out_batch_size - in_batch_size) // 2
    idx = jnp.arange(in_arr.shape[-1])
    out_arr = put_along_axis(
        arr=jnp.full(out_shape, fill, dtype=in_arr.dtype),
        indices=(idx // in_batch_size) * out_batch_size
        + offset
        + (idx % in_batch_size),
        values=in_arr,
        axis=-1,
    )
    return out_arr
