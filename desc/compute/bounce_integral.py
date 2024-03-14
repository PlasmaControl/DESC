"""Methods for computing bounce integrals."""

from interpax import Akima1DInterpolator, CubicHermiteSpline

from desc.backend import complex_sqrt, flatnonzero, jnp, put_along_axis, vmap
from desc.compute.utils import mask_diff, mask_take, safediv
from desc.grid import Grid, LinearGrid, _meshgrid_expand

from .data_index import data_index

NUM_ROOTS = 3  # max number of roots of a cubic polynomial
# returns index of first nonzero element in a
v_first_flatnonzero = vmap(lambda a: flatnonzero(a, size=1, fill_value=a.size))
v_mask_diff = vmap(mask_diff)
v_mask_take = vmap(lambda a, mask: mask_take(a, mask, size=a.size, fill_value=jnp.nan))


def _inner_product(knots, pitch, w, X, f, B_sup_z, B, B_z_ra):
    """Compute bounce integrals for every pitch along a particular field line.

    Parameters
    ----------
    knots : ndarray, shape(knots.size, )
        Field line-following ζ coordinates of spline knots.
    pitch : ndarray, shape(pitch.size, )
        λ values.
    w : ndarray, shape(w.size, )
        Quadrature weights.
    X : ndarray, shape(pitch.size, X.shape[1], w.size)
        Quadrature points.
    f : ndarray, shape(knots.size, )
        Spline of function to compute bounce integral of.
    B_sup_z : ndarray, shape(knots.size, )
        Contravariant field-line following toroidal component of magnetic field.
    B : ndarray, shape(knots.size, )
        Norm of magnetic field.
    B_z_ra : ndarray, shape(knots.size, )
        Norm of magnetic field derivative with respect to field-line following label.

    Returns
    -------
    inner_product : ndarray, shape(pitch.size, X.shape[1])
        Bounce integrals for every pitch along a particular field line.

    """
    assert X.shape == (pitch.size, X.shape[1], w.size)
    # FIXME: https://github.com/f0uriest/interpax/issues/26
    f = Akima1DInterpolator(knots, f, check=False)
    B_sup_z = Akima1DInterpolator(knots, B_sup_z, check=False)
    B = CubicHermiteSpline(knots, B, B_z_ra, check=False)
    pitch = pitch[:, jnp.newaxis, jnp.newaxis]
    return jnp.dot(f(X) / (B_sup_z(X) * jnp.sqrt(1 - pitch * B(X))), w)


"""Compute bounce integrals for every pitch along every field line."""
_compute_quad = vmap(_inner_product, in_axes=(None, None, None, 1, 0, 0, 0, 0))


def tanh_sinh_quadrature(resolution):
    """
    tanh_sinh quadrature.

    This function outputs the quadrature points and weights
    for a tanh-sinh quadrature.

    ∫₋₁¹ f(x) dx = ∑ₖ wₖ f(xₖ)

    Parameters
    ----------
    resolution: int
        Number of quadrature points, preferably odd

    Returns
    -------
    x : numpy array
        Quadrature points
    w : numpy array
        Quadrature weights

    """
    # https://github.com/f0uriest/quadax/blob/main/quadax/utils.py#L166
    # x = 1 - eps with some buffer
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


def polyint(c, k=jnp.array([0])):
    """Coefficients for the primitives of the given set of polynomials.

    Parameters
    ----------
    c : ndarray
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.
    k : ndarray
        Integration constants.

    Returns
    -------
    poly : ndarray
        Coefficients of polynomial primitive.
        That is, ``poly[i]`` stores the coefficient of the monomial xⁿ⁻ⁱ⁺¹,
        where n is ``c.shape[0] - 1``.

    """
    poly = (c.T / jnp.arange(c.shape[0], 0, -1)).T
    poly = jnp.append(poly, jnp.broadcast_to(k, c.shape[1:])[jnp.newaxis], axis=0)
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

    Parameters
    ----------
    x : ndarray
        Coordinates at which to evaluate the set of polynomials.
        The first ``c.ndim`` axes should have shape ``c.shape[1:]``.
    c : ndarray
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.

    Returns
    -------
    val : ndarray
        ``val[j, k, ...]`` is the polynomial with coefficients ``c[:, j, k, ...]``
        evaluated at the point ``x[j, k, ...]``.

    Notes
    -----
    This function does not perform the same operation as
    ``np.polynomial.polynomial.polyval(x, c)``.
    An example usage of this function is shown in
    tests/test_compute_utils.py::TestComputeUtils::test_polyval.

    """
    X = x[..., jnp.newaxis] ** jnp.arange(c.shape[0] - 1, -1, -1)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    s = alphabet[: c.ndim]
    val = jnp.einsum(f"{s},{s[1:]}...{s[0]}->{s[1:]}...", c, X)
    return val


def cubic_poly_roots(coef, k=jnp.array([0]), a_min=None, a_max=None, sort=False):
    """Roots of cubic polynomial.

    Parameters
    ----------
    coef : ndarray
        First axis should store coefficients of a polynomial. For a polynomial
        given by c₁ x³ + c₂ x² + c₃ x + c₄, ``coef[i]`` should store cᵢ.
        It is assumed that c₁ is nonzero.
    k : ndarray, shape(k.size, )
        Specify to instead find solutions to c₁ x³ + c₂ x² + c₃ x + c₄ = ``k``.
    a_min, a_max : ndarray
        Minimum and maximum value to return roots between.
        If specified only real roots are returned.
        If None, returns all complex roots.
        Both arrays are broadcast against arrays of shape ``coef.shape[1:]``.
    sort : bool
        Whether to sort the roots.

    Returns
    -------
    roots : ndarray, shape(k.size, coef.shape, 3)
        If ``k`` has one element, the first axis will be squeezed out.
        The roots of the cubic polynomial.

    """
    # https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
    # The common libraries use root-finding which isn't compatible with JAX.
    clip = not (a_min is None and a_max is None)
    if a_min is None:
        a_min = -jnp.inf
    if a_max is None:
        a_max = jnp.inf

    a, b, c, d = coef
    d = jnp.squeeze(jnp.moveaxis(d[..., jnp.newaxis] - k, -1, 0))
    t_0 = b**2 - 3 * a * c
    t_1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d
    C = ((t_1 + complex_sqrt(t_1**2 - 4 * t_0**3)) / 2) ** (1 / 3)
    C_is_zero = jnp.isclose(C, 0)

    def compute_root(xi):
        return -(b + xi * C + jnp.where(C_is_zero, 0, t_0 / (xi * C))) / (3 * a)

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


# TODO: fix up some logic with the periodic boundary bounce integral thing
def _compute_bp(pitch, knots, poly_B, poly_B_z):
    """Compute the bounce points given |B| and pitch λ.

    Parameters
    ----------
    pitch : ndarray
        λ values.
    knots : ndarray
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray
        Polynomial coefficients of the cubic spline of |B|.
    poly_B_z : ndarray
        Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.

    Returns
    -------
    bp1, bp2 : ndarray, ndarray
        Field line-following ζ coordinates of bounce points for a given pitch
        along a field line. Has shape (pitch.size, M * L, N * NUM_ROOTS).
        If there were less than N * NUM_ROOTS bounce points along a field line,
        then the last axis is padded with nan.

    """
    ML = jnp.array(poly_B.shape[1:-1]).prod()
    N = poly_B.shape[-1]

    # The polynomials' intersection points with 1 / λ is given by ``intersect``.
    # In order to be JIT compilable, this must have a shape that accommodates the
    # case where each cubic polynomial intersects 1 / λ thrice.
    # nan values in ``intersect`` denote a polynomial has less than three intersects.
    intersect = cubic_poly_roots(
        coef=poly_B, k=1 / pitch, a_min=knots[:-1], a_max=knots[1:], sort=True
    ).reshape(pitch.size, ML, N, NUM_ROOTS)
    # Reshape to unsqueeze pitch axis.

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    # Condense the first and second axes to vmap over them.
    B_z = polyval(intersect, poly_B_z[:, jnp.newaxis]).reshape(
        pitch.size * ML, N * NUM_ROOTS
    )
    intersect = intersect.reshape(pitch.size * ML, N * NUM_ROOTS)
    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = ~jnp.isnan(intersect)

    # Rearrange so that all intersects along a field line are contiguous.
    intersect = v_mask_take(intersect, is_intersect)
    B_z = v_mask_take(B_z, is_intersect)
    # The boolean masks ``bp1`` and ``bp2`` will encode whether a given entry in
    # ``intersect`` is a valid starting and ending bounce point, respectively.
    # Sign of derivative determines whether an intersect is a valid bounce point.
    bp1 = B_z <= 0
    bp2 = B_z >= 0
    # B_z <= 0 at intersect i implies B_z >= 0 at intersect i+1 by continuity.

    #   extend bp1 and bp2 by single element and then test
    # # index of last intersect along a field line
    # idx = jnp.squeeze(v_first_flatnonzero(~is_intersect)) - 1  # noqa: E800
    # assert idx.shape == (pitch.size * ML,)  # noqa: E800
    # Consider the boundary to be periodic to compute bounce integrals of
    # particles trapped outside this snapshot of the field lines.
    # Roll such that first intersect is moved to index of last intersect.

    # Get ζ values of bounce points from the masks.
    bp1 = v_mask_take(intersect, bp1).reshape(pitch.size, ML, N * NUM_ROOTS)
    bp2 = v_mask_take(intersect, bp2).reshape(pitch.size, ML, N * NUM_ROOTS)
    return bp1, bp2


def _compute_bp_include_knots(pitch, knots, poly_B, poly_B_z):
    """Compute the bounce points given |B| and pitch λ.

    Like ``_compute_bp`` but returns ingredients needed by the
    algorithm in the direct method in ``bounce_integral``.

    Parameters
    ----------
    pitch : ndarray
        λ values.
    knots : ndarray
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray
        Polynomial coefficients of the cubic spline of |B|.
    poly_B_z : ndarray
        Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.

    Returns
    -------
    intersect_nan_to_right_knot, is_intersect, is_bp
        The boolean mask ``is_bp`` encodes whether a given pair of intersects
        are the endpoints of a bounce integral.

    """
    ML = jnp.array(poly_B.shape[1:-1]).prod()
    N = poly_B.shape[-1]
    a_min = knots[:-1]
    a_max = knots[1:]

    # The polynomials' intersection points with 1 / λ is given by ``roots``.
    # In order to be JIT compilable, this must have a shape that accommodates the
    # case where each cubic polynomial intersects 1 / λ thrice.
    # nan values in ``roots`` denote a polynomial has less than three intersects.
    roots = cubic_poly_roots(
        coef=poly_B, k=1 / pitch, a_min=a_min, a_max=a_max, sort=True
    ).reshape(pitch.size, ML, N, NUM_ROOTS)
    # Reshape to unsqueeze pitch axis.

    # Include the knots of the splines along with the intersection points.
    # This preprocessing makes the ``direct`` algorithm in ``bounce_integral`` simpler.
    roots = (roots[..., 0], roots[..., 1], roots[..., 2])
    a_min = jnp.broadcast_to(a_min, shape=(pitch.size, ML, N))
    a_max = jnp.broadcast_to(a_max, shape=(pitch.size, ML, N))
    intersect = jnp.stack((a_min, *roots, a_max), axis=-1)

    # Reshape so that last axis enumerates intersects of a pitch along a field line.
    # Condense the first and second axes to vmap over them.
    B_z = polyval(intersect, poly_B_z[:, jnp.newaxis]).reshape(
        pitch.size * ML, N * (NUM_ROOTS + 2)
    )
    # Only consider intersect if it is within knots that bound that polynomial.
    is_intersect = jnp.reshape(
        jnp.array([False, True, True, True, False], dtype=bool) & ~jnp.isnan(intersect),
        newshape=(pitch.size * ML, N * (NUM_ROOTS + 2)),
    )

    # Rearrange so that all the intersects along field line are contiguous.
    B_z = v_mask_take(B_z, is_intersect)
    # The boolean masks ``bp1`` and ``bp2`` will encode whether a given entry in
    # ``intersect`` is a valid starting and ending bounce point, respectively.
    # Sign of derivative determines whether an intersect is a valid bounce point.
    bp1 = B_z <= 0
    bp2 = B_z >= 0
    # B_z <= 0 at intersect i implies B_z >= 0 at intersect i+1 by continuity.

    # index of last intersect
    idx = jnp.squeeze(v_first_flatnonzero(~is_intersect)) - 1
    assert idx.shape == (pitch.size * ML,)
    # Consider the boundary to be periodic to compute bounce integrals of
    # particles trapped outside this snapshot of the field lines.
    # Roll such that first intersect is moved to index of last intersect.
    is_bp = bp1 & put_along_axis(jnp.roll(bp2, -1, axis=-1), idx, bp2[..., 0], axis=-1)

    # Returning this makes the ``direct`` algorithm in ``bounce_integral`` simpler.
    # Replace nan values with right knots of the spline.
    intersect_nan_to_right_knot = jnp.stack(
        (a_min, *tuple(map(lambda r: jnp.where(jnp.isnan(r), a_max, r), roots)), a_max),
        axis=-1,
    ).reshape(pitch.size * ML, N, (NUM_ROOTS + 2))

    return intersect_nan_to_right_knot, is_intersect, is_bp


def _compute_bp_if_given_pitch(
    pitch, knots, poly_B, poly_B_z, compute_bp, *original, err=False
):
    """Return the ingredients needed by the ``bounce_integral`` function.

    Parameters
    ----------
    pitch : ndarray
        λ values.
        If None, returns the given ``original`` tuple.
    knots : ndarray
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray
        Polynomial coefficients of the cubic spline of |B|.
    poly_B_z : ndarray
        Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.
    compute_bp : callable
        Method to compute bounce points.
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
        pitch = jnp.atleast_1d(pitch)
        return pitch, *compute_bp(pitch, knots, poly_B, poly_B_z)


def bounce_integral(
    eq,
    rho=None,
    alpha=None,
    zeta_max=10 * jnp.pi,
    pitch=None,
    resolution=20,
    method="tanh_sinh",
):
    """Returns a method to compute the bounce integral of any quantity.

    The bounce integral is defined as F_ℓ(λ) = ∫ f(ℓ) / √(1 − λ |B|) dℓ, where
        dℓ parameterizes the distance along the field line,
        λ is a constant proportional to the magnetic moment over energy,
        |B| is the norm of the magnetic field,
        f(ℓ) is the quantity to integrate along the field line,
        and the endpoints of the integration are at the bounce points.
    For a particle with fixed λ, bounce points are defined to be the location
    on the field line such that the particle's velocity parallel to the
    magnetic field is zero, i.e. λ |B| = 1.

    The bounce integral is defined up to a sign.
    We choose the sign that corresponds the particle's guiding center trajectory
    traveling in the direction of increasing field-line-following label.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium on which the bounce integral is defined.
    rho : ndarray
        Unique flux surface label coordinates.
    alpha : ndarray
        Unique field line label coordinates over a constant rho surface.
    zeta_max : float
        Max value for field line following coordinate.
    pitch : ndarray
        λ values to evaluate the bounce integral at.
    resolution : int
        Number of interpolation points (knots) used for splines in the quadrature.
        A maximum of three bounce points can be detected in between knots.
        The accuracy of the quadrature will increase as some function of
        the number of knots over the number of detected bounce points.
        So for well-behaved magnetic fields increasing resolution should increase
        the accuracy of the quadrature.
    method : str
        The quadrature scheme used to evaluate the integral.
        The "direct" method exactly integrates a cubic spline of the integrand.
        The "tanh_sinh" method performs a Tanh-sinh quadrature, where independent cubic
        splines are used for components in the integrand so that the singularity near
        the bounce points can be captured more accurately than can be represented by a
        polynomial.

    Returns
    -------
    bi : callable
        This callable method computes the bounce integral F_ℓ(λ) for every
        specified field line ℓ (constant rho and alpha), for every λ value in
        ``pitch``.

    Examples
    --------
    .. code-block:: python

        bi = bounce_integral(eq)
        F = bi(name, pitch)

    """
    if rho is None:
        rho = jnp.linspace(0, 1, 10)
    if alpha is None:
        alpha = jnp.linspace(0, (2 - eq.sym) * jnp.pi, 20)
    rho = jnp.atleast_1d(rho)
    alpha = jnp.atleast_1d(alpha)
    knots = jnp.linspace(0, zeta_max, resolution)
    L = rho.size
    M = alpha.size
    N = resolution - 1  # number of piecewise cubic polynomials per field line

    grid, data = field_line_to_desc_coords(eq, rho, alpha, knots)
    data = eq.compute(["B^zeta", "|B|", "|B|_z|r,a"], grid=grid, data=data)
    B_sup_z = data["B^zeta"].reshape(M * L, resolution)
    B = data["|B|"].reshape(M * L, resolution)
    B_z_ra = data["|B|_z|r,a"].reshape(M * L, resolution)
    poly_B = CubicHermiteSpline(knots, B, B_z_ra, axis=-1, check=False).c
    poly_B = jnp.moveaxis(poly_B, 1, -1)
    poly_B_z = polyder(poly_B)
    assert poly_B.shape == (4, M * L, N)
    assert poly_B_z.shape == (3, M * L, N)

    def quad(name, pitch=None):
        """Compute the bounce integral of the named quantity.

        Parameters
        ----------
        name : ndarray
            Name of quantity in ``data_index`` to compute the bounce integral of.
        pitch : ndarray
            λ values to evaluate the bounce integral at.
            If None, uses the values given to the parent function.

        Returns
        -------
        F : ndarray, shape(pitch, alpha, rho, (resolution - 1) * 3)
            The last axis iterates through every bounce integral performed
            along that field line padded by nan.

        """
        pitch, bp1, bp2 = _compute_bp_if_given_pitch(
            pitch, knots, poly_B, poly_B_z, _compute_bp, *original, err=True
        )
        X = x * (bp2 - bp1)[..., jnp.newaxis] + bp2[..., jnp.newaxis]
        f = eq.compute(name, grid=grid, data=data)[name].reshape(M * L, resolution)
        F = jnp.reshape(
            _compute_quad(knots, pitch, w, X, f, B_sup_z, B, B_z_ra)
            * jnp.pi
            / (bp2 - bp1),
            newshape=(pitch.size, M, L, N * NUM_ROOTS),
        )
        return F

    def direct(name, pitch=None):
        """Compute the bounce integral of the named quantity.

        Parameters
        ----------
        name : ndarray
            Name of quantity in ``data_index`` to compute the bounce integral of.
        pitch : ndarray
            λ values to evaluate the bounce integral at.
            If None, uses the values given to the parent function.

        Returns
        -------
        F : ndarray, shape(pitch, alpha, rho, (resolution - 1) * 3)
            The last axis iterates through every bounce integral performed
            along that field line padded by nan.

        """
        (
            pitch,
            intersect_nan_to_right_knot,
            is_intersect,
            is_bp,
        ) = _compute_bp_if_given_pitch(
            pitch,
            knots,
            poly_B,
            poly_B_z,
            _compute_bp_include_knots,
            *original,
            err=True,
        )

        integrand = jnp.nan_to_num(
            eq.compute(name, grid=grid, data=data)[name]
            / (data["B^zeta"] * jnp.sqrt(1 - pitch[:, jnp.newaxis] * data["|B|"]))
        ).reshape(pitch.size * M * L, resolution)
        integrand = Akima1DInterpolator(knots, integrand, axis=-1, check=False).c
        integrand = jnp.moveaxis(integrand, 1, -1)
        assert integrand.shape == (4, pitch.size * M * L, N)
        # For this algorithm, computing integrals via differences of primitives
        # is preferable to any numerical quadrature. For example, even if the
        # intersection points were evenly spaced, a composite Simpson's quadrature
        # would require computing the spline on 1.8x more knots for the same accuracy.
        primitive = polyval(intersect_nan_to_right_knot, polyint(integrand)).reshape(
            pitch.size * M * L, N * (NUM_ROOTS + 2)
        )
        sums = jnp.cumsum(
            # Periodic boundary to compute bounce integrals of particles
            # trapped outside this snapshot of the field lines.
            jnp.diff(primitive, axis=-1, append=primitive[..., 0, jnp.newaxis])
            # We didn't enforce continuity of the piecewise primitives, so
            # multiply by mask that is false at shared knots of piecewise spline
            # to avoid adding difference between primitives of splines at knots.
            * jnp.append(
                jnp.arange(1, N * (NUM_ROOTS + 2)) % (NUM_ROOTS + 2) != 0, True
            ),
            axis=-1,
        )
        F = jnp.reshape(
            # Compute difference of ``sums`` between bounce points.
            v_mask_diff(v_mask_take(sums, is_intersect), is_bp)[..., : N * NUM_ROOTS],
            # Guaranteed to have at most N * NUM_ROOTS entries where is_bp is true.
            newshape=(pitch.size, M, L, N * NUM_ROOTS),
        )
        return F

    if method == "direct":
        original = _compute_bp_if_given_pitch(
            pitch, knots, poly_B, poly_B_z, _compute_bp_include_knots, err=False
        )
        return direct
    else:
        original = _compute_bp_if_given_pitch(
            pitch, knots, poly_B, poly_B_z, _compute_bp, err=False
        )
        x, w = tanh_sinh_quadrature(resolution)
        x = jnp.arcsin(x) / jnp.pi - 0.5
        return quad


def bounce_average(
    eq,
    rho=None,
    alpha=None,
    zeta_max=10 * jnp.pi,
    pitch=None,
    resolution=20,
    method="quad",
):
    """Returns a method to compute the bounce average of any quantity.

    The bounce average is defined as
    G_ℓ(λ) = (∫ g(ℓ) / √(1 − λ |B|) dℓ) / (∫ 1 / √(1 − λ |B|) dℓ), where
        dℓ parameterizes the distance along the field line,
        λ is a constant proportional to the magnetic moment over energy,
        |B| is the norm of the magnetic field,
        g(ℓ) is the quantity to integrate along the field line,
        and the endpoints of the integration are at the bounce points.
    For a particle with fixed λ, bounce points are defined to be the location
    on the field line such that the particle's velocity parallel to the
    magnetic field is zero, i.e. λ |B| = 1.

    The bounce integral is defined up to a sign.
    We choose the sign that corresponds the particle's guiding center trajectory
    traveling in the direction of increasing field-line-following label.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium on which the bounce average is defined.
    rho : ndarray
        Unique flux surface label coordinates.
    alpha : ndarray
        Unique field line label coordinates over a constant rho surface.
    zeta_max : float
        Max value for field line following coordinate.
    pitch : ndarray
        λ values to evaluate the bounce average at.
        Defaults to linearly spaced values between min and max of |B|.
    resolution : int
        Number of interpolation points (knots) used for splines in the quadrature.
        A maximum of three bounce points can be detected in between knots.
        The accuracy of the quadrature will increase as some function of
        the number of knots over the number of detected bounce points.
        So for well-behaved magnetic fields increasing resolution should increase
        the accuracy of the quadrature.
    method : str
        The method to evaluate the integral.
        The "spline" method exactly integrates a cubic spline of the integrand.
        The "quad" method performs a Gauss quadrature and estimates the integrand
        by using distinct cubic splines for components in the integrand so that
        the singularity from the division by zero near the bounce points can be
        captured more accurately than can be represented by a polynomial.

    Returns
    -------
    ba : callable
        This callable method computes the bounce average G_ℓ(λ) for every
        specified field line ℓ (constant rho and alpha), for every λ value in
        ``lambdas``.

    Examples
    --------
    .. code-block:: python

        ba = bounce_average(eq)
        G = ba(name, pitch)

    """
    bi = bounce_integral(eq, rho, alpha, zeta_max, pitch, resolution, method)

    def _bounce_average(name, pitch=None):
        """Compute the bounce average of the named quantity using the spline method.

        Parameters
        ----------
        name : ndarray
            Name of quantity in ``data_index`` to compute the bounce average of.
        pitch : ndarray
            λ values to evaluate the bounce average at.
            If None, uses the values given to the parent function.

        Returns
        -------
        G : ndarray, shape(pitch, alpha, rho, (resolution - 1) * 3)
            The last axis iterates through every bounce average performed
            along that field line padded by nan.

        """
        return safediv(bi(name, pitch), bi("1", pitch))

    return _bounce_average


def field_line_to_desc_coords(eq, rho, alpha, zeta):
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
    lg_data = eq.compute("iota", grid=lg)
    data = {
        d: _meshgrid_expand(lg.compress(lg_data[d]), rho.size, alpha.size, zeta.size)
        for d in lg_data
        if data_index["desc.equilibrium.equilibrium.Equilibrium"][d]["coordinates"]
        == "r"
    }
    sfl_coords = jnp.column_stack([r, a + data["iota"] * z, z])
    desc_coords = eq.compute_theta_coords(sfl_coords)
    grid = Grid(desc_coords, jitable=True)
    return grid, data
