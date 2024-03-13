"""Methods for computing bounce integrals."""

from interpax import interp1d
from scipy.interpolate import Akima1DInterpolator, CubicHermiteSpline

from desc.backend import (
    complex_sqrt,
    flatnonzero,
    fori_loop,
    jnp,
    put,
    put_along_axis,
    vmap,
)
from desc.compute.utils import mask_diff, mask_take, safediv
from desc.grid import Grid, LinearGrid, _meshgrid_expand

from .data_index import data_index


def polyint(c, k=jnp.array([0])):
    """Coefficients for the primitives of the given set of polynomials.

    Parameters
    ----------
    c : ndarray
        First axis should store coefficients of a polynomial.
        For a polynomial given by ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0] - 1``,
        coefficient cᵢ should be stored at ``c[n - i]``.
    k : ndarray or float
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
    X = (x[jnp.newaxis].T ** jnp.arange(c.shape[0] - 1, -1, -1)).T
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    sub = alphabet[: c.ndim]
    val = jnp.einsum(f"{sub},{sub}...->{sub[1:]}...", c, X)
    return val


def tanh_sinh_quadrature(N, quad_limit=3.16):
    """
    tanh_sinh quadrature.

    This function outputs the quadrature points and weights
    for a tanh-sinh quadrature.

    ∫₋₁¹ f(x) dx = ∑ₖ wₖ f(xₖ)

    Parameters
    ----------
    N: int
        Number of quadrature points, preferably odd
    quad_limit: float
        The range of quadrature points to be mapped.
        Larger quad_limit implies better result but limited due to overflow in sinh

    Returns
    -------
    x : numpy array
        Quadrature points
    w : numpy array
        Quadrature weights

    """
    points = jnp.linspace(-quad_limit, quad_limit, N)
    h = 2 * quad_limit / (N - 1)
    sinh = jnp.sinh(points)
    x = jnp.tanh(0.5 * jnp.pi * sinh)
    w = 0.5 * jnp.pi * h * jnp.cosh(points) / jnp.cosh(0.5 * jnp.pi * sinh) ** 2
    return x, w


def cubic_poly_roots(coef, k=jnp.array([0]), a_min=None, a_max=None, sort=False):
    """Roots of cubic polynomial.

    Parameters
    ----------
    coef : ndarray
        First axis should store coefficients of a polynomial. For a polynomial
        given by c₁ x³ + c₂ x² + c₃ x + c₄, ``coef[i]`` should store cᵢ.
        It is assumed that c₁ is nonzero.
    k : ndarray, shape(constant.size, )
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
    d = jnp.squeeze((d[jnp.newaxis].T - k).T)
    t_0 = b**2 - 3 * a * c
    t_1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d
    C = ((t_1 + complex_sqrt(t_1**2 - 4 * t_0**3)) / 2) ** (1 / 3)
    C_is_zero = jnp.isclose(C, 0)

    def compute_roots(xi_k):
        t_3 = jnp.where(C_is_zero, 0, t_0 / (xi_k * C))
        r = -(b + xi_k * C + t_3) / (3 * a)
        return r

    def clip_to_nan(r):
        r = jnp.where(jnp.isreal(r) & (a_min <= r) & (r <= a_max), jnp.real(r), jnp.nan)
        return r

    xi_1 = (-1 + (-3) ** 0.5) / 2
    xi_2 = xi_1**2
    xi_3 = 1
    roots = tuple(map(compute_roots, (xi_1, xi_2, xi_3)))
    if clip:
        roots = tuple(map(clip_to_nan, roots))
    roots = jnp.stack(roots, axis=-1)
    if sort:
        roots = jnp.sort(roots, axis=-1)
    return roots


def _get_bounce_points(pitch, zeta, poly_B, poly_B_z):
    """Get the bounce points given |B| and 1 / λ.

    Parameters
    ----------
    pitch : ndarray
        λ values representing the constant function 1 / λ.
    zeta : ndarray
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray
        Polynomial coefficients of the cubic spline of |B|.
    poly_B_z : ndarray
        Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.

    Returns
    -------
    intersect, bp1, bp2 : ndarray, ndarray, ndarray
        The polynomials' intersection points with 1 / λ is given by ``intersect``.
        In order to be JIT compilable, the returned array must have a shape that
        accommodates the case where each cubic polynomial intersects 1 / λ thrice.
        So ``intersect`` has shape (pitch.size * M * L, N * NUM_ROOTS),
        where the last axis is padded with nan at the end to be JIT compilable.
        The boolean masks ``bp1`` and ``bp2`` encode whether a given entry in
        ``intersect`` is a valid starting and ending bounce point, respectively.

    """
    ML = poly_B.shape[1]
    N = poly_B.shape[2]
    NUM_ROOTS = 3
    a_min = zeta[:-1]
    a_max = zeta[1:]

    intersect = cubic_poly_roots(poly_B, 1 / pitch, a_min, a_max, sort=True).reshape(
        pitch.size, ML, N, NUM_ROOTS
    )
    B_z = polyval(intersect, poly_B_z[:, jnp.newaxis]).reshape(
        pitch.size * ML, N * NUM_ROOTS
    )
    intersect = intersect.reshape(pitch.size * ML, N * NUM_ROOTS)
    is_intersect = ~jnp.isnan(intersect)
    # Rearrange so that all the intersects along field line are contiguous.
    contiguous = vmap(
        lambda args: mask_take(*args, size=N * NUM_ROOTS, fill_value=jnp.nan)
    )
    intersect = contiguous((intersect, is_intersect))
    B_z = contiguous((B_z, is_intersect))
    # Check sign of derivative to determine whether root is a valid bounce point.
    bp1 = B_z <= 0
    bp2 = B_z >= 0

    # index of last intersect
    idx = (
        jnp.squeeze(
            vmap(lambda a: flatnonzero(a, size=1, fill_value=a.size))(~is_intersect)
        )
        - 1
    )
    assert idx.shape == (pitch.size * ML,)
    # Periodic boundary to compute bounce integrals of particles trapped outside
    # this snapshot of the field lines.
    # Roll such that first intersect is moved to index of last intersect.
    is_bp = bp1 & put_along_axis(jnp.roll(bp2, -1, axis=-1), idx, bp2[:, 0], axis=-1)
    # B_z<=0 at intersect_i implies B_z>=0 at intersect_i+1 by continuity.
    # I think this step is only needed to determine if the boundaries are bounce points.
    bp1 = bp1 & is_bp
    bp2 = bp2 & is_bp
    return intersect, bp1, bp2


def _get_bounce_points_include_knots(pitch, zeta, poly_B, poly_B_z):
    """Get the bounce points given |B| and 1 / λ.

    Like ``_get_bounce_points`` but returns additional ingredients
    needed by the algorithm in the direct method in ``bounce_integral``.

    Parameters
    ----------
    pitch : ndarray
        λ values representing the constant function 1 / λ.
    zeta : ndarray
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray
        Polynomial coefficients of the cubic spline of |B|.
    poly_B_z : ndarray
        Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.

    Returns
    -------
    intersect_nan_to_right_knot, contiguous, is_intersect, is_bp
        The polynomials' intersection points with 1 / λ is given by
        ``intersect_nan_to_right_knot``.
        In order to be JIT compilable, the returned array must have a shape that
        accommodates the case where each cubic polynomial intersects 1 / λ thrice.
        Rather than padding the nan values to the end, ``intersect_nan_to_right_knot``
        replaces the nan values with the right knot of the splines. This array
        has shape (pitch.size * M * L, N, NUM_ROOTS + 2).
        The boolean mask ``is_bp`` encodes whether a given entry in

        .. code-block:: python
            contiguous(
                (intersect_nan_to_right_knot.reshape(pitch.size * ML, -1), is_intersect)
            )

        is a valid bounce point.

    """
    ML = poly_B.shape[1]
    N = poly_B.shape[2]
    NUM_ROOTS = 3
    R = NUM_ROOTS + 2
    a_min = zeta[:-1]
    a_max = zeta[1:]

    roots = cubic_poly_roots(poly_B, 1 / pitch, a_min, a_max, sort=True).reshape(
        pitch.size, ML, N, 3
    )
    roots = (roots[..., 0], roots[..., 1], roots[..., 2])
    nan_to_right_knot = tuple(map(lambda r: jnp.where(jnp.isnan(r), a_max, r), roots))
    a_min = jnp.broadcast_to(a_min, shape=(pitch.size, ML, N))
    a_max = jnp.broadcast_to(a_max, shape=(pitch.size, ML, N))
    # Include the knots of the splines along with the intersection points.
    intersect = jnp.stack((a_min, *roots, a_max), axis=-1)
    intersect_nan_to_right_knot = jnp.stack(
        (a_min, *nan_to_right_knot, a_max), axis=-1
    ).reshape(pitch.size * ML, N, R)

    B_z = polyval(intersect, poly_B_z[:, jnp.newaxis]).reshape(pitch.size * ML, N * R)
    is_intersect = jnp.reshape(
        jnp.array([False, True, True, True, False], dtype=bool) & ~jnp.isnan(intersect),
        newshape=(pitch.size * ML, N * R),
    )
    # Rearrange so that all the intersects along field line are contiguous.
    contiguous = vmap(lambda args: mask_take(*args, size=N * R, fill_value=jnp.nan))
    B_z = contiguous((B_z, is_intersect))
    # Check sign of derivative to determine whether root is a valid bounce point.
    bp1 = B_z <= 0
    bp2 = B_z >= 0

    # index of last intersect
    idx = (
        jnp.squeeze(
            vmap(lambda a: flatnonzero(a, size=1, fill_value=a.size))(~is_intersect)
        )
        - 1
    )
    assert idx.shape == (pitch.size * ML,)
    # Periodic boundary to compute bounce integrals of particles trapped outside
    # this snapshot of the field lines.
    # Roll such that first intersect is moved to index of last intersect.
    is_bp = bp1 & put_along_axis(jnp.roll(bp2, -1, axis=-1), idx, bp2[:, 0], axis=-1)
    return intersect_nan_to_right_knot, contiguous, is_intersect, is_bp


def _compute_bp_if_given_pitch(
    pitch, zeta, poly_B, poly_B_z, get_bounce_points, *original, err=False
):
    """Return the ingredients needed by the ``bounce_integrals`` function.

    Parameters
    ----------
    pitch : ndarray
        λ values representing the constant function 1 / λ.
        If None, returns the given ``original`` tuple.
    zeta : ndarray
        Field line-following ζ coordinates of spline knots.
    poly_B : ndarray
        Polynomial coefficients of the cubic spline of |B|.
    poly_B_z : ndarray
        Polynomial coefficients of the cubic spline of ∂|B|/∂_ζ.
    get_bounce_points : callable
        Method to return bounce points.
    original : tuple
        pitch, intersect, is_bp, bp1, bp2.
    err : bool
        Whether to raise an error if ``pitch`` is None and ``original`` is empty.

    """
    if pitch is None:
        if err and not original:
            raise ValueError("No pitch values were given.")
        return original
    else:
        pitch = jnp.atleast_1d(pitch)
        return pitch, *get_bounce_points(pitch, zeta, poly_B, poly_B_z)


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
    zeta = jnp.linspace(0, zeta_max, resolution)
    L = rho.size
    M = alpha.size
    N = resolution - 1  # number of piecewise cubic polynomials per field line
    NUM_ROOTS = 3  # number of roots for cubic polynomial

    grid, data = field_line_to_desc_coords(eq, rho, alpha, zeta)
    data = eq.compute(
        ["B^zeta", "|B|", "|B|_z constant rho alpha"], grid=grid, data=data
    )
    B = data["|B|"].reshape(M * L, resolution)

    # TODO: https://github.com/f0uriest/interpax/issues/19
    poly_B = CubicHermiteSpline(
        zeta,
        B,
        data["|B|_z constant rho alpha"].reshape(M * L, resolution),
        axis=-1,
        extrapolate="periodic",
    ).c

    poly_B = jnp.moveaxis(poly_B, 1, -1)
    poly_B_z = polyder(poly_B)
    assert poly_B.shape == (4, M * L, N)
    assert poly_B_z.shape == (3, M * L, N)

    def _direct(name, pitch=None):
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
        F : ndarray, shape(pitch, alpha, rho, (resolution - 1) * 2)
            The last axis iterates through every bounce integral performed
            along that field line padded by zeros.

        """
        (
            pitch,
            intersect_nan_to_right_knot,
            contiguous,
            is_intersect,
            is_bp,
        ) = _compute_bp_if_given_pitch(
            pitch,
            zeta,
            poly_B,
            poly_B_z,
            _get_bounce_points_include_knots,
            *original,
            err=True,
        )

        integrand = jnp.nan_to_num(
            eq.compute(name, grid=grid, override_grid=False, data=data)[name]
            / (data["B^zeta"] * jnp.sqrt(1 - pitch[:, jnp.newaxis] * data["|B|"]))
        ).reshape(pitch.size * M * L, resolution)

        # TODO: https://github.com/f0uriest/interpax/issues/19
        integrand = Akima1DInterpolator(zeta, integrand, axis=-1).c

        integrand = jnp.moveaxis(integrand, 1, -1)
        assert integrand.shape == (4, pitch.size * M * L, N)
        # For this algorithm, computing integrals via differences of primitives
        # is preferable to any numerical quadrature. For example, even if the
        # intersection points were evenly spaced, a composite Simpson's quadrature
        # would require computing the spline on 1.8x more knots for the same accuracy.
        R = NUM_ROOTS + 2
        primitive = polyval(intersect_nan_to_right_knot, polyint(integrand)).reshape(
            pitch.size * M * L, N * R
        )
        sums = jnp.cumsum(
            # Periodic boundary to compute bounce integrals of particles
            # trapped outside this snapshot of the field lines.
            jnp.diff(primitive, axis=-1, append=primitive[..., 0, jnp.newaxis])
            # Multiply by mask that is false at shared knots of piecewise spline
            # to avoid adding difference between primitives of splines at knots.
            * jnp.append(jnp.arange(1, N * R) % R != 0, True),
            axis=-1,
        )
        F = jnp.nan_to_num(
            fun((contiguous((sums, is_intersect)), is_bp)), posinf=0, neginf=0
        )
        return F.reshape(pitch.size, M, L, N * R // 2)

    def _quad_sin(name, pitch=None):
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
        F : ndarray, shape(pitch, alpha, rho, (resolution - 1) * 2)
            The last axis iterates through every bounce integral performed
            along that field line padded by zeros.

        """
        pitch, intersect, bp1, bp2 = _compute_bp_if_given_pitch(
            pitch, zeta, poly_B, poly_B_z, _get_bounce_points, *original, err=True
        )
        bp1 = fun((intersect, bp1))
        bp2 = fun((intersect, bp2))
        X = x * (bp2 - bp1)[..., jnp.newaxis] + bp2[..., jnp.newaxis]
        assert X.shape == (pitch.size * M * L, N * 2, x.size)

        def body(i, integral):
            k = i % (N * 2)
            j = i // (N * 2)
            p = i // (M * L * N * 2)
            v = j % pitch.size
            # TODO: Add Hermite spline to interpax to pass in B_z[i].
            integrand = interp1d(X[j, k], zeta, f[v]) / (
                interp1d(X[j, k], zeta, B_sup_z[v])
                * jnp.sqrt(1 - pitch[p] * interp1d(X[j, k], zeta, B[v]))
            )
            integral = put(integral, i, jnp.sum(w * integrand))
            return integral

        f = eq.compute(name, grid=grid, override_grid=False, data=data)[name].reshape(
            M * L, resolution
        )
        B_sup_z = data["B^zeta"].reshape(M * L, resolution)
        F = jnp.nan_to_num(
            # TODO: Vectorize interpax to do this with 1 call with einsum.
            fori_loop(0, pitch.size * M * L * N * 2, body, jnp.zeros(X.shape[:-1]))
            * jnp.pi
            / (bp2 - bp1),
            posinf=0,
            neginf=0,
        )
        return F.reshape(pitch.size, M, L, N * 2)

    if method == "direct":
        bi = _direct
        fun = vmap(lambda args: mask_diff(*args)[::2])
        get_bounce_points = _get_bounce_points_include_knots
    else:
        bi = _quad_sin
        fun = vmap(lambda args: mask_take(*args, size=N * 2, fill_value=jnp.nan))
        get_bounce_points = _get_bounce_points
        x, w = tanh_sinh_quadrature(resolution)
        x = jnp.arcsin(x) / jnp.pi - 0.5
    original = _compute_bp_if_given_pitch(
        pitch, zeta, poly_B, poly_B_z, get_bounce_points, err=False
    )
    return bi


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
        G : ndarray, shape(pitch, alpha, rho, (resolution - 1) * 2)
            The last axis iterates through every bounce average performed
            along that field line padded by zeros.

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
