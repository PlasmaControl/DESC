"""Interpolation utilities."""

from functools import partial

from orthax.chebyshev import chebvander
from orthax.polynomial import polyvander

from desc.backend import dct, jnp, rfft, rfft2, take
from desc.compute.utils import safediv
from desc.integrals.quad_utils import bijection_from_disc
from desc.utils import Index, errorif

# TODO: Transformation to make nodes more uniform Boyd eq. 16.46 pg. 336.
#  Have a hunch it won't change locations of complex poles much, so using
#  more uniformly spaced nodes could speed up convergence.


def cheb_pts(N, lobatto=False, domain=(-1, 1)):
    """Get ``N`` Chebyshev points mapped to given domain.

    Notes
    -----
    This is a common definition of the Chebyshev points (see Boyd, Chebyshev and
    Fourier Spectral Methods p. 498). These are the points demanded by discrete
    cosine transformations to interpolate Chebyshev series because the cosine
    basis for the DCT is defined on [0, π].

    They differ in ordering from the points returned by
    ``numpy.polynomial.chebyshev.chebpts1`` and
    ``numpy.polynomial.chebyshev.chebpts2``.

    Parameters
    ----------
    N : int
        Number of points.
    lobatto : bool
        Whether to return the Gauss-Lobatto (extrema-plus-endpoint)
        instead of the interior roots for Chebyshev points.
    domain : (float, float)
        Domain for points.

    Returns
    -------
    pts : jnp.ndarray
        Shape (N, ).
        Chebyshev points mapped to given domain.

    """
    n = jnp.arange(N)
    if lobatto:
        y = jnp.cos(jnp.pi * n / (N - 1))
    else:
        y = jnp.cos(jnp.pi * (2 * n + 1) / (2 * N))
    return bijection_from_disc(y, domain[0], domain[-1])


def fourier_pts(M):
    """Get ``M`` Fourier points."""
    m = jnp.arange(1, M + 1)
    return -jnp.pi + 2 * jnp.pi * m / M


def harmonic(a, M, axis=-1):
    """Spectral coefficients of the Nyquist trigonometric interpolant.

    Parameters
    ----------
    a : jnp.ndarray
        Fourier coefficients ``a=rfft(f,norm="forward",axis=axis)``.
    M : int
        Spectral resolution of ``a``.
    axis : int
        Axis along which coefficients are stored.

    Returns
    -------
    h : jnp.ndarray
        Nyquist trigonometric interpolant coefficients.
        Coefficients ordered along ``axis`` of size ``M`` to match ordering of
        [1, cos(x), ..., cos(mx), sin(x), sin(2x), ..., sin(mx)] basis.

    """
    is_even = (M % 2) == 0
    # cos(mx) coefficients
    an = 2.0 * (
        a.real.at[Index.get(0, axis, a.ndim)]
        .divide(2.0)
        .at[Index.get(-1, axis, a.ndim)]
        .divide(1.0 + is_even)
    )
    # sin(mx) coefficients
    bn = -2.0 * take(
        a.imag,
        jnp.arange(1, a.shape[axis] - is_even),
        axis,
        unique_indices=True,
        indices_are_sorted=True,
    )
    h = jnp.concatenate([an, bn], axis=axis)
    assert h.shape[axis] == M
    return h


def harmonic_vander(x, M):
    """Nyquist trigonometric interpolant basis evaluated at ``x``.

    Parameters
    ----------
    x : jnp.ndarray
        Points at which to evaluate pseudo-Vandermonde matrix.
    M : int
        Spectral resolution.

    Returns
    -------
    basis : jnp.ndarray
        Shape (*x.shape, M).
        Pseudo-Vandermonde matrix of degree ``M-1`` and sample points ``x``.
        Last axis ordered as [1, cos(x), ..., cos(mx), sin(x), sin(2x), ..., sin(mx)].

    """
    m = jnp.fft.rfftfreq(M, d=1 / M)
    mx = m * x[..., jnp.newaxis]
    basis = jnp.concatenate(
        [jnp.cos(mx), jnp.sin(mx[..., 1 : m.size - ((M % 2) == 0)])], axis=-1
    )
    assert basis.shape == (*x.shape, M)
    return basis


# TODO: For inverse transforms, do multipoint evaluation with FFT.
#   FFT cost is 𝒪(M N log[M N]) while direct evaluation is 𝒪(M² N²).
#   Chapter 10, https://doi.org/10.1017/CBO9781139856065.
#   Right now we just do an MMT with the Vandermode matrix.
#   Multipoint is likely better than using NFFT to evaluate f(xq) given fourier
#   coefficients because evaluation points are quadratically packed near edges as
#   required by quadrature to avoid runge. NFFT is only approximation anyway.
#   https://github.com/flatironinstitute/jax-finufft.


def interp_rfft(xq, f, axis=-1):
    """Interpolate real-valued ``f`` to ``xq`` with FFT.

    Parameters
    ----------
    xq : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``xq`` must broadcast with arrays of shape ``np.delete(f.shape,axis)``.
    f : jnp.ndarray
        Real 2π periodic function values on uniform grid to interpolate.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    assert f.ndim >= 1
    a = rfft(f, axis=axis, norm="forward")
    fq = irfft_non_uniform(xq, a, f.shape[axis], axis)
    return fq


def irfft_non_uniform(xq, a, n, axis=-1):
    """Evaluate Fourier coefficients ``a`` at ``xq`` ∈ [0, 2π] periodic.

    Parameters
    ----------
    xq : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``xq`` must broadcast with arrays of shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Fourier coefficients ``a=rfft(f,axis=axis,norm="forward")``.
    n : int
        Spectral resolution of ``a``.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    assert a.ndim >= 1
    a = (
        (2.0 * a)
        .at[Index.get(0, axis, a.ndim)]
        .divide(2.0)
        .at[Index.get(-1, axis, a.ndim)]
        .divide(1.0 + ((n % 2) == 0))
    )
    a = jnp.moveaxis(a, axis, -1)
    m = jnp.fft.rfftfreq(n, d=1 / n)
    basis = jnp.exp(-1j * m * xq[..., jnp.newaxis])
    fq = jnp.linalg.vecdot(basis, a).real
    # TODO: Test JAX does this optimization automatically.
    # ℜ〈 basis, a 〉= cos(m xq)⋅ℜ(a) − sin(m xq)⋅ℑ(a)
    return fq


def interp_rfft2(xq, f, axes=(-2, -1)):
    """Interpolate real-valued ``f`` to ``xq`` with FFT.

    Parameters
    ----------
    xq : jnp.ndarray
        Shape (..., 2).
        Real query points where interpolation is desired.
        Last axis must hold coordinates for a given point.
        Shape ``xq.shape[:-1]`` must broadcast with shape ``np.delete(f.shape,axes)``.
    f : jnp.ndarray
        Shape (..., f.shape[-2], f.shape[-1]).
        Real (2π × 2π) periodic function values on uniform tensor-product grid
        to interpolate.
    axes : tuple[int, int]
        Axes along which to transform.
        The real transform is done along ``axes[-1]``, so it will be more
        efficient for that to denote the larger size axis in ``axes``.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    assert xq.shape[-1] == 2
    assert f.ndim >= 2
    a = rfft2(f, axes=axes, norm="forward")
    fq = irfft2_non_uniform(xq, a, f.shape[axes[0]], f.shape[axes[-1]], axes)
    return fq


def irfft2_non_uniform(xq, a, M, N, axes=(-2, -1)):
    """Evaluate Fourier coefficients ``a`` at ``xq`` ∈ [0, 2π]² periodic.

    Parameters
    ----------
    xq : jnp.ndarray
        Shape (..., 2).
        Real query points where interpolation is desired.
        Last axis must hold coordinates for a given point.
        Shape ``xq.shape[:-1]`` must broadcast with shape ``np.delete(a.shape,axes)``.
    a : jnp.ndarray
        Shape (..., a.shape[-2], a.shape[-1]).
        Fourier coefficients ``a=rfft2(f,axes=axes,norm="forward")``.
    M : int
        Spectral resolution of ``a`` along ``axes[0]``.
    N : int
        Spectral resolution of ``a`` along ``axes[-1]``.
    axes : tuple[int, int]
        Axes along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    assert xq.shape[-1] == 2
    assert a.ndim >= 2
    a = (
        (2.0 * a)
        .at[Index.get(0, axes[-1], a.ndim)]
        .divide(2.0)
        .at[Index.get(-1, axes[-1], a.ndim)]
        .divide(1.0 + ((N % 2) == 0))
    )
    a = jnp.moveaxis(a, source=axes, destination=(-2, -1))
    a = a.reshape(*a.shape[:-2], -1)

    m = jnp.fft.fftfreq(M, d=1 / M)
    n = jnp.fft.rfftfreq(N, d=1 / N)
    basis = jnp.exp(
        -1j
        * (
            (m * xq[..., 0, jnp.newaxis])[..., jnp.newaxis]
            + (n * xq[..., -1, jnp.newaxis])[..., jnp.newaxis, :]
        )
    ).reshape(*xq.shape[:-1], m.size * n.size)

    fq = jnp.linalg.vecdot(basis, a).real
    return fq


def transform_to_desc(grid, f):
    """Transform to DESC spectral domain.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (θ, ζ) with uniformly spaced nodes in
        (2π × 2π) poloidal and toroidal coordinates.
    f : jnp.ndarray
        Function evaluated on ``grid``.

    Returns
    -------
    a : jnp.ndarray
        Shape (grid.num_rho, grid.num_theta // 2 + 1, grid.num_zeta)
        Coefficients of 2D real FFT.

    """
    f = grid.meshgrid_reshape(f, order="rtz")
    a = rfft2(f, axes=(-1, -2), norm="forward")
    # Real fft done over poloidal since grid.num_theta > grid.num_zeta usually.
    assert a.shape == (grid.num_rho, grid.num_theta // 2 + 1, grid.num_zeta)
    return a


def cheb_from_dct(a, axis=-1):
    """Get discrete Chebyshev transform from discrete cosine transform.

    Parameters
    ----------
    a : jnp.ndarray
        Discrete cosine transform coefficients, e.g.
        ``a=dct(f,type=2,axis=axis,norm="forward")``.
        The discrete cosine transformation used by scipy is defined here.
        docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct
    axis : int
        Axis along which to transform.

    Returns
    -------
    cheb : jnp.ndarray
        Chebyshev coefficients along ``axis``.

    """
    cheb = a.copy().at[Index.get(0, axis, a.ndim)].divide(2.0)
    return cheb


def interp_dct(xq, f, lobatto=False, axis=-1):
    """Interpolate ``f`` to ``xq`` with discrete Chebyshev transform.

    Parameters
    ----------
    xq : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``xq`` must broadcast with shape ``np.delete(f.shape,axis)``.
    f : jnp.ndarray
        Real function values on Chebyshev points to interpolate.
    lobatto : bool
        Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
        or interior roots grid for Chebyshev points.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    lobatto = bool(lobatto)
    errorif(lobatto, NotImplementedError, "JAX hasn't implemented type 1 DCT.")
    assert f.ndim >= 1
    a = cheb_from_dct(dct(f, type=2 - lobatto, axis=axis), axis) / (
        f.shape[axis] - lobatto
    )
    fq = idct_non_uniform(xq, a, f.shape[axis], axis)
    return fq


def idct_non_uniform(xq, a, n, axis=-1):
    """Evaluate discrete Chebyshev transform coefficients ``a`` at ``xq`` ∈ [-1, 1].

    Parameters
    ----------
    xq : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``xq`` must broadcast with shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Discrete Chebyshev transform coefficients.
    n : int
        Spectral resolution of ``a``.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    assert a.ndim >= 1
    a = jnp.moveaxis(a, axis, -1)
    # Could use Clenshaw recursion with fq = chebval(xq, a, tensor=False).
    basis = chebvander(xq, n - 1)
    fq = jnp.linalg.vecdot(basis, a)
    return fq


def polyder_vec(c):
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


def polyval_vec(x, c):
    """Evaluate the set of polynomials ``c`` at the points ``x``.

    Note this function is not the same as ``np.polynomial.polynomial.polyval(x,c)``.

    Parameters
    ----------
    x : jnp.ndarray
        Real coordinates at which to evaluate the set of polynomials.
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

        val = polyval_vec(x, c)
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
    # Better than Horner's method as we expect to evaluate low order polynomials.
    # No need to use fast multipoint evaluation techniques for the same reason.
    val = jnp.linalg.vecdot(
        polyvander(x, c.shape[0] - 1), jnp.moveaxis(jnp.flipud(c), 0, -1)
    )
    return val


# TODO: Eventually do a PR to move this stuff into interpax.


_roots = jnp.vectorize(partial(jnp.roots, strip_zeros=False), signature="(m)->(n)")


def poly_root(
    c,
    k=0,
    a_min=None,
    a_max=None,
    sort=False,
    sentinel=jnp.nan,
    # About 2e-12 for 64 bit jax.
    eps=min(jnp.finfo(jnp.array(1.0).dtype).eps * 1e4, 1e-8),
    distinct=False,
):
    """Roots of polynomial with given coefficients.

    Parameters
    ----------
    c : jnp.ndarray
        First axis should store coefficients of a polynomial. For a polynomial given by
        ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[0]-1``, coefficient cᵢ should be stored at
        ``c[n-i]``.
    k : jnp.ndarray
        Specify to find solutions to ∑ᵢⁿ cᵢ xⁱ = ``k``. Should broadcast with arrays of
        shape ``c.shape[1:]``.
    a_min : jnp.ndarray
        Minimum ``a_min`` and maximum ``a_max`` value to return roots between.
        If specified only real roots  are returned. If None, returns all complex roots.
        Should broadcast with arrays of shape ``c.shape[1:]``.
    a_max : jnp.ndarray
        Minimum ``a_min`` and maximum ``a_max`` value to return roots between.
        If specified only real roots  are returned. If None, returns all complex roots.
        Should broadcast with arrays of shape ``c.shape[1:]``.
    sort : bool
        Whether to sort the roots.
    sentinel : float
        Value with which to pad array in place of filtered elements.
        Anything less than ``a_min`` or greater than ``a_max`` plus some floating point
        error buffer will work just like nan while avoiding nan gradient.
    eps : float
        Absolute tolerance with which to consider value as zero.
    distinct : bool
        Whether to only return the distinct roots. If true, when the multiplicity is
        greater than one, the repeated roots are set to ``sentinel``.

    Returns
    -------
    r : jnp.ndarray
        Shape (..., c.shape[1:], c.shape[0] - 1).
        The roots of the polynomial, iterated over the last axis.

    """
    get_only_real_roots = not (a_min is None and a_max is None)

    func = {2: _root_linear, 3: _root_quadratic, 4: _root_cubic}
    if (
        c.shape[0] in func
        and get_only_real_roots
        and not (jnp.iscomplexobj(c) or jnp.iscomplexobj(k))
    ):
        # Compute from analytic formula to avoid the issue of complex roots with small
        # imaginary parts and to avoid nan in gradient.
        r = func[c.shape[0]](*c[:-1], c[-1] - k, sentinel, eps, distinct)
        # We already filtered distinct roots for quadratics.
        distinct = distinct and c.shape[0] > 3
    else:
        # Compute from eigenvalues of polynomial companion matrix.
        c_n = c[-1] - k
        c = [jnp.broadcast_to(c_i, c_n.shape) for c_i in c[:-1]]
        c.append(c_n)
        c = jnp.stack(c, axis=-1)
        r = _roots(c)
    if get_only_real_roots:
        a_min = -jnp.inf if a_min is None else a_min[..., jnp.newaxis]
        a_max = +jnp.inf if a_max is None else a_max[..., jnp.newaxis]
        r = jnp.where(
            (jnp.abs(r.imag) <= eps) & (a_min <= r.real) & (r.real <= a_max),
            r.real,
            sentinel,
        )

    if sort or distinct:
        r = jnp.sort(r, axis=-1)
    return _filter_distinct(r, sentinel, eps) if distinct else r


def _root_cubic(a, b, c, d, sentinel, eps, distinct):
    """Return r such that a r³ + b r² + c r + d = 0, assuming real coef and roots."""
    # numerical.recipes/book.html, page 228

    def irreducible(Q, R, b, mask):
        # Three irrational real roots.
        theta = jnp.arccos(R / jnp.sqrt(jnp.where(mask, Q**3, R**2 + 1)))
        return jnp.moveaxis(
            -2
            * jnp.sqrt(Q)
            * jnp.stack(
                [
                    jnp.cos(theta / 3),
                    jnp.cos((theta + 2 * jnp.pi) / 3),
                    jnp.cos((theta - 2 * jnp.pi) / 3),
                ]
            )
            - b / 3,
            source=0,
            destination=-1,
        )

    def reducible(Q, R, b):
        # One real and two complex roots.
        A = -jnp.sign(R) * (jnp.abs(R) + jnp.sqrt(jnp.abs(R**2 - Q**3))) ** (1 / 3)
        B = safediv(Q, A)
        r1 = (A + B) - b / 3
        return _concat_sentinel(r1[..., jnp.newaxis], sentinel, num=2)

    def root(b, c, d):
        b = safediv(b, a)
        c = safediv(c, a)
        d = safediv(d, a)
        Q = (b**2 - 3 * c) / 9
        R = (2 * b**3 - 9 * b * c + 27 * d) / 54
        mask = R**2 < Q**3
        return jnp.where(
            mask[..., jnp.newaxis],
            irreducible(jnp.abs(Q), R, b, mask),
            reducible(Q, R, b),
        )

    return jnp.where(
        # Tests catch failure here if eps < 1e-12 for 64 bit jax.
        jnp.expand_dims(jnp.abs(a) <= eps, axis=-1),
        _concat_sentinel(_root_quadratic(b, c, d, sentinel, eps, distinct), sentinel),
        root(b, c, d),
    )


def _root_quadratic(a, b, c, sentinel, eps, distinct):
    """Return r such that a r² + b r + c = 0, assuming real coefficients and roots."""
    # numerical.recipes/book.html, page 227
    discriminant = b**2 - 4 * a * c
    q = -0.5 * (b + jnp.sign(b) * jnp.sqrt(jnp.abs(discriminant)))
    r1 = jnp.where(
        discriminant < 0,
        sentinel,
        safediv(q, a, _root_linear(b, c, sentinel, eps)),
    )
    r2 = jnp.where(
        # more robust to remove repeated roots with discriminant
        (discriminant < 0) | (distinct & (discriminant <= eps)),
        sentinel,
        safediv(c, q, sentinel),
    )
    return jnp.stack([r1, r2], axis=-1)


def _root_linear(a, b, sentinel, eps, distinct=False):
    """Return r such that a r + b = 0."""
    return safediv(-b, a, jnp.where(jnp.abs(b) <= eps, 0, sentinel))


def _concat_sentinel(r, sentinel, num=1):
    """Concat ``sentinel`` ``num`` times to ``r`` on last axis."""
    sent = jnp.broadcast_to(sentinel, (*r.shape[:-1], num))
    return jnp.append(r, sent, axis=-1)


def _filter_distinct(r, sentinel, eps):
    """Set all but one of matching adjacent elements in ``r``  to ``sentinel``."""
    # eps needs to be low enough that close distinct roots do not get removed.
    # Otherwise, algorithms relying on continuity will fail.
    mask = jnp.isclose(jnp.diff(r, axis=-1, prepend=sentinel), 0, atol=eps)
    r = jnp.where(mask, sentinel, r)
    return r
