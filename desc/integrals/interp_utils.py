"""Fast interpolation utilities.

Notes
-----
These polynomial utilities are chosen for performance on gpu among
methods that have the best (asymptotic) algorithmic complexity.
For example, we prefer to not use Horner's method.
"""

from functools import partial

import numpy as np
from interpax import interp1d
from orthax.chebyshev import chebroots, chebvander

from desc.backend import dct, jnp, rfft, rfft2, take
from desc.integrals.quad_utils import bijection_from_disc
from desc.utils import Index, errorif, safediv

# TODO: Boyd's method 𝒪(N²) instead of Chebyshev companion matrix 𝒪(N³).
#  John P. Boyd, Computing real roots of a polynomial in Chebyshev series
#  form through subdivision. https://doi.org/10.1016/j.apnum.2005.09.007.
chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")


# TODO: Transformation to make nodes more uniform Boyd eq. 16.46 pg. 336.
#  Have a hunch more uniformly spaced nodes could speed up convergence
#  Edit: Seems unnecessary for now. Chebyshev part converges fine.


def cheb_pts(N, domain=(-1, 1), lobatto=False):
    """Get ``N`` Chebyshev points mapped to given domain.

    Warnings
    --------
    This is a common definition of the Chebyshev points (see Boyd, Chebyshev and
    Fourier Spectral Methods p. 498). These are the points demanded by discrete
    cosine transformations to interpolate Chebyshev series because the cosine
    basis for the DCT is defined on [0, π]. They differ in ordering from the
    points returned by ``numpy.polynomial.chebyshev.chebpts1`` and
    ``numpy.polynomial.chebyshev.chebpts2``.

    Parameters
    ----------
    N : int
        Number of points.
    domain : (float, float)
        Domain for points.
    lobatto : bool
        Whether to return the Gauss-Lobatto (extrema-plus-endpoint)
        instead of the interior roots for Chebyshev points.

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
    """Get ``M`` Fourier points in [0, 2π]."""
    # [0, 2π] instead of [-π, π] required to match our definition of α.
    return 2 * jnp.pi * jnp.arange(M) / M


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
#   Right now we do an MMT with the Vandermode matrix.
#   Multipoint is likely better than using NFFT (for strong singular bounce
#   integrals) to evaluate f(xq) given fourier coefficients because evaluation
#   points are quadratically packed near edges for efficient quadrature. For
#   weak singularities (e.g. effective ripple) NFFT should work well.
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
    a = rfft(f, axis=axis, norm="forward")
    fq = irfft_non_uniform(xq, a, f.shape[axis], axis)
    return fq


def irfft_non_uniform(xq, a, n, axis=-1):
    """Evaluate Fourier coefficients ``a`` at ``xq`` ∈ [0, 2π].

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
    # |a| << |basis|, so move a instead of basis
    a = (
        jnp.moveaxis(a, axis, -1)
        .at[..., 0]
        .divide(2.0)
        .at[..., -1]
        .divide(1.0 + ((n % 2) == 0))
    )
    m = jnp.fft.rfftfreq(n, d=1 / n)
    basis = jnp.exp(-1j * m * xq[..., jnp.newaxis])
    fq = 2.0 * jnp.linalg.vecdot(basis, a).real
    # ℜ〈 basis, a 〉= cos(m xq)⋅ℜ(a) − sin(m xq)⋅ℑ(a)
    return fq


def interp_rfft2(xq, f, axes=(-2, -1)):
    """Interpolate real-valued ``f`` to ``xq`` with FFT.

    Parameters
    ----------
    xq : jnp.ndarray
        Shape (..., 2).
        Real query points where interpolation is desired.
        Shape ``xq.shape[:-1]`` must broadcast with shape ``np.delete(f.shape,axes)``.
        Last axis must hold coordinates for a given point. The coordinates stored
        along ``xq[...,0]`` (``xq[...,1]``) must be the same coordinate enumerated
        across axis ``min(axes)`` (``max(axes)``) of the function values ``f``.
    f : jnp.ndarray
        Shape (..., f.shape[-2], f.shape[-1]).
        Real (2π × 2π) periodic function values on uniform tensor-product grid
        to interpolate.
    axes : tuple[int, int]
        Axes along which to transform.
        The real transform is done along ``axes[1]``, so it will be more
        efficient for that to denote the larger size axis in ``axes``.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    a = rfft2(f, axes=axes, norm="forward")
    fq = irfft2_non_uniform(xq, a, f.shape[axes[0]], f.shape[axes[1]], axes)
    return fq


def irfft2_non_uniform(xq, a, M, N, axes=(-2, -1)):
    """Evaluate Fourier coefficients ``a`` at ``xq`` ∈ [0, 2π]².

    Parameters
    ----------
    xq : jnp.ndarray
        Shape (..., 2).
        Real query points where interpolation is desired.
        Last axis must hold coordinates for a given point.
        Shape ``xq.shape[:-1]`` must broadcast with shape ``np.delete(a.shape,axes)``.
        Last axis must hold coordinates for a given point. The coordinates stored
        along ``xq[...,0]`` (``xq[...,1]``) must be the same coordinate enumerated
        across axis ``min(axes)`` (``max(axes)``) of the Fourier coefficients ``a``.
    a : jnp.ndarray
        Shape (..., a.shape[-2], a.shape[-1]).
        Fourier coefficients ``a=rfft2(f,axes=axes,norm="forward")``.
    M : int
        Spectral resolution of ``a`` along ``axes[0]``.
    N : int
        Spectral resolution of ``a`` along ``axes[1]``.
    axes : tuple[int, int]
        Axes along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    errorif(not (len(axes) == xq.shape[-1] == 2), msg="This is a 2D transform.")
    errorif(a.ndim < 2, msg=f"Dimension mismatch, a.shape: {a.shape}.")

    # |a| << |basis|, so move a instead of basis
    a = (
        jnp.moveaxis(a, source=axes, destination=(-2, -1))
        .at[..., 0]
        .divide(2.0)
        .at[..., -1]
        .divide(1.0 + ((N % 2) == 0))
    )

    m = jnp.fft.fftfreq(M, d=1 / M)
    n = jnp.fft.rfftfreq(N, d=1 / N)
    idx = np.argsort(axes)
    basis = jnp.exp(
        1j
        * (
            (m * xq[..., idx[0], jnp.newaxis])[..., jnp.newaxis]
            + (n * xq[..., idx[1], jnp.newaxis])[..., jnp.newaxis, :]
        )
    )
    fq = 2.0 * (basis * a).real.sum(axis=(-2, -1))
    return fq


def cheb_from_dct(a, axis=-1):
    """Get discrete Chebyshev transform from discrete cosine transform.

    Parameters
    ----------
    a : jnp.ndarray
        Discrete cosine transform coefficients, e.g.
        ``a=dct(f,type=2,axis=axis,norm="forward")``.
        The discrete cosine transformation used by scipy is defined here:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html.
    axis : int
        Axis along which to transform.

    Returns
    -------
    cheb : jnp.ndarray
        Chebyshev coefficients along ``axis``.

    """
    cheb = a.copy().at[Index.get(0, axis, a.ndim)].divide(2.0)
    return cheb


def dct_from_cheb(cheb, axis=-1):
    """Get discrete cosine transform from discrete Chebyshev transform.

    Parameters
    ----------
    cheb : jnp.ndarray
        Discrete Chebyshev transform coefficients, e.g.``cheb_from_dct(a)``.
    axis : int
        Axis along which to transform.

    Returns
    -------
    a : jnp.ndarray
        Chebyshev coefficients along ``axis``.

    """
    a = cheb.copy().at[Index.get(0, axis, cheb.ndim)].multiply(2.0)
    return a


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
    a = jnp.moveaxis(a, axis, -1)
    # Could use Clenshaw recursion with fq = chebval(xq, a, tensor=False).
    basis = chebvander(xq, n - 1)
    fq = jnp.linalg.vecdot(basis, a)
    return fq


# Warning: method must be specified as keyword argument.
interp1d_vec = jnp.vectorize(
    interp1d, signature="(m),(n),(n)->(m)", excluded={"method"}
)


@partial(jnp.vectorize, signature="(m),(n),(n),(n)->(m)")
def interp1d_Hermite_vec(xq, x, f, fx, /):
    """Vectorized cubic Hermite spline."""
    return interp1d(xq, x, f, method="cubic", fx=fx)


def polyder_vec(c):
    """Coefficients for the derivatives of the given set of polynomials.

    Parameters
    ----------
    c : jnp.ndarray
        Last axis should store coefficients of a polynomial. For a polynomial given by
        ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[-1]-1``, coefficient cᵢ should be stored at
        ``c[...,n-i]``.

    Returns
    -------
    poly : jnp.ndarray
        Coefficients of polynomial derivative, ignoring the arbitrary constant. That is,
        ``poly[...,i]`` stores the coefficient of the monomial xⁿ⁻ⁱ⁻¹,  where n is
        ``c.shape[-1]-1``.

    """
    return c[..., :-1] * jnp.arange(c.shape[-1] - 1, 0, -1)


def polyval_vec(*, x, c):
    """Evaluate the set of polynomials ``c`` at the points ``x``.

    Parameters
    ----------
    x : jnp.ndarray
        Coordinates at which to evaluate the set of polynomials.
    c : jnp.ndarray
        Last axis should store coefficients of a polynomial. For a polynomial given by
        ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[-1]-1``, coefficient cᵢ should be stored at
        ``c[...,n-i]``.

    Returns
    -------
    val : jnp.ndarray
        Polynomial with given coefficients evaluated at given points.

    Examples
    --------
    .. code-block:: python

        np.testing.assert_allclose(
            polyval_vec(x=x, c=c),
            np.sum(polyvander(x, c.shape[-1] - 1) * c[..., ::-1], axis=-1),
        )

    """
    # Better than Horner's method as we expect to evaluate low order polynomials.
    # No need to use fast multipoint evaluation techniques for the same reason.
    return jnp.sum(
        c * x[..., jnp.newaxis] ** jnp.arange(c.shape[-1] - 1, -1, -1),
        axis=-1,
    )


# TODO: Eventually do a PR to move this stuff into interpax.


def _subtract_first(c, k):
    """Subtract ``k`` from first index of last axis of ``c``.

    Semantically same as ``return c.copy().at[...,0].add(-k)``,
    but allows dimension to increase.
    """
    c_0 = c[..., 0] - k
    c = jnp.concatenate(
        [
            c_0[..., jnp.newaxis],
            jnp.broadcast_to(c[..., 1:], (*c_0.shape, c.shape[-1] - 1)),
        ],
        axis=-1,
    )
    return c


def _subtract_last(c, k):
    """Subtract ``k`` from last index of last axis of ``c``.

    Semantically same as ``return c.copy().at[...,-1].add(-k)``,
    but allows dimension to increase.
    """
    c_1 = c[..., -1] - k
    c = jnp.concatenate(
        [
            jnp.broadcast_to(c[..., :-1], (*c_1.shape, c.shape[-1] - 1)),
            c_1[..., jnp.newaxis],
        ],
        axis=-1,
    )
    return c


def _filter_distinct(r, sentinel, eps):
    """Set all but one of matching adjacent elements in ``r``  to ``sentinel``."""
    # eps needs to be low enough that close distinct roots do not get removed.
    # Otherwise, algorithms relying on continuity will fail.
    mask = jnp.isclose(jnp.diff(r, axis=-1, prepend=sentinel), 0, atol=eps)
    r = jnp.where(mask, sentinel, r)
    return r


_polyroots_vec = jnp.vectorize(
    partial(jnp.roots, strip_zeros=False), signature="(m)->(n)"
)
_eps = max(jnp.finfo(jnp.array(1.0).dtype).eps, 2.5e-12)


def polyroot_vec(
    c,
    k=0,
    a_min=None,
    a_max=None,
    sort=False,
    sentinel=jnp.nan,
    eps=_eps,
    distinct=False,
):
    """Roots of polynomial with given coefficients.

    Parameters
    ----------
    c : jnp.ndarray
        Last axis should store coefficients of a polynomial. For a polynomial given by
        ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[-1]-1``, coefficient cᵢ should be stored at
        ``c[...,n-i]``.
    k : jnp.ndarray
        Shape (..., *c.shape[:-1]).
        Specify to find solutions to ∑ᵢⁿ cᵢ xⁱ = ``k``.
    a_min : jnp.ndarray
        Shape (..., *c.shape[:-1]).
        Minimum ``a_min`` and maximum ``a_max`` value to return roots between.
        If specified only real roots are returned, otherwise returns all complex roots.
    a_max : jnp.ndarray
        Shape (..., *c.shape[:-1]).
        Minimum ``a_min`` and maximum ``a_max`` value to return roots between.
        If specified only real roots are returned, otherwise returns all complex roots.
    sort : bool
        Whether to sort the roots.
    sentinel : float
        Value with which to pad array in place of filtered elements.
        Anything less than ``a_min`` or greater than ``a_max`` plus some floating point
        error buffer will work just like nan while avoiding ``nan`` gradient.
    eps : float
        Absolute tolerance with which to consider value as zero.
    distinct : bool
        Whether to only return the distinct roots. If true, when the multiplicity is
        greater than one, the repeated roots are set to ``sentinel``.

    Returns
    -------
    r : jnp.ndarray
        Shape (..., *c.shape[:-1], c.shape[-1] - 1).
        The roots of the polynomial, iterated over the last axis.

    """
    get_only_real_roots = not (a_min is None and a_max is None)
    num_coef = c.shape[-1]
    c = _subtract_last(c, k)
    func = {2: _root_linear, 3: _root_quadratic, 4: _root_cubic}

    if (
        num_coef in func
        and get_only_real_roots
        and not (jnp.iscomplexobj(c) or jnp.iscomplexobj(k))
    ):
        # Compute from analytic formula to avoid the issue of complex roots with small
        # imaginary parts and to avoid nan in gradient.
        r = func[num_coef](C=c, sentinel=sentinel, eps=eps, distinct=distinct)
        # We already filtered distinct roots for quadratics.
        distinct = distinct and num_coef > 3
    else:
        # Compute from eigenvalues of polynomial companion matrix.
        r = _polyroots_vec(c)

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
    r = _filter_distinct(r, sentinel, eps) if distinct else r
    assert r.shape[-1] == num_coef - 1
    return r


def _root_cubic(C, sentinel, eps, distinct):
    """Return real cubic root assuming real coefficients."""
    # numerical.recipes/book.html, page 228

    def irreducible(Q, R, b, mask):
        # Three irrational real roots.
        theta = R / jnp.sqrt(jnp.where(mask, Q**3, 1.0))
        theta = jnp.arccos(jnp.where(jnp.abs(theta) < 1.0, theta, 0.0))
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

    a = C[..., 0]
    b = C[..., 1]
    c = C[..., 2]
    d = C[..., 3]
    return jnp.where(
        # Tests catch failure here if eps < 1e-12 for 64 bit precision.
        jnp.expand_dims(jnp.abs(a) <= eps, axis=-1),
        _concat_sentinel(
            _root_quadratic(
                C=C[..., 1:], sentinel=sentinel, eps=eps, distinct=distinct
            ),
            sentinel,
        ),
        root(b, c, d),
    )


def _root_quadratic(C, sentinel, eps, distinct):
    """Return real quadratic root assuming real coefficients."""
    # numerical.recipes/book.html, page 227
    a = C[..., 0]
    b = C[..., 1]
    c = C[..., 2]

    discriminant = b**2 - 4 * a * c
    q = -0.5 * (b + jnp.sign(b) * jnp.sqrt(jnp.abs(discriminant)))
    r1 = jnp.where(
        discriminant < 0,
        sentinel,
        safediv(q, a, _root_linear(C=C[..., 1:], sentinel=sentinel, eps=eps)),
    )
    r2 = jnp.where(
        # more robust to remove repeated roots with discriminant
        (discriminant < 0) | (distinct & (discriminant <= eps)),
        sentinel,
        safediv(c, q, sentinel),
    )
    return jnp.stack([r1, r2], axis=-1)


def _root_linear(C, sentinel, eps, distinct=False):
    """Return real linear root assuming real coefficients."""
    a = C[..., 0]
    b = C[..., 1]
    return safediv(-b, a, jnp.where(jnp.abs(b) <= eps, 0, sentinel))


def _concat_sentinel(r, sentinel, num=1):
    """Append ``sentinel`` ``num`` times to ``r`` on last axis."""
    sent = jnp.broadcast_to(sentinel, (*r.shape[:-1], num))
    return jnp.append(r, sent, axis=-1)
