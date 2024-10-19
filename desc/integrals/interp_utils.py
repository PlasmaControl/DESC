"""Fast interpolation utilities.

Notes
-----
These utilities are chosen for performance on gpu among
methods that have the best (asymptotic) algorithmic complexity.
For example, we prefer to not use Horner's method.
"""

from functools import partial

import numpy as np
from interpax import interp1d
from orthax.chebyshev import chebroots

from desc.backend import dct, jnp, rfft, rfft2, take
from desc.integrals.quad_utils import bijection_from_disc
from desc.utils import Index, errorif, safediv

# TODO:
#  1. Boyd's method 𝒪(n²) instead of Chebyshev companion matrix 𝒪(n³).
#  John P. Boyd, Computing real roots of a polynomial in Chebyshev series
#  form through subdivision. https://doi.org/10.1016/j.apnum.2005.09.007.
#  Use that once to find extrema of |B| if Y_B > 64.
#  2. Then to find roots of bounce points use the closed formula in Boyd's
#  spectral methods section 19.6. Can isolate interval to search for root by
#  observing whether B - 1/pitch changes sign at extrema. Only need to do
#  evaluate Chebyshev series at quadrature points once, and can use that to
#  compute the integral for every pitch. The integral will converge rapidly
#  since a low order polynomial approximates |B| well in between adjacent
#  extrema. This is cheaper and non-iterative, so jax and gpu will like it.
#  Implementing 1 and 2 will remove all eigenvalue solves from computation.
#  2 is a larger improvement than 1. Implement this in later PR.
chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")


def cheb_pts(n, domain=(-1, 1), lobatto=False):
    """Get ``n`` Chebyshev points mapped to given domain.

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
    n : int
        Number of points.
    domain : tuple[float]
        Domain for points.
    lobatto : bool
        Whether to return the Gauss-Lobatto (extrema-plus-endpoint)
        instead of the interior roots for Chebyshev points.

    Returns
    -------
    pts : jnp.ndarray
        Shape (n, ).
        Chebyshev points mapped to given domain.

    """
    N = jnp.arange(n)
    if lobatto:
        y = jnp.cos(jnp.pi * N / (n - 1))
    else:
        y = jnp.cos(jnp.pi * (2 * N + 1) / (2 * n))
    return bijection_from_disc(y, domain[0], domain[-1])


def fourier_pts(n):
    """Get ``n`` Fourier points in [0, 2π)."""
    # [0, 2π] instead of [-π, π] required to match our definition of α.
    return 2 * jnp.pi * jnp.arange(n) / n


def harmonic(a, n, axis=-1):
    """Spectral coefficients of the Nyquist trigonometric interpolant.

    Parameters
    ----------
    a : jnp.ndarray
        Fourier coefficients ``a=rfft(f,norm="forward",axis=axis)``.
    n : int
        Spectral resolution of ``a``.
    axis : int
        Axis along which coefficients are stored.

    Returns
    -------
    h : jnp.ndarray
        Nyquist trigonometric interpolant coefficients.
        Coefficients ordered along ``axis`` of size ``n`` to match ordering of
        [1, cos(x), ..., cos(nx), sin(x), sin(2x), ..., sin(nx)] basis.

    """
    is_even = (n % 2) == 0
    # cos(nx) coefficients
    an = 2.0 * (
        a.real.at[Index.get(0, axis, a.ndim)]
        .divide(2.0)
        .at[Index.get(-1, axis, a.ndim)]
        .divide(1.0 + is_even)
    )
    # sin(nx) coefficients
    bn = -2.0 * take(
        a.imag,
        jnp.arange(1, a.shape[axis] - is_even),
        axis,
        unique_indices=True,
        indices_are_sorted=True,
    )
    h = jnp.concatenate([an, bn], axis=axis)
    assert h.shape[axis] == n
    return h


def harmonic_vander(x, n, domain=(0, 2 * jnp.pi)):
    """Nyquist trigonometric interpolant basis evaluated at ``x``.

    Parameters
    ----------
    x : jnp.ndarray
        Points at which to evaluate Vandermonde matrix.
    n : int
        Spectral resolution.
    domain : tuple[float]
        Domain over which samples will be taken.
        This domain should span an open period of the function to interpolate.

    Returns
    -------
    basis : jnp.ndarray
        Shape (*x.shape, n).
        Vandermonde matrix of degree ``n-1`` and sample points ``x``.
        Last axis ordered as [1, cos(x), ..., cos(nx), sin(x), sin(2x), ..., sin(nx)].

    """
    is_even = (n % 2) == 0
    n_rfft = jnp.fft.rfftfreq(n, d=(domain[-1] - domain[0]) / (2 * jnp.pi * n))
    nx = n_rfft * (x - domain[0])[..., jnp.newaxis]
    basis = jnp.concatenate(
        [jnp.cos(nx), jnp.sin(nx[..., 1 : n_rfft.size - is_even])], axis=-1
    )
    assert basis.shape[-1] == n
    return basis


# TODO: For inverse transforms, use non-uniform fast transforms (NFFT).
#   https://github.com/flatironinstitute/jax-finufft.
#   Let spectral resolution be F, (e.g. F = M N for 2D transform),
#   and number of points (non-uniform) to evaluate be Q. A non-uniform
#   fast transform cost is 𝒪([F+Q] log[F] log[1/ε]) where ε is the
#   interpolation error term (depending on implementation how ε appears
#   may change, but it is always logarithmic). Direct evaluation is 𝒪(F Q).
#   Note that for the inverse Chebyshev transforms, we can also use fast
#   multipoint methods Chapter 10, https://doi.org/10.1017/CBO9781139856065.
#   Unlike NFFTs, multipoint methods are exact and reduce to using FFTs.
#   The cost is 𝒪([F+Q] log²[F + Q]). This might be useful to evaluating
#   |B|, since the integrands are not smooth functions of |B|, which we know
#   as a Chebyshev series, and the nodes are packed more tightly near the
#   singular regions.


def interp_rfft(xq, f, domain=(0, 2 * jnp.pi), axis=-1):
    """Interpolate real-valued ``f`` to ``xq`` with FFT.

    Parameters
    ----------
    xq : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``xq`` must broadcast with arrays of shape ``np.delete(f.shape,axis)``.
    f : jnp.ndarray
        Real function values on uniform grid over an open period to interpolate.
    domain : tuple[float]
        Domain over which samples were taken.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    a = rfft(f, axis=axis, norm="forward")
    return irfft_non_uniform(xq, a, f.shape[axis], domain, axis)


def irfft_non_uniform(xq, a, n, domain=(0, 2 * jnp.pi), axis=-1):
    """Evaluate Fourier coefficients ``a`` at ``xq``.

    Parameters
    ----------
    xq : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``xq`` must broadcast with arrays of shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Fourier coefficients ``a=rfft(f,axis=axis,norm="forward")``.
    n : int
        Spectral resolution of ``a``.
    domain : tuple[float]
        Domain over which samples were taken.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    # |a| << |xq|, so move a instead
    a = (
        jnp.moveaxis(a, axis, -1)
        .at[..., 0]
        .divide(2.0)
        .at[..., -1]
        .divide(1.0 + ((n % 2) == 0))
    )
    n = jnp.fft.rfftfreq(n, d=(domain[-1] - domain[0]) / (2 * jnp.pi * n))
    basis = jnp.exp(1j * n * (xq - domain[0])[..., jnp.newaxis])
    return 2.0 * (basis * a).real.sum(axis=-1)


def interp_rfft2(
    xq0, xq1, f, domain0=(0, 2 * jnp.pi), domain1=(0, 2 * jnp.pi), axes=(-2, -1)
):
    """Interpolate real-valued ``f`` to coordinates ``(xq0,xq1)`` with FFT.

    Parameters
    ----------
    xq0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``min(axes)`` of the function values ``f``.
    xq1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``max(axes)`` of the function values ``f``.
    f : jnp.ndarray
        Shape (..., f.shape[-2], f.shape[-1]).
        Real function values on uniform tensor-product grid over an open period.
    domain0 : tuple[float]
        Domain of coordinate specified by ``xq0`` over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by ``xq1`` over which samples were taken.
    axes : tuple[int]
        Axes along which to transform.
        The real transform is done along ``axes[1]``, so it will be more
        efficient for that to denote the larger size axis in ``axes``.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    a = rfft2(f, axes=axes, norm="forward")
    n0, n1 = sorted(axes)
    return irfft2_non_uniform(
        xq0, xq1, a, f.shape[n0], f.shape[n1], domain0, domain1, axes
    )


def irfft2_non_uniform(
    xq0, xq1, a, n0, n1, domain0=(0, 2 * jnp.pi), domain1=(0, 2 * jnp.pi), axes=(-2, -1)
):
    """Evaluate Fourier coefficients ``a`` at coordinates ``(xq0,xq1)``.

    Parameters
    ----------
    xq0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``min(axes)`` of the Fourier coefficients ``a``.
    xq1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``max(axes)`` of the Fourier coefficients ``a``.
    a : jnp.ndarray
        Shape (..., a.shape[-2], a.shape[-1]).
        Fourier coefficients ``a=rfft2(f,axes=axes,norm="forward")``.
    n0 : int
        Spectral resolution of ``a`` for ``domain0``.
    n1 : int
        Spectral resolution of ``a`` for ``domain1``.
    domain0 : tuple[float]
        Domain of coordinate specified by ``xq0`` over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by ``xq1`` over which samples were taken.
    axes : tuple[int]
        Axes along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    idx = np.argsort(axes)
    d = (domain0, domain1)
    n = (n0, n1)
    d_fft, d_rfft = d[idx[0]], d[idx[1]]
    n_fft, n_rfft = n[idx[0]], n[idx[1]]

    # |a| << |xq|, so move a instead
    a = (
        jnp.moveaxis(a, source=axes, destination=(-2, -1))
        .at[..., 0]
        .divide(2.0)
        .at[..., -1]
        .divide(1.0 + ((n_rfft % 2) == 0))
    )
    n_fft = jnp.fft.fftfreq(n_fft, d=(d_fft[1] - d_fft[0]) / (2 * jnp.pi * n_fft))
    n_rfft = jnp.fft.rfftfreq(n_rfft, d=(d_rfft[1] - d_rfft[0]) / (2 * jnp.pi * n_rfft))
    xq = (xq0 - domain0[0], xq1 - domain1[0])
    basis = jnp.exp(
        1j
        * (
            (n_fft * xq[idx[0]][..., jnp.newaxis])[..., jnp.newaxis]
            + (n_rfft * xq[idx[1]][..., jnp.newaxis])[..., jnp.newaxis, :]
        )
    )
    return 2.0 * (basis * a).real.sum(axis=(-2, -1))


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
    return a.copy().at[Index.get(0, axis, a.ndim)].divide(2.0)


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
    return cheb.copy().at[Index.get(0, axis, cheb.ndim)].multiply(2.0)


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
    return idct_non_uniform(xq, a, f.shape[axis], axis)


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
    # |a| << |xq|, so move a instead
    a = jnp.moveaxis(a, axis, -1)
    # Same as Clenshaw recursion ``chebval(xq,a,tensor=False)`` but better on GPU.
    n = jnp.arange(n)
    return jnp.sum(jnp.cos(n * jnp.arccos(xq)[..., jnp.newaxis]) * a, axis=-1)


def _fourier(grid, f, is_reshaped=False):
    """Transform to DESC spectral domain.

    Parameters
    ----------
    grid : Grid
        Tensor-product grid in (θ, ζ) with uniformly spaced nodes [0, 2π) × [0, 2π/NFP).
        Preferably power of 2 for ``grid.num_theta`` and ``grid.num_zeta``.
    f : jnp.ndarray
        Function evaluated on ``grid``.

    Returns
    -------
    a : jnp.ndarray
        Shape (..., grid.num_theta // 2 + 1, grid.num_zeta)
        Complex coefficients of 2D real FFT of ``f``.

    """
    if not is_reshaped:
        f = grid.meshgrid_reshape(f, "rtz")
    # real fft over poloidal since usually M > N
    return rfft2(f, axes=(-1, -2), norm="forward")


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
    return jnp.where(mask, sentinel, r)


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
    """Concatenate ``sentinel`` ``num`` times to ``r`` on last axis."""
    sent = jnp.broadcast_to(sentinel, (*r.shape[:-1], num))
    return jnp.append(r, sent, axis=-1)
