"""Interpolation utilities."""

from functools import partial

from desc.backend import jnp, rfft, rfft2
from desc.compute.utils import safediv

# TODO: For inverse transforms, do multipoint evaluation with FFT.
#   FFT cost is 𝒪(M N log[M N]) while direct evaluation is 𝒪(M² N²).
#   Chapter 10, https://doi.org/10.1017/CBO9781139856065.
#   Likely better than using NFFT to evaluate f(xq) given fourier
#   coefficients because evaluation points are quadratically packed near edges as
#   required by quadrature to avoid runge. NFFT is only approximation anyway.


def interp_rfft(xq, f):
    """Interpolate real-valued ``f`` to ``xq`` with FFT.

    Parameters
    ----------
    xq : jnp.ndarray
        Shape (..., xq.shape[-1]).
        Query points where interpolation is desired.
    f : jnp.ndarray
        Shape (..., f.shape[-1]).
        Function values on 2π periodic grid to interpolate.

    Returns
    -------
    fq : jnp.ndarray
        Shape (..., xq.shape[-1])
        Function value at query points.

    """
    assert xq.ndim == f.ndim >= 1
    return irfft_non_uniform(xq, rfft(f, norm="forward"), f.shape[-1])


def irfft_non_uniform(xq, a, M):
    """Evaluate Fourier coefficients ``a`` at ``xq`` ∈ [0, 2π] periodic.

    Parameters
    ----------
    xq : jnp.ndarray
        Shape (..., xq.shape[-1]).
        Query points where interpolation is desired.
        Dimension should match ``a``, though size of last axis may differ.
    a : jnp.ndarray
        Fourier coefficients ``a = rfft(f, norm="forward")``.
    M : int
        Spectral resolution of ``a``.

    Returns
    -------
    fq : jnp.ndarray
        Shape (..., xq.shape[-1])
        Function value at query points.

    """
    a = a.at[..., 0].divide(2.0).at[..., -1].divide(1.0 + ((M % 2) == 0))
    m = jnp.fft.rfftfreq(M, d=1 / M)
    basis = jnp.exp(1j * m * xq[..., jnp.newaxis])
    fq = 2 * jnp.real(jnp.linalg.vecdot(jnp.conj(basis), a))
    return fq


def interp_rfft2(xq, f):
    """Interpolate real-valued ``f`` to ``xq`` with FFT.

    Parameters
    ----------
    xq : jnp.ndarray
        Shape (..., xq.shape[-2], 2).
        Query points where interpolation is desired.
    f : jnp.ndarray
        Shape (..., f.shape[-2], f.shape[-1]).
        Function values on (2π × 2π) periodic tensor-product grid to interpolate.

    Returns
    -------
    fq : jnp.ndarray
        Shape (..., xq.shape[-2]).
        Function value at query points.

    """
    assert xq.ndim == f.ndim >= 2
    return irfft2_non_uniform(xq, rfft2(f, norm="forward"), *f.shape[-2:])


def irfft2_non_uniform(xq, a, M, N):
    """Evaluate Fourier coefficients ``a`` at ``xq`` ∈ [0, 2π]² periodic.

    Parameters
    ----------
    xq : jnp.ndarray
        Shape (..., xq.shape[-2], 2).
        Query points where interpolation is desired.
    a : jnp.ndarray
        Fourier coefficients ``a = rfft2(f, norm="forward")``.
    M : int
        Spectral resolution of ``a`` along second to last axis.
    N : int
        Spectral resolution of ``a`` along last axis.

    Returns
    -------
    fq : jnp.ndarray
        Shape (..., xq.shape[-2]).
        Function value at query points.

    """
    a = a.at[..., 0].divide(2.0).at[..., -1].divide(1.0 + ((N % 2) == 0))
    m = jnp.fft.fftfreq(M, d=1 / M)
    n = jnp.fft.rfftfreq(N, d=1 / N)
    basis = jnp.exp(
        1j
        * (
            (m * xq[..., 0, jnp.newaxis])[..., jnp.newaxis]
            + (n * xq[..., -1, jnp.newaxis])[..., jnp.newaxis, :]
        )
    )
    fq = 2 * jnp.real(jnp.einsum("...mn,...mn", basis, a))
    return fq


# TODO: upstream cubic spline polynomial root finding to interpax


def _filter_distinct(r, sentinel, eps):
    """Set all but one of matching adjacent elements in ``r``  to ``sentinel``."""
    # eps needs to be low enough that close distinct roots do not get removed.
    # Otherwise, algorithms relying on continuity will fail.
    mask = jnp.isclose(jnp.diff(r, axis=-1, prepend=sentinel), 0, atol=eps)
    r = jnp.where(mask, sentinel, r)
    return r


def _sentinel_append(r, sentinel, num=1):
    """Concat ``sentinel`` ``num`` times to ``r`` on last axis."""
    sent = jnp.broadcast_to(sentinel, (*r.shape[:-1], num))
    return jnp.append(r, sent, axis=-1)


def _root_linear(a, b, sentinel, eps, distinct=False):
    """Return r such that a r + b = 0."""
    return safediv(-b, a, jnp.where(jnp.abs(b) <= eps, 0, sentinel))


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
        return _sentinel_append(r1[..., jnp.newaxis], sentinel, num=2)

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
        _sentinel_append(_root_quadratic(b, c, d, sentinel, eps, distinct), sentinel),
        root(b, c, d),
    )


_roots = jnp.vectorize(partial(jnp.roots, strip_zeros=False), signature="(m)->(n)")


def _poly_root(
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
    is_real = not (jnp.iscomplexobj(c) or jnp.iscomplexobj(k))
    get_only_real_roots = not (a_min is None and a_max is None)

    func = {2: _root_linear, 3: _root_quadratic, 4: _root_cubic}
    if c.shape[0] in func and is_real and get_only_real_roots:
        # Compute from analytic formula to avoid the issue of complex roots with small
        # imaginary parts and to avoid nan in gradient.
        r = func[c.shape[0]](*c[:-1], c[-1] - k, sentinel, eps, distinct)
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
            (jnp.abs(jnp.imag(r)) <= eps) & (a_min <= r) & (r <= a_max),
            jnp.real(r),
            sentinel,
        )

    if sort or distinct:
        r = jnp.sort(r, axis=-1)
    return _filter_distinct(r, sentinel, eps) if distinct else r
