"""Fast interpolation utilities.

Notes
-----
These utilities are chosen for performance on gpu among
methods that have the best (asymptotic) algorithmic complexity.
For example, we prefer to not use Horner's method.
"""

import warnings
from functools import partial

import numpy as np
from interpax import interp1d

try:
    from jax_finufft import nufft2, options
except ImportError:
    warnings.warn(
        "\njax-finufft is not installed.\n"
        "If you want to use NUFFTs, follow the DESC installation instructions.\n"
        "Otherwise you must set the parameter nufft_eps to zero\n"
        "when computing effective ripple, Gamma_c, and any other\n"
        "computations that involve bounce integrals.\n"
    )

from desc.backend import dct, jnp, rfft, rfft2, take
from desc.integrals.quad_utils import bijection_from_disc
from desc.utils import Index, errorif, safediv


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
    return 2 * jnp.pi * jnp.arange(n) / n


def interp_rfft(x, f, domain=(0, 2 * jnp.pi), axis=-1):
    """Interpolate real-valued ``f`` to ``x`` with FFT and MMT.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with arrays of shape ``np.delete(f.shape,axis)``.
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
    return irfft_mmt(x, rfft(f, axis=axis, norm="forward"), f.shape[axis], domain, axis)


def irfft_mmt(x, a, n, domain=(0, 2 * jnp.pi), axis=-1, *, _modes=None):
    """Evaluate Fourier coefficients ``a`` at ``x``.

    Uses matrix multiplication transform.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with arrays of shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Fourier coefficients ``a=rfft(f,axis=axis,norm="forward")``.
    n : int
        Spectral resolution of ``a``.
    domain : tuple[float]
        Domain over which samples were taken.
    axis : int
        Axis along which to transform.
    _modes : jnp.ndarray
        If supplied, just builds the Vandermonde array and computes the dot product.
        Assumes the Fourier coefficients have the correct factors for the DC and
        Nyquist frequency. Assumes ``axis=-1``.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    if _modes is None:
        _modes = jnp.fft.rfftfreq(n, (domain[1] - domain[0]) / (2 * jnp.pi * n))
        i = (0, -1) if (n % 2 == 0) else 0
        a = jnp.moveaxis(a, axis, -1).at[..., i].divide(2) * 2
    vander = jnp.exp(1j * _modes * (x - domain[0])[..., jnp.newaxis])
    return (vander * a).real.sum(-1)


def ifft_mmt(x, a, domain=(0, 2 * jnp.pi), axis=-1, *, vander=None, modes=None):
    """Evaluate Fourier coefficients ``a`` at ``x``.

    Uses matrix multiplication transform.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with arrays of shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Fourier coefficients ``a=fft(f,axis=axis,norm="forward")``.
    domain : tuple[float]
        Domain over which samples were taken.
    axis : int
        Axis along which to transform.
    vander : jnp.ndarray
        Precomputed transform matrix.
        If given returns ``(vander*a).sum(axis)``.
    modes : jnp.ndarray
        Precomputed modes.

    Returns
    -------
    fq : jnp.ndarray
        Function value at query points.

    """
    if vander is None:
        if modes is None:
            n = a.shape[axis]
            modes = jnp.fft.fftfreq(n, (domain[1] - domain[0]) / (2 * jnp.pi * n))
        vander = jnp.exp(1j * modes * (x - domain[0])[..., jnp.newaxis])
        a = jnp.moveaxis(a, axis, -1)
        axis = -1
    return (vander * a).sum(axis)


def interp_rfft2(
    x0, x1, f, domain0=(0, 2 * jnp.pi), domain1=(0, 2 * jnp.pi), axes=(-2, -1)
):
    """Interpolate real-valued ``f`` to coordinates ``(x0,x1)`` with FFT and MMT.

    Parameters
    ----------
    x0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``min(axes)`` of the function values ``f``.
    x1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``max(axes)`` of the function values ``f``.
    f : jnp.ndarray
        Shape (..., f.shape[-2], f.shape[-1]).
        Real function values on uniform tensor-product grid over an open period.
    domain0 : tuple[float]
        Domain of coordinate specified by x₀ over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by x₁ over which samples were taken.
    axes : tuple[int]
        Axes along which to transform.
        The real transform is done along ``axes[1]``, so it will be more
        efficient for that to denote the smaller size axis in ``axes``.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    i = (0, -1) if (f.shape[axes[1]] % 2 == 0) else 0
    a = rfft2(f, axes=axes, norm="forward")
    a = jnp.moveaxis(a, axes, (-2, -1)).at[..., i].divide(2) * 2
    n0, n1 = sorted(axes)
    return _irfft2_mmt(
        x0,
        x1,
        a,
        f.shape[n0],
        f.shape[n1],
        domain0,
        domain1,
        axes,
    )


def _irfft2_mmt(
    x0, x1, a, n0, n1, domain0=(0, 2 * jnp.pi), domain1=(0, 2 * jnp.pi), axes=(-2, -1)
):
    """Evaluate Fourier coefficients ``a`` at coordinates ``(x0,x1)``.

    Uses matrix multiplication transform.

    Parameters
    ----------
    x0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``min(axes)`` of the Fourier coefficients ``a``.
    x1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``max(axes)`` of the Fourier coefficients ``a``.
    a : jnp.ndarray
        Shape (..., a.shape[-2], a.shape[-1]).
        Fourier coefficients.
        ``f=rfft2(f,axes=axes,norm="forward")``
        ``a=jnp.moveaxis(f,axes,(-2,-1)).at[...,i].divide(2)*2``.
    n0 : int
        Spectral resolution of ``a`` for ``domain0``.
    n1 : int
        Spectral resolution of ``a`` for ``domain1``.
    domain0 : tuple[float]
        Domain of coordinate specified by x₀ over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by x₁ over which samples were taken.
    axes : tuple[int]
        Axes along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    x = (x0, x1)
    n = (n0, n1)
    d = (domain0, domain1)
    f, r = np.argsort(axes)
    modes_f, modes_r = rfft2_modes(n[f], n[r], d[f], d[r])
    vander = rfft2_vander(x[f], x[r], modes_f, modes_r, d[f][0], d[r][0])
    return (vander * a).real.sum((-2, -1))


def rfft2_vander(
    x_fft,
    x_rfft,
    modes_fft,
    modes_rfft,
    x_fft0=0,
    x_rfft0=0,
    inverse_idx_fft=None,
    inverse_idx_rfft=None,
):
    """Return Vandermonde matrix for complex Fourier modes.

    Warnings
    --------
    It is vital to not perform any operations on Vandermonde array and immediately
    reduce it. For example, to transform from spectral to real space do
      ``a=jnp.fft.rfft2(f).at[...,i].divide(2)*2``

      ``(vander*a).real.sum((-2,-1))``

    Performing the scaling on the Vandermonde array would triple the memory consumption.
    Perhaps this is required for the compiler to fuse operations.

    Notes
    -----
    When the Vandermonde matrix is large, care needs to be taken to ensure the compiler
    fuses the operation to transform from spectral to real space. For JAX, this is up
    to the JIT compiler's whim, and it helps to make the code as suggestive as possible
    for that. Basically do not do anything besides the relevant matmuls after making
    the Vandermonde; even things like adding a new axis to the coefficient array or
    creating local variables after the Vandermonde array is made can prevent this.

    Parameters
    ----------
    x_fft : jnp.ndarray
        Real query points of coordinate in ``domain_fft`` where interpolation is
        desired.
    x_rfft : jnp.ndarray
        Real query points of coordinate in ``domain_rfft`` where interpolation is
        desired.
    modes_fft : jnp.ndarray
        FFT Fourier modes.
    modes_rfft : jnp.ndarray
        Real FFT Fourier modes.
    x_fft0 : float
        Left boundary of domain of coordinate specified by ``x_fft`` over which
        samples were taken.
    x_rfft0 : float
        Left boundary of domain of coordinate specified by ``x_rfft`` over which
        samples were taken.
    inverse_idx_fft : jnp.ndarray
        Optional. Inverse idx along axis 0 to ensure query points broadcast.
    inverse_idx_rfft : jnp.ndarray
        Optional. Inverse idx along axis 0 to ensure query points broadcast.

    Returns
    -------
    vander : jnp.ndarray
        Shape (..., modes_fft.size, modes_rfft.size).
        Vandermonde matrix to evaluate complex Fourier series.

    """
    vf = jnp.exp(1j * modes_fft * (x_fft - x_fft0)[..., jnp.newaxis])
    vr = jnp.exp(1j * modes_rfft * (x_rfft - x_rfft0)[..., jnp.newaxis])
    if inverse_idx_fft is not None:
        vf = vf[inverse_idx_fft]
    if inverse_idx_rfft is not None:
        vr = vr[inverse_idx_rfft]
    return vf[..., jnp.newaxis] * vr[..., jnp.newaxis, :]


def rfft2_modes(n_fft, n_rfft, domain_fft=(0, 2 * jnp.pi), domain_rfft=(0, 2 * jnp.pi)):
    """Modes for complex exponential basis for real Fourier transform.

    Parameters
    ----------
    n_fft : int
        Spectral resolution for ``domain_fft``.
    n_rfft : int
        Spectral resolution for ``domain_rfft``.
    domain_fft : tuple[float]
        Domain of coordinate over which samples are taken.
    domain_rfft : tuple[float]
        Domain of coordinate over which samples are taken.

    Returns
    -------
    modes_fft : jnp.ndarray
        Shape (n_fft, ).
        FFT Fourier modes.
    modes_rfft : jnp.ndarray
        Shape (n_rfft // 2 + 1, ).
        Real FFT Fourier modes.

    """
    modes_fft = jnp.fft.fftfreq(
        n_fft, (domain_fft[1] - domain_fft[0]) / (2 * jnp.pi * n_fft)
    )
    modes_rfft = jnp.fft.rfftfreq(
        n_rfft, (domain_rfft[1] - domain_rfft[0]) / (2 * jnp.pi * n_rfft)
    )
    return modes_fft, modes_rfft


def nufft1d2r(x, f, domain=(0, 2 * jnp.pi), vec=False, eps=1e-6):
    """Non-uniform 1D real fast Fourier transform of second type.

    Examples
    --------
    [Tutorial](https://finufft.readthedocs.io/en/latest/tutorial/realinterp1d.html#id1).
    Also see the tests in the following directory.

     - ``tests/test_interp_utils.py::TestFastInterp::test_non_uniform_real_FFT``
     - ``tests/test_interp_utils.py::TestFastInterp::test_nufft2_vec``

    Parameters
    ----------
    x : jnp.ndarray
        Real query points of coordinate in ``domain`` where interpolation is desired.
        The coordinates stored here must be the same coordinate enumerated across
        axis ``-1`` of ``f``.
    f : jnp.ndarray
        Fourier coefficients fₙ of the map x ↦ c(x) such that c(x) = ∑ₙ fₙ exp(i n x)
        where n >= 0.
    domain : tuple[float]
        Domain of coordinate specified by x over which samples were taken.
    vec : bool
        If set to ``True``, then it is assumed that multiple Fourier series are
        to be evaluated at the same non-uniform points. In that case, this flag
        must be set to retain the function signature for vectorization
        of ``(x),(b,f)->(b,x)``.
    eps : float
        Precision requested. Default is ``1e-6``.

    Returns
    -------
    c(x) : jnp.ndarray
        Real function value at query points.

    """
    # This is optimized away under JIT if the operation is an idenity.
    s = 2 * jnp.pi / (domain[1] - domain[0])
    x = (x - domain[0]) * s

    s = f.shape[-1] // 2
    s = jnp.exp(1j * s * x)
    s = s[..., jnp.newaxis, :] if vec else s

    opts = options.Opts(modeord=0)
    return (nufft2(f, x, iflag=1, eps=eps, opts=opts) * s).real


def nufft2d2r(
    x0,
    x1,
    f,
    domain0=(0, 2 * jnp.pi),
    domain1=(0, 2 * jnp.pi),
    rfft_axis=-1,
    vec=False,
    eps=1e-6,
):
    """Non-uniform 2D real fast Fourier transform of second type.

    Examples
    --------
    [Tutorial](https://finufft.readthedocs.io/en/latest/tutorial/realinterp1d.html#id1).
    Also see the tests in the following directory.

     - ``tests/test_interp_utils.py::TestFastInterp::test_non_uniform_real_FFT_2D``
     - ``tests/test_interp_utils.py::TestFastInterp::test_nufft2_vec``

    Parameters
    ----------
    x0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        The coordinates stored here must be the same coordinate
        enumerated across axis ``-2`` of ``f``.
    x1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        The coordinates stored here must be the same coordinate
        enumerated across axis ``-1`` of ``f``.
    f : jnp.ndarray
        Fourier coefficients fₘₙ of the map x₀,x₁ ↦ c(x₀,x₁) such that
        c(x₀,x₁) = ∑ₘₙ fₘₙ exp(i m x₀) exp(i n x₁).
    domain0 : tuple[float]
        Domain of coordinate specified by x₀ over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by x₁ over which samples were taken.
    rfft_axis : int
        Axis along which real FFT was performed.
        If -1 (-2), assumes c(x₀,x₁) = ∑ₘₙ fₘₙ exp(i m x₀) exp(i n x₁) where
            n ( m) >= 0, respectively.
    vec : bool
        If set to ``True``, then it is assumed that multiple Fourier series are
        to be evaluated at the same non-uniform points. In that case, this flag
        must be set to retain the function signature for vectorization
        of ``(x),(x),(b,f0,f1)->(b,x)``.
    eps : float
        Precision requested. Default is ``1e-6``.

    Returns
    -------
    c(x₀,x₁) : jnp.ndarray
        Real function value at query points.

    """
    # This is optimized away under JIT if the operation is an idenity.
    s0 = 2 * jnp.pi / (domain0[1] - domain0[0])
    s1 = 2 * jnp.pi / (domain1[1] - domain1[0])
    x0 = (x0 - domain0[0]) * s0
    x1 = (x1 - domain1[0]) * s1

    if rfft_axis is None:
        s = 1
    elif rfft_axis != -1 and rfft_axis != -2:
        raise NotImplementedError(f"rfft_axis must be -1 or -2, but got {rfft_axis}.")
    else:
        s = f.shape[rfft_axis] // 2
        s = jnp.exp(1j * s * (x1 if rfft_axis == -1 else x0))
        s = s[..., jnp.newaxis, :] if vec else s
        f = jnp.fft.ifftshift(f, rfft_axis)

    opts = options.Opts(modeord=1)
    JF_BUG = True
    if JF_BUG:
        # https://github.com/flatironinstitute/jax-finufft/pull/159
        opts = options.Opts(modeord=0)
        f = jnp.fft.fftshift(f, (-2, -1))

    return (nufft2(f, x0, x1, iflag=1, eps=eps, opts=opts) * s).real


def cheb_from_dct(a, axis=-1):
    """Get discrete Chebyshev transform from discrete cosine transform.

    Parameters
    ----------
    a : jnp.ndarray
        Discrete cosine transform coefficients, e.g.
        ``a=dct(f,type=2,axis=axis,norm="forward")``.
    axis : int
        Axis along which to transform.

    Returns
    -------
    cheb : jnp.ndarray
        Chebyshev coefficients along ``axis``.

    """
    return a.at[Index.get(0, axis, a.ndim)].divide(2)


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
    return cheb.at[Index.get(0, axis, cheb.ndim)].multiply(2)


def interp_dct(x, f, lobatto=False, axis=-1):
    """Interpolate ``f`` to ``x`` with discrete Chebyshev transform.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with shape ``np.delete(f.shape,axis)``.
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
    errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
    return idct_mmt(
        x,
        cheb_from_dct(dct(f, type=2 - lobatto, axis=axis), axis)
        / (f.shape[axis] - lobatto),
        axis,
    )


def idct_mmt(x, a, axis=-1, vander=None):
    """Evaluate Chebyshev coefficients ``a`` at ``x`` ∈ [-1, 1].

    Uses matrix multiplication transform.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Discrete Chebyshev transform coefficients.
    axis : int
        Axis along which to transform.
    vander : jnp.ndarray
        Precomputed transform matrix.
        If given returns ``(vander*a).sum(axis)``.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    if vander is None:
        vander = jnp.cos(jnp.arange(a.shape[axis]) * jnp.arccos(x)[..., jnp.newaxis])
        a = jnp.moveaxis(a, axis, -1)
        axis = -1
    # Better than Clenshaw recursion ``chebval(x,a,tensor=False)`` on GPU.
    return (vander * a).sum(axis)


# Warning: method must be specified as keyword argument.
interp1d_vec = jnp.vectorize(
    interp1d, signature="(m),(n),(n)->(m)", excluded={"method"}
)


@partial(jnp.vectorize, signature="(m),(n),(n),(n)->(m)")
def interp1d_Hermite_vec(xq, x, f, fx, /):
    """Vectorized cubic Hermite interpolation."""
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


def _subtract_first(c, k):
    """Subtract ``k`` from first index of last axis of ``c``.

    Semantically same as ``return c.at[...,0].subtract(k)``,
    but allows dimension to increase.
    """
    c_0 = c[..., 0] - k
    return jnp.concatenate(
        [
            c_0[..., jnp.newaxis],
            jnp.broadcast_to(c[..., 1:], (*c_0.shape, c.shape[-1] - 1)),
        ],
        axis=-1,
    )


def _subtract_last(c, k):
    """Subtract ``k`` from last index of last axis of ``c``.

    Semantically same as ``return c.at[...,-1].subtract(k)``,
    but allows dimension to increase.
    """
    c_1 = c[..., -1] - k
    return jnp.concatenate(
        [
            jnp.broadcast_to(c[..., :-1], (*c_1.shape, c.shape[-1] - 1)),
            c_1[..., jnp.newaxis],
        ],
        axis=-1,
    )


# TODO (#1388): Move this stuff into interpax.


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
        # TODO: Differentiate through root alone with custom_linear_solve.
        # Compute from analytic formula to avoid the issue of complex roots with small
        # imaginary parts and to avoid nan in gradient. Also consumes less memory.
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


def rfft_to_trig(a, n, axis=-1):
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

        Coefficients are ordered along ``axis`` of size ``n`` to match
        Vandermonde matrix with order
        [sin(k𝐱), ..., sin(𝐱), 1, cos(𝐱), ..., cos(k𝐱)].
        When ``n`` is even the sin(k𝐱) coefficient is zero and is excluded.

    """
    is_even = (n % 2) == 0
    an = -2 * jnp.flip(
        take(
            a.imag,
            jnp.arange(1, a.shape[axis] - is_even),
            axis,
            unique_indices=True,
            indices_are_sorted=True,
        ),
        axis=axis,
    )
    i = (0, -1) if is_even else 0
    bn = a.real.at[Index.get(i, axis, a.ndim)].divide(2) * 2
    h = jnp.concatenate([an, bn], axis=axis)
    assert h.shape[axis] == n
    return h


def trig_vander(x, n, domain=(0, 2 * jnp.pi)):
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
    vander : jnp.ndarray
        Shape (*x.shape, n).
        Vandermonde matrix of degree ``n-1`` and sample points ``x``.
        Last axis ordered as [sin(k𝐱), ..., sin(𝐱), 1, cos(𝐱), ..., cos(k𝐱)].
        When ``n`` is even the sin(k𝐱) basis function is excluded.

    """
    is_even = (n % 2) == 0
    n_rfft = jnp.fft.rfftfreq(n, d=(domain[-1] - domain[0]) / (2 * jnp.pi * n))
    nx = n_rfft * (x - domain[0])[..., jnp.newaxis]
    return jnp.concatenate(
        [jnp.sin(nx[..., n_rfft.size - is_even - 1 : 0 : -1]), jnp.cos(nx)], axis=-1
    )
