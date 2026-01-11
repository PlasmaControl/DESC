"""Interpolation utilities."""

import warnings
from functools import partial

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

from desc.backend import jnp
from desc.utils import safediv


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


# Warning: method must be specified as keyword argument.
interp1d_vec = jnp.vectorize(
    interp1d, signature="(m),(n),(n)->(m)", excluded={"method"}
)


@partial(jnp.vectorize, signature="(m),(n),(n),(n)->(m)")
def interp1d_Hermite_vec(xq, x, f, fx, /):
    """Vectorized cubic Hermite interpolation."""
    return interp1d(xq, x, f, method="cubic", fx=fx)


def cubic_val(*, x, c, der=False):
    """Evaluate cubic polynomial ``c`` at the points ``x``.

    Parameters
    ----------
    x : jnp.ndarray
        Coordinates at which to evaluate the set of polynomials.
    c : jnp.ndarray
        Last axis should store coefficients of a polynomial. For a polynomial given by
        ∑ᵢⁿ cᵢ xⁱ, where n is ``c.shape[-1]-1``, coefficient cᵢ should be stored at
        ``c[...,n-i]``.
    der : bool
        Whether to evaluate the derivative instead.

    Returns
    -------
    val : jnp.ndarray
        Polynomial with given coefficients evaluated at given points.

    Examples
    --------
    .. code-block:: python

        np.testing.assert_allclose(
            cubic_val(x=x, c=c),
            np.sum(polyvander(x, c.shape[-1] - 1) * c[..., ::-1], axis=-1),
        )

    """
    assert c.shape[-1] == 4
    if der:
        return (3 * c[..., 0] * x + 2 * c[..., 1]) * x + c[..., 2]
    return ((c[..., 0] * x + c[..., 1]) * x + c[..., 2]) * x + c[..., 3]


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


def _filter_distinct(r, sentinel, eps):
    """Set all but one of matching adjacent elements in ``r``  to ``sentinel``."""
    # eps needs to be low enough that close distinct roots do not get removed.
    # Otherwise, algorithms relying on continuity will fail.
    mask = jnp.isclose(jnp.diff(r, axis=-1, prepend=sentinel), 0, atol=eps)
    return jnp.where(mask, sentinel, r)


_root_companion = jnp.vectorize(
    partial(jnp.roots, strip_zeros=False), signature="(m)->(n)"
)
_eps = max(jnp.finfo(jnp.array(1.0).dtype).eps, 2.5e-12)

# TODO (#1388): Move to interpax.


def polyroot_vec(
    c,
    k=0.0,
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
    distinct = distinct and num_coef > 2
    func = {2: _root_linear, 3: _root_quadratic, 4: _root_cubic}

    if (
        num_coef in func
        and get_only_real_roots
        and not (jnp.iscomplexobj(c) or jnp.iscomplexobj(k))
    ):
        # Compute from analytic formula to avoid the issue of complex roots with small
        # imaginary parts and to avoid nan in gradient. Also consumes less memory.
        c = jnp.moveaxis(c, -1, 0)
        r = func[num_coef](*c[:-1], c[-1] - k, sentinel, eps, distinct)
        if num_coef == 2:
            r = r[jnp.newaxis]
        r = jnp.moveaxis(r, 0, -1)

        # We already filtered distinct roots for quadratics.
        distinct = distinct and num_coef > 3
    else:
        r = _root_companion(_subtract_last(c, k))

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
    if distinct:
        r = _filter_distinct(r, sentinel, eps)
    assert r.shape[-1] == num_coef - 1
    return r


def _root_cubic(a, b, c, d, sentinel, eps, distinct):
    """Return real cubic root assuming real coefficients."""
    # numerical.recipes/book.html, page 228

    def irreducible(Q, R, b, mask):
        # Three irrational real roots.
        theta = R / jnp.sqrt(jnp.where(mask, Q**3, 1.0))
        theta = jnp.arccos(jnp.where(jnp.abs(theta) < 1.0, theta, 0.0))
        return (
            -2
            * jnp.sqrt(Q)
            * jnp.stack(
                [
                    jnp.cos(theta / 3),
                    jnp.cos((theta + 2 * jnp.pi) / 3),
                    jnp.cos((theta - 2 * jnp.pi) / 3),
                ]
            )
            - b / 3
        )

    def reducible(Q, R, b):
        # One real and two complex roots.
        A = -jnp.sign(R) * jnp.cbrt(jnp.abs(R) + jnp.sqrt(jnp.abs(R**2 - Q**3)))
        B = safediv(Q, A)
        r1 = (A + B) - b / 3
        return _concat_sentinel(r1[jnp.newaxis], sentinel, num=2)

    def root(b, c, d):
        b = safediv(b, a)
        c = safediv(c, a)
        Q = (b**2 - 3 * c) / 9
        R = (2 * b**3 - 9 * b * c) / 54 + safediv(d, 2 * a)
        mask = R**2 < Q**3
        return jnp.where(
            mask,
            irreducible(jnp.abs(Q), R, b, mask),
            reducible(Q, R, b),
        )

    return jnp.where(
        # Tests catch failure here if eps < 1e-12 for double precision.
        jnp.abs(a) <= eps,
        _concat_sentinel(
            _root_quadratic(b, c, d, sentinel, eps, distinct),
            sentinel,
        ),
        root(b, c, d),
    )


def _root_quadratic(a, b, c, sentinel, eps, distinct):
    """Return real quadratic root assuming real coefficients."""
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
    return jnp.stack([r1, r2])


def _root_linear(a, b, sentinel, eps, distinct=False):
    """Return real linear root assuming real coefficients."""
    return safediv(-b, a, jnp.where(jnp.abs(b) <= eps, 0, sentinel))


def _concat_sentinel(r, sentinel, num=1):
    """Concatenate ``sentinel`` ``num`` times to ``r`` on first axis."""
    return jnp.concatenate((r, jnp.broadcast_to(sentinel, (num, *r.shape[1:]))))


# TODO: replace the inner loop in orthax with this
def chebder(c, m=1, scl=1.0, axis=0, keepdims=False):
    """Same as orthax.chebder but fast enough to use in optimization loop."""
    assert m == 1
    c = jnp.flip(c.swapaxes(axis, 0), 0)

    N = c.shape[0]
    n = jnp.arange(N - 1, -1, -1).reshape((N,) + (1,) * (c.ndim - 1))
    w = (2 * scl) * n * c

    dc = jnp.flip(
        jnp.zeros(c.shape)
        .at[1::2]
        .set(jnp.cumsum(w[::2], 0)[: N // 2])
        .at[2::2]
        .set(jnp.cumsum(w[1::2], 0)[: (N - 1) // 2])
        .at[-1]
        .multiply(0.5),
        0,
    )
    if not keepdims:
        dc = dc[:-1]
    dc = dc.swapaxes(axis, 0)
    return dc
