"""Interpolation utilities."""

import warnings
from functools import partial

from interpax import interp1d

try:
    from jax_finufft import nufft2, options

except (ImportError, ModuleNotFoundError):
    warnings.warn(
        "jax_finufft is not installed. NUFFT functions will not be available.",
        UserWarning,
    )
except Exception as e:
    error_str = str(e)
    # This error will probably happen pretty often, we skip it to prevent breaking
    # codes that doesn't use jax_finufft but still want to use desc
    if "XLA FFI handler registration" in error_str:
        warnings.warn(
            "jax_finufft XLA FFI handler registration failed. "
            "This is likely due to a mismatch between the JAX version and the "
            "jax_finufft version. Change package versions to resolve this issue. "
            "NUFFT functions will not be available.",
            UserWarning,
        )
    # If we face any other specific error related to jax_finufft, we can catch it
    # in an elif block and provide a more specific warning.
    else:
        warnings.warn(
            "Unknown error occurred while importing jax_finufft. NUFFT functions "
            f"will not be available: {e}",
            UserWarning,
        )

from desc.backend import jax, jnp

_JF_BUG = True
"""https://github.com/flatironinstitute/jax-finufft/issues/158.

   Wait for jax-finufft to merge
   https://github.com/flatironinstitute/jax-finufft/pull/216
   then bump min version and set this to False.
"""


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
    # This is optimized away under JIT if the operation is an identity.
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
    mask=None,
    fill_value=None,
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
    mask : jnp.ndarray, optional
        Boolean mask of points to interpolate to. Should have same shape as ``x0``
        and ``x1``. This does nothing until the merge of
        https://github.com/flatironinstitute/jax-finufft/pull/216.
    fill_value : float
        Value to pad array where the mask is false.
        Default is 0.0.

    Returns
    -------
    c(x₀,x₁) : jnp.ndarray
        Real function value at query points.

    """
    # This is optimized away under JIT if the operation is an identity.
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

    if _JF_BUG:
        opts = options.Opts(modeord=0)
        f = jnp.fft.fftshift(f, (-2, -1))
        return (nufft2(f, x0, x1, iflag=1, eps=eps, opts=opts) * s).real

    opts = options.Opts(modeord=1)
    f = (nufft2(f, x0, x1, points_mask=mask, iflag=1, eps=eps, opts=opts) * s).real
    if mask is not None and fill_value is not None:
        f = jnp.where(mask[..., jnp.newaxis, :] if vec else mask, f, fill_value)
    return f


# Warning: method must be specified as keyword argument.
interp1d_vec = jnp.vectorize(
    interp1d, signature="(m),(n),(n)->(m)", excluded={"method"}
)


@partial(jnp.vectorize, signature="(m),(n),(n),(n)->(m)")
def interp1d_Hermite_vec(xq, x, f, fx, /):
    """Vectorized cubic Hermite interpolation."""
    return interp1d(xq, x, f, method="cubic", fx=fx)


def poly_val(*, x, c, der=False):
    """Evaluate polynomial ``c`` at the points ``x``.

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
    if c.shape[-1] == 4:
        if der:
            return (3 * c[..., 0] * x + 2 * c[..., 1]) * x + c[..., 2]
        return ((c[..., 0] * x + c[..., 1]) * x + c[..., 2]) * x + c[..., 3]

    if c.shape[-1] == 3:
        if der:
            return 2 * c[..., 0] * x + c[..., 1]
        return (c[..., 0] * x + c[..., 1]) * x + c[..., 2]

    if der:
        c = c[..., :-1] * jnp.arange(c.shape[-1] - 1, 0, -1)
    return jnp.sum(c * x[..., None] ** jnp.arange(c.shape[-1] - 1, -1, -1), axis=-1)


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


_root_companion = jnp.vectorize(
    partial(jnp.roots, strip_zeros=False), signature="(m)->(n)"
)


def _root_eps():
    # Safer to make this a callable since output depends on whether
    # double precision is enabled before it is called.
    return max(jnp.finfo(jnp.array(1.0).dtype).eps, 1e-11)


@partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6, 7))
def polyroot_vec(
    c,
    k=0.0,
    a_min=None,
    a_max=None,
    sort=False,
    sentinel=jnp.nan,
    eps=-1.0,
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
    if eps < 0:
        eps = _root_eps()
    get_only_real_roots = not (a_min is None and a_max is None)
    degree = c.shape[-1] - 1
    distinct = distinct and (degree > 1)

    if degree <= 3 and get_only_real_roots and jnp.isrealobj(c) and jnp.isrealobj(k):
        backward_stable = degree < 3

        c = jnp.moveaxis(c, -1, 0)
        r = {1: _root_linear, 2: _root_quadratic, 3: _root_cubic}[degree](
            *c[:-1], c[-1] - k
        )
        r = jnp.moveaxis(r, 0, -1)
        c = jnp.moveaxis(c, 0, -1)
    else:
        backward_stable = True

        r = _root_companion(_subtract_last(c, k))
        # If the complex part is too big, then these would not be real roots of
        # a nearby perturbed problem, so we set to nan so that they are not
        # classified as candidates after the correction step.
        if get_only_real_roots:
            r = jnp.where(
                jnp.abs(r.imag) <= eps**0.5,
                r.real,
                jnp.nan,
            )

    # Schröder first kind correction to push the roots of the perturbed problem
    # toward roots of the original problem.
    if degree > 1:
        k = jnp.expand_dims(k, -1)
        c = c[..., None, :]
        p0 = poly_val(x=r, c=c) - k
        p1 = poly_val(x=r, c=c, der=True)
        p2 = poly_val(x=r, c=c[..., :-1] * jnp.arange(degree, 0, -1), der=True)
        candidate = r - (p0 * p1) / (p1**2 - p0 * p2)

        residual = jnp.abs(p0)
        residual_new = jnp.abs(poly_val(x=candidate, c=c) - k)
        r = jnp.where(residual_new < residual, candidate, r)

        if not backward_stable:
            r = jnp.where(
                jnp.minimum(residual_new, residual) <= eps**0.5,
                r,
                jnp.nan,
            )

        del p0, p1, p2, candidate, residual, residual_new

    if distinct:
        # Then we need to ensure the returned roots have a consistent multiplicity.
        #
        # The correction above can merge roots that were artificially split due to
        # conditioning issues (e.g. if multiple reds or greens are adjacent).
        # The final block of this function sweeps through the roots and discards
        # duplicates that lie within ε of each other.
        #
        # The purpose of returning only distinct roots is that downstream algorithms
        # assume continuity of the underlying function. If we return an ordering of
        # distinct roots where the derivative does not change sign between adjacent
        # roots, this violates the behavior implied by the intermediate value theorem
        # and can mislead such algorithms.
        #
        # There is an edge case when the roots are near a tangent crossing. A computed
        # pair of roots may lie on opposite sides of a minimum within ε of each other.
        # One should either retain both roots as distinct or discard both as justified
        # by the fact that in a nearby perturbed problem neither point would be a root.
        # The best choice is the latter, since it is possible the full pair was not
        # detected in the first place due to the poor condition number, and to ensure
        # the duplicate removal sweep does not mistakenly discard only one element
        # of the pair.
        #
        # To detect such cases, we note that near a multiplicity m > 1 root, the
        # residual of the derivative is of order ε^(m-1) where ε is the distance
        # from the root to the tangent extrema. So we discard roots when the derivative
        # residual is sufficiently small near the root.
        r = jnp.where(
            # There is probably a way to make this comparison more robust,
            # but we do not appear to have issues in practice.
            jnp.abs(poly_val(x=r, c=c, der=True)) > eps,
            r,
            sentinel,
        )

    if get_only_real_roots:
        a_min = -jnp.inf if a_min is None else jnp.expand_dims(a_min, -1)
        a_max = +jnp.inf if a_max is None else jnp.expand_dims(a_max, -1)
        r = jnp.where(
            (a_min <= r.real) & (r.real <= a_max),
            r.real,
            sentinel,
        )
    elif not distinct and not jnp.isnan(sentinel):
        r = jnp.where(jnp.isfinite(r), r, sentinel)

    if sort or distinct:
        r = jnp.sort(r, stable=False)
    if distinct:
        r = jnp.where(
            jnp.isclose(jnp.diff(r, prepend=sentinel), 0.0, atol=eps),
            sentinel,
            r,
        )
    assert r.shape[-1] == degree
    return r


@polyroot_vec.defjvp
def _polyroot_vec_jvp(sort, sentinel, eps, distinct, primals, tangents):
    """Implicit function theorem with regularization.

    Regularization used to smooth the discretized system so that it recognizes
    any non-differentiable sample it has observed actually has zero measure in
    the continuous system.

    References
    ----------
    See supplementary information in DESC/publications/unalmis2025.

    """
    c, k, a_min, a_max = primals
    dc, dk, _, _ = tangents

    if eps < 0:
        eps = _root_eps()
    r = polyroot_vec(c, k, a_min, a_max, sort, sentinel, eps, distinct)

    dc_dr = poly_val(x=r, c=c[..., None, :], der=True)
    dc_dr = jnp.where(
        jnp.abs(dc_dr) > eps,
        dc_dr,
        dc_dr + jnp.copysign(eps, dc_dr.real),
    )
    dr = jnp.where(
        r == sentinel,
        0.0,
        (jnp.expand_dims(dk, -1) - poly_val(x=r, c=dc[..., None, :])) / dc_dr,
    )
    return r, dr


def _irreducible(Q, R, b):
    # Three irrational real roots.
    theta = jnp.arccos(R / jnp.sqrt(Q**3))
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


def _reducible(Q, R, b):
    # One real and two complex roots.
    A = -jnp.sign(R) * jnp.cbrt(jnp.abs(R) + jnp.sqrt(R**2 - Q**3))
    B = jnp.where(A == 0.0, 0.0, Q / A)
    return _concat_nan(((A + B) - b / 3)[None], num=2)


def _cubic(a, b, c, d):
    b = b / a
    c = c / a
    Q = (b**2 - 3 * c) / 9
    R = (2 * b**3 - 9 * b * c) / 54 + d / (2 * a)
    return jnp.where(R**2 < Q**3, _irreducible(Q, R, b), _reducible(Q, R, b))


def _root_cubic(a, b, c, d):
    """Return real cubic root assuming real coefficients.

    Uses numerical.recipes/book.html, page 228, which is not backwards stable.
    This can generate fake root with O(1) residual, so post-processing is needed.
    Advantage is it is much more performant than eigenvalue solve, especially
    when d is higher dimensional than a, b, c.
    """
    return jnp.where(
        a == 0.0,
        _concat_nan(_root_quadratic(b, c, d)),
        _cubic(a, b, c, d),
    )


def _root_quadratic(a, b, c):
    """Return real quadratic root assuming real coefficients."""
    # numerical.recipes/book.html, page 227
    q = -0.5 * (b + jnp.sign(b) * jnp.sqrt(b**2 - 4 * a * c))
    return jnp.stack(
        [
            jnp.where(a == 0.0, _root_linear(b, c).squeeze(0), q / a),
            c / q,
        ]
    )


def _root_linear(a, b):
    """Return real linear root assuming real coefficients."""
    return (-b / a)[None]


def _concat_nan(r, num=1):
    """Concatenate nan ``num`` times to ``r`` on first axis."""
    return jnp.concatenate((r, jnp.broadcast_to(jnp.nan, (num,) + r.shape[1:])))


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
