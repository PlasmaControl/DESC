"""Test bounce integral methods."""

import inspect
from functools import partial

import numpy as np
import pytest
from jax import grad
from matplotlib import pyplot as plt
from orthax.legendre import leggauss
from scipy import integrate
from scipy.interpolate import CubicHermiteSpline
from scipy.special import ellipkm1
from tests.test_plotting import tol_1d

from desc.backend import flatnonzero, jnp
from desc.compute.bounce_integral import (
    _composite_linspace,
    _filter_nonzero_measure,
    _filter_not_nan,
    _poly_der,
    _poly_root,
    _poly_val,
    _take_mask,
    affine_bijection,
    automorphism_arcsin,
    automorphism_sin,
    bounce_integral,
    bounce_points,
    get_extrema,
    get_pitch,
    grad_affine_bijection,
    grad_automorphism_arcsin,
    grad_automorphism_sin,
    plot_field_line,
    tanh_sinh,
)
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.utils import only1


def _affine_bijection_forward(x, a, b):
    """[a, b] ∋ x ↦ y ∈ [−1, 1]."""
    y = 2 * (x - a) / (b - a) - 1
    return y


@partial(np.vectorize, signature="(m)->()")
def _last_value(a):
    """Return the last non-nan value in ``a``."""
    a = a[::-1]
    idx = np.squeeze(flatnonzero(~np.isnan(a), size=1, fill_value=0))
    return a[idx]


@pytest.mark.unit
def test_mask_operations():
    """Test custom masked array operation."""
    rows = 5
    cols = 7
    a = np.random.rand(rows, cols)
    nan_idx = np.random.choice(rows * cols, size=(rows * cols) // 2, replace=False)
    a.ravel()[nan_idx] = np.nan
    taken = _take_mask(a, ~np.isnan(a))
    last = _last_value(taken)
    for i in range(rows):
        desired = a[i, ~np.isnan(a[i])]
        assert np.array_equal(
            taken[i],
            np.pad(desired, (0, cols - desired.size), constant_values=np.nan),
            equal_nan=True,
        ), "take_mask has bugs."
        assert np.array_equal(
            last[i],
            desired[-1] if desired.size else np.nan,
            equal_nan=True,
        ), "flatnonzero has bugs."


@pytest.mark.unit
def test_reshape_convention():
    """Test the reshaping convention separates data across field lines."""
    rho = np.linspace(0, 1, 3)
    alpha = np.linspace(0, 2 * np.pi, 4)
    zeta = np.linspace(0, 6 * np.pi, 5)
    grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz")
    r, a, z = grid.nodes.T
    # functions of zeta should separate along first two axes
    # since those are contiguous, this should work
    f = z.reshape(-1, zeta.size)
    for i in range(1, f.shape[0]):
        np.testing.assert_allclose(f[i - 1], f[i])
    # likewise for rho
    f = r.reshape(rho.size, -1)
    for i in range(1, f.shape[-1]):
        np.testing.assert_allclose(f[:, i - 1], f[:, i])
    # test reshaping result won't mix data
    f = (a**2 + z).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[0]):
        np.testing.assert_allclose(f[i - 1], f[i])
    f = (r**2 + z).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[1]):
        np.testing.assert_allclose(f[:, i - 1], f[:, i])
    f = (r**2 + a).reshape(rho.size, alpha.size, zeta.size)
    for i in range(1, f.shape[-1]):
        np.testing.assert_allclose(f[..., i - 1], f[..., i])

    err_msg = "The ordering conventions are required for correctness."
    assert "P, S, N" in inspect.getsource(bounce_points), err_msg
    assert "S, knots.size" in inspect.getsource(bounce_integral), err_msg
    assert 'meshgrid(a, b, c, indexing="ij")' in inspect.getsource(
        Grid.create_meshgrid
    ), err_msg


@pytest.mark.unit
def test_poly_root():
    """Test vectorized computation of cubic polynomial exact roots."""
    cubic = 4
    c = np.arange(-24, 24).reshape(cubic, 6, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    constant = np.broadcast_to(np.arange(c.shape[-1]), c.shape[1:])
    constant = np.stack([constant, constant])
    root = _poly_root(c, constant, sort=True)

    for i in range(constant.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                d = c[-1, j, k] - constant[i, j, k]
                np.testing.assert_allclose(
                    actual=root[i, j, k],
                    desired=np.sort(np.roots([*c[:-1, j, k], d])),
                )

    c = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -1, -8, 12],
            [1, -6, 11, -6],
            [0, -6, 11, -2],
        ]
    )
    root = _poly_root(c.T, sort=True, distinct=True)
    for j in range(c.shape[0]):
        unique_roots = np.unique(np.roots(c[j]))
        root_filter = _filter_not_nan(root[j], check=True)
        assert root_filter.size == unique_roots.size, j
        np.testing.assert_allclose(
            actual=root_filter,
            desired=unique_roots,
            err_msg=str(j),
        )
    c = np.array([0, 1, -1, -8, 12])
    root = _filter_not_nan(_poly_root(c, sort=True, distinct=True), check=True)
    unique_root = np.unique(np.roots(c))
    assert root.size == unique_root.size
    np.testing.assert_allclose(root, unique_root)


@pytest.mark.unit
def test_poly_der():
    """Test vectorized computation of polynomial derivative."""
    quintic = 6
    c = np.arange(-18, 18).reshape(quintic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    derivative = _poly_der(c)
    for j in range(c.shape[1]):
        for k in range(c.shape[2]):
            np.testing.assert_allclose(
                actual=derivative[:, j, k], desired=np.polyder(c[:, j, k])
            )


@pytest.mark.unit
def test_poly_val():
    """Test vectorized computation of polynomial evaluation."""

    def test(x, c):
        val = _poly_val(x=x, c=c)
        if val.ndim != max(x.ndim, c.ndim - 1):
            raise ValueError(f"Incompatible shapes {x.shape} and {c.shape}.")
        for index in np.ndindex(c.shape[1:]):
            idx = (..., *index)
            np.testing.assert_allclose(
                actual=val[idx],
                desired=np.poly1d(c[idx])(x[idx]),
                err_msg=f"Failed with shapes {x.shape} and {c.shape}.",
            )

    quartic = 5
    c = np.arange(-60, 60).reshape(quartic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    x = np.linspace(0, 20, c.shape[1] * c.shape[2]).reshape(c.shape[1], c.shape[2])
    test(x, c)

    x = np.stack([x, x * 2], axis=0)
    x = np.stack([x, x * 2, x * 3, x * 4], axis=0)
    # make sure broadcasting won't hide error in implementation
    assert np.unique(x.shape).size == x.ndim
    assert c.shape[1:] == x.shape[x.ndim - (c.ndim - 1) :]
    assert np.unique((c.shape[0],) + x.shape[c.ndim - 1 :]).size == x.ndim - 1
    test(x, c)


@pytest.mark.unit
def test_get_extrema():
    """Test that these pitch intersect extrema of |B|."""
    start = -np.pi
    end = -2 * start
    k = np.linspace(start, end, 5)
    B = CubicHermiteSpline(
        k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
    )
    B_z_ra = B.derivative()
    extrema_scipy = np.sort(B(B_z_ra.roots(extrapolate=False)))
    rtol = 1e-7
    extrema = get_extrema(k, B.c, B_z_ra.c, relative_shift=rtol)
    eps = 100 * np.finfo(float).eps
    extrema = np.sort(_filter_not_nan(extrema))
    assert extrema.size == extrema_scipy.size
    np.testing.assert_allclose(extrema, extrema_scipy, rtol=rtol + eps)


@pytest.mark.unit
def test_composite_linspace():
    """Test this utility function useful for Newton-Cotes integration over pitch."""
    B_min_tz = np.array([0.1, 0.2])
    B_max_tz = np.array([1, 3])
    breaks = np.linspace(B_min_tz, B_max_tz, num=5)
    b = _composite_linspace(breaks, num=3)
    for i in range(breaks.shape[0]):
        for j in range(breaks.shape[1]):
            assert only1(np.isclose(breaks[i, j], b[:, j]).tolist())


@pytest.mark.unit
def test_bounce_points():
    """Test that bounce points are computed correctly."""

    def test_bp1_first():
        start = np.pi / 3
        end = 6 * np.pi
        knots = np.linspace(start, end, 5)
        B = CubicHermiteSpline(knots, np.cos(knots), -np.sin(knots))
        pitch = 2.0
        intersect = B.solve(1 / pitch, extrapolate=False)
        bp1, bp2 = bounce_points(pitch, knots, B.c, B.derivative().c, check=True)
        bp1, bp2 = _filter_nonzero_measure(bp1, bp2)
        assert bp1.size and bp2.size
        np.testing.assert_allclose(bp1, intersect[0::2])
        np.testing.assert_allclose(bp2, intersect[1::2])

    def test_bp2_first():
        start = -3 * np.pi
        end = -start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(k, np.cos(k), -np.sin(k))
        pitch = 2.0
        intersect = B.solve(1 / pitch, extrapolate=False)
        bp1, bp2 = bounce_points(pitch, k, B.c, B.derivative().c, check=True)
        bp1, bp2 = _filter_nonzero_measure(bp1, bp2)
        assert bp1.size and bp2.size
        # Don't include intersect[-1] for now as it doesn't have a paired bp2.
        np.testing.assert_allclose(bp1, intersect[1:-1:2])
        np.testing.assert_allclose(bp2, intersect[0::2][1:])

    def test_bp1_before_extrema():
        start = -np.pi
        end = -2 * start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(
            k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[3] + 1e-13
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = _filter_nonzero_measure(bp1, bp2)
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[1], 1.982767, rtol=1e-6)
        np.testing.assert_allclose(bp1, intersect[[1, 2]], rtol=1e-6)
        # intersect array could not resolve double root as single at index 2,3
        np.testing.assert_allclose(intersect[2], intersect[3], rtol=1e-6)
        np.testing.assert_allclose(bp2, intersect[[3, 4]], rtol=1e-6)

    def test_bp2_before_extrema():
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 4,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 4,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[2]
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = _filter_nonzero_measure(bp1, bp2)
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[[0, -2]])
        np.testing.assert_allclose(bp2, intersect[[1, -1]])

    def test_extrema_first_and_before_bp1():
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 20,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 20,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[2] - 1e-13
        bp1, bp2 = bounce_points(
            pitch, k[2:], B.c[:, 2:], B_z_ra.c[:, 2:], check=True, plot=False
        )
        plot_field_line(B, pitch, bp1, bp2, start=k[2])
        bp1, bp2 = _filter_nonzero_measure(bp1, bp2)
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], 0.835319, rtol=1e-6)
        intersect = intersect[intersect >= k[2]]
        np.testing.assert_allclose(bp1, intersect[[0, 2, 4]], rtol=1e-6)
        np.testing.assert_allclose(bp2, intersect[[0, 3, 5]], rtol=1e-6)

    def test_extrema_first_and_before_bp2():
        start = -1.2 * np.pi
        end = -2 * start + 1
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 10,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 10,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[1] + 1e-13
        # If a regression fails this test, this note will save many hours of debugging.
        # If the filter in place to return only the distinct roots is too coarse,
        # in particular atol < 1e-15, then this test will error. In the resulting
        # plot that the error will produce the red bounce point on the first hump
        # disappears. The true sequence is green, double red, green, red, green.
        # The first green was close to the double red and hence the first of the
        # double red root pair was erased as it was falsely detected as a duplicate.
        # The second of the double red root pair is correctly erased. All that is
        # left is the green. Now the bounce_points method assumes the intermediate
        # value theorem holds for the continuous spline, so when fed these sequence
        # of roots, the correct action is to ignore the first green root since
        # otherwise the interior of the bounce points would be hills and not valleys.
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True)
        bp1, bp2 = _filter_nonzero_measure(bp1, bp2)
        assert bp1.size and bp2.size
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], -0.671904, rtol=1e-6)
        np.testing.assert_allclose(bp1, intersect[[0, 3, 5]], rtol=1e-5)
        # intersect array could not resolve double root as single at index 0,1
        np.testing.assert_allclose(intersect[0], intersect[1], rtol=1e-5)
        np.testing.assert_allclose(bp2, intersect[[2, 4, 6]], rtol=1e-5)

    test_bp1_first()
    test_bp2_first()
    test_bp1_before_extrema()
    test_bp2_before_extrema()
    # In theory, this test should only pass if distinct=True when computing the
    # intersections in bounce points. However, we can get lucky due to floating
    # point errors, and it may also pass when distinct=False.
    test_extrema_first_and_before_bp1()
    test_extrema_first_and_before_bp2()


@pytest.mark.unit
def test_automorphism():
    """Test automorphisms."""
    a, b = -312, 786
    x = np.linspace(a, b, 10)
    y = _affine_bijection_forward(x, a, b)
    x_1 = affine_bijection(y, a, b)
    np.testing.assert_allclose(x_1, x)
    np.testing.assert_allclose(_affine_bijection_forward(x_1, a, b), y)
    np.testing.assert_allclose(automorphism_arcsin(automorphism_sin(y)), y, atol=5e-7)
    np.testing.assert_allclose(automorphism_sin(automorphism_arcsin(y)), y, atol=5e-7)

    np.testing.assert_allclose(grad_affine_bijection(a, b), 1 / (2 / (b - a)))
    np.testing.assert_allclose(
        grad_automorphism_sin(y),
        1 / grad_automorphism_arcsin(automorphism_sin(y)),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        1 / grad_automorphism_arcsin(y),
        grad_automorphism_sin(automorphism_arcsin(y)),
        atol=2e-6,
    )

    # test that floating point error is acceptable
    x = tanh_sinh(19)[0]
    assert np.all(np.abs(x) < 1)
    y = 1 / np.sqrt(1 - np.abs(x))
    assert np.isfinite(y).all()
    y = 1 / np.sqrt(1 - np.abs(automorphism_sin(x)))
    assert np.isfinite(y).all()
    y = 1 / np.sqrt(1 - np.abs(automorphism_arcsin(x)))
    assert np.isfinite(y).all()


@pytest.mark.unit
def test_bounce_quadrature():
    """Test bounce integral matches elliptic integral."""
    p = 1e-4
    m = 1 - p
    # Some prime number that doesn't appear anywhere in calculation.
    # Ensures no lucky cancellation occurs from this test case since otherwise
    # (bp2 - bp1) / pi = pi / (bp2 - bp1) which could mask errors since pi
    # appears often in transformations.
    v = 7
    truth = v * 2 * ellipkm1(p)
    rtol = 1e-4

    def integrand(B, pitch):
        return jnp.reciprocal(jnp.sqrt(1 - pitch * m * B))

    bp1 = -np.pi / 2 * v
    bp2 = -bp1
    knots = np.linspace(bp1, bp2, 50)
    B = np.clip(np.sin(knots / v) ** 2, 1e-7, 1)
    B_z_ra = np.sin(2 * knots / v) / v
    pitch = 1 + 50 * jnp.finfo(jnp.array(1.0).dtype).eps

    bounce_integrate, _ = bounce_integral(
        B, B, B_z_ra, knots, quad=tanh_sinh(40), automorphism=None, check=True
    )
    tanh_sinh_vanilla = bounce_integrate(integrand, [], pitch)
    assert np.count_nonzero(tanh_sinh_vanilla) == 1
    np.testing.assert_allclose(np.sum(tanh_sinh_vanilla), truth, rtol=rtol)
    bounce_integrate, _ = bounce_integral(
        B, B, B_z_ra, knots, quad=leggauss(25), check=True
    )
    leg_gauss_sin = bounce_integrate(integrand, [], pitch, batch=False)
    assert np.count_nonzero(tanh_sinh_vanilla) == 1
    np.testing.assert_allclose(np.sum(leg_gauss_sin), truth, rtol=rtol)


@pytest.mark.unit
def test_bounce_integral_checks():
    """Test that all the internal correctness checks pass for real example."""

    def numerator(g_zz, B, pitch):
        f = (1 - pitch * B / 2) * g_zz
        # You may need to clip and safediv to avoid nan gradient.
        return f / jnp.sqrt(1 - pitch * B)

    def denominator(B, pitch):
        # You may need to clip and safediv to avoid nan gradient.
        return 1 / jnp.sqrt(1 - pitch * B)

    # Suppose we want to compute a bounce average of the function
    # f(ℓ) = (1 − λ|B|/2) * g_zz, where g_zz is the squared norm of the
    # toroidal basis vector on some set of field lines specified by (ρ, α)
    # coordinates. This is defined as
    # [∫ f(ℓ) / √(1 − λ|B|) dℓ] / [∫ 1 / √(1 − λ|B|) dℓ]
    eq = get("HELIOTRON")
    # Clebsch-Type field-line coordinates ρ, α, ζ.
    rho = np.linspace(0.1, 1, 6)
    alpha = np.array([0])
    knots = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    grid = eq.rtz_grid(
        rho, alpha, knots, coordinates="raz", period=(np.inf, 2 * np.pi, np.inf)
    )
    data = eq.compute(
        ["B^zeta", "|B|", "|B|_z|r,a", "min_tz |B|", "max_tz |B|", "g_zz"], grid=grid
    )
    bounce_integrate, spline = bounce_integral(
        data["B^zeta"],
        data["|B|"],
        data["|B|_z|r,a"],
        knots,
        check=True,
        plot=False,
        quad=leggauss(3),  # not checking quadrature accuracy in this test
    )
    pitch = get_pitch(
        grid.compress(data["min_tz |B|"]), grid.compress(data["max_tz |B|"]), 10
    )
    # You can also plot the field line by uncommenting the following line.
    # Useful to see if the knot density was sufficient to reconstruct the field line.
    # _, _ = bounce_points(pitch, **spline, check=True, num=50000) # noqa: E800
    num = bounce_integrate(numerator, data["g_zz"], pitch)
    den = bounce_integrate(denominator, [], pitch)
    avg = num / den

    # Sum all bounce integrals across each particular field line.
    avg = np.nansum(avg, axis=-1)
    assert np.count_nonzero(avg)
    # Split the resulting data by field line.
    avg = avg.reshape(pitch.shape[0], rho.size, alpha.size)
    # The sum stored at index i, j
    i, j = 0, 0
    print(avg[:, i, j])
    # is the summed bounce average among wells along the field line with nodes
    # given in Clebsch-Type field-line coordinates ρ, α, ζ
    raz_grid = grid.source_grid
    nodes = raz_grid.nodes.reshape(rho.size, alpha.size, -1, 3)
    print(nodes[i, j])
    # for the pitch values stored in
    pitch = pitch.reshape(pitch.shape[0], rho.size, alpha.size)
    print(pitch[:, i, j])


@partial(np.vectorize, excluded={0})
def _adaptive_elliptic(integrand, k):
    a = 0
    b = 2 * np.arcsin(k)
    return integrate.quad(integrand, a, b, args=(k,), points=b)[0]


def _fixed_elliptic(integrand, k, deg):
    k = np.atleast_1d(k)
    a = np.zeros_like(k)
    b = 2 * np.arcsin(k)
    x, w = leggauss(deg)
    w = w * grad_automorphism_sin(x)
    x = automorphism_sin(x)
    Z = affine_bijection(x, a[..., np.newaxis], b[..., np.newaxis])
    k = k[..., np.newaxis]
    quad = np.dot(integrand(Z, k), w) * grad_affine_bijection(a, b)
    return quad


def _elliptic_incomplete(k2):
    K_integrand = lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * (k / 4)
    E_integrand = lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) / (k * 4)
    # Scipy's elliptic integrals are broken.
    # https://github.com/scipy/scipy/issues/20525.
    k = np.sqrt(k2)
    K = _adaptive_elliptic(K_integrand, k)
    E = _adaptive_elliptic(E_integrand, k)
    # Make sure scipy's adaptive quadrature is not broken.
    np.testing.assert_allclose(K, _fixed_elliptic(K_integrand, k, 10))
    np.testing.assert_allclose(E, _fixed_elliptic(E_integrand, k, 10))

    I_0 = 4 / k * K
    I_1 = 4 * k * E
    I_2 = 16 * k * E
    I_3 = 16 * k / 9 * (2 * (-1 + 2 * k2) * E - (-1 + k2) * K)
    I_4 = 16 * k / 3 * ((-1 + 2 * k2) * E - 2 * (-1 + k2) * K)
    I_5 = 32 * k / 30 * (2 * (1 - k2 + k2**2) * E - (1 - 3 * k2 + 2 * k2**2) * K)
    I_6 = 4 / k * (2 * k2 * E + (1 - 2 * k2) * K)
    I_7 = 2 * k / 3 * ((-2 + 4 * k2) * E - 4 * (-1 + k2) * K)
    # Check for math mistakes.
    np.testing.assert_allclose(
        I_2,
        _adaptive_elliptic(
            lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * Z * np.sin(Z), k
        ),
    )
    np.testing.assert_allclose(
        I_3,
        _adaptive_elliptic(
            lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * Z * np.sin(Z), k
        ),
    )
    np.testing.assert_allclose(
        I_4,
        _adaptive_elliptic(
            lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.sin(Z) ** 2, k
        ),
    )
    np.testing.assert_allclose(
        I_5,
        _adaptive_elliptic(
            lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.sin(Z) ** 2, k
        ),
    )
    # scipy fails
    np.testing.assert_allclose(
        I_6,
        _fixed_elliptic(
            lambda Z, k: 2 / np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.cos(Z),
            k,
            deg=10,
        ),
    )
    np.testing.assert_allclose(
        I_7,
        _adaptive_elliptic(
            lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.cos(Z), k
        ),
    )
    return I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_drift():
    """Test bounce-averaged drift with analytical expressions."""
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")
    psi_boundary = eq.Psi / (2 * np.pi)
    psi = 0.25 * psi_boundary
    rho = np.sqrt(psi / psi_boundary)
    np.testing.assert_allclose(rho, 0.5)

    # Make a set of nodes along a single fieldline.
    grid_fsa = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, sym=eq.sym, NFP=eq.NFP)
    data = eq.compute(["iota"], grid=grid_fsa)
    iota = grid_fsa.compress(data["iota"]).item()
    alpha = 0
    zeta = np.linspace(-np.pi / iota, np.pi / iota, (2 * eq.M_grid) * 4 + 1)
    grid = eq.rtz_grid(
        rho, alpha, zeta, coordinates="raz", period=(np.inf, 2 * np.pi, np.inf)
    )

    data = eq.compute(
        [
            "B^zeta",
            "|B|",
            "|B|_z|r,a",
            "cvdrift",
            "gbdrift",
            "g^pa",
            "shear",
            "iota",
            "psi",
            "a",
        ],
        grid=grid,
    )
    np.testing.assert_allclose(data["psi"], psi)
    np.testing.assert_allclose(data["iota"], iota)
    assert np.all(np.sign(data["B^zeta"]) > 0)
    data["iota"] = grid.compress(data["iota"]).item()
    data["shear"] = grid.compress(data["shear"]).item()

    L_ref = data["a"]
    B_ref = 2 * np.abs(psi_boundary) / L_ref**2
    bounce_integrate, _ = bounce_integral(
        data["B^zeta"],
        data["|B|"],
        data["|B|_z|r,a"],
        knots=zeta,
        B_ref=B_ref,
        L_ref=L_ref,
        quad=leggauss(28),  # converges to absolute and relative tolerance of 1e-7
        check=True,
    )

    B = data["|B|"] / B_ref
    B0 = np.mean(B)
    # epsilon should be changed to dimensionless, and computed in a way that
    # is independent of normalization length scales, like "effective r/R0".
    epsilon = L_ref * rho  # Aspect ratio of the flux surface.
    np.testing.assert_allclose(epsilon, 0.05)
    theta_PEST = alpha + data["iota"] * zeta
    # same as 1 / (1 + epsilon cos(theta)) assuming epsilon << 1
    B_analytic = B0 * (1 - epsilon * np.cos(theta_PEST))
    np.testing.assert_allclose(B, B_analytic, atol=3e-3)

    gradpar = L_ref * data["B^zeta"] / data["|B|"]
    # This method of computing G0 suggests a fixed point iteration.
    G0 = data["a"]
    gradpar_analytic = G0 * (1 - epsilon * np.cos(theta_PEST))
    gradpar_theta_analytic = data["iota"] * gradpar_analytic
    G0 = np.mean(gradpar_theta_analytic)
    np.testing.assert_allclose(gradpar, gradpar_analytic, atol=5e-3)

    # Comparing coefficient calculation here with coefficients from compute/_metric
    normalization = -np.sign(psi) * B_ref * L_ref**2
    cvdrift = data["cvdrift"] * normalization
    gbdrift = data["gbdrift"] * normalization
    dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * data["|B|"] ** 2)
    alpha_MHD = -0.5 * dPdrho / data["iota"] ** 2
    gds21 = -np.sign(data["iota"]) * data["shear"] * data["g^pa"] / B_ref
    gds21_analytic = -data["shear"] * (
        data["shear"] * theta_PEST - alpha_MHD / B**4 * np.sin(theta_PEST)
    )
    gds21_analytic_low_order = -data["shear"] * (
        data["shear"] * theta_PEST - alpha_MHD / B0**4 * np.sin(theta_PEST)
    )
    np.testing.assert_allclose(gds21, gds21_analytic, atol=2e-2)
    np.testing.assert_allclose(gds21, gds21_analytic_low_order, atol=2.7e-2)

    fudge_1 = 0.19
    gbdrift_analytic = fudge_1 * (
        -data["shear"]
        + np.cos(theta_PEST)
        - gds21_analytic / data["shear"] * np.sin(theta_PEST)
    )
    gbdrift_analytic_low_order = fudge_1 * (
        -data["shear"]
        + np.cos(theta_PEST)
        - gds21_analytic_low_order / data["shear"] * np.sin(theta_PEST)
    )
    fudge_2 = 0.07
    cvdrift_analytic = gbdrift_analytic + fudge_2 * alpha_MHD / B**2
    cvdrift_analytic_low_order = (
        gbdrift_analytic_low_order + fudge_2 * alpha_MHD / B0**2
    )
    np.testing.assert_allclose(gbdrift, gbdrift_analytic, atol=1e-2)
    np.testing.assert_allclose(cvdrift, cvdrift_analytic, atol=2e-2)
    np.testing.assert_allclose(gbdrift, gbdrift_analytic_low_order, atol=1e-2)
    np.testing.assert_allclose(cvdrift, cvdrift_analytic_low_order, atol=2e-2)

    pitch = get_pitch(np.min(B), np.max(B), 100)[1:]
    k2 = 0.5 * ((1 - pitch * B0) / (epsilon * pitch * B0) + 1)
    I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7 = _elliptic_incomplete(k2)
    y = np.sqrt(2 * epsilon * pitch * B0)
    I_0, I_2, I_4, I_6 = map(lambda I: I / y, (I_0, I_2, I_4, I_6))
    I_1, I_3, I_5, I_7 = map(lambda I: I * y, (I_1, I_3, I_5, I_7))

    drift_analytic_num = (
        fudge_2 * alpha_MHD / B0**2 * I_1
        - 0.5
        * fudge_1
        * (
            data["shear"] * (I_0 + I_1 - I_2 - I_3)
            + alpha_MHD / B0**4 * (I_4 + I_5)
            - (I_6 + I_7)
        )
    ) / G0
    drift_analytic_den = I_0 / G0
    drift_analytic = drift_analytic_num / drift_analytic_den

    def integrand_num(cvdrift, gbdrift, B, pitch):
        g = jnp.sqrt(1 - pitch * B)
        return (cvdrift * g) - (0.5 * g * gbdrift) + (0.5 * gbdrift / g)

    def integrand_den(B, pitch):
        return jnp.reciprocal(jnp.sqrt(1 - pitch * B))

    drift_numerical_num = bounce_integrate(
        integrand=integrand_num,
        f=[cvdrift, gbdrift],
        pitch=pitch[:, np.newaxis],
    )
    drift_numerical_den = bounce_integrate(
        integrand=integrand_den,
        f=[],
        pitch=pitch[:, np.newaxis],
    )

    drift_numerical_num = np.squeeze(drift_numerical_num[drift_numerical_num != 0])
    drift_numerical_den = np.squeeze(drift_numerical_den[drift_numerical_den != 0])
    drift_numerical = drift_numerical_num / drift_numerical_den
    msg = "There should be one bounce integral per pitch in this example."
    assert drift_numerical.size == drift_analytic.size, msg
    np.testing.assert_allclose(drift_numerical, drift_analytic, atol=5e-3, rtol=5e-2)

    fig, ax = plt.subplots()
    ax.plot(1 / pitch, drift_analytic)
    ax.plot(1 / pitch, drift_numerical)

    # Test if differentiable.
    def dummy_fun(pitch):
        return jnp.sum(bounce_integrate(integrand_num, [cvdrift, gbdrift], pitch))

    assert np.isclose(grad(dummy_fun)(1.0), 650, rtol=1e-3)

    return fig
