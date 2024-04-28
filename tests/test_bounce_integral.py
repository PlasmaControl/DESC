"""Test bounce integral methods."""

import inspect
from functools import partial

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.interpolate import CubicHermiteSpline
from scipy.special import ellipkm1

from desc.backend import complex_sqrt, flatnonzero
from desc.compute import data_index
from desc.compute.bounce_integral import (
    _affine_bijection_forward,
    _filter_not_nan,
    _poly_der,
    _poly_root,
    _poly_val,
    affine_bijection_reverse,
    automorphism_arcsin,
    automorphism_sin,
    bounce_integral,
    bounce_points,
    composite_linspace,
    desc_grid_from_field_line_coords,
    get_extrema,
    grad_affine_bijection_reverse,
    grad_automorphism_arcsin,
    grad_automorphism_sin,
    plot_field_line_with_ripple,
    take_mask,
    tanh_sinh_quad,
)
from desc.compute.utils import dot, get_data_deps, safediv
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import Grid, LinearGrid
from desc.utils import errorif, only1


def _sqrt(x):
    """Reproduces jnp.sqrt with np.sqrt."""
    x = complex_sqrt(x)
    x = np.where(np.isclose(np.imag(x), 0), np.real(x), np.nan)
    return x


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
    taken = take_mask(a, ~np.isnan(a))
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
    grid = Grid.create_meshgrid(rho, alpha, zeta)
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
    # test final reshape of bounce integral result won't mix data
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
    src = inspect.getsource(bounce_integral)
    assert "S, knots.size" in src, err_msg
    assert "pitch.shape[0], rho.size, alpha.size" in src, err_msg


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
        root_filter = _filter_not_nan(root[j])
        assert root_filter.size == unique_roots.size
        np.testing.assert_allclose(
            actual=root_filter,
            desired=unique_roots,
            err_msg=str(j),
        )
    c = np.array([0, 1, -1, -8, 12])
    root = _filter_not_nan(_poly_root(c, sort=True, distinct=True))
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
    extrema = _filter_not_nan(extrema)
    assert extrema.size == extrema_scipy.size
    np.testing.assert_allclose(extrema, extrema_scipy, rtol=rtol + eps)


@pytest.mark.unit
def test_composite_linspace():
    """Test this utility function useful for Newton-Cotes integration over pitch."""
    B_min_tz = np.array([0.1, 0.2])
    B_max_tz = np.array([1, 3])
    breaks = np.linspace(B_min_tz, B_max_tz, num=5)
    b = composite_linspace(breaks, resolution=3)
    print(breaks)
    print(b)
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
        pitch = 2
        bp1, bp2 = bounce_points(
            pitch, knots, B.c, B.derivative().c, check=True, plot=False
        )
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[0::2])
        np.testing.assert_allclose(bp2, intersect[1::2])

    def test_bp2_first():
        start = -3 * np.pi
        end = -start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(k, np.cos(k), -np.sin(k))
        pitch = 2
        bp1, bp2 = bounce_points(
            pitch, k, B.c, B.derivative().c, check=True, plot=False
        )
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[1::2])
        np.testing.assert_allclose(bp2, intersect[0::2][1:])

    def test_bp1_before_extrema():
        start = -np.pi
        end = -2 * start
        k = np.linspace(start, end, 5)
        B = CubicHermiteSpline(
            k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[3]
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True, plot=False)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        assert bp1.size and bp2.size
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[1], 1.9827671337414938)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[1]), bp1[1])
        np.testing.assert_allclose(bp1, intersect[[1, 2]])
        np.testing.assert_allclose(bp2, intersect[[2, 3]])

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
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True, plot=False)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        assert bp1.size and bp2.size
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1, intersect[[0, -2]])
        np.testing.assert_allclose(bp2, intersect[[1, -1]])

    def test_extrema_first_and_before_bp1(plot=False):
        start = -1.2 * np.pi
        end = -2 * start
        k = np.linspace(start, end, 7)
        B = CubicHermiteSpline(
            k,
            np.cos(k) + 2 * np.sin(-2 * k) + k / 20,
            -np.sin(k) - 4 * np.cos(-2 * k) + 1 / 20,
        )
        B_z_ra = B.derivative()
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[2]
        bp1, bp2 = bounce_points(
            pitch, k[2:], B.c[:, 2:], B_z_ra.c[:, 2:], check=True, plot=False
        )
        if plot:
            plot_field_line_with_ripple(B, pitch, bp1, bp2, start=k[2])
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        assert bp1.size and bp2.size
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], 0.8353192766102349)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[0]), bp1[0])
        intersect = intersect[intersect >= k[2]]
        np.testing.assert_allclose(bp1, intersect[[0, 1, 3]])
        np.testing.assert_allclose(bp2, intersect[[0, 2, 4]])

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
        pitch = 1 / B(B_z_ra.roots(extrapolate=False))[1]
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
        bp1, bp2 = bounce_points(pitch, k, B.c, B_z_ra.c, check=True, plot=False)
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        assert bp1.size and bp2.size
        # Our routine correctly detects intersection, while scipy, jnp.root fails.
        intersect = B.solve(1 / pitch, extrapolate=False)
        np.testing.assert_allclose(bp1[0], -0.6719044147510538)
        intersect = np.insert(intersect, np.searchsorted(intersect, bp1[0]), bp1[0])
        np.testing.assert_allclose(bp1, intersect[0::2])
        np.testing.assert_allclose(bp2, intersect[1::2])

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
    x_1 = affine_bijection_reverse(y, a, b)
    np.testing.assert_allclose(x_1, x)
    np.testing.assert_allclose(_affine_bijection_forward(x_1, a, b), y)
    np.testing.assert_allclose(automorphism_arcsin(automorphism_sin(y)), y, atol=5e-7)
    np.testing.assert_allclose(automorphism_sin(automorphism_arcsin(y)), y, atol=5e-7)

    np.testing.assert_allclose(
        grad_affine_bijection_reverse(a, b),
        1 / (2 / (b - a)),
    )
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
    x, w = tanh_sinh_quad(19)
    assert np.all(np.abs(x) < 1)
    y = 1 / (1 - np.abs(x))
    assert np.isfinite(y).all()
    y = 1 / (1 - np.abs(automorphism_sin(x)))
    assert np.isfinite(y).all()
    y = 1 / (1 - np.abs(automorphism_arcsin(x)))
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
    rtol = 1e-3

    def integrand(B, pitch, Z):
        return 1 / np.sqrt(1 - pitch * m * B)

    bp1 = -np.pi / 2 * v
    bp2 = -bp1
    knots = np.linspace(bp1, bp2, 15)
    B_sup_z = np.ones(knots.size)
    B = np.clip(np.sin(knots / v) ** 2, 1e-7, 1)
    B_z_ra = np.sin(2 * knots / v) / v
    pitch = 1 + np.finfo(np.array(1.0).dtype).eps

    bounce_integrate, _ = bounce_integral(
        B_sup_z,
        B,
        B_z_ra,
        knots,
        quad=tanh_sinh_quad,
        automorphism=(automorphism_arcsin, grad_automorphism_arcsin),
        resolution=18,
        check=True,
        plot=False,
    )
    tanh_sinh_arcsin = _filter_not_nan(bounce_integrate(integrand, [], pitch))
    assert tanh_sinh_arcsin.size == 1
    np.testing.assert_allclose(tanh_sinh_arcsin, truth, rtol=rtol)

    bounce_integrate, _ = bounce_integral(
        B_sup_z,
        B,
        B_z_ra,
        knots,
        quad=np.polynomial.legendre.leggauss,
        automorphism=(automorphism_sin, grad_automorphism_sin),
        deg=16,
        check=True,
        plot=False,
    )
    leg_gauss_sin = _filter_not_nan(bounce_integrate(integrand, [], pitch))
    assert leg_gauss_sin.size == 1
    np.testing.assert_allclose(leg_gauss_sin, truth, rtol=rtol)


@pytest.mark.unit
def test_example_bounce_integral():
    """Test example code in bounce_integral docstring."""
    # This test also stress tests the bounce_points routine because
    # the |B| spline that is generated from this combination of knots
    # equilibrium etc. has many edge cases for bounce point computations.
    eq = get("HELIOTRON")
    rho = np.linspace(1e-12, 1, 6)
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 5)
    knots = np.linspace(-3 * np.pi, 3 * np.pi, 40)
    grid_desc, grid_fl = desc_grid_from_field_line_coords(eq, rho, alpha, knots)
    data = eq.compute(
        ["B^zeta", "|B|", "|B|_z|r,a", "g_zz"],
        grid=grid_desc,
        override_grid=False,  # Need to have this.
    )
    bounce_integrate, spline = bounce_integral(
        data["B^zeta"],
        data["|B|"],
        data["|B|_z|r,a"],
        knots,
        check=True,
        plot=False,
    )

    def integrand_num(g_zz, B, pitch, Z):
        """Integrand in integral in numerator of bounce average."""
        f = (1 - pitch * B) * g_zz
        return safediv(f, _sqrt(1 - pitch * B))

    def integrand_den(B, pitch, Z):
        """Integrand in integral in denominator of bounce average."""
        return safediv(1, _sqrt(1 - pitch * B))

    pitch = 1 / get_extrema(**spline)
    num = bounce_integrate(integrand_num, data["g_zz"], pitch)
    den = bounce_integrate(integrand_den, [], pitch)
    average = num / den
    assert np.isfinite(average).any()

    # Now we can group the data by field line.
    average = average.reshape(pitch.shape[0], rho.size, alpha.size, -1)
    # The bounce averages stored at index i, j
    i, j = 0, 0
    print(average[:, i, j])
    # are the bounce averages along the field line with nodes
    # given in Clebsch-Type field-line coordinates ρ, α, ζ
    nodes = grid_fl.nodes.reshape(rho.size, alpha.size, -1, 3)
    print(nodes[i, j])
    # for the pitch values stored in
    pitch = pitch.reshape(pitch.shape[0], rho.size, alpha.size)
    print(pitch[:, i, j])
    # Some of these bounce averages will evaluate as nan.
    # You should filter out these nan values when computing stuff.
    print(np.nansum(average, axis=-1))


@partial(np.vectorize, excluded={0})
def _adaptive_elliptic(integrand, k):
    return integrate.quad(integrand, 0, 2 * np.arcsin(k), args=(k,))[0]


def _fixed_elliptic(integrand, k, resolution):
    k = np.atleast_1d(k)
    a = np.zeros_like(k)
    b = 2 * np.arcsin(k)
    x, w = tanh_sinh_quad(resolution, grad_automorphism_arcsin)
    Z = affine_bijection_reverse(
        automorphism_arcsin(x), a[..., np.newaxis], b[..., np.newaxis]
    )
    k = k[..., np.newaxis]
    quad = np.dot(integrand(Z, k), w) * grad_affine_bijection_reverse(a, b)
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
    np.testing.assert_allclose(K, _fixed_elliptic(K_integrand, k, 9), rtol=1e-3)
    np.testing.assert_allclose(E, _fixed_elliptic(E_integrand, k, 9), rtol=1e-3)

    # Here are the notes that explain these integrals.
    # https://github.com/PlasmaControl/DESC/files/15010927/bavg.pdf.
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
            resolution=9,
        ),
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        I_7,
        _adaptive_elliptic(
            lambda Z, k: 2 * np.sqrt(k**2 - np.sin(Z / 2) ** 2) * np.cos(Z), k
        ),
    )
    return I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7


def _compute_field_line_data(
    eq, rho, alpha, field_line_names, other_0d_or_1dr_names=None
):
    """Compute field line quantities on correct grids.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to compute on.
    rho : Array
        Field line radial label.
    alpha : Array
        Field line poloidal label.
    field_line_names : list
        Field line quantities that will be computed on the returned field line grid.
    other_0d_or_1dr_names : list, optional
        Other quantities to compute that are constant throughout volume or over
        flux surface.

    Returns
    -------
    data : dict
        Computed quantities.
    grid_desc : Grid
        Grid on which the returned quantities can be broadcast on.
    grid_fl : Grid
        Clebsch-Type field-line coordinates corresponding to above grid.
    zeta : Array
        Zeta values along field line.

    """
    errorif(alpha != 0, NotImplementedError)
    if other_0d_or_1dr_names is None:
        other_0d_or_1dr_names = []
    other_0d_or_1dr_names.append("iota")
    p = "desc.equilibrium.equilibrium.Equilibrium"
    # Gather dependencies of given quantities.
    deps = (
        get_data_deps(field_line_names + other_0d_or_1dr_names, obj=p, has_axis=False)
        + other_0d_or_1dr_names
    )
    deps = list(set(deps))
    # Create grid with given flux surfaces.
    grid1dr = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, sym=eq.sym, NFP=eq.NFP)
    # Compute dependencies on correct grids.
    seed_data = eq.compute(deps, grid=grid1dr)
    dep1dr = {dep for dep in deps if data_index[p][dep]["coordinates"] == "r"}
    dep0d = {dep for dep in deps if data_index[p][dep]["coordinates"] == ""}

    # Make a set of nodes along a single fieldline.
    iota = grid1dr.compress(seed_data["iota"]).item()
    zeta = np.linspace(-np.pi / iota, np.pi / iota, (2 * eq.M_grid) * 4 + 1)
    # Make grid that can separate into field lines via a reshape operation,
    # as expected by bounce_integral().
    grid_desc, grid_fl = desc_grid_from_field_line_coords(eq, rho, alpha, zeta)

    # Collect quantities that can be used as a seed to compute the
    # field line quantities over the grid mapped from field line coordinates.
    # (Single field line grid won't have enough poloidal resolution to
    # compute these quantities accurately).
    data0d = {key: val for key, val in seed_data.items() if key in dep0d}
    data1d = {
        key: grid_desc.copy_data_from_other(val, grid1dr)
        for key, val in seed_data.items()
        if key in dep1dr
    }
    data = {}
    data.update(data0d)
    data.update(data1d)
    # Compute field line quantities with precomputed dependencies.
    data = eq.compute(
        names=field_line_names, grid=grid_desc, data=data, override_grid=False
    )
    return data, grid_desc, grid_fl, zeta


@pytest.mark.unit
def test_bounce_averaged_drifts():
    """Test bounce-averaged drift with analytical expressions.

    Calculate bounce-averaged drifts using the bounce-average routine and
    compare it with the analytical expression
    # Note 2: Remove tests/test_equilibrium :: test_shifted_circle_geometry
    # once all the epsilons and Gammas have been implemented and tested
    """
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")
    psi_boundary = eq.Psi / (2 * np.pi)
    psi = 0.25 * psi_boundary
    rho = np.sqrt(psi / psi_boundary)
    assert np.isclose(rho, 0.5)
    alpha = 0
    data, grid, grid_fl, zeta = _compute_field_line_data(
        eq,
        rho,
        alpha,
        field_line_names=[
            "B^zeta",
            "|B|",
            "|B|_z|r,a",
            "cvdrift",
            "gbdrift",
            "grad(alpha)",
            "grad(psi)",
        ],
        other_0d_or_1dr_names=["iota_r", "a", "psi"],
    )
    assert np.allclose(data["psi"], psi)

    # normalization
    L_ref = data["a"]
    # FIXME:
    #  When we use psi, numerical and analytic match better.
    #  When we use the psi at the lcfs, plots match worse.
    #  Need to make sure we are using the correct psi in the numerical
    #  computations in this test and elsewhere in DESC.
    B_ref = 2 * np.abs(psi_boundary) / L_ref**2

    quad_resolution = 50
    monotonic = False
    bounce_integrate, _ = bounce_integral(
        data["B^zeta"],
        data["|B|"],
        data["|B|_z|r,a"],
        knots=zeta,
        B_ref=B_ref,
        L_ref=L_ref,
        quad=tanh_sinh_quad,
        automorphism=(automorphism_arcsin, grad_automorphism_arcsin),
        resolution=quad_resolution,
        # quad=np.polynomial.legendre.leggauss,  # noqa: E800
        # automorphism=(automorphism_sin, grad_automorphism_sin),  # noqa: E800
        # deg=quad_resolution,  # noqa: E800
        check=True,
        plot=False,
        monotonic=monotonic,
    )

    epsilon = L_ref * rho
    # I wouldn't really consider 0.05 << 1... maybe for a rough approximation.
    assert np.isclose(epsilon, 0.05)
    x = L_ref * rho

    iota = grid.compress(data["iota"]).item()
    theta_PEST = alpha + iota * zeta
    B_normalized = data["|B|"] / B_ref
    B0 = np.mean(B_normalized)
    # same as 1 / (1 + epsilon cos(theta)) assuming epsilon << 1
    taylor = 1 - epsilon * np.cos(theta_PEST)
    B_normalized_analytic = B0 * taylor
    np.testing.assert_allclose(B_normalized, B_normalized_analytic, atol=3e-3)

    shear = grid.compress(data["iota_r"]).item()
    s_hat = -x / iota * shear / L_ref
    gradpar = L_ref * data["B^zeta"] / data["|B|"]
    gradpar_analytic = L_ref * taylor
    gradpar_theta_analytic = iota * gradpar_analytic
    G0 = np.mean(gradpar_theta_analytic)
    np.testing.assert_allclose(gradpar, gradpar_analytic, atol=5e-3)

    # Comparing coefficient calculation here with coefficients from compute/_metric
    cvdrift = -2 * np.sign(psi) * B_ref * L_ref**2 * rho * data["cvdrift"]
    gbdrift = -2 * np.sign(psi) * B_ref * L_ref**2 * rho * data["gbdrift"]
    dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * data["|B|"] ** 2)
    alpha_MHD = -np.mean(dPdrho / iota**2 * 0.5)
    gds21 = -np.sign(iota) * dot(data["grad(psi)"], data["grad(alpha)"]) * s_hat / B_ref
    gds21_analytic = (
        -1
        * s_hat
        * (s_hat * theta_PEST - alpha_MHD / B_normalized**4 * np.sin(theta_PEST))
    )
    np.testing.assert_allclose(gds21, gds21_analytic, atol=2e-2)

    fudge_factor_gbdrift = 0.19
    gbdrift_analytic = fudge_factor_gbdrift * (
        -s_hat + (np.cos(theta_PEST) - gds21_analytic / s_hat * np.sin(theta_PEST))
    )
    fudge_factor_cvdrift = 0.07
    cvdrift_analytic = (
        gbdrift_analytic + fudge_factor_cvdrift * alpha_MHD / B_normalized**2
    )
    np.testing.assert_allclose(gbdrift, gbdrift_analytic, atol=1e-2)
    np.testing.assert_allclose(cvdrift, cvdrift_analytic, atol=2e-2)

    # Values of pitch angle lambda for which to evaluate the bounce averages.
    delta_shift = 1e-6
    pitch_resolution = 50
    pitch = np.linspace(
        1 / np.max(B_normalized) + delta_shift,
        1 / np.min(B_normalized) - delta_shift,
        pitch_resolution,
    )
    k2 = 0.5 * ((1 - pitch * B0) / (epsilon * pitch * B0) + 1)
    I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7 = _elliptic_incomplete(k2)
    bounce_drift_analytic = (
        fudge_factor_cvdrift * alpha_MHD / B0**2 * I_1
        - 0.5
        * fudge_factor_gbdrift
        * (
            s_hat * (I_0 + I_1 - I_2 - I_3)
            + alpha_MHD / B0**4 * (I_4 + I_5)
            - (I_6 + I_7)
        )
    ) / G0

    def integrand(cvdrift, gbdrift, B, pitch, Z):
        g = _sqrt(1 - pitch * B)
        return (cvdrift * g) - (0.5 * g * gbdrift) + (0.5 * gbdrift / g)

    method = "akima"
    bounce_drift = bounce_integrate(
        integrand=integrand,
        f=[cvdrift, gbdrift],
        pitch=pitch.reshape(pitch_resolution, -1),
        method=method,
    )
    bounce_drift = np.squeeze(_filter_not_nan(bounce_drift))
    msg = "There should be one bounce integral per pitch in this example."
    assert bounce_drift.size == bounce_drift_analytic.size, msg

    fig, ax = plt.subplots(2)
    ax[0].plot(1 / pitch, bounce_drift_analytic, marker="o", label="analytic")
    ax[0].plot(1 / pitch, bounce_drift, marker="x", label="numerical")
    ax[0].set_xlabel(r"$1 / \lambda$")
    ax[0].set_ylabel("Bounce averaged drift")
    ax[0].legend()
    ax[1].plot(1 / pitch, bounce_drift_analytic, marker="o", label="analytic")
    ax[1].set_xlabel(r"$1 / \lambda$")
    ax[1].set_ylabel("Bounce averaged drift")
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    msg = (
        f"Quadrature resolution is {quad_resolution}.\n"
        f"Delta shift is {delta_shift}.\n"
        f"Spline method for integrand quantities is {method}.\n"
        f"Spline method for |B| is monotonic? (as opposed to Hermite): {monotonic}.\n"
    )
    np.testing.assert_allclose(
        bounce_drift, bounce_drift_analytic, atol=2e-2, rtol=1e-2, err_msg=msg
    )
