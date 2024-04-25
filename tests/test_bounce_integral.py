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
from desc.compute.bounce_integral import (
    _affine_bijection_forward,
    _bounce_quadrature,
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
    grad_affine_bijection_reverse,
    grad_automorphism_arcsin,
    grad_automorphism_sin,
    pitch_of_extrema,
    plot_field_line_with_ripple,
    take_mask,
    tanh_sinh_quad,
)
from desc.compute.utils import dot, safediv
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.objectives import (
    ObjectiveFromUser,
    ObjectiveFunction,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile
from desc.utils import only1


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
    r, a, z = map(np.ravel, np.meshgrid(rho, alpha, zeta, indexing="ij"))
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
    src = inspect.getsource(desc_grid_from_field_line_coords)
    assert 'indexing="ij"' in src, err_msg
    assert 'meshgrid(rho, alpha, zeta, indexing="ij")' in src, err_msg


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
        np.testing.assert_allclose(
            actual=_filter_not_nan(root[j]),
            desired=unique_roots,
            err_msg=str(j),
        )
    c = np.array([0, 1, -1, -8, 12])
    np.testing.assert_allclose(
        actual=_filter_not_nan(_poly_root(c, sort=True, distinct=True)),
        desired=np.unique(np.roots(c)),
    )


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
def test_pitch_of_extrema():
    """Test that these pitch intersect extrema of |B|."""
    start = -np.pi
    end = -2 * start
    k = np.linspace(start, end, 5)
    B = CubicHermiteSpline(
        k, np.cos(k) + 2 * np.sin(-2 * k), -np.sin(k) - 4 * np.cos(-2 * k)
    )
    B_z_ra = B.derivative()
    pitch_scipy = 1 / B(B_z_ra.roots(extrapolate=False))
    rtol = 1e-7
    pitch = pitch_of_extrema(k, B.c, B_z_ra.c, relative_shift=rtol)
    eps = 100 * np.finfo(float).eps
    np.testing.assert_allclose(
        np.sort(_filter_not_nan(pitch)), np.sort(pitch_scipy), rtol=rtol + eps
    )


@pytest.mark.unit
def test_composite_linspace():
    """Test this utility function useful for Newton-Cotes integration over pitch."""
    B_min_tz = np.array([0.1, 0.2])
    B_max_tz = np.array([1, 3])
    pitch_knot = np.linspace(1 / B_min_tz, 1 / B_max_tz, num=5)
    b_knot = 1 / pitch_knot
    b = composite_linspace(b_knot, resolution=3)
    print(b_knot)
    print(b)
    np.testing.assert_allclose(b, np.sort(b, axis=0), atol=0, rtol=0)
    for i in range(pitch_knot.shape[0]):
        for j in range(pitch_knot.shape[1]):
            assert only1(np.isclose(b_knot[i, j], b[:, j]).tolist())


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
    np.testing.assert_allclose(automorphism_arcsin(automorphism_sin(y)), y, atol=1e-6)
    np.testing.assert_allclose(automorphism_sin(automorphism_arcsin(y)), y, atol=1e-6)

    np.testing.assert_allclose(
        grad_affine_bijection_reverse(a, b),
        1 / (2 / (b - a)),
    )
    np.testing.assert_allclose(
        grad_automorphism_sin(y),
        1 / grad_automorphism_arcsin(automorphism_sin(y)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        1 / grad_automorphism_arcsin(y),
        grad_automorphism_sin(automorphism_arcsin(y)),
        atol=2e-6,
    )

    # test that floating point error is acceptable
    x, w = tanh_sinh_quad(19)
    assert np.all(np.abs(x) < 1)
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

    bp1 = -np.pi / 2 * v
    bp2 = -bp1
    knots = np.linspace(bp1, bp2, 15)
    bp1 = np.atleast_3d(bp1)
    bp2 = np.atleast_3d(bp2)
    B_sup_z = np.ones((1, knots.size))
    B = (np.sin(knots / v) ** 2).reshape(1, -1)
    B_z_ra = (np.sin(2 * knots / v) / v).reshape(1, -1)
    pitch = np.ones((1, 1))

    def integrand(B, pitch, Z):
        return 1 / np.sqrt(1 - pitch * m * B)

    # augment the singularity
    x_t, w_t = tanh_sinh_quad(18, grad_automorphism_arcsin)
    x_t = automorphism_arcsin(x_t)
    tanh_sinh_arcsin = _bounce_quadrature(
        bp1,
        bp2,
        x_t,
        w_t,
        integrand,
        [],
        B_sup_z,
        B,
        B_z_ra,
        pitch,
        knots,
        check=True,
    )
    np.testing.assert_allclose(tanh_sinh_arcsin, truth, rtol=rtol)
    x_g, w_g = np.polynomial.legendre.leggauss(16)
    # suppress the singularity
    w_g = w_g * grad_automorphism_sin(x_g)
    x_g = automorphism_sin(x_g)
    leg_gauss_sin = _bounce_quadrature(
        bp1,
        bp2,
        x_g,
        w_g,
        integrand,
        [],
        B_sup_z,
        B,
        B_z_ra,
        pitch,
        knots,
        check=True,
    )
    np.testing.assert_allclose(leg_gauss_sin, truth, rtol=rtol)


@pytest.mark.unit
def test_example_bounce_integral():
    """Test example code in bounce_integral docstring."""
    # This test also stress tests the bounce_points routine because
    # the |B| spline that is generated from this combination of knots
    # equilibrium etc. has many edge cases for bounce point computations.

    def integrand_num(g_zz, B, pitch, Z):
        """Integrand in integral in numerator of bounce average."""
        f = (1 - pitch * B) * g_zz
        return safediv(f, _sqrt(1 - pitch * B))

    def integrand_den(B, pitch, Z):
        """Integrand in integral in denominator of bounce average."""
        return safediv(1, _sqrt(1 - pitch * B))

    eq = get("HELIOTRON")
    rho = np.linspace(1e-12, 1, 6)
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 5)

    bounce_integrate, items = bounce_integral(eq, rho, alpha, check=True, plot=False)
    g_zz = eq.compute("g_zz", grid=items["grid_desc"])["g_zz"]
    pitch = pitch_of_extrema(items["knots"], items["B.c"], items["B_z_ra.c"])
    num = bounce_integrate(integrand_num, g_zz, pitch)
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
    nodes = items["grid_fl"].nodes.reshape(rho.size, alpha.size, -1, 3)
    print(nodes[i, j])
    # for the pitch values stored in
    pitch = pitch.reshape(pitch.shape[0], rho.size, alpha.size)
    print(pitch[:, i, j])
    # Some of these bounce averages will evaluate as nan.
    # You should filter out these nan values when computing stuff.
    print(np.nansum(average, axis=-1))


@pytest.mark.unit
def test_integral_0(k=0.9, resolution=10):
    """4 / k * ellipkinc(np.arcsin(k), 1 / k**2)."""
    k = np.atleast_1d(k)
    bp1 = np.zeros_like(k)
    bp2 = np.arcsin(k)
    x, w = tanh_sinh_quad(resolution, grad_automorphism_arcsin)
    Z = affine_bijection_reverse(
        automorphism_arcsin(x), bp1[..., np.newaxis], bp2[..., np.newaxis]
    )
    k = k[..., np.newaxis]

    def integrand(Z, k):
        return safediv(4 / k, np.sqrt(1 - 1 / k**2 * np.sin(Z) ** 2))

    quad = np.dot(integrand(Z, k), w) * grad_affine_bijection_reverse(bp1, bp2)
    if k.size == 1:
        q = integrate.quad(integrand, bp1.item(), bp2.item(), args=(k.item(),))[0]
        np.testing.assert_allclose(quad, q, rtol=1e-5)
    return quad


@pytest.mark.unit
def test_integral_1(k=0.9, resolution=10):
    """4 * k * ellipeinc(np.arcsin(k), 1 / k**2)."""
    k = np.atleast_1d(k)
    bp1 = np.zeros_like(k)
    bp2 = np.arcsin(k)
    x, w = tanh_sinh_quad(resolution, grad_automorphism_arcsin)
    Z = affine_bijection_reverse(
        automorphism_arcsin(x), bp1[..., np.newaxis], bp2[..., np.newaxis]
    )
    k = k[..., np.newaxis]

    def integrand(Z, k):
        return 4 * k * np.sqrt(1 - 1 / k**2 * np.sin(Z) ** 2)

    quad = np.dot(integrand(Z, k), w) * grad_affine_bijection_reverse(bp1, bp2)
    if k.size == 1:
        q = integrate.quad(integrand, bp1.item(), bp2.item(), args=(k.item(),))[0]
        np.testing.assert_allclose(quad, q, rtol=1e-4)
    return quad


@pytest.mark.unit
def test_bounce_averaged_drifts():
    """Test bounce-averaged drift with analytical expressions.

    Calculate bounce-averaged drifts using the bounce-average routine and
    compare it with the analytical expression
    # Note 1: This test can be merged with the low beta test
    # Note 2: Remove tests/test_equilibrium :: test_shifted_circle_geometry
    # once all the epsilons and Gammas have been implemented and tested
    """
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")
    psi = 0.25  # normalized psi
    rho = np.sqrt(psi)
    data = eq.compute(["iota", "iota_r", "a", "rho", "psi"])

    # normalization
    L_ref = data["a"]
    epsilon = L_ref * rho
    psi_boundary = data["psi"][np.argmax(np.abs(data["psi"]))]
    B_ref = 2 * np.abs(psi_boundary) / L_ref**2

    # Creating a grid along a field line
    iota = np.interp(rho, data["rho"], data["iota"])
    shear = np.interp(rho, data["rho"], data["iota_r"])
    N = (2 * eq.M_grid) * 4 + 1
    zeta = np.linspace(-np.pi / iota, np.pi / iota, N)
    alpha = 0
    theta_PEST = alpha + iota * zeta
    # TODO: Request: The bounce integral operator should be able to take a grid.
    #       Response: Currently the API is such that the method does all the
    #                 above preprocessing for you. Let's test it for correctness
    #                 first then do this later.

    resolution = 50
    # Whether to use monotonic or Hermite splines to interpolate |B|.
    monotonic = False
    bounce_integrate, items = bounce_integral(
        eq=eq,
        rho=rho,
        alpha=alpha,
        knots=zeta,
        B_ref=B_ref,
        L_ref=L_ref,
        check=True,
        plot=True,
        resolution=resolution,
        monotonic=monotonic,
    )
    data_keys = [
        "|grad(psi)|^2",
        "grad(psi)",
        "B",
        "iota",
        "|B|",
        "B^zeta",
        "cvdrift0",
        "cvdrift",
        "gbdrift",
    ]
    # FIXME (outside scope of the bounce branch):
    #  override_grid should not be required for the test to pass.
    #  and anytime override_grid is true we should print a blue warning.
    data_bounce = eq.compute(data_keys, grid=items["grid_desc"], override_grid=False)

    # normalizations
    bmag = data_bounce["|B|"] / B_ref
    B0 = np.mean(bmag)
    bmag_analytic = B0 * (1 - epsilon * np.cos(theta_PEST))
    np.testing.assert_allclose(bmag, bmag_analytic, atol=5e-3, rtol=5e-3)

    x = L_ref * rho  # same as epsilon?
    s_hat = -x / iota * shear / L_ref
    gradpar = L_ref * data_bounce["B^zeta"] / data_bounce["|B|"]
    gradpar_analytic = (
        2 * L_ref * data_bounce["iota"] * (1 - epsilon * np.cos(theta_PEST))
    )
    np.testing.assert_allclose(gradpar, gradpar_analytic, atol=9e-3, rtol=5e-3)

    # Comparing coefficient calculation here with coefficients from compute/_metric
    cvdrift = (
        -2 * np.sign(psi_boundary) * B_ref * L_ref**2 * rho * data_bounce["cvdrift"]
    )
    gbdrift = (
        -2 * np.sign(psi_boundary) * B_ref * L_ref**2 * rho * data_bounce["gbdrift"]
    )
    dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * data_bounce["|B|"] ** 2)
    alpha_MHD = -np.mean(dPdrho * 1 / data_bounce["iota"] ** 2 * 0.5)

    gds21 = (
        -np.sign(iota)
        * dot(data_bounce["grad(psi)"], data_bounce["grad(alpha)"])
        * s_hat
        / B_ref
    )
    gds21_analytic = (
        -1 * s_hat * (s_hat * theta_PEST - alpha_MHD / bmag**4 * np.sin(theta_PEST))
    )
    np.testing.assert_allclose(gds21, gds21_analytic, atol=1.7e-2, rtol=5e-4)

    fudge_factor_gbdrift = 0.19
    gbdrift_analytic = fudge_factor_gbdrift * (
        -s_hat + (np.cos(theta_PEST) - gds21_analytic / s_hat * np.sin(theta_PEST))
    )
    fudge_factor_cvdrift = 0.07
    cvdrift_analytic = gbdrift_analytic + fudge_factor_cvdrift * alpha_MHD / bmag**2
    np.testing.assert_allclose(gbdrift, gbdrift_analytic, atol=1.2e-2, rtol=5e-3)
    np.testing.assert_allclose(cvdrift, cvdrift_analytic, atol=1.8e-2, rtol=5e-3)

    # Values of pitch angle lambda for which to evaluate the bounce averages.
    delta_shift = 1e-6
    pitch_resolution = 50
    pitch = np.linspace(
        1 / np.max(bmag) + delta_shift, 1 / np.min(bmag) - delta_shift, pitch_resolution
    )
    k2 = 0.5 * ((1 - pitch * B0) / (pitch * B0 * epsilon) + 1)
    k = np.sqrt(k2)
    # Here are the notes that explain these integrals.
    # https://github.com/PlasmaControl/DESC/files/15010927/bavg.pdf.
    I_0 = test_integral_0(k, resolution)
    I_1 = test_integral_1(k, resolution)
    I_2 = 16 * k * I_0
    I_3 = 4 / 9 * (8 * k * (-1 + 2 * k2) * I_1 - 4 * k * (-1 + k2) * I_0)
    I_4 = (
        2
        * np.sqrt(2)
        / 3
        * (4 * np.sqrt(2) * k * (-1 + 2 * k2) * I_0 - 2 * (-1 + k2) * I_1)
    )
    I_5 = (
        2
        / 30
        * (32 * k * (1 - k2 + k2**2) * I_0 - 16 * k * (1 - 3 * k2 + 2 * k2**2) * I_1)
    )
    I_6 = 2 / 3 * (k * (-2 + 4 * k2) * I_0 - 4 * (-1 + k2) * I_1)
    I_7 = 4 / k * (2 * k2 * I_0 + (1 - 2 * k2) * I_1)

    bounce_drift_analytic = (
        fudge_factor_cvdrift * dPdrho / B0**2 * I_1
        - 0.5
        * fudge_factor_gbdrift
        * (
            s_hat * (I_0 + I_1 + I_2 + I_3)
            + alpha_MHD / B0**4 * (I_4 + I_5)
            - (I_6 + I_7)
        )
    )

    def integrand(cvdrift, gbdrift, B, pitch, Z):
        g = _sqrt(1 - pitch * B)
        return (cvdrift * g) - (0.5 * g * gbdrift) + (0.5 * gbdrift / g)

    # Can choose method of interpolation for all quantities besides |B| from
    # interpax.readthedocs.io/en/latest/_api/interpax.interp1d.html#interpax.interp1d.
    method = "akima"
    bounce_drift = bounce_integrate(
        integrand=integrand,
        f=[cvdrift, gbdrift],
        pitch=pitch.reshape(pitch_resolution, -1),
        method=method,
    )
    # There is only one bounce integral per pitch in this example.
    bounce_drift = np.squeeze(_filter_not_nan(bounce_drift))
    assert bounce_drift.shape == bounce_drift_analytic.shape

    plt.plot(1 / pitch, bounce_drift_analytic, marker="o", label="analytic")
    plt.plot(1 / pitch, bounce_drift, marker="x", label="numerical")
    plt.xlabel(r"$1 / \lambda$")
    plt.ylabel("Bounce averaged drift")
    plt.legend()
    plt.tight_layout()
    plt.show()
    msg = (
        "Maybe tune these parameters?\n"
        f"Quadrature resolution is {resolution}.\n"
        f"Delta shift is {delta_shift}.\n"
        f"Spline method for integrand quantities is {method}.\n"
        f"Spline method for |B| is monotonic? (as opposed to Hermite): {monotonic}.\n"
        f"Fudge factors: {fudge_factor_gbdrift}, {fudge_factor_cvdrift}.\n"
    )
    np.testing.assert_allclose(
        bounce_drift, bounce_drift_analytic, atol=2e-2, rtol=1e-2, err_msg=msg
    )


@pytest.mark.regression
def test_bounce_averaged_drifts_low_beta():
    """Test bounce integrals in low beta limit."""
    assert False, "Test not finished yet."
    L, M, N, NFP, sym = 6, 6, 6, 1, True
    surface = FourierRZToroidalSurface(
        R_lmn=[1.0, 0.1],
        Z_lmn=[0.0, -0.1],
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
        sym=sym,
        NFP=NFP,
    )
    eq = Equilibrium(
        L=L,
        M=M,
        N=N,
        NFP=NFP,
        surface=surface,
        pressure=PowerSeriesProfile([1e2, 0, -1e2]),
        iota=PowerSeriesProfile([1, 0, 2]),
        Psi=1.0,
    )
    eq = solve_continuation_automatic(eq)[-1]

    def beta(grid, data):
        return data["<beta>_vol"]

    low_beta = 0.01
    # todo: error that objective function has no linear attribute?
    objective = ObjectiveFunction(
        (ObjectiveFromUser(fun=beta, eq=eq, target=low_beta),)
    )

    constraints = (*get_fixed_boundary_constraints(eq), get_equilibrium_objective(eq))
    opt = Optimizer("proximal-lsq-exact")
    eq, result = eq.optimize(
        objective=objective, constraints=constraints, optimizer=opt
    )
    print(result)

    rho = np.array([0.5])
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 10)
    knots = np.linspace(0, 6 * np.pi, 20)
    # TODO now compare result to elliptic integral
    bounce_integrate, items = bounce_integral(
        eq, rho, alpha, knots, check=True, plot=False
    )
    pitch = pitch_of_extrema(knots, items["B.c"], items["B_z_ra.c"])
    bp1, bp2 = bounce_points(
        pitch, knots, items["B.c"], items["B_z_ra.c"], check=True, plot=False
    )
