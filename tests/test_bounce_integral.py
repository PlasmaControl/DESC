"""Test bounce integral methods."""

import inspect

import numpy as np
import pytest
from interpax import Akima1DInterpolator, CubicHermiteSpline
from matplotlib import pyplot as plt
from scipy.special import ellipe, ellipk

from desc.backend import flatnonzero, fori_loop, jnp, put, root_scalar
from desc.compute.bounce_integral import (
    bounce_average,
    bounce_integral,
    bounce_points,
    pitch_of_extrema,
    poly_der,
    poly_int,
    poly_root,
    poly_val,
    take_mask,
)
from desc.compute.utils import dot
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import desc_grid_from_field_line_coords
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid
from desc.objectives import (
    ObjectiveFromUser,
    ObjectiveFunction,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile


@np.vectorize(signature="(m)->()")
def _last_value(a):
    """Return the last non-nan value in ``a``."""
    a = jnp.ravel(a)[::-1]
    idx = jnp.squeeze(flatnonzero(~jnp.isnan(a), size=1, fill_value=0))
    return a[idx]


def _filter_not_nan(a):
    """Filter out nan while making sure they have correct padding."""
    is_nan = np.isnan(a)
    assert np.array_equal(is_nan, np.sort(is_nan, axis=-1))
    return a[~is_nan]


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
        ), "take_mask() has bugs."
        assert np.array_equal(last[i], desired[-1]), "flatnonzero() has bugs."


@pytest.mark.unit
def test_reshape_convention():
    """Test the reshaping convention separates data across field lines."""
    rho = np.linspace(0, 1, 3)
    alpha = np.linspace(0, 2 * np.pi, 4)
    zeta = np.linspace(0, 10 * np.pi, 5)
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
    assert "S, zeta.size" in src, err_msg
    assert "pitch_res, rho.size, alpha.size" in src, err_msg
    src = inspect.getsource(desc_grid_from_field_line_coords)
    assert 'indexing="ij"' in src, err_msg
    assert 'meshgrid(rho, alpha, zeta, indexing="ij")' in src, err_msg


@pytest.mark.unit
def test_poly_root():
    """Test vectorized computation of cubic polynomial exact roots."""
    cubic = 4
    poly = np.arange(-24, 24).reshape(cubic, 6, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(poly.shape).size == poly.ndim
    constant = np.broadcast_to(np.arange(poly.shape[-1]), poly.shape[1:])
    constant = np.stack([constant, constant])
    root = poly_root(poly, constant, sort=True)

    for i in range(constant.shape[0]):
        for j in range(poly.shape[1]):
            for k in range(poly.shape[2]):
                d = poly[-1, j, k] - constant[i, j, k]
                np.testing.assert_allclose(
                    actual=root[i, j, k],
                    desired=np.sort(np.roots([*poly[:-1, j, k], d])),
                )

    poly = np.array(
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
    root = poly_root(poly.T, sort=True, distinct=True)
    for j in range(poly.shape[0]):
        np.testing.assert_allclose(
            actual=_filter_not_nan(root[j]),
            desired=np.unique(np.roots(poly[j])),
            err_msg=str(j),
        )
    poly = np.array([0, 1, -1, -8, 12])
    np.testing.assert_allclose(
        actual=_filter_not_nan(poly_root(poly, sort=True, distinct=True)),
        desired=np.unique(np.roots(poly)),
    )


@pytest.mark.unit
def test_poly_int():
    """Test vectorized computation of polynomial primitive."""
    quintic = 6
    poly = np.arange(-18, 18).reshape(quintic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(poly.shape).size == poly.ndim
    constant = np.broadcast_to(np.arange(poly.shape[-1]), poly.shape[1:])
    primitive = poly_int(poly, k=constant)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            np.testing.assert_allclose(
                actual=primitive[:, j, k],
                desired=np.polyint(poly[:, j, k], k=constant[j, k]),
            )
    assert poly_int(poly).shape == primitive.shape, "Failed broadcasting default k."


@pytest.mark.unit
def test_poly_der():
    """Test vectorized computation of polynomial derivative."""
    quintic = 6
    poly = np.arange(-18, 18).reshape(quintic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(poly.shape).size == poly.ndim
    derivative = poly_der(poly)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            np.testing.assert_allclose(
                actual=derivative[:, j, k], desired=np.polyder(poly[:, j, k])
            )


@pytest.mark.unit
def test_poly_val():
    """Test vectorized computation of polynomial evaluation."""
    quartic = 5
    c = np.arange(-60, 60).reshape(quartic, 3, -1) * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    x = np.linspace(0, 20, c.shape[1] * c.shape[2]).reshape(c.shape[1], c.shape[2])
    val = poly_val(x=x, c=c)
    for index in np.ndindex(c.shape[1:]):
        idx = (..., *index)
        np.testing.assert_allclose(
            actual=val[idx],
            desired=np.poly1d(c[idx])(x[idx]),
            err_msg=f"Failed with shapes {x.shape} and {c.shape}.",
        )

    x = np.stack([x, x * 2], axis=0)
    x = np.stack([x, x * 2, x * 3, x * 4], axis=0)
    # make sure broadcasting won't hide error in implementation
    assert np.unique(x.shape).size == x.ndim
    assert c.shape[1:] == x.shape[x.ndim - (c.ndim - 1) :]
    assert np.unique((c.shape[0],) + x.shape[c.ndim - 1 :]).size == x.ndim - 1
    val = poly_val(x=x, c=c)
    for index in np.ndindex(c.shape[1:]):
        idx = (..., *index)
        np.testing.assert_allclose(
            actual=val[idx],
            desired=np.poly1d(c[idx])(x[idx]),
            err_msg=f"Failed with shapes {x.shape} and {c.shape}.",
        )

    # integrate piecewise polynomial and set constants to preserve continuity
    y = np.arange(2, 8)
    y = np.arange(y.prod()).reshape(*y)
    x = np.arange(y.shape[-1])
    a1d = Akima1DInterpolator(x, y, axis=-1)
    primitive = poly_int(a1d.c)
    # choose evaluation points at d just to match choice made in a1d.antiderivative()
    d = np.diff(x)
    # evaluate every spline at d
    k = poly_val(x=d, c=primitive)
    # don't want to use jax.ndarray.at[].add() in case jax is not installed
    primitive = np.array(primitive)
    primitive[-1, 1:] += np.cumsum(k, axis=-1)[:-1]
    np.testing.assert_allclose(primitive, a1d.antiderivative().c)


@pytest.mark.unit
def test_bounce_points():
    """Test that the bounce points are computed correctly."""

    def plot_field_line(B, pitch, start, end):
        fig, ax = plt.subplots()
        for knot in B.x:
            ax.axvline(x=knot, color="red", linestyle="--")
        z = np.linspace(start, end, 50)
        ax.plot(z, B(z))
        ax.plot(z, np.full(z.size, 1 / pitch))
        plt.show()

    def assert_case_1(plot=False):
        # 1/pitch does not intersect extrema
        pitch = 2
        start = np.pi / 3
        end = 6 * np.pi
        knots = np.linspace(start, end, 5)
        B = CubicHermiteSpline(knots, np.cos(knots), -np.sin(knots))
        # Can observe correctness of bounce points through this plot.
        if plot:
            plot_field_line(B, pitch, start, end)
        bp1, bp2 = bounce_points(
            pitch,
            knots,
            B.c[:, np.newaxis],
            B.derivative().c[:, np.newaxis],
            check=True,
        )
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        # Hardcode because CubicHermiteSpline.solve not yet implemented.
        np.testing.assert_allclose(bp1, np.array([1.04719755, 7.13120418]))
        np.testing.assert_allclose(bp2, np.array([5.19226163, 17.57830469]))

    def assert_case_2(plot=False):
        # 1/pitch intersects extrema
        pitch = 1
        start = np.pi / 3
        end = 6 * np.pi
        knots = np.linspace(start, end, 5)
        B = CubicHermiteSpline(knots, np.cos(knots), -np.sin(knots))
        # Can observe correctness of bounce points through this plot.
        if plot:
            plot_field_line(B, pitch, start, end)
        bp1, bp2 = bounce_points(
            pitch,
            knots,
            B.c[:, np.newaxis],
            B.derivative().c[:, np.newaxis],
            check=True,
        )
        bp1, bp2 = map(_filter_not_nan, (bp1, bp2))
        # Hardcode because CubicHermiteSpline.solve not yet implemented.
        np.testing.assert_allclose(bp1, np.array([1.04719755, 7.13120418]))
        np.testing.assert_allclose(bp2, np.array([5.19226163, 17.57830469]))

    # TODO: add all the edge cases I parameterized
    assert_case_1()


@pytest.mark.unit
def test_pitch_and_hairy_ball():
    """Test different ways of specifying pitch and ensure B does not vanish."""
    eq = get("HELIOTRON")
    rho = np.linspace(1e-12, 1, 6)
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 5)
    zeta = jnp.linspace(0, 6 * jnp.pi, 20)
    ba, items = bounce_average(eq, rho=rho, alpha=alpha, zeta=zeta, return_items=True)
    B = items["data"]["B"]
    assert not np.isclose(B, 0, atol=1e-19).any(), "B should never vanish."

    name = "g_zz"
    f = eq.compute(name, grid=items["grid"], data=items["data"])[name]
    # specify pitch per field line
    pitch_res = 30
    B = B.reshape(rho.size * alpha.size, -1)
    pitch = np.linspace(1 / B.max(axis=-1), 1 / B.min(axis=-1), pitch_res)
    result = ba(f, pitch)
    assert np.isfinite(result).any()
    # specify pitch from extrema of |B|
    pitch = pitch_of_extrema(zeta, items["poly_B"], items["poly_B_z"])
    result = ba(f, pitch)
    assert np.isfinite(result).any()


# @pytest.mark.unit
def test_elliptic_integral_limit():
    """Test bounce integral matches elliptic integrals.

    In the limit of a low beta, large aspect ratio tokamak the bounce integral
    should converge to the elliptic integrals of the first kind.
    todo: would be nice to understand physics for why these are supposed
        to be proportional to bounce integral. Is this discussed in any book?
        Also, looking at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html
        Are we saying that in this limit, we expect that |B| ~ sin(t)^2, with m as the
        pitch angle? I assume that we want to add g_zz to the integrand in the
        definition of the function in the scipy documentation above,
        and after a change of variables the bounce points will be the endpoints of
        the integration.
        So this test will test whether the quadrature is accurate
        (and not whether the bounce points were accurate).

    """
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
    zeta = np.linspace(0, 10 * np.pi, 20)
    bi, items = bounce_integral(
        eq, rho=rho, alpha=alpha, zeta=zeta, return_items=True, check=True
    )
    B = items["data"]["B"]
    pitch_res = 15
    pitch = np.linspace(1 / B.max(), 1 / B.min(), pitch_res)
    name = "g_zz"
    f = eq.compute(name, grid=items["grid"], data=items["data"])[name]
    result = bi(f, pitch)
    assert np.isfinite(result).any(), "tanh_sinh quadrature failed."

    # TODO now compare result to elliptic integral
    bp1, bp2 = bounce_points(pitch, zeta, items["poly_B"], items["poly_B_z"])


@pytest.mark.unit
def test_bounce_averaged_drifts():
    """Test bounce-averaged drift with analytical expressions.

    Calculate bounce-averaged drifts using the bounce-average routine and
    compare it with the analytical expression
    # Note 1: This test can be merged with the elliptic integral test as
    we do calculate elliptic integrals here
    # Note 2: Remove tests/test_equilibrium :: test_shifted_circle_geometry
    # once all the epsilons and Gammas have been implemented and tested
    """
    eq = Equilibrium.load(".//tests//inputs//low-beta-shifted-circle.h5")

    psi = 0.25  # rho^2 (or normalized psi)
    alpha = 0

    eq_keys = ["iota", "iota_r", "a", "rho", "psi"]

    data_eq = eq.compute(eq_keys)

    iotas = np.interp(np.sqrt(psi), data_eq["rho"], data_eq["iota"])
    shears = np.interp(np.sqrt(psi), data_eq["rho"], data_eq["iota_r"])

    N = int((2 * eq.M_grid) * 4 + 1)

    zeta = np.linspace(-1.0 * np.pi / iotas, 1.0 * np.pi / iotas, N)
    theta_PEST = alpha * np.ones(N, dtype=int) + iotas * zeta

    coords1 = np.zeros((N, 3))
    coords1[:, 0] = np.sqrt(psi) * np.ones(N, dtype=int)
    coords1[:, 1] = theta_PEST
    coords1[:, 2] = zeta

    # Creating a grid along a field line
    c1 = eq.compute_theta_coords(coords1)
    grid = Grid(c1, sort=False)

    # The bounce integral operator should be able to take a grid
    bi, items = bounce_integral(eq, grid=grid, return_items=True, check=True)

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

    data = eq.compute(data_keys, grid=grid, override_grid=False)

    psib = data_eq["psi"][-1]

    # signs
    sign_psi = psib / np.abs(psib)
    sign_iota = iotas / np.abs(iotas)

    # normalizations
    Lref = data_eq["a"]
    Bref = 2 * np.abs(psib) / Lref**2

    modB = data["|B|"]
    bmag = modB / Bref

    x = Lref * np.sqrt(psi)
    s_hat = -x / iotas * shears / Lref

    iota = data["iota"]
    gradpar = Lref * data["B^zeta"] / modB

    ## Comparing coefficient calculation here with coefficients from compute/_mtric
    cvdrift = -2 * sign_psi * Bref * Lref**2 * np.sqrt(psi) * data["cvdrift"]
    gbdrift = -2 * sign_psi * Bref * Lref**2 * np.sqrt(psi) * data["gbdrift"]

    a0_over_R0 = Lref * np.sqrt(psi)

    bmag_an = np.mean(bmag) * (1 - a0_over_R0 * np.cos(theta_PEST))
    np.testing.assert_allclose(bmag, bmag_an, atol=5e-3, rtol=5e-3)

    gradpar_an = 2 * Lref * iota * (1 - a0_over_R0 * np.cos(theta_PEST))
    np.testing.assert_allclose(gradpar, gradpar_an, atol=9e-3, rtol=5e-3)

    dPdrho = np.mean(-0.5 * (cvdrift - gbdrift) * modB**2)
    alpha_MHD = -dPdrho * 1 / iota**2 * 0.5

    grad_psi = data["grad(psi)"]
    grad_alpha = data["grad(alpha)"]

    gds21 = -sign_iota * np.array(dot(grad_psi, grad_alpha)) * s_hat / Bref

    fudge_factor2 = 0.19
    gbdrift_an = fudge_factor2 * (
        -1 * s_hat + (np.cos(theta_PEST) - 1.0 * gds21 / s_hat * np.sin(theta_PEST))
    )

    fudge_factor3 = 0.07
    cvdrift_an = gbdrift_an + fudge_factor3 * alpha_MHD / bmag**2

    # Comparing coefficients with their analytical expressions
    np.testing.assert_allclose(gbdrift, gbdrift_an, atol=1.5e-2, rtol=5e-3)
    np.testing.assert_allclose(cvdrift, cvdrift_an, atol=9e-3, rtol=5e-3)

    # Values of pitch angle for which to evaluate the bounce averages
    lambdas = np.linspace(1 / np.min(bmag), 1 / np.max(bmag), 11)

    bavg_drift_an = (
        0.5 * cvdrift_an * ellipe(lambdas)
        + gbdrift_an * ellipk(lambdas)
        + dPdrho / bmag**2 * ellipe(lambdas)
    )

    # The quantities are already calculated along a field line
    bavg_drift_num = bi(
        np.sqrt(1 - lambdas * bmag) * 0.5 * cvdrift
        + gbdrift * 1 / np.sqrt(1 - lambdas * bmag)
        + dPdrho / bmag**2 * np.sqrt(1 - lambdas * bmag),
        lambdas,
    )

    np.testing.assert_allclose(bavg_drift_num, bavg_drift_an, atol=2e-2, rtol=1e-2)


# TODO: if deemed useful finish details using methods in desc.compute.bounce_integral
def _compute_bounce_points_with_root_finding(
    eq, pitch, rho, alpha, resolution=20, zeta_max=10 * np.pi
):
    # TODO: avoid separate root finding routines in residual and jac
    #       and use previous desc coords as initial guess for next iteration
    def residual(zeta, i):
        grid, data = desc_grid_from_field_line_coords(rho, alpha, zeta, eq)
        data = eq.compute(["|B|"], grid=grid, data=data)
        return data["|B|"] - pitch[i]

    def jac(zeta):
        grid, data = desc_grid_from_field_line_coords(rho, alpha, zeta, eq)
        data = eq.compute(["|B|_z|r,a"], grid=grid, data=data)
        return data["|B|_z|r,a"]

    # Compute |B| - 1/pitch on a dense grid.
    # For every field line, find the roots of this linear spline.
    # These estimates for the true roots will serve as an initial guess, and
    # let us form a boundary mesh around root estimates to limit search domain
    # of the root finding algorithms.
    zeta = np.linspace(0, zeta_max, 3 * resolution)
    grid, data = desc_grid_from_field_line_coords(rho, alpha, zeta, eq)
    data = eq.compute(["|B|"], grid=grid, data=data)
    B_norm = data["|B|"].reshape(alpha.size, rho.size, -1)  # constant field line chunks

    boundary_lt = np.zeros((pitch.size, resolution, alpha.size, rho.size))
    boundary_rt = np.zeros((pitch.size, resolution, alpha.size, rho.size))
    guess = np.zeros((pitch.size, resolution, alpha.size, rho.size))
    # todo: scan over this
    for i in range(pitch.size):
        for j in range(alpha.size):
            for k in range(rho.size):
                # indices of zeta values observed prior to sign change
                idx = np.nonzero(np.diff(np.sign(B_norm[j, k] - pitch[i])))[0]
                guess[i, :, j, k] = grid.nodes[idx, 2]
                boundary_lt[i, :, j, k] = np.append(zeta[0], guess[:-1])
                boundary_rt[i, :, j, k] = np.append(guess[1:], zeta[-1])
    guess = guess.reshape(pitch.size, resolution, alpha.size * rho.size)
    boundary_lt = boundary_lt.reshape(pitch.size, resolution, alpha.size * rho.size)
    boundary_rt = boundary_rt.reshape(pitch.size, resolution, alpha.size * rho.size)

    def body_pitch(i, out):
        def body_roots(j, out_i):
            def fixup(z):
                return np.clip(z, boundary_lt[i, j], boundary_rt[i, j])

            # todo: call vmap to vectorize on guess[i, j] so that we solve
            #  guess[i, j].size independent root finding problems
            root = root_scalar(residual, guess[i, j], jac=jac, args=i, fixup=fixup)
            out_i = put(out_i, j, root)
            return out_i

        out = put(out, i, fori_loop(0, resolution, body_roots, out[i]))
        return out

    bounce_points = np.zeros(shape=(pitch.size, alpha.size, rho.size, resolution))
    bounce_points = fori_loop(0, pitch.size, body_pitch, bounce_points)
    return bounce_points
