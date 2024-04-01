"""Test bounce integral methods."""

import inspect

import numpy as np
import pytest
from interpax import Akima1DInterpolator

from desc.backend import fori_loop, jnp, put, put_along_axis, root_scalar, vmap
from desc.compute.bounce_integral import (
    _last_value,
    _roll_and_replace_if_shift,
    bounce_average,
    bounce_integral,
    compute_bounce_points,
    cubic_poly_roots,
    polyder,
    polyint,
    polyval,
    take_mask,
)
from desc.continuation import solve_continuation_automatic
from desc.equilibrium import Equilibrium
from desc.equilibrium.coords import desc_grid_from_field_line_coords
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


@pytest.mark.unit
def test_mask_operation():
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
        np.testing.assert_allclose(
            actual=taken[i],
            desired=np.pad(desired, (0, cols - desired.size), constant_values=np.nan),
            err_msg="take_mask() has bugs.",
        )
        np.testing.assert_allclose(
            actual=last[i], desired=desired[-1], err_msg="_last_value() has bugs."
        )

    shift = np.random.choice([True, False], size=rows)
    replacement = last * shift + a[:, 0] * (~shift)
    # This might be a better way to perform this computation, without
    # the jax.cond, which will get transformed to jax.select under vmap
    # which performs both branches of the computation.
    # But perhaps computing replacement as above, while fine for jit,
    # will make the computation non-differentiable...
    desired = put_along_axis(
        vmap(jnp.roll)(a, shift), jnp.array([0]), replacement[:, np.newaxis], axis=-1
    )
    np.testing.assert_allclose(
        actual=_roll_and_replace_if_shift(a, shift, replacement),
        desired=desired,
        err_msg="_roll_and_replace_if_shift() has bugs.",
    )


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
    src = inspect.getsource(bounce_integral)
    assert "R, A" in src and "A, R" not in src, err_msg
    assert "A, zeta.size" in src, err_msg
    src = inspect.getsource(desc_grid_from_field_line_coords)
    assert 'indexing="ij"' in src, err_msg
    assert 'meshgrid(rho, alpha, zeta, indexing="ij")' in src, err_msg


@pytest.mark.unit
def test_cubic_poly_roots():
    """Test vectorized computation of cubic polynomial exact roots."""
    cubic = 4
    poly = np.arange(-24, 24).reshape(cubic, 6, -1)
    poly[0] = np.where(poly[0] == 0, np.ones_like(poly[0]), poly[0])
    poly = poly * np.e * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(poly.shape).size == poly.ndim
    constant = np.broadcast_to(np.arange(poly.shape[-1]), poly.shape[1:])
    roots = cubic_poly_roots(poly, constant, sort=True)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            a, b, c, d = poly[:, j, k]
            np.testing.assert_allclose(
                actual=roots[j, k],
                desired=np.sort_complex(np.roots([a, b, c, d - constant[j, k]])),
            )


@pytest.mark.unit
def test_polyint():
    """Test vectorized computation of polynomial primitive."""
    quintic = 6
    poly = np.arange(-18, 18).reshape(quintic, 3, -1) * np.e * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(poly.shape).size == poly.ndim
    constant = np.broadcast_to(np.arange(poly.shape[-1]), poly.shape[1:])
    primitive = polyint(poly, k=constant)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            np.testing.assert_allclose(
                actual=primitive[:, j, k],
                desired=np.polyint(poly[:, j, k], k=constant[j, k]),
            )
    assert polyint(poly).shape == primitive.shape, "Failed broadcasting default k."


@pytest.mark.unit
def test_polyder():
    """Test vectorized computation of polynomial derivative."""
    quintic = 6
    poly = np.arange(-18, 18).reshape(quintic, 3, -1) * np.e * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(poly.shape).size == poly.ndim
    derivative = polyder(poly)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            np.testing.assert_allclose(
                actual=derivative[:, j, k], desired=np.polyder(poly[:, j, k])
            )


@pytest.mark.unit
def test_polyval():
    """Test vectorized computation of polynomial evaluation."""
    quartic = 5
    c = np.arange(-60, 60).reshape(quartic, 3, -1) * np.e * np.pi
    # make sure broadcasting won't hide error in implementation
    assert np.unique(c.shape).size == c.ndim
    x = np.linspace(0, 20, c.shape[1] * c.shape[2]).reshape(c.shape[1], c.shape[2])
    val = polyval(x=x, c=c)
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
    val = polyval(x=x, c=c)
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
    primitive = polyint(a1d.c)
    # choose evaluation points at d just to match choice made in a1d.antiderivative()
    d = np.diff(x)
    # evaluate every spline at d
    k = polyval(x=d, c=primitive)
    # don't want to use jax.ndarray.at[].add() in case jax is not installed
    primitive = np.array(primitive)
    primitive[-1, 1:] += np.cumsum(k, axis=-1)[:-1]
    np.testing.assert_allclose(primitive, a1d.antiderivative().c)


@pytest.mark.unit
def test_pitch_and_hairy_ball():
    """Test different ways of specifying pitch and ensure B does not vanish."""
    eq = get("HELIOTRON")
    rho = np.linspace(1e-12, 1, 6)
    alpha = np.linspace(0, (2 - eq.sym) * np.pi, 5)
    ba, items = bounce_average(eq, rho=rho, alpha=alpha, return_items=True)
    B = items["data"]["B"]
    assert not np.isclose(B, 0, atol=1e-19).any(), "B should never vanish."

    name = "g_zz"
    f = eq.compute(name, grid=items["grid"], data=items["data"])[name]
    # Same pitch for every field line may give sparse result.
    pitch_res = 30
    pitch = np.linspace(1 / B.max(), 1 / B.min(), pitch_res)[:, np.newaxis, np.newaxis]
    result = ba(f, pitch)
    assert np.isfinite(result).any()

    # specify pitch per field line
    B = B.reshape(rho.size * alpha.size, -1)
    pitch = np.linspace(1 / B.max(axis=-1), 1 / B.min(axis=-1), pitch_res).reshape(
        -1, rho.size, alpha.size
    )
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
    bi, items = bounce_integral(eq, rho=rho, alpha=alpha, zeta=zeta, return_items=True)
    B = items["data"]["B"]
    pitch_res = 15
    pitch = np.linspace(1 / B.max(), 1 / B.min(), pitch_res)
    name = "g_zz"
    f = eq.compute(name, grid=items["grid"], data=items["data"])[name]
    result = bi(f, pitch)
    assert np.isfinite(result).any(), "tanh_sinh quadrature failed."

    # TODO now compare result to elliptic integral
    bp1, bp2 = compute_bounce_points(pitch, zeta, items["poly_B"], items["poly_B_z"])


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
