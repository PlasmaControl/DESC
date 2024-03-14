"""Tests bounce integral."""

import numpy as np
import pytest
from interpax import Akima1DInterpolator

from desc.backend import fori_loop, put, root_scalar
from desc.compute.bounce_integral import (
    cubic_poly_roots,
    field_line_to_desc_coords,
    polyder,
    polyint,
    polyval,
)


@pytest.mark.unit
def test_cubic_poly_roots():
    """Test vectorized computation of cubic polynomial exact roots."""
    cubic = 4
    poly = np.arange(-60, 60).reshape(cubic, 6, -1)
    poly[0] = np.where(poly[0] == 0, np.ones_like(poly[0]), poly[0])
    poly = poly * np.e * np.pi
    assert np.unique(poly.shape).size == poly.ndim
    constant = np.arange(10)
    assert np.unique(poly.shape + constant.shape).size == poly.ndim + constant.ndim
    roots = cubic_poly_roots(poly, constant, sort=True)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            for s in range(constant.size):
                a, b, c, d = poly[:, j, k]
                d = d - constant[s]
                np.testing.assert_allclose(
                    roots[s, j, k],
                    np.sort_complex(np.roots([a, b, c, d])),
                )


@pytest.mark.unit
def test_polyint():
    """Test vectorized computation of polynomial primitive."""
    quintic = 6
    poly = np.arange(-90, 90).reshape(quintic, 3, -1) * np.e * np.pi
    assert np.unique(poly.shape).size == poly.ndim
    constant = np.pi
    out = polyint(poly, k=constant)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            np.testing.assert_allclose(
                out[:, j, k], np.polyint(poly[:, j, k], k=constant)
            )


@pytest.mark.unit
def test_polyder():
    """Test vectorized computation of polynomial derivative."""
    quintic = 6
    poly = np.arange(-90, 90).reshape(quintic, 3, -1) * np.e * np.pi
    assert np.unique(poly.shape).size == poly.ndim
    out = polyder(poly)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            np.testing.assert_allclose(out[:, j, k], np.polyder(poly[:, j, k]))


@pytest.mark.unit
def test_polyval():
    """Test vectorized computation of polynomial evaluation."""
    quintic = 6
    poly = np.arange(-90, 90).reshape(quintic, 3, -1) * np.e * np.pi
    assert np.unique(poly.shape).size == poly.ndim
    x = np.linspace(0, 20, poly.shape[1] * poly.shape[2]).reshape(
        poly.shape[1], poly.shape[2]
    )
    x = np.stack([x, x * 2], axis=-1)
    x = np.stack([x, x * 2, x * 3, x * 4], axis=-1)
    assert np.unique(x.shape).size == x.ndim
    assert poly.shape[1:] == x.shape[: poly.ndim - 1]
    assert np.unique((poly.shape[0],) + x.shape[poly.ndim - 1 :]).size == x.ndim - 1
    val = polyval(x, poly)
    for j in range(poly.shape[1]):
        for k in range(poly.shape[2]):
            np.testing.assert_allclose(val[j, k], np.poly1d(poly[:, j, k])(x[j, k]))

    y = np.arange(1, 6)
    y = np.arange(y.prod()).reshape(*y)
    x = np.arange(y.shape[-1])
    a1d = Akima1DInterpolator(x, y, axis=-1)
    primitive = polyint(a1d.c)
    d = np.diff(x)
    k = polyval(d.reshape(d.size, *np.ones(primitive.ndim - 2, dtype=int)), primitive)
    primitive = primitive.at[-1, 1:].add(np.cumsum(k, axis=-1)[:-1])
    np.testing.assert_allclose(primitive, a1d.antiderivative().c)


# TODO: finish up details if deemed useful
def bounce_point(
    self, eq, lambdas, rho, alpha, max_bounce_points=20, max_field_line=10 * np.pi
):
    """Find bounce points."""
    # TODO:
    #     1. make another version of desc.backend.root_scalar
    #        to avoid separate root finding routines in residual and jac
    #        and use previous desc coords as initial guess for next iteration
    #     2. write docstrings and use transforms in api instead of eq
    def residual(zeta, i):
        grid, data = field_line_to_desc_coords(rho, alpha, zeta, eq)
        data = eq.compute(["|B|"], grid=grid, data=data)
        return data["|B|"] - lambdas[i]

    def jac(zeta):
        grid, data = field_line_to_desc_coords(rho, alpha, zeta, eq)
        data = eq.compute(["|B|_z|r,a"], grid=grid, data=data)
        return data["|B|_z|r,a"]

    # Compute |B| - lambda on a dense grid.
    # For every field line, find the roots of this linear spline.
    # These estimates for the true roots will serve as an initial guess, and
    # let us form a boundary mesh around root estimates to limit search domain
    # of the root finding algorithms.
    zeta = np.linspace(0, max_field_line, 3 * max_bounce_points)
    grid, data = field_line_to_desc_coords(rho, alpha, zeta, eq)
    data = eq.compute(["|B|"], grid=grid, data=data)
    B_norm = data["|B|"].reshape(alpha.size, rho.size, -1)  # constant field line chunks

    boundary_lt = np.zeros((lambdas.size, max_bounce_points, alpha.size, rho.size))
    boundary_rt = np.zeros((lambdas.size, max_bounce_points, alpha.size, rho.size))
    guess = np.zeros((lambdas.size, max_bounce_points, alpha.size, rho.size))
    # todo: scan over this
    for i in range(lambdas.size):
        for j in range(alpha.size):
            for k in range(rho.size):
                # indices of zeta values observed prior to sign change
                idx = np.nonzero(np.diff(np.sign(B_norm[j, k] - lambdas[i])))[0]
                guess[i, :, j, k] = grid.nodes[idx, 2]
                boundary_lt[i, :, j, k] = np.append(zeta[0], guess[:-1])
                boundary_rt[i, :, j, k] = np.append(guess[1:], zeta[-1])
    guess = guess.reshape(lambdas.size, max_bounce_points, alpha.size * rho.size)
    boundary_lt = boundary_lt.reshape(
        lambdas.size, max_bounce_points, alpha.size * rho.size
    )
    boundary_rt = boundary_rt.reshape(
        lambdas.size, max_bounce_points, alpha.size * rho.size
    )

    def body_lambdas(i, out):
        def body_roots(j, out_i):
            def fixup(z):
                return np.clip(z, boundary_lt[i, j], boundary_rt[i, j])

            # todo: call vmap to vectorize on guess[i, j] so that we solve
            #  guess[i, j].size independent root finding problems
            root = root_scalar(residual, guess[i, j], jac=jac, args=i, fixup=fixup)
            out_i = put(out_i, j, root)
            return out_i

        out = put(out, i, fori_loop(0, max_bounce_points, body_roots, out[i]))
        return out

    bounce_points = np.zeros(
        shape=(lambdas.size, alpha.size, rho.size, max_bounce_points)
    )
    bounce_points = fori_loop(0, lambdas.size, body_lambdas, bounce_points)
    return bounce_points
