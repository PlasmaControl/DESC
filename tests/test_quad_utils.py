"""Tests for quadrature utilities."""

import numpy as np
import pytest
import scipy
from jax import grad
from numpy.polynomial.chebyshev import chebgauss, chebweight
from scipy.special import roots_chebyu

from desc.backend import jnp
from desc.integrals.quad_utils import (
    automorphism_arcsin,
    automorphism_sin,
    bijection_from_disc,
    bijection_to_disc,
    chebgauss2,
    grad_automorphism_arcsin,
    grad_automorphism_sin,
    grad_bijection_from_disc,
    leggauss_lob,
    simpson2,
    tanh_sinh,
    uniform,
)


@pytest.mark.unit
def test_automorphism():
    """Test automorphisms."""
    a, b = -312, 786
    x = np.linspace(a, b, 10)
    y = bijection_to_disc(x, a, b)
    x_1 = bijection_from_disc(y, a, b)
    np.testing.assert_allclose(x_1, x)
    np.testing.assert_allclose(bijection_to_disc(x_1, a, b), y)
    np.testing.assert_allclose(
        automorphism_arcsin(automorphism_sin(y), gamma=1), y, atol=5e-7
    )
    np.testing.assert_allclose(
        automorphism_sin(automorphism_arcsin(y, gamma=1)), y, atol=5e-7
    )

    np.testing.assert_allclose(grad_bijection_from_disc(a, b), 1 / (2 / (b - a)))
    np.testing.assert_allclose(
        grad_automorphism_sin(y),
        1 / grad_automorphism_arcsin(automorphism_sin(y), gamma=1),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        1 / grad_automorphism_arcsin(y, gamma=1),
        grad_automorphism_sin(automorphism_arcsin(y, gamma=1)),
        atol=2e-6,
    )

    # test that floating point error is acceptable
    x = tanh_sinh(19)[0]
    assert np.all(np.abs(x) < 1)
    y = 1 / np.sqrt(1 - np.abs(x))
    assert np.isfinite(y).all()
    y = 1 / np.sqrt(1 - np.abs(automorphism_sin(x)))
    assert np.isfinite(y).all()
    y = 1 / np.sqrt(1 - np.abs(automorphism_arcsin(x, gamma=1)))
    assert np.isfinite(y).all()


@pytest.mark.unit
def test_leggauss_lobatto():
    """Test quadrature points and weights against known values."""
    with pytest.raises(ValueError):
        x, w = leggauss_lob(1)
    x, w = leggauss_lob(0, True)
    assert x.size == w.size == 0

    x, w = leggauss_lob(2)
    np.testing.assert_allclose(x, [-1, 1])
    np.testing.assert_allclose(w, [1, 1])

    x, w = leggauss_lob(3)
    np.testing.assert_allclose(x, [-1, 0, 1])
    np.testing.assert_allclose(w, [1 / 3, 4 / 3, 1 / 3])
    np.testing.assert_allclose(leggauss_lob(x.size - 2, True), (x[1:-1], w[1:-1]))

    x, w = leggauss_lob(4)
    np.testing.assert_allclose(x, [-1, -np.sqrt(1 / 5), np.sqrt(1 / 5), 1])
    np.testing.assert_allclose(w, [1 / 6, 5 / 6, 5 / 6, 1 / 6])
    np.testing.assert_allclose(leggauss_lob(x.size - 2, True), (x[1:-1], w[1:-1]))

    x, w = leggauss_lob(5)
    np.testing.assert_allclose(x, [-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1])
    np.testing.assert_allclose(w, [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10])
    np.testing.assert_allclose(leggauss_lob(x.size - 2, True), (x[1:-1], w[1:-1]))

    def fun(a):
        x, w = leggauss_lob(a.size)
        return jnp.dot(x * a, w)

    # make sure differentiable
    # https://github.com/PlasmaControl/DESC/pull/854#discussion_r1733323161
    assert np.isfinite(grad(fun)(jnp.arange(10) * np.pi)).all()


@pytest.mark.unit
def test_chebgauss():
    """Test Chebyshev quadratures."""

    def f(y):
        return 5.2 * y**7 - 3.6 * y**3 + y**4

    deg = 4
    yk, wk = chebgauss(deg)
    x, w = uniform(deg)
    np.testing.assert_allclose(yk[::-1], automorphism_sin(x))
    np.testing.assert_allclose(wk, 0.5 * jnp.pi * w)
    np.testing.assert_allclose(np.diff(x), x[1] - x[0])
    np.testing.assert_allclose(
        f(automorphism_sin(x)).dot(w),
        2 / jnp.pi * scipy.integrate.quad(lambda y: f(y) / np.sqrt(1 - y**2), -1, 1)[0],
    )
    x, w = roots_chebyu(deg)
    w *= chebweight(x)
    np.testing.assert_allclose(chebgauss2(deg), (x, w))

    n = 5
    x, w = simpson2(n)
    np.testing.assert_allclose(x, [-5 / 6, -2 / 3, 0, 2 / 3, 5 / 6])
    np.testing.assert_allclose(w.sum(), 2)
    np.testing.assert_allclose(
        f(x).dot(w), scipy.integrate.quad(f, -1, 1)[0], rtol=2.5e-2
    )
