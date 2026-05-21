"""Tests for quadrature utilities."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy
from jax import grad
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import chebgauss, chebweight
from scipy.special import roots_chebyu
from tests.test_plotting import tol_1d

from desc.backend import jnp
from desc.compute._fast_ion import _fold_wells_to_alpha, _reduction_gamma_alpha
from desc.integrals.quad_utils import (
    _LossCone,
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


def _manufactured_gamma_alpha(num_alpha):
    """Manufactured Gamma_alpha data with analytically known loss interval."""
    alpha0 = 0.37
    thresh = 0.23
    alpha = np.linspace(0, 2 * np.pi, num_alpha, endpoint=False)
    v_tau = (
        1.2 + 0.3 * np.cos(alpha) + 0.2 * np.sin(2 * alpha) - 0.1 * np.cos(3 * alpha)
    )[:, None, None]
    radial = (-np.sin(alpha - alpha0))[:, None, None]
    poloidal = np.ones_like(radial)
    start = alpha0 + np.arcsin(thresh)
    stop = start + np.pi
    exact = (
        1.2 * (stop - start)
        + 0.3 * (np.sin(stop) - np.sin(start))
        + 0.1 * (np.cos(2 * start) - np.cos(2 * stop))
        - (0.1 / 3) * (np.sin(3 * stop) - np.sin(3 * start))
    ) / (2 * np.pi)
    opts = SimpleNamespace(alpha=alpha, thresh=thresh)
    return v_tau, radial, poloidal, opts, exact


def _alpha_and_mask(opts, radial):
    """Uniform alpha grid and valid mask for manufactured Gamma_alpha data."""
    return jnp.asarray(opts.alpha[:, None, None]), jnp.ones_like(radial, dtype=bool)


@pytest.mark.unit
def test_fold_wells_to_alpha_midpoint_mapping():
    """Test long-field-line wells fold to midpoint effective alpha labels."""
    nfp = 2
    period = 2 * jnp.pi / nfp
    opts = SimpleNamespace(alpha=jnp.array([0.0, 1.0]), num_field_periods=3, thresh=0.2)
    z1 = jnp.array([[[0.1, 0.5, period + 0.1]], [[0.2, 0.6, period + 0.2]]])
    z2 = z1 + 0.1
    values = [jnp.ones_like(z1)]

    value, alpha, mask = _fold_wells_to_alpha(
        values, (z1, z2), opts, jnp.array(0.4), nfp
    )

    expected_alpha = np.array(
        [0.0, 0.4 * np.pi, 0.8 * np.pi, 1.0, 1.0 + 0.4 * np.pi, 1.0 + 0.8 * np.pi]
    )
    np.testing.assert_allclose(alpha[:, 0, 0], expected_alpha)
    np.testing.assert_allclose(value[mask], 1.0)
    np.testing.assert_array_equal(mask.sum(axis=0), np.array([[4, 2, 0]]))


@pytest.mark.unit
def test_nonuniform_loss_cone_matches_uniform_grid():
    """Test nonuniform loss-cone indicator agrees on a uniform grid."""
    _, radial, poloidal, opts, _ = _manufactured_gamma_alpha(32)
    alpha, mask = _alpha_and_mask(opts, radial)
    thresh = opts.thresh * jnp.abs(poloidal)
    alpha_out_candidate = radial - thresh
    alpha_in_candidate = -radial - thresh
    dist = (opts.alpha - opts.alpha[:, None]) % (2 * jnp.pi)
    da = 2 * jnp.pi / opts.alpha.size

    uniform = _LossCone.indicator(
        alpha_in_candidate, alpha_out_candidate, dist, da, order=1
    )
    nonuniform = _LossCone.indicator_nonuniform(
        alpha_in_candidate, alpha_out_candidate, alpha, mask, order=1
    )

    np.testing.assert_allclose(nonuniform, uniform)


@pytest.mark.unit
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=tol_1d)
def test_loss_cone_convergence():
    """Test Gamma_alpha reduction on a manufactured alpha loss cone."""
    num_alpha = np.array([16, 32, 64, 128])
    lowes = []
    highs = []
    for num in num_alpha:
        v_tau, radial, poloidal, opts, exact = _manufactured_gamma_alpha(num)
        alpha, mask = _alpha_and_mask(opts, radial)
        lo = _reduction_gamma_alpha(
            v_tau, radial, poloidal, opts, alpha, mask, order=0
        ).item()
        hi = _reduction_gamma_alpha(
            v_tau, radial, poloidal, opts, alpha, mask, order=1
        ).item()
        lowes.append(abs(lo - exact))
        highs.append(abs(hi - exact))
    lowes, highs = np.asarray(lowes), np.asarray(highs)

    assert highs[-1] < 2e-5
    assert highs[-1] < highs[-2] < highs[-3]

    fig, ax = plt.subplots()
    ax.loglog(num_alpha, lowes, "o-")
    ax.loglog(num_alpha, highs, "o-")
    ax.loglog(num_alpha, lowes[0] * (num_alpha[0] / num_alpha), "k--")
    ax.loglog(num_alpha, highs[2] * (num_alpha[2] / num_alpha) ** 2, "k:")
    return fig


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
