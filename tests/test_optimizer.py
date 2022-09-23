import unittest
import numpy as np
import pytest
from desc.backend import jnp
from desc.optimize import fmintr, lsqtr
from scipy.optimize import rosen, rosen_der, rosen_hess
from desc.derivatives import Derivative
from numpy.random import default_rng
from scipy.optimize import BFGS


def vector_fun(x, p):
    """Complicated-ish vector valued function for testing least squares."""
    a0 = x * p[0]
    a1 = jnp.exp(-(x ** 2) * p[1])
    a2 = jnp.cos(jnp.sin(x * p[2] - x ** 2 * p[3]))
    a3 = jnp.sum(
        jnp.array([(x + 2) ** -(i * 2) * pi ** (i + 1) for i, pi in enumerate(p[3:])]),
        axis=0,
    )
    return a0 + a1 + 3 * a2 + a3


def scalar_fun(x):
    """Simple convex function for testing scalar minimization."""
    return x[0] ** 2 + x[1] ** 2 - jnp.log(x[0] + 2) + 10 - jnp.log(x[1] + 2)


scalar_grad = Derivative(scalar_fun, mode="grad")
scalar_hess = Derivative(scalar_fun, mode="hess")


class TestFmin(unittest.TestCase):
    def test_convex_full_hess_dogleg(self):
        rando = default_rng(seed=2)

        x0 = 10 * rando.random(2)

        out = fmintr(
            scalar_fun,
            x0,
            scalar_grad,
            scalar_hess,
            verbose=1,
            method="dogleg",
            x_scale="hess",
            ftol=0,
            xtol=0,
            gtol=1e-12,
            options={"ga_accept_threshold": 0},
        )
        assert out["success"] is True
        np.testing.assert_allclose(scalar_grad(out["x"]), 0, atol=1e-12)

    def test_convex_full_hess_subspace(self):
        rando = default_rng(seed=2)

        x0 = rando.random(2)

        out = fmintr(
            scalar_fun,
            x0,
            scalar_grad,
            scalar_hess,
            verbose=1,
            method="subspace",
            x_scale="hess",
            ftol=0,
            xtol=0,
            gtol=1e-12,
            options={"ga_accept_threshold": 1},
        )
        assert out["success"] is True
        np.testing.assert_allclose(scalar_grad(out["x"]), 0, atol=1e-12)

    @pytest.mark.slow
    def test_rosenbrock_bfgs_dogleg(self):
        rando = default_rng(seed=3)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmintr(
            rosen,
            x0,
            rosen_der,
            hess="bfgs",
            verbose=1,
            method="dogleg",
            x_scale=1,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"ga_accept_threshold": 0},
        )
        np.testing.assert_allclose(out["x"], true_x)

    @pytest.mark.slow
    def test_rosenbrock_bfgs_subspace(self):
        rando = default_rng(seed=4)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmintr(
            rosen,
            x0,
            rosen_der,
            hess=BFGS(),
            verbose=1,
            method="subspace",
            x_scale=1,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"ga_accept_threshold": 0},
        )
        np.testing.assert_allclose(out["x"], true_x)


class TestLSQTR(unittest.TestCase):
    def test_lsqtr_exact(self):

        p = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
        x = np.linspace(-1, 1, 100)
        y = vector_fun(x, p)

        def res(p):
            return vector_fun(x, p) - y

        rando = default_rng(seed=0)
        p0 = p + 0.25 * (rando.random(p.size) - 0.5)

        jac = Derivative(res, 0, "fwd")

        out = lsqtr(
            res,
            p0,
            jac,
            verbose=0,
            x_scale=1,
            tr_method="cho",
            options={"initial_trust_radius": 0.15, "max_trust_radius": 0.25},
        )
        np.testing.assert_allclose(out["x"], p)

        out = lsqtr(
            res,
            p0,
            jac,
            verbose=0,
            x_scale=1,
            tr_method="svd",
            options={"initial_trust_radius": 0.15, "max_trust_radius": 0.25},
        )
        np.testing.assert_allclose(out["x"], p)


def test_no_iterations():
    """Make sure giving the correct answer works correctly"""

    np.random.seed(0)
    A = np.random.random((20, 10))
    b = np.random.random(20)
    x0 = np.linalg.lstsq(A, b)[0]

    vecfun = lambda x: A @ x - b
    vecjac = Derivative(vecfun)

    fun = lambda x: np.sum(vecfun(x) ** 2)
    grad = Derivative(fun, 0, mode="grad")
    hess = Derivative(fun, 0, mode="hess")

    out1 = fmintr(fun, x0, grad, hess)
    out2 = lsqtr(vecfun, x0, vecjac)

    np.testing.assert_allclose(x0, out1["x"])
    np.testing.assert_allclose(x0, out2["x"])
