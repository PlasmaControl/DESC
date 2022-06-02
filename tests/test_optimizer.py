import unittest
import numpy as np
import pytest
from desc.backend import jnp
from desc.optimize import fmintr, lsqtr
from desc.optimize.utils import make_spd, chol_U_update
from scipy.optimize import rosen, rosen_der, rosen_hess
from desc.derivatives import Derivative
from numpy.random import default_rng


def fun(x, p):
    a0 = x * p[0]
    a1 = jnp.exp(-(x ** 2) * p[1])
    a2 = jnp.cos(jnp.sin(x * p[2] - x ** 2 * p[3]))
    a3 = jnp.sum(
        jnp.array([(x + 2) ** -(i * 2) * pi ** (i + 1) for i, pi in enumerate(p[3:])]),
        axis=0,
    )
    return a0 + a1 + 3 * a2 + a3


class TestUtils(unittest.TestCase):
    def test_spd(self):
        rando = default_rng(seed=0)

        n = 100
        A = rando.random((n, n))
        A = A + A.T - 5
        mineig = sorted(np.linalg.eig(A)[0])[0]
        self.assertTrue(mineig < 0)
        B = make_spd(A)
        mineig = sorted(np.linalg.eig(B)[0])[0]
        self.assertTrue(mineig > 0)

    def test_chol_update(self):
        rando = default_rng(seed=0)

        n = 100
        A = rando.random((n, n))
        v = rando.random(n)
        A = A + A.T - 5
        B = make_spd(A)
        U = np.linalg.cholesky(B).T
        Bv = B + np.outer(v, v)
        Uv = np.linalg.cholesky(Bv).T
        Uva = chol_U_update(U, v, 1)

        np.testing.assert_allclose(Uv, Uva)


class TestFmin(unittest.TestCase):
    def test_rosenbrock_full_hess_dogleg(self):
        rando = default_rng(seed=1)

        x0 = rando.random(7)
        true_x = np.ones(7)

        out = fmintr(
            rosen,
            x0,
            rosen_der,
            hess=rosen_hess,
            verbose=1,
            method="dogleg",
            x_scale="hess",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"ga_accept_threshold": 1},
        )

        np.testing.assert_allclose(out["x"], true_x)

    def test_rosenbrock_full_hess_subspace(self):
        rando = default_rng(seed=2)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmintr(
            rosen,
            x0,
            rosen_der,
            hess=rosen_hess,
            verbose=1,
            method="subspace",
            x_scale="hess",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"ga_accept_threshold": 1},
        )

        np.testing.assert_allclose(out["x"], true_x)

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
            x_scale="hess",
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
            hess="bfgs",
            verbose=1,
            method="subspace",
            x_scale="hess",
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
        y = fun(x, p)

        def res(p):
            return fun(x, p) - y

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
        print(out)
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
