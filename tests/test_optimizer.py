import unittest
import numpy as np

from desc.optimize import fmin_scalar
from desc.optimize.utils import make_spd, chol_U_update
from scipy.optimize import rosen, rosen_der, rosen_hess
from desc.derivatives import Derivative
from numpy.random import default_rng


def ellipsoid(x, p, q):
    i = jnp.arange(x.size) + 1
    d = x.size
    return jnp.sum((i / d) ** p * x ** (2 + q * 2 * i))


ellipsoid_grad = Derivative(ellipsoid, argnum=0, mode="grad")
ellipsoid_hess = Derivative(ellipsoid, argnum=0, mode="hess")


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

        out = fmin_scalar(
            rosen,
            x0,
            rosen_der,
            hess=rosen_hess,
            verbose=1,
            method="dogleg",
            x_scale="hess",
            options={"ga_accept_threshold": 1},
        )

        np.testing.assert_allclose(out["x"], true_x)

    def test_rosenbrock_full_hess_subspace(self):
        rando = default_rng(seed=2)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmin_scalar(
            rosen,
            x0,
            rosen_der,
            hess=rosen_hess,
            verbose=1,
            method="subspace",
            x_scale="hess",
            options={"ga_accept_threshold": 1},
        )

        np.testing.assert_allclose(out["x"], true_x)

    def test_rosenbrock_bfgs_dogleg(self):
        rando = default_rng(seed=3)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmin_scalar(
            rosen,
            x0,
            rosen_der,
            hess="bfgs",
            verbose=1,
            method="dogleg",
            x_scale="hess",
            options={"ga_accept_threshold": 0},
        )
        np.testing.assert_allclose(out["x"], true_x)

    def test_rosenbrock_bfgs_subspace(self):
        rando = default_rng(seed=4)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmin_scalar(
            rosen,
            x0,
            rosen_der,
            hess="bfgs",
            verbose=1,
            method="subspace",
            x_scale="hess",
            options={"ga_accept_threshold": 0},
        )
        np.testing.assert_allclose(out["x"], true_x)
