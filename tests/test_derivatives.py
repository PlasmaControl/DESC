import unittest
import numpy as np

from desc.backend import jnp
from desc.derivatives import AutoDiffDerivative, FiniteDiffDerivative

from numpy.random import default_rng


class TestDerivative(unittest.TestCase):
    """Tests Grid classes"""

    def test_finite_diff_vec(self):
        def test_fun(x, y, a):
            return x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = FiniteDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = np.diag(y)

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

    def test_finite_diff_scalar(self):
        def test_fun(x, y, a):
            return np.dot(x, y) + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = FiniteDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = y

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

        jac.argnum = 1
        J = jac.compute(x, y, a)
        np.testing.assert_allclose(J, x, atol=1e-8)

    def test_auto_diff(self):
        def test_fun(x, y, a):
            return jnp.cos(x) + x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = AutoDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = np.diag(-np.sin(x) + y)

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

    def test_compare_AD_FD(self):
        def test_fun(x, y, a):
            return jnp.cos(x) + x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac_AD = AutoDiffDerivative(test_fun, argnum=0)
        J_AD = jac_AD.compute(x, y, a)

        jac_FD = AutoDiffDerivative(test_fun, argnum=0)
        J_FD = jac_FD.compute(x, y, a)

        np.testing.assert_allclose(J_FD, J_AD, atol=1e-8)

    def test_fd_hessian(self):
        rando = default_rng(seed=0)

        n = 5
        A = rando.random((n, n))
        A = A + A.T
        g = rando.random(n)

        def f(x):
            return 5 + g.dot(x) + x.dot(1 / 2 * A.dot(x))

        hess = FiniteDiffDerivative(f, argnum=0, mode="hess")

        y = rando.random(n)
        A1 = hess(y)

        np.testing.assert_allclose(A1, A)
