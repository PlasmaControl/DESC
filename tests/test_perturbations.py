import unittest
import numpy as np
from desc.backend import jnp
from desc.perturbations import perturb_continuation_params


class TestPerturbations(unittest.TestCase):
    """tests for pertubations functions"""

    def test_perturb_continuation_params_1D(self):
        """1D test function where linear perturb is exact"""

        def test_fun(x, a0, a1, a2, a3, a4, c0, c1, c2, c3):
            return jnp.array([(x[0] - a0) * (x[0] - c0)])

        a0 = 1.0
        c0 = 2.0
        x = np.array([c0])  # initial solution
        args = [a0, 0, 0, 0, 0, c0, 0, 0, 0]
        dc = np.array([0.1, 0.25, 0.5, 1.0])

        n = len(dc)
        err = np.zeros((2, n))
        for i in range(n):
            deltas = np.array([dc[i], 0, 0, 0])
            y1, timer = perturb_continuation_params(
                x, test_fun, deltas, args, pert_order=1, verbose=0)
            y2, timer = perturb_continuation_params(
                x, test_fun, deltas, args, pert_order=2, verbose=0)
            # correct answer
            z = np.array([c0 - (a0-c0) / (2*c0 - a0 - c0) * dc[i]])
            err[0, i] = np.abs(y1-z)[0]  # 1st order error
            err[1, i] = np.abs(y2-z)[0]  # 2nd order error

        self.assertEqual(np.max(err[0, :]), 0)
        np.testing.assert_allclose(
            np.argsort(err[1, :]), np.linspace(0, n-1, n), atol=1e-8)

    def test_perturb_continuation_params_2D(self):
        """2D test function to check convergence rates"""

        def test_fun(x, a0, a1, a2, a3, a4, c0, c1, c2, c3):
            return jnp.array([a0 + c0*x[0] + c1*x[1]**2,
                              a1 + c2*x[1] + c3*x[0]**2])

        x = np.array([1.0, 1.0])  # initial solution
        args = [0.0, -1.0, 9.0, 9.0, 9.0, 1.0, -1.0, 2.0, -1.0]
        deltas = jnp.array([[2e-3, 2e-3, -2e-3, 2e-3],
                            [5e-3, 5e-3, -5e-3, 5e-3],
                            [0.01, 0.01, -0.01, 0.01],
                            [0.02, 0.02, -0.02, 0.02],
                            [0.05, 0.05, -0.05, 0.05],
                            [0.10, 0.10, -0.10, 0.10],
                            [0.25, 0.25, -0.25, 0.25],
                            [0.50, 0.50, -0.50, 0.50]])
        z = np.array([[1.00400, 1.00400],
                      [1.01000, 1.01003],
                      [1.02000, 1.02010],
                      [1.04004, 1.04043],
                      [1.10056, 1.10291],
                      [1.20416, 1.21316],
                      [1.55762, 1.61122],
                      [2.48979, 2.73301]])

        n = deltas.shape[0]
        err = np.zeros((2, n))
        for i in range(n):
            y1, timer = perturb_continuation_params(
                x, test_fun, deltas[i, :], args, pert_order=1, verbose=0)
            y2, timer = perturb_continuation_params(
                x, test_fun, deltas[i, :], args, pert_order=2, verbose=0)
            err[0, i] = np.linalg.norm(y1-z[i, :])  # 1st order error
            err[1, i] = np.linalg.norm(y2-z[i, :])  # 2nd order error

        np.testing.assert_allclose(
            np.argsort(err[0, :]), np.linspace(0, n-1, n), atol=1e-8)
        np.testing.assert_allclose(
            np.argsort(err[1, :]), np.linspace(0, n-1, n), atol=1e-8)
