import unittest
from desc.backend import jnp
from desc.continuation import perturb


class TestContinuation(unittest.TestCase):
    """tests for continuation functions"""

    def test_perturb(self):
        
        def test_fun(x, a0, a1, a2, a3, a4, c0, c1, c2, c3):
            return jnp.array([a0 + c0*x[0] + c1*x[1]**2,
                              a1 + c2*x[1] + c3*x[0]**2])
        
        x = jnp.array([1.0, 1.0])
        args = [0.0, -1.0, 9.0, 9.0, 9.0, 1.0, -1.0, 2.0, -1.0]
        deltas = jnp.array([0.5, 1, -0.5, 0.5])
        y1 = perturb(x, test_fun, deltas, args, pert_order=1, verbose=0)
        y2 = perturb(x, test_fun, deltas, args, pert_order=2, verbose=0)
        z = jnp.array([0.0, 2/3.0])
        self.assertLess(abs(z-y2), abs(z-y1))
        self.assertAlmostEqual(y2, z, delta=0.6)
