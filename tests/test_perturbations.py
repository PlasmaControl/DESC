import unittest
import numpy as np

from desc.equilibrium import Equilibrium
from desc.objectives import (
    ObjectiveFunction,
    FixedBoundaryR,
    FixedBoundaryZ,
    FixedPressure,
    LCFSBoundary,
    Volume,
)
from desc.perturbations import perturb


class TestPerturbations(unittest.TestCase):
    """Test pertubations."""

    def test_perturb_1D(self):
        """Linear test function where perturb order=1 is exact."""

        # circular tokamak
        inputs = {
            "sym": True,
            "NFP": 1,
            "Psi": 1.0,
            "L": 2,
            "M": 1,
            "N": 0,
            "pressure": np.array([[0, 0], [2, 0]]),
            "iota": np.array([[0, 0]]),
            "surface": np.array([[0, -1, 0, 0, 1], [0, 0, 0, 3, 0], [0, 1, 0, 1, 0]]),
        }
        eq_orig = Equilibrium(**inputs)
        eq_orig.change_resolution(N=1)

        # perturbations
        dp = 1e3 * np.array([1, 0, -1])
        dRb = np.array([-0.2, 0, 0, 0.1, 0.2])
        dZb = np.array([0.1, -0.2, 0, -0.2])

        obj_pres = ObjectiveFunction(FixedPressure(), eq=eq_orig)
        obj_bdry = ObjectiveFunction(
            Volume(),
            (FixedBoundaryR(), FixedBoundaryZ(), LCFSBoundary()),
            eq=eq_orig,
        )

        eq_pres = perturb(eq_orig, obj_pres, dp=dp, order=1, verbose=2, copy=True)
        eq_bdry = perturb(
            eq_orig, obj_bdry, dRb=dRb, dZb=dZb, order=1, verbose=2, copy=True
        )

        np.testing.assert_allclose(eq_pres.p_l, eq_orig.p_l + dp)
        np.testing.assert_allclose(eq_bdry.Rb_lmn, eq_orig.Rb_lmn + dRb)
        np.testing.assert_allclose(eq_bdry.Zb_lmn, eq_orig.Zb_lmn + dZb)

    def test_perturb_2D(self):
        """Nonlinear test function to check perturb convergence rates."""
        pass
