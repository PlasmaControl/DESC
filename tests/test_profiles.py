import numpy as np
import unittest
import pytest
from desc.io import InputReader
from desc.equilibrium import Equilibrium
from desc.profiles import PowerSeriesProfile


class TestProfiles(unittest.TestCase):
    def test_same_result(self):
        input_path = "examples/DESC/SOLOVEV"
        ir = InputReader(input_path)

        eq1 = Equilibrium(ir.inputs[-1])
        eq2 = eq1.copy()
        eq2.pressure = eq1.pressure.to_spline()
        eq2.iota = eq1.iota.to_spline()

        eq1.solve()
        eq2.solve()

        np.testing.assert_allclose(
            eq1.x,
            eq2.x,
            rtol=1e-05,
            atol=1e-08,
        )

    def test_close_values(self):

        pp = PowerSeriesProfile(modes=np.array([0, 2, 4]), params=np.array([1, -2, 1]))
        sp = pp.to_spline()
        with pytest.warns(UserWarning):
            mp = pp.to_mtanh(order=4, ftol=1e-12, xtol=1e-12)
        x = np.linspace(0, 1, 100)

        np.testing.assert_allclose(pp(x), sp(x), rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(pp(x), mp(x), rtol=1e-3, atol=1e-2)

        pp1 = sp.to_powerseries(order=4)
        np.testing.assert_allclose(pp.params, pp1.params, rtol=1e-5, atol=1e-2)
        pp2 = mp.to_powerseries(order=4)
        np.testing.assert_allclose(pp.params, pp2.params, rtol=1e-5, atol=1e-2)
