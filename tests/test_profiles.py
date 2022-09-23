import numpy as np
import unittest
import pytest
from desc.io import InputReader
from desc.profiles import PowerSeriesProfile, FourierZernikeProfile
from desc.equilibrium import Equilibrium
from .utils import compute_coords, area_difference


class TestProfiles(unittest.TestCase):
    @pytest.mark.slow
    def test_same_result(self):
        input_path = "./tests/inputs/SOLOVEV"
        ir = InputReader(input_path)

        eq1 = Equilibrium(**ir.inputs[-1])
        print(eq1.pressure)
        eq2 = eq1.copy()
        eq2.pressure = eq1.pressure.to_spline()
        eq2.iota = eq1.iota.to_spline()

        eq1.solve()
        eq2.solve()

        Rr1, Zr1, Rv1, Zv1 = compute_coords(eq1, check_all_zeta=True)
        Rr2, Zr2, Rv2, Zv2 = compute_coords(eq2, check_all_zeta=True)
        rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
        np.testing.assert_allclose(rho_err, 0, atol=1e-7)
        np.testing.assert_allclose(theta_err, 0, atol=2e-11)

        assert True

    @pytest.mark.slow
    def test_close_values(self):

        pp = PowerSeriesProfile(
            modes=np.array([0, 2, 4]), params=np.array([1, -2, 1]), sym=False
        )
        sp = pp.to_spline()
        with pytest.warns(UserWarning):
            mp = pp.to_mtanh(order=4, ftol=1e-12, xtol=1e-12)
        zp = pp.to_fourierzernike()
        x = np.linspace(0, 0.8, 10)

        np.testing.assert_allclose(pp(x), sp(x), rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(pp(x, dr=2), mp(x, dr=2), rtol=1e-2, atol=1e-1)
        np.testing.assert_allclose(pp(x), zp(x), rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(pp(x, dr=2), zp(x, dr=2), rtol=1e-2, atol=1e-1)

        pp1 = sp.to_powerseries(order=4)
        np.testing.assert_allclose(pp.params, pp1.params, rtol=1e-5, atol=1e-2)
        pp2 = mp.to_powerseries(order=4)
        np.testing.assert_allclose(pp.params, pp2.params, rtol=1e-5, atol=1e-2)

        with pytest.warns(UserWarning):
            mp1 = sp.to_mtanh()
        np.testing.assert_allclose(mp1(x, dr=1), sp(x, dr=1), rtol=1e-5, atol=1e-3)

        pp3 = pp.to_powerseries()
        sp3 = sp.to_spline()
        np.testing.assert_allclose(sp3(x), pp3(x), rtol=1e-5, atol=1e-3)
        sp4 = mp.to_spline()
        np.testing.assert_allclose(sp3(x), sp4(x), rtol=1e-5, atol=1e-2)

    def test_repr(self):

        pp = PowerSeriesProfile(modes=np.array([0, 2, 4]), params=np.array([1, -2, 1]))
        sp = pp.to_spline()
        mp = pp.to_mtanh(order=4, ftol=1e-4, xtol=1e-4)
        zp = pp.to_fourierzernike()
        assert "PowerSeriesProfile" in str(pp)
        assert "SplineProfile" in str(sp)
        assert "MTanhProfile" in str(mp)
        assert "FourierZernikeProfile" in str(zp)
        assert "SumProfile" in str(pp + zp)
        assert "ProductProfile" in str(pp * zp)
        assert "ScaledProfile" in str(2 * zp)

    def test_get_set(self):

        pp = PowerSeriesProfile(
            modes=np.array([0, 2, 4]), params=np.array([1, -2, 1]), sym=False
        )

        assert pp.get_params(2) == -2
        assert pp.basis.get_idx(4) == 4
        pp.set_params(3, 22)

        assert pp.get_params(3) == 22
        pp.change_resolution(L=2)
        assert pp.params.size == 3

        sp = pp.to_spline()
        sp.params = sp.params + 1
        np.testing.assert_allclose(sp.params, 1 + pp(sp._knots))

        zp = FourierZernikeProfile([1 - 1 / 3 - 1 / 6, -1 / 2, 1 / 6])
        assert zp.get_params(2, 0, 0) == -1 / 2
        zp.set_params(2, 0, 0, 1 / 4)
        assert zp.get_params(2, 0, 0) == 1 / 4
        zp.change_resolution(L=0)
        assert len(zp.params) == 1

    def test_auto_sym(self):
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )
        assert pp.sym == "even"
        assert pp.basis.num_modes == 3
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 1, -2, 1]), sym="auto"
        )
        assert pp.sym is False
        assert pp.basis.num_modes == 5

    def test_sum_profiles(self):
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )
        sp = pp.to_spline()
        zp = pp.to_fourierzernike()

        f = pp + sp - (-zp)
        x = np.linspace(0, 1, 50)
        f.grid = 50
        np.testing.assert_allclose(f(), 3 * (pp(x)), atol=1e-3)

        params = f.params
        assert all(params[0] == pp.params)
        assert all(params[1] == sp.params)
        assert all(params[2][1][1] == zp.params)

        f.params = (None, 2 * sp.params, None)
        f.grid = x
        np.testing.assert_allclose(f(), 4 * (pp(x)), atol=1e-3)

    def test_product_profiles(self):
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )
        sp = pp.to_spline()
        zp = pp.to_fourierzernike()

        f = pp * sp * zp
        x = np.linspace(0, 1, 50)
        f.grid = 50
        np.testing.assert_allclose(f(), pp(x) ** 3, atol=1e-3)

        params = f.params
        assert all(params[0] == pp.params)
        assert all(params[1] == sp.params)
        assert all(params[2] == zp.params)

        f.params = (None, 2 * sp.params, None)
        f.grid = x
        np.testing.assert_allclose(f(), 2 * pp(x) ** 3, atol=1e-3)

    def test_scaled_profiles(self):
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )

        f = 3 * pp
        x = np.linspace(0, 1, 50)
        f.grid = 50
        np.testing.assert_allclose(f(), 3 * (pp(x)), atol=1e-3)

        params = f.params
        assert params[0] == 3
        assert all(params[1] == pp.params)

        f.params = 2
        f.grid = x
        np.testing.assert_allclose(f(), 2 * (pp(x)), atol=1e-3)

        f.params = 4 * pp.params
        f.grid = x

        params = f.params
        assert params[0] == 2
        np.testing.assert_allclose(params[1], [4, -8, 4])
        np.testing.assert_allclose(pp.params, [1, -2, 1])
        np.testing.assert_allclose(f(), 8 * (pp(x)), atol=1e-3)

    def test_profile_errors(self):
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )
        sp = pp.to_spline()
        zp = pp.to_fourierzernike()
        mp = pp.to_mtanh(order=4, ftol=1e-4, xtol=1e-4)

        with pytest.raises(ValueError):
            zp.params = 4
        with pytest.raises(ValueError):
            mp.params = np.arange(4)
        with pytest.raises(ValueError):
            sp.params = np.arange(4)
        with pytest.raises(ValueError):
            pp.params = np.arange(8)
        with pytest.raises(NotImplementedError):
            a = sp + 4
        with pytest.raises(NotImplementedError):
            a = sp * [1, 2, 3]
        with pytest.raises(TypeError):
            sp.grid = None
        with pytest.raises(TypeError):
            sp.grid = None
        with pytest.raises(ValueError):
            a = sp + pp
            a.params = pp.params
        with pytest.raises(ValueError):
            a = sp * pp
            a.params = sp.params
