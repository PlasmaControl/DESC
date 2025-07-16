"""Tests for profile classes."""

import numpy as np
import pytest
from scipy.constants import elementary_charge
from scipy.interpolate import interp1d

from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.grid import LinearGrid
from desc.io import InputReader
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
)
from desc.profiles import (
    FourierZernikeProfile,
    HermiteSplineProfile,
    MTanhProfile,
    PowerSeriesProfile,
    SplineProfile,
    TwoPowerProfile,
)

from .utils import area_difference, compute_coords


class TestProfiles:
    """Tests for Profile classes."""

    @pytest.mark.slow
    @pytest.mark.regression
    @pytest.mark.solve
    def test_same_result(self):
        """Test that different representations of the same profile give the same eq."""
        input_path = "./tests/inputs/SOLOVEV"
        ir = InputReader(input_path)

        eq1 = Equilibrium(**ir.inputs[-1], check_kwargs=False)
        eq2 = eq1.copy()
        eq2.pressure = eq1.pressure.to_spline()
        eq2.iota = eq1.iota.to_spline()

        eq1.solve()
        eq2.solve()

        Rr1, Zr1, Rv1, Zv1 = compute_coords(eq1, Nz=6)
        Rr2, Zr2, Rv2, Zv2 = compute_coords(eq2, Nz=6)
        rho_err, theta_err = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
        np.testing.assert_allclose(rho_err, 0, atol=1e-7)
        np.testing.assert_allclose(theta_err, 0, atol=2e-11)

    @pytest.mark.unit
    def test_close_values(self):
        """Test that different forms of the same profile give similar values."""
        pp = PowerSeriesProfile(
            modes=np.array([0, 2, 4]), params=np.array([1, -2, 1]), sym=False
        )
        sp = pp.to_spline()
        with pytest.warns(UserWarning):
            mp = pp.to_mtanh(order=4, ftol=1e-12, xtol=1e-12)
        zp = pp.to_fourierzernike()
        # don't test to rho=1 bc mtanh is very non-polynomial there,
        # don't test at axis bc splines and mtanh don't enforce zero slope exactly.
        x = np.linspace(0.1, 0.8, 10)

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

        np.testing.assert_allclose(pp(x, dt=1), 0)
        np.testing.assert_allclose(pp(x, dz=1), 0)
        np.testing.assert_allclose(sp(x, dt=1), 0)
        np.testing.assert_allclose(sp(x, dz=1), 0)
        np.testing.assert_allclose(mp(x, dt=1), 0)
        np.testing.assert_allclose(mp(x, dz=1), 0)

        pp = PowerSeriesProfile([0.5, 0, -1, 0, 0.5])
        tp = TwoPowerProfile([0.5, 2, 2])
        np.testing.assert_allclose(pp(x), tp(x))
        np.testing.assert_allclose(pp(x, dr=1), tp(x, dr=1))
        np.testing.assert_allclose(pp(x, dr=2), tp(x, dr=2))
        np.testing.assert_allclose(
            pp.params, tp.to_powerseries(order=4, sym=True).params
        )

    @pytest.mark.unit
    def test_PowerSeriesProfile_even_sym(self):
        """Test that even symmetry is enforced properly in PowerSeriesProfile."""
        pp = PowerSeriesProfile(params=np.array([4, 0, -2, 0, 3, 0, -1]), sym="auto")
        assert pp.sym == "even"  # auto symmetry detects it is even
        sp = pp.to_spline()

        pp_o = sp.to_powerseries(sym="auto")
        assert not pp_o.sym  # default conversion from spline is not symmetric

        pp_e = sp.to_powerseries(sym="even")
        assert pp_e.sym == "even"  # check that even symmetry is enforced
        # check that this matches the original parameters
        np.testing.assert_allclose(pp.params, pp_e.params, rtol=1e-3)

    @pytest.mark.unit
    def test_SplineProfile_methods(self):
        """Test that all methods of SplineProfile work as intended without errors."""
        pp = PowerSeriesProfile(
            modes=np.array([0, 2, 4]), params=np.array([1, -2, 1]), sym=False
        )  # base profile to work off of
        knots = np.linspace(0, 1.0, 40, endpoint=True)
        x = np.linspace(0, 0.99, 100)

        method = "nearest"
        sp = pp.to_spline(knots=knots, method=method)
        # should be exactly same if evaluated at knots + eps
        np.testing.assert_allclose(pp(knots), sp(knots + 0.4 * (x[1] - x[0])))

        method = "linear"
        sp = pp.to_spline(knots=knots, method=method)
        # should match linear interpolation
        scipy_interp = interp1d(x=knots, y=pp(knots))
        np.testing.assert_allclose(scipy_interp(x), sp(x))

        method = "cubic"
        sp = pp.to_spline(knots=knots, method=method)
        np.testing.assert_allclose(pp(x), sp(x), rtol=1e-5, atol=1e-3)

        method = "cubic2"
        sp = pp.to_spline(knots=knots, method=method)
        np.testing.assert_allclose(pp(x), sp(x), rtol=1e-5, atol=1e-3)

        method = "catmull-rom"
        sp = pp.to_spline(knots=knots, method=method)
        np.testing.assert_allclose(pp(x), sp(x), rtol=1e-5, atol=1e-3)

        method = "cardinal"
        sp = pp.to_spline(knots=knots, method=method)
        np.testing.assert_allclose(pp(x), sp(x), rtol=1e-5, atol=1e-3)

        method = "monotonic"
        sp = pp.to_spline(knots=knots, method=method)
        np.testing.assert_allclose(pp(x), sp(x), rtol=1e-5, atol=1e-3)

        method = "monotonic-0"
        sp = pp.to_spline(knots=knots, method=method)
        np.testing.assert_allclose(pp(x), sp(x), rtol=1e-5, atol=5e-3)

        # test monotonic splines preserve monotonicity
        f = np.heaviside(knots - 0.5, 0) + 0.1 * knots
        spm = SplineProfile(values=f, knots=knots, method="monotonic")
        spm0 = SplineProfile(values=f, knots=knots, method="monotonic-0")
        spc = SplineProfile(values=f, knots=knots, method="cubic")

        dfc = spc(x, dr=1)
        dfm = spm(x, dr=1)
        dfm0 = spm0(x, dr=1)
        assert dfc.min() < 0  # cubic interpolation undershoots, giving negative slope
        assert dfm.min() > 0  # monotonic interpolation doesn't
        assert dfm0.min() >= 0  # monotonic-0 doesn't overshoot either

    @pytest.mark.unit
    def test_repr(self):
        """Test string representation of profile classes."""
        pp = PowerSeriesProfile(modes=np.array([0, 2, 4]), params=np.array([1, -2, 1]))
        tp = TwoPowerProfile([1, 2, 1])
        sp = pp.to_spline()
        mp = pp.to_mtanh(order=4, ftol=1e-4, xtol=1e-4)
        zp = pp.to_fourierzernike()
        assert "PowerSeriesProfile" in str(pp)
        assert "TwoPowerProfile" in str(tp)
        assert "SplineProfile" in str(sp)
        assert "MTanhProfile" in str(mp)
        assert "FourierZernikeProfile" in str(zp)
        assert "SumProfile" in str(pp + zp)
        assert "ProductProfile" in str(pp * zp)
        assert "ScaledProfile" in str(2 * zp)
        assert "PowerProfile" in str(zp**2)

    @pytest.mark.unit
    def test_get_set(self):
        """Test getting/setting of profile attributes."""
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

        tp = TwoPowerProfile([1, 2, 3])
        np.testing.assert_allclose(tp.params, [1, 2, 3])
        tp.params = [1 / 2, 3 / 2, 4 / 3]
        np.testing.assert_allclose(tp.params, [1 / 2, 3 / 2, 4 / 3])

        zp = FourierZernikeProfile([1 - 1 / 3 - 1 / 6, -1 / 2, 1 / 6])
        assert zp.get_params(2, 0, 0) == -1 / 2
        zp.set_params(2, 0, 0, 1 / 4)
        assert zp.get_params(2, 0, 0) == 1 / 4
        zp.change_resolution(L=0)
        assert len(zp.params) == 1

    @pytest.mark.unit
    def test_auto_sym(self):
        """Test that even parity is enforced automatically."""
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

    @pytest.mark.unit
    def test_sum_profiles(self):
        """Test adding two profiles together."""
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )
        sp = pp.to_spline()
        zp = -pp.to_fourierzernike()

        f = pp + sp - zp
        x = np.linspace(0, 1, 50)
        np.testing.assert_allclose(f(x), 3 * (pp(x)), atol=1e-3)

        params = f.params
        assert params.size == len(sp.params) + len(pp.params) + len(zp.params) + 1
        params = f._parse_params(params)
        assert all(params[0] == pp.params)
        assert all(params[1] == sp.params)
        # offset by 1 because of two - signs
        assert all(params[2][1:] == zp.params)
        assert params[2][0] == -1

        f.params = (pp.params, 2 * sp.params, zp.params)
        np.testing.assert_allclose(f(x), 4 * (pp(x)), atol=1e-3)

    @pytest.mark.unit
    def test_product_profiles(self):
        """Test multiplying two profiles together."""
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )
        sp = pp.to_spline()
        zp = pp.to_fourierzernike()

        f = pp * sp * zp
        x = np.linspace(0, 1, 50)
        np.testing.assert_allclose(f(x), pp(x) ** 3, atol=1e-3)

        params = f.params
        assert params.size == len(sp.params) + len(pp.params) + len(zp.params)
        params = f._parse_params(params)
        assert all(params[0] == pp.params)
        assert all(params[1] == sp.params)
        assert all(params[2] == zp.params)

        f.params = (pp.params, 2 * sp.params, zp.params)
        np.testing.assert_allclose(f(x), 2 * pp(x) ** 3, atol=1e-3)

    @pytest.mark.unit
    def test_product_profiles_derivative(self):
        """Test that product profiles computes the derivative correctly."""
        p1 = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 3, -2, 4]), sym="auto"
        )
        p2 = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 3]),
            params=np.array([2.02, 4.93, 0.22, 0.46]),
            sym="auto",
        )
        p3 = PowerSeriesProfile(
            modes=np.array([0, 1, 3, 4]),
            params=np.array([1.79, 3.19, 1.82, 2.07]),
            sym="auto",
        )

        f = p1 * p2 * p3
        x = np.linspace(0, 1, 50)

        # Below is a simpler method to compute first derivative of product series
        # than the more general combinatorics algorithm in implementation.
        # Analytic formula derived from logarithmic differentiation.
        _sum = 0
        _sum_r = 0
        for profile in f._profiles:
            result = profile(x)
            result_r = profile(x, dr=1)
            result_rr = profile(x, dr=2)
            _sum += result_r / result
            _sum_r += result_rr / result - (result_r / result) ** 2
        f_r = f(x) * _sum
        f_rr = f_r * _sum + f(x) * _sum_r

        np.testing.assert_allclose(f_r, f(x, dr=1))
        np.testing.assert_allclose(f_rr, f(x, dr=2))

    @pytest.mark.unit
    def test_scaled_profiles(self):
        """Test scaling profiles by a constant."""
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )

        f = 3 * pp
        x = np.linspace(0, 1, 50)
        np.testing.assert_allclose(f(x), 3 * (pp(x)), atol=1e-3)

        params = f.params
        assert params[0] == 3
        assert all(params[1:] == pp.params)

        f.params = 2
        np.testing.assert_allclose(f(x), 2 * (pp(x)), atol=1e-3)

        f.params = 4 * pp.params

        params = f.params
        assert params.size == len(pp.params) + 1
        assert params[0] == 2
        np.testing.assert_allclose(params[1:], [4, -8, 4])
        np.testing.assert_allclose(pp.params, [1, -2, 1])
        np.testing.assert_allclose(f(x), 8 * (pp(x)), atol=1e-3)

    @pytest.mark.unit
    def test_powered_profiles(self):
        """Test raising profiles to a power."""
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )

        f = pp**3
        x = np.linspace(0, 1, 50)
        np.testing.assert_allclose(f(x), (pp(x)) ** 3, atol=1e-3)

        params = f.params
        assert params[0] == 3
        assert all(params[1:] == pp.params)

        f.params = 2
        np.testing.assert_allclose(f(x), (pp(x)) ** 2, atol=1e-3)

        f.params = 0.5
        np.testing.assert_allclose(f(x), np.sqrt(pp(x)), atol=1e-3)

    @pytest.mark.unit
    def test_powered_profiles_derivative(self):
        """Test that powered profiles computes the derivative correctly."""
        x = np.linspace(0, 1, 50)
        p1 = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 3, -2, 4]), sym="auto"
        )
        p2 = p1 * p1
        p3 = p1 * p2

        f3 = p1**3
        np.testing.assert_allclose(f3(x, dr=0), p3(x, dr=0))
        np.testing.assert_allclose(f3(x, dr=1), p3(x, dr=1))
        np.testing.assert_allclose(f3(x, dr=2), p3(x, dr=2))

        f2 = f3 ** (2 / 3)
        np.testing.assert_allclose(f2(x, dr=0), p2(x, dr=0))
        np.testing.assert_allclose(f2(x, dr=1), p2(x, dr=1))
        np.testing.assert_allclose(f2(x, dr=2), p2(x, dr=2))

    @pytest.mark.unit
    def test_profile_errors(self):
        """Test error checking when creating and working with profiles."""
        pp = PowerSeriesProfile(
            modes=np.array([0, 1, 2, 4]), params=np.array([1, 0, -2, 1]), sym="auto"
        )
        sp = pp.to_spline()
        zp = pp.to_fourierzernike()
        mp = pp.to_mtanh(order=4, ftol=1e-4, xtol=1e-4)
        tp = TwoPowerProfile(params=np.array([0.5, 2, 1.5]))
        grid = LinearGrid(L=9)

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
        with pytest.raises(ValueError):
            a = sp + pp
            a.params = pp.params
        with pytest.raises(ValueError):
            a = sp * pp
            a.params = sp.params
        with pytest.raises(ValueError):
            _ = TwoPowerProfile([1, 2, 3, 4])
        with pytest.raises(NotImplementedError):
            tp.compute(grid, dr=3)
        with pytest.raises(NotImplementedError):
            mp.compute(grid, dr=3)
        with pytest.raises(UserWarning):
            tp.params = [1, 0.3, 0.7]
        with pytest.raises(UserWarning):
            a = sp**-1

    @pytest.mark.unit
    def test_default_profiles(self):
        """Test that default profiles are just zeros."""
        pp = PowerSeriesProfile()
        tp = TwoPowerProfile()
        sp = SplineProfile()
        mp = MTanhProfile()
        zp = FourierZernikeProfile()

        x = np.linspace(0, 1, 10)
        np.testing.assert_allclose(pp(x), 0)
        np.testing.assert_allclose(tp(x), 0)
        np.testing.assert_allclose(sp(x), 0)
        np.testing.assert_allclose(mp(x), 0)
        np.testing.assert_allclose(zp(x), 0)

    @pytest.mark.regression
    def test_solve_with_combined(self):
        """Make sure combined profiles work correctly for solving equilibrium.

        Test for GH issue #347.
        """
        ne = PowerSeriesProfile(3.0e19 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(2.0e3 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        pressure = elementary_charge * (ne * Te + ne * Ti)

        LM_resolution = 6
        eq1 = Equilibrium(
            pressure=pressure,
            iota=PowerSeriesProfile([1.61]),
            Psi=np.pi,  # so B ~ 1 T
            NFP=1,
            L=LM_resolution,
            M=LM_resolution,
            N=0,
            L_grid=2 * LM_resolution,
            M_grid=2 * LM_resolution,
            N_grid=0,
            sym=True,
        )
        eq1.solve(
            constraints=get_fixed_boundary_constraints(eq=eq1),
            objective=ObjectiveFunction(objectives=ForceBalance(eq=eq1)),
            maxiter=5,
        )
        eq2 = Equilibrium(
            electron_temperature=Te,
            electron_density=ne,
            atomic_number=1,
            ion_temperature=Ti,
            iota=PowerSeriesProfile([1.61]),
            Psi=np.pi,  # so B ~ 1 T
            NFP=1,
            L=LM_resolution,
            M=LM_resolution,
            N=0,
            L_grid=2 * LM_resolution,
            M_grid=2 * LM_resolution,
            N_grid=0,
            sym=True,
        )
        eq2.solve(
            constraints=get_fixed_boundary_constraints(eq=eq2),
            objective=ObjectiveFunction(objectives=ForceBalance(eq=eq2)),
            maxiter=5,
        )
        np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn, atol=1e-14)
        np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn, atol=1e-14)
        np.testing.assert_allclose(eq1.L_lmn, eq2.L_lmn, atol=1e-14)

    @pytest.mark.unit
    def test_kinetic_pressure(self):
        """Test that both ways of computing pressure are equivalent."""
        ne = PowerSeriesProfile(3.0e19 * np.array([1, -1]), modes=[0, 10])
        Te = PowerSeriesProfile(2.0e3 * np.array([1, -1]), modes=[0, 2])
        Ti = Te
        pressure = elementary_charge * (ne * Te + ne * Ti)

        LM_resolution = 6
        eq1 = Equilibrium(
            pressure=pressure,
            iota=PowerSeriesProfile([1.61]),
            Psi=np.pi,  # so B ~ 1 T
            NFP=1,
            L=LM_resolution,
            M=LM_resolution,
            N=0,
            L_grid=2 * LM_resolution,
            M_grid=2 * LM_resolution,
            N_grid=0,
            sym=True,
        )
        eq2 = Equilibrium(
            electron_temperature=Te,
            electron_density=ne,
            iota=PowerSeriesProfile([1.61]),
            Psi=np.pi,  # so B ~ 1 T
            NFP=1,
            L=LM_resolution,
            M=LM_resolution,
            N=0,
            L_grid=2 * LM_resolution,
            M_grid=2 * LM_resolution,
            N_grid=0,
            sym=True,
        )
        grid = LinearGrid(L=20)
        data1 = eq1.compute(["p", "p_r"], grid=grid)
        data2 = eq2.compute(["p", "p_r"], grid=grid)

        assert np.all(np.isnan(data1["ne"]))
        assert np.all(np.isnan(data1["Te"]))
        assert np.all(np.isnan(data1["Ti"]))
        assert np.all(np.isnan(data1["Zeff"]))
        assert np.all(np.isnan(data1["ne_r"]))
        assert np.all(np.isnan(data1["Te_r"]))
        assert np.all(np.isnan(data1["Ti_r"]))
        assert np.all(np.isnan(data1["Zeff_r"]))
        assert np.all(data2["Te"] == data2["Ti"])
        assert np.all(data2["Te_r"] == data2["Ti_r"])
        np.testing.assert_allclose(data1["p"], data2["p"])
        np.testing.assert_allclose(data1["p_r"], data2["p_r"])

    @pytest.mark.unit
    def test_hermite_spline_solve(self):
        """Test that spline with double number of parameters is optimized."""
        eq = get("DSHAPE")
        rho = np.linspace(0, 1.0, 20, endpoint=True)
        eq.pressure = HermiteSplineProfile(
            eq.pressure(rho), eq.pressure(rho, dr=1), rho
        )
        eq.solve()
        assert eq.is_nested()
