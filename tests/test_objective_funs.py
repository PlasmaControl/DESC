"""Tests for objective functions.

These generally don't test the accuracy of the computation for realistic examples,
that is done in test_compute_functions or regression tests.

This module primarily tests the constructing/building/calling methods.
"""

import numpy as np
import pytest
from scipy.constants import mu_0

from desc.compute import get_transforms
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio,
    Elongation,
    Energy,
    GenericObjective,
    MagneticWell,
    MeanCurvature,
    MercierStability,
    ObjectiveFunction,
    PlasmaVesselDistance,
    PrincipalCurvature,
    QuasisymmetryBoozer,
    QuasisymmetryTripleProduct,
    QuasisymmetryTwoTerm,
    RotationalTransform,
    ToroidalCurrent,
    Volume,
)
from desc.objectives.objective_funs import _Objective
from desc.profiles import PowerSeriesProfile
from desc.vmec_utils import ptolemy_linear_transform


class TestObjectiveFunction:
    """Test ObjectiveFunction classes."""

    @pytest.mark.unit
    def test_generic(self):
        """Test GenericObjective for arbitrary quantities."""

        def test(f, eq):
            obj = GenericObjective(f, eq=eq)
            kwargs = {
                "R_lmn": eq.R_lmn,
                "Z_lmn": eq.Z_lmn,
                "L_lmn": eq.L_lmn,
                "i_l": eq.i_l,
                "c_l": eq.c_l,
                "Psi": eq.Psi,
            }
            np.testing.assert_allclose(
                obj.compute(**kwargs),
                eq.compute(f, grid=obj.grid)[f] * obj.grid.weights,
            )

        test("sqrt(g)", Equilibrium())
        test("current", Equilibrium(iota=PowerSeriesProfile(0)))
        test("iota", Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_volume(self):
        """Test calculation of plasma volume."""

        def test(eq):
            obj = Volume(
                target=10 * np.pi**2, weight=1 / np.pi**2, eq=eq, normalize=False
            )
            V = obj.compute(eq.R_lmn, eq.Z_lmn)
            V_scaled = obj.compute_scaled(eq.R_lmn, eq.Z_lmn)
            V_scalar = obj.compute_scalar(eq.R_lmn, eq.Z_lmn)
            np.testing.assert_allclose(V, 20 * np.pi**2)
            np.testing.assert_allclose(V_scaled, 10)
            np.testing.assert_allclose(V_scalar, 10)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_aspect_ratio(self):
        """Test calculation of aspect ratio."""

        def test(eq):
            obj = AspectRatio(target=5, weight=1, eq=eq)
            AR = obj.compute(eq.R_lmn, eq.Z_lmn)
            AR_scaled = obj.compute_scaled(eq.R_lmn, eq.Z_lmn)
            np.testing.assert_allclose(AR, 10)
            np.testing.assert_allclose(AR_scaled, 5)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_elongation(self):
        """Test calculation of elongation."""

        def test(eq):
            obj = Elongation(target=0, weight=2, eq=eq)
            f = obj.compute(eq.R_lmn, eq.Z_lmn)
            f_scaled = obj.compute_scaled(eq.R_lmn, eq.Z_lmn)
            np.testing.assert_allclose(f, 1.3 / 0.7, rtol=5e-3)
            np.testing.assert_allclose(f_scaled, 2 * (1.3 / 0.7), rtol=5e-3)

        test(get("HELIOTRON"))

    @pytest.mark.unit
    def test_energy(self):
        """Test calculation of MHD energy."""

        def test(eq):
            obj = Energy(target=0, weight=mu_0, eq=eq, normalize=False)
            W = obj.compute(*obj.xs(eq))
            W_scaled = obj.compute_scaled(*obj.xs(eq))
            np.testing.assert_allclose(W, 10 / mu_0)
            np.testing.assert_allclose(W_scaled, 10)

        test(Equilibrium(node_pattern="quad", iota=PowerSeriesProfile(0)))
        test(Equilibrium(node_pattern="quad", current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_target_iota(self):
        """Test calculation of iota profile."""

        def test(eq):
            obj = RotationalTransform(target=1, weight=2, eq=eq)
            iota = obj.compute(*obj.xs(eq))
            iota_scaled = obj.compute_scaled(*obj.xs(eq))
            np.testing.assert_allclose(iota, 0)
            np.testing.assert_allclose(iota_scaled, -2 / np.sqrt(3))

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_toroidal_current(self):
        """Test calculation of toroidal current."""

        def test(eq):
            obj = ToroidalCurrent(target=1, weight=2, eq=eq, normalize=False)
            I = obj.compute(*obj.xs(eq))
            I_scaled = obj.compute_scaled(*obj.xs(eq))
            np.testing.assert_allclose(I, 0)
            np.testing.assert_allclose(I_scaled, -2 / np.sqrt(3))

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_qa_boozer(self):
        """Test calculation of Boozer QA metric."""

        def test(eq):
            obj = QuasisymmetryBoozer(eq=eq)
            fb = obj.compute(*obj.xs(eq))
            np.testing.assert_allclose(fb, 0, atol=1e-12)

        test(Equilibrium(L=2, M=2, N=1, iota=PowerSeriesProfile(0)))
        test(Equilibrium(L=2, M=2, N=1, current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_qh_boozer(self):
        """Test calculation of Boozer QH metric."""
        eq = get("WISTELL-A")  # WISTELL-A is optimized for QH symmetry
        helicity = (1, -eq.NFP)
        M_booz = eq.M
        N_booz = eq.N
        grid = LinearGrid(M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP, sym=False)

        # objective function returns amplitudes of non-symmetric modes
        obj = QuasisymmetryBoozer(
            helicity=helicity,
            M_booz=M_booz,
            N_booz=N_booz,
            grid=grid,
            normalize=False,
            eq=eq,
        )
        f = obj.compute(*obj.xs(eq))
        idx_f = np.argsort(np.abs(f))

        # compute all amplitudes in the Boozer spectrum
        transforms = get_transforms(
            "|B|_mn",
            eq=eq,
            grid=grid,
            M_booz=M_booz,
            N_booz=N_booz,
        )
        matrix, modes, idx = ptolemy_linear_transform(
            transforms["B"].basis.modes, helicity=helicity, NFP=eq.NFP
        )
        data = eq.compute("|B|_mn", helicity=helicity, grid=grid, transforms=transforms)
        B_mn = matrix @ data["|B|_mn"]
        idx_B = np.argsort(np.abs(B_mn))

        # check that largest amplitudes are the QH modes
        np.testing.assert_allclose(B_mn[idx_B[-3:]], np.flip(B_mn[~idx][:3]))
        # check that these QH modes are not returned by the objective
        assert [b not in f for b in B_mn[idx_B[-3:]]]
        # check that the objective returns the lowest amplitudes
        np.testing.assert_allclose(f[idx_f][:131], B_mn[idx_B][:131])

    @pytest.mark.unit
    def test_qs_twoterm(self):
        """Test calculation of two term QS metric."""

        def test(eq):
            obj = QuasisymmetryTwoTerm(eq=eq)
            fc = obj.compute(*obj.xs(eq))
            np.testing.assert_allclose(fc, 0)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_qs_tripleproduct(self):
        """Test calculation of triple product QS metric."""

        def test(eq):
            obj = QuasisymmetryTripleProduct(eq=eq)
            ft = obj.compute(*obj.xs(eq))
            np.testing.assert_allclose(ft, 0)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_qs_boozer_grids(self):
        """Test grid compatability with QS objectives."""
        eq = get("QAS")

        # symmetric grid
        grid = LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP, sym=True)
        with pytest.raises(AssertionError):
            _ = QuasisymmetryBoozer(eq=eq, grid=grid)

        # multiple flux surfaces
        grid = LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP, rho=[0.25, 0.5, 0.75, 1])
        with pytest.raises(AssertionError):
            _ = QuasisymmetryBoozer(eq=eq, grid=grid)

    @pytest.mark.unit
    def test_mercier_stability(self):
        """Test calculation of mercier stability criteria."""

        def test(eq):
            obj = MercierStability(eq=eq)
            DMerc = obj.compute(*obj.xs(eq))
            np.testing.assert_equal(len(DMerc), obj.grid.num_rho)
            np.testing.assert_allclose(DMerc, 0)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_magnetic_well(self):
        """Test calculation of magnetic well stability criteria."""

        def test(eq):
            obj = MagneticWell(eq=eq)
            magnetic_well = obj.compute(*obj.xs(eq))
            np.testing.assert_equal(len(magnetic_well), obj.grid.num_rho)
            np.testing.assert_allclose(magnetic_well, 0, atol=1e-15)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))


@pytest.mark.unit
def test_derivative_modes():
    """Test equality of derivatives using batched and blocked methods."""
    eq = Equilibrium(M=2, N=1, L=2)
    obj1 = ObjectiveFunction(MagneticWell(), deriv_mode="batched", use_jit=False)
    obj2 = ObjectiveFunction(MagneticWell(), deriv_mode="blocked", use_jit=False)

    obj1.build(eq)
    obj2.build(eq)
    x = obj1.x(eq)
    g1 = obj1.grad(x)
    g2 = obj2.grad(x)
    np.testing.assert_allclose(g1, g2, atol=1e-10)
    J1 = obj1.jac(x)
    J2 = obj2.jac(x)
    np.testing.assert_allclose(J1, J2, atol=1e-10)
    H1 = obj1.hess(x)
    H2 = obj2.hess(x)
    np.testing.assert_allclose(np.diag(H1), np.diag(H2), atol=1e-10)


@pytest.mark.unit
def test_rejit():
    """Test that updating attributes and recompiling correctly updates."""

    class DummyObjective(_Objective):
        def __init__(self, y, eq=None, target=0, weight=1, name="dummy"):
            self.y = y
            super().__init__(eq=eq, target=target, weight=weight, name=name)

        def build(self, eq, use_jit=True, verbose=1):
            self._dim_f = 1
            super().build(eq, use_jit, verbose)

        def compute(self, R_lmn):
            return 200 + self.target * self.weight - self.y * R_lmn**3

    obj = DummyObjective(3)
    eq = Equilibrium()
    obj.build(eq)
    assert obj.compute(4) == 8
    assert obj.compute_scaled(4) == 8
    obj.target = 1
    obj.weight = 2
    assert obj.compute(4) == 10  # compute method is not JIT compiled
    assert obj.compute_scaled(4) == 8  # only compute_scaled is JIT compiled
    obj.jit()
    assert obj.compute(4) == 10
    assert obj.compute_scaled(4) == 18

    objFun = ObjectiveFunction(obj)
    objFun.build(eq)
    x = objFun.x(eq)

    f = objFun.compute(x)
    J = objFun.jac(x)
    np.testing.assert_allclose(f, [-5598, 402, 396])
    np.testing.assert_allclose(J, np.diag([-1800, 0, -18]))
    objFun.objectives[0].target = 3
    objFun.objectives[0].weight = 4
    objFun.objectives[0].y = 2
    np.testing.assert_allclose(objFun.compute(x), f)
    np.testing.assert_allclose(objFun.jac(x), J)
    objFun.jit()
    np.testing.assert_allclose(objFun.compute(x), [-7164, 836, 828])
    np.testing.assert_allclose(objFun.jac(x), J * 4 / 3)


@pytest.mark.unit
def test_generic_compute():
    """Test for gh issue #388."""
    eq = Equilibrium()
    obj = ObjectiveFunction(AspectRatio(target=2, weight=1), eq=eq)
    a1 = obj.compute_scalar(obj.x(eq))
    obj = ObjectiveFunction(GenericObjective("R0/a", target=2, weight=1), eq=eq)
    a2 = obj.compute_scalar(obj.x(eq))
    assert np.allclose(a1, a2)


@pytest.mark.unit
def test_getter_setter():
    """Test getter and setter methods of Objectives."""
    eq = Equilibrium()
    obj = GenericObjective("R", eq=eq)
    R = obj.compute(*obj.xs(eq))

    # target
    target = R - 0.5
    obj.target = target
    np.testing.assert_allclose(obj.target, target)

    # bounds
    bounds = (0.5 * R, 2 * R)
    obj.bounds = bounds
    np.testing.assert_allclose(obj.bounds, bounds)

    # weight
    weight = R
    obj.weight = weight
    np.testing.assert_allclose(obj.weight, weight)


@pytest.mark.unit
def test_bounds_format():
    """Test that tuple targets are in the format (lower bound, upper bound)."""
    eq = Equilibrium()
    with pytest.raises(AssertionError):
        _ = GenericObjective("R", bounds=(1,), eq=eq)
    with pytest.raises(AssertionError):
        _ = GenericObjective("R", bounds=(1, 2, 3), eq=eq)
    with pytest.raises(ValueError):
        _ = GenericObjective("R", bounds=(1, -1), eq=eq)


@pytest.mark.unit
def test_target_profiles():
    """Tests for using Profile objects as targets for profile objectives."""
    iota = PowerSeriesProfile([1, 0, -0.3])
    current = PowerSeriesProfile([4, 0, 1, 0, -1])
    eqi = Equilibrium(L=5, N=3, M=3, iota=iota)
    eqc = Equilibrium(L=3, N=3, M=3, current=current)
    obji = RotationalTransform(target=iota)
    obji.build(eqc)
    np.testing.assert_allclose(
        obji.target, iota(obji.grid.nodes[obji.grid.unique_rho_idx])
    )
    objc = ToroidalCurrent(target=current)
    objc.build(eqi)
    np.testing.assert_allclose(
        objc.target, current(objc.grid.nodes[objc.grid.unique_rho_idx])
    )


@pytest.mark.unit
def test_plasma_vessel_distance():
    """Test calculation of min distance from plasma to vessel."""
    R0 = 10.0
    a_p = 1.0
    a_s = 2.0
    # default eq has R0=10, a=1
    eq = Equilibrium(M=3, N=2)
    # surface with same R0, a=2, so true d=1 for all pts
    surface = FourierRZToroidalSurface(
        R_lmn=[R0, a_s], Z_lmn=[-a_s], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
    )
    # For equally spaced grids, should get true d=1
    surf_grid = LinearGrid(M=5, N=6)
    plas_grid = LinearGrid(M=5, N=6)
    obj = PlasmaVesselDistance(
        eq=eq, plasma_grid=plas_grid, surface_grid=surf_grid, surface=surface
    )
    d = obj.compute(*obj.xs(eq))
    np.testing.assert_allclose(d, a_s - a_p)

    # for unequal M, should have error of order M_spacing*a_p
    surf_grid = LinearGrid(M=5, N=6)
    plas_grid = LinearGrid(M=10, N=6)
    obj = PlasmaVesselDistance(
        eq=eq, plasma_grid=plas_grid, surface_grid=surf_grid, surface=surface
    )
    d = obj.compute(*obj.xs(eq))
    assert abs(d.min() - (a_s - a_p)) < 1e-14
    assert abs(d.max() - (a_s - a_p)) < surf_grid.spacing[0, 1] * a_p

    # for unequal N, should have error of order N_spacing*R0
    surf_grid = LinearGrid(M=5, N=6)
    plas_grid = LinearGrid(M=5, N=12)
    obj = PlasmaVesselDistance(
        eq=eq, plasma_grid=plas_grid, surface_grid=surf_grid, surface=surface
    )
    d = obj.compute(*obj.xs(eq))
    assert abs(d.min() - (a_s - a_p)) < 1e-14
    assert abs(d.max() - (a_s - a_p)) < surf_grid.spacing[0, 2] * R0

    grid = LinearGrid(L=3, M=3, N=3)
    eq = Equilibrium()
    surf = FourierRZToroidalSurface()
    obj = PlasmaVesselDistance(surface=surf, surface_grid=grid, plasma_grid=grid)
    with pytest.warns(UserWarning):
        obj.build(eq)


@pytest.mark.unit
def test_mean_curvature():
    """Test for mean curvature objective function."""
    # simple case like dshape should have mean curvature negative everywhere
    eq = get("DSHAPE")
    obj = MeanCurvature(eq=eq)
    H = obj.compute(*obj.xs(eq))
    assert np.all(H <= 0)

    # more shaped case like NCSX should have some positive curvature
    eq = get("NCSX")
    obj = MeanCurvature(eq=eq)
    H = obj.compute(*obj.xs(eq))
    assert np.any(H > 0)


@pytest.mark.unit
def test_principal_curvature():
    """Test for principal curvature objective function."""
    eq1 = get("DSHAPE")
    eq2 = get("NCSX")
    obj1 = PrincipalCurvature(eq=eq1, normalize=False)
    K1 = obj1.compute(*obj1.xs(eq1))
    obj2 = PrincipalCurvature(eq=eq2, normalize=False)
    K2 = obj2.compute(*obj2.xs(eq2))

    # simple test: NCSX should have higher mean absolute curvature than DSHAPE
    assert K1.mean() < K2.mean()


@pytest.mark.unit
def test_objective_print(capsys):
    """Test that the profile objectives prints correctly."""
    eq = Equilibrium()
    grid = LinearGrid(L=10, M=10, N=5, axis=False)

    def test(obj, values, normalize=False):

        obj.print_value(*obj.xs(eq))
        out = capsys.readouterr()

        corr_out = str(
            "Precomputing transforms\n"
            + "Maximum "
            + obj._print_value_fmt.format(np.max(values))
            + obj._units
            + "\n"
            + "Minimum "
            + obj._print_value_fmt.format(np.min(values))
            + obj._units
            + "\n"
            + "Average "
            + obj._print_value_fmt.format(np.mean(values))
            + obj._units
            + "\n"
        )
        if normalize:
            corr_out += str(
                "Maximum "
                + obj._print_value_fmt.format(np.max(values / obj.normalization))
                + "(normalized)"
                + "\n"
                + "Minimum "
                + obj._print_value_fmt.format(np.min(values / obj.normalization))
                + "(normalized)"
                + "\n"
                + "Average "
                + obj._print_value_fmt.format(np.mean(values / obj.normalization))
                + "(normalized)"
                + "\n"
            )

        assert out.out == corr_out

    iota = eq.compute("iota", grid=grid)["iota"]
    obj = RotationalTransform(eq=eq, grid=grid)
    test(obj, iota)
    curr = eq.compute("current", grid=grid)["current"]
    obj = ToroidalCurrent(eq=eq, grid=grid)
    test(obj, curr, normalize=True)
