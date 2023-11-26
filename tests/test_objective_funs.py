"""Tests for objective functions.

These generally don't test the accuracy of the computation for realistic examples,
that is done in test_compute_functions or regression tests.

This module primarily tests the constructing/building/calling methods.
"""

import numpy as np
import pytest
from scipy.constants import elementary_charge, mu_0

import desc.examples
from desc.backend import jnp
from desc.compute import get_transforms
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid
from desc.objectives import (
    AspectRatio,
    BootstrapRedlConsistency,
    BScaleLength,
    CurrentDensity,
    Elongation,
    Energy,
    ForceBalance,
    ForceBalanceAnisotropic,
    GenericObjective,
    GoodCoordinates,
    HelicalForceBalance,
    Isodynamicity,
    MagneticWell,
    MeanCurvature,
    MercierStability,
    ObjectiveFromUser,
    ObjectiveFunction,
    PlasmaVesselDistance,
    Pressure,
    PrincipalCurvature,
    QuasisymmetryBoozer,
    QuasisymmetryTripleProduct,
    QuasisymmetryTwoTerm,
    RadialForceBalance,
    RotationalTransform,
    Shear,
    ToroidalCurrent,
    Volume,
)
from desc.objectives.objective_funs import _Objective
from desc.objectives.utils import softmax, softmin
from desc.profiles import FourierZernikeProfile, PowerSeriesProfile
from desc.vmec_utils import ptolemy_linear_transform


class TestObjectiveFunction:
    """Test ObjectiveFunction classes."""

    @pytest.mark.unit
    def test_generic(self):
        """Test GenericObjective for arbitrary quantities."""

        def test(f, eq, compress=False):
            obj = GenericObjective(f, eq=eq)
            obj.build()
            val = eq.compute(f, grid=obj.constants["transforms"]["grid"])[f]
            if compress:
                val = obj.constants["transforms"]["grid"].compress(val)
            np.testing.assert_allclose(
                obj.compute(eq.params_dict),
                val,
            )

        test("sqrt(g)", Equilibrium())
        test("current", Equilibrium(iota=PowerSeriesProfile(0)), True)
        test("iota", Equilibrium(current=PowerSeriesProfile(0)), True)

    @pytest.mark.unit
    def test_objective_from_user(self):
        """Test ObjectiveFromUser for arbitrary callable."""

        def myfun(grid, data):
            x = data["X"]
            y = data["Y"]
            r = jnp.sqrt(x * data["X"] + y**2)
            return r

        eq = Equilibrium()
        grid = LinearGrid(2, 2, 2)
        objective = ObjectiveFromUser(myfun, eq=eq, grid=grid)
        objective.build()
        R1 = objective.compute(*objective.xs(eq))
        R2 = eq.compute("R", grid=grid)["R"]
        np.testing.assert_allclose(R1, R2)

    @pytest.mark.unit
    def test_volume(self):
        """Test calculation of plasma volume."""

        def test(eq):
            obj = Volume(
                target=10 * np.pi**2, weight=1 / np.pi**2, eq=eq, normalize=False
            )
            obj.build()
            V = obj.compute_unscaled(*obj.xs(eq))
            V_scaled = obj.compute_scaled_error(*obj.xs(eq))
            V_scalar = obj.compute_scalar(*obj.xs(eq))
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
            obj.build()
            AR = obj.compute_unscaled(*obj.xs(eq))
            AR_scaled = obj.compute_scaled_error(*obj.xs(eq))
            np.testing.assert_allclose(AR, 10)
            np.testing.assert_allclose(AR_scaled, 5)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_elongation(self):
        """Test calculation of elongation."""

        def test(eq):
            obj = Elongation(target=0, weight=2, eq=eq)
            obj.build()
            f = obj.compute_unscaled(*obj.xs(eq))
            f_scaled = obj.compute_scaled_error(*obj.xs(eq))
            np.testing.assert_allclose(f, 1.3 / 0.7, rtol=5e-3)
            np.testing.assert_allclose(f_scaled, 2 * (1.3 / 0.7), rtol=5e-3)

        test(get("HELIOTRON"))

    @pytest.mark.unit
    def test_energy(self):
        """Test calculation of MHD energy."""

        def test(eq):
            obj = Energy(target=0, weight=mu_0, eq=eq, normalize=False)
            obj.build()
            W = obj.compute_unscaled(*obj.xs(eq))
            W_scaled = obj.compute_scaled_error(*obj.xs(eq))
            np.testing.assert_allclose(W, 10 / mu_0)
            np.testing.assert_allclose(W_scaled, 10)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_target_iota(self):
        """Test calculation of iota profile."""

        def test(eq):
            obj = RotationalTransform(target=1, weight=2, eq=eq)
            obj.build()
            iota = obj.compute_unscaled(*obj.xs(eq))
            iota_scaled = obj.compute_scaled_error(*obj.xs(eq))
            np.testing.assert_allclose(iota, 0)
            np.testing.assert_allclose(iota_scaled, -2 / np.sqrt(3))

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_target_shear(self):
        """Test calculation of shear profile."""

        def test(eq, raw, scaled):
            obj = Shear(target=-1, weight=2, eq=eq)
            obj.build()
            shear = obj.compute_unscaled(*obj.xs(eq))
            shear_scaled = obj.compute_scaled_error(*obj.xs(eq))
            np.testing.assert_allclose(shear, raw)
            np.testing.assert_allclose(shear_scaled, scaled)

        test(Equilibrium(iota=PowerSeriesProfile(0)), 0, 2 / np.sqrt(3))
        test(Equilibrium(current=PowerSeriesProfile(0)), 0, 2 / np.sqrt(3))
        test(Equilibrium(iota=PowerSeriesProfile([0, 0, 0.5])), -2, -2 / np.sqrt(3))

    @pytest.mark.unit
    def test_toroidal_current(self):
        """Test calculation of toroidal current."""

        def test(eq):
            obj = ToroidalCurrent(target=1, weight=2, eq=eq, normalize=False)
            obj.build()
            I = obj.compute_unscaled(*obj.xs(eq))
            I_scaled = obj.compute_scaled_error(*obj.xs(eq))
            np.testing.assert_allclose(I, 0)
            np.testing.assert_allclose(I_scaled, -2 / np.sqrt(3))

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_pressure(self):
        """Test calculation of pressure objective."""

        def test(eq):
            obj = Pressure(target=1, weight=2, eq=eq, normalize=False)
            obj.build()
            p = obj.compute_unscaled(*obj.xs(eq))
            p_scaled = obj.compute_scaled_error(*obj.xs(eq))
            np.testing.assert_allclose(p, 12)
            # (value - target) * objective weight * quadrature weights
            # in this case, both value and target are constant wrt rho
            np.testing.assert_allclose(p_scaled, (12 - 1) * 2 / np.sqrt(3))

        test(Equilibrium(pressure=PowerSeriesProfile(12)))
        test(
            Equilibrium(
                electron_temperature=PowerSeriesProfile(2),
                electron_density=PowerSeriesProfile(3 / elementary_charge),
            )
        )

    @pytest.mark.unit
    def test_qa_boozer(self):
        """Test calculation of Boozer QA metric."""

        def test(eq):
            obj = QuasisymmetryBoozer(eq=eq)
            obj.build()
            fb = obj.compute_unscaled(*obj.xs(eq))
            np.testing.assert_allclose(fb, 0, atol=1e-12)

        test(Equilibrium(L=2, M=2, N=1, iota=PowerSeriesProfile(0)))
        test(Equilibrium(L=2, M=2, N=1, current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_jax_compile_boozer(self):
        """Test compilation of Boozer QA metric in ObjectiveFunction."""

        def test(eq):
            """Ensure compilation without any errors from JAX, related to issue #625."""
            obj = ObjectiveFunction(QuasisymmetryBoozer(eq=eq))
            obj.build()
            obj.compile()
            fb = obj.compute_scaled_error(obj.x(eq))
            np.testing.assert_allclose(fb, 0, atol=1e-12)

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
        obj.build()
        f = obj.compute_unscaled(*obj.xs(eq))
        idx_f = np.argsort(np.abs(f))

        # compute all amplitudes in the Boozer spectrum
        transforms = get_transforms(
            "|B|_mn", obj=eq, grid=grid, M_booz=M_booz, N_booz=N_booz
        )
        matrix, modes, idx = ptolemy_linear_transform(
            transforms["B"].basis.modes, helicity=helicity, NFP=eq.NFP
        )
        data = eq.compute("|B|_mn", helicity=helicity, grid=grid, transforms=transforms)
        B_mn = matrix @ data["|B|_mn"]
        idx_B = np.argsort(np.abs(B_mn))

        # check that largest amplitudes are the QH modes
        np.testing.assert_allclose(B_mn[idx_B[-3:]], np.flip(np.delete(B_mn, idx)[:3]))
        # check that these QH modes are not returned by the objective
        assert [b not in f for b in B_mn[idx_B[-3:]]]
        # check that the objective returns the lowest amplitudes
        # 120 ~ smallest amplitudes BEFORE QH modes show up so that sorting both arrays
        # should have the same values up until then
        np.testing.assert_allclose(f[idx_f][:120], B_mn[idx_B][:120])

    @pytest.mark.unit
    def test_qs_twoterm(self):
        """Test calculation of two term QS metric."""

        def test(eq):
            obj = QuasisymmetryTwoTerm(eq=eq)
            obj.build()
            fc = obj.compute_unscaled(*obj.xs(eq))
            np.testing.assert_allclose(fc, 0)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

        # also make sure helicity is set correctly
        eq1 = desc.examples.get("precise_QA")
        eq2 = desc.examples.get("precise_QH")

        helicity_QA = (1, 0)
        helicity_QH = (1, eq2.NFP)

        # precise_QA should have lower QA than QH
        obj = QuasisymmetryTwoTerm(eq=eq1, helicity=helicity_QA)
        obj.build()
        f1 = obj.compute_scalar(*obj.xs(eq1), constants=obj.constants)
        obj.helicity = helicity_QH
        obj.build()
        f2 = obj.compute_scalar(*obj.xs(eq1), constants=obj.constants)
        assert f1 < f2

        # precise_QH should have lower QH than QA
        obj = QuasisymmetryTwoTerm(eq=eq2, helicity=helicity_QH)
        obj.build()
        f1 = obj.compute_scalar(*obj.xs(eq2), constants=obj.constants)
        obj.helicity = helicity_QA
        obj.build()
        f2 = obj.compute_scalar(*obj.xs(eq2), constants=obj.constants)
        assert f1 < f2

    @pytest.mark.unit
    def test_qs_tripleproduct(self):
        """Test calculation of triple product QS metric."""

        def test(eq):
            obj = QuasisymmetryTripleProduct(eq=eq)
            obj.build()
            ft = obj.compute_unscaled(*obj.xs(eq))
            np.testing.assert_allclose(ft, 0, atol=5e-35)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_isodynamicity(self):
        """Test calculation of isodynamicity metric."""

        def test(eq):
            obj = Isodynamicity(eq=eq)
            obj.build()
            iso = obj.compute(*obj.xs(eq))
            np.testing.assert_allclose(iso, 0, atol=1e-14)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_qs_boozer_grids(self):
        """Test grid compatibility with QS objectives."""
        eq = get("NCSX")

        # symmetric grid
        grid = LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP, sym=True)
        with pytest.raises(AssertionError):
            QuasisymmetryBoozer(eq=eq, grid=grid).build()

        # multiple flux surfaces
        grid = LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP, rho=[0.25, 0.5, 0.75, 1])
        with pytest.raises(AssertionError):
            QuasisymmetryBoozer(eq=eq, grid=grid).build()

    @pytest.mark.unit
    def test_mercier_stability(self):
        """Test calculation of mercier stability criteria."""

        def test(eq):
            obj = MercierStability(eq=eq)
            obj.build()
            DMerc = obj.compute_unscaled(*obj.xs(eq))
            np.testing.assert_equal(
                len(DMerc), obj.constants["transforms"]["grid"].num_rho
            )
            np.testing.assert_allclose(DMerc, 0)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_magnetic_well(self):
        """Test calculation of magnetic well stability criteria."""

        def test(eq):
            obj = MagneticWell(eq=eq)
            obj.build()
            magnetic_well = obj.compute_unscaled(*obj.xs(eq))
            np.testing.assert_equal(
                len(magnetic_well), obj.constants["transforms"]["grid"].num_rho
            )
            np.testing.assert_allclose(magnetic_well, 0, atol=1e-15)

        test(Equilibrium(iota=PowerSeriesProfile(0)))
        test(Equilibrium(current=PowerSeriesProfile(0)))

    @pytest.mark.unit
    def test_target_mean_iota(self):
        """Test calculation of iota profile average."""

        def test(eq):
            grid = LinearGrid(L=5, M=1, N=1, NFP=eq.NFP)
            mean_iota = jnp.mean(eq.compute("iota", grid=grid)["iota"])
            obj = RotationalTransform(
                target=mean_iota, weight=1, eq=eq, loss_function="mean", grid=grid
            )
            obj.build()
            mean_iota_unscaled = obj.compute_unscaled(*obj.xs(eq))
            mean_iota_scaled_error = obj.compute_scaled_error(*obj.xs(eq))
            mean_iota_scaled = obj.compute_scaled(*obj.xs(eq))
            np.testing.assert_allclose(mean_iota, mean_iota_unscaled, atol=1e-16)
            np.testing.assert_allclose(mean_iota_scaled_error, 0, atol=5e-16)
            np.testing.assert_allclose(mean_iota_scaled, mean_iota, atol=5e-16)

        test(get("DSHAPE"))
        test(get("HELIOTRON"))

    @pytest.mark.unit
    def test_target_max_iota(self):
        """Test calculation of iota profile max."""

        def test(eq):
            grid = LinearGrid(L=5, M=1, N=1, NFP=eq.NFP)
            max_iota = jnp.max(eq.compute("iota", grid=grid)["iota"])
            obj = RotationalTransform(
                target=max_iota, weight=1, eq=eq, loss_function="max", grid=grid
            )
            obj.build()
            max_iota_unscaled = obj.compute_unscaled(*obj.xs(eq))
            max_iota_scaled_error = obj.compute_scaled_error(*obj.xs(eq))
            max_iota_scaled = obj.compute_scaled(*obj.xs(eq))
            np.testing.assert_allclose(max_iota, max_iota_unscaled, atol=1e-16)
            np.testing.assert_allclose(max_iota_scaled_error, 0, atol=5e-16)
            np.testing.assert_allclose(max_iota_scaled, max_iota, atol=5e-16)

        test(get("DSHAPE"))
        test(get("HELIOTRON"))

    @pytest.mark.unit
    def test_target_min_iota(self):
        """Test calculation of iota profile min."""

        def test(eq):
            grid = LinearGrid(L=5, M=1, N=1, NFP=eq.NFP)
            min_iota = jnp.min(eq.compute("iota", grid=grid)["iota"])
            obj = RotationalTransform(
                target=min_iota, weight=1, eq=eq, loss_function="min", grid=grid
            )
            obj.build()
            min_iota_unscaled = obj.compute_unscaled(*obj.xs(eq))
            min_iota_scaled_error = obj.compute_scaled_error(*obj.xs(eq))
            min_iota_scaled = obj.compute_scaled(*obj.xs(eq))
            np.testing.assert_allclose(min_iota, min_iota_unscaled, atol=1e-16)
            np.testing.assert_allclose(min_iota_scaled_error, 0, atol=5e-16)
            np.testing.assert_allclose(min_iota_scaled, min_iota, atol=5e-16)

        test(get("DSHAPE"))
        test(get("HELIOTRON"))


@pytest.mark.unit
def test_derivative_modes():
    """Test equality of derivatives using batched, looped methods."""
    eq = Equilibrium(M=2, N=1, L=2)
    surf = FourierRZToroidalSurface()
    obj1 = ObjectiveFunction(
        [
            PlasmaVesselDistance(eq, surf),
            MagneticWell(eq),
        ],
        deriv_mode="batched",
        use_jit=False,
    )
    obj2 = ObjectiveFunction(
        [
            PlasmaVesselDistance(eq, surf),
            MagneticWell(eq),
        ],
        deriv_mode="blocked",
        use_jit=False,
    )
    obj3 = ObjectiveFunction(
        [
            PlasmaVesselDistance(eq, surf),
            MagneticWell(eq),
        ],
        deriv_mode="looped",
        use_jit=False,
    )

    obj1.build()
    obj2.build()
    obj3.build()
    x = obj1.x(eq, surf)
    g1 = obj1.grad(x)
    g2 = obj2.grad(x)
    g3 = obj3.grad(x)
    np.testing.assert_allclose(g1, g2, atol=1e-10)
    np.testing.assert_allclose(g1, g3, atol=1e-10)
    J1 = obj1.jac_scaled(x)
    J2 = obj2.jac_scaled(x)
    J3 = obj3.jac_scaled(x)
    np.testing.assert_allclose(J1, J2, atol=1e-10)
    np.testing.assert_allclose(J1, J3, atol=1e-10)
    J1 = obj1.jac_unscaled(x)
    J2 = obj2.jac_unscaled(x)
    J3 = obj3.jac_unscaled(x)
    np.testing.assert_allclose(J1, J2, atol=1e-10)
    np.testing.assert_allclose(J1, J3, atol=1e-10)
    H1 = obj1.hess(x)
    H2 = obj2.hess(x)
    H3 = obj3.hess(x)
    np.testing.assert_allclose(H1, H2, atol=1e-10)
    np.testing.assert_allclose(H1, H3, atol=1e-10)


@pytest.mark.unit
@pytest.mark.xfail
def test_rejit():
    """Test that updating attributes and recompiling correctly updates."""

    class DummyObjective(_Objective):
        def __init__(self, y, eq=None, target=0, weight=1, name="dummy"):
            self.y = y
            super().__init__(things=eq, target=target, weight=weight, name=name)

        def build(self, use_jit=True, verbose=1):
            self._dim_f = 1
            super().build(use_jit, verbose)

        def compute(self, params, constants=None):
            return 200 + self.target * self.weight - self.y * params["R_lmn"] ** 3

    eq = Equilibrium()
    obj = DummyObjective(3, eq=eq)
    obj.build()
    assert obj.compute_unscaled({"R_lmn": 4}) == 8
    assert obj.compute_scaled_error({"R_lmn": 4}) == 8
    obj.target = 1
    obj.weight = 2
    assert obj.compute({"R_lmn": 4}) == 10  # compute method is not JIT compiled
    assert (
        obj.compute_scaled_error({"R_lmn": 4}) == 8
    )  # only compute_scaled is JIT compiled
    obj.jit()
    assert obj.compute({"R_lmn": 4}) == 10
    assert obj.compute_scaled_error({"R_lmn": 4}) == 18

    objFun = ObjectiveFunction(obj)
    objFun.build()
    x = objFun.x(eq)

    f = objFun.compute_scaled_error(x)
    J = objFun.jac_scaled(x)
    np.testing.assert_allclose(f, [-5598, 402, 396])
    np.testing.assert_allclose(J[:, eq.x_idx["R_lmn"]], np.diag([-1800, 0, -18]))
    objFun.objectives[0].target = 3
    objFun.objectives[0].weight = 4
    objFun.objectives[0].y = 2
    np.testing.assert_allclose(objFun.compute_scaled_error(x), f)
    np.testing.assert_allclose(objFun.jac_scaled(x), J)
    objFun.jit()
    np.testing.assert_allclose(objFun.compute_scaled_error(x), [-7164, 836, 828])
    np.testing.assert_allclose(objFun.jac_scaled(x), J * 4 / 3)


@pytest.mark.unit
def test_generic_compute():
    """Test for gh issue #388."""
    eq = Equilibrium()
    obj = ObjectiveFunction(AspectRatio(target=2, weight=1, eq=eq))
    obj.build()
    a1 = obj.compute_scalar(obj.x(eq))
    obj = ObjectiveFunction(GenericObjective("R0/a", target=2, weight=1, eq=eq))
    obj.build()
    a2 = obj.compute_scalar(obj.x(eq))
    assert np.allclose(a1, a2)


@pytest.mark.unit
def test_getter_setter():
    """Test getter and setter methods of Objectives."""
    eq = Equilibrium()
    obj = GenericObjective("R", eq=eq)
    obj.build()
    R = obj.compute_unscaled(*obj.xs(eq))

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
        GenericObjective("R", bounds=(1,), eq=eq).build()
    with pytest.raises(AssertionError):
        GenericObjective("R", bounds=(1, 2, 3), eq=eq).build()
    with pytest.raises(ValueError):
        GenericObjective("R", bounds=(1, -1), eq=eq).build()


@pytest.mark.unit
def test_target_profiles():
    """Tests for using Profile objects as targets for profile objectives."""
    iota = PowerSeriesProfile([1, 0, -0.3])
    shear = PowerSeriesProfile([0, -0.6])
    current = PowerSeriesProfile([4, 0, 1, 0, -1])
    merc = PowerSeriesProfile([1, 0, -1])
    well = PowerSeriesProfile([2, 0, -2])
    pres = PowerSeriesProfile([3, 0, -3])
    eqi = Equilibrium(L=5, N=3, M=3, iota=iota)
    eqc = Equilibrium(L=3, N=3, M=3, current=current)
    obji = RotationalTransform(target=iota, eq=eqi)
    obji.build()
    np.testing.assert_allclose(
        obji.target,
        iota(
            obji.constants["transforms"]["grid"].nodes[
                obji.constants["transforms"]["grid"].unique_rho_idx
            ]
        ),
    )
    objs = Shear(target=shear, eq=eqi)
    objs.build()
    np.testing.assert_allclose(
        objs.target,
        shear(
            objs.constants["transforms"]["grid"].nodes[
                objs.constants["transforms"]["grid"].unique_rho_idx
            ]
        ),
    )
    objc = ToroidalCurrent(target=current, eq=eqc)
    objc.build()
    np.testing.assert_allclose(
        objc.target,
        current(
            objc.constants["transforms"]["grid"].nodes[
                objc.constants["transforms"]["grid"].unique_rho_idx
            ]
        ),
    )
    objm = MercierStability(bounds=(merc, np.inf), eq=eqi)
    objm.build()
    np.testing.assert_allclose(
        objm.bounds[0],
        merc(
            objm.constants["transforms"]["grid"].nodes[
                objm.constants["transforms"]["grid"].unique_rho_idx
            ]
        ),
    )
    np.testing.assert_allclose(objm.bounds[1], np.inf)
    objw = MagneticWell(bounds=(merc, well), eq=eqi)
    objw.build()
    np.testing.assert_allclose(
        objw.bounds[0],
        merc(
            objw.constants["transforms"]["grid"].nodes[
                objw.constants["transforms"]["grid"].unique_rho_idx
            ]
        ),
    )
    np.testing.assert_allclose(
        objw.bounds[1],
        well(
            objw.constants["transforms"]["grid"].nodes[
                objw.constants["transforms"]["grid"].unique_rho_idx
            ]
        ),
    )
    objp = Pressure(target=pres, eq=eqc)
    objp.build()
    np.testing.assert_allclose(
        objp.target,
        pres(
            objp.constants["transforms"]["grid"].nodes[
                objp.constants["transforms"]["grid"].unique_rho_idx
            ]
        ),
    )
    objp = Pressure(target=lambda x: 2 * x, eq=eqc)
    objp.build()
    np.testing.assert_allclose(
        objp.target,
        2
        * objp.constants["transforms"]["grid"].nodes[
            objp.constants["transforms"]["grid"].unique_rho_idx, 0
        ],
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
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq, surface))
    np.testing.assert_allclose(d, a_s - a_p)

    # for unequal M, should have error of order M_spacing*a_p
    surf_grid = LinearGrid(M=5, N=6)
    plas_grid = LinearGrid(M=10, N=6)
    obj = PlasmaVesselDistance(
        eq=eq,
        plasma_grid=plas_grid,
        surface_grid=surf_grid,
        surface=surface,
        surface_fixed=True,
    )
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq, surface))
    assert abs(d.min() - (a_s - a_p)) < 1e-14
    assert abs(d.max() - (a_s - a_p)) < surf_grid.spacing[0, 1] * a_p

    # for unequal N, should have error of order N_spacing*R0
    surf_grid = LinearGrid(M=5, N=6)
    plas_grid = LinearGrid(M=5, N=12)
    obj = PlasmaVesselDistance(
        eq=eq, plasma_grid=plas_grid, surface_grid=surf_grid, surface=surface
    )
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq, surface))
    assert abs(d.min() - (a_s - a_p)) < 1e-14
    assert abs(d.max() - (a_s - a_p)) < surf_grid.spacing[0, 2] * R0
    # ensure that it works (dimension-wise) when compute_scaled is called
    _ = obj.compute_scaled(*obj.xs(eq, surface))

    grid = LinearGrid(L=3, M=3, N=3)
    eq = Equilibrium()
    surf = FourierRZToroidalSurface()
    obj = PlasmaVesselDistance(surface=surf, surface_grid=grid, plasma_grid=grid, eq=eq)
    with pytest.warns(UserWarning):
        obj.build()

    # test softmin, should give value less than true minimum
    surf_grid = LinearGrid(M=5, N=6)
    plas_grid = LinearGrid(M=5, N=6)
    obj = PlasmaVesselDistance(
        eq=eq,
        plasma_grid=plas_grid,
        surface_grid=surf_grid,
        surface=surface,
        use_softmin=True,
    )
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq, surface))
    assert np.all(np.abs(d) < a_s - a_p)

    # for large enough alpha, should be same as actual min
    obj = PlasmaVesselDistance(
        eq=eq,
        plasma_grid=plas_grid,
        surface_grid=surf_grid,
        surface=surface,
        use_softmin=True,
        alpha=100,
    )
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq, surface))
    np.testing.assert_allclose(d, a_s - a_p)


@pytest.mark.unit
def test_mean_curvature():
    """Test for mean curvature objective function."""
    # torus should have mean curvature negative everywhere
    eq = Equilibrium()
    obj = MeanCurvature(eq=eq)
    obj.build()
    H = obj.compute_unscaled(*obj.xs(eq))
    assert np.all(H <= 0)

    # more shaped case like NCSX should have some positive curvature
    eq = get("NCSX")
    obj = MeanCurvature(eq=eq)
    obj.build()
    H = obj.compute_unscaled(*obj.xs(eq))
    assert np.any(H > 0)


@pytest.mark.unit
def test_principal_curvature():
    """Test for principal curvature objective function."""
    eq1 = get("DSHAPE")
    eq2 = get("NCSX")
    obj1 = PrincipalCurvature(eq=eq1, normalize=False)
    obj1.build()
    K1 = obj1.compute_unscaled(*obj1.xs(eq1))
    obj2 = PrincipalCurvature(eq=eq2, normalize=False)
    obj2.build()
    K2 = obj2.compute_unscaled(*obj2.xs(eq2))

    # simple test: NCSX should have higher mean absolute curvature than DSHAPE
    assert K1.mean() < K2.mean()


@pytest.mark.unit
def test_field_scale_length():
    """Test for B field scale length objective function."""
    surf1 = FourierRZToroidalSurface(
        R_lmn=[5, 1], Z_lmn=[-1], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]], NFP=1
    )
    surf2 = FourierRZToroidalSurface(
        R_lmn=[10, 2], Z_lmn=[-2], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]], NFP=1
    )
    eq1 = Equilibrium(L=2, M=2, N=0, surface=surf1)
    eq2 = Equilibrium(L=2, M=2, N=0, surface=surf2)
    eq1.solve()
    eq2.solve()

    obj1 = BScaleLength(eq=eq1, normalize=False)
    obj2 = BScaleLength(eq=eq2, normalize=False)
    obj1.build()
    obj2.build()

    L1 = obj1.compute_unscaled(*obj1.xs(eq1))
    L2 = obj2.compute_unscaled(*obj2.xs(eq2))

    np.testing.assert_array_less(L1, L2)


@pytest.mark.unit
def test_profile_objective_print(capsys):
    """Test that the profile objectives print correctly."""
    eq = Equilibrium(
        iota=PowerSeriesProfile([1, 0, 0.5]), pressure=PowerSeriesProfile([1, 0, -1])
    )
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
    obj = RotationalTransform(eq=eq, target=1, grid=grid)
    obj.build()
    test(obj, iota)
    shear = eq.compute("shear", grid=grid)["shear"]
    obj = Shear(eq=eq, target=1, grid=grid)
    obj.build()
    test(obj, shear)
    curr = eq.compute("current", grid=grid)["current"]
    obj = ToroidalCurrent(eq=eq, target=1, grid=grid)
    obj.build()
    test(obj, curr, normalize=True)
    pres = eq.compute("p", grid=grid)["p"]
    obj = Pressure(eq=eq, target=1, grid=grid)
    obj.build()
    test(obj, pres, normalize=True)


@pytest.mark.unit
def test_plasma_vessel_distance_print(capsys):
    """Test that the PlasmaVesselDistance objective prints correctly."""
    R0 = 10.0
    a_p = 1.0
    a_s = 2.0
    # default eq has R0=10, a=1
    eq = Equilibrium(M=3, N=2)
    # surface with same R0, a=2, so true d=1 for all pts
    surface = FourierRZToroidalSurface(
        R_lmn=[R0, a_s], Z_lmn=[-a_s], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
    )
    surf_grid = LinearGrid(M=5, N=0)
    plas_grid = LinearGrid(M=5, N=0)
    obj = PlasmaVesselDistance(
        eq=eq, plasma_grid=plas_grid, surface_grid=surf_grid, surface=surface
    )
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq, surface))
    np.testing.assert_allclose(d, a_s - a_p)

    obj.print_value(*obj.xs(eq, surface))
    out = capsys.readouterr()

    corr_out = str(
        "Precomputing transforms\n"
        + "Maximum "
        + obj._print_value_fmt.format(np.max(d))
        + obj._units
        + "\n"
        + "Minimum "
        + obj._print_value_fmt.format(np.min(d))
        + obj._units
        + "\n"
        + "Average "
        + obj._print_value_fmt.format(np.mean(d))
        + obj._units
        + "\n"
        + "Maximum "
        + obj._print_value_fmt.format(np.max(d / obj.normalization))
        + "(normalized)"
        + "\n"
        + "Minimum "
        + obj._print_value_fmt.format(np.min(d / obj.normalization))
        + "(normalized)"
        + "\n"
        + "Average "
        + obj._print_value_fmt.format(np.mean(d / obj.normalization))
        + "(normalized)"
        + "\n"
    )
    assert out.out == corr_out


@pytest.mark.unit
def test_rebuild():
    """Test that the objective is rebuilt correctly when needed."""
    eq = Equilibrium(L=3, M=3)
    f_obj = ForceBalance(eq=eq)
    obj = ObjectiveFunction(f_obj)
    eq.solve(maxiter=2, objective=obj)

    # this would fail before v0.8.2 when trying to get objective.x
    eq.change_resolution(L=5, M=5)
    obj.build(eq)
    eq.solve(maxiter=2, objective=obj)

    eq = Equilibrium(L=3, M=3)
    f_obj = ForceBalance(eq=eq)
    obj = ObjectiveFunction(f_obj)
    eq.solve(maxiter=2, objective=obj)
    eq.change_resolution(L=5, M=5)
    # this would fail at objective.compile
    obj = ObjectiveFunction(f_obj)
    obj.build(eq)
    eq.solve(maxiter=2, objective=obj)


@pytest.mark.unit
def test_objective_fun_things():
    """Test that the objective things logic works correctly."""
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
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq, surface))
    np.testing.assert_allclose(d, a_s - a_p)

    surface2 = surface.copy()
    eq2 = eq.copy()
    obj.things = [eq2, surface2]
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq2, surface2))
    np.testing.assert_allclose(d, a_s - a_p)

    # change objects, and surface resolution as well
    a_s2 = 2.5
    surface2 = FourierRZToroidalSurface(
        R_lmn=[R0, a_s2], Z_lmn=[-a_s2], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
    )
    eq2 = Equilibrium(M=3, N=2)
    surface2.change_resolution(M=4, N=4)
    obj.things = [eq2, surface2]
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq2, surface2))
    np.testing.assert_allclose(d, a_s2 - a_p)

    # test that works correctly when changing both objects
    # with the resolution of the equilibrium surface also changing
    a_s2 = 2.5
    surface2 = FourierRZToroidalSurface(
        R_lmn=[R0, a_s2], Z_lmn=[-a_s2], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
    )
    eq2 = Equilibrium(surface=surface, M=3, N=2)
    obj.things = [eq2, surface2]
    obj.build()
    d = obj.compute_unscaled(*obj.xs(eq2, surface2))
    np.testing.assert_allclose(d, a_s2 - a_s)

    with pytest.raises(AssertionError):
        # one of these is not optimizable, throws error
        obj.things = [eq, 2.0]
    with pytest.raises(AssertionError):
        # these are not the expected types
        obj.things = [eq, eq2]
    with pytest.raises(AssertionError):
        # these are not in the correct order for the objective
        obj.things = [surface, eq]


@pytest.mark.unit
def test_jvp_scaled():
    """Test that jvps are scaled correctly."""
    eq = Equilibrium()
    weight = 3
    target = 5
    objective = ObjectiveFunction(
        Volume(target=target, normalize=True, weight=weight, eq=eq)
    )
    objective.build()
    x = objective.x(eq)
    dx = x / 100
    jvp1u = objective.jvp_unscaled((dx,), x)
    jvp2u = objective.jvp_unscaled((dx, dx), x)
    jvp3u = objective.jvp_unscaled((dx, dx, dx), x)
    jvp1s = objective.jvp_scaled((dx,), x)
    jvp2s = objective.jvp_scaled((dx, dx), x)
    jvp3s = objective.jvp_scaled((dx, dx, dx), x)

    np.testing.assert_allclose(
        jvp1u / objective._objectives[0].normalization * weight, jvp1s
    )
    np.testing.assert_allclose(
        jvp2u / objective._objectives[0].normalization * weight, jvp2s
    )
    np.testing.assert_allclose(
        jvp3u / objective._objectives[0].normalization * weight, jvp3s
    )

    with pytest.raises(NotImplementedError):
        _ = objective.jvp_scaled((dx, dx, dx, dx), x)

    with pytest.raises(NotImplementedError):
        _ = objective.jvp_unscaled((dx, dx, dx, dx), x)


@pytest.mark.unit
def test_vjp():
    """Test that vjps are scaled correctly."""
    eq = Equilibrium()
    weight = 3
    target = 5
    objective = ObjectiveFunction(
        ForceBalance(target=target, normalize=True, weight=weight, eq=eq)
    )
    objective.build()
    x = objective.x(eq)
    y = np.linspace(0, 1, objective.dim_f)
    vjp1u = objective.vjp_unscaled(y, x)
    vjp1s = objective.vjp_scaled(y, x)
    vjp2u = y @ objective.jac_unscaled(x)
    vjp2s = y @ objective.jac_scaled(x)

    np.testing.assert_allclose(vjp1u, vjp2u, atol=1e-8)
    np.testing.assert_allclose(vjp1s, vjp2s, atol=1e-8)


@pytest.mark.unit
def test_objective_target_bounds():
    """Test that the target_scaled and bounds_scaled etc. return the right things."""
    eq = Equilibrium()

    vol = Volume(target=3, normalize=True, weight=2, eq=eq)
    asp = AspectRatio(bounds=(2, 3), normalize=False, weight=3, eq=eq)
    fbl = ForceBalance(normalize=True, bounds=(-1, 2), weight=5, eq=eq)

    objective = ObjectiveFunction((vol, asp, fbl))
    objective.build()

    target = objective.target_scaled
    bounds = objective.bounds_scaled
    weight = objective.weights

    assert bounds[0][0] == 3 / vol.normalization * vol.weight
    assert bounds[1][0] == 3 / vol.normalization * vol.weight
    assert bounds[0][1] == 2 * asp.weight
    assert bounds[1][1] == 3 * asp.weight
    np.testing.assert_allclose(
        bounds[0][2:],
        (-1 / fbl.normalization * fbl.weight * fbl.constants["quad_weights"]),
    )
    np.testing.assert_allclose(
        bounds[1][2:],
        (2 / fbl.normalization * fbl.weight * fbl.constants["quad_weights"]),
    )

    assert target[0] == 3 / vol.normalization * vol.weight
    assert target[1] == 2.5 * asp.weight
    np.testing.assert_allclose(
        target[2:],
        (0.5 / fbl.normalization * fbl.weight * fbl.constants["quad_weights"]),
    )

    assert weight[0] == 2
    assert weight[1] == 3
    assert np.all(weight[2:] == 5)

    eq = Equilibrium(L=8, M=2, N=2, iota=PowerSeriesProfile(0.42))

    con = ObjectiveFunction(RotationalTransform(eq=eq, bounds=(0.41, 0.43)))
    con.build()

    np.testing.assert_allclose(con.compute_scaled_error(con.x(eq)), 0)
    np.testing.assert_array_less(con.bounds_scaled[0], con.compute_scaled(con.x(eq)))
    np.testing.assert_array_less(con.compute_scaled(con.x(eq)), con.bounds_scaled[1])


@pytest.mark.unit
def test_softmax_and_softmin():
    """Test softmax and softmin function."""
    arr = np.arange(-17, 17, 5)
    # expect this to not be equal to the max but rather be more
    # since softmax is a conservative estimate of the max
    sftmax = softmax(arr, alpha=1)
    assert sftmax >= np.max(arr)

    # expect this to be equal to the max
    # as alpha -> infinity, softmax -> max
    sftmax = softmax(arr, alpha=100)
    np.testing.assert_almost_equal(sftmax, np.max(arr))

    # expect this to not be equal to the min but rather be less
    # since softmin is a conservative estimate of the min
    sftmin = softmin(arr, alpha=1)
    assert sftmin <= np.min(arr)

    # expect this to be equal to the min
    # as alpha -> infinity, softmin -> min
    sftmin = softmin(arr, alpha=100)
    np.testing.assert_almost_equal(sftmin, np.min(arr))
    sftmin = softmin(arr, alpha=100)
    np.testing.assert_almost_equal(sftmin, np.min(arr))


@pytest.mark.unit
def test_loss_function_asserts():
    """Test the checks on loss function for _Objective."""
    eq = Equilibrium()
    # ensure passed-in loss_function is callable
    with pytest.raises(AssertionError):
        RotationalTransform(eq=eq, loss_function=1)
    # ensure passed-in loss_function takes only one argument
    fun = lambda x, y: x + y
    with pytest.raises(Exception):
        RotationalTransform(eq=eq, loss_function=fun)
    # ensure passed-in loss_function returns a single 0D or 1D array
    fun = lambda x: jnp.vstack((x, x))
    with pytest.raises(AssertionError):
        RotationalTransform(eq=eq, loss_function=fun)
    fun = lambda x: (x, x)
    with pytest.raises(AssertionError):
        RotationalTransform(eq=eq, loss_function=fun)


@pytest.mark.unit
@pytest.mark.slow
def test_compute_scalar_resolution():  # noqa: C901
    """Test that compute_scalar values are roughly independent of grid resolution."""
    eq = get("HELIOTRON")
    res_array = np.array([2, 2.5, 3])

    # BootstrapRedlConsistency
    # this is already covered in tests/test_bootstrap.py
    # by TestBootstrapObjectives.test_BootstrapRedlConsistency_resolution

    # CurrentDensity
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(CurrentDensity(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=3e-2)

    # Energy
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(Energy(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # ForceBalance
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(ForceBalance(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    # ForceBalanceAnisotropic
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(ForceBalanceAnisotropic(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    # HelicalForceBalance
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(HelicalForceBalance(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-1)

    # RadialForceBalance
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(RadialForceBalance(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-1)

    # GenericObjective
    # scalar
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = QuadratureGrid(
            L=int(eq.L * res), M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP
        )
        obj = ObjectiveFunction(
            GenericObjective("<beta>_vol", eq=eq, grid=grid), verbose=0
        )
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)
    # radial profile
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = LinearGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
            axis=False,
        )
        obj = ObjectiveFunction(GenericObjective("<J*B>", eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=2e-2)
    # volume quantity
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(
            GenericObjective("sqrt(g)", eq=eq, grid=grid), verbose=0
        )
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    # AspectRatio
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = QuadratureGrid(
            L=int(eq.L * res), M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP
        )
        obj = ObjectiveFunction(AspectRatio(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # BScaleLength
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = LinearGrid(M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP, sym=eq.sym)
        obj = ObjectiveFunction(BScaleLength(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # Elongation
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = QuadratureGrid(
            L=int(eq.L * res), M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP
        )
        obj = ObjectiveFunction(Elongation(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # MeanCurvature
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = LinearGrid(M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP, sym=eq.sym)
        obj = ObjectiveFunction(MeanCurvature(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # PlasmaVesselDistance
    f = np.zeros_like(res_array, dtype=float)
    surface = FourierRZToroidalSurface(
        R_lmn=[10, 1.5], Z_lmn=[-1.5], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
    )
    for i, res in enumerate(res_array):
        grid = LinearGrid(M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP)
        obj = ObjectiveFunction(
            PlasmaVesselDistance(
                surface=surface, eq=eq, surface_grid=grid, plasma_grid=grid
            ),
            verbose=0,
        )
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq, surface))
    np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    # PrincipalCurvature
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = LinearGrid(M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP, sym=eq.sym)
        obj = ObjectiveFunction(PrincipalCurvature(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # Volume
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = QuadratureGrid(
            L=int(eq.L * res), M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP
        )
        obj = ObjectiveFunction(Volume(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # RotationalTransform
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = LinearGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
            axis=False,
        )
        obj = ObjectiveFunction(RotationalTransform(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    # ToroidalCurrent
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = LinearGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
            axis=False,
        )
        obj = ObjectiveFunction(ToroidalCurrent(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=4e-2)

    # Isodynamicity
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(Isodynamicity(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    # QuasisymmetryBoozer
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate((res_array + 4) * 2):
        grid = LinearGrid(M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP)
        obj = ObjectiveFunction(
            QuasisymmetryBoozer(eq=eq, helicity=(1, -eq.NFP), grid=grid),
            verbose=0,
        )
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # QuasisymmetryTripleProduct
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(QuasisymmetryTripleProduct(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    # QuasisymmetryTwoTerm
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        grid = ConcentricGrid(
            L=int(eq.L * res),
            M=int(eq.M * res),
            N=int(eq.N * res),
            NFP=eq.NFP,
            sym=eq.sym,
        )
        obj = ObjectiveFunction(
            QuasisymmetryTwoTerm(eq=eq, helicity=(1, -eq.NFP), grid=grid),
            verbose=0,
        )
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    # MagneticWell
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        rho = np.linspace(0.2, 1, int(eq.L * res))
        grid = LinearGrid(
            rho=rho, M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP, sym=eq.sym
        )
        obj = ObjectiveFunction(MagneticWell(eq=eq, grid=grid, target=0), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    # MercierStability
    f = np.zeros_like(res_array, dtype=float)
    for i, res in enumerate(res_array):
        rho = np.linspace(0.2, 1, int(eq.L * res))
        grid = LinearGrid(
            rho=rho, M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP, sym=eq.sym
        )
        obj = ObjectiveFunction(MercierStability(eq=eq, grid=grid), verbose=0)
        obj.build(verbose=0)
        f[i] = obj.compute_scalar(obj.x(eq))
    np.testing.assert_allclose(f, f[-1], rtol=1e-2)


@pytest.mark.unit
def test_objective_no_nangrad():
    """Make sure reverse mode AD works correctly for all objectives."""
    # these need some special logic
    eq = Equilibrium(L=2, M=2, N=2)
    surf = FourierRZToroidalSurface()
    obj = ObjectiveFunction(PlasmaVesselDistance(eq, surf))
    obj.build()
    g = obj.grad(obj.x(eq, surf))
    assert not np.any(np.isnan(g)), "plasma vessel distance"

    eq = Equilibrium(L=2, M=2, N=2, anisotropy=FourierZernikeProfile())
    obj = ObjectiveFunction(ForceBalanceAnisotropic(eq))
    obj.build()
    g = obj.grad(obj.x(eq))
    assert not np.any(np.isnan(g)), "anisotropic"

    eq = Equilibrium(
        L=2,
        M=2,
        N=2,
        electron_density=PowerSeriesProfile([1e19, 0, -1e19]),
        electron_temperature=PowerSeriesProfile([1e3, 0, -1e3]),
        current=PowerSeriesProfile([1, 0, -1]),
    )
    obj = ObjectiveFunction(BootstrapRedlConsistency(eq))
    obj.build()
    g = obj.grad(obj.x(eq))
    assert not np.any(np.isnan(g)), "redl bootstrap"

    # these only need basic equilibrium
    eq = Equilibrium(L=2, M=2, N=2)
    objectives = [
        CurrentDensity,
        Energy,
        ForceBalance,
        HelicalForceBalance,
        RadialForceBalance,
        AspectRatio,
        BScaleLength,
        Elongation,
        GoodCoordinates,
        MeanCurvature,
        PrincipalCurvature,
        Volume,
        Pressure,
        RotationalTransform,
        Shear,
        ToroidalCurrent,
        Isodynamicity,
        QuasisymmetryBoozer,
        QuasisymmetryTripleProduct,
        QuasisymmetryTwoTerm,
        MagneticWell,
        MercierStability,
    ]

    for objective in objectives:
        obj = ObjectiveFunction(objective(eq))
        obj.build()
        g = obj.grad(obj.x(eq))
        assert not np.any(np.isnan(g)), str(objective)
