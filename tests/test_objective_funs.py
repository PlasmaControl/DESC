"""Tests for objective functions.

These generally don't test the accuracy of the computation for realistic examples,
that is done in test_compute_functions or regression tests.

This module primarily tests the constructing/building/calling methods.
"""

import warnings

import numpy as np
import pytest
from scipy.constants import elementary_charge, mu_0

import desc.examples
from desc.backend import jnp
from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    FourierRZCoil,
    FourierXYZCoil,
    MixedCoilSet,
)
from desc.compute import get_transforms
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import FourierPlanarCurve, FourierRZToroidalSurface, FourierXYZCurve
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid
from desc.integrals import Bounce2D
from desc.io import load
from desc.magnetic_fields import (
    CurrentPotentialField,
    FourierCurrentPotentialField,
    OmnigenousField,
    PoloidalMagneticField,
    SplineMagneticField,
    ToroidalMagneticField,
    VerticalMagneticField,
)
from desc.objectives import (
    AspectRatio,
    BallooningStability,
    BootstrapRedlConsistency,
    BoundaryError,
    BScaleLength,
    CoilArclengthVariance,
    CoilCurrentLength,
    CoilCurvature,
    CoilIntegratedCurvature,
    CoilLength,
    CoilSetLinkingNumber,
    CoilSetMinDistance,
    CoilTorsion,
    EffectiveRipple,
    Elongation,
    Energy,
    ForceBalance,
    ForceBalanceAnisotropic,
    FusionPower,
    GammaC,
    GenericObjective,
    HeatingPowerISS04,
    Isodynamicity,
    LinearObjectiveFromUser,
    LinkingCurrentConsistency,
    MagneticWell,
    MeanCurvature,
    MercierStability,
    MirrorRatio,
    ObjectiveFromUser,
    ObjectiveFunction,
    Omnigenity,
    PlasmaCoilSetMinDistance,
    PlasmaVesselDistance,
    Pressure,
    PrincipalCurvature,
    QuadraticFlux,
    QuasisymmetryBoozer,
    QuasisymmetryTripleProduct,
    QuasisymmetryTwoTerm,
    RotationalTransform,
    Shear,
    SurfaceCurrentRegularization,
    SurfaceQuadraticFlux,
    ToroidalCurrent,
    ToroidalFlux,
    VacuumBoundaryError,
    Volume,
)
from desc.objectives._free_boundary import BoundaryErrorNESTOR
from desc.objectives.normalization import compute_scaling_factors
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.objectives.utils import softmax, softmin
from desc.profiles import FourierZernikeProfile, PowerSeriesProfile
from desc.utils import PRINT_WIDTH
from desc.vmec_utils import ptolemy_linear_transform


class TestObjectiveFunction:
    """Test ObjectiveFunction classes."""

    @pytest.mark.unit
    def test_generic(self):
        """Test GenericObjective for arbitrary Equilibrium quantities."""

        def test(f, thing, grid=None, compress=False):
            obj = GenericObjective(f, thing=thing, grid=grid)
            obj.build()
            val = thing.compute(f, grid=obj.constants["transforms"]["grid"])[f]
            if compress:
                val = obj.constants["transforms"]["grid"].compress(val)
            np.testing.assert_allclose(
                obj.compute(thing.params_dict),
                val,
            )

        test("curvature", FourierXYZCurve(Y_n=[-1, 0, 0]), LinearGrid(0, 0, 12))
        test("length", FourierPlanarCoil(r_n=0.5), LinearGrid(0, 0, 12))
        test(
            "Phi",
            FourierCurrentPotentialField(Phi_mn=np.array([0.2])),
            LinearGrid(0, 4, 4),
        )
        test("sqrt(g)", Equilibrium())
        test("current", Equilibrium(iota=PowerSeriesProfile(0)), None, True)
        test("iota", Equilibrium(current=PowerSeriesProfile(0)), None, True)

    @pytest.mark.unit
    def test_objective_from_user(self):
        """Test ObjectiveFromUser for arbitrary callable."""

        def myfun(grid, data):
            x = data["X"]
            y = data["Y"]
            r = jnp.sqrt(x * data["X"] + y**2)
            return r

        def test(thing, grid):
            objective = ObjectiveFromUser(myfun, thing=thing, grid=grid)
            objective.build()
            R1 = objective.compute(*objective.xs(thing))
            R2 = thing.compute("R", grid=grid)["R"]
            np.testing.assert_allclose(R1, R2)

        curve = FourierXYZCurve()
        grid = LinearGrid(0, 0, 5)
        test(curve, grid)

        surf = FourierRZToroidalSurface()
        grid = LinearGrid(2, 2, 2)
        test(surf, grid)

        eq = Equilibrium()
        grid = LinearGrid(2, 2, 2)
        test(eq, grid)

    @pytest.mark.unit
    def test_linear_objective_from_user(self):
        """Test LinearObjectiveFromUser for arbitrary callable."""

        def myfun(params):
            L_lmn = params["L_lmn"]
            p_l = params["p_l"]
            c_l = params["c_l"]
            Zb_lmn = params["Zb_lmn"]
            r = jnp.array([p_l[0] + Zb_lmn[0], c_l[2] + L_lmn[1]])
            return r

        eq = Equilibrium(pressure=np.array([1, 0, -1]), current=np.array([0, 0, 2]))
        objective = LinearObjectiveFromUser(myfun, eq)
        objective.build()
        f = objective.compute(*objective.xs(eq))
        np.testing.assert_allclose(f, np.array([0, 2]))

    @pytest.mark.unit
    def test_volume(self):
        """Test calculation of plasma volume."""

        def test(eq):
            obj = Volume(
                target=10 * np.pi**2,
                weight=1 / np.pi**2,
                eq=eq,
                normalize=False,
            )
            obj.build()
            V = obj.compute_unscaled(*obj.xs(eq))
            V_scaled = obj.compute_scaled_error(*obj.xs(eq))
            V_scalar = obj.compute_scalar(*obj.xs(eq))
            np.testing.assert_allclose(V, 20 * np.pi**2)
            np.testing.assert_allclose(V_scaled, 10)
            np.testing.assert_allclose(V_scalar, 10)

        eqi = Equilibrium(iota=PowerSeriesProfile(0))
        test(eqi)
        test(Equilibrium(current=PowerSeriesProfile(0)))
        # test that it can compute with a surface object
        test(eqi.surface)

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
        test(Equilibrium(iota=PowerSeriesProfile(0)).surface)

    @pytest.mark.unit
    def test_elongation(self):
        """Test calculation of elongation."""

        def test(eq):
            obj = Elongation(target=0, weight=2, eq=eq)
            obj.build()
            f = obj.compute_unscaled(*obj.xs(eq))
            f_scaled = obj.compute_scaled_error(*obj.xs(eq))
            np.testing.assert_allclose(f, 1.3 / 0.7, rtol=8e-3)
            np.testing.assert_allclose(f_scaled, 2 * (1.3 / 0.7), rtol=8e-3)

        test(get("HELIOTRON"))
        test(get("HELIOTRON").surface)

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
            obj.compile("all")
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
        np.testing.assert_allclose(f[idx_f][:120], B_mn[idx_B][:120], rtol=1e-6)

    @pytest.mark.unit
    def test_qh_boozer_multiple_surfaces(self):
        """Test for computing Boozer error on multiple surfaces."""
        eq = get("WISTELL-A")  # WISTELL-A is optimized for QH symmetry
        helicity = (1, -eq.NFP)
        M_booz = eq.M
        N_booz = eq.N
        grid1 = LinearGrid(rho=0.5, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP, sym=False)
        grid2 = LinearGrid(rho=1.0, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP, sym=False)
        grid3 = LinearGrid(
            rho=np.array([0.5, 1.0]), M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP, sym=False
        )

        obj1 = QuasisymmetryBoozer(
            helicity=helicity,
            M_booz=M_booz,
            N_booz=N_booz,
            grid=grid1,
            normalize=False,
            eq=eq,
        )
        obj2 = QuasisymmetryBoozer(
            helicity=helicity,
            M_booz=M_booz,
            N_booz=N_booz,
            grid=grid2,
            normalize=False,
            eq=eq,
        )
        obj3 = QuasisymmetryBoozer(
            helicity=helicity,
            M_booz=M_booz,
            N_booz=N_booz,
            grid=grid3,
            normalize=False,
            eq=eq,
        )
        obj1.build()
        obj2.build()
        obj3.build()
        f1 = obj1.compute_unscaled(*obj1.xs(eq))
        f2 = obj2.compute_unscaled(*obj2.xs(eq))
        f3 = obj3.compute_unscaled(*obj3.xs(eq))
        np.testing.assert_allclose(f3, np.concatenate([f1, f2]), atol=1e-14)

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
        with pytest.raises(ValueError):
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
    def test_boundary_error_biestsc(self):
        """Test calculation of boundary error using BIEST w/ sheet current."""
        coil = FourierXYZCoil(5e5)
        coilset = CoilSet.linspaced_angular(coil, n=100, check_intersection=False)
        coil_grid = LinearGrid(N=20)
        eq = Equilibrium(L=3, M=3, N=3, Psi=np.pi)
        eq.surface = FourierCurrentPotentialField.from_surface(
            eq.surface, M_Phi=eq.M, N_Phi=eq.N
        )
        eq.solve()
        obj = BoundaryError(eq, coilset, field_grid=coil_grid)
        obj.build()
        f = obj.compute_scaled_error(*obj.xs())
        n = len(f) // 3
        # first n should be B*n errors
        np.testing.assert_allclose(f[:n], 0, atol=1e-4)
        # next n should be B^2 errors
        np.testing.assert_allclose(f[n : 2 * n], 0, atol=5e-2)
        # last n should be K errors
        np.testing.assert_allclose(f[2 * n :], 0, atol=3e-2)

    @pytest.mark.unit
    def test_boundary_error_biest(self):
        """Test calculation of boundary error using BIEST."""
        coil = FourierXYZCoil(5e5)
        coilset = CoilSet.linspaced_angular(coil, n=100, check_intersection=False)
        coil_grid = LinearGrid(N=20)
        eq = Equilibrium(L=3, M=3, N=3, Psi=np.pi)
        eq.solve()
        obj = BoundaryError(eq, coilset, field_grid=coil_grid)
        obj.build()
        f = obj.compute_scaled_error(*obj.xs())
        n = len(f) // 2
        # first n should be B*n errors
        np.testing.assert_allclose(f[:n], 0, atol=1e-4)
        # next n should be B^2 errors
        np.testing.assert_allclose(f[n : 2 * n], 0, atol=5e-2)

    @pytest.mark.unit
    def test_boundary_error_vacuum(self):
        """Test calculation of vacuum boundary error."""
        coil = FourierXYZCoil(5e5)
        coilset = CoilSet.linspaced_angular(coil, n=100, check_intersection=False)
        coil_grid = LinearGrid(N=20)
        eq = Equilibrium(L=3, M=3, N=3, Psi=np.pi)
        eq.solve()
        obj = VacuumBoundaryError(eq, coilset, field_grid=coil_grid)
        obj.build()
        f = obj.compute_scaled_error(*obj.xs())
        n = len(f) // 2
        # first n should be B*n errors
        np.testing.assert_allclose(f[:n], 0, atol=1e-4)
        # next n should be B^2 errors
        np.testing.assert_allclose(f[n : 2 * n], 0, atol=4e-2)

    @pytest.mark.unit
    def test_boundary_error_nestor(self):
        """Test calculation of boundary error using NESTOR."""
        coil = FourierXYZCoil(5e5)
        coilset = CoilSet.linspaced_angular(coil, n=100, check_intersection=False)
        coil_grid = LinearGrid(N=20)
        eq = Equilibrium(L=3, M=3, N=3, Psi=np.pi)
        eq.solve()
        obj = BoundaryErrorNESTOR(eq, coilset, field_grid=coil_grid)
        obj.build()
        f = obj.compute_scaled_error(*obj.xs())
        np.testing.assert_allclose(f, 0, atol=2e-3)

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
    def test_plasma_vessel_distance(self):
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
        d = obj.compute_unscaled(*obj.xs(eq))
        assert d.size == obj.dim_f
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
        obj = PlasmaVesselDistance(
            surface=surf, surface_grid=grid, plasma_grid=grid, eq=eq
        )
        with pytest.raises(UserWarning):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                obj.build()

        # test softmin, should give approximate value
        surf_grid = LinearGrid(M=5, N=6)
        plas_grid = LinearGrid(M=5, N=6)
        obj = PlasmaVesselDistance(
            eq=eq,
            plasma_grid=plas_grid,
            surface_grid=surf_grid,
            surface=surface,
            use_softmin=True,
            softmin_alpha=5,
        )
        obj.build()
        d = obj.compute_unscaled(*obj.xs(eq, surface))
        assert d.size == obj.dim_f
        np.testing.assert_allclose(np.abs(d).min(), a_s - a_p, rtol=1.5e-1)

        # for large enough alpha, should be same as actual min
        obj = PlasmaVesselDistance(
            eq=eq,
            plasma_grid=plas_grid,
            surface_grid=surf_grid,
            surface=surface,
            use_softmin=True,
            softmin_alpha=100,
        )
        obj.build()
        d = obj.compute_unscaled(*obj.xs(eq, surface))
        np.testing.assert_allclose(d, a_s - a_p)

    @pytest.mark.unit
    def test_mean_curvature(self):
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

        # check using the surface
        obj = MeanCurvature(eq=eq.surface)
        obj.build()
        H = obj.compute_unscaled(*obj.xs(eq.surface))
        assert np.any(H > 0)

    @pytest.mark.unit
    def test_principal_curvature(self):
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

        # same test but using the surface directly
        obj1 = PrincipalCurvature(eq=eq1.surface, normalize=False)
        obj1.build()
        K1 = obj1.compute_unscaled(*obj1.xs(eq1.surface))
        obj2 = PrincipalCurvature(eq=eq2.surface, normalize=False)
        obj2.build()
        K2 = obj2.compute_unscaled(*obj2.xs(eq2.surface))

        # simple test: NCSX should have higher mean absolute curvature than DSHAPE
        assert K1.mean() < K2.mean()

    @pytest.mark.unit
    def test_field_scale_length(self):
        """Test for B field scale length objective function."""
        surf1 = FourierRZToroidalSurface(
            R_lmn=[5, 1], Z_lmn=[-1], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]], NFP=1
        )
        surf2 = FourierRZToroidalSurface(
            R_lmn=[10, 2],
            Z_lmn=[-2],
            modes_R=[[0, 0], [1, 0]],
            modes_Z=[[-1, 0]],
            NFP=1,
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
    def test_coil_length(self):
        """Tests coil length."""

        def test(coil, grid=None):
            obj = CoilLength(coil, grid=grid)
            obj.build()
            f = obj.compute(params=coil.params_dict)
            np.testing.assert_allclose(f, 4 * np.pi, rtol=1e-8)
            assert len(f) == obj.dim_f

        coil = FourierPlanarCoil(r_n=2, basis="rpz")
        coils = CoilSet.linspaced_linear(
            coil, n=3, displacement=[0, 3, 0], check_intersection=False
        )
        mixed_coils = MixedCoilSet.linspaced_linear(
            coil, n=2, displacement=[0, 7, 0], check_intersection=False
        )
        nested_coils = MixedCoilSet(coils, mixed_coils, check_intersection=False)

        grid = None  # default grid

        test(coil)
        test(coils)
        test(mixed_coils)
        test(nested_coils, grid=grid)

    @pytest.mark.unit
    def test_coil_current_length(self):
        """Tests coil current length."""

        def test(coil, grid=None):
            obj = CoilCurrentLength(coil, grid=grid)
            obj.build()
            f = obj.compute(params=coil.params_dict)
            np.testing.assert_allclose(f, 4 * np.pi, rtol=1e-8)
            assert f.shape == (obj.dim_f,)

        coil = FourierPlanarCoil(r_n=2, basis="rpz")
        coils = CoilSet.linspaced_linear(
            coil, n=3, displacement=[0, 3, 0], check_intersection=False
        )
        mixed_coils = MixedCoilSet.linspaced_linear(
            coil, n=2, displacement=[0, 7, 0], check_intersection=False
        )
        nested_coils = MixedCoilSet(coils, mixed_coils, check_intersection=False)

        grid = LinearGrid(N=5)  # single grid

        test(coil)
        test(coils)
        test(mixed_coils)
        test(nested_coils, grid=grid)

    @pytest.mark.unit
    def test_coil_curvature(self):
        """Tests coil curvature."""

        def test(coil, grid=None):
            obj = CoilCurvature(coil, grid=grid)
            obj.build()
            f = obj.compute(params=coil.params_dict)
            np.testing.assert_allclose(f, 1 / 2, rtol=1e-8)
            assert len(f) == obj.dim_f

        coil = FourierPlanarCoil(r_n=2, basis="rpz")
        coils = CoilSet.linspaced_linear(
            coil, n=3, displacement=[0, 3, 0], check_intersection=False
        )
        mixed_coils = MixedCoilSet.linspaced_linear(
            coil, n=2, displacement=[0, 7, 0], check_intersection=False
        )
        nested_coils = MixedCoilSet(coils, mixed_coils, check_intersection=False)

        grid = [LinearGrid(N=5)] * 5  # single list of grids

        test(coil)
        test(coils)
        test(mixed_coils)
        test(nested_coils, grid=grid)

    @pytest.mark.unit
    def test_coil_torsion(self):
        """Tests coil torsion."""

        def test(coil, grid=None):
            obj = CoilTorsion(coil, grid=grid)
            obj.build()
            f = obj.compute(params=coil.params_dict)
            np.testing.assert_allclose(f, 0, atol=1e-8)
            assert len(f) == obj.dim_f

        coil = FourierPlanarCoil(r_n=2, basis="rpz")
        coils = CoilSet.linspaced_linear(
            coil, n=3, displacement=[0, 3, 0], check_intersection=False
        )
        mixed_coils = MixedCoilSet.linspaced_linear(
            coil, n=2, displacement=[0, 7, 0], check_intersection=False
        )
        nested_coils = MixedCoilSet(coils, mixed_coils, check_intersection=False)

        grid = [[LinearGrid(N=5)] * 3, [LinearGrid(N=5)] * 2]  # nested list of grids

        test(coil)
        test(coils)
        test(mixed_coils)
        test(nested_coils, grid=grid)

    @pytest.mark.unit
    def test_integrated_curvature(self):
        """Tests integrated_curvature."""

        def test(coil, grid=None, ans=2 * np.pi):
            obj = CoilIntegratedCurvature(coil, grid=grid)
            obj.build()
            f = obj.compute(params=coil.params_dict)
            np.testing.assert_allclose(f, ans, atol=2e-5)
            assert f.shape == (obj.dim_f,)

        # convex coils
        coil = FourierPlanarCoil(r_n=[0.3, 1, 0.3], basis="rpz")
        coils = CoilSet.linspaced_linear(
            coil, n=3, displacement=[0, 3, 0], check_intersection=False
        )
        mixed_coils = MixedCoilSet.linspaced_linear(
            coil, n=2, displacement=[0, 7, 0], check_intersection=False
        )
        nested_coils = MixedCoilSet(coils, mixed_coils, check_intersection=False)
        test(coil)
        test(coils)
        test(mixed_coils)
        test(nested_coils)

        # not convex coils
        coil = FourierPlanarCoil(r_n=[0.5, 1, 0.5], basis="rpz")
        coils = CoilSet.linspaced_linear(
            coil, n=3, displacement=[0, 3, 0], check_intersection=False
        )
        mixed_coils = MixedCoilSet.linspaced_linear(
            coil, n=2, displacement=[0, 7, 0], check_intersection=False
        )
        nested_coils = MixedCoilSet(coils, mixed_coils, check_intersection=False)
        ans = 1.104044 + 2 * np.pi
        test(coil, ans=ans)
        test(coils, ans=ans)
        test(mixed_coils, ans=ans)
        test(nested_coils, ans=ans)

    @pytest.mark.unit
    def test_coil_type_error(self):
        """Tests error when objective is not passed a coil."""
        curve = FourierPlanarCurve(r_n=2, basis="rpz")
        obj = CoilLength(curve)
        with pytest.raises(TypeError):
            obj.build()

    @pytest.mark.unit
    def test_coil_min_distance(self):
        """Tests minimum distance between coils in a coilset."""

        def test(coils, mindist, grid=None, expect_intersect=False, tol=None):
            obj = CoilSetMinDistance(coils, grid=grid)
            obj.build()
            f = obj.compute(params=coils.params_dict)
            assert f.size == coils.num_coils
            np.testing.assert_allclose(f, mindist)
            assert coils.is_self_intersecting(grid=grid, tol=tol) == expect_intersect
            obj2 = CoilSetMinDistance(
                coils, grid=grid, use_softmin=True, softmin_alpha=10
            )
            obj2.build()
            f = obj2.compute(params=coils.params_dict)
            assert f.size == coils.num_coils
            np.testing.assert_allclose(f, mindist, rtol=5e-2, atol=1e-3)

        # linearly spaced planar coils, all coils are min distance from their neighbors
        n = 3
        disp = 5
        coil = FourierPlanarCoil(r_n=1, normal=[0, 0, 1])
        coils_linear = CoilSet.linspaced_linear(
            coil, n=n, displacement=[0, 0, disp], check_intersection=False
        )
        test(coils_linear, disp / n)

        # planar toroidal coils, without symmetry
        # min points are at the inboard midplane and are corners of a square inscribed
        # in a circle of radius = center - r
        center = 3
        r = 1
        coil = FourierPlanarCoil(center=[center, 0, 0], normal=[0, 1, 0], r_n=r)
        coils_angular = CoilSet.linspaced_angular(coil, n=4, check_intersection=False)
        test(
            coils_angular, np.sqrt(2) * (center - r), grid=LinearGrid(zeta=4), tol=1e-5
        )

        # planar toroidal coils, with symmetry
        # min points are at the inboard midplane and are corners of an octagon inscribed
        # in a circle of radius = center - r
        center = 3
        r = 1
        coil = FourierPlanarCoil(center=[center, 0, 0], normal=[0, 1, 0], r_n=r)
        coils = CoilSet.linspaced_angular(
            coil, angle=np.pi / 2, n=5, endpoint=True, check_intersection=False
        )
        coils_sym = CoilSet(coils[1::2], NFP=2, sym=True)
        test(coils_sym, 2 * (center - r) * np.sin(np.pi / 8), grid=LinearGrid(zeta=4))

        # mixture of toroidal field coilset, vertical field coilset, and extra coil
        # TF coils instersect with the middle VF coil
        # extra coil is 5 m from middle VF coil
        tf_coil = FourierPlanarCoil(center=[2, 0, 0], normal=[0, 1, 0], r_n=1)
        tf_coilset = CoilSet.linspaced_angular(tf_coil, n=4, check_intersection=False)
        vf_coil = FourierRZCoil(R_n=3, Z_n=-1)
        vf_coilset = CoilSet.linspaced_linear(
            vf_coil,
            displacement=[0, 0, 2],
            n=3,
            endpoint=True,
            check_intersection=False,
        )
        xyz_coil = FourierXYZCoil(X_n=[0, 6, 1], Y_n=[0, 0, 0], Z_n=[-1, 0, 0])
        coils_mixed = MixedCoilSet(
            (tf_coilset, vf_coilset, xyz_coil), check_intersection=False
        )
        with pytest.warns(UserWarning, match="nearly intersecting"):
            test(
                coils_mixed,
                [0, 0, 0, 0, 1, 0, 1, 2],
                grid=LinearGrid(zeta=4),
                expect_intersect=True,
            )
        # TODO (#1400, 914): move this coil set to conftest?

    @pytest.mark.unit
    def test_plasma_coil_min_distance(self):
        """Tests minimum distance between plasma and a coilset."""

        def test(
            eq,
            coils,
            mindist,
            plasma_grid=None,
            coil_grid=None,
            eq_fixed=False,
            coils_fixed=False,
        ):
            obj = PlasmaCoilSetMinDistance(
                eq=eq,
                coil=coils,
                plasma_grid=plasma_grid,
                coil_grid=coil_grid,
                eq_fixed=eq_fixed,
                coils_fixed=coils_fixed,
            )
            obj.build()
            if eq_fixed:
                f = obj.compute(params_1=coils.params_dict)
            elif coils_fixed:
                f = obj.compute(params_1=eq.params_dict)
            else:
                f = obj.compute(params_1=eq.params_dict, params_2=coils.params_dict)
            assert f.size == coils.num_coils
            np.testing.assert_allclose(f, mindist)
            obj2 = PlasmaCoilSetMinDistance(
                eq=eq,
                coil=coils,
                plasma_grid=plasma_grid,
                coil_grid=coil_grid,
                eq_fixed=eq_fixed,
                coils_fixed=coils_fixed,
                use_softmin=True,
                softmin_alpha=40,
            )
            obj2.build()
            if eq_fixed:
                f = obj2.compute(params_1=coils.params_dict)
            elif coils_fixed:
                f = obj2.compute(params_1=eq.params_dict)
            else:
                f = obj2.compute(params_1=eq.params_dict, params_2=coils.params_dict)
            assert f.size == coils.num_coils
            np.testing.assert_allclose(f, mindist, rtol=5e-2, atol=1e-3)

        plasma_grid = LinearGrid(M=4, zeta=16)
        coil_grid = LinearGrid(N=8)

        # planar toroidal coils without symmetry, around fixed circular tokamak
        R0 = 3
        a = 1
        offset = 0.5
        surf = FourierRZToroidalSurface(
            R_lmn=np.array([R0, a]),
            Z_lmn=np.array([0, -a]),
            modes_R=np.array([[0, 0], [1, 0]]),
            modes_Z=np.array([[0, 0], [-1, 0]]),
        )
        eq = Equilibrium(surface=surf, NFP=1, M=2, N=0, sym=True)
        coil = FourierPlanarCoil(center=[R0, 0, 0], normal=[0, 1, 0], r_n=[a + offset])
        coils = CoilSet.linspaced_angular(coil, n=8, check_intersection=False)
        test(
            eq,
            coils,
            offset,
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=True,
        )
        test(
            eq.surface,
            coils,
            offset,
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=True,
        )

        # planar toroidal coils with symmetry, around unfixed circular tokamak
        R0 = 5
        a = 1.5
        offset = 0.75
        surf = FourierRZToroidalSurface(
            R_lmn=np.array([R0, a]),
            Z_lmn=np.array([0, -a]),
            modes_R=np.array([[0, 0], [1, 0]]),
            modes_Z=np.array([[0, 0], [-1, 0]]),
        )
        eq = Equilibrium(surface=surf, NFP=1, M=2, N=0, sym=True)
        coil = FourierPlanarCoil(center=[R0, 0, 0], normal=[0, 1, 0], r_n=[a + offset])
        coils = CoilSet.linspaced_angular(
            coil, angle=np.pi / 2, n=5, endpoint=True, check_intersection=False
        )
        coils = CoilSet(coils[1::2], NFP=2, sym=True)
        test(
            eq,
            coils,
            offset,
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=False,
        )
        test(
            eq.surface,
            coils,
            offset,
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=False,
        )

        # fixed planar toroidal coils with symmetry, around unfixed circular tokamak
        R0 = 5
        a = 1.5
        offset = 0.75
        surf = FourierRZToroidalSurface(
            R_lmn=np.array([R0, a]),
            Z_lmn=np.array([0, -a]),
            modes_R=np.array([[0, 0], [1, 0]]),
            modes_Z=np.array([[0, 0], [-1, 0]]),
        )
        eq = Equilibrium(surface=surf, NFP=1, M=2, N=0, sym=True)
        coil = FourierPlanarCoil(center=[R0, 0, 0], normal=[0, 1, 0], r_n=[a + offset])
        coils = CoilSet.linspaced_angular(
            coil, angle=np.pi / 2, n=5, endpoint=True, check_intersection=False
        )
        coils = CoilSet(coils[1::2], NFP=2, sym=True)
        test(
            eq,
            coils,
            offset,
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=False,
            coils_fixed=True,
        )

    @pytest.mark.unit
    def test_quadratic_flux(self):
        """Test calculation of quadratic flux on the boundary."""
        t_field = ToroidalMagneticField(1, 1)

        # test that torus (axisymmetric) Bnorm is exactly 0
        eq = load("./tests/inputs/vacuum_circular_tokamak.h5")
        obj = QuadraticFlux(eq, t_field)
        obj.build(eq, verbose=2)
        f = obj.compute(field_params=t_field.params_dict)
        np.testing.assert_allclose(f, 0, rtol=1e-14, atol=1e-14)

        # test non-axisymmetric surface
        eq = desc.examples.get("precise_QA", "all")[0]
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(4, 4, 4, 8, 8, 8)
        eval_grid = LinearGrid(
            rho=np.array([1.0]),
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=False,
        )
        source_grid = LinearGrid(
            rho=np.array([1.0]),
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=False,
        )

        obj = QuadraticFlux(eq, t_field, eval_grid=eval_grid, source_grid=source_grid)
        Bnorm = t_field.compute_Bnormal(
            eq, eval_grid=eval_grid, source_grid=source_grid
        )[0]
        obj.build(eq)
        dA = eq.compute("|e_theta x e_zeta|", grid=eval_grid)["|e_theta x e_zeta|"]
        f = obj.compute_unscaled(t_field.params_dict)

        np.testing.assert_allclose(f, Bnorm * np.sqrt(dA), atol=2e-4, rtol=1e-2)

        # equilibrium that has B_plasma == 0
        eq = load("./tests/inputs/vacuum_nonaxisym.h5")

        eval_grid = LinearGrid(
            rho=np.array([1.0]),
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP,
            sym=False,
        )
        obj = QuadraticFlux(eq, t_field, vacuum=True, eval_grid=eval_grid)
        Bnorm = t_field.compute_Bnormal(eq.surface, eval_grid=eval_grid)[0]
        obj.build(eq)
        f = obj.compute(field_params=t_field.params_dict)
        dA = eq.compute("|e_theta x e_zeta|", grid=eval_grid)["|e_theta x e_zeta|"]
        # check that they're the same since we set B_plasma = 0
        np.testing.assert_allclose(f, Bnorm * np.sqrt(dA), atol=1e-14)

    @pytest.mark.unit
    def test_quadratic_flux_minimizing_surface(self):
        """Test calculation of quadratic flux on a surface."""
        t_field = ToroidalMagneticField(1, 1)

        # test that torus (axisymmetric) Bnorm is exactly 0
        eq = load("./tests/inputs/vacuum_circular_tokamak.h5")
        surf = eq.surface
        obj = SurfaceQuadraticFlux(surf, t_field)
        obj.build(eq, verbose=2)
        f = obj.compute(params_1=surf.params_dict, params_2=t_field.params_dict)
        np.testing.assert_allclose(f, 0, rtol=1e-14, atol=1e-14)

        # test non-axisymmetric surface
        eq = desc.examples.get("precise_QA", "all")[0]
        surf = eq.surface
        surf.change_resolution(4, 4)
        eval_grid = LinearGrid(
            rho=np.array([1.0]),
            M=surf.M * 2,
            N=surf.N * 2,
            NFP=eq.NFP,
            sym=False,
        )

        obj = SurfaceQuadraticFlux(surf, t_field, eval_grid=eval_grid, field_fixed=True)
        Bnorm = t_field.compute_Bnormal(
            eq.surface, eval_grid=eval_grid, source_grid=eval_grid
        )[0]
        obj.build(surf)
        dA = surf.compute("|e_theta x e_zeta|", grid=eval_grid)["|e_theta x e_zeta|"]
        f = obj.compute(params_1=surf.params_dict)

        np.testing.assert_allclose(f, Bnorm * np.sqrt(dA), atol=2e-4, rtol=1e-2)

    @pytest.mark.unit
    def test_toroidal_flux(self):
        """Test calculation of toroidal flux from coils."""
        grid1 = LinearGrid(L=0, M=40, zeta=np.array(0.0))

        def test(
            eq,
            field,
            correct_value,
            rtol=1e-14,
            grid=None,
            eq_fixed=True,
            field_fixed=True,
        ):
            obj = ToroidalFlux(
                eq=eq,
                field=field,
                eval_grid=grid,
                eq_fixed=eq_fixed,
                field_fixed=field_fixed,
            )
            obj.build(verbose=2)
            if eq_fixed:
                torflux = obj.compute_unscaled(*obj.xs(field))
            elif field_fixed:
                torflux = obj.compute_unscaled(*obj.xs(eq))
            else:
                torflux = obj.compute_unscaled(*obj.xs(eq, field))
            np.testing.assert_allclose(torflux, correct_value, rtol=rtol)

        eq = Equilibrium(iota=PowerSeriesProfile(0))
        test(eq, VerticalMagneticField(B0=1), 0, grid=grid1, field_fixed=False)
        test(eq, VerticalMagneticField(B0=1), 0, grid=grid1, eq_fixed=False)
        test(
            eq,
            VerticalMagneticField(B0=1),
            0,
            grid=grid1,
            field_fixed=False,
            eq_fixed=False,
        )

        field = ToroidalMagneticField(B0=1, R0=1)
        # calc field Psi

        data = eq.compute(["R", "phi", "Z", "e_theta"], grid=grid1)
        field_A = field.compute_magnetic_vector_potential(
            np.vstack([data["R"], data["phi"], data["Z"]]).T
        )

        A_dot_e_theta = jnp.sum(field_A * data["e_theta"], axis=1)

        psi_from_field = np.sum(grid1.spacing[:, 1] * A_dot_e_theta)
        eq.change_resolution(L_grid=20, M_grid=20)

        test(eq, field, psi_from_field, field_fixed=False)
        test(eq, field, psi_from_field, rtol=1e-3, field_fixed=False)

        with pytest.raises(ValueError, match="Cannot have"):
            ToroidalFlux(eq, field, eq_fixed=True, field_fixed=True)

        # test on field with no vector potential
        pfield = PoloidalMagneticField(1, 1, 1)
        test(eq, pfield, 0.0, field_fixed=False)
        test(eq, pfield, 0.0, eq_fixed=False)
        test(eq, pfield, 0.0, eq_fixed=False, field_fixed=False)

        with pytest.raises(ValueError, match="vector potential"):
            obj = ToroidalFlux(eq.surface, pfield)
            obj.build()

    @pytest.mark.unit
    def test_coil_linking_number(self):
        """Test for linking number objective."""
        coil = FourierPlanarCoil(center=[10, 1, 0])
        # regular modular coilset from symmetry, so that there are 10 coils, half going
        # one way and half going the other way
        coilset = CoilSet.from_symmetry(coil, NFP=5, sym=True, check_intersection=False)
        coil2 = FourierRZCoil()
        # add a coil along the axis that links all the other coils
        coilset2 = MixedCoilSet(coilset, coil2, check_intersection=False)

        obj = CoilSetLinkingNumber(coilset2)
        obj.build()
        out = obj.compute_scaled_error(coilset2.params_dict)
        # the modular coils all link 1 other coil (the axis)
        # while the axis links all 10 modular coils
        expected = np.array([1] * 10 + [10])
        np.testing.assert_allclose(out, expected, rtol=1e-3)

    @pytest.mark.unit
    def test_signed_plasma_vessel_distance(self):
        """Test calculation of signed distance from plasma to vessel."""
        R0 = 10.0
        a_p = 1.0
        a_s = 2.0
        # default eq has R0=10, a=1
        eq = Equilibrium(M=3, N=2)
        # surface with same R0, a=2, so true d=1 for all pts
        surface = FourierRZToroidalSurface(
            R_lmn=[R0, a_s], Z_lmn=[-a_s], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
        )
        grid = LinearGrid(M=5, N=6)
        obj = PlasmaVesselDistance(
            eq=eq,
            surface_grid=grid,
            plasma_grid=grid,
            surface=surface,
            use_signed_distance=True,
        )
        obj.build()
        d = obj.compute_unscaled(*obj.xs(eq, surface))
        assert obj.dim_f == d.size
        np.testing.assert_allclose(d, a_s - a_p)

        # ensure that it works (dimension-wise) when compute_scaled is called
        _ = obj.compute_scaled(*obj.xs(eq, surface))

        # For plasma outside surface, should get signed distance
        a_s = 0.5 * a_p
        surface = FourierRZToroidalSurface(
            R_lmn=[R0, a_s],
            Z_lmn=[-a_s],
            modes_R=[[0, 0], [1, 0]],
            modes_Z=[[-1, 0]],
        )
        grid = LinearGrid(M=5, N=6)
        obj = PlasmaVesselDistance(
            eq=eq,
            surface_grid=grid,
            plasma_grid=grid,
            surface=surface,
            use_signed_distance=True,
        )
        obj.build()
        d = obj.compute_unscaled(*obj.xs(eq, surface))
        assert obj.dim_f == d.size
        np.testing.assert_allclose(d, a_s - a_p)

        # ensure it works with different sized grids (poloidal resolution different)
        grid = LinearGrid(M=5, N=6)
        obj = PlasmaVesselDistance(
            eq=eq,
            surface_grid=grid,
            plasma_grid=LinearGrid(M=10, N=6),
            surface=surface,
            use_signed_distance=True,
        )
        obj.build()
        d = obj.compute_unscaled(*obj.xs(eq, surface))
        assert obj.dim_f == d.size
        assert abs(d.max() - (-a_s)) < 1e-14
        assert abs(d.min() - (-a_s)) < grid.spacing[0, 1] * a_s

        # ensure it works with different sized grids (poloidal resolution different)
        # and using softmin (with deprecated name alpha)
        grid = LinearGrid(M=5, N=6)
        with pytest.raises(FutureWarning):
            obj = PlasmaVesselDistance(
                eq=eq,
                surface_grid=grid,
                plasma_grid=LinearGrid(M=10, N=6),
                surface=surface,
                use_signed_distance=True,
                use_softmin=True,
                alpha=4000,
            )
        obj.build()
        d = obj.compute_unscaled(*obj.xs(eq, surface))
        assert obj.dim_f == d.size
        assert abs(d.max() - (-a_s)) < 1e-14
        assert abs(d.min() - (-a_s)) < grid.spacing[0, 1] * a_s
        # test errors
        # differing grid zetas, same num_zeta
        with pytest.raises(ValueError):
            obj = PlasmaVesselDistance(
                eq=eq,
                surface_grid=grid,
                plasma_grid=LinearGrid(M=grid.M, N=grid.N, NFP=2),
                surface=surface,
                use_signed_distance=True,
            )
            obj.build()
        # test with differing grid.num_zeta
        with pytest.raises(ValueError):
            obj = PlasmaVesselDistance(
                eq=eq,
                surface_grid=grid,
                plasma_grid=LinearGrid(M=grid.M, N=grid.N - 2),
                surface=surface,
                use_signed_distance=True,
            )
            obj.build()

    @pytest.mark.unit
    def test_mirror_ratio_equilibrium(self):
        """Test mirror ratio objective for Equilibrium."""
        # axisymmetry, no iota, so B ~ B0/R
        eq = Equilibrium(L=8, M=8)
        eq.solve()
        # R0 = 10, a=1, so Bmax = B0/9, Bmin = B0/11
        mirror_ratio = (1 / 9 - 1 / 11) / (1 / 9 + 1 / 11)
        obj = MirrorRatio(eq)
        obj.build()
        f = obj.compute(eq.params_dict)
        # not perfect agreement bc eq is low res, so B isnt exactly B0/R
        np.testing.assert_allclose(f, mirror_ratio, rtol=3e-3)

    @pytest.mark.unit
    def test_mirror_ratio_omni_field(self):
        """Test mirror ratio objective for OmnigenousField."""
        field = OmnigenousField(
            L_B=1,
            M_B=3,
            L_x=1,
            M_x=1,
            N_x=1,
            NFP=1,
            helicity=(0, 1),
            B_lm=np.array(
                [
                    # f(r) = B0 + B1*(2r-1)
                    # f(0) = [0.8, 1.0, 1.2]
                    # f(1) = [1.0, 1.0, 1.0]
                    [0.9, 1.0, 1.1],  # B0
                    [0.1, 0.0, -0.1],  # B1
                ]
            ).flatten(),
        )

        mirror_ratio_axis = (1.2 - 0.8) / (1.2 + 0.8)
        mirror_ratio_edge = 0.0
        grid = LinearGrid(L=5, theta=6, N=2)
        rho = grid.nodes[grid.unique_rho_idx, 0]
        obj = MirrorRatio(field, grid=grid)
        obj.build()
        f = obj.compute(field.params_dict)
        np.testing.assert_allclose(
            f, mirror_ratio_axis * (1 - rho) + mirror_ratio_edge * rho
        )

    @pytest.mark.unit
    def test_linking_current(self):
        """Test calculation of signed linking current from coils to plasma."""
        eq = Equilibrium()
        G = eq.compute("G", grid=LinearGrid(rho=1.0))["G"][0] * 2 * jnp.pi / mu_0
        c = G / 8
        coil1 = FourierPlanarCoil(current=1.5 * c, center=[10, 1, 0])
        coil2 = FourierPlanarCoil(current=0.5 * c, center=[10, 2, 0])
        # explicit symmetry coils
        coilset1 = CoilSet.from_symmetry((coil1, coil2), NFP=2, sym=True)
        expected_currents = [
            c * 1.5,  # these are the 2 actual coils, with different currents
            c * 0.5,
            -c * 0.5,  # these are the stellarator symmetric ones in first field period
            -c * 1.5,
            c * 1.5,  # these next 4 are the ones from the 2nd field period
            c * 0.5,
            -c * 0.5,
            -c * 1.5,
        ]
        np.testing.assert_allclose(coilset1._all_currents(), expected_currents)
        obj = LinkingCurrentConsistency(eq, coilset1)
        obj.build()
        f = obj.compute(coilset1.params_dict, eq.params_dict)
        np.testing.assert_allclose(f, 0)

        # same with virtual coils
        coilset2 = CoilSet(coil1, coil2, NFP=2, sym=True)
        np.testing.assert_allclose(coilset2._all_currents(), expected_currents)
        obj = LinkingCurrentConsistency(eq, coilset2)
        obj.build()
        f = obj.compute(coilset2.params_dict, eq.params_dict)
        np.testing.assert_allclose(f, 0)

        # both coilsets together. These have overlapping coils but it doesn't
        # affect the linking number
        coilset3 = MixedCoilSet(coilset1, coilset2, check_intersection=False)
        np.testing.assert_allclose(
            coilset3._all_currents(), expected_currents + expected_currents
        )
        obj = LinkingCurrentConsistency(eq, coilset3)
        obj.build()
        f = obj.compute(coilset3.params_dict, eq.params_dict)
        np.testing.assert_allclose(f, -G)  # coils provide 2G so error is -G

        # CoilSet + 1 extra coil
        coilset4 = MixedCoilSet(coilset1, coil2, check_intersection=False)
        np.testing.assert_allclose(
            coilset4._all_currents(), expected_currents + [0.5 * G / 8]
        )
        obj = LinkingCurrentConsistency(eq, coilset4)
        obj.build()
        f = obj.compute(coilset4.params_dict, eq.params_dict)
        np.testing.assert_allclose(f, -0.5 * G / 8)

    @pytest.mark.unit
    def test_omnigenity_multiple_surfaces(self):
        """Test omnigenity transform vectorized over multiple surfaces."""
        surf = FourierRZToroidalSurface.from_qp_model(
            major_radius=1,
            aspect_ratio=20,
            elongation=6,
            mirror_ratio=0.2,
            torsion=0.1,
            NFP=1,
            sym=True,
        )
        eq = Equilibrium(
            Psi=6e-3,
            M=4,
            N=4,
            surface=surf,
            iota=PowerSeriesProfile(1, 0, -1),  # ensure diff surfs have diff iota
        )
        field = OmnigenousField(
            L_B=1,
            M_B=3,
            L_x=1,
            M_x=1,
            N_x=1,
            NFP=eq.NFP,
            helicity=(1, 1),
            B_lm=np.array(
                [
                    [0.8, 1.0, 1.2],
                    [-0.4, 0.0, 0.6],  # radially varying B
                ]
            ).flatten(),
        )
        grid1 = LinearGrid(rho=0.5, M=eq.M_grid, N=eq.N_grid)
        grid2 = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid)
        grid3 = LinearGrid(rho=np.array([0.5, 1.0]), M=eq.M_grid, N=eq.N_grid)
        obj1 = Omnigenity(eq=eq, field=field, eq_grid=grid1)
        obj2 = Omnigenity(eq=eq, field=field, eq_grid=grid2)
        obj3 = Omnigenity(eq=eq, field=field, eq_grid=grid3)
        obj1.build()
        obj2.build()
        obj3.build()
        f1 = obj1.compute(*obj1.xs(eq, field))
        f2 = obj2.compute(*obj2.xs(eq, field))
        f3 = obj3.compute(*obj3.xs(eq, field))
        # the order will be different but the values should be the same so we sort
        # before comparing
        np.testing.assert_allclose(
            np.sort(np.concatenate([f1, f2])), np.sort(f3), atol=1e-14
        )

    @pytest.mark.unit
    def test_surface_current_regularization(self):
        """Test SurfaceCurrentRegularization Calculation."""

        def test(field, grid):
            obj = SurfaceCurrentRegularization(field, source_grid=grid)
            obj.build()
            return obj.compute(field.params_dict)

        field1 = FourierCurrentPotentialField(
            I=0, G=10, NFP=10, Phi_mn=[[0]], modes_Phi=[[2, 2]]
        )
        grid = LinearGrid(M=5, N=5, NFP=field1.NFP)
        result1 = test(field1, grid)
        result2 = test(field1, grid=None)

        # test with CurrentPotentialField
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([10, 1]),
            Z_lmn=jnp.array([0, -1]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        surface.change_resolution(M=field1._M_Phi, N=field1._N_Phi)
        # make a current potential corresponding a purely poloidal current
        G = 10  # net poloidal current
        potential = lambda theta, zeta, G: G * zeta / 2 / jnp.pi
        potential_dtheta = lambda theta, zeta, G: jnp.zeros_like(theta)
        potential_dzeta = lambda theta, zeta, G: G * jnp.ones_like(theta) / 2 / jnp.pi

        params = {"G": -G}

        field2 = CurrentPotentialField(
            potential,
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            params=params,
            potential_dtheta=potential_dtheta,
            potential_dzeta=potential_dzeta,
            NFP=surface.NFP,
        )
        result3 = test(field2, grid)
        result4 = test(field2, grid=None)
        field1.G = 2 * field1.G
        result5 = test(field1, grid)

        np.testing.assert_allclose(result1, result3)
        np.testing.assert_allclose(result1 * 2, result5)
        np.testing.assert_allclose(result2, result4)

    @pytest.mark.unit
    def test_objective_compute(self):
        """To avoid issues such as #1424."""
        eq = get("W7-X")
        rho = np.linspace(0.1, 1, 3)
        grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
        X = 16
        Y = 32
        num_transit = 4
        num_well = 15 * num_transit
        num_quad = 16
        num_pitch = 16
        data = eq.compute(
            ["effective ripple", "Gamma_c"],
            grid=grid,
            theta=Bounce2D.compute_theta(eq, X=X, Y=Y, rho=rho),
            num_transit=num_transit,
            num_well=num_well,
            num_quad=num_quad,
            num_pitch=num_pitch,
        )
        obj = EffectiveRipple(
            eq,
            grid=grid,
            X=X,
            Y=Y,
            num_transit=num_transit,
            num_quad=num_quad,
            num_pitch=num_pitch,
        )
        obj.build()
        # TODO(#1094)
        np.testing.assert_allclose(
            obj.compute(eq.params_dict),
            grid.compress(data["effective ripple"]),
            rtol=0.004,
        )
        obj = GammaC(
            eq,
            grid=grid,
            X=X,
            Y=Y,
            num_transit=num_transit,
            num_quad=num_quad,
            num_pitch=num_pitch,
        )
        obj.build()
        np.testing.assert_allclose(
            obj.compute(eq.params_dict), grid.compress(data["Gamma_c"])
        )


@pytest.mark.regression
def test_derivative_modes():
    """Test equality of derivatives using batched, looped methods."""
    eq = Equilibrium(M=2, N=1, L=2)
    surf = FourierRZToroidalSurface()
    obj1 = ObjectiveFunction(
        [
            PlasmaVesselDistance(eq, surf, jac_chunk_size=1),
            MagneticWell(eq),
            AspectRatio(eq),
        ],
        deriv_mode="batched",
        use_jit=False,
    )
    obj2 = ObjectiveFunction(
        [
            PlasmaVesselDistance(eq, surf, jac_chunk_size=2),
            MagneticWell(eq),
            AspectRatio(eq, jac_chunk_size=None),
        ],
        deriv_mode="blocked",
        jac_chunk_size=10,
        use_jit=False,
    )
    with pytest.warns(DeprecationWarning, match="looped"):
        obj3 = ObjectiveFunction(
            [
                PlasmaVesselDistance(eq, surf),
                MagneticWell(eq),
                AspectRatio(eq),
            ],
            deriv_mode="looped",
            use_jit=False,
        )
    with pytest.raises(ValueError, match="jac_chunk_size"):
        obj1.build()
    with pytest.raises(ValueError, match="jac_chunk_size"):
        obj2.build()
    obj1 = ObjectiveFunction(
        [
            PlasmaVesselDistance(eq, surf),
            MagneticWell(eq),
            AspectRatio(eq),
        ],
        deriv_mode="batched",
        jac_chunk_size="auto",
        use_jit=False,
    )
    obj2 = ObjectiveFunction(
        [
            PlasmaVesselDistance(eq, surf, jac_chunk_size=2),
            MagneticWell(eq),
            AspectRatio(eq, jac_chunk_size=None),
        ],
        deriv_mode="blocked",
        jac_chunk_size="auto",
        use_jit=False,
    )
    obj1.build()
    obj2.build()
    # check that default size works for blocked
    assert obj2.objectives[0]._jac_chunk_size == 2
    assert obj2.objectives[1]._jac_chunk_size > 0
    assert obj2.objectives[2]._jac_chunk_size > 0
    # hard to say what size auto will give, just check it is >0
    assert obj1._jac_chunk_size > 0
    obj3.build()
    x = obj1.x(eq, surf)
    v = jnp.ones_like(x)
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
    j1 = obj1.jvp_scaled(v, x)
    j2 = obj2.jvp_scaled(v, x)
    j3 = obj3.jvp_scaled(v, x)
    np.testing.assert_allclose(j1, j2, atol=1e-10)
    np.testing.assert_allclose(j1, j3, atol=1e-10)


@pytest.mark.unit
def test_fwd_rev():
    """Test that forward and reverse mode jvps etc give same results."""
    eq = Equilibrium()
    obj1 = MeanCurvature(eq, deriv_mode="fwd")
    obj2 = MeanCurvature(eq, deriv_mode="rev")
    obj1.build()
    obj2.build()

    x = eq.pack_params(eq.params_dict)
    J1 = obj1.jac_scaled(x)
    J2 = obj2.jac_scaled(x)
    np.testing.assert_allclose(J1, J2, atol=1e-14)

    jvp1 = obj1.jvp_scaled(x, jnp.ones_like(x))
    jvp2 = obj2.jvp_scaled(x, jnp.ones_like(x))
    np.testing.assert_allclose(jvp1, jvp2, atol=1e-14)

    surf = FourierRZToroidalSurface()
    obj1 = PlasmaVesselDistance(eq, surf, deriv_mode="fwd")
    obj2 = PlasmaVesselDistance(eq, surf, deriv_mode="rev")
    obj1.build()
    obj2.build()

    x1 = eq.pack_params(eq.params_dict)
    x2 = surf.pack_params(surf.params_dict)

    J1a, J1b = obj1.jac_scaled(x1, x2)
    J2a, J2b = obj2.jac_scaled(x1, x2)
    np.testing.assert_allclose(J1a, J2a, atol=1e-14)
    np.testing.assert_allclose(J1b, J2b, atol=1e-14)

    jvp1 = obj1.jvp_scaled((x1, x2), (jnp.ones_like(x1), jnp.ones_like(x2)))
    jvp2 = obj2.jvp_scaled((x1, x2), (jnp.ones_like(x1), jnp.ones_like(x2)))
    np.testing.assert_allclose(jvp1, jvp2, atol=1e-14)


@pytest.mark.unit
def test_getter_setter():
    """Test getter and setter methods of Objectives."""
    eq = Equilibrium()
    obj = GenericObjective("R", thing=eq)
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
        GenericObjective("R", bounds=(1,), thing=eq).build()
    with pytest.raises(AssertionError):
        GenericObjective("R", bounds=(1, 2, 3), thing=eq).build()
    with pytest.raises(ValueError):
        GenericObjective("R", bounds=(1, -1), thing=eq).build()


@pytest.mark.unit
def test_target_profiles():
    """Tests for using Profile objects as targets for profile objectives."""
    iota = PowerSeriesProfile([1, 0, -0.3])
    shear = PowerSeriesProfile([0, -0.6])
    current = PowerSeriesProfile([0, 0, 1, 0, -1])
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
def test_profile_objective_print(capsys):
    """Test that the profile objectives print correctly."""
    eq = Equilibrium(
        iota=PowerSeriesProfile([1, 0, 0.5]), pressure=PowerSeriesProfile([1, 0, -1])
    )
    grid = LinearGrid(L=10, M=10, N=5, axis=False)
    pre_width = len("Maximum ")

    def test(obj, values, print_init=False, normalize=False):
        if print_init:
            # print the initial value too. For this test, it is the
            # same as the final value
            obj.print_value(obj.xs(eq), obj.xs(eq))
            print_fmt = (
                f"{obj._print_value_fmt:<{PRINT_WIDTH-pre_width}}"
                + "{:10.3e}  -->  {:10.3e} "
            )
        else:
            obj.print_value(obj.xs(eq))
            print_fmt = f"{obj._print_value_fmt:<{PRINT_WIDTH-pre_width}}" + "{:10.3e} "
        out = capsys.readouterr()

        corr_out = str(
            "Precomputing transforms\n"
            + "Maximum "
            + print_fmt.format(np.max(values), np.max(values))
            + obj._units
            + "\n"
            + "Minimum "
            + print_fmt.format(np.min(values), np.min(values))
            + obj._units
            + "\n"
            + "Average "
            + print_fmt.format(np.mean(values), np.mean(values))
            + obj._units
            + "\n"
        )
        if normalize:
            corr_out += str(
                "Maximum "
                + print_fmt.format(
                    np.max(values / obj.normalization),
                    np.max(values / obj.normalization),
                )
                + "(normalized)"
                + "\n"
                + "Minimum "
                + print_fmt.format(
                    np.min(values / obj.normalization),
                    np.min(values / obj.normalization),
                )
                + "(normalized)"
                + "\n"
                + "Average "
                + print_fmt.format(
                    np.mean(values / obj.normalization),
                    np.mean(values / obj.normalization),
                )
                + "(normalized)"
                + "\n"
            )

        assert out.out == corr_out

    iota = eq.compute("iota", grid=grid)["iota"]
    obj = RotationalTransform(eq=eq, target=1, grid=grid)
    obj.build()
    test(obj, iota, print_init=True)
    shear = eq.compute("shear", grid=grid)["shear"]
    obj = Shear(eq=eq, target=1, grid=grid)
    obj.build()
    test(obj, shear)
    curr = eq.compute("current", grid=grid)["current"]
    obj = ToroidalCurrent(eq=eq, target=1, grid=grid)
    obj.build()
    test(obj, curr, print_init=True, normalize=True)
    pres = eq.compute("p", grid=grid)["p"]
    obj = Pressure(eq=eq, target=1, grid=grid)
    obj.build()
    test(obj, pres, normalize=True)


@pytest.mark.unit
def test_plasma_vessel_distance_print(capsys):
    """Test that the PlasmaVesselDistance objective prints correctly."""
    pre_width = len("Maximum ")

    def test(obj, eq, surface, d, print_init=False):
        if print_init:
            if isinstance(obj, ObjectiveFunction):
                obj.print_value(obj.x(eq, surface), obj.x(eq, surface))
                print_fmt = (
                    f"{obj.objectives[0]._print_value_fmt:<{PRINT_WIDTH-pre_width}}"  # noqa: E501
                    + "{:10.3e}  -->  {:10.3e} "
                )
                units = obj.objectives[0]._units
                norm = obj.objectives[0].normalization
            else:
                obj.print_value(obj.xs(eq, surface), obj.xs(eq, surface))
                print_fmt = (
                    f"{obj._print_value_fmt:<{PRINT_WIDTH-pre_width}}"
                    + "{:10.3e}  -->  {:10.3e} "
                )
                units = obj._units
                norm = obj.normalization
        else:
            if isinstance(obj, ObjectiveFunction):
                obj.print_value(obj.x(eq, surface))
                print_fmt = (
                    f"{obj.objectives[0]._print_value_fmt:<{PRINT_WIDTH-pre_width}}"  # noqa: E501
                    + "{:10.3e} "
                )
                units = obj.objectives[0]._units
                norm = obj.objectives[0].normalization
            else:
                obj.print_value(obj.xs(eq, surface))
                print_fmt = (
                    f"{obj._print_value_fmt:<{PRINT_WIDTH-pre_width}}" + "{:10.3e} "
                )
                units = obj._units
                norm = obj.normalization
        out = capsys.readouterr()

        corr_out = str(
            "Maximum "
            + print_fmt.format(np.max(d), np.max(d))
            + units
            + "\n"
            + "Minimum "
            + print_fmt.format(np.min(d), np.min(d))
            + units
            + "\n"
            + "Average "
            + print_fmt.format(np.mean(d), np.mean(d))
            + units
            + "\n"
            + "Maximum "
            + print_fmt.format(np.max(d / norm), np.max(d / norm))
            + "(normalized)"
            + "\n"
            + "Minimum "
            + print_fmt.format(np.min(d / norm), np.min(d / norm))
            + "(normalized)"
            + "\n"
            + "Average "
            + print_fmt.format(np.mean(d / norm), np.mean(d / norm))
            + "(normalized)"
            + "\n"
        )
        if isinstance(obj, ObjectiveFunction):
            f = obj.compute_scalar(obj.x(eq, surface))
            if print_init:
                corr_out = (
                    str(
                        f"{'Total (sum of squares): ':<{PRINT_WIDTH}}"
                        + "{:10.3e}  -->  {:10.3e}, \n".format(f, f)
                    )
                    + corr_out
                )
            else:
                corr_out = (
                    str(
                        f"{'Total (sum of squares): ':<{PRINT_WIDTH}}"
                        + "{:10.3e}, \n".format(f)
                    )
                    + corr_out
                )
        assert out.out == corr_out

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
    obj.build(verbose=0)
    d = obj.compute_unscaled(*obj.xs(eq, surface))
    np.testing.assert_allclose(d, a_s - a_p)
    test(obj, eq, surface, d)
    test(obj, eq, surface, d, print_init=True)

    obj = ObjectiveFunction(
        PlasmaVesselDistance(
            eq=eq, plasma_grid=plas_grid, surface_grid=surf_grid, surface=surface
        )
    )
    obj.build(verbose=0)
    d = obj.compute_unscaled(obj.x(eq, surface))
    test(obj, eq, surface, d)
    test(obj, eq, surface, d, print_init=True)


@pytest.mark.unit
def test_boundary_error_print(capsys):
    """Test that the boundary error objectives print correctly."""
    coil = FourierXYZCoil(5e5)
    coilset = CoilSet.linspaced_angular(coil, n=100, check_intersection=False)
    coil_grid = LinearGrid(N=20)
    eq = Equilibrium(L=3, M=3, N=3, Psi=np.pi)

    obj = VacuumBoundaryError(eq, coilset, field_grid=coil_grid)
    obj.build()

    f = np.abs(obj.compute_unscaled(*obj.xs(eq, coilset)))
    n = len(f) // 2
    f1 = f[:n]
    f2 = f[n:]
    obj.print_value(obj.xs())
    out = capsys.readouterr()
    pre_width = len("Maximum absolute ")

    corr_out = str(
        "Precomputing transforms\n"
        + "Maximum absolute "
        + f"{'Boundary normal field error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f1))
        + "(T*m^2)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary normal field error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f1))
        + "(T*m^2)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary normal field error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f1))
        + "(T*m^2)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary normal field error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary normal field error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary normal field error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary magnetic pressure error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary magnetic pressure error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary magnetic pressure error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary magnetic pressure error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f2 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary magnetic pressure error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f2 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary magnetic pressure error: ':<{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f2 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
    )
    assert out.out == corr_out

    obj = BoundaryError(eq, coilset, field_grid=coil_grid)
    obj.build()

    f = np.abs(obj.compute_unscaled(*obj.xs(eq, coilset)))
    n = len(f) // 2
    f1 = f[:n]
    f2 = f[n:]
    obj.print_value(obj.xs())
    out = capsys.readouterr()

    corr_out = str(
        "Precomputing transforms\n"
        + "Maximum absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f1))
        + "(T*m^2)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f1))
        + "(T*m^2)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f1))
        + "(T*m^2)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f2 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f2 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f2 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
    )
    assert out.out == corr_out

    eq.surface = FourierCurrentPotentialField.from_surface(eq.surface)
    obj = BoundaryError(eq, coilset, field_grid=coil_grid)
    obj.build()

    f = np.abs(obj.compute_unscaled(*obj.xs(eq, coilset)))
    n = len(f) // 3
    f1 = f[:n]
    f2 = f[n : 2 * n]
    f3 = f[2 * n :]
    obj.print_value(obj.xs())
    out = capsys.readouterr()

    corr_out = str(
        "Precomputing transforms\n"
        + "Maximum absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f1))
        + "(T*m^2)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f1))
        + "(T*m^2)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f1))
        + "(T*m^2)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary normal field error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f1 / obj.normalization[0]))
        + "(normalized)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f2))
        + "(T^2*m^2)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f2 / obj.normalization[n]))
        + "(normalized)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f2 / obj.normalization[n]))
        + "(normalized)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary magnetic pressure error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f2 / obj.normalization[n]))
        + "(normalized)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary field jump error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f3))
        + "(T*m^2)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary field jump error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f3))
        + "(T*m^2)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary field jump error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f3))
        + "(T*m^2)"
        + "\n"
        + "Maximum absolute "
        + f"{'Boundary field jump error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.max(f3 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
        + "Minimum absolute "
        + f"{'Boundary field jump error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.min(f3 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
        + "Average absolute "
        + f"{'Boundary field jump error: ':{PRINT_WIDTH-pre_width}}"
        + "{:10.3e} ".format(np.mean(f3 / obj.normalization[-1]))
        + "(normalized)"
        + "\n"
    )
    assert out.out == corr_out


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
        Volume(target=target, normalize=True, weight=weight, eq=eq), use_jit=False
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
        ForceBalance(target=target, normalize=True, weight=weight, eq=eq), use_jit=False
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

    objective = ObjectiveFunction((vol, asp, fbl), use_jit=False)
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

    con = ObjectiveFunction(
        RotationalTransform(eq=eq, bounds=(0.41, 0.43)), use_jit=False
    )
    con.build()

    np.testing.assert_allclose(con.compute_scaled_error(con.x(eq)), 0)
    np.testing.assert_array_less(con.bounds_scaled[0], con.compute_scaled(con.x(eq)))
    np.testing.assert_array_less(con.compute_scaled(con.x(eq)), con.bounds_scaled[1])


@pytest.mark.unit
def test_softmax_and_softmin():
    """Test softmax and softmin function."""
    arr = np.arange(-17, 17, 5)
    # expect this to not be equal to the max but approximately so
    sftmax = softmax(arr, alpha=1)
    np.testing.assert_allclose(sftmax, np.max(arr), rtol=1e-2)

    # expect this to be equal to the max
    # as alpha -> infinity, softmax -> max
    sftmax = softmax(arr, alpha=100)
    np.testing.assert_almost_equal(sftmax, np.max(arr))

    # expect this to not be equal to the min but approximately so
    sftmin = softmin(arr, alpha=1)
    np.testing.assert_allclose(sftmin, np.min(arr), rtol=1e-2)

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


def _reduced_resolution_objective(eq, objective):
    """Speed up testing suite by defining rules to reduce objective resolution."""
    kwargs = {}
    if objective in {EffectiveRipple, GammaC}:
        kwargs["X"] = 8
        kwargs["Y"] = 16
        kwargs["num_transit"] = 4
        kwargs["num_well"] = 15 * kwargs["num_transit"]
        kwargs["num_pitch"] = 16
        kwargs["num_quad"] = 16
    return objective(eq=eq, **kwargs)


class TestComputeScalarResolution:
    """Test that compute_scalar values are roughly independent of grid resolution."""

    # get a list of all the objectives
    objectives = [
        getattr(desc.objectives, obj)
        for obj in dir(desc.objectives)
        if obj[0].isupper()
        and (not obj.startswith("Fix"))
        and (obj != "ObjectiveFunction")
        and ("SelfConsistency" not in obj)
    ]
    specials = [
        # these require special logic
        BootstrapRedlConsistency,
        BoundaryError,
        CoilArclengthVariance,
        CoilCurrentLength,
        CoilCurvature,
        CoilIntegratedCurvature,
        CoilLength,
        CoilSetLinkingNumber,
        CoilSetMinDistance,
        CoilTorsion,
        FusionPower,
        GenericObjective,
        HeatingPowerISS04,
        LinkingCurrentConsistency,
        Omnigenity,
        PlasmaCoilSetMinDistance,
        PlasmaVesselDistance,
        QuadraticFlux,
        SurfaceQuadraticFlux,
        ToroidalFlux,
        SurfaceCurrentRegularization,
        VacuumBoundaryError,
        # need to avoid blowup near the axis
        MercierStability,
        # don't test these since they depend on what user wants
        LinearObjectiveFromUser,
        ObjectiveFromUser,
    ]
    other_objectives = list(set(objectives) - set(specials))

    eq = get("HELIOTRON")
    res_array = np.array([2, 2.5, 3])

    @pytest.mark.regression
    def test_compute_scalar_resolution_plasma_vessel(self):
        """PlasmaVesselDistance."""
        f = np.zeros_like(self.res_array, dtype=float)
        surface = FourierRZToroidalSurface(
            R_lmn=[10, 1.5], Z_lmn=[-1.5], modes_R=[[0, 0], [1, 0]], modes_Z=[[-1, 0]]
        )
        for i, res in enumerate(self.res_array):
            grid = LinearGrid(
                M=int(self.eq.M * res), N=int(self.eq.N * res), NFP=self.eq.NFP
            )
            obj = ObjectiveFunction(
                PlasmaVesselDistance(
                    surface=surface, eq=self.eq, surface_grid=grid, plasma_grid=grid
                ),
                use_jit=False,
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_bootstrap(self):
        """BootstrapRedlConsistency."""
        eq = self.eq.copy()
        eq.electron_density = PowerSeriesProfile([1e19, 0, -1e19])
        eq.electron_temperature = PowerSeriesProfile([1e3, 0, -1e3])
        eq.ion_temperature = PowerSeriesProfile([1e3, 0, -1e3])
        eq.atomic_number = 1.0

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            grid = LinearGrid(
                M=int(self.eq.M * res), N=int(self.eq.N * res), NFP=self.eq.NFP
            )
            obj = ObjectiveFunction(
                BootstrapRedlConsistency(eq=eq, grid=grid), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_fusion_power(self):
        """FusionPower."""
        eq = self.eq.copy()
        eq.electron_density = PowerSeriesProfile([1e19, 0, -1e19])
        eq.electron_temperature = PowerSeriesProfile([1e3, 0, -1e3])
        eq.ion_temperature = PowerSeriesProfile([1e3, 0, -1e3])
        eq.atomic_number = 1.0

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            grid = QuadratureGrid(
                L=int(self.eq.L * res),
                M=int(self.eq.M * res),
                N=int(self.eq.N * res),
                NFP=self.eq.NFP,
            )
            obj = ObjectiveFunction(FusionPower(eq=eq, grid=grid))
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_heating_power(self):
        """HeatingPowerISS04."""
        eq = self.eq.copy()
        eq.electron_density = PowerSeriesProfile([1e19, 0, -1e19])
        eq.electron_temperature = PowerSeriesProfile([1e3, 0, -1e3])
        eq.ion_temperature = PowerSeriesProfile([1e3, 0, -1e3])
        eq.atomic_number = 1.0

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            grid = QuadratureGrid(
                L=int(self.eq.L * res),
                M=int(self.eq.M * res),
                N=int(self.eq.N * res),
                NFP=self.eq.NFP,
            )
            obj = ObjectiveFunction(HeatingPowerISS04(eq=eq, grid=grid))
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_boundary_error(self):
        """BoundaryError."""
        with pytest.warns(UserWarning):
            # user warning because saved mgrid no vector potential
            ext_field = SplineMagneticField.from_mgrid(r"tests/inputs/mgrid_solovev.nc")

        pres = PowerSeriesProfile([1.25e-1, 0, -1.25e-1])
        iota = PowerSeriesProfile([-4.9e-1, 0, 3.0e-1])
        surf = FourierRZToroidalSurface(
            R_lmn=[4.0, 1.0],
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=[-1.0],
            modes_Z=[[-1, 0]],
            NFP=1,
        )
        eq = Equilibrium(M=6, N=0, Psi=1.0, surface=surf, pressure=pres, iota=iota)

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            eq.change_resolution(
                L_grid=int(eq.L * res), M_grid=int(eq.M * res), N_grid=int(eq.N * res)
            )
            obj = ObjectiveFunction(BoundaryError(eq, ext_field), use_jit=False)
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_vacuum_boundary_error(self):
        """VacuumBoundaryError."""
        with pytest.warns(UserWarning):
            # user warning because saved mgrid no vector potential
            ext_field = SplineMagneticField.from_mgrid(r"tests/inputs/mgrid_solovev.nc")

        pres = PowerSeriesProfile([1.25e-1, 0, -1.25e-1])
        iota = PowerSeriesProfile([-4.9e-1, 0, 3.0e-1])
        surf = FourierRZToroidalSurface(
            R_lmn=[4.0, 1.0],
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=[-1.0],
            modes_Z=[[-1, 0]],
            NFP=1,
        )
        eq = Equilibrium(M=6, N=0, Psi=1.0, surface=surf, pressure=pres, iota=iota)

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            eq.change_resolution(
                L_grid=int(eq.L * res), M_grid=int(eq.M * res), N_grid=int(eq.N * res)
            )
            obj = ObjectiveFunction(VacuumBoundaryError(eq, ext_field), use_jit=False)
            with pytest.warns(UserWarning):
                obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_quadratic_flux(self):
        """QuadraticFlux."""
        with pytest.warns(UserWarning):
            ext_field = SplineMagneticField.from_mgrid(r"tests/inputs/mgrid_solovev.nc")

        pres = PowerSeriesProfile([1.25e-1, 0, -1.25e-1])
        iota = PowerSeriesProfile([-4.9e-1, 0, 3.0e-1])
        surf = FourierRZToroidalSurface(
            R_lmn=[4.0, 1.0],
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=[-1.0],
            modes_Z=[[-1, 0]],
            NFP=1,
        )
        eq = Equilibrium(M=6, N=0, Psi=1.0, surface=surf, pressure=pres, iota=iota)

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            eq.change_resolution(
                L_grid=int(eq.L * res), M_grid=int(eq.M * res), N_grid=int(eq.N * res)
            )
            obj = ObjectiveFunction(QuadraticFlux(eq, ext_field), use_jit=False)
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_quadratic_flux_minimizing(self):
        """SurfaceQuadraticFlux."""
        ext_field = ToroidalMagneticField(1.0, 1.0)

        surf = FourierRZToroidalSurface(
            R_lmn=[4.0, 1.0],
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=[-1.0],
            modes_Z=[[-1, 0]],
            NFP=1,
        )

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            surf.change_resolution(
                M=int(surf.M * res),
                N=int(surf.N * res),
            )
            obj = ObjectiveFunction(
                SurfaceQuadraticFlux(surf, ext_field), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_toroidal_flux_A(self):
        """ToroidalFlux."""
        ext_field = ToroidalMagneticField(1, 1)
        eq = get("precise_QA")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(4, 4, 4, 8, 8, 8)

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            eq.change_resolution(
                L_grid=int(eq.L * res), M_grid=int(eq.M * res), N_grid=int(eq.N * res)
            )
            obj = ObjectiveFunction(
                ToroidalFlux(eq, ext_field, eq_fixed=True), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_toroidal_flux_B(self):
        """ToroidalFlux."""
        ext_field = ToroidalMagneticField(1, 1)
        eq = get("precise_QA")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(4, 4, 4, 8, 8, 8)

        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            eq.change_resolution(
                L_grid=int(eq.L * res), M_grid=int(eq.M * res), N_grid=int(eq.N * res)
            )
            obj = ObjectiveFunction(
                ToroidalFlux(eq, ext_field, eq_fixed=True), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_surface_current_reg(self):
        """SurfaceCurrentRegularization."""
        field = FourierCurrentPotentialField(
            I=1, G=1, Phi_mn=np.array([1, 1]), modes_Phi=np.array([[1, 1], [4, 4]])
        )
        M0 = 5
        N0 = 5
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            grid = LinearGrid(M=round(M0 * res), N=round(N0 * res))
            obj = ObjectiveFunction(
                SurfaceCurrentRegularization(field, source_grid=grid), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=5e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_generic_scalar(self):
        """Generic objective with scalar qty."""
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            grid = QuadratureGrid(
                L=int(self.eq.L * res),
                M=int(self.eq.M * res),
                N=int(self.eq.N * res),
                NFP=self.eq.NFP,
            )
            obj = ObjectiveFunction(
                GenericObjective("<beta>_vol", thing=self.eq, grid=grid), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=1e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_generic_profile(self):
        """Generic objective with profile qty."""
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            grid = LinearGrid(
                L=int(self.eq.L * res),
                M=int(self.eq.M * res),
                N=int(self.eq.N * res),
                NFP=self.eq.NFP,
                sym=self.eq.sym,
                axis=False,
            )
            obj = ObjectiveFunction(
                GenericObjective("<J*B>", thing=self.eq, grid=grid), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_generic_volume(self):
        """Generic objective with volume qty."""
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            grid = ConcentricGrid(
                L=int(self.eq.L * res),
                M=int(self.eq.M * res),
                N=int(self.eq.N * res),
                NFP=self.eq.NFP,
                sym=self.eq.sym,
            )
            obj = ObjectiveFunction(
                GenericObjective("sqrt(g)", thing=self.eq, grid=grid), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_mercier(self):
        """Mercier stability."""
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            rho = np.linspace(0.2, 1, int(self.eq.L * res))
            grid = LinearGrid(
                rho=rho,
                M=int(self.eq.M * res),
                N=int(self.eq.N * res),
                NFP=self.eq.NFP,
                sym=self.eq.sym,
            )
            obj = ObjectiveFunction(
                MercierStability(eq=self.eq, grid=grid), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=2e-2)

    @pytest.mark.regression
    def test_compute_scalar_resolution_omnigenity(self):
        """Omnigenity."""
        surf = FourierRZToroidalSurface.from_qp_model(
            major_radius=1,
            aspect_ratio=20,
            elongation=6,
            mirror_ratio=0.2,
            torsion=0.1,
            NFP=1,
            sym=True,
        )
        eq = Equilibrium(Psi=6e-3, M=4, N=4, surface=surf)
        eq, _ = eq.solve(objective="force", verbose=3)
        field = OmnigenousField(
            L_B=0,
            M_B=2,
            L_x=0,
            M_x=0,
            N_x=0,
            NFP=eq.NFP,
            helicity=(0, eq.NFP),
            B_lm=np.array([0.8, 1.2]),
        )
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array + 0.5):  # omnigenity needs higher res
            grid = LinearGrid(M=int(eq.M * res), N=int(eq.N * res), NFP=eq.NFP)
            obj = ObjectiveFunction(
                Omnigenity(eq=eq, field=field, eq_grid=grid, field_grid=grid)
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x(eq, field))
        np.testing.assert_allclose(f, f[-1], rtol=1e-3)

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "objective", sorted(other_objectives, key=lambda x: str(x.__name__))
    )
    def test_compute_scalar_resolution_others(self, objective):
        """All other objectives."""
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            # just change eq resolution and let objective pick the right grid type
            self.eq.change_resolution(
                L_grid=int(self.eq.L * res),
                M_grid=int(self.eq.M * res),
                N_grid=int(self.eq.N * res),
            )
            obj = ObjectiveFunction(
                _reduced_resolution_objective(self.eq, objective), use_jit=False
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(
            f, f[-1], rtol=6e-2, atol=1e-4 if np.max(f) < 1e-3 else 0
        )

    @pytest.mark.regression
    @pytest.mark.parametrize(
        "objective",
        [
            CoilArclengthVariance,
            CoilCurrentLength,
            CoilCurvature,
            CoilIntegratedCurvature,
            CoilLength,
            CoilTorsion,
            CoilSetLinkingNumber,
            CoilSetMinDistance,
        ],
    )
    def test_compute_scalar_resolution_coils(self, objective):
        """Coil objectives."""
        coil = FourierXYZCoil()
        coilset = CoilSet.linspaced_angular(coil, check_intersection=False)
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            obj = ObjectiveFunction(
                objective(coilset, grid=LinearGrid(N=int(5 + 3 * res))),
                use_jit=False,
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=1e-2, atol=1e-12)

    @pytest.mark.unit
    def test_compute_scalar_resolution_linking_current(self):
        """LinkingCurrentConsistency."""
        coil = FourierPlanarCoil(center=[10, 1, 0])
        eq = Equilibrium()
        coilset = CoilSet.from_symmetry(coil, NFP=4, sym=True)
        f = np.zeros_like(self.res_array, dtype=float)
        for i, res in enumerate(self.res_array):
            obj = ObjectiveFunction(
                LinkingCurrentConsistency(
                    eq,
                    coilset,
                    grid=LinearGrid(M=int(eq.M_grid * res), N=int(eq.N_grid * res)),
                ),
                use_jit=False,
            )
            obj.build(verbose=0)
            f[i] = obj.compute_scalar(obj.x())
        np.testing.assert_allclose(f, f[-1], rtol=1e-2, atol=1e-12)


class TestObjectiveNaNGrad:
    """Make sure reverse mode AD works correctly for all objectives."""

    # get a list of all the objectives
    objectives = [
        getattr(desc.objectives, obj)
        for obj in dir(desc.objectives)
        if obj[0].isupper()
        and (not obj.startswith("Fix"))
        and (obj != "ObjectiveFunction")
        and ("SelfConsistency" not in obj)
    ]
    specials = [
        # these require special logic
        BallooningStability,
        BootstrapRedlConsistency,
        BoundaryError,
        CoilArclengthVariance,
        CoilCurrentLength,
        CoilCurvature,
        CoilIntegratedCurvature,
        CoilLength,
        CoilSetLinkingNumber,
        CoilSetMinDistance,
        CoilTorsion,
        EffectiveRipple,
        ForceBalanceAnisotropic,
        FusionPower,
        GammaC,
        HeatingPowerISS04,
        LinkingCurrentConsistency,
        Omnigenity,
        PlasmaCoilSetMinDistance,
        PlasmaVesselDistance,
        QuadraticFlux,
        SurfaceCurrentRegularization,
        SurfaceQuadraticFlux,
        ToroidalFlux,
        VacuumBoundaryError,
        # we don't test these since they depend too much on what exactly the user wants
        GenericObjective,
        LinearObjectiveFromUser,
        ObjectiveFromUser,
    ]
    other_objectives = list(set(objectives) - set(specials))

    @pytest.mark.unit
    def test_objective_no_nangrad_plasma_vessel(self):
        """PlasmaVesselDistance."""
        eq = Equilibrium(L=2, M=2, N=2)
        surf = FourierRZToroidalSurface()
        obj = ObjectiveFunction(PlasmaVesselDistance(eq, surf), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(eq, surf))
        assert not np.any(np.isnan(g)), "plasma vessel distance"

    @pytest.mark.unit
    def test_objective_no_nangrad_anisotropy(self):
        """ForceBalanceAnisotropic."""
        eq = Equilibrium(L=2, M=2, N=2, anisotropy=FourierZernikeProfile())
        obj = ObjectiveFunction(ForceBalanceAnisotropic(eq), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(eq))
        assert not np.any(np.isnan(g)), "anisotropic"

    @pytest.mark.unit
    def test_objective_no_nangrad_bootstrap(self):
        """BootstrapRedlConsistency."""
        eq = Equilibrium(
            L=2,
            M=2,
            N=2,
            electron_density=PowerSeriesProfile([1e19, 0, -1e19]),
            electron_temperature=PowerSeriesProfile([1e3, 0, -1e3]),
            current=PowerSeriesProfile([0, 0, -1]),
        )
        obj = ObjectiveFunction(BootstrapRedlConsistency(eq), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(eq))
        assert not np.any(np.isnan(g)), "redl bootstrap"

    @pytest.mark.unit
    def test_objective_no_nangrad_fusion_power(self):
        """FusionPower."""
        eq = Equilibrium(
            L=2,
            M=2,
            N=2,
            electron_density=PowerSeriesProfile([1e19, 0, -1e19]),
            electron_temperature=PowerSeriesProfile([1e3, 0, -1e3]),
            current=PowerSeriesProfile([0, 0, -1]),
        )
        obj = ObjectiveFunction(FusionPower(eq))
        obj.build()
        g = obj.grad(obj.x(eq))
        assert not np.any(np.isnan(g)), "fusion power"

    @pytest.mark.unit
    def test_objective_no_nangrad_heating_power(self):
        """HeatingPowerISS04."""
        eq = Equilibrium(
            L=2,
            M=2,
            N=2,
            electron_density=PowerSeriesProfile([1e19, 0, -1e19]),
            electron_temperature=PowerSeriesProfile([1e3, 0, -1e3]),
            current=PowerSeriesProfile([0, 0, -1]),
        )
        obj = ObjectiveFunction(HeatingPowerISS04(eq))
        obj.build()
        g = obj.grad(obj.x(eq))
        assert not np.any(np.isnan(g)), "heating power"

    @pytest.mark.unit
    def test_objective_no_nangrad_boundary_error(self):
        """BoundaryError."""
        with pytest.warns(UserWarning):
            # user warning because saved mgrid no vector potential
            ext_field = SplineMagneticField.from_mgrid(r"tests/inputs/mgrid_solovev.nc")

        pres = PowerSeriesProfile([1.25e-1, 0, -1.25e-1])
        iota = PowerSeriesProfile([-4.9e-1, 0, 3.0e-1])
        surf = FourierRZToroidalSurface(
            R_lmn=[4.0, 1.0],
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=[-1.0],
            modes_Z=[[-1, 0]],
            NFP=1,
        )

        eq = Equilibrium(M=6, N=0, Psi=1.0, surface=surf, pressure=pres, iota=iota)
        obj = ObjectiveFunction(BoundaryError(eq, ext_field), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(eq, ext_field))
        assert not np.any(np.isnan(g)), "boundary error"

    @pytest.mark.unit
    def test_objective_no_nangrad_vacuum_boundary_error(self):
        """VacuumBoundaryError."""
        with pytest.warns(UserWarning):
            # user warning because saved mgrid no vector potential
            ext_field = SplineMagneticField.from_mgrid(r"tests/inputs/mgrid_solovev.nc")

        pres = PowerSeriesProfile([1.25e-1, 0, -1.25e-1])
        iota = PowerSeriesProfile([-4.9e-1, 0, 3.0e-1])
        surf = FourierRZToroidalSurface(
            R_lmn=[4.0, 1.0],
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=[-1.0],
            modes_Z=[[-1, 0]],
            NFP=1,
        )

        eq = Equilibrium(M=6, N=0, Psi=1.0, surface=surf, pressure=pres, iota=iota)

        obj = ObjectiveFunction(VacuumBoundaryError(eq, ext_field), use_jit=False)
        with pytest.warns(UserWarning):
            obj.build()
        g = obj.grad(obj.x(eq, ext_field))
        assert not np.any(np.isnan(g)), "vacuum boundary error"

    @pytest.mark.unit
    def test_objective_no_nangrad_quadratic_flux(self):
        """QuadraticFlux."""
        with pytest.warns(UserWarning):
            # user warning because saved mgrid no vector potential
            ext_field = SplineMagneticField.from_mgrid(r"tests/inputs/mgrid_solovev.nc")

        pres = PowerSeriesProfile([1.25e-1, 0, -1.25e-1])
        iota = PowerSeriesProfile([-4.9e-1, 0, 3.0e-1])
        surf = FourierRZToroidalSurface(
            R_lmn=[4.0, 1.0],
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=[-1.0],
            modes_Z=[[-1, 0]],
            NFP=1,
        )

        eq = Equilibrium(M=6, N=0, Psi=1.0, surface=surf, pressure=pres, iota=iota)

        obj = ObjectiveFunction(QuadraticFlux(eq, ext_field), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(ext_field))
        assert not np.any(np.isnan(g)), "quadratic flux"

    @pytest.mark.unit
    def test_objective_no_nangrad_quadratic_flux_minimizing(self):
        """SurfaceQuadraticFlux."""
        surf = FourierRZToroidalSurface(
            R_lmn=[4.0, 1.0],
            modes_R=[[0, 0], [1, 0]],
            Z_lmn=[-1.0],
            modes_Z=[[-1, 0]],
            NFP=1,
        )

        def test(normal):
            ext_field = FourierPlanarCoil(center=[0, 0, 3.0], normal=normal)
            obj = ObjectiveFunction(
                SurfaceQuadraticFlux(surf, ext_field), use_jit=False
            )
            obj.build()
            g = obj.grad(obj.x(surf, ext_field))
            assert not np.any(np.isnan(g)), "quadratic flux"

        test([0, 0, 1])  # normal parallel to Z-axis
        test([0, 0, -1])  # antiparallel
        test([0, 1e-4, 1])  # nearly parallel
        test([0, 1e-4, -1])  # nearly antiparallel

    @pytest.mark.unit
    def test_objective_no_nangrad_toroidal_flux(self):
        """ToroidalFlux."""
        ext_field = ToroidalMagneticField(1, 1)

        eq = get("precise_QA")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(4, 4, 4, 8, 8, 8)

        obj = ObjectiveFunction(ToroidalFlux(eq, ext_field), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(eq, ext_field))
        assert not np.any(np.isnan(g)), "toroidal flux A"

        obj = ObjectiveFunction(ToroidalFlux(eq, ext_field), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(eq, ext_field))
        assert not np.any(np.isnan(g)), "toroidal flux B"

    @pytest.mark.unit
    def test_objective_no_nangrad_surface_current_reg(self):
        """SurfaceCurrentRegularization."""
        field = FourierCurrentPotentialField()

        obj = ObjectiveFunction(SurfaceCurrentRegularization(field), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(field))
        assert not np.any(np.isnan(g)), "surface current regularization"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "objective", sorted(other_objectives, key=lambda x: str(x.__name__))
    )
    def test_objective_no_nangrad(self, objective):
        """Generic test for other objectives."""
        eq = Equilibrium(L=2, M=2, N=2)
        obj = ObjectiveFunction(objective(eq), use_jit=False)
        obj.build()
        g = obj.grad(obj.x(eq))
        assert not np.any(np.isnan(g)), str(objective)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "objective",
        [
            CoilArclengthVariance,
            CoilCurrentLength,
            CoilCurvature,
            CoilIntegratedCurvature,
            CoilLength,
            CoilTorsion,
            CoilSetLinkingNumber,
            CoilSetMinDistance,
        ],
    )
    def test_objective_no_nangrad_coils(self, objective):
        """Coil objectives."""
        coil = FourierXYZCoil()
        coilset = CoilSet.linspaced_angular(coil, n=3, check_intersection=False)
        obj = ObjectiveFunction(objective(coilset), use_jit=False)
        obj.build(verbose=0)
        g = obj.grad(obj.x())
        assert not np.any(np.isnan(g)), str(objective)

    @pytest.mark.unit
    @pytest.mark.parametrize("helicity", [(1, 0), (1, 1), (0, 1)])
    def test_objective_no_nangrad_omnigenity(self, helicity):
        """Omnigenity."""
        surf = FourierRZToroidalSurface.from_qp_model(
            major_radius=1,
            aspect_ratio=20,
            elongation=6,
            mirror_ratio=0.2,
            torsion=0.1,
            NFP=1,
            sym=True,
        )
        eq = Equilibrium(Psi=6e-3, M=4, N=4, surface=surf)
        field = OmnigenousField(
            L_B=0,
            M_B=2,
            L_x=1,
            M_x=1,
            N_x=1,
            NFP=eq.NFP,
            helicity=helicity,
            B_lm=np.array([0.8, 1.2]),
        )
        obj = ObjectiveFunction(Omnigenity(eq=eq, field=field))
        obj.build()
        g = obj.grad(obj.x())
        assert not np.any(np.isnan(g)), str(helicity)

    @pytest.mark.unit
    def test_objective_no_nangrad_effective_ripple(self):
        """Effective ripple."""
        eq = get("ESTELL")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(2, 2, 2, 4, 4, 4)
        obj = ObjectiveFunction(_reduced_resolution_objective(eq, EffectiveRipple))
        obj.build(verbose=0)
        g = obj.grad(obj.x())
        assert not np.any(np.isnan(g))

    @pytest.mark.unit
    def test_objective_no_nangrad_Gamma_c(self):
        """Gamma_c."""
        eq = get("ESTELL")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(2, 2, 2, 4, 4, 4)
        obj = ObjectiveFunction(_reduced_resolution_objective(eq, GammaC))
        obj.build(verbose=0)
        g = obj.grad(obj.x())
        assert not np.any(np.isnan(g))

    @pytest.mark.unit
    def test_objective_no_nangrad_ballooning(self):
        """BallooningStability."""
        eq = get("HELIOTRON")
        obj = ObjectiveFunction(BallooningStability(eq=eq))
        obj.build()
        g = obj.grad(obj.x())
        assert not np.any(np.isnan(g))

    @pytest.mark.unit
    def test_objective_no_nangrad_linking_current(self):
        """LinkingCurrentConsistency."""
        coil = FourierPlanarCoil(center=[10, 1, 0])
        coilset = CoilSet.from_symmetry(coil, NFP=4, sym=True)
        eq = Equilibrium()
        obj = ObjectiveFunction(LinkingCurrentConsistency(eq, coilset))
        obj.build()
        g = obj.grad(obj.x())
        assert not np.any(np.isnan(g))


@pytest.mark.unit
def test_asymmetric_normalization():
    """Tests normalizations for asymmetric equilibrium."""
    # related to PR #821
    a = 0.6
    # make a asym equilibrium with 0 for R_1_0 and Z_-1_0
    surf = FourierRZToroidalSurface(
        R_lmn=[10, -a],
        Z_lmn=[0, -a],
        modes_R=np.array([[0, 0], [-1, 0]]),
        modes_Z=np.array([[0, 0], [1, 0]]),
        sym=False,
    )
    eq = Equilibrium(surface=surf)
    scales_surf = compute_scaling_factors(surf)
    scales_eq = compute_scaling_factors(eq)

    for val in scales_surf.values():
        assert np.all(np.isfinite(val))
    for val in scales_eq.values():
        assert np.all(np.isfinite(val))


@pytest.mark.unit
def test_objective_print_widths():
    """Test that the objective's name is shorter than max."""
    subclasses = _Objective.__subclasses__()
    max_prewidth = len("Maximum Absolute ")
    max_width = PRINT_WIDTH - max_prewidth
    # check every subclass of _Objective class
    for subclass in subclasses:
        try:
            assert len(subclass._print_value_fmt) <= max_width, (
                f"{subclass.__name__} is too long for PRINT_WIDTH.\n"
                + "Note to Devs: If this is a new objective, please make sure the "
                + "name is short enough to fit in the PRINT_WIDTH. Either "
                + "change the name or increase the PRINT_WIDTH in the "
                + "desc/utils.py file. The former is preferred."
            )
        except AttributeError:
            # if the subclass has subclasses, check those
            subsubclasses = subclass.__subclasses__()
            for subsubclass in subsubclasses:
                assert len(subsubclass._print_value_fmt) <= max_width, (
                    f"{subsubclass.__name__} is too long for PRINT_WIDTH.\n"
                    + "Note to Devs: If this is a new objective, please make sure the "
                    + "name is short enough to fit in the PRINT_WIDTH. Either "
                    + "change the name or increase the PRINT_WIDTH in the "
                    + "desc/utils.py file. The former is preferred."
                )


@pytest.mark.unit
def test_objective_docstring():
    """Test that the objective docstring and collect_docs are consistent."""
    objective_docs = _Objective.__doc__.rstrip()
    doc_header = (
        "Objective (or constraint) used in the optimization of an Equilibrium.\n\n"
        + "    Parameters\n"
        + "    ----------\n"
        + "    things : Optimizable or tuple/list of Optimizable\n"
        + "        Objects that will be optimized to satisfy the Objective.\n"
    )
    collected_docs = collect_docs().strip()
    collected_docs = doc_header + "    " + collected_docs

    assert objective_docs == collected_docs
