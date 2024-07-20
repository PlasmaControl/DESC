"""Tests for _Configuration base class."""

import warnings

import numpy as np
import pytest

import desc.examples
from desc.backend import put
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.equilibrium.initial_guess import _initial_guess_surface
from desc.geometry import (
    FourierRZCurve,
    FourierRZToroidalSurface,
    ZernikeRZToroidalSection,
)
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid
from desc.profiles import PowerSeriesProfile, SplineProfile


class TestConstructor:
    """Tests for creating equilibrium objects."""

    @pytest.mark.unit
    def test_defaults(self):
        """Tests that default attribute values get set correctly."""
        eq = Equilibrium()

        assert eq.spectral_indexing == "ansi"
        assert eq.NFP == 1
        assert eq.L == 1
        assert eq.M == 1
        assert eq.N == 0
        assert eq.sym is False
        assert eq.surface.equiv(FourierRZToroidalSurface(sym=False))
        assert isinstance(eq.pressure, PowerSeriesProfile)
        assert isinstance(eq.current, PowerSeriesProfile)
        np.testing.assert_allclose(eq.p_l, [0])
        np.testing.assert_allclose(eq.c_l, [0])

    @pytest.mark.unit
    def test_supplied_objects(self):
        """Tests that profile and surface objects are parsed correctly."""
        pressure = SplineProfile([1, 2, 3])
        iota = SplineProfile([2, 3, 4])
        surface = FourierRZToroidalSurface(NFP=2, sym=False)
        axis = FourierRZCurve([-0.2, 10, 0.3], [0.3, 0, -0.2], NFP=2, sym=False)
        eq = Equilibrium(
            M=2,
            pressure=pressure,
            iota=iota,
            surface=surface,
            axis=axis,
            N=1,
            sym=False,
        )

        assert eq.pressure.equiv(pressure)
        assert eq.iota.equiv(iota)
        assert eq.spectral_indexing == "ansi"
        assert eq.NFP == 2
        assert eq.axis.NFP == 2

        np.testing.assert_allclose(axis.R_n, eq.Ra_n)
        np.testing.assert_allclose(axis.Z_n, eq.Za_n)

        surface2 = ZernikeRZToroidalSection(spectral_indexing="ansi")
        eq2 = Equilibrium(surface=surface2)
        assert eq2.surface.equiv(surface2)

        surface3 = FourierRZToroidalSurface(NFP=3)
        eq3 = Equilibrium(surface=surface3)
        assert eq3.NFP == 3
        assert eq3.axis.NFP == 3

        eq4 = Equilibrium(surface=surface2, axis=None)
        np.testing.assert_allclose(eq4.axis.R_n, [10])

    @pytest.mark.unit
    def test_dict(self):
        """Test creating an equilibrium from a dictionary of arrays."""
        inputs = {
            "L": 4,
            "M": 2,
            "N": 2,
            "NFP": 3,
            "sym": False,
            "spectral_indexing": "ansi",
            "surface": np.array(
                [
                    [0, 0, 0, 10, 0],
                    [0, 1, 0, 1, 0],
                    [0, -1, 0, 0, -1],
                    [0, -1, 1, 0.1, 0.1],
                ]
            ),
            "axis": np.array([[0, 10, 0]]),
            "pressure": np.array([[0, 10], [2, 5]]),
            "iota": np.array([[0, 1], [2, 3]]),
        }
        eq = Equilibrium(**inputs)

        assert eq.L == 4
        assert eq.M == 2
        assert eq.N == 2
        assert eq.NFP == 3
        assert eq.spectral_indexing == "ansi"
        np.testing.assert_allclose(eq.p_l, [10, 5, 0])
        np.testing.assert_allclose(eq.i_l, [1, 3, 0])
        assert isinstance(eq.surface, FourierRZToroidalSurface)
        np.testing.assert_allclose(
            eq.Rb_lmn,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                10.0,
                1.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
        np.testing.assert_allclose(
            eq.Zb_lmn,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )

        inputs["surface"] = np.array(
            [
                [0, 0, 0, 10, 0],
                [1, 1, 0, 1, 0.1],
                [1, -1, 0, 0.2, -1],
            ]
        )

        eq = Equilibrium(**inputs)
        assert eq.bdry_mode == "poincare"
        np.testing.assert_allclose(
            eq.Rb_lmn, [10.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            eq.Zb_lmn, [0.0, -1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking in equilibrium creation."""
        with pytest.raises(ValueError):
            eq = Equilibrium(L=3.4)
        with pytest.raises(ValueError):
            eq = Equilibrium(M=3.4)
        with pytest.raises(ValueError):
            eq = Equilibrium(N=3.4)
        with pytest.raises(ValueError):
            eq = Equilibrium(NFP=3.4j)
        with pytest.raises(ValueError):
            eq = Equilibrium(surface=np.array([[1, 1, 1, 10, 2]]))
        with pytest.raises(TypeError):
            eq = Equilibrium(surface=FourierRZCurve())
        with pytest.raises(TypeError):
            eq = Equilibrium(axis=2)
        with pytest.raises(ValueError):
            eq = Equilibrium(surface=FourierRZToroidalSurface(NFP=1), NFP=2)
        with pytest.raises(TypeError):
            eq = Equilibrium(pressure="abc")
        with pytest.raises(TypeError):
            eq = Equilibrium(iota="def")
        with pytest.raises(TypeError):
            eq = Equilibrium(current="def")
        with pytest.raises(ValueError):  # change to TypeError if allow both
            eq = Equilibrium(iota="def", current="def")
        with pytest.raises(ValueError):
            eq = Equilibrium(iota=None)
            eq.i_l = None
        with pytest.raises(ValueError):
            eq = Equilibrium(iota=PowerSeriesProfile(params=[1, 3], modes=[0, 2]))
            eq.c_l = None

    @pytest.mark.unit
    def test_supplied_coeffs(self):
        """Test passing in specific spectral coefficients."""
        R_lmn = np.random.random(3)
        Z_lmn = np.random.random(3)
        L_lmn = np.random.random(3)
        eq = Equilibrium(R_lmn=R_lmn, Z_lmn=Z_lmn, L_lmn=L_lmn, check_orientation=False)
        np.testing.assert_allclose(R_lmn, eq.R_lmn)
        np.testing.assert_allclose(Z_lmn, eq.Z_lmn)
        np.testing.assert_allclose(L_lmn, eq.L_lmn)

        with pytest.raises(AssertionError):
            eq = Equilibrium(L=4, R_lmn=R_lmn)


class TestInitialGuess:
    """Tests for setting initial guess for an equilibrium."""

    @pytest.mark.unit
    def test_default_set(self):
        """Test the default initial guess."""
        eq = Equilibrium()
        eq.set_initial_guess()
        np.testing.assert_allclose(eq.compute("V")["V"], 2 * 10 * np.pi * np.pi * 1 * 1)
        del eq._axis
        eq.set_initial_guess()
        np.testing.assert_allclose(eq.compute("V")["V"], 2 * 10 * np.pi * np.pi * 1 * 1)

    @pytest.mark.unit
    def test_errors(self):
        """Test error checking for setting initial guess."""
        eq = Equilibrium()
        with pytest.raises(ValueError):
            eq.set_initial_guess(1, "a", 4, 5, 6)
        with pytest.raises(ValueError):
            eq.set_initial_guess(1, 2)
        with pytest.raises(ValueError):
            eq.set_initial_guess(eq, eq.surface)
        with pytest.raises(TypeError):
            eq.set_initial_guess(eq.surface, [1, 2, 3])
        del eq._surface
        with pytest.raises(ValueError):
            eq.set_initial_guess()

        with pytest.raises(ValueError):
            eq.set_initial_guess("path", 3)
        with pytest.raises(ValueError):
            eq.set_initial_guess("path", "hdf5")
        with pytest.raises(ValueError):
            eq.surface = eq.get_surface_at(rho=1)
            eq.change_resolution(2, 2, 2)
            _ = _initial_guess_surface(eq.R_basis, eq.R_lmn, eq.R_basis)
        with pytest.raises(ValueError):
            _ = _initial_guess_surface(
                eq.R_basis, eq.surface.R_lmn, eq.surface.R_basis, mode="foo"
            )

    @pytest.mark.unit
    def test_guess_from_other(self):
        """Test using one equilibrium as the initial guess for another."""
        eq1 = Equilibrium(L=4, M=2)
        eq2 = Equilibrium(L=2, M=1)
        eq2.set_initial_guess(eq1)

        eq2.change_resolution(L=4, M=2)
        np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn)
        np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn)

    @pytest.mark.unit
    def test_guess_from_surface(self):
        """Test initial guess by scaling boundary surface."""
        eq = Equilibrium()
        surface = FourierRZToroidalSurface()
        # turn the circular cross-section into an ellipse w AR=2
        surface.set_coeffs(m=-1, n=0, R=None, Z=-2)
        # move z axis up to 0.5 for no good reason
        axis = FourierRZCurve([0, 10, 0], [0, 0.5, 0])
        eq.set_initial_guess(surface, axis)
        np.testing.assert_allclose(eq.compute("V")["V"], 2 * 10 * np.pi * np.pi * 2 * 1)

    @pytest.mark.unit
    def test_guess_from_surface2(self):
        """Test initial guess by scaling interior surface."""
        eq = Equilibrium()
        # specify an interior flux surface
        surface = FourierRZToroidalSurface(rho=0.5)
        eq.set_initial_guess(surface)
        np.testing.assert_allclose(eq.compute("V")["V"], 2 * 10 * np.pi * np.pi * 2**2)

    @pytest.mark.unit
    def test_guess_from_points(self):
        """Test initial guess by fitting R,Z at specified points."""
        eq = Equilibrium(L=3, M=3, N=1)
        # these are just the default circular tokamak with a random normal
        # perturbation with std=0.03, fixed for repeatability
        eq.R_lmn = np.array(
            [
                3.94803875e-02,
                7.27321367e-03,
                -8.88095373e-03,
                1.47523628e-02,
                1.18518478e-02,
                -2.61657165e-02,
                -1.27473081e-02,
                3.26441003e-02,
                4.47427817e-03,
                1.24734770e-02,
                9.99231496e00,
                -2.74400311e-03,
                1.00447777e00,
                3.22285107e-02,
                1.16571026e-02,
                -3.15868165e-03,
                -6.77657739e-04,
                -1.97894171e-02,
                2.13535622e-02,
                -2.19703593e-02,
                5.15586341e-02,
                3.39651128e-02,
                -1.66077603e-02,
                -2.20514583e-02,
                -3.13335598e-02,
                7.16090760e-02,
                -1.30064709e-03,
                -4.00687024e-02,
                5.25583677e-02,
                4.04325991e-03,
            ]
        )
        eq.Z_lmn = np.array(
            [
                2.58179465e-02,
                -6.58108612e-03,
                3.67459870e-02,
                9.32236734e-04,
                -2.07982449e-03,
                -1.67700140e-02,
                2.56951390e-02,
                -4.49230035e-04,
                9.93325894e-02,
                4.28162330e-03,
                9.39812383e-03,
                9.95829268e-01,
                4.14468984e-02,
                -3.10725101e-02,
                -1.42026152e-02,
                -2.20423483e-02,
                -1.37389716e-02,
                -1.31592276e-02,
                -3.13922472e-02,
                1.88145630e-03,
                2.72255620e-02,
                -9.42746650e-03,
                2.15264372e-02,
                2.43549268e-02,
                5.33383228e-02,
                1.65948808e-02,
                1.45908076e-03,
                1.85101895e-02,
                1.25967662e-02,
                -2.07374046e-02,
            ]
        )
        grid = ConcentricGrid(L=6, M=6, N=2, node_pattern="ocs")
        coords = eq.compute(["R", "Z", "lambda"], grid=grid)
        eq2 = Equilibrium(L=3, M=3, N=1)
        eq2.set_initial_guess(grid.nodes, coords["R"], coords["Z"], coords["lambda"])
        np.testing.assert_allclose(eq.R_lmn, eq2.R_lmn, atol=1e-8)
        np.testing.assert_allclose(eq.Z_lmn, eq2.Z_lmn, atol=1e-8)
        np.testing.assert_allclose(eq.L_lmn, eq2.L_lmn, atol=1e-8)

    @pytest.mark.unit
    def test_NFP_error(self):
        """Check for ValueError when eq, axis, and surface NFPs do not agree."""
        axis = FourierRZCurve([-1, 10, 1], [1, 0, -1], NFP=2)
        surface2 = FourierRZToroidalSurface(NFP=2)
        surface3 = FourierRZToroidalSurface(NFP=3)

        # test axis and eq NFP not agreeing
        with pytest.raises(ValueError):
            _ = Equilibrium(surface=surface3, axis=axis, NFP=3)

        # test axis and surface NFP not agreeing
        with pytest.raises(ValueError):
            _ = Equilibrium(surface=surface2, axis=axis, NFP=3)

        # test surface and eq NFP not agreeing
        with pytest.raises(ValueError):
            _ = Equilibrium(surface=surface3, axis=axis, NFP=2)

    @pytest.mark.unit
    def test_guess_from_file(self):
        """Test setting initial guess from saved equilibrium file."""
        path = "tests//inputs//iotest_HELIOTRON.h5"
        eq1 = Equilibrium(L=9, M=14, N=3, sym=True, spectral_indexing="ansi")
        eq1.set_initial_guess(path)
        eq2 = EquilibriaFamily.load(path)[-1]

        np.testing.assert_allclose(eq1.R_lmn, eq2.R_lmn)
        np.testing.assert_allclose(eq1.Z_lmn, eq2.Z_lmn)

    @pytest.mark.unit
    def test_guess_from_coordinate_mapping(self):
        """Test that we can initialize strongly shaped equilibria correctly."""
        Rb = np.array([3.51, 1.1, 1.5, -0.3])
        R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        Zb = np.array([0.0, 0.16, -2])
        Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
        surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes)
        with pytest.warns(UserWarning):
            eq = Equilibrium(M=surf.M, N=surf.N, surface=surf)

        assert eq.is_nested()

    @pytest.mark.unit
    def test_guess_from_coordinate_mapping_no_sym(self):
        """Test that we can initialize strongly shaped equilibria correctly.

        (without axisymmetry or stellarator symmetry)
        """
        Rb = np.array([10, 1, 0.8, -0.2, 0.3, 0.02])
        R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 1], [2, 1], [2, 2]])
        Zb = np.array([0.01, 0.2, -1.5, 0.2])
        Z_modes = np.array([[-3, -2], [2, -1], [-1, 0], [1, 1]])
        surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes)
        with pytest.warns(UserWarning):
            eq = Equilibrium(M=surf.M, N=surf.N, surface=surf)

        assert eq.is_nested()


class TestGetSurfaces:
    """Tests for get_surface method."""

    @pytest.mark.unit
    def test_get_rho_surface(self):
        """Test getting a constant rho surface."""
        eq = Equilibrium()
        R0 = 10
        rho = 0.5
        surf = eq.get_surface_at(rho=rho)
        assert surf.rho == rho
        np.testing.assert_allclose(surf.compute("S")["S"], 4 * np.pi**2 * R0 * rho)

    @pytest.mark.unit
    @pytest.mark.xfail(reason="GitHub issue 1127.")
    def test_get_zeta_surface(self):
        """Test getting a constant zeta surface."""
        eq = Equilibrium()
        surf = eq.get_surface_at(zeta=np.pi)
        assert surf.zeta == np.pi
        rho = 1
        np.testing.assert_allclose(surf.compute("A")["A"], np.pi * rho**2)

    @pytest.mark.unit
    def test_get_theta_surface(self):
        """Test that getting a constant theta surface doesn't work yet."""
        eq = Equilibrium()
        with pytest.raises(NotImplementedError):
            _ = eq.get_surface_at(theta=np.pi)

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking in get_surface method."""
        eq = Equilibrium()
        with pytest.raises(ValueError):
            _ = eq.get_surface_at(rho=1, zeta=2)
        with pytest.raises(AssertionError):
            _ = eq.get_surface_at(rho=1.2)


@pytest.mark.unit
def test_magnetic_axis():
    """Test that Configuration.axis returns the true axis location."""
    eq = desc.examples.get("HELIOTRON")
    axis = eq.axis
    grid = LinearGrid(N=3 * eq.N_grid, NFP=eq.NFP, rho=np.array(0.0))

    data = eq.compute(["R", "Z"], grid=grid)
    coords = axis.compute("x", grid=grid)["x"]

    np.testing.assert_allclose(coords[:, 0], data["R"])
    np.testing.assert_allclose(coords[:, 2], data["Z"])


@pytest.mark.unit
def test_is_nested():
    """Test that jacobian sign indicates whether surfaces are nested."""
    eq = Equilibrium()
    grid = QuadratureGrid(L=10, M=10, N=0)
    assert eq.is_nested(grid=grid)

    eq.change_resolution(L=2, M=2)
    eq.R_lmn = put(eq.R_lmn, eq.R_basis.get_idx(L=1, M=1, N=0), 1)
    # make unnested by setting higher order mode to same amplitude as lower order mode
    eq.R_lmn = put(eq.R_lmn, eq.R_basis.get_idx(L=2, M=2, N=0), 1)

    assert not eq.is_nested(grid=grid)
    with pytest.warns(Warning) as record:
        assert not eq.is_nested(grid=grid, msg="auto")
    assert len(record) == 1
    assert "Automatic" in str(record[0].message)
    with pytest.warns(Warning) as record:
        assert not eq.is_nested(grid=grid, msg="manual")
    assert len(record) == 1
    assert "perturbation" in str(record[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not eq.is_nested(grid=grid, msg=None)


@pytest.mark.unit
def test_is_nested_theta():
    """Test that new version of is nested also catches messed up theta contours."""
    eq = Equilibrium(L=6, M=6, N=0, iota=1)
    # just mess with lambda, so rho contours are the same
    eq.L_lmn += 1e-1 * np.random.default_rng(seed=3).random(eq.L_lmn.shape)
    grid = QuadratureGrid(10, 10, 0, NFP=eq.NFP)
    g1 = eq.compute("sqrt(g)", grid=grid)["sqrt(g)"]
    g2 = eq.compute("sqrt(g)_PEST", grid=grid)["sqrt(g)_PEST"]
    assert np.all(g1 > 0)  # regular jacobian will still be fine
    assert np.any(g2 < 0)  # PEST jacobian should be negative
    assert not eq.is_nested()


@pytest.mark.unit
def test_get_profile():
    """Test getting/setting iota and current profiles."""
    eq = desc.examples.get("DSHAPE_CURRENT")
    current0 = eq.current
    current1 = eq.get_profile("current", kind="power_series")
    current2 = eq.get_profile("current", kind="spline")
    current3 = eq.get_profile("current", kind="fourier_zernike")

    x = np.linspace(0, 1, 20)
    np.testing.assert_allclose(current0(x), current1(x), rtol=1e-6, atol=1e-1)
    np.testing.assert_allclose(current0(x), current2(x), rtol=1e-6, atol=1e-1)
    np.testing.assert_allclose(current0(x), current3(x), rtol=1e-6, atol=1e-1)


@pytest.mark.unit
def test_kinetic_errors():
    """Test that we can't set nonexistent profile values."""
    eqp = Equilibrium(L=3, M=3, N=3, pressure=np.array([1, 0, -1]))
    eqk = Equilibrium(
        L=3,
        M=3,
        N=3,
        electron_temperature=np.array([1, 0, -1]),
        electron_density=np.array([2, 0, -2]),
    )
    params = np.arange(3)
    with pytest.raises(ValueError):
        eqk.p_l = params
    with pytest.raises(ValueError):
        eqp.Te_l = params
    with pytest.raises(ValueError):
        eqp.ne_l = params
    with pytest.raises(ValueError):
        eqp.Ti_l = params
    with pytest.raises(ValueError):
        eqp.Zeff_l = params

    params = np.ones((3, 4))
    profile = PowerSeriesProfile()
    eqk.pressure = profile
    eqp.electron_temperature = profile
    eqp.electron_density = profile
    eqp.ion_temperature = profile
    eqp.atomic_number = profile
    with pytest.raises(TypeError):
        eqk.pressure = params
    with pytest.raises(TypeError):
        eqp.electron_temperature = params
    with pytest.raises(TypeError):
        eqp.electron_density = params
    with pytest.raises(TypeError):
        eqp.ion_temperature = params
    with pytest.raises(TypeError):
        eqp.atomic_number = params

    with pytest.raises(ValueError):
        _ = Equilibrium(pressure=1, electron_density=1, electron_temperature=1)
    with pytest.raises(ValueError):
        _ = Equilibrium(electron_temperature=1)
