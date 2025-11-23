"""Tests for magnetic field classes."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from diffrax import (
    Dopri5,
    Event,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
)
from scipy.constants import mu_0

from desc.backend import jax, jit, jnp
from desc.basis import DoubleFourierSeries
from desc.compute.utils import get_params, get_transforms
from desc.derivatives import FiniteDiffDerivative as Derivative
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface, FourierXYZCurve
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import (
    CurrentPotentialField,
    DommaschkPotentialField,
    FourierCurrentPotentialField,
    MagneticFieldFromUser,
    OmnigenousField,
    PoloidalMagneticField,
    ScalarPotentialField,
    SplineMagneticField,
    ToroidalMagneticField,
    VectorPotentialField,
    VerticalMagneticField,
    field_line_integrate,
    read_BNORM_file,
    solve_regularized_surface_current,
)
from desc.magnetic_fields._core import _field_line_integrate
from desc.magnetic_fields._dommaschk import CD_m_k, CN_m_k
from desc.plotting import poincare_plot
from desc.utils import dot, rpz2xyz, rpz2xyz_vec, xyz2rpz_vec


def phi_lm(R, phi, Z, a, m):
    """Scalar potential test function."""
    CNm0 = (R**m - R**-m) / (2 * m)
    Nm1 = CNm0 * Z
    CDm0 = (R**m + R**-m) / 2
    c1 = -m * (m - 1)
    c2 = (m + 1) * (m - 2)
    c3 = m * (m + 1)
    c4 = -(m + 2) * (m - 1)
    CDm1 = (c1 * R ** (m + 2) + c2 * R ** (m) + c3 * R ** (-m + 2) + c4 * R ** (-m)) / (
        8 * m * (m**2 - 1)
    )
    Dm2 = CDm0 * Z**2 / 2 + CDm1
    return phi + a * Dm2 * jnp.sin(m * phi) + a * Nm1 * jnp.cos(m * phi)


a = -1.489
m = 5
args = {"a": a, "m": m}


class TestMagneticFields:
    """Tests for MagneticField classes."""

    @pytest.mark.unit
    def test_basic_fields(self):
        """Tests for basic field types (toroidal, vertical, poloidal)."""
        tfield = ToroidalMagneticField(2, 1)
        vfield = VerticalMagneticField(1)
        pfield = PoloidalMagneticField(2, 1, 2)

        def tfield_A(R, phi, Z, B0=2, R0=1):
            az = -B0 * R0 * jnp.log(R)
            arp = jnp.zeros_like(az)
            A = jnp.array([arp, arp, az]).T
            return A

        tfield_from_A = VectorPotentialField(tfield_A, params={"B0": 2, "R0": 1})

        def vfield_A(R, phi, Z, B0=None):
            coords_rpz = jnp.vstack([R, phi, Z]).T
            coords_xyz = rpz2xyz(coords_rpz)
            ax = B0 / 2 * coords_xyz[:, 1]
            ay = -B0 / 2 * coords_xyz[:, 0]

            az = jnp.zeros_like(ax)
            A = jnp.array([ax, -ay, az]).T
            A = xyz2rpz_vec(A, phi=coords_rpz[:, 1])
            return A

        vfield_params = {"B0": 1}
        vfield_from_A = VectorPotentialField(vfield_A, params=vfield_params)

        np.testing.assert_allclose(tfield([1, 0, 0]), [[0, 2, 0]])
        np.testing.assert_allclose((4 * tfield)([2, 0, 0]), [[0, 4, 0]])
        np.testing.assert_allclose(tfield_from_A([1, 0, 0]), [[0, 2, 0]])
        np.testing.assert_allclose(
            tfield_A(1, 0, 0),
            tfield_from_A.compute_magnetic_vector_potential([1, 0, 0]).squeeze(),
        )
        np.testing.assert_allclose(
            vfield_A(1, 0, 0, **vfield_params),
            vfield_from_A.compute_magnetic_vector_potential([1, 0, 0]),
        )

        np.testing.assert_allclose((tfield + vfield)([1, 0, 0]), [[0, 2, 1]])
        np.testing.assert_allclose(
            (tfield + vfield - pfield)([1, 0, 0.1]), [[0.4, 2, 1]]
        )

    @pytest.mark.unit
    def test_field_from_user(self):
        """Test for MagneticFieldFromUser."""
        tfield = ToroidalMagneticField(2, 1)

        def fun(coords, params):
            R0, B0 = params
            coords = jnp.atleast_2d(jnp.asarray(coords))
            bp = B0 * R0 / coords[:, 0]
            brz = jnp.zeros_like(bp)
            B = jnp.array([brz, bp, brz]).T
            return B

        ufield = MagneticFieldFromUser(fun, [tfield.R0, tfield.B0])
        np.testing.assert_allclose(
            tfield([1, 0, 0]), ufield([1, 0, 0], params=[tfield.R0, tfield.B0])
        )
        np.testing.assert_allclose(
            tfield([1, 1, 0], basis="xyz"),
            ufield([1, 1, 0], params=[tfield.R0, tfield.B0], basis="xyz"),
        )

    @pytest.mark.unit
    def test_combined_fields(self):
        """Tests for sum/scaled fields."""
        tfield = ToroidalMagneticField(2, 1)
        vfield = VerticalMagneticField(1)
        pfield = PoloidalMagneticField(2, 1, 2)
        tfield.R0 = 1
        tfield.B0 = 2
        vfield.B0 = 3.2
        pfield.R0 = 1
        pfield.B0 = 2
        pfield.iota = 1.2
        scaled_field = 3.1 * tfield
        assert scaled_field.B0 == 2
        assert scaled_field.scale == 3.1
        np.testing.assert_allclose(scaled_field([1.0, 0, 0]), np.array([[0, 6.2, 0]]))
        np.testing.assert_allclose(
            scaled_field.compute_magnetic_vector_potential([2.0, 0, 0]),
            np.array([[0, 0, -3.1 * 2 * 1 * np.log(2)]]),
        )

        scaled_field.R0 = 1.3
        scaled_field.scale = 1.0
        np.testing.assert_allclose(scaled_field([1.3, 0, 0]), np.array([[0, 2, 0]]))
        np.testing.assert_allclose(
            scaled_field.compute_magnetic_vector_potential([2.0, 0, 0]),
            np.array([[0, 0, -2 * 1.3 * np.log(2)]]),
        )
        assert scaled_field.optimizable_params == ["B0", "R0", "scale"]
        assert hasattr(scaled_field, "B0")

        sum_field = vfield + pfield + tfield
        sum_field_tv = vfield + tfield  # to test A since pfield does not have A
        assert len(sum_field) == 3
        assert len(sum_field_tv) == 2

        np.testing.assert_allclose(
            sum_field([1.3, 0, 0.0]), [[0.0, 2, 3.2 + 2 * 1.2 * 0.3]]
        )

        tfield_A = np.array([[0, 0, -tfield.B0 * tfield.R0 * np.log(tfield.R0)]])
        x = tfield.R0 * np.cos(np.pi / 4)
        y = tfield.R0 * np.sin(np.pi / 4)
        vfield_A = np.array([[vfield.B0 * y, -vfield.B0 * x, 0]]) / 2

        np.testing.assert_allclose(
            sum_field_tv.compute_magnetic_vector_potential([x, y, 0.0], basis="xyz"),
            tfield_A + vfield_A,
        )

        assert sum_field.optimizable_params == [
            ["B0"],
            ["B0", "R0", "iota"],
            ["B0", "R0"],
        ]
        assert sum_field.dimensions == [
            {"B0": 1},
            {"B0": 1, "R0": 1, "iota": 1},
            {"B0": 1, "R0": 1},
        ]
        assert sum_field.x_idx == [
            {"B0": np.array([0])},
            {"B0": np.array([1]), "R0": np.array([2]), "iota": np.array([3])},
            {"B0": np.array([4]), "R0": np.array([5])},
        ]
        assert sum_field.dim_x == 6
        p = sum_field.pack_params(sum_field.params_dict)
        np.testing.assert_allclose(p, [3.2, 2, 1, 1.2, 2, 1.3])
        p *= 1.3
        sum_field.params_dict = sum_field.unpack_params(p)
        assert sum_field[0].B0 == 1.3 * 3.2
        assert sum_field[1].B0 == 1.3 * 2
        assert sum_field[2].B0 == 1.3 * 2
        sum_field.params_dict = sum_field.unpack_params(p / 1.3)

        del sum_field[-1]
        assert len(sum_field) == 2
        np.testing.assert_allclose(
            sum_field([1.3, 0, 0.0]), [[0.0, 0.0, (3.2 + 2 * 1.2 * 0.3)]]
        )
        assert sum_field.optimizable_params == [
            ["B0"],
            ["B0", "R0", "iota"],
        ]
        sum_field.insert(1, tfield)
        assert len(sum_field) == 3
        np.testing.assert_allclose(
            sum_field([1.3, 0, 0.0]), [[0.0, 2.0, (3.2 + 2 * 1.2 * 0.3)]]
        )
        assert sum_field.optimizable_params == [
            ["B0"],
            ["B0", "R0"],
            ["B0", "R0", "iota"],
        ]
        sum_field[1] = pfield
        sum_field[2] = tfield
        assert sum_field.optimizable_params == [
            ["B0"],
            ["B0", "R0", "iota"],
            ["B0", "R0"],
        ]

    @pytest.mark.unit
    def test_scalar_field(self):
        """Test scalar potential magnetic field against analytic result."""
        field = ScalarPotentialField(phi_lm, args)
        np.testing.assert_allclose(
            field.compute_magnetic_field([1.0, 0, 0]), [[0, 1, 0]]
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field([1.0, np.pi / 4, 0]), [[0, 1, 0]]
        )

    @pytest.mark.unit
    def test_current_potential_field(self):
        """Test current potential magnetic field against analytic result."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([10, 1]),
            Z_lmn=jnp.array([0, -1]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        # make a current potential corresponding a purely poloidal current
        G = 10  # net poloidal current
        potential = lambda theta, zeta, G: G * zeta / 2 / jnp.pi
        potential_dtheta = lambda theta, zeta, G: jnp.zeros_like(theta)
        potential_dzeta = lambda theta, zeta, G: G * jnp.ones_like(theta) / 2 / jnp.pi

        params = {"G": -G}
        correct_field = lambda R, phi, Z: jnp.array([[0, mu_0 * G / 2 / jnp.pi / R, 0]])

        field = CurrentPotentialField(
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

        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, 0, 0]),
            correct_field(10.0, 0, 0),
            atol=1e-16,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, np.pi / 4, 0]),
            correct_field(10.0, np.pi / 4, 0),
            atol=1e-16,
            rtol=1e-8,
        )

        potential = lambda theta, zeta, G: G * zeta / 2 / jnp.pi * 2
        potential_dtheta = lambda theta, zeta, G: jnp.zeros_like(theta)
        potential_dzeta = (
            lambda theta, zeta, G: G * jnp.ones_like(theta) / 2 / jnp.pi * 2
        )

        correct_field = lambda R, phi, Z: 2 * jnp.array(
            [[0, mu_0 * G / 2 / jnp.pi / R, 0]]
        )
        field.potential = potential
        field.potential_dtheta = potential_dtheta
        field.potential_dzeta = potential_dzeta

        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, 0, 0]),
            correct_field(10.0, 0, 0),
            atol=1e-16,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, np.pi / 4, 0]),
            correct_field(10.0, np.pi / 4, 0),
            atol=1e-16,
            rtol=1e-8,
        )

        field = CurrentPotentialField.from_surface(
            surface=surface,
            potential=potential,
            params=params,
            potential_dtheta=potential_dtheta,
            potential_dzeta=potential_dzeta,
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, 0, 0]),
            correct_field(10.0, 0, 0),
            atol=1e-16,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, np.pi / 4, 0]),
            correct_field(10.0, np.pi / 4, 0),
            atol=1e-16,
            rtol=1e-8,
        )

        # add a ToroidalField and check passing in/not passing in

        B_TF = ToroidalMagneticField(1, 10)
        sumfield = B_TF + field

        np.testing.assert_allclose(
            sumfield.compute_magnetic_field([10.0, 0, 0]),
            correct_field(10.0, 0, 0) + B_TF([10.0, 0, 0]),
            atol=1e-16,
            rtol=1e-8,
        )

        np.testing.assert_allclose(
            sumfield.compute_magnetic_field(
                [10.0, 0, 0],
                source_grid=[None, LinearGrid(M=30, N=30, NFP=surface.NFP)],
            ),
            correct_field(10.0, 0, 0) + B_TF([10.0, 0, 0]),
            atol=1e-16,
            rtol=1e-8,
        )

        with pytest.raises(IOError):
            field.save("test_field.h5")
        with pytest.warns(UserWarning):
            # check that if passing in a longer params dict, that
            # the warning is thrown
            field.params = {"key1": None, "key2": None}
        # check assert callable statement
        with pytest.raises(AssertionError):
            field.potential = 1
        with pytest.raises(AssertionError):
            field.potential_dtheta = 1
        with pytest.raises(AssertionError):
            field.potential_dzeta = 1

    @pytest.mark.unit
    def test_current_potential_vector_potential(self):
        """Test current potential field vector potential against analytic result."""
        R0 = 10
        a = 1
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([R0, a]),
            Z_lmn=jnp.array([0, -a]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        # make a current potential corresponding a purely poloidal current
        G = 100  # net poloidal current
        potential = lambda theta, zeta, G: G * zeta / 2 / jnp.pi
        potential_dtheta = lambda theta, zeta, G: jnp.zeros_like(theta)
        potential_dzeta = lambda theta, zeta, G: G * jnp.ones_like(theta) / 2 / jnp.pi

        params = {"G": -G}

        field = CurrentPotentialField(
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
        # test the loop integral of A around a curve encompassing the torus
        # against the analytic result for flux in an ideal toroidal solenoid
        prefactors = mu_0 * G / 2 / jnp.pi
        correct_flux = -2 * np.pi * prefactors * (np.sqrt(R0**2 - a**2) - R0)

        curve = FourierXYZCurve()  # curve to integrate A over
        curve_grid = LinearGrid(zeta=20)
        curve_data = curve.compute(["x", "x_s"], grid=curve_grid, basis="xyz")
        curve_data_rpz = curve.compute(["x", "x_s"], grid=curve_grid, basis="rpz")

        surface_grid = LinearGrid(M=60, N=60, NFP=10)

        A_xyz = field.compute_magnetic_vector_potential(
            curve_data["x"], basis="xyz", source_grid=surface_grid
        )
        A_rpz = field.compute_magnetic_vector_potential(
            curve_data_rpz["x"], basis="rpz", source_grid=surface_grid
        )

        # integrate to get the flux
        flux_xyz = jnp.sum(
            dot(A_xyz, curve_data["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )
        flux_rpz = jnp.sum(
            dot(A_rpz, curve_data_rpz["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )

        np.testing.assert_allclose(correct_flux, flux_xyz, rtol=1e-8)
        np.testing.assert_allclose(correct_flux, flux_rpz, rtol=1e-8)

        field.params["G"] = -2 * field.params["G"]

        A_xyz = field.compute_magnetic_vector_potential(
            curve_data["x"], basis="xyz", source_grid=surface_grid
        )
        A_rpz = field.compute_magnetic_vector_potential(
            curve_data_rpz["x"], basis="rpz", source_grid=surface_grid
        )

        # integrate to get the flux
        flux_xyz = jnp.sum(
            dot(A_xyz, curve_data["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )
        flux_rpz = jnp.sum(
            dot(A_rpz, curve_data_rpz["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )

        np.testing.assert_allclose(-2 * correct_flux, flux_xyz, rtol=1e-8)
        np.testing.assert_allclose(-2 * correct_flux, flux_rpz, rtol=1e-8)

    @pytest.mark.unit
    def test_fourier_current_potential_field(self):
        """Test Fourier current potential magnetic field against analytic result."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([10, 1]),
            Z_lmn=jnp.array([0, -1]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        basis = DoubleFourierSeries(M=2, N=2, sym="sin")
        phi_mn = np.ones((basis.num_modes,))
        # make a current potential corresponding a purely poloidal current
        G = 10  # net poloidal current
        correct_field = lambda R, phi, Z: jnp.array([[0, mu_0 * G / 2 / jnp.pi / R, 0]])

        field = FourierCurrentPotentialField(
            Phi_mn=phi_mn,
            modes_Phi=basis.modes[:, 1:],
            I=0,
            G=-G,
            sym_Phi="sin",
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )
        surface_grid = LinearGrid(M=120, N=120, NFP=10)

        phi_mn = np.zeros((basis.num_modes,))

        field.Phi_mn = phi_mn

        field.change_resolution(3, 3)
        field.change_Phi_resolution(2, 2)

        params = get_params(["K", "x"], field)
        transforms = get_transforms(["K", "x"], field, grid=surface_grid)

        np.testing.assert_allclose(
            field.compute_magnetic_field(
                [10.0, 0, 0],
                source_grid=surface_grid,
                params=params,
                transforms=transforms,
            ),
            correct_field(10.0, 0, 0),
            atol=1e-16,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field(
                [10.0, np.pi / 4, 0], source_grid=surface_grid
            ),
            correct_field(10.0, np.pi / 4, 0),
            atol=1e-16,
            rtol=1e-8,
        )

        field.G = -2 * G
        field.I = 0

        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, 0, 0], source_grid=surface_grid),
            correct_field(10.0, 0, 0) * 2,
            atol=1e-16,
            rtol=1e-8,
        )
        # use default grid
        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, np.pi / 4, 0], source_grid=None),
            correct_field(10.0, np.pi / 4, 0) * 2,
            atol=1e-12,
            rtol=1e-8,
        )

        field = FourierCurrentPotentialField.from_surface(
            surface=surface,
            Phi_mn=phi_mn,
            modes_Phi=basis.modes[:, 1:],
            I=0,
            G=-G,
        )

        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, 0, 0], source_grid=surface_grid),
            correct_field(10.0, 0, 0),
            atol=1e-16,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field(
                [10.0, np.pi / 4, 0], source_grid=surface_grid
            ),
            correct_field(10.0, np.pi / 4, 0),
            atol=1e-16,
            rtol=1e-8,
        )

        K_xyz = field.compute(["K", "x"], basis="xyz", grid=surface_grid)
        K_rpz = field.compute(["K", "x"], basis="rpz", grid=surface_grid)

        np.testing.assert_allclose(
            K_xyz["K"], rpz2xyz_vec(K_rpz["K"], phi=K_rpz["x"][:, 1]), atol=1e-16
        )
        np.testing.assert_allclose(
            K_rpz["K"],
            xyz2rpz_vec(K_xyz["K"], x=K_xyz["x"][:, 0], y=K_xyz["x"][:, 1]),
            atol=1e-16,
        )

    @pytest.mark.unit
    def test_fourier_current_potential_change_Phi_resolution(self):
        """Test Fourier current potential changing Phi resolution."""
        surface = FourierRZToroidalSurface(sym=False)
        current_field = FourierCurrentPotentialField.from_surface(
            surface, sym_Phi=False, M_Phi=4, N_Phi=0
        )
        assert current_field.M_Phi == 4
        assert current_field.N_Phi == 0
        current_field.change_Phi_resolution(M=8, N=0)
        assert current_field.M_Phi == 8
        assert current_field.N_Phi == 0

    @pytest.mark.unit
    @pytest.mark.mpl_image_compare(remove_text=True, tolerance=10)
    def test_fourier_current_potential_field_modular_coil_cut(self):
        """Test Fourier current potential coil cut against analytic solenoid."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([20, 3]),
            Z_lmn=jnp.array([0, -3]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=5,
        )
        # make a current potential corresponding a purely poloidal current
        G = 1e4  # net poloidal current
        correct_field = lambda R, phi, Z: jnp.array([[0, mu_0 * G / 2 / jnp.pi / R, 0]])

        field = FourierCurrentPotentialField(
            I=0,
            G=-G,
            sym_Phi="sin",
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=surface.NFP,
        )
        coils = field.to_CoilSet(10, stell_sym=True, show_plots=True).to_FourierXYZ(
            N=2, grid=LinearGrid(N=8), check_intersection=False
        )

        np.testing.assert_allclose(
            coils.compute_magnetic_field(
                [20.0, 0, 0],
            ),
            correct_field(20.0, 0, 0),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            coils.compute_magnetic_field([20.0, np.pi / 4, 0]),
            correct_field(20.0, np.pi / 4, 0),
            atol=1e-8,
            rtol=1e-8,
        )
        return plt.gcf()

    @pytest.mark.unit
    def test_fourier_current_potential_field_helical_coil_cut(self):
        """Test Fourier current potential helix coil cut against analytic solenoid."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([20, 3]),
            Z_lmn=jnp.array([0, -3]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        # make a current potential corresponding a helical current
        # with a sharp pitch (approximating a toroidal solenoid)
        G = 2e2  # net poloidal current
        I = 1
        correct_field = lambda R, phi, Z: jnp.array([[0, mu_0 * G / 2 / jnp.pi / R, 0]])

        field = FourierCurrentPotentialField(
            I=I,
            G=-G,
            sym_Phi="sin",
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )

        coils = field.to_CoilSet(1)

        np.testing.assert_allclose(
            coils.compute_magnetic_field([20.0, 0, 0], source_grid=LinearGrid(N=700)),
            correct_field(20.0, 0, 0),
            atol=2e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            coils.compute_magnetic_field(
                [20.0, np.pi / 4, 0], source_grid=LinearGrid(N=700)
            ),
            correct_field(20.0, np.pi / 4, 0),
            atol=2e-8,
            rtol=1e-8,
        )
        # check with opposite helicity current
        field = FourierCurrentPotentialField(
            I=I,
            G=G,
            sym_Phi="sin",
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )

        coils = field.to_CoilSet(1)

        np.testing.assert_allclose(
            -coils.compute_magnetic_field([20.0, 0, 0], source_grid=LinearGrid(N=700)),
            correct_field(20.0, 0, 0),
            atol=2e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            -coils.compute_magnetic_field(
                [20.0, np.pi / 4, 0], source_grid=LinearGrid(N=700)
            ),
            correct_field(20.0, np.pi / 4, 0),
            atol=2e-8,
            rtol=1e-8,
        )

    @pytest.mark.unit
    def test_fourier_current_potential_vector_potential(self):
        """Test Fourier current potential vector potential against analytic result."""
        R0 = 10
        a = 1
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([R0, a]),
            Z_lmn=jnp.array([0, -a]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )

        basis = DoubleFourierSeries(M=2, N=2, sym="sin")
        phi_mn = np.ones((basis.num_modes,))
        # make a current potential corresponding a purely poloidal current
        G = 100  # net poloidal current

        # test the loop integral of A around a curve encompassing the torus
        # against the analytic result for flux in an ideal toroidal solenoid
        ## expression for flux inside of toroidal solenoid of radius a
        prefactors = mu_0 * G / 2 / jnp.pi
        correct_flux = -2 * np.pi * prefactors * (np.sqrt(R0**2 - a**2) - R0)

        curve = FourierXYZCurve()  # curve to integrate A over
        curve_grid = LinearGrid(zeta=20)
        curve_data = curve.compute(["x", "x_s"], grid=curve_grid)
        curve_data_rpz = curve.compute(["x", "x_s"], grid=curve_grid, basis="rpz")

        field = FourierCurrentPotentialField(
            Phi_mn=phi_mn,
            modes_Phi=basis.modes[:, 1:],
            I=0,
            G=-G,  # to get a positive B_phi, we must put G negative
            # since -G is the net poloidal current on the surface
            # ( with  G=-(net_current) meaning that we have net_current
            # flowing poloidally (in clockwise direction) around torus)
            sym_Phi="sin",
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )
        surface_grid = LinearGrid(M=60, N=60, NFP=10)

        phi_mn = np.zeros((basis.num_modes,))

        field.Phi_mn = phi_mn

        field.change_resolution(3, 3)
        field.change_Phi_resolution(2, 2)

        A_xyz = field.compute_magnetic_vector_potential(
            curve_data["x"], basis="xyz", source_grid=surface_grid
        )
        A_rpz = field.compute_magnetic_vector_potential(
            curve_data_rpz["x"], basis="rpz", source_grid=surface_grid
        )

        # integrate to get the flux
        flux_xyz = jnp.sum(
            dot(A_xyz, curve_data["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )
        flux_rpz = jnp.sum(
            dot(A_rpz, curve_data_rpz["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )

        np.testing.assert_allclose(correct_flux, flux_xyz, rtol=1e-8)
        np.testing.assert_allclose(correct_flux, flux_rpz, rtol=1e-8)

        field.G = -2 * field.G
        field.I = 0

        A_xyz = field.compute_magnetic_vector_potential(
            curve_data["x"], basis="xyz", source_grid=surface_grid
        )
        A_rpz = field.compute_magnetic_vector_potential(
            curve_data_rpz["x"], basis="rpz", source_grid=surface_grid
        )

        # integrate to get the flux
        flux_xyz = jnp.sum(
            dot(A_xyz, curve_data["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )
        flux_rpz = jnp.sum(
            dot(A_rpz, curve_data_rpz["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )

        np.testing.assert_allclose(-2 * correct_flux, flux_xyz, rtol=1e-8)
        np.testing.assert_allclose(-2 * correct_flux, flux_rpz, rtol=1e-8)

        field = FourierCurrentPotentialField.from_surface(
            surface=surface,
            Phi_mn=phi_mn,
            modes_Phi=basis.modes[:, 1:],
            I=0,
            G=-G,
        )

        A_xyz = field.compute_magnetic_vector_potential(
            curve_data["x"], basis="xyz", source_grid=surface_grid
        )
        A_rpz = field.compute_magnetic_vector_potential(
            curve_data_rpz["x"], basis="rpz", source_grid=surface_grid
        )

        # integrate to get the flux
        flux_xyz = jnp.sum(
            dot(A_xyz, curve_data["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )
        flux_rpz = jnp.sum(
            dot(A_rpz, curve_data_rpz["x_s"], axis=-1) * curve_grid.spacing[:, 2]
        )

        np.testing.assert_allclose(correct_flux, flux_xyz, rtol=1e-8)
        np.testing.assert_allclose(correct_flux, flux_rpz, rtol=1e-8)

    @pytest.mark.unit
    def test_fourier_current_potential_field_symmetry(self):
        """Test Fourier current potential magnetic field Phi symmetry logic."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([10, 1]),
            Z_lmn=jnp.array([0, -1]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        basis = DoubleFourierSeries(M=2, N=2, sym="cos")
        phi_mn = np.ones((basis.num_modes,))
        # make a current potential corresponding a purely poloidal current
        field = FourierCurrentPotentialField(
            Phi_mn=phi_mn,
            modes_Phi=basis.modes[:, 1:],
            sym_Phi="cos",
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )
        assert field.sym_Phi == "cos"

        basis = DoubleFourierSeries(M=2, N=2, sym=False)
        phi_mn = np.ones((basis.num_modes,))
        # make a current potential corresponding a purely poloidal current
        field = FourierCurrentPotentialField(
            Phi_mn=phi_mn,
            modes_Phi=basis.modes[:, 1:],
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )
        assert field.sym_Phi is False

        # check error thrown if new array is different size than old
        with pytest.raises(ValueError):
            field.Phi_mn = np.ones((basis.num_modes + 1,))

    @pytest.mark.unit
    def test_io_fourier_current_field(self, tmpdir_factory):
        """Test that i/o works for FourierCurrentPotentialField."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([10, 1]),
            Z_lmn=jnp.array([0, -1]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        basis = DoubleFourierSeries(M=2, N=2, sym="cos")
        phi_mn = np.ones((basis.num_modes,))
        field = FourierCurrentPotentialField(
            Phi_mn=phi_mn,
            G=1000,
            I=-50,
            modes_Phi=basis.modes[:, 1:],
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )
        tmpdir = tmpdir_factory.mktemp("test_io_fourier_current_field")
        field.save(tmpdir.join("test_field.h5"))
        field2 = load(tmpdir.join("test_field.h5"))
        assert field.equiv(field2)

    @pytest.mark.unit
    def test_fourier_current_potential_field_asserts(self):
        """Test Fourier current potential magnetic field assert statements."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([10, 1]),
            Z_lmn=jnp.array([0, -1]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        basis = DoubleFourierSeries(M=2, N=2, sym="sin")
        phi_mn = np.ones((basis.num_modes,))
        # make a current potential corresponding a purely poloidal current
        G = 10  # net poloidal current

        field = FourierCurrentPotentialField(
            Phi_mn=phi_mn,
            modes_Phi=basis.modes[:, 1:],
            I=0,
            G=-G,
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )
        # check that we can change I,G correctly

        # with scalars
        field.I = 1
        field.G = 2
        np.testing.assert_allclose(field.I, 1)
        np.testing.assert_allclose(field.G, 2)
        # with 0D array
        field.I = np.array(1)
        field.G = np.array(2)
        np.testing.assert_allclose(field.I, 1)
        np.testing.assert_allclose(field.G, 2)
        # with 1D array of size 1
        field.I = np.array([1])
        field.G = np.array([2])
        np.testing.assert_allclose(field.I, 1)
        np.testing.assert_allclose(field.G, 2)

        # check that we can't set it with a size>1 array
        with pytest.raises(TypeError):
            field.I = np.array([1, 2])
        with pytest.raises(TypeError):
            field.G = np.array([1, 2])

        # check that we cant initialize with different size
        # Phi_mn and Phi_modes arrays
        with pytest.raises(AssertionError):
            field = FourierCurrentPotentialField(
                Phi_mn=phi_mn[0:-1],  # too short by 1
                modes_Phi=basis.modes[:, 1:],
                I=0,
                G=-G,
                R_lmn=surface.R_lmn,
                Z_lmn=surface.Z_lmn,
                modes_R=surface._R_basis.modes[:, 1:],
                modes_Z=surface._Z_basis.modes[:, 1:],
                NFP=10,
            )

    @pytest.mark.unit
    def test_change_Phi_basis_fourier_current_field(self):
        """Test that change_Phi_resolution works for FourierCurrentPotentialField."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([10, 1]),
            Z_lmn=jnp.array([0, -1]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        M = N = 2
        basis = DoubleFourierSeries(M=M, N=N, NFP=surface.NFP, sym="cos")

        phi_mn = np.ones((basis.num_modes,))
        field = FourierCurrentPotentialField(
            Phi_mn=phi_mn,
            modes_Phi=basis.modes[:, 1:],
            sym_Phi="cos",
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )

        np.testing.assert_allclose(abs(np.max(field.Phi_basis.modes[:, 1:])), M)
        np.testing.assert_allclose(field.Phi_basis.modes, basis.modes)
        assert field.Phi_basis.sym == "cos"

        M = N = 5
        basis = DoubleFourierSeries(M=M, N=N, NFP=surface.NFP, sym="cos")
        field.change_Phi_resolution(M=M, N=N)

        np.testing.assert_allclose(abs(np.max(field.Phi_basis.modes[:, 1:])), M)
        np.testing.assert_allclose(field.Phi_basis.modes, basis.modes)
        assert field.Phi_basis.sym == "cos"

        M = 7
        N = 9
        basis = DoubleFourierSeries(M=M, N=N, NFP=surface.NFP, sym="cos")
        field.change_Phi_resolution(M=M, N=N)

        np.testing.assert_allclose(abs(np.max(field.Phi_basis.modes[:, 1])), M)
        np.testing.assert_allclose(abs(np.max(field.Phi_basis.modes[:, 2])), N)
        np.testing.assert_allclose(field.Phi_basis.modes, basis.modes)
        assert field.Phi_basis.sym == "cos"

        M = 3
        N = 3
        basis = DoubleFourierSeries(M=M, N=N, NFP=surface.NFP, sym="sin")
        field.change_Phi_resolution(M=M, N=N, sym_Phi="sin")

        np.testing.assert_allclose(abs(np.max(field.Phi_basis.modes[:, 1])), M)
        np.testing.assert_allclose(abs(np.max(field.Phi_basis.modes[:, 2])), N)
        np.testing.assert_allclose(field.Phi_basis.modes, basis.modes)
        assert field.Phi_basis.sym == "sin"

    @pytest.mark.unit
    def test_init_Phi_mn_fourier_current_field(self):
        """Test initial Phi_mn size is correct for FourierCurrentPotentialField."""
        surface = FourierRZToroidalSurface(
            R_lmn=jnp.array([10, 1]),
            Z_lmn=jnp.array([0, -1]),
            modes_R=jnp.array([[0, 0], [1, 0]]),
            modes_Z=jnp.array([[0, 0], [-1, 0]]),
            NFP=10,
        )
        init_modes = np.array([[1, 1], [3, 3]])
        init_coeffs = np.array([1, 1])

        field = FourierCurrentPotentialField(
            Phi_mn=init_coeffs,
            modes_Phi=init_modes,
            R_lmn=surface.R_lmn,
            Z_lmn=surface.Z_lmn,
            modes_R=surface._R_basis.modes[:, 1:],
            modes_Z=surface._Z_basis.modes[:, 1:],
            NFP=10,
        )
        assert field.Phi_mn.size == field.Phi_basis.num_modes
        inds_nonzero = []
        for coef, modes in zip(init_coeffs, init_modes):
            ind = field.Phi_basis.get_idx(M=modes[0], N=modes[1])
            assert coef == field.Phi_mn[ind]
            inds_nonzero.append(ind)
        inds_zero = np.setdiff1d(np.arange(field.Phi_basis.num_modes), inds_nonzero)
        np.testing.assert_allclose(field.Phi_mn[inds_zero], 0)
        # ensure can compute field at a point without incompatible size error
        field.compute_magnetic_field([10.0, 0, 0])

    @pytest.mark.unit
    def test_fourier_current_potential_field_coil_cut_warnings(self):
        """Test Fourier current potential coil cut method warning."""
        curr = 1e4
        # with this choice of Phi_mn, the constant Phi contours
        # move so much that they intersect the boundaries of where we
        # plot them, that should return a warning
        # TODO: When we switch from the visual coil cutting method, remove this
        field = FourierCurrentPotentialField(
            I=curr,
            G=curr,
            Phi_mn=np.array([-4 * curr / 13]),
            modes_Phi=np.array([[-1, 0]]),
        )

        with pytest.warns(
            UserWarning,
            match="Detected",
        ):
            field.to_CoilSet(2)

    @pytest.mark.slow
    @pytest.mark.unit
    def test_spline_field(self, tmpdir_factory):
        """Test accuracy of spline magnetic field."""
        field1 = ScalarPotentialField(phi_lm, args, NFP=5)
        R = np.linspace(0.5, 1.5, 20)
        Z = np.linspace(-1.5, 1.5, 20)
        p = np.linspace(0, 2 * np.pi / 5, 40)
        # add source_grid here just for code coverage
        field2 = SplineMagneticField.from_field(
            field1, R, p, Z, source_grid=LinearGrid(N=1)
        )
        # this is just to test the logic when
        # compute_vector_potential returns a ValueError
        _ = SplineMagneticField.from_field(field2, R, p, Z, source_grid=LinearGrid(N=1))

        np.testing.assert_allclose(
            field1([1.0, 1.0, 1.0]), field2([1.0, 1.0, 1.0]), rtol=1e-2, atol=1e-2
        )

        extcur = [4700.0, 1000.0]
        mgrid = "tests/inputs/mgrid_test.nc"
        field3 = SplineMagneticField.from_mgrid(mgrid, extcur)
        # test saving and loading from mgrid
        tmpdir = tmpdir_factory.mktemp("spline_mgrid_with_A")
        path = tmpdir.join("spline_mgrid_with_A.nc")
        field3.save_mgrid(
            path,
            Rmin=np.min(field3._R),
            Rmax=np.max(field3._R),
            Zmin=np.min(field3._Z),
            Zmax=np.max(field3._Z),
            nR=field3._R.size,
            nZ=field3._Z.size,
            nphi=field3._phi.size,
            # just to test the source_grid function, the
            # field is independent of source_Grid
            source_grid=LinearGrid(N=0),
        )
        # no need for extcur b/c is saved in "raw" format, no need to scale again
        field4 = SplineMagneticField.from_mgrid(path)
        attrs_4d = ["_AR", "_Aphi", "_AZ", "_BR", "_Bphi", "_BZ"]
        for attr in attrs_4d:
            np.testing.assert_allclose(
                (getattr(field3, attr) * np.array(extcur)).sum(axis=-1),
                getattr(field4, attr).squeeze(),
                err_msg=attr,
            )
        attrs_3d = ["_R", "_phi", "_Z"]
        for attr in attrs_3d:
            np.testing.assert_allclose(getattr(field3, attr), getattr(field4, attr))

        r = 0.70
        p = 0
        z = 0
        # use finite diff derivatives to check A accuracy
        tfield_A = lambda R, phi, Z: field3.compute_magnetic_vector_potential(
            jnp.vstack([R, phi, Z]).T
        )
        funR = lambda x: tfield_A(x, p, z)
        funP = lambda x: tfield_A(r, x, z)
        funZ = lambda x: tfield_A(r, p, x)

        ap = tfield_A(r, p, z)[:, 1]

        # these are the gradients of each component of A
        dAdr = Derivative.compute_jvp(funR, 0, (jnp.ones_like(r),), r)
        dAdp = Derivative.compute_jvp(funP, 0, (jnp.ones_like(p),), p)
        dAdz = Derivative.compute_jvp(funZ, 0, (jnp.ones_like(z),), z)

        # form the B components with the appropriate combinations
        B2 = jnp.array(
            [
                dAdp[:, 2] / r - dAdz[:, 1],
                dAdz[:, 0] - dAdr[:, 2],
                dAdr[:, 1] + (ap - dAdp[:, 0]) / r,
            ]
        ).T

        np.testing.assert_allclose(
            field3([0.70, 0, 0]), np.array([[0, -0.671, 0.0858]]), rtol=1e-3, atol=1e-8
        )

        np.testing.assert_allclose(field3([0.70, 0, 0]), B2, rtol=1e-3, atol=5e-3)

        field3.currents *= 2
        np.testing.assert_allclose(
            field3([0.70, 0, 0]),
            2 * np.array([[0, -0.671, 0.0858]]),
            rtol=1e-3,
            atol=1e-8,
        )

    @pytest.mark.unit
    def test_spline_field_axisym(self):
        """Test computing axisymmetric magnetic field using SplineMagneticField."""
        extcur = [
            -1.370985e03,
            -1.609154e03,
            -2.751331e03,
            -2.524384e03,
            -3.435372e03,
            -3.466123e03,
            3.670919e03,
            3.450196e03,
            2.908027e03,
            3.404695e03,
            -4.148967e03,
            -4.294406e03,
            -3.059939e03,
            -2.990609e03,
            3.903818e03,
            3.727301e03,
            -3.049484e03,
            -3.086940e03,
            -1.488703e07,
            -2.430716e04,
            -2.380229e04,
        ]
        with pytest.warns(UserWarning):
            # user warning because saved mgrid no vector potential
            field = SplineMagneticField.from_mgrid(
                "tests/inputs/mgrid_d3d.nc", extcur=extcur
            )
        # make sure field is invariant to shift in phi
        B1 = field.compute_magnetic_field(np.array([1.75, 0.0, 0.0]))
        B2 = field.compute_magnetic_field(np.array([1.75, 1.0, 0.0]))
        np.testing.assert_allclose(B1, B2)

        # test the error when no vec pot values exist
        with pytest.raises(ValueError, match="no vector potential"):
            field.compute_magnetic_vector_potential(np.array([1.75, 0.0, 0.0]))

    @pytest.mark.unit
    def test_field_line_integrate(self):
        """Test field line integration."""
        # q=4, field line should rotate 1/4 turn after 1 toroidal transit
        # from outboard midplane to top center
        field = ToroidalMagneticField(2, 10) + PoloidalMagneticField(2, 10, 0.25)
        r0 = [10.001]
        z0 = [0.0]
        phis = [0, 2 * np.pi]
        r, z = field_line_integrate(r0, z0, phis, field)
        np.testing.assert_allclose(r[-1], 10, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z[-1], 0.001, rtol=1e-6, atol=1e-6)

    @pytest.mark.unit
    def test_field_line_integrate_jax_transforms(self, capsys):
        """Test field line integration is JAX transformable."""
        field = ToroidalMagneticField(2, 10) + PoloidalMagneticField(2, 10, 0.25)
        r0 = np.array([10.001])
        z0 = np.array([0.0])
        phis = np.array([0, 2 * np.pi])

        def default_terminating_event(t, y, args, **kwargs):
            return jnp.logical_or(y[0] < 0, y[0] > np.inf)

        # close over unhashable objects
        solver = Tsit5()
        saveat = SaveAt(ts=phis)
        stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)
        event = Event(default_terminating_event)
        adjoint = RecursiveCheckpointAdjoint()

        def fun0(
            r0,
            z0,
            phis,
            field,
            params,
            source_grid,
            max_steps,
            min_step_size,
            chunk_size,
            options,
        ):
            return _field_line_integrate(
                r0,
                z0,
                phis=phis,
                field=field,
                params=params,
                source_grid=source_grid,
                solver=solver,
                max_steps=max_steps,
                min_step_size=min_step_size,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                event=event,
                adjoint=adjoint,
                chunk_size=chunk_size,
                bs_chunk_size=None,
                options=options,
                return_aux=False,
            )

        # check if it is jittable
        r, z = jit(fun0, static_argnames="max_steps")(
            r0,
            z0,
            phis,
            field=field,
            params=None,
            source_grid=None,
            max_steps=1000,
            min_step_size=1e-8,
            chunk_size=None,
            options={},
        )
        np.testing.assert_allclose(r[-1], 10, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z[-1], 0.001, rtol=1e-6, atol=1e-6)

        # make sure that the function is not recompiled
        with jax.log_compiles():
            r, z = jit(fun0, static_argnames="max_steps")(
                np.array([10.002]),
                z0,
                phis,
                field=field,
                params=None,
                source_grid=None,
                max_steps=1000,
                min_step_size=1e-8,
                chunk_size=None,
                options={},
            )

        out = capsys.readouterr()
        assert out.out == ""
        # check the grad works
        # For toroidal field, r doesn't change with integration f(r) = r
        # so the derivative of the field line with respect to r should be 1
        fieldT = ToroidalMagneticField(2, 10)

        def fun(r0):
            r0 = jnp.array([r0])
            r, _ = _field_line_integrate(
                r0,
                z0,
                phis=phis,
                field=fieldT,
                params=None,
                source_grid=None,
                solver=solver,
                max_steps=1000,
                min_step_size=1e-8,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                event=event,
                adjoint=adjoint,
                chunk_size=None,
                bs_chunk_size=None,
                options={},
                return_aux=False,
            )
            return jnp.squeeze(r[-1])

        df_dr = jax.grad(jit(fun))(10.1)
        np.testing.assert_allclose(df_dr, 1, rtol=1e-8, atol=1e-8)

    @pytest.mark.unit
    def test_field_line_integrate_long(self):
        """Test field line integration for long distance along line."""
        # q=4, field line should rotate 1/4 turn after 1 toroidal transit
        # from outboard midplane to top center
        field = ToroidalMagneticField(2, 10) + PoloidalMagneticField(2, 10, 0.25)
        r0 = [10.001]
        z0 = [0.0]
        phis = [0, 2 * np.pi * 25]
        r, z = field_line_integrate(r0, z0, phis, field, solver=Dopri5())
        np.testing.assert_allclose(r[-1], 10, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z[-1], 0.001, rtol=1e-6, atol=1e-6)

    @pytest.mark.unit
    def test_field_line_integrate_early_terminate_default(self):
        """Test field line integration with default early termination criterion."""
        # q=4, field line should rotate 1/4 turn after 1 toroidal transit
        # from outboard midplane to top center
        # early terminate when it crosses towards the inboard side (R=10),
        field1 = ToroidalMagneticField(2, 10) + PoloidalMagneticField(2, 10, 0.25)
        # make a SplineMagneticField only defined in a tiny region around initial point
        field = SplineMagneticField.from_field(
            field=field1,
            R=np.linspace(10.0, 10.005, 40),
            phi=np.linspace(0, 2 * np.pi, 40),
            Z=np.linspace(-5e-3, 5e-3, 40),
            extrap=True,
        )
        r0 = [10.001]
        z0 = [0.0]
        phis = [0, 2 * np.pi, 2 * np.pi * 2]

        r, z = field_line_integrate(
            r0,
            z0,
            phis,
            field,
            bounds_R=(np.min(field._R), np.max(field._R)),
            bounds_Z=(np.min(field._Z), np.max(field._Z)),
            min_step_size=1e-2,
        )
        np.testing.assert_allclose(r[1], 10, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z[1], 0.001, rtol=1e-6, atol=1e-6)
        # if early terinated, the values at the un-integrated phi points are inf
        assert np.isnan(r[-1])
        assert np.isnan(z[-1])

    @pytest.mark.unit
    def test_Bnormal_calculation(self):
        """Tests Bnormal calculation for simple toroidal field."""
        tfield = ToroidalMagneticField(2, 1)
        surface = get("DSHAPE").surface
        Bnorm, _ = tfield.compute_Bnormal(surface)
        # should have 0 Bnormal because surface is axisymmetric
        np.testing.assert_allclose(Bnorm, 0, atol=1e-14)

    @pytest.mark.unit
    def test_Bnormal_save_and_load_DSHAPE(self, tmpdir_factory):
        """Tests Bnormal save/load for simple toroidal field with DSHAPE."""
        ### test on simple field first with tokamak
        tmpdir = tmpdir_factory.mktemp("BNORM_files")
        path = tmpdir.join("BNORM_desc.txt")
        tfield = ToroidalMagneticField(2, 1)
        eq = get("DSHAPE")
        grid = LinearGrid(rho=np.array(1.0), M=20, N=20, NFP=eq.NFP)
        x = eq.surface.compute("x", grid=grid, basis="rpz")["x"]
        Bnorm, x_from_Bnorm = tfield.compute_Bnormal(
            eq.surface, eval_grid=grid, source_grid=grid, basis="rpz"
        )

        # make sure x calculation is the same
        np.testing.assert_allclose(x[:, 0], x_from_Bnorm[:, 0], atol=1e-16)
        np.testing.assert_allclose(x[:, 2], x_from_Bnorm[:, 2], atol=1e-16)

        np.testing.assert_allclose(
            x[:, 1] % (2 * np.pi), x_from_Bnorm[:, 1] % (2 * np.pi), atol=1e-16
        )

        # should have 0 Bnormal because surface is axisymmetric
        np.testing.assert_allclose(Bnorm, 0, atol=1e-14)

        tfield.save_BNORM_file(eq, path, scale_by_curpol=False)
        Bnorm_from_file = read_BNORM_file(path, eq, grid, scale_by_curpol=False)
        np.testing.assert_allclose(Bnorm, Bnorm_from_file, atol=1e-14)

        # check that loading/saving with scale_by_curpol true
        # but no eq passed raises error
        with pytest.raises(RuntimeError):
            Bnorm_from_file = read_BNORM_file(path, eq.surface, grid)
        with pytest.raises(RuntimeError):
            tfield.save_BNORM_file(eq.surface, path)

    @pytest.mark.unit
    def test_Bnormal_save_and_load_HELIOTRON(self, tmpdir_factory):
        """Tests Bnormal save/load for simple toroidal field with HELIOTRON."""
        # test on simple field with stellarator
        tmpdir = tmpdir_factory.mktemp("BNORM_files")
        path = tmpdir.join("BNORM_desc_heliotron.txt")
        tfield = ToroidalMagneticField(2, 1)
        eq = get("HELIOTRON")
        grid = LinearGrid(rho=np.array(1.0), M=20, N=20, NFP=eq.NFP)
        x = eq.surface.compute("x", grid=grid, basis="xyz")["x"]
        Bnorm, x_from_Bnorm = tfield.compute_Bnormal(
            eq.surface, eval_grid=grid, basis="xyz"
        )

        # make sure x calculation is the same
        np.testing.assert_allclose(x, x_from_Bnorm, atol=1e-16)

        tfield.save_BNORM_file(eq, path, 40, 40)
        Bnorm_from_file = read_BNORM_file(path, eq, grid)
        np.testing.assert_allclose(Bnorm, Bnorm_from_file, atol=1e-8)

        asym_surf = FourierRZToroidalSurface(sym=False)
        with pytest.raises(AssertionError, match="sym"):
            Bnorm_from_file = read_BNORM_file(path, asym_surf, grid)

    @pytest.mark.unit
    def test_spline_field_jit(self):
        """Test that the spline field can be passed to a jitted function."""
        extcur = [4700.0, 1000.0]
        mgrid = "tests/inputs/mgrid_test.nc"
        field = SplineMagneticField.from_mgrid(mgrid, extcur)

        x = jnp.array([0.70, 0, 0])

        @jit
        def foo(field, x):
            return field.compute_magnetic_field(x)

        np.testing.assert_allclose(
            foo(field, x), np.array([[0, -0.671, 0.0858]]), rtol=1e-3, atol=1e-8
        )

    @pytest.mark.unit
    def test_mgrid_io(self, tmpdir_factory):
        """Test saving to and loading from an mgrid file."""
        tmpdir = tmpdir_factory.mktemp("mgrid_dir")
        path = tmpdir.join("mgrid.nc")

        # field to test on
        toroidal_field = ToroidalMagneticField(B0=1, R0=5)
        poloidal_field = PoloidalMagneticField(B0=1, R0=5, iota=2 / np.pi)
        vertical_field = VerticalMagneticField(B0=0.2)
        save_field = toroidal_field + poloidal_field + vertical_field

        # save and load mgrid file
        Rmin = 3
        Rmax = 7
        Zmin = -2
        Zmax = 2
        with pytest.raises(NotImplementedError):
            # Raises error because poloidal field has no vector potential
            # and so cannot save the vector potential
            save_field.save_mgrid(path, Rmin, Rmax, Zmin, Zmax)
        save_field.save_mgrid(path, Rmin, Rmax, Zmin, Zmax, save_vector_potential=False)
        with pytest.warns(UserWarning):
            # user warning because saved mgrid has no vector potential
            # and so cannot load the vector potential
            load_field = SplineMagneticField.from_mgrid(path)

        # check that the fields are the same
        num_nodes = 50
        grid = np.array(
            [
                np.linspace(Rmin, Rmax, num_nodes),
                np.linspace(0, 2 * np.pi, num_nodes, endpoint=False),
                np.linspace(Zmin, Zmax, num_nodes),
            ]
        ).T
        B_saved = save_field.compute_magnetic_field(grid)
        B_loaded = load_field.compute_magnetic_field(grid)
        np.testing.assert_allclose(B_loaded, B_saved, rtol=1e-6)

    @pytest.mark.unit
    def test_omnigenous_field_change_resolution_B(self):
        """Test OmnigenousField.change_resolution() of the B_lm parameters."""
        L_B_old = 1
        L_B_new = 2
        M_B_old = 3
        M_B_new = 6
        NFP = 4
        field = OmnigenousField(
            L_B=L_B_old,
            M_B=M_B_old,
            L_x=0,
            M_x=0,
            N_x=0,
            NFP=NFP,
            helicity=(0, NFP),
            B_lm=np.array([0.9, 1.0, 1.1, 0.2, 0.05, -0.2]),
        )
        grid_axis = LinearGrid(rho=[0.0], M=50)
        grid_half = LinearGrid(rho=[0.5], M=50)
        grid_lcfs = LinearGrid(rho=[1.0], M=50)
        B_axis_lowres = field.compute("|B|", grid=grid_axis)["|B|"]
        B_half_lowres = field.compute("|B|", grid=grid_half)["|B|"]
        B_lcfs_lowres = field.compute("|B|", grid=grid_lcfs)["|B|"]
        field.change_resolution(L_B=L_B_new, M_B=M_B_new)
        B_axis_highres = field.compute("|B|", grid=grid_axis)["|B|"]
        B_half_highres = field.compute("|B|", grid=grid_half)["|B|"]
        B_lcfs_highres = field.compute("|B|", grid=grid_lcfs)["|B|"]
        np.testing.assert_allclose(B_axis_lowres, B_axis_highres, rtol=6e-3)
        np.testing.assert_allclose(B_half_lowres, B_half_highres, rtol=3e-3)
        np.testing.assert_allclose(B_lcfs_lowres, B_lcfs_highres, rtol=4e-3)

    @pytest.mark.unit
    def test_solve_current_potential_warnings_and_errors(self):
        """Test solve current potential warnings/errors."""
        field = FourierCurrentPotentialField(I=0, G=1, sym_Phi="sin")
        eq = get("SOLOVEV")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(L=1, M=1, N=1, L_grid=1, M_grid=1, N_grid=1)
        with pytest.raises(ValueError, match="length"):
            solve_regularized_surface_current(field, eq, current_helicity=(1, 1, 1))
        with pytest.raises(ValueError, match="integer"):
            solve_regularized_surface_current(field, eq, current_helicity=(1.2, 1))
        with pytest.raises(ValueError, match="regularization"):
            solve_regularized_surface_current(
                field, eq, regularization_type="not a valid option"
            )
        with pytest.raises(ValueError, match="Expected Fourier"):
            solve_regularized_surface_current(ToroidalMagneticField(1, 1), eq)
        with pytest.raises(AssertionError, match="Expected MagneticField"):
            solve_regularized_surface_current(field, eq, external_field=eq)
        field = FourierCurrentPotentialField(I=0, G=1, sym_Phi="cos")
        grid = LinearGrid(M=1, N=1)
        # nested with pytest.warns, if a warning is not detected it is
        # re-emitted and goes through the higher level context,
        #  this lets us test 3 different warnings with one fxn
        # call here
        with pytest.warns(UserWarning, match="Detected"):
            with pytest.warns(UserWarning, match="Pressure"):
                with pytest.warns(UserWarning, match="Current"):
                    solve_regularized_surface_current(
                        field,
                        eq,
                        eval_grid=grid,
                        source_grid=grid,
                        vc_source_grid=grid,
                        verbose=0,
                        vacuum=True,
                    )


@pytest.mark.unit
def test_dommaschk_CN_CD_m_0():
    """Test of CD_m_k and CN_m_k when k=0."""
    # based off eqn 8 and 9 of Dommaschk paper
    # https://doi.org/10.1016/0010-4655(86)90109-8
    for m in range(1, 6):
        # test of CD_m_k based off eqn 8
        R = np.linspace(0.1, 1, 100)
        res1 = CD_m_k(R, m, 0)
        res2 = 0.5 * (R**m + R ** (-m))
        np.testing.assert_allclose(res1, res2)

        # test of CN_m_k based off eqn 9
        res1 = CN_m_k(R, m, 0)
        res2 = 0.5 * (R**m - R ** (-m)) / m
        np.testing.assert_allclose(res1, res2, atol=1e-15)


@pytest.mark.unit
def test_dommaschk_field_errors():
    """Test the assert statements of the DommaschkField function."""
    ms = [1]
    ls = [1]
    a_arr = [1]
    b_arr = [1]
    c_arr = [1]
    d_arr = [1, 1]  # length is not equal to the rest
    with pytest.raises(AssertionError, match="size"):
        DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr)
    d_arr = [1]  # test with incorrect NFP
    with pytest.raises(AssertionError, match="desired NFP"):
        DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr, NFP=2)
    ms = [-1]  # negative mode number
    with pytest.raises(AssertionError, match=">= 0"):
        DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr)


@pytest.mark.unit
def test_dommaschk_radial_field():
    """Test the Dommaschk potential for a pure toroidal (Bphi~1/R) field."""
    phi = np.linspace(0, 2 * np.pi, 10)
    R = np.linspace(0.1, 1.5, 50)
    Z = np.linspace(-0.05, 0.5, 50)
    R, phi, Z = np.meshgrid(R, phi, Z)
    coords = np.vstack((R.flatten(), phi.flatten(), Z.flatten())).T

    ms = [0]
    ls = [0]
    a_arr = [0]
    b_arr = [0]
    c_arr = [0]
    d_arr = [0]
    B = DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr)
    B_dom = B.compute_magnetic_field(coords)
    np.testing.assert_allclose(B_dom[:, 0], 0)
    np.testing.assert_array_equal(B_dom[:, 1], 1 / R.flatten())
    np.testing.assert_allclose(B_dom[:, 2], 0)


@pytest.mark.unit
def test_dommaschk_vertical_field():
    """Test the Dommaschk potential for a 1/R toroidal + pure vertical field."""
    phi = jnp.linspace(0, 2 * jnp.pi, 10)
    R = jnp.linspace(0.1, 1.5, 50)
    Z = jnp.linspace(-0.5, 0.5, 50)
    R, phi, Z = jnp.meshgrid(R, phi, Z)
    coords = jnp.vstack((R.flatten(), phi.flatten(), Z.flatten())).T

    ms = [0]
    ls = [1]
    a_arr = [1]
    b_arr = [0]
    c_arr = [0]
    d_arr = [0]
    B = DommaschkPotentialField(ms, ls, a_arr, b_arr, c_arr, d_arr)
    B_dom = B.compute_magnetic_field(coords)
    ones = jnp.ones_like(B_dom[:, 0])
    np.testing.assert_allclose(B_dom[:, 0], 0, atol=1e-14)
    np.testing.assert_allclose(B_dom[:, 1], 1 / R.flatten(), atol=1e-14)
    np.testing.assert_allclose(B_dom[:, 2], ones, atol=5e-15)


@pytest.mark.unit
def test_dommaschk_fit_vertical_and_toroidal_field():
    """Test the Dommaschk potential fit for a toroidal and a vertical field."""
    phi = np.linspace(0, 2 * np.pi, 3)
    R = np.linspace(0.1, 1.5, 3)
    Z = np.linspace(-0.5, 0.5, 3)
    R, phi, Z = np.meshgrid(R, phi, Z)
    coords = np.vstack((R.flatten(), phi.flatten(), Z.flatten())).T

    max_l = 1
    max_m = 1
    B0 = 2  # scale strength for to 1/R field to fit
    B0_Z = 1  # scale strength for to uniform vertical field to fit
    field = ToroidalMagneticField(B0=B0, R0=1) + VerticalMagneticField(B0=B0_Z)

    B = DommaschkPotentialField.fit_magnetic_field(
        field, coords, max_m, max_l, sym=True, NFP=2
    )

    B_dom = B.compute_magnetic_field(coords)
    np.testing.assert_allclose(B_dom[:, 0], 0, atol=1e-13)
    np.testing.assert_allclose(B_dom[:, 1], B0 / R.flatten(), atol=1e-13)
    np.testing.assert_allclose(B_dom[:, 2], B0_Z, atol=1e-13)

    np.testing.assert_allclose(B._params["B0"], B0)

    # only nonzero coefficient of the field should be the B0 and a_ml = a_01
    np.testing.assert_allclose(B._params["B0"], B0, atol=1e-13)
    for coef, m, l in zip(
        B._params["a_arr"], B._full_params["ms"], B._full_params["ls"]
    ):
        if m == 0 and l == 1:
            np.testing.assert_allclose(coef, B0_Z)
        else:
            np.testing.assert_allclose(coef, 0, atol=1e-13)
    for name in ["b_arr", "c_arr", "d_arr"]:
        np.testing.assert_allclose(B._params[name], 0, atol=1e-13)


@pytest.mark.unit
def test_domm_field_is_nonzero_and_continuous_across_Z_0():
    """Test that at Z=0 the Bphi of domm field is not discontinuous."""
    # following field should, at constant R and Z, have Bphi
    # be nonzero everywhere.
    # related to issue noted in PRs #966 and #961, where
    # before the fix in #966 was implemented the field Bphi
    # would drop to 0 discontinuously at Z=0.
    new_field = DommaschkPotentialField(1, 1, 10, 0, 0, 9, B0=0)
    Zs = np.linspace(-0.05, 0.05, 101)
    Rs = 1.1 * np.ones_like(Zs)
    phis = 0 * np.ones_like(Zs)

    Bnew = new_field.compute_magnetic_field(np.vstack([Rs, phis, Zs]).T)

    assert not np.any(np.isclose(Bnew[:, 1], 0))


@pytest.mark.mpl_image_compare(remove_text=True, tolerance=10)
@pytest.mark.unit
def test_domm_W7AS():
    """Test getting flux surfaces from W7-AS Dommaschk coefficients."""
    # W7-AS
    # The coefficients are taken from IPP-Report IPP_0_48 by Dommaschk et al.
    B0 = 1.00
    # fmt: off
    # toroidal harmonics
    ms = [ 0, 5, 10, 15,
        0, 5, 10, 15,
        0, 5, 10, 15,
        0, 5, 10, 15,
        0, 5, 10, 15,
        0, 5, 10, 15,
        0, 5, 10, 15 ]
    # poloidal harmonics
    ls = [ 0, 0,  0,  0,
        1, 1,  1,  1,
        2, 2,  2,  2,
        3, 3,  3,  3,
        4, 4,  4,  4,
        5, 5,  5,  5,
        6, 6,  6,  6 ]
    # Arrays:
    # m      0          5         10         15           # l
    a_w7as=[0.0      , 0.0      , 0.0       , 0.0,        # 0
            0.0094324, 0.0112792, 0.0147238 ,-0.00100904, # 1
            0.0      , 0.0      , 0.0       , 0.0,        # 2
            0.0410242,-4.15263  ,-3.33006   , 0.0359866,  # 3
            0.0      , 0.0      , 0.0       , 0.0,        # 4
            -29.9792 ,-135.852  , 60.1416   ,-215.314 ,   # 5
            0.0      , 0.0      , 0.0       , 0.0       ] # 6

    #        0          5         10         15
    b_w7as=[0.0      ,-0.0121413,0.000471873, 4.58071E-05,# 0
            0.0      , 0.0      , 0.0       , 0.0     ,   # 1
            0.0      , 1.25878  , 0.237015  , -0.0403158, # 2
            0.0      , 0.0      , 0.0      , 0.0     ,    # 3
            0.0      , 4.06992  , 38.1763  , 15.0701 ,    # 4
            0.0      , 0.0      , 0.0      , 0.0     ,    # 5
            0.0      ,-3192.22  , 376.945  , 2621.02  ]   # 6

    #        0          5         10         15
    c_w7as=[0.0      , 0.0      , 0.0      , 0.0,         # 0
            0.0      , 0.0      , 0.0      , 0.0     ,    # 1
            0.255341 , 1.35643  , 0.351854 , 0.0165178,   # 2
            0.0      , 0.0      , 0.0      , 0.0     ,    # 3
            -14.022  , -9.04059  , 40.75   , 5.76041 ,    # 4
            0.0      , 0.0      , 0.0      , 0.0     ,    # 5
            -3877.26 , -17583.9 ,-20976.   , 937.2     ]  # 6

    #        0          5         10         15
    d_w7as=[0.0      , 0.0     , 0.0      , 0.0,          # 0
            0.0      , 0.190922,-0.0105725, 0.00335741,   # 1
            0.0      , 0.0     , 0.0      , 0.0     ,     # 2
            0.0      , 0.163836, 1.60217  ,-0.136241,     # 3
            0.0      , 0.0     , 0.0      , 0.0     ,     # 4
            0.0      , 120.441 , 344.756  , 193.934 ,     # 5
            0.0      , 0.0     , 0.0      , 0.0       ]   # 6
    # fmt: on
    Bw7as_dom = DommaschkPotentialField(
        B0=B0,
        a_arr=a_w7as,
        b_arr=b_w7as,
        c_arr=c_w7as,
        d_arr=d_w7as,
        ms=ms,
        ls=ls,
        NFP=5,
    )
    ntransit = 300  # how many toroidal transits to trace
    NFP = 5
    r0 = np.linspace(0.88245, 1.0, 11)
    z0 = np.zeros_like(r0)
    fig, _ = poincare_plot(
        field=Bw7as_dom,
        R0=r0,  # initial R positions for the field line trajectories
        Z0=z0,  # initial R positions for the field line trajectories
        ntransit=ntransit,  # number of toroidal transits we want to trace
        NFP=NFP,
        # bounds_R and bounds_Z set a cylindrical shell where,
        # if the B trajectory exits, it will stop the integration.
        # this saves time by not tracking trajectories which are going off to infinity
        bounds_R=[0.75, 1.25],
        bounds_Z=[-0.25, 0.25],
        size=0.10,  # markersize for the plotted points
        marker="d",
    )
    return fig
