"""Tests for magnetic field classes."""

import numpy as np
import pytest
from scipy.constants import mu_0

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.compute import rpz2xyz_vec, xyz2rpz_vec
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.magnetic_fields import (
    CurrentPotentialField,
    FourierCurrentPotentialField,
    PoloidalMagneticField,
    ScalarPotentialField,
    SplineMagneticField,
    ToroidalMagneticField,
    VerticalMagneticField,
    field_line_integrate,
    read_BNORM_file,
)


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
        np.testing.assert_allclose(tfield([1, 0, 0]), [[0, 2, 0]])
        np.testing.assert_allclose((4 * tfield)([2, 0, 0]), [[0, 4, 0]])
        np.testing.assert_allclose((tfield + vfield)([1, 0, 0]), [[0, 2, 1]])
        np.testing.assert_allclose(
            (tfield + vfield - pfield)([1, 0, 0.1]), [[0.4, 2, 1]]
        )

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

        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, 0, 0], grid=surface_grid),
            correct_field(10.0, 0, 0),
            atol=1e-16,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, np.pi / 4, 0], grid=surface_grid),
            correct_field(10.0, np.pi / 4, 0),
            atol=1e-16,
            rtol=1e-8,
        )

        field.G = -2 * G
        field.I = 0

        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, 0, 0], grid=surface_grid),
            correct_field(10.0, 0, 0) * 2,
            atol=1e-16,
            rtol=1e-8,
        )
        # use default grid
        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, np.pi / 4, 0], grid=None),
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
            field.compute_magnetic_field([10.0, 0, 0], grid=surface_grid),
            correct_field(10.0, 0, 0),
            atol=1e-16,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            field.compute_magnetic_field([10.0, np.pi / 4, 0], grid=surface_grid),
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

    @pytest.mark.slow
    @pytest.mark.unit
    def test_spline_field(self):
        """Test accuracy of spline magnetic field."""
        field1 = ScalarPotentialField(phi_lm, args)
        R = np.linspace(0.5, 1.5, 20)
        Z = np.linspace(-1.5, 1.5, 20)
        p = np.linspace(0, 2 * np.pi / 5, 40)
        field2 = SplineMagneticField.from_field(field1, R, p, Z, period=2 * np.pi / 5)

        np.testing.assert_allclose(
            field1([1.0, 1.0, 1.0]), field2([1.0, 1.0, 1.0]), rtol=1e-2, atol=1e-2
        )

        extcur = [4700.0, 1000.0]
        mgrid = "tests/inputs/mgrid_test.nc"
        field3 = SplineMagneticField.from_mgrid(mgrid, extcur)

        np.testing.assert_allclose(
            field3([0.70, 0, 0]), [[0, -0.671, 0.0858]], rtol=1e-3, atol=1e-8
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
        field = SplineMagneticField.from_mgrid(
            "tests/inputs/mgrid_d3d.nc", extcur=extcur
        )
        # make sure field is invariant to shift in phi
        B1 = field.compute_magnetic_field(np.array([1.75, 0.0, 0.0]))
        B2 = field.compute_magnetic_field(np.array([1.75, 1.0, 0.0]))
        np.testing.assert_allclose(B1, B2)

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
    def test_Bnormal_calculation(self):
        """Tests Bnormal calculation for simple toroidal field."""
        tfield = ToroidalMagneticField(2, 1)
        surface = get("DSHAPE").surface
        Bnorm, _ = tfield.compute_Bnormal(surface)
        # should have 0 Bnormal because surface is axisymmetric
        np.testing.assert_allclose(Bnorm, 0, atol=3e-15)

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
        ### test on simple field with stellarator
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
