"""Tests for coils and coilsets."""

import shutil

import numpy as np
import pytest
import scipy
import scipy.constants

from desc.backend import jnp
from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    FourierRZCoil,
    FourierXYZCoil,
    MixedCoilSet,
    SplineXYZCoil,
    initialize_helical_coils,
    initialize_modular_coils,
    initialize_saddle_coils,
)
from desc.compute import get_params, get_transforms, rpz2xyz, xyz2rpz, xyz2rpz_vec
from desc.compute.geom_utils import copy_rpz_periods
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import FourierRZCurve, FourierRZToroidalSurface, FourierXYZCurve
from desc.grid import Grid, LinearGrid
from desc.io import load
from desc.magnetic_fields import SumMagneticField, VerticalMagneticField
from desc.objectives import LinkingCurrentConsistency
from desc.utils import dot


class TestCoil:
    """Tests for singular coil objects."""

    @pytest.mark.unit
    def test_biot_savart_all_coils(self):
        """Test biot-savart implementation against analytic formula."""
        coil_grid = LinearGrid(zeta=100, endpoint=False)

        R = 2
        y = 1
        I = 1e7

        By_true = scipy.constants.mu_0 / 2 * R**2 * I / (y**2 + R**2) ** (3 / 2)
        Bz_true = scipy.constants.mu_0 / 2 * R**2 * I / (y**2 + R**2) ** (3 / 2)

        B_true_xyz = np.atleast_2d([0, By_true, 0])
        grid_xyz = np.atleast_2d([10, y, 0])
        grid_rpz = xyz2rpz(grid_xyz)
        B_true_rpz_xy = xyz2rpz_vec(B_true_xyz, x=grid_xyz[:, 0], y=grid_xyz[:, 1])
        B_true_rpz_phi = xyz2rpz_vec(B_true_xyz, phi=grid_rpz[:, 1])

        # FourierXYZCoil
        coil = FourierXYZCoil(I)
        transforms = get_transforms(["x", "x_s", "ds"], coil, coil_grid)
        params = get_params(["x", "x_s", "ds"], coil)
        B_xyz = coil.compute_magnetic_field(
            grid_xyz,
            basis="xyz",
            source_grid=coil_grid,
            transforms=transforms,
            params=params,
        )
        B_rpz = coil.compute_magnetic_field(
            grid_rpz, basis="rpz", source_grid=coil_grid
        )
        np.testing.assert_allclose(
            B_true_xyz, B_xyz, rtol=1e-3, atol=1e-10, err_msg="Using FourierXYZCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_xy, B_rpz, rtol=1e-3, atol=1e-10, err_msg="Using FourierXYZCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_phi, B_rpz, rtol=1e-3, atol=1e-10, err_msg="Using FourierXYZCoil"
        )

        # SplineXYZCoil
        x = coil.compute("x", grid=coil_grid, basis="xyz")["x"]
        coil = SplineXYZCoil(I, X=x[:, 0], Y=x[:, 1], Z=x[:, 2])
        B_xyz = coil.compute_magnetic_field(
            grid_xyz, basis="xyz", source_grid=coil_grid
        )
        B_rpz = coil.compute_magnetic_field(
            grid_rpz, basis="rpz", source_grid=coil_grid
        )
        np.testing.assert_allclose(
            B_true_xyz, B_xyz, rtol=1e-3, atol=1e-10, err_msg="Using SplineXYZCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_xy, B_rpz, rtol=1e-3, atol=1e-10, err_msg="Using SplineXYZCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_phi, B_rpz, rtol=1e-3, atol=1e-10, err_msg="Using SplineXYZCoil"
        )

        # FourierPlanarCoil
        coil = FourierPlanarCoil(I)
        B_xyz = coil.compute_magnetic_field(
            grid_xyz, basis="xyz", source_grid=coil_grid
        )
        B_rpz = coil.compute_magnetic_field(
            grid_rpz, basis="rpz", source_grid=coil_grid
        )
        np.testing.assert_allclose(
            B_true_xyz, B_xyz, rtol=1e-3, atol=1e-10, err_msg="Using FourierPlanarCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_xy,
            B_rpz,
            rtol=1e-3,
            atol=1e-10,
            err_msg="Using FourierPlanarCoil",
        )
        np.testing.assert_allclose(
            B_true_rpz_phi,
            B_rpz,
            rtol=1e-3,
            atol=1e-10,
            err_msg="Using FourierPlanarCoil",
        )

        B_true_xyz = np.atleast_2d([0, 0, Bz_true])
        grid_xyz = np.atleast_2d([0, 0, y])
        grid_rpz = xyz2rpz(grid_xyz)
        B_true_rpz_xy = xyz2rpz_vec(B_true_xyz, x=grid_xyz[:, 0], y=grid_xyz[:, 1])
        B_true_rpz_phi = xyz2rpz_vec(B_true_xyz, phi=grid_rpz[:, 1])

        # FourierRZCoil
        coil = FourierRZCoil(I, R_n=np.array([R]), modes_R=np.array([0]))
        B_xyz = coil.compute_magnetic_field(
            grid_xyz, basis="xyz", source_grid=coil_grid
        )
        B_rpz = coil.compute_magnetic_field(
            grid_rpz, basis="rpz", source_grid=coil_grid
        )
        np.testing.assert_allclose(
            B_true_xyz, B_xyz, rtol=1e-3, atol=1e-10, err_msg="Using FourierRZCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_xy, B_rpz, rtol=1e-3, atol=1e-10, err_msg="Using FourierRZCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_phi, B_rpz, rtol=1e-3, atol=1e-10, err_msg="Using FourierRZCoil"
        )

        # FourierRZCoil with NFP>1
        coil = FourierRZCoil(I, R_n=np.array([R]), modes_R=np.array([0]), NFP=2)
        B_xyz = coil.compute_magnetic_field(grid_xyz, basis="xyz", source_grid=None)
        B_rpz = coil.compute_magnetic_field(grid_rpz, basis="rpz", source_grid=None)
        np.testing.assert_allclose(
            B_true_xyz, B_xyz, rtol=1e-3, atol=1e-10, err_msg="Using FourierRZCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_xy, B_rpz, rtol=1e-3, atol=1e-10, err_msg="Using FourierRZCoil"
        )
        np.testing.assert_allclose(
            B_true_rpz_phi, B_rpz, rtol=1e-3, atol=1e-10, err_msg="Using FourierRZCoil"
        )

    @pytest.mark.unit
    def test_biot_savart_vector_potential_all_coils(self):
        """Test biot-savart vec potential implementation against analytic formula."""
        coil_grid = LinearGrid(zeta=100, endpoint=False)

        R = 2
        y = 1
        I = 1e7

        A_true = np.atleast_2d([0, 0, 0])
        grid_xyz = np.atleast_2d([10, y, 0])
        grid_rpz = xyz2rpz(grid_xyz)

        def test(coil, grid_xyz, grid_rpz):
            A_xyz = coil.compute_magnetic_vector_potential(
                grid_xyz, basis="xyz", source_grid=coil_grid
            )
            A_rpz = coil.compute_magnetic_vector_potential(
                grid_rpz, basis="rpz", source_grid=coil_grid
            )
            np.testing.assert_allclose(
                A_true, A_xyz, rtol=1e-3, atol=1e-10, err_msg=f"Using {coil}"
            )
            np.testing.assert_allclose(
                A_true, A_rpz, rtol=1e-3, atol=1e-10, err_msg=f"Using {coil}"
            )
            np.testing.assert_allclose(
                A_true, A_rpz, rtol=1e-3, atol=1e-10, err_msg=f"Using {coil}"
            )

        # FourierXYZCoil
        coil = FourierXYZCoil(I)
        test(coil, grid_xyz, grid_rpz)

        # SplineXYZCoil
        x = coil.compute("x", grid=coil_grid, basis="xyz")["x"]
        coil = SplineXYZCoil(I, X=x[:, 0], Y=x[:, 1], Z=x[:, 2])
        test(coil, grid_xyz, grid_rpz)

        # FourierPlanarCoil
        coil = FourierPlanarCoil(I)
        test(coil, grid_xyz, grid_rpz)

        grid_xyz = np.atleast_2d([0, 0, y])
        grid_rpz = xyz2rpz(grid_xyz)

        # FourierRZCoil
        coil = FourierRZCoil(I, R_n=np.array([R]), modes_R=np.array([0]))
        test(coil, grid_xyz, grid_rpz)
        # test in a CoilSet
        coil2 = CoilSet(coil)
        test(coil2, grid_xyz, grid_rpz)
        # test in a MixedCoilSet
        coil3 = MixedCoilSet(coil2, coil, check_intersection=False)
        coil3[1].current = 0
        test(coil3, grid_xyz, grid_rpz)

    @pytest.mark.unit
    def test_biot_savart_vector_potential_integral_all_coils(self):
        """Test analytic expression of flux integral for all coils."""
        # taken from analytic benchmark in
        # "A Magnetic Diagnostic Code for 3D Fusion Equilibria", Lazerson 2013
        # find flux for concentric loops of varying radii to a circular coil

        coil_grid = LinearGrid(zeta=1000, endpoint=False)

        R = 1
        I = 1e7

        # analytic eqn for "A_phi" (phi is in dl direction for loop)
        def _A_analytic(r):
            # elliptic integral arguments must be k^2, not k,
            # error in original paper and apparently in Jackson EM book too.
            theta = np.pi / 2
            arg = R**2 + r**2 + 2 * r * R * np.sin(theta)
            term_1_num = I * R * scipy.constants.mu_0 / np.pi
            term_1_den = np.sqrt(arg)
            k_sqd = 4 * r * R * np.sin(theta) / arg
            term_2_num = (2 - k_sqd) * scipy.special.ellipk(
                k_sqd
            ) - 2 * scipy.special.ellipe(k_sqd)
            term_2_den = k_sqd
            return term_1_num * term_2_num / term_1_den / term_2_den

        # we only evaluate it at theta=np.pi/2 (b/c it is in spherical coords)
        rs = np.linspace(0.1, 3, 10, endpoint=True)
        N = 200
        curve_grid = LinearGrid(zeta=N)

        def test(
            coil, grid_xyz, grid_rpz, A_true_rpz, correct_flux, rtol=1e-10, atol=1e-12
        ):
            """Test that we compute the correct flux for the given coil."""
            A_xyz = coil.compute_magnetic_vector_potential(
                grid_xyz, basis="xyz", source_grid=coil_grid
            )
            A_rpz = coil.compute_magnetic_vector_potential(
                grid_rpz, basis="rpz", source_grid=coil_grid
            )
            flux_xyz = jnp.sum(
                dot(A_xyz, curve_data["x_s"], axis=-1) * curve_grid.spacing[:, 2]
            )
            flux_rpz = jnp.sum(
                dot(A_rpz, curve_data_rpz["x_s"], axis=-1) * curve_grid.spacing[:, 2]
            )

            np.testing.assert_allclose(
                correct_flux, flux_xyz, rtol=rtol, err_msg=f"Using {coil}"
            )
            np.testing.assert_allclose(
                correct_flux, flux_rpz, rtol=rtol, err_msg=f"Using {coil}"
            )
            np.testing.assert_allclose(
                A_true_rpz,
                A_rpz,
                rtol=rtol,
                atol=atol,
                err_msg=f"Using {coil}",
            )

        for r in rs:
            # A_phi is constant around the loop (no phi dependence)
            A_true_phi = _A_analytic(r) * np.ones(N)
            A_true_rpz = np.vstack(
                (np.zeros_like(A_true_phi), A_true_phi, np.zeros_like(A_true_phi))
            ).T
            correct_flux = np.sum(r * A_true_phi * 2 * np.pi / N)

            curve = FourierXYZCurve(
                X_n=[-r, 0, 0], Y_n=[0, 0, r], Z_n=[0, 0, 0]
            )  # flux loop to integrate A over

            curve_data = curve.compute(["x", "x_s"], grid=curve_grid, basis="xyz")
            curve_data_rpz = curve.compute(["x", "x_s"], grid=curve_grid, basis="rpz")

            grid_rpz = np.vstack(
                [
                    curve_data_rpz["x"][:, 0],
                    curve_data_rpz["x"][:, 1],
                    curve_data_rpz["x"][:, 2],
                ]
            ).T
            grid_xyz = rpz2xyz(grid_rpz)
            # FourierXYZCoil
            coil = FourierXYZCoil(I, X_n=[-R, 0, 0], Y_n=[0, 0, R], Z_n=[0, 0, 0])
            test(
                coil,
                grid_xyz,
                grid_rpz,
                A_true_rpz,
                correct_flux,
                rtol=1e-8,
                atol=1e-12,
            )

            # SplineXYZCoil
            x = coil.compute("x", grid=coil_grid, basis="xyz")["x"]
            coil = SplineXYZCoil(I, X=x[:, 0], Y=x[:, 1], Z=x[:, 2])
            test(
                coil,
                grid_xyz,
                grid_rpz,
                A_true_rpz,
                correct_flux,
                rtol=1e-4,
                atol=1e-12,
            )

            # FourierPlanarCoil
            coil = FourierPlanarCoil(I, center=[0, 0, 0], normal=[0, 0, -1], r_n=R)
            test(
                coil,
                grid_xyz,
                grid_rpz,
                A_true_rpz,
                correct_flux,
                rtol=1e-8,
                atol=1e-12,
            )

            # FourierRZCoil
            coil = FourierRZCoil(I, R_n=np.array([R]), modes_R=np.array([0]))
            test(
                coil,
                grid_xyz,
                grid_rpz,
                A_true_rpz,
                correct_flux,
                rtol=1e-8,
                atol=1e-12,
            )

    @pytest.mark.unit
    def test_properties(self):
        """Test getting/setting attributes for Coil class."""
        current = 4.34
        coil = FourierPlanarCoil(current)
        assert coil.current == current
        new_current = 3.5
        coil.current = new_current
        assert coil.current == new_current

    @pytest.mark.unit
    def test_SumMagneticField_with_Coil(self):
        """Test SumMagneticField working with Coil and MagneticField objects."""
        R = 2
        y = 1
        I = 1e7
        B_Z = 2  # add constant vertical field of 2T
        By_true = scipy.constants.mu_0 / 2 * R**2 * I / (y**2 + R**2) ** (3 / 2)
        B_true = np.array([0, By_true, 2])
        coil = FourierXYZCoil(I)

        field = SumMagneticField(coil, VerticalMagneticField(B_Z))
        B_approx = field.compute_magnetic_field(
            np.array([[10, y, 0], [10, -y, 0]]), basis="xyz", source_grid=100
        )[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    @pytest.mark.unit
    def test_adding_MagneticField_with_Coil_or_CoilSet(self):
        """Test MagneticField plus Coil/CoilSet and vice versa."""
        R = 2
        y = 1
        I = 1e7
        B_Z = 2  # add constant vertical field of 2T
        By_true = scipy.constants.mu_0 / 2 * R**2 * I / (y**2 + R**2) ** (3 / 2)
        B_true = np.array([0, By_true, 2])
        coil = FourierXYZCoil(I)
        coilset = CoilSet(coil)
        mixedcoilset = MixedCoilSet(coil)

        field1 = coil + VerticalMagneticField(B_Z)
        field2 = VerticalMagneticField(B_Z) + coil
        # coilset + magnetic field (tests __radd__ of field)
        field3 = coilset + VerticalMagneticField(B_Z)
        field4 = VerticalMagneticField(B_Z) + coilset
        field5 = mixedcoilset + VerticalMagneticField(B_Z)
        field6 = VerticalMagneticField(B_Z) + mixedcoilset

        for i, field in enumerate([field1, field2, field3, field4, field5, field6]):
            B_approx = field.compute_magnetic_field(
                np.array([[10, y, 0], [10, -y, 0]]), basis="xyz", source_grid=100
            )[0]
            np.testing.assert_allclose(
                B_true, B_approx, rtol=1e-3, atol=1e-10, err_msg=f"field {i}"
            )

    @pytest.mark.unit
    def test_convert_type(self):
        """Test conversions between coil representations."""
        s = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        coil1 = FourierRZCoil(1e6, [0, 10, 1], [0, 0, 0])
        coil2 = coil1.to_FourierXYZ(s=s)
        coil3 = coil1.to_SplineXYZ(knots=s)
        coil4 = coil1.to_FourierRZ(N=coil1.N)
        coil5 = coil1.to_FourierPlanar(N=10, basis="rpz")

        grid = LinearGrid(zeta=s)
        x1 = coil1.compute("x", grid=grid, basis="xyz")["x"]
        x2 = coil2.compute("x", grid=grid, basis="xyz")["x"]
        x3 = coil3.compute("x", grid=grid, basis="xyz")["x"]
        x4 = coil4.compute("x", grid=grid, basis="xyz")["x"]
        zeta = np.arctan2(  # zeta = polar angle for planar coil for same points
            x1[:, 1] - coil5.center[1],
            x1[:, 0] - coil5.center[0],
        )  # use Grid instead of LinearGrid to prevent node sorting
        grid_planar = Grid(np.array([np.zeros_like(zeta), np.zeros_like(zeta), zeta]).T)
        x5 = coil5.compute("x", grid=grid_planar, basis="xyz")["x"]

        B1 = coil1.compute_magnetic_field(
            np.zeros((1, 3)), source_grid=grid, basis="xyz"
        )
        B2 = coil2.compute_magnetic_field(
            np.zeros((1, 3)), source_grid=grid, basis="xyz"
        )
        B3 = coil3.compute_magnetic_field(
            np.zeros((1, 3)), source_grid=grid, basis="xyz"
        )
        B4 = coil4.compute_magnetic_field(
            np.zeros((1, 3)), source_grid=grid, basis="xyz"
        )
        B5 = coil5.compute_magnetic_field(
            np.zeros((1, 3)), source_grid=grid, basis="xyz"
        )

        np.testing.assert_allclose(x1, x2, atol=1e-12)
        np.testing.assert_allclose(x1, x3, atol=1e-12)
        np.testing.assert_allclose(x1, x4, atol=1e-12)
        np.testing.assert_allclose(x1, x5, atol=1e-12)
        np.testing.assert_allclose(B1, B2, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(B1, B3, rtol=1e-3, atol=1e-8)
        np.testing.assert_allclose(B1, B4, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(B1, B5, rtol=1e-6, atol=1e-7)


class TestCoilSet:
    """Tests for sets of multiple coils."""

    @pytest.mark.unit
    def test_linspaced_linear(self):
        """Field from straight solenoid."""
        R = 10
        z = np.linspace(0, 10, 10)
        I = 1e7
        n = 10
        Bz_true = np.sum(scipy.constants.mu_0 / 2 * R**2 * I / (z**2 + R**2) ** (3 / 2))
        B_true = np.array([0, 0, Bz_true])
        coil = FourierRZCoil(0.1)
        coils = CoilSet.linspaced_linear(
            coil, displacement=[0, 0, 10], n=n, endpoint=True
        )
        coils.current = I
        np.testing.assert_allclose(coils.current, I)
        B_approx = coils.compute_magnetic_field(
            [0, 0, z[-1]], basis="xyz", source_grid=32
        )[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    @pytest.mark.unit
    def test_linspaced_angular(self):
        """Field from uniform toroidal solenoid."""
        R = 10
        N = 50
        I = 1e7
        Bp_true = np.sum(scipy.constants.mu_0 * N * I / 2 / np.pi / R)
        B_true = np.array([0, Bp_true, 0])
        coil = FourierPlanarCoil()
        coil.current = I
        coils = CoilSet.linspaced_angular(coil, n=N)
        grid = LinearGrid(N=32, endpoint=False)
        transforms = get_transforms(["x", "x_s", "ds"], coil, grid=grid)
        B_approx = coils.compute_magnetic_field(
            [10, 0, 0], basis="rpz", source_grid=grid, transforms=transforms
        )[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

        surf = FourierRZToroidalSurface(
            R_lmn=np.array([10, 0.1]),
            Z_lmn=np.array([-0.1]),
            modes_R=np.array([[0, 0], [1, 0]]),
            modes_Z=np.array([[-1, 0]]),
        )

        B_normal, _ = coils.compute_Bnormal(surf)
        np.testing.assert_allclose(B_normal, 0, atol=1e-9)

    @pytest.mark.unit
    def test_from_symmetry(self):
        """Same toroidal solenoid field, but different construction."""
        R = 10
        N = 48
        I = 1e7
        Bp_true = np.sum(scipy.constants.mu_0 * N * I / 2 / np.pi / R)
        B_true = np.array([0, Bp_true, 0])
        coil = FourierPlanarCoil(I)
        coils = CoilSet.linspaced_angular(coil, angle=np.pi / 2, n=N // 4)
        coils = MixedCoilSet.from_symmetry(coils, NFP=4)
        grid = LinearGrid(N=32, endpoint=False)
        transforms = get_transforms(["x", "x_s", "ds"], coil, grid=grid)
        B_approx = coils.compute_magnetic_field(
            [10, 0, 0], basis="rpz", source_grid=grid, transforms=transforms
        )[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

        # with stellarator symmetry
        NFP = 4
        coil = FourierXYZCoil()
        coil.rotate(angle=np.pi / N)
        coils = CoilSet.linspaced_angular(
            coil, I, [0, 0, 1], np.pi / NFP, N // NFP // 2
        )
        coils2 = MixedCoilSet.from_symmetry(coils, NFP, True)
        B_approx = coils2.compute_magnetic_field(
            [10, 0, 0], basis="rpz", source_grid=32
        )[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

        # With a MixedCoilSet as the base coils and only rotation
        coil = FourierPlanarCoil(I)
        coils = [coil] + [FourierXYZCoil(I) for i in range(N // 4 - 1)]
        for i, c in enumerate(coils[1:]):
            c.rotate(angle=2 * np.pi / N * (i + 1))
        coils = MixedCoilSet.from_symmetry(coils, NFP=4)
        grid = LinearGrid(N=32, endpoint=False)
        transforms = get_transforms(["x", "x_s", "ds"], coil, grid=grid)
        B_approx = coils.compute_magnetic_field(
            [10, 0, 0], basis="rpz", source_grid=grid
        )[0]
        np.testing.assert_allclose(B_true, B_approx, rtol=1e-3, atol=1e-10)

    @pytest.mark.unit
    def test_is_self_intersecting_warnings(self):
        """Test warning in from_symmetry for self-intersection."""
        N = 40
        # tilt coils so they cross the symmetry plane
        # and the resulting coils are self-intersecting
        coil = FourierPlanarCoil(normal=[1e-4, 1, 3])
        coils_list_sym = [coil.copy()] + [coil.copy() for i in range(N // 8 - 1)]

        for i, c in enumerate(coils_list_sym[1:]):
            c.rotate(angle=2 * np.pi / N * (i + 1))

        # test the warning for self-intersecting coils, as two
        #  of the coils in each field period lie nearly
        # in the same physical space (intersecting at 2 points) after reflection
        with pytest.warns(UserWarning) as warninfo:
            _ = CoilSet.from_symmetry(coils_list_sym, NFP=4, sym=True)
        assert "nearly intersecting" in str(warninfo[0].message)

    @pytest.mark.unit
    def test_properties(self):
        """Test getting/setting of CoilSet attributes."""
        coil = FourierPlanarCoil()
        coils = CoilSet.linspaced_linear(coil, n=4, displacement=[0, 2, 0])
        data = coils.compute(
            [
                "x",
                "curvature",
                "torsion",
                "frenet_tangent",
                "frenet_normal",
                "frenet_binormal",
            ],
            grid=0,
            basis="xyz",
        )
        np.testing.assert_allclose(
            [dat["x"] for dat in data],
            np.array(
                [
                    [12, 0, 0],
                    [12, 0.5, 0],
                    [12, 1, 0],
                    [12, 1.5, 0],
                ]
            ).reshape((4, 1, 3)),
        )
        np.testing.assert_allclose([dat["curvature"] for dat in data], 1 / 2)
        np.testing.assert_allclose([dat["torsion"] for dat in data], 0)
        T = [dat["frenet_tangent"] for dat in data]
        N = [dat["frenet_normal"] for dat in data]
        B = [dat["frenet_binormal"] for dat in data]
        np.testing.assert_allclose(
            T,
            np.array(
                [
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            N,
            np.array(
                [
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [-1, 0, 0],
                    [-1, 0, 0],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            B,
            np.array(
                [
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        data = coils.compute("length", grid=32)
        np.testing.assert_allclose([dat["length"] for dat in data], 2 * 2 * np.pi)
        coils.translate([1, 1, 1])
        data = coils.compute("length", grid=32)
        np.testing.assert_allclose([dat["length"] for dat in data], 2 * 2 * np.pi)
        coils.flip([1, 0, 0])
        data = coils.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"],
            grid=0,
            basis="xyz",
        )
        T = [dat["frenet_tangent"] for dat in data]
        N = [dat["frenet_normal"] for dat in data]
        B = [dat["frenet_binormal"] for dat in data]
        np.testing.assert_allclose(
            T,
            np.array(
                [
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                    [0, 0, -1],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            N,
            np.array(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            B,
            np.array(
                [
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ]
            ).reshape((4, 1, 3)),
            atol=1e-12,
        )

    @pytest.mark.unit
    def test_dunder_methods(self):
        """Test methods for combining and calling CoilSet objects."""
        coil1 = FourierXYZCoil()
        coils1 = MixedCoilSet.from_symmetry(coil1, NFP=4)
        coil2 = FourierPlanarCoil(center=[100, 0, 0])
        coils2 = coils1 + [coil2]
        assert coils2[-1] is coil2
        with pytest.warns(UserWarning, match="nearly intersecting"):
            coils2 = coils1 + MixedCoilSet([coil2, coil2], check_intersection=False)
        assert coils2[-1] is coil2

        with pytest.raises(TypeError):
            _ = coils1 + FourierRZCurve()

        with pytest.raises(TypeError):
            coils1[-1] = FourierRZCurve()

        coils1[-1] = coil2
        assert coils1[-1] is coil2

        coils1.insert(-1, coil2)
        with pytest.raises(TypeError):
            coils1.insert(-1, FourierRZCurve())

        assert len(coils1) == 5

        assert coils1[-1] is coil2
        assert coils1[-2] is coil2

        s = coils1[-2:]
        assert s[-1] is coil2

        del coils1[-2]
        assert len(coils1) == 4
        assert coils1[-1] is coil2
        assert coils1[-2].__class__ is coil1.__class__

        coils2 = CoilSet.linspaced_angular(coil1)
        assert coils2[0].equiv(coil1) and not (coils2[0] is coil1)
        coils2[0] = coil1
        assert coils2[0] is coil1
        with pytest.raises(TypeError):
            coils2[1] = coil2
        with pytest.raises(TypeError):
            coils2.insert(4, coil2)

    @pytest.mark.unit
    def test_convert_type(self):
        """Test converting coilsets between different representations."""
        grid = LinearGrid(N=20)
        coil = FourierRZCoil(1e6, [0, 10, 1], [0, 0, 0])

        # MixedCoilSet
        coils1 = MixedCoilSet.linspaced_linear(coil, displacement=[0, 0, 2.5], n=4)
        coils2 = coils1.to_SplineXYZ(grid=grid, check_intersection=False)
        coils3 = coils1.to_FourierXYZ(grid=grid, check_intersection=False)
        coils4 = coils1.to_FourierPlanar(grid=grid, check_intersection=False)
        assert isinstance(coils2, MixedCoilSet)
        assert isinstance(coils3, MixedCoilSet)
        assert isinstance(coils4, MixedCoilSet)
        assert all(isinstance(coil, SplineXYZCoil) for coil in coils2)
        assert all(isinstance(coil, FourierXYZCoil) for coil in coils3)
        assert all(isinstance(coil, FourierPlanarCoil) for coil in coils4)
        x1 = coils1.compute("x", grid=grid, basis="xyz")
        x2 = coils2.compute("x", grid=grid, basis="xyz")
        x3 = coils3.compute("x", grid=grid, basis="xyz")
        zeta = np.arctan2(  # zeta = polar angle for planar coil for same points
            x1[0]["x"][:, 1] - coils4[0].center[1],
            x1[0]["x"][:, 0] - coils4[0].center[0],
        )  # use Grid instead of LinearGrid to prevent node sorting
        grid_planar = Grid(np.array([np.zeros_like(zeta), np.zeros_like(zeta), zeta]).T)
        x4 = coils4.compute("x", grid=grid_planar, basis="xyz")
        np.testing.assert_allclose(
            [xi["x"] for xi in x1], [xi["x"] for xi in x2], atol=1e-12
        )
        np.testing.assert_allclose(
            [xi["x"] for xi in x1], [xi["x"] for xi in x3], atol=1e-12
        )
        np.testing.assert_allclose(
            [xi["x"] for xi in x1], [xi["x"] for xi in x4], atol=1e-12
        )
        B1 = coils1.compute_magnetic_field(np.array([[5, 2, 1]]), source_grid=grid)
        B2 = coils2.compute_magnetic_field(np.array([[5, 2, 1]]), source_grid=grid)
        B3 = coils3.compute_magnetic_field(np.array([[5, 2, 1]]), source_grid=grid)
        B4 = coils4.compute_magnetic_field(np.array([[5, 2, 1]]), source_grid=grid)
        np.testing.assert_allclose(B1, B2, rtol=1e-2)
        np.testing.assert_allclose(B1, B3, rtol=1e-2)
        np.testing.assert_allclose(B1, B4, rtol=1e-2)

        # CoilSet
        coil = coils3[0]  # FourierXYZCoil
        coils5 = CoilSet.linspaced_linear(coil, displacement=[0, 0, 3.5], n=6)
        coils6 = coils5.to_SplineXYZ(grid=grid, check_intersection=False)
        coils7 = coils5.to_FourierRZ(grid=grid, check_intersection=False)
        coils8 = coils5.to_FourierPlanar(grid=grid, check_intersection=False)
        assert isinstance(coils6, CoilSet)
        assert isinstance(coils7, CoilSet)
        assert isinstance(coils8, CoilSet)
        assert all(isinstance(coil, SplineXYZCoil) for coil in coils6)
        assert all(isinstance(coil, FourierRZCoil) for coil in coils7)
        assert all(isinstance(coil, FourierPlanarCoil) for coil in coils8)
        x5 = coils5.compute("x", grid=grid, basis="xyz")
        x6 = coils6.compute("x", grid=grid, basis="xyz")
        x7 = coils7.compute("x", grid=grid, basis="xyz")
        zeta = np.arctan2(  # zeta = polar angle for planar coil for same points
            x5[0]["x"][:, 1] - coils8[0].center[1],
            x5[0]["x"][:, 0] - coils8[0].center[0],
        )  # use Grid instead of LinearGrid to prevent node sorting
        grid_planar = Grid(np.array([np.zeros_like(zeta), np.zeros_like(zeta), zeta]).T)
        x8 = coils8.compute("x", grid=grid_planar, basis="xyz")
        np.testing.assert_allclose(
            [xi["x"] for xi in x5], [xi["x"] for xi in x6], atol=1e-12
        )
        np.testing.assert_allclose(
            [xi["x"] for xi in x5], [xi["x"] for xi in x7], atol=1e-12
        )
        np.testing.assert_allclose(
            [xi["x"] for xi in x5], [xi["x"] for xi in x8], atol=1e-12
        )
        B5 = coils5.compute_magnetic_field(np.array([[5, 2, 1]]), source_grid=grid)
        B6 = coils6.compute_magnetic_field(np.array([[5, 2, 1]]), source_grid=grid)
        B7 = coils6.compute_magnetic_field(np.array([[5, 2, 1]]), source_grid=grid)
        B8 = coils6.compute_magnetic_field(np.array([[5, 2, 1]]), source_grid=grid)
        np.testing.assert_allclose(B5, B6, rtol=1e-2)
        np.testing.assert_allclose(B5, B7, rtol=1e-2)
        np.testing.assert_allclose(B5, B8, rtol=1e-2)


@pytest.mark.unit
def test_symmetry_position(DummyCoilSet):
    """Tests that compute position is correct from symmetry."""
    # same coil sets with vs without symmetry
    coilset_sym = load(
        load_from=str(DummyCoilSet["output_path_sym"]), file_format="hdf5"
    )
    coilset_asym = load(
        load_from=str(DummyCoilSet["output_path_asym"]), file_format="hdf5"
    )
    coilset_mixed = MixedCoilSet(*coilset_asym)
    grid = LinearGrid(N=30)

    # check that positions of CoilSets are the same with xyz basis
    x_sym = coilset_sym._compute_position(basis="xyz", grid=grid)
    x_asym = coilset_asym._compute_position(basis="xyz", grid=grid)
    x_mixed = coilset_mixed._compute_position(basis="xyz", grid=grid)

    np.testing.assert_allclose(x_sym, x_asym)
    np.testing.assert_allclose(x_sym, x_mixed)

    # check that positions of CoilSets are the same with rpz basis
    x_sym = coilset_sym._compute_position(basis="rpz", grid=grid)
    x_asym = coilset_asym._compute_position(basis="rpz", grid=grid)
    x_mixed = coilset_mixed._compute_position(basis="rpz", grid=grid)

    np.testing.assert_allclose(x_sym, x_asym)
    np.testing.assert_allclose(x_sym, x_mixed)


@pytest.mark.unit
def test_symmetry_magnetic_field(DummyCoilSet):
    """Tests that compute magnetic field is correct from symmetry."""
    # same coil sets with vs without symmetry
    eq = get("precise_QH")
    coilset_sym = load(
        load_from=str(DummyCoilSet["output_path_sym"]), file_format="hdf5"
    )
    coilset_asym = load(
        load_from=str(DummyCoilSet["output_path_asym"]), file_format="hdf5"
    )

    # test that both coil sets compute the same field on the plasma surface
    grid = LinearGrid(rho=[1.0], M=eq.M_grid, N=eq.N_grid, NFP=1, sym=False)

    with pytest.warns(UserWarning):  # because eq.NFP != grid.NFP
        data = eq.compute(["phi", "R", "X", "Y", "Z"], grid)

    # test in (R, phi, Z) coordinates
    nodes_rpz = np.array([data["R"], data["phi"], data["Z"]]).T
    B_sym_rpz = coilset_sym.compute_magnetic_field(nodes_rpz, basis="rpz")
    B_asym_rpz = coilset_asym.compute_magnetic_field(nodes_rpz, basis="rpz")
    np.testing.assert_allclose(B_sym_rpz, B_asym_rpz, atol=1e-14)

    # test in (X, Y, Z) coordinates
    nodes_xyz = np.array([data["X"], data["Y"], data["Z"]]).T
    B_sym_xyz = coilset_sym.compute_magnetic_field(nodes_xyz, basis="xyz")
    B_asym_xyz = coilset_asym.compute_magnetic_field(nodes_xyz, basis="xyz")
    np.testing.assert_allclose(B_sym_xyz, B_asym_xyz, atol=1e-14)


@pytest.mark.unit
def test_load_and_save_makegrid_coils(tmpdir_factory):
    """Test loading in and saving CoilSets from MAKEGRID format files."""
    Ncoils = 22
    input_path = f"./tests/inputs/coils.MAKEGRID_format_{Ncoils}_coils"
    tmpdir = tmpdir_factory.mktemp("coil_files")
    tmp_path = tmpdir.join(f"coils.MAKEGRID_format_{Ncoils}_coils")
    shutil.copyfile(input_path, tmp_path)

    coilset = CoilSet.from_makegrid_coilfile(str(tmp_path))
    assert len(coilset) == Ncoils  # correct number of coils

    path = tmpdir.join("coils.MAKEGRID_format_desc")
    coilset.save_in_makegrid_format(
        str(path), grid=LinearGrid(zeta=coilset[0].knots, theta=0, endpoint=True)
    )

    coilset2 = CoilSet.from_makegrid_coilfile(str(path))

    grid = LinearGrid(N=200, endpoint=False)

    # check values at saved points, ensure they match
    for i, (c1, c2) in enumerate(zip(coilset, coilset2)):
        # make sure knots are exactly the same
        np.testing.assert_allclose(c1.knots, c2.knots, err_msg=f"Coil {i}")

        grid = LinearGrid(zeta=coilset2[0].knots, endpoint=False)
        coords1 = c1.compute("x", grid=grid, basis="xyz")["x"]
        X1 = coords1[:, 0]
        Y1 = coords1[:, 1]
        Z1 = coords1[:, 2]

        coords2 = c2.compute("x", grid=grid, basis="xyz")["x"]
        X2 = coords2[:, 0]
        Y2 = coords2[:, 1]
        Z2 = coords2[:, 2]

        np.testing.assert_allclose(c1.current, c2.current, err_msg=f"Coil {i}")
        np.testing.assert_allclose(X1, X2, err_msg=f"Coil {i}")
        np.testing.assert_allclose(Y1, Y2, err_msg=f"Coil {i}")
        np.testing.assert_allclose(Z1, Z2, atol=2e-7, err_msg=f"Coil {i}")

    # check magnetic field from both, check that matches
    grid = LinearGrid(N=200, endpoint=False)
    B1 = coilset.compute_magnetic_field(
        np.array([[0.7, 0, 0]]), basis="xyz", source_grid=grid
    )
    B2 = coilset2.compute_magnetic_field(
        np.array([[0.7, 0, 0]]), basis="xyz", source_grid=grid
    )

    np.testing.assert_allclose(B1, B2, atol=1e-7)


@pytest.mark.unit
def test_load_and_save_makegrid_coils_diff_length_of_knots(tmpdir_factory):
    """Test loading and saving coils from MAKEGRID file that are not uniform length."""
    Ncoils = 2
    input_path = f"./tests/inputs/coils.MAKEGRID_format_{Ncoils}_coils_diff_lengths"
    tmpdir = tmpdir_factory.mktemp("coil_files")
    tmp_path = tmpdir.join(f"coils.MAKEGRID_format_{Ncoils}_coils")
    shutil.copyfile(input_path, tmp_path)

    with pytest.raises(ValueError, match="CoilSet"):
        coilset = CoilSet.from_makegrid_coilfile(str(tmp_path))
    coilset = MixedCoilSet.from_makegrid_coilfile(str(tmp_path), ignore_groups=True)
    assert len(coilset) == Ncoils  # correct number of coils
    # if the coils are not all the same number of knots, then making a CoilSet
    # will fail (as each underyling coil must have same length knots),
    # instead the function should make a MixedCoilSet
    assert isinstance(coilset, MixedCoilSet)

    path = tmpdir.join("coils.MAKEGRID_format_desc")
    # save using the default grids
    coilset.save_in_makegrid_format(str(path))

    coilset2 = MixedCoilSet.from_makegrid_coilfile(str(path))
    assert isinstance(coilset2, MixedCoilSet)

    grid = LinearGrid(N=50, endpoint=False)

    # check values, ensure they are close
    for i, (c1, c2) in enumerate(zip(coilset, coilset2)):

        coords1 = c1.compute("x", grid=grid, basis="xyz")["x"]
        X1 = coords1[:, 0]
        Y1 = coords1[:, 1]
        Z1 = coords1[:, 2]

        coords2 = c2.compute("x", grid=grid, basis="xyz")["x"]
        X2 = coords2[:, 0]
        Y2 = coords2[:, 1]
        Z2 = coords2[:, 2]
        # knots are not the exact same, so these points will be close but not the
        # same.
        np.testing.assert_allclose(c1.current, c2.current, err_msg=f"Coil {i}")
        np.testing.assert_allclose(X1, X2, rtol=2e-5, err_msg=f"Coil {i}")
        np.testing.assert_allclose(Y1, Y2, rtol=2e-5, err_msg=f"Coil {i}")
        np.testing.assert_allclose(Z1, Z2, atol=2e-7, rtol=1e-3, err_msg=f"Coil {i}")

    # check magnetic field from both, check that matches
    grid = LinearGrid(N=200, endpoint=False)
    B1 = coilset.compute_magnetic_field(
        np.array([[0.7, 0, 0]]), basis="xyz", source_grid=grid
    )
    B2 = coilset2.compute_magnetic_field(
        np.array([[0.7, 0, 0]]), basis="xyz", source_grid=grid
    )

    np.testing.assert_allclose(B1, B2, atol=1e-7)


@pytest.mark.unit
def test_load_and_save_makegrid_coils_groups(tmpdir_factory):
    """Test loading and saving CoilSets from MAKEGRID format files with coilgroups."""
    Ncoils_per_group = 2
    coilgroups = ["groupone", "grouptwo"]
    input_path = "./tests/inputs/coils.MAKEGRID_format_two_groups"
    tmpdir = tmpdir_factory.mktemp("coil_files")
    tmp_path = tmpdir.join("coils.MAKEGRID_format_two_groups")
    shutil.copyfile(input_path, tmp_path)

    coilset = MixedCoilSet.from_makegrid_coilfile(str(tmp_path))
    assert len(coilset) == len(coilgroups)  # correct number of coils
    for i, (coils, groupname) in enumerate(zip(coilset, coilgroups)):
        assert len(coils) == Ncoils_per_group
        assert groupname in coils.name
        assert str(i + 1) in coils.name  # make sure the correct number is in the name
    path = tmpdir.join("coils.MAKEGRID_format_groups_desc")

    coilset.save_in_makegrid_format(
        str(path), grid=LinearGrid(zeta=coilset[0][0].knots, theta=0, endpoint=True)
    )
    coilset2 = MixedCoilSet.from_makegrid_coilfile(str(path), ignore_groups=False)
    # also compare to flattened
    coilset_flat = MixedCoilSet.from_makegrid_coilfile(str(path), ignore_groups=True)
    assert len(coilset_flat) == Ncoils_per_group * len(coilgroups)

    assert len(coilset2) == len(coilgroups)  # correct number of coils groups
    for i, (coils, groupname) in enumerate(zip(coilset2, coilgroups)):
        assert len(coils) == Ncoils_per_group
        assert groupname in coils.name
        assert str(i + 1) in coils.name  # make sure the correct number is in the name

    grid = LinearGrid(zeta=coilset[0][0].knots, endpoint=False)

    # check values at saved points, ensure they match
    for i, (cs1, cs2) in enumerate(zip(coilset, coilset2)):
        for j, (c1, c2) in enumerate(zip(cs1, cs2)):
            c3 = coilset_flat[2 * i + j]
            print(c1)
            print(c2)
            print(c3)
            # make sure knots are exactly the same
            np.testing.assert_allclose(
                c1.knots, c2.knots, err_msg=f"CoilSet {i} Coil {j}"
            )
            np.testing.assert_allclose(
                c3.knots, c2.knots, err_msg=f"CoilSet {i} Coil {j}"
            )

            coords1 = c1.compute("x", grid=grid, basis="xyz")["x"]
            X1 = coords1[:, 0]
            Y1 = coords1[:, 1]
            Z1 = coords1[:, 2]

            coords2 = c2.compute("x", grid=grid, basis="xyz")["x"]
            X2 = coords2[:, 0]
            Y2 = coords2[:, 1]
            Z2 = coords2[:, 2]

            coords3 = c3.compute("x", grid=grid, basis="xyz")["x"]
            X3 = coords3[:, 0]
            Y3 = coords3[:, 1]
            Z3 = coords3[:, 2]

            np.testing.assert_allclose(
                c1.current, c2.current, err_msg=f"CoilSet {i} Coil {j}"
            )
            np.testing.assert_allclose(
                c3.current, c2.current, err_msg=f"CoilSet {i} Coil {j}"
            )
            np.testing.assert_allclose(X1, X2, err_msg=f"CoilSet {i} Coil {j}")
            np.testing.assert_allclose(X3, X2, err_msg=f"CoilSet {i} Coil {j}")
            np.testing.assert_allclose(Y1, Y2, err_msg=f"CoilSet {i} Coil {j}")
            np.testing.assert_allclose(Y3, Y2, err_msg=f"CoilSet {i} Coil {j}")
            np.testing.assert_allclose(
                Z1, Z2, atol=2e-7, err_msg=f"CoilSet {i} Coil {j}"
            )
            np.testing.assert_allclose(
                Z3, Z2, atol=2e-7, err_msg=f"CoilSet {i} Coil {j}"
            )

    # check magnetic field from both, check that matches
    grid = LinearGrid(N=200, endpoint=False)
    B1 = coilset.compute_magnetic_field(
        np.array([[0.7, 0, 0]]), basis="xyz", source_grid=grid
    )
    B2 = coilset2.compute_magnetic_field(
        np.array([[0.7, 0, 0]]), basis="xyz", source_grid=grid
    )
    B3 = coilset_flat.compute_magnetic_field(
        np.array([[0.7, 0, 0]]), basis="xyz", source_grid=grid
    )

    np.testing.assert_allclose(B1, B2, atol=1e-7)
    np.testing.assert_allclose(B3, B2, atol=1e-7)


@pytest.mark.unit
def test_save_and_load_makegrid_coils_rotated(tmpdir_factory):
    """Test saving and reloading CoilSet linspaced angular from MAKEGRID file."""
    tmpdir = tmpdir_factory.mktemp("coil_files")
    path = tmpdir.join("coils.MAKEGRID_format_angular_coil")

    # make a coilset with angular coilset
    N = 22
    coil = FourierPlanarCoil()
    coil.current = 1
    coilset = CoilSet.linspaced_angular(coil, n=N, angle=2 * np.pi)

    grid = LinearGrid(N=200, endpoint=False)
    coilset.save_in_makegrid_format(str(path), grid=grid, NFP=2)

    coilset2 = CoilSet.from_makegrid_coilfile(str(path))

    # check values at saved points, ensure they match
    for i, (c1, c2) in enumerate(zip(coilset, coilset2)):
        grid = LinearGrid(zeta=coilset2[0].knots, endpoint=False)
        coords1 = c1.compute("x", grid=grid, basis="xyz")["x"]
        X1 = coords1[:, 0]
        Y1 = coords1[:, 1]
        Z1 = coords1[:, 2]

        coords2 = c2.compute("x", grid=grid, basis="xyz")["x"]
        X2 = coords2[:, 0]
        Y2 = coords2[:, 1]
        Z2 = coords2[:, 2]

        np.testing.assert_allclose(c1.current, c2.current, err_msg=f"Coil {i}")
        np.testing.assert_allclose(X1, X2, err_msg=f"Coil {i}")
        np.testing.assert_allclose(Y1, Y2, err_msg=f"Coil {i}")
        np.testing.assert_allclose(Z1, Z2, atol=2e-7, err_msg=f"Coil {i}")

    # check values at interpolated points, ensure they match closely
    grid = LinearGrid(N=51, endpoint=False)
    for c1, c2 in zip(coilset, coilset2):
        coords1 = c1.compute("x", grid=grid, basis="xyz")["x"]
        X1 = coords1[:, 0]
        Y1 = coords1[:, 1]
        Z1 = coords1[:, 2]

        coords2 = c2.compute("x", grid=grid, basis="xyz")["x"]
        X2 = coords2[:, 0]
        Y2 = coords2[:, 1]
        Z2 = coords2[:, 2]

        np.testing.assert_allclose(c1.current, c2.current, err_msg=f"Coil {i}")
        np.testing.assert_allclose(X1, X2, err_msg=f"Coil {i}", atol=1e-16)
        np.testing.assert_allclose(Y1, Y2, err_msg=f"Coil {i}", atol=1e-16)
        np.testing.assert_allclose(Z1, Z2, atol=2e-7, err_msg=f"Coil {i}")

    # check Bnormal on torus and ensure is near zero
    surf = FourierRZToroidalSurface(
        R_lmn=np.array([10, 0.1]),
        Z_lmn=np.array([-0.1]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
    )

    B_normal, _ = coilset.compute_Bnormal(surf, source_grid=grid)
    np.testing.assert_allclose(B_normal, 0, atol=1e-16)
    B_normal2, _ = coilset2.compute_Bnormal(surf)
    np.testing.assert_allclose(B_normal2, 0, atol=1e-16)

    # check B btwn the two coilsets
    B1 = coilset.compute_magnetic_field(
        np.array([[10, 0, 0]]), basis="xyz", source_grid=32
    )
    B2 = coilset2.compute_magnetic_field(
        np.array([[10, 0, 0]]), basis="xyz", source_grid=1000
    )

    # coilset uses fourier discretization so biot savart is more accurate
    # coilset2 uses hanson hirshman which is only 2nd order
    np.testing.assert_allclose(B1, B2, atol=1e-16, rtol=1e-6)


@pytest.mark.unit
def test_save_and_load_makegrid_coils_rotated_int_grid(tmpdir_factory):
    """Test save/load CoilSet linspaced angular from MAKEGRID file with int grid."""
    tmpdir = tmpdir_factory.mktemp("coil_files")
    path = tmpdir.join("coils.MAKEGRID_format_angular_coil")

    # make a coilset with angular coilset
    N = 10
    coil = FourierPlanarCoil()
    coil.current = 1
    coilset = CoilSet.linspaced_angular(coil, n=N, angle=2 * np.pi)

    grid = 200
    coilset.save_in_makegrid_format(str(path), grid=grid, NFP=2)

    coilset2 = CoilSet.from_makegrid_coilfile(str(path))

    # check values at saved points, ensure they match
    for i, (c1, c2) in enumerate(zip(coilset, coilset2)):
        grid = LinearGrid(zeta=coilset2[0].knots, endpoint=False)
        coords1 = c1.compute("x", grid=grid, basis="xyz")["x"]
        X1 = coords1[:, 0]
        Y1 = coords1[:, 1]
        Z1 = coords1[:, 2]

        coords2 = c2.compute("x", grid=grid, basis="xyz")["x"]
        X2 = coords2[:, 0]
        Y2 = coords2[:, 1]
        Z2 = coords2[:, 2]

        np.testing.assert_allclose(c1.current, c2.current, err_msg=f"Coil {i}")
        np.testing.assert_allclose(X1, X2, err_msg=f"Coil {i}")
        np.testing.assert_allclose(Y1, Y2, err_msg=f"Coil {i}")
        np.testing.assert_allclose(Z1, Z2, atol=2e-7, err_msg=f"Coil {i}")

    # check values at interpolated points, ensure they match closely
    grid = LinearGrid(N=101, endpoint=False)
    for c1, c2 in zip(coilset, coilset2):
        coords1 = c1.compute("x", grid=grid, basis="xyz")["x"]
        X1 = coords1[:, 0]
        Y1 = coords1[:, 1]
        Z1 = coords1[:, 2]

        coords2 = c2.compute("x", grid=grid, basis="xyz")["x"]
        X2 = coords2[:, 0]
        Y2 = coords2[:, 1]
        Z2 = coords2[:, 2]

        np.testing.assert_allclose(c1.current, c2.current, err_msg=f"Coil {i}")
        np.testing.assert_allclose(X1, X2, err_msg=f"Coil {i}", atol=1e-16)
        np.testing.assert_allclose(Y1, Y2, err_msg=f"Coil {i}", atol=1e-16)
        np.testing.assert_allclose(Z1, Z2, atol=2e-7, err_msg=f"Coil {i}")

    # check Bnormal on torus and ensure is near zero
    surf = FourierRZToroidalSurface(
        R_lmn=np.array([10, 0.1]),
        Z_lmn=np.array([-0.1]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
    )

    B_normal, _ = coilset.compute_Bnormal(surf, source_grid=grid)
    np.testing.assert_allclose(B_normal, 0, atol=1e-16)
    B_normal2, _ = coilset2.compute_Bnormal(surf)
    np.testing.assert_allclose(B_normal2, 0, atol=1e-16)

    # check B btwn the two coilsets
    B1 = coilset.compute_magnetic_field(
        np.array([[10, 0, 0]]), basis="xyz", source_grid=grid
    )
    B2 = coilset2.compute_magnetic_field(
        np.array([[10, 0, 0]]), basis="xyz", source_grid=grid
    )

    np.testing.assert_allclose(B1, B2, atol=1e-10)


@pytest.mark.unit
def test_save_and_load_makegrid_coils_nested(tmpdir_factory):
    """Test saving and reloading a nested CoilSet from MAKEGRID file."""
    tmpdir = tmpdir_factory.mktemp("coil_files")
    path = tmpdir.join("coils.MAKEGRID_format_nested")

    # make a coilset with angular coilset
    N = 22
    coil = FourierPlanarCoil()
    coil.current = 1
    coilset_NFP = CoilSet(coil, NFP=N, sym=False)
    coilset_sym = CoilSet(
        FourierPlanarCoil(r_n=3, center=[10, 2 * np.pi / 7, 0], basis="rpz"),
        NFP=1,
        sym=True,
    )
    coilset = MixedCoilSet(coilset_NFP, coilset_sym)

    grid = LinearGrid(N=25, endpoint=False)
    coilset.save_in_makegrid_format(str(path), grid=grid, NFP=2)

    coilset2 = CoilSet.from_makegrid_coilfile(str(path))

    assert coilset2.num_coils == coilset.num_coils

    # check length of each coil
    # first 22 are the coilset_NFP coils
    correct_length = coilset_NFP.compute("length")[0]["length"]
    loaded_coil_lengths = [c.compute("length")["length"] for c in coilset2[:22]]
    np.testing.assert_allclose(correct_length, loaded_coil_lengths, rtol=1e-2)
    # last 2 are the coilset_sym coils
    correct_length = coilset_sym.compute("length")[0]["length"]
    loaded_coil_lengths = [c.compute("length")["length"] for c in coilset2[22:]]
    np.testing.assert_allclose(correct_length, loaded_coil_lengths, rtol=1e-2)


@pytest.mark.unit
def test_save_makegrid_coils_assert_NFP(tmpdir_factory):
    """Test saving CoilSet that with incompatible NFP throws an error."""
    Ncoils = 22
    input_path = f"./tests/inputs/coils.MAKEGRID_format_{Ncoils}_coils"
    tmpdir = tmpdir_factory.mktemp("coil_files")
    tmp_path = tmpdir.join("coils.MAKEGRID_format_{Ncoils}_coils")
    shutil.copyfile(input_path, tmp_path)

    coilset = CoilSet.from_makegrid_coilfile(str(tmp_path))
    assert len(coilset) == Ncoils  # correct number of coils
    path = tmpdir.join("coils.MAKEGRID_format_desc")
    assert len(coilset) % 3 != 0
    with pytest.raises(AssertionError):
        coilset.save_in_makegrid_format(str(path), NFP=3)


@pytest.mark.unit
def test_load_makegrid_coils_header_asserts(tmpdir_factory):
    """Test loading in CoilSets from incorrect MAKEGRID format files throws error."""
    Ncoils = 22
    input_path = f"./tests/inputs/coils.MAKEGRID_format_{Ncoils}_coils_header_too_long"
    tmpdir = tmpdir_factory.mktemp("coil_files")
    tmp_path = tmpdir.join("coils.MAKEGRID_format_{Ncoils}_coils_header_too_long")
    shutil.copyfile(input_path, tmp_path)
    with pytest.raises(IOError):
        CoilSet.from_makegrid_coilfile(str(tmp_path))
    input_path = f"./tests/inputs/coils.MAKEGRID_format_{Ncoils}_coils_header_too_short"
    shutil.copyfile(input_path, tmp_path)
    with pytest.raises(IOError):
        CoilSet.from_makegrid_coilfile(str(tmp_path))


@pytest.mark.unit
def test_repr():
    """Test string representation of Coil objects."""
    coil = FourierPlanarCoil()
    assert "FourierPlanarCoil" in str(coil)
    assert "current=1" in str(coil)

    coils = CoilSet.linspaced_angular(coil, n=4)
    assert "CoilSet" in str(coils)
    assert "4 submembers" in str(coils)

    coils.name = "MyCoils"
    assert "MyCoils" in str(coils)


@pytest.mark.unit
def test_linking_number():
    """Test calculation of linking number."""
    coil = FourierPlanarCoil(center=[10, 1, 0])
    grid = LinearGrid(N=25)
    # regular modular coilset from symmetry, so that there are 10 coils, half going
    # one way and half going the other way
    coilset = CoilSet(coil, NFP=5, sym=True)
    coil2 = FourierRZCoil()
    # add a coil along the axis that links all the other coils
    coilset2 = MixedCoilSet(coilset, coil2)
    link = coilset2._compute_linking_number(grid=grid)

    # modular coils don't link each other
    np.testing.assert_allclose(link[:-1, :-1], 0, atol=1e-14)
    # axis coil doesn't link itself
    np.testing.assert_allclose(link[-1, -1], 0, atol=1e-14)
    # we expect the axis coil to link all the modular coils, with alternating sign
    # due to alternating orientation of the coils due to symmetry.
    expected = [1, -1] * 5
    np.testing.assert_allclose(link[-1, :-1], expected, rtol=1e-3)


@pytest.mark.unit
def test_initialize_modular():
    """Test initializing a modular coilset."""
    eq = Equilibrium(NFP=2, sym=True)
    coilset = initialize_modular_coils(eq, 3, 2.0)
    assert len(coilset) == 3
    np.testing.assert_allclose(coilset[0].r_n, 2.0)  # a=1, so r/a of 2 gives r=2
    x = coilset[1]._compute_position(basis="rpz")[0]
    # eq is axisymmetric so coils should each be at const zeta
    # with symmetry and 2 field periods, each half period goes from 0 to pi/2
    # with 3 coils, one coil should be right in the middle at pi/4
    np.testing.assert_allclose(x[:, 1], np.pi / 4)
    np.testing.assert_allclose(x[:, 0].min(), 8, rtol=1e-2)  # Rmin ~ 10-2
    np.testing.assert_allclose(x[:, 0].max(), 12, rtol=1e-2)  # Rmax ~ 10+2
    y = coilset._compute_position()
    assert len(y) == 12  # 3 coils/fp * 2 fp * 2 sym


@pytest.mark.unit
def test_initialize_saddle():
    """Test initializing a saddle coilset."""
    eq = Equilibrium(NFP=2, sym=False)
    coilset = initialize_saddle_coils(eq, 3, offset=2.0, r_over_a=1.0, position="inner")
    assert len(coilset) == 3
    y = coilset._compute_position()
    assert len(y) == 6  # 3 coils/fp * 2 fp
    np.testing.assert_allclose(coilset[0].r_n, 1.0)  # a=1, so r/a of 1 gives r=1
    x = coilset[1]._compute_position(grid=LinearGrid(N=50), basis="xyz")[0]
    # 2 field periods, each half period goes from 0 to pi
    # with 3 coils, one coil should be right in the middle at pi/2, eg parallel to
    # x axis
    np.testing.assert_allclose(x[:, 1], 8)  # R ~ 10-2
    np.testing.assert_allclose(x[:, 0].min(), -1, rtol=1e-2)  # xmin
    np.testing.assert_allclose(x[:, 0].max(), 1, rtol=1e-2)  # xmax

    coilset = initialize_saddle_coils(eq, 1, offset=2.0, r_over_a=1.0, position="outer")
    assert len(coilset) == 1
    y = coilset._compute_position()
    assert len(y) == 2  # 3 coils/fp * 2 fp
    np.testing.assert_allclose(coilset[0].r_n, 1.0)  # a=1, so r/a of 1 gives r=1
    x = coilset[0]._compute_position(grid=LinearGrid(N=50), basis="xyz")[0]
    # 2 field periods, each half period goes from 0 to pi
    # with 1 coils, it should be at pi/2
    np.testing.assert_allclose(x[:, 1], 12)  # R ~ 10+2
    np.testing.assert_allclose(x[:, 0].min(), -1, rtol=1e-2)  # xmin
    np.testing.assert_allclose(x[:, 0].max(), 1, rtol=1e-2)  # xmax

    offset = 3.0
    coilset = initialize_saddle_coils(
        eq, 1, offset=offset, r_over_a=1.0, position="top"
    )
    assert len(coilset) == 1
    y = coilset._compute_position()
    assert len(y) == 2  # 3 coils/fp * 2 fp
    np.testing.assert_allclose(coilset[0].r_n, 1.0)  # a=1, so r/a of 1 gives r=1
    x = coilset[0]._compute_position(grid=LinearGrid(N=50), basis="xyz")[0]
    # 2 field periods, each half period goes from 0 to pi
    # with 1 coils, it should be at pi/2
    np.testing.assert_allclose(x[:, 2], offset)  # Z ~ 3
    np.testing.assert_allclose(x[:, 0].min(), -1, rtol=1e-2)  # xmin
    np.testing.assert_allclose(x[:, 0].max(), 1, rtol=1e-2)  # xmax

    coilset = initialize_saddle_coils(
        eq, 1, offset=offset, r_over_a=1.0, position="bottom"
    )
    assert len(coilset) == 1
    y = coilset._compute_position()
    assert len(y) == 2  # 3 coils/fp * 2 fp
    np.testing.assert_allclose(coilset[0].r_n, 1.0)  # a=1, so r/a of 1 gives r=1
    x = coilset[0]._compute_position(grid=LinearGrid(N=50), basis="xyz")[0]
    # 2 field periods, each half period goes from 0 to pi
    # with 1 coils, it should be at pi/2
    np.testing.assert_allclose(x[:, 2], -offset)  # Z ~ -3
    np.testing.assert_allclose(x[:, 0].min(), -1, rtol=1e-2)  # xmin
    np.testing.assert_allclose(x[:, 0].max(), 1, rtol=1e-2)  # xmax


@pytest.mark.unit
def test_initialize_helical():
    """Test initializing a helical coilset."""
    eq = get("NCSX")
    coilset = initialize_helical_coils(eq, 2, r_over_a=2.0, helicity=(3, 1), npts=100)
    assert len(coilset) == 2
    obj = LinkingCurrentConsistency(eq, coilset)
    obj.build()
    np.testing.assert_allclose(
        obj.compute(coilset.params_dict, eq.params_dict), 0, atol=1e-8
    )
    assert obj.constants["link"][0] == 9  # M=3 per period * 3 periods

    coils_pts = coilset._compute_position()
    a = eq.compute("a")["a"]
    data = eq.compute(
        ["R", "phi", "Z"],
        grid=LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP),
    )
    rpz = jnp.array([data["R"], data["phi"], data["Z"]]).T
    rpz = copy_rpz_periods(rpz, eq.NFP)
    plasma_pts = rpz2xyz(rpz)
    dist = np.linalg.norm(coils_pts[:, None, :, :] - plasma_pts[:, None, :], axis=-1)
    # dist is distance from every point on the plasma to every point on the coil
    # first take a min over plasma pts to get distance from each coil pt to plasma
    # then we expect the avg of that to be ~a since r/a=2 so offset is 1*a
    np.testing.assert_allclose(dist.min(axis=1).mean(axis=-1), a, rtol=3e-2)
