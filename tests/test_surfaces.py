"""Tests for Surface classes."""

import numpy as np
import pytest

import desc.examples
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface, ZernikeRZToroidalSection
from desc.grid import LinearGrid
from desc.utils import rpz2xyz


class TestFourierRZToroidalSurface:
    """Tests for FourierRZToroidalSurface class."""

    @pytest.mark.unit
    def test_area(self):
        """Test calculation of surface area."""
        s = FourierRZToroidalSurface()
        grid = LinearGrid(M=24, N=24)
        area = 4 * np.pi**2 * 10
        np.testing.assert_allclose(s.compute("S", grid=grid)["S"], area)

    @pytest.mark.unit
    def test_compute_ndarray_error(self):
        """Test raising TypeError if ndarray is passed in."""
        s = FourierRZToroidalSurface()
        with pytest.raises(TypeError):
            s.compute("S", grid=1)
        with pytest.raises(TypeError):
            s.compute("S", grid=np.linspace(0, 1, 10))

    @pytest.mark.unit
    def test_normal(self):
        """Test calculation of surface normal vector."""
        s = FourierRZToroidalSurface()
        grid = LinearGrid(theta=np.pi / 2, zeta=np.pi)
        N = s.compute("n_rho", grid=grid)["n_rho"]
        np.testing.assert_allclose(N[0], [0, 0, -1], atol=1e-14)
        grid = LinearGrid(theta=0.0, zeta=0.0)
        N = s.compute("n_rho", grid=grid)["n_rho"]
        np.testing.assert_allclose(N[0], [1, 0, 0], atol=1e-12)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting attributes of surface."""
        c = FourierRZToroidalSurface()

        R, Z = c.get_coeffs(0, 0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 0, 5, None)
        c.set_coeffs(-1, 0, None, 2)
        np.testing.assert_allclose(c.R_lmn, [5, 1])
        np.testing.assert_allclose(c.Z_lmn, [2])

        s = c.copy()
        assert s.equiv(c)

        c.change_resolution(0, 5, 5)
        with pytest.raises(ValueError):
            c.R_lmn = s.R_lmn
        with pytest.raises(ValueError):
            c.Z_lmn = s.Z_lmn

        c.name = "my curve"
        assert "my" in c.name
        assert c.name in str(c)
        assert "FourierRZToroidalSurface" in str(c)

        # test assert statement for array sizes matching
        with pytest.raises(AssertionError):
            c = FourierRZToroidalSurface(
                R_lmn=np.array([1, 2, 3]), modes_R=np.array([[0, 0], [1, 0]])
            )
        with pytest.raises(AssertionError):
            c = FourierRZToroidalSurface(
                Z_lmn=np.array([1, 2, 3]), modes_Z=np.array([[0, 0], [1, 0]])
            )

    @pytest.mark.unit
    def test_from_input_file(self):
        """Test reading a surface from a vmec or desc input file."""
        vmec_path = ".//tests//inputs//input.DSHAPE"
        desc_path = ".//tests//inputs//DSHAPE"
        with pytest.warns(UserWarning):
            vmec_surf = FourierRZToroidalSurface.from_input_file(vmec_path)
        desc_surf = FourierRZToroidalSurface.from_input_file(desc_path)
        true_surf = desc.examples.get("DSHAPE", "boundary")

        vmec_surf.change_resolution(M=6, N=0)
        desc_surf.change_resolution(M=6, N=0)
        true_surf.change_resolution(M=6, N=0)

        np.testing.assert_allclose(
            true_surf.R_lmn, vmec_surf.R_lmn, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            true_surf.Z_lmn, vmec_surf.Z_lmn, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            true_surf.R_lmn, desc_surf.R_lmn, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            true_surf.Z_lmn, desc_surf.Z_lmn, atol=1e-10, rtol=1e-10
        )

    @pytest.mark.unit
    def test_from_near_axis(self):
        """Test constructing approximate QI surface from near axis parameters."""
        surf = FourierRZToroidalSurface.from_qp_model(1, 10, 4, 0.3, 0.2)
        np.testing.assert_allclose(
            surf.R_lmn,
            np.array([-0.075, 0, 1, -0.125, 0, 0.030707, -0.2, -0.075]),
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            surf.Z_lmn,
            np.array([0.2, 0.075, 0, 0, 0.125, -0.007677, -0.075]),
            rtol=1e-4,
            atol=1e-6,
        )

    @pytest.mark.unit
    def test_curvature(self):
        """Tests for gaussian, mean, principle curvatures."""
        s = FourierRZToroidalSurface()
        grid = LinearGrid(theta=np.pi / 2, zeta=np.pi)
        data = s.compute(
            [
                "curvature_K_rho",
                "curvature_H_rho",
                "curvature_k1_rho",
                "curvature_k2_rho",
            ],
            grid=grid,
        )
        np.testing.assert_allclose(data["curvature_K_rho"], 0)
        np.testing.assert_allclose(data["curvature_H_rho"], -1 / 2)
        np.testing.assert_allclose(data["curvature_k1_rho"], 0)
        np.testing.assert_allclose(data["curvature_k2_rho"], -1)

    @pytest.mark.unit
    def test_constant_offset_surface_circle(self):
        """Test constant offset algorithm for a circular torus."""
        s = FourierRZToroidalSurface()
        grid = LinearGrid(M=3, N=2)
        offset = 1
        (s_offset, data, _) = s.constant_offset_surface(
            offset, grid, M=1, N=1, full_output=True
        )
        r_offset_surf = data["x_offset_surface"]
        r_surf = data["x"]
        dists = np.linalg.norm(r_surf - r_offset_surf, axis=1)
        np.testing.assert_allclose(dists, 1, atol=1e-16)
        R00_offset_ind = s_offset.R_basis.get_idx(M=0, N=0)
        R00_offset = s_offset.R_lmn[R00_offset_ind]
        R10_offset_ind = s_offset.R_basis.get_idx(M=1, N=0)
        R10_offset = s_offset.R_lmn[R10_offset_ind]
        Zneg10_offset_ind = s_offset.Z_basis.get_idx(M=-1, N=0)
        Zneg10_offset = s_offset.Z_lmn[Zneg10_offset_ind]

        np.testing.assert_allclose(R00_offset, 10)
        np.testing.assert_allclose(R10_offset, 2)
        np.testing.assert_allclose(Zneg10_offset, -2)
        np.testing.assert_allclose(
            np.delete(
                s_offset.R_lmn,
                np.array([R00_offset_ind, R10_offset_ind]),
            ),
            0,
            atol=9e-15,
        )
        np.testing.assert_allclose(
            np.delete(
                s_offset.Z_lmn,
                Zneg10_offset_ind,
            ),
            0,
            atol=9e-15,
        )
        grid_compute = LinearGrid(M=10, N=10)
        data = s.compute(["x", "e_theta", "e_zeta"], basis="rpz", grid=grid_compute)
        data_offset = s_offset.compute(
            ["x", "e_theta", "e_zeta"], basis="rpz", grid=grid_compute
        )
        dists = np.linalg.norm(data["x"] - data_offset["x"], axis=1)
        np.testing.assert_allclose(dists, 1, atol=1e-16)
        correct_data_offset = {
            "e_theta": np.vstack(
                (
                    -2 * np.sin(grid_compute.nodes[:, 1]),
                    np.zeros_like(grid_compute.nodes[:, 1]),
                    -2 * np.cos(grid_compute.nodes[:, 1]),
                )
            ).T,
            "e_zeta": np.vstack(
                (
                    np.zeros_like(grid_compute.nodes[:, 1]),
                    data_offset["x"][:, 0],
                    np.zeros_like(grid_compute.nodes[:, 1]),
                )
            ).T,
        }
        for key in ["e_theta", "e_zeta"]:
            np.testing.assert_allclose(
                correct_data_offset[key],
                data_offset[key],
                atol=1e-4,
                err_msg=f"Failed test at comparison of {key}",
            )

    @pytest.mark.slow
    @pytest.mark.unit
    def test_constant_offset_surface_rot_ellipse(self):
        """Test constant offset algorithm for a rotating ellipse."""
        eq = desc.examples.get("HELIOTRON")
        s = eq.surface
        s.change_resolution(M=2, N=2)
        offset = 0.1
        (s_offset, data, _) = s.constant_offset_surface(
            offset, grid=None, M=2, N=2, full_output=True
        )
        r_offset_surf = data["x_offset_surface"]
        r_surf = data["x"]

        dists = np.linalg.norm(rpz2xyz(r_surf) - rpz2xyz(r_offset_surf), axis=1)

        np.testing.assert_allclose(dists, offset, atol=1e-16)

        R00_offset_ind = s_offset.R_basis.get_idx(M=0, N=0)
        R00_offset = s_offset.R_lmn[R00_offset_ind]

        R10_offset_ind = s_offset.R_basis.get_idx(M=1, N=0)
        R10_offset = s_offset.R_lmn[R10_offset_ind]
        R11_offset_ind = s_offset.R_basis.get_idx(M=1, N=1)
        R11_offset = s_offset.R_lmn[R11_offset_ind]
        Rneg1neg1_offset_ind = s_offset.R_basis.get_idx(M=-1, N=-1)
        Rneg1neg1_offset = s_offset.R_lmn[Rneg1neg1_offset_ind]

        Zneg10_offset_ind = s_offset.Z_basis.get_idx(M=-1, N=0)
        Zneg10_offset = s_offset.Z_lmn[Zneg10_offset_ind]
        Zneg11_offset_ind = s_offset.Z_basis.get_idx(M=-1, N=1)
        Zneg11_offset = s_offset.Z_lmn[Zneg11_offset_ind]
        Z1neg1_offset_ind = s_offset.Z_basis.get_idx(M=1, N=-1)
        Z1neg1_offset = s_offset.Z_lmn[Z1neg1_offset_ind]

        # a is the cos or sin (theta) term for the ellipse
        # b is the cos(theta)*sin(phi) mixed terms for the ellipse
        a = 1
        b = 0.3

        np.testing.assert_allclose(
            R00_offset,
            s.R_lmn[s.R_basis.get_idx(M=0, N=0)],
            atol=1e-3,
        )
        np.testing.assert_allclose(R10_offset, -a - offset, atol=1e-2)
        np.testing.assert_allclose(R11_offset, -b, atol=3e-2)
        np.testing.assert_allclose(Rneg1neg1_offset, b, atol=3e-2)

        np.testing.assert_allclose(Zneg10_offset, a + offset, atol=1e-2)
        np.testing.assert_allclose(Zneg11_offset, -b, atol=3e-2)
        np.testing.assert_allclose(Z1neg1_offset, -b, atol=3e-2)

        np.testing.assert_allclose(
            R00_offset,
            s.R_lmn[s.R_basis.get_idx(M=0, N=0)],
            atol=1e-3,
        )
        # cannot do the same sort of distance test as the axisymmetric test
        # because of the non-axisymmetry making the unit normal vector not
        # purely having R,Z components (and the offset is along the normal vector).
        # so, we cannot know which points on the surface to compare to see if
        # the distance is = offset without doing rootfinding (like is done to
        # find the surface in the first place)

        # we can check that that avg XS area is what we expect
        semi_major = a + b
        semi_minor = a - b
        correct_offset_XS_area = np.pi * (offset + semi_major) * (offset + semi_minor)
        np.testing.assert_allclose(
            Equilibrium(surface=s_offset).compute("A")["A"],
            correct_offset_XS_area,
            rtol=2e-2,
        )

    @pytest.mark.unit
    def test_position(self):
        """Tests for position on surface."""
        s = FourierRZToroidalSurface()
        grid = LinearGrid(theta=0, zeta=np.pi)
        data = s.compute(["x", "R", "phi", "Z"], grid=grid, basis="xyz")
        np.testing.assert_allclose(data["R"], 11)
        np.testing.assert_allclose(data["x"][0, 0], -11)
        np.testing.assert_allclose(data["phi"], np.pi)
        np.testing.assert_allclose(data["x"][0, 1], 0, atol=1e-14)  # this is y
        np.testing.assert_allclose(data["Z"], 0)
        np.testing.assert_allclose(data["x"][0, 2], 0)

    @pytest.mark.unit
    def test_surface_from_values(self):
        """Test for constructing elliptical surface from values."""
        surface = get("HELIOTRON", "boundary")
        grid = LinearGrid(M=20, N=20, sym=False, NFP=surface.NFP, endpoint=False)
        data = surface.compute(["R", "phi", "Z"], grid=grid)

        theta = grid.nodes[:, 1]

        coords = np.vstack([data["R"], data["phi"], data["Z"]]).T
        surface2 = FourierRZToroidalSurface.from_values(
            coords,
            theta,
            M=surface.M,
            N=surface.N,
            NFP=surface.NFP,
            sym=True,
            w=np.ones_like(theta),
        )
        grid = LinearGrid(M=25, N=25, sym=False, NFP=surface.NFP)
        np.testing.assert_allclose(
            surface.compute("x", grid=grid)["x"], surface2.compute("x", grid=grid)["x"]
        )

        # with a different poloidal angle
        theta = -np.arctan2(data["Z"] - 0, data["R"] - 10)
        surface2 = FourierRZToroidalSurface.from_values(
            coords,
            theta,
            M=surface.M,
            N=surface.N,
            NFP=surface.NFP,
            sym=True,
        )
        # cannot compare x directly because thetas are different
        np.testing.assert_allclose(
            surface.compute("V")["V"], surface2.compute("V")["V"], rtol=1e-4
        )
        np.testing.assert_allclose(
            surface.compute("S")["S"], surface2.compute("S")["S"], rtol=1e-3
        )

        # test assert statements
        with pytest.raises(NotImplementedError):
            FourierRZToroidalSurface.from_values(
                coords,
                theta,
                zeta=theta,
                M=surface.M,
                N=surface.N,
                NFP=surface.NFP,
                sym=True,
            )

    @pytest.mark.unit
    def test_surface_from_shape_parameters(self):
        """Test that making a surface with specified R0,a etc gives correct shape."""
        R0 = 8
        a = 2
        e = 2.1
        # basic rotating ellipse, parameters should be ~exact
        surf = FourierRZToroidalSurface.from_shape_parameters(
            major_radius=R0,
            aspect_ratio=R0 / a,
            elongation=e,
            triangularity=0.0,
            squareness=0,
            eccentricity=0,
            torsion=0,
            twist=1,
            NFP=2,
            sym=True,
        )
        eq = Equilibrium(surface=surf)
        np.testing.assert_allclose(R0, eq.compute("R0")["R0"], rtol=1e-8)
        np.testing.assert_allclose(a, eq.compute("a")["a"], rtol=1e-8)
        np.testing.assert_allclose(
            e, eq.compute("a_major/a_minor")["a_major/a_minor"], rtol=1e-4
        )

        # slightly more complex shape, parameters are only approximate
        surf = FourierRZToroidalSurface.from_shape_parameters(
            major_radius=R0,
            aspect_ratio=R0 / a,
            elongation=e,
            triangularity=0.3,
            squareness=0.1,
            eccentricity=0.1,
            torsion=0.2,
            twist=1,
            NFP=2,
            sym=True,
        )
        eq = Equilibrium(surface=surf)
        np.testing.assert_allclose(R0, eq.compute("R0")["R0"], rtol=1e-2)
        np.testing.assert_allclose(a, eq.compute("a")["a"], rtol=5e-2)
        np.testing.assert_allclose(
            e, eq.compute("a_major/a_minor")["a_major/a_minor"], rtol=2e-2
        )


class TestZernikeRZToroidalSection:
    """Tests for ZernikeRZToroidalSection class."""

    @pytest.mark.unit
    def test_area(self):
        """Test calculation of surface area."""
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(L=10, M=10)
        area = np.pi * 1**2
        np.testing.assert_allclose(s.compute("A", grid=grid)["A"], area)

    @pytest.mark.unit
    def test_normal(self):
        """Test calculation of surface normal vector."""
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(L=8, M=4, N=0, axis=False)
        N = s.compute("n_zeta", grid=grid)["n_zeta"]
        np.testing.assert_allclose(N, np.broadcast_to([0, 1, 0], N.shape), atol=1e-12)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting surface attributes."""
        c = ZernikeRZToroidalSection()

        R, Z = c.get_coeffs(0, 0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 0, 5, None)
        c.set_coeffs(1, -1, None, 2)
        np.testing.assert_allclose(c.R_lmn, [5, 1])
        np.testing.assert_allclose(c.Z_lmn, [2])
        with pytest.raises(ValueError):
            c.set_coeffs(0, 0, None, 2)
        s = c.copy()
        assert s.equiv(c)

        c.change_resolution(5, 5, 0)
        with pytest.raises(ValueError):
            c.R_lmn = s.R_lmn
        with pytest.raises(ValueError):
            c.Z_lmn = s.Z_lmn

        assert c.sym

    @pytest.mark.unit
    def test_curvature(self):
        """Tests for gaussian, mean, principle curvatures.

        (kind of pointless since it's a flat surface so its always 0)
        """
        s = ZernikeRZToroidalSection()
        grid = LinearGrid(theta=np.pi / 2, rho=0.5)
        data = s.compute(
            [
                "curvature_K_zeta",
                "curvature_H_zeta",
                "curvature_k1_zeta",
                "curvature_k2_zeta",
            ],
            grid=grid,
        )
        np.testing.assert_allclose(data["curvature_K_zeta"], 0)
        np.testing.assert_allclose(data["curvature_H_zeta"], 0)
        np.testing.assert_allclose(data["curvature_k1_zeta"], 0)
        np.testing.assert_allclose(data["curvature_k2_zeta"], 0)


@pytest.mark.unit
def test_surface_orientation():
    """Tests for computing the orientation of a surface in weird edge cases."""
    # this has the axis outside the surface, and negative orientation
    Rb = np.array([3.41, 0.8, 0.706, -0.3])
    R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    Zb = np.array([0.0, -0.16, 1.47])
    Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
    surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes, check_orientation=False)
    assert surf._compute_orientation() == -1
    eq = Equilibrium(
        M=surf.M, N=surf.N, surface=surf, check_orientation=False, ensure_nested=False
    )
    assert np.sign(eq.compute("sqrt(g)")["sqrt(g)"].mean()) == -1

    # same surface but flipped to have positive orientation
    Rb = np.array([3.41, 0.8, 0.706, -0.3])
    R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    Zb = np.array([0.0, 0.16, -1.47])
    Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
    surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes, check_orientation=False)
    assert surf._compute_orientation() == 1
    eq = Equilibrium(
        M=surf.M, N=surf.N, surface=surf, check_orientation=False, ensure_nested=False
    )
    assert np.sign(eq.compute("sqrt(g)")["sqrt(g)"].mean()) == 1

    # this has theta=0 on inboard side and positive orientation
    Rb = np.array([3.51, -1.3, -0.506, 0.1])
    R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    Zb = np.array([0.0, -0.16, 1.47])
    Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
    surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes, check_orientation=False)
    assert surf._compute_orientation() == 1
    eq = Equilibrium(M=surf.M, N=surf.N, surface=surf, check_orientation=False)
    assert np.sign(eq.compute("sqrt(g)")["sqrt(g)"].mean()) == 1

    # same but with negative orientation
    Rb = np.array([3.51, -1.3, -0.506, 0.1])
    R_modes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    Zb = np.array([0.0, 0.16, -1.47])
    Z_modes = np.array([[-3, 0], [-2, 0], [-1, 0]])
    surf = FourierRZToroidalSurface(Rb, Zb, R_modes, Z_modes, check_orientation=False)
    assert surf._compute_orientation() == -1
    eq = Equilibrium(M=surf.M, N=surf.N, surface=surf, check_orientation=False)
    assert np.sign(eq.compute("sqrt(g)")["sqrt(g)"].mean()) == -1


@pytest.mark.unit
def test_surface_change_only_symmetry():
    """Test that sym correctly changes when only sym is passed to change_resolution."""
    surf = FourierRZToroidalSurface(sym=False)
    surf.change_resolution(sym=True)
    assert surf.sym
    assert surf.R_basis.sym == "cos"
    assert surf.Z_basis.sym == "sin"

    surf.change_resolution(sym=False)
    assert surf.sym is False
    assert surf.R_basis.sym is False
    assert surf.Z_basis.sym is False


@pytest.mark.unit
def test_section_change_only_symmetry():
    """Test that sym correctly changes when only sym is passed to change_resolution."""
    surf = ZernikeRZToroidalSection(sym=False)
    surf.change_resolution(sym=True)
    assert surf.sym
    assert surf.R_basis.sym == "cos"
    assert surf.Z_basis.sym == "sin"

    surf.change_resolution(sym=False)
    assert surf.sym is False
    assert surf.R_basis.sym is False
    assert surf.Z_basis.sym is False
