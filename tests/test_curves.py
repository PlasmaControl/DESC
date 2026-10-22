"""Tests for different Curve classes."""

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.optimize import brentq

from desc.compute.spline_utils import b_p_deriv3, uniform_knots
from desc.equilibrium import Equilibrium
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierXYCurve,
    FourierXYZCurve,
    NurbsRPZCurve,
    SplineXYZCurve,
)
from desc.grid import Grid, LinearGrid
from desc.io import InputReader
from desc.utils import rpz2xyz, rpz2xyz_vec


class TestFourierRZCurve:
    """Tests for FourierRZCurve class."""

    @pytest.mark.unit
    def test_center(self):
        """Test center of curve."""
        c = FourierRZCurve(R_n=[-2, 10, 4], Z_n=[1, 3, 2])
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [2, -1, 3]
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [-3, 0, 4], atol=1e-15
        )

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        c = FourierRZCurve()
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], 10 * 2 * np.pi
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], 10 * 2 * np.pi
        )

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / 10)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / 10)

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute("torsion", grid=20)["torsion"], 0)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute("torsion", grid=20)["torsion"], 0)

    @pytest.mark.unit
    def test_frenet(self):
        """Test frenet-serret frame of circular curve."""
        c = FourierRZCurve()
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="xyz", grid=0
        )
        T, N, B = data["frenet_tangent"], data["frenet_normal"], data["frenet_binormal"]
        np.testing.assert_allclose(T, np.array([[0, 1, 0]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 0, 1]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="xyz", grid=0
        )
        T, N, B = data["frenet_tangent"], data["frenet_normal"], data["frenet_binormal"]
        np.testing.assert_allclose(T, np.array([[0, 1, 0]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 0, 1]]), atol=1e-12)

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        c = FourierRZCurve()
        x, y, z = c.compute("x", grid=0, basis="xyz")["x"].T
        np.testing.assert_allclose(x, 10)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        r, p, z = c.compute("x", grid=0, basis="rpz")["x"].T
        np.testing.assert_allclose(r, np.sqrt(1**2 + 9**2))
        np.testing.assert_allclose(p, np.arctan2(-9, 1))
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of FourierRZCurve."""
        c = FourierRZCurve()

        R, Z = c.get_coeffs(0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 5, None)
        np.testing.assert_allclose(c.R_n, [5])
        np.testing.assert_allclose(c.Z_n, [])

        s = c.copy()
        assert s.equiv(c)

        c.change_resolution(5)
        assert c.N == 5
        c.set_coeffs(-1, None, 2)
        np.testing.assert_allclose(c.R_n, [5, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(c.Z_n, [0, 0, 0, 0, 2])

        with pytest.raises(ValueError):
            c.R_n = s.R_n
        with pytest.raises(ValueError):
            c.Z_n = s.Z_n

        c.name = "my curve"
        assert "my" in c.name
        assert c.name in str(c)
        assert "FourierRZCurve" in str(c)
        assert c.sym

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking when creating FourierRZCurve."""
        with pytest.raises(ValueError):
            _ = FourierRZCurve(R_n=[])
        with pytest.raises(AssertionError):
            _ = FourierRZCurve(R_n=[1], modes_R=[1, 2])
        with pytest.raises(AssertionError):
            _ = FourierRZCurve(Z_n=[1], modes_Z=[1, 2])

    @pytest.mark.unit
    def test_to_FourierXYZCurve(self):
        """Test conversion to FourierXYZCurve."""
        rz = FourierRZCurve(R_n=[0, 10, 1], Z_n=[-1, 0, 0])
        grid = LinearGrid(N=20, endpoint=False)
        xyz = rz.to_FourierXYZ(N=2, grid=grid, s=grid.nodes[:, 2])

        np.testing.assert_allclose(
            rz.compute("curvature", grid=grid)["curvature"],
            xyz.compute("curvature", grid=grid)["curvature"],
        )
        np.testing.assert_allclose(
            rz.compute("torsion", grid=grid)["torsion"],
            xyz.compute("torsion", grid=grid)["torsion"],
        )
        np.testing.assert_allclose(
            rz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
        )
        np.testing.assert_allclose(
            rz.compute("x", grid=grid, basis="xyz")["x"],
            xyz.compute("x", basis="xyz", grid=grid)["x"],
            atol=1e-12,
        )
        # same thing but pass in a closed grid
        grid = LinearGrid(N=20, endpoint=True)
        xyz = rz.to_FourierXYZ(N=2, grid=grid, s=grid.nodes[:, 2])

        np.testing.assert_allclose(
            rz.compute("curvature", grid=grid)["curvature"],
            xyz.compute("curvature", grid=grid)["curvature"],
        )
        np.testing.assert_allclose(
            rz.compute("torsion", grid=grid)["torsion"],
            xyz.compute("torsion", grid=grid)["torsion"],
        )
        np.testing.assert_allclose(
            rz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
        )
        np.testing.assert_allclose(
            rz.compute("x", grid=grid, basis="xyz")["x"],
            xyz.compute("x", basis="xyz", grid=grid)["x"],
            atol=1e-12,
        )

        # same thing but with arclength angle
        grid = LinearGrid(N=20, endpoint=False)
        xyz = rz.to_FourierXYZ(N=2, grid=grid, s="arclength")

        np.testing.assert_allclose(
            rz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
            atol=3e-3,
        )

        # pass in non-monotonic s
        grid = LinearGrid(N=20, endpoint=False)
        s = grid.nodes[:, 2]
        s[-2] = s[-1]
        with pytest.raises(ValueError):
            xyz = rz.to_FourierXYZ(N=2, grid=grid, s=s)

    @pytest.mark.unit
    def test_to_SplineXYZCurve(self):
        """Test conversion to SplineXYZCurve."""
        rz = FourierRZCurve(R_n=[0, 10, 1], Z_n=[-1, 0, 0])
        xyz = rz.to_SplineXYZ(grid=500, knots="arclength")

        grid = LinearGrid(N=20, endpoint=False)

        np.testing.assert_allclose(
            rz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
            atol=1e-2,
        )
        coords_xyz = np.asarray(xyz.compute("x", basis="rpz", grid=grid)["x"])
        phi_xyz = (coords_xyz[:, 1] + 1e-4) % (2 * np.pi)
        coords_rpz = rz.compute("x", grid=grid, basis="rpz")["x"]
        phi_rpz = (coords_rpz[:, 1] + 1e-4) % (2 * np.pi)
        np.testing.assert_allclose(
            coords_rpz[:, 0::2],
            coords_xyz[:, 0::2],
            atol=1e-1,
        )
        np.testing.assert_allclose(
            phi_rpz,
            phi_xyz,
            atol=1e-1,
        )

    @pytest.mark.unit
    def test_from_input_file(self):
        """Test getting a curve from axis guess in input file."""
        path = "tests/inputs/input.QSC_r2_5.5_desc"

        curve1 = FourierRZCurve.from_input_file(path)
        curve2 = Equilibrium(**InputReader(path).inputs[0], check_kwargs=False).axis
        curve1.change_resolution(curve2.N)

        np.testing.assert_allclose(curve1.R_n, curve2.R_n)
        np.testing.assert_allclose(curve1.Z_n, curve2.Z_n)
        np.testing.assert_allclose(curve1.NFP, curve2.NFP)
        np.testing.assert_allclose(curve1.sym, curve2.sym)

        path = "tests/inputs/input.QSC_r2_5.5_vmec"

        with pytest.warns(UserWarning):
            curve3 = FourierRZCurve.from_input_file(path)
            curve4 = Equilibrium(**InputReader(path).inputs[0], check_kwargs=False).axis
        curve3.change_resolution(curve4.N)

        np.testing.assert_allclose(curve3.R_n, curve4.R_n)
        np.testing.assert_allclose(curve3.Z_n, curve4.Z_n)
        np.testing.assert_allclose(curve3.NFP, curve4.NFP)
        np.testing.assert_allclose(curve3.sym, curve4.sym)

    @pytest.mark.unit
    def test_to_FourierRZCurve(self):
        """Test conversion to FourierRZCurve."""
        xyz = FourierXYZCurve(modes=[-1, 1], X_n=[0, 10], Y_n=[10, 0], Z_n=[0, 0])
        grid = LinearGrid(N=10, endpoint=False)
        # convert back and check now
        rzz = xyz.to_FourierRZ(N=2, grid=grid)
        np.testing.assert_allclose(rzz.R_n[rzz.R_basis.get_idx(0)], 10)
        np.testing.assert_allclose(rzz.Z_n, 0)

        np.testing.assert_allclose(
            rzz.compute("curvature", grid=grid)["curvature"],
            xyz.compute("curvature", grid=grid)["curvature"],
        )
        np.testing.assert_allclose(
            rzz.compute("torsion", grid=grid)["torsion"],
            xyz.compute("torsion", grid=grid)["torsion"],
        )
        np.testing.assert_allclose(
            rzz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
        )
        np.testing.assert_allclose(
            rzz.compute("x", grid=grid, basis="xyz")["x"],
            xyz.compute("x", basis="xyz", grid=grid)["x"],
            atol=1e-12,
        )

        # same thing but pass in a closed grid
        grid = LinearGrid(N=10, endpoint=True)
        rzz = xyz.to_FourierRZ(N=2, grid=grid)
        np.testing.assert_allclose(rzz.R_n[rzz.R_basis.get_idx(0)], 10)
        np.testing.assert_allclose(rzz.Z_n, 0)
        np.testing.assert_allclose(
            rzz.compute("curvature", grid=grid)["curvature"],
            xyz.compute("curvature", grid=grid)["curvature"],
        )
        np.testing.assert_allclose(
            rzz.compute("torsion", grid=grid)["torsion"],
            xyz.compute("torsion", grid=grid)["torsion"],
        )
        np.testing.assert_allclose(
            rzz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
        )
        np.testing.assert_allclose(
            rzz.compute("x", grid=grid, basis="xyz")["x"],
            xyz.compute("x", basis="xyz", grid=grid)["x"],
            atol=1e-12,
        )
        # pass in non-monotonic phi
        phi_non_monotonic = np.array([0, 3, 2, 4, 1])
        grid = Grid(
            np.vstack(
                [
                    np.zeros_like(phi_non_monotonic),
                    np.zeros_like(phi_non_monotonic),
                    phi_non_monotonic,
                ]
            ).T,
            sort=False,
        )
        with pytest.raises(ValueError):
            xyz.to_FourierRZ(N=1, grid=grid)

    @pytest.mark.unit
    def test_change_symmetry(self):
        """Test correct sym changes when only sym is passed to change_resolution."""
        c = FourierRZCurve(sym=False)
        c.change_resolution(sym=True)
        assert c.sym
        assert c.R_basis.sym == "cos"
        assert c.Z_basis.sym == "sin"

        c.change_resolution(sym=False)
        assert c.sym is False
        assert c.R_basis.sym is False
        assert c.Z_basis.sym is False

    @pytest.mark.unit
    def test_from_values_np(self):
        """Test that from_values works when numpy array passed."""
        phis = np.linspace(0, 2 * np.pi, 4)
        coords = np.vstack([np.cos(phis), phis, np.zeros_like(phis)]).T
        c = FourierRZCurve.from_values(coords, N=1)
        np.testing.assert_allclose(c.R_n[c.R_basis.get_idx(N=1)], 1.0)


class TestFourierXYZCurve:
    """Tests for FourierXYZCurve class."""

    @pytest.mark.unit
    def test_center(self):
        """Test center of curve."""
        c = FourierXYZCurve(X_n=[1, 10, 2], Y_n=[0, -3, 1], Z_n=[-2, 2, 3])
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [10, -3, 2]
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [-11, -2, 3]
        )

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        c = FourierXYZCurve()
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], 2 * 2 * np.pi
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], 2 * 2 * np.pi
        )

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / 2)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / 2)

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        c = FourierXYZCurve(modes=[-1, 0, 1])
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-12
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-12
        )

    @pytest.mark.unit
    def test_frenet(self):
        """Test frenet-serret frame of circular curve."""
        c = FourierXYZCurve()
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="xyz", grid=0
        )
        T, N, B = data["frenet_tangent"], data["frenet_normal"], data["frenet_binormal"]
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="xyz", grid=0
        )
        T, N, B = data["frenet_tangent"], data["frenet_normal"], data["frenet_binormal"]
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        c = FourierXYZCurve()
        x, y, z = c.compute("x", grid=0, basis="xyz")["x"].T
        np.testing.assert_allclose(x, 12)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        r, p, z = c.compute("x", grid=0, basis="rpz")["x"].T
        np.testing.assert_allclose(r, np.sqrt(1**2 + 11**2))
        np.testing.assert_allclose(p, np.arctan2(-11, 1))
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_to_FourierXYZCurve(self):
        """Test fitting FourierXYZCurve from SplineXYZCurve object."""
        npts = 4000
        # make a simple circular curve of radius 2
        R = 2
        # make initial points non-uniform in angle
        phi = 2 * np.pi * np.linspace(0, 1, 1001, endpoint=True) ** 2
        c = SplineXYZCurve(
            X=R * np.cos(phi),
            Y=R * np.sin(phi),
            Z=np.zeros_like(phi),
            knots="arclength",
        )
        c2 = c.to_FourierXYZ(N=1, grid=1000)

        np.testing.assert_allclose(
            c.compute("length", grid=npts)["length"], R * 2 * np.pi, atol=2e-3
        )
        np.testing.assert_allclose(
            c2.compute("length", grid=npts)["length"], R * 2 * np.pi, atol=2e-3
        )

        grid = LinearGrid(N=20, endpoint=False)
        coords1 = c.compute("x", grid=grid, basis="xyz")["x"]
        coords2 = c2.compute("x", grid=grid, basis="xyz")["x"]

        np.testing.assert_allclose(coords1, coords2, atol=8e-3)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of FourierXYZCurve."""
        c = FourierXYZCurve()

        X, Y, Z = c.get_coeffs(0)
        np.testing.assert_allclose(X, 10)
        np.testing.assert_allclose(Y, 0)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 5, 2, 3)
        np.testing.assert_allclose(c.X_n, [0, 5, 2])
        np.testing.assert_allclose(c.Y_n, [0, 2, 0])
        np.testing.assert_allclose(c.Z_n, [-2, 3, 0])

        s = c.copy()
        assert s.equiv(c)

        c.change_resolution(5)
        assert c.N == 5
        with pytest.raises(ValueError):
            c.X_n = s.X_n
        with pytest.raises(ValueError):
            c.Y_n = s.Y_n
        with pytest.raises(ValueError):
            c.Z_n = s.Z_n

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking when creating FourierXYZCurve."""
        with pytest.raises(AssertionError):
            _ = FourierXYZCurve(X_n=[1], modes=[1, 2])
        with pytest.raises(AssertionError):
            _ = FourierXYZCurve(Y_n=[1], modes=[1, 2])
        with pytest.raises(AssertionError):
            _ = FourierXYZCurve(Z_n=[1], modes=[1, 2])

    @pytest.mark.unit
    def test_from_values_rpz(self):
        """Test from_values method with rpz coords."""
        t = np.linspace(0, 2 * np.pi, 10)
        R = np.cos(t)
        Z = np.sin(t)
        phi = np.zeros_like(t)
        coords = np.vstack([R, phi, Z]).T
        coil = FourierXYZCurve.from_values(coords, basis="rpz", N=1)
        np.testing.assert_allclose(coil.X_n[-1], 1.0)
        np.testing.assert_allclose(coil.Z_n[0], 1.0)


class TestFourierPlanarCurve:
    """Tests for FourierPlanarCurve class."""

    @pytest.mark.unit
    def test_center(self):
        """Test center of curve."""
        c = FourierPlanarCurve(center=[5, 4, 3], r_n=[1, 2, 0.5], basis="xyz")
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [5, 4, 3]
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [-6, 5, 4]
        )
        c = FourierPlanarCurve(center=[5, 1, -2], r_n=[1, 2, 0.5], basis="rpz")
        np.testing.assert_allclose(
            c.compute("center", basis="rpz")["center"][0, :], [5, 1, -2]
        )

    @pytest.mark.unit
    def test_rotation(self):
        """Test rotation of planar curve."""
        cx = FourierPlanarCurve(center=[0, 0, 0], normal=[1, 0, 0], r_n=1)
        cy = FourierPlanarCurve(center=[0, 0, 0], normal=[0, 1, 0], r_n=1)
        cz = FourierPlanarCurve(center=[0, 0, 0], normal=[0, 0, 1], r_n=1)
        datax = cx.compute("x", grid=20, basis="xyz")
        datay = cy.compute("x", grid=20, basis="xyz")
        dataz = cz.compute("x", grid=20, basis="xyz")
        np.testing.assert_allclose(datax["x"][:, 0], 0, atol=2e-16)  # only in Y-Z plane
        np.testing.assert_allclose(datay["x"][:, 1], 0, atol=2e-16)  # only in X-Z plane
        np.testing.assert_allclose(dataz["x"][:, 2], 0, atol=2e-16)  # only in X-Y plane

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        c = FourierPlanarCurve(modes=[0])
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], 2 * 2 * np.pi
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], 2 * 2 * np.pi
        )

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / 2)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / 2)

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        c = FourierPlanarCurve()
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-12
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-12
        )

    @pytest.mark.unit
    def test_frenet(self):
        """Test frenet-serret frame of circular curve."""
        c = FourierPlanarCurve()
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="xyz", grid=0
        )
        T, N, B = data["frenet_tangent"], data["frenet_normal"], data["frenet_binormal"]
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="xyz", grid=0
        )
        T, N, B = data["frenet_tangent"], data["frenet_normal"], data["frenet_binormal"]
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        c = FourierPlanarCurve()
        r, p, z = c.compute("x", grid=0, basis="rpz")["x"].T
        np.testing.assert_allclose(r, 12)
        np.testing.assert_allclose(p, 0)
        np.testing.assert_allclose(z, 0)
        dr, dp, dz = c.compute("x_sss", grid=0, basis="rpz")["x_sss"].T
        np.testing.assert_allclose(dr, 0)
        np.testing.assert_allclose(dp, 0, atol=1e-14)
        np.testing.assert_allclose(dz, 2)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        x, y, z = c.compute("x", grid=0, basis="xyz")["x"].T
        np.testing.assert_allclose(x, 1)
        np.testing.assert_allclose(y, -11)
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_basis(self):
        """Test xyz vs rpz basis."""
        cxyz = FourierPlanarCurve(center=[1, 1, 0], normal=[-1, 1, 0], basis="xyz")
        crpz = FourierPlanarCurve(
            center=[np.sqrt(2), np.pi / 4, 0], normal=[0, 1, 0], basis="rpz"
        )

        x_xyz = cxyz.compute("x")["x"]
        x_rpz = crpz.compute("x")["x"]
        np.testing.assert_allclose(x_xyz, x_rpz)

        xs_xyz = cxyz.compute("x_s")["x_s"]
        xs_rpz = crpz.compute("x_s")["x_s"]
        np.testing.assert_allclose(xs_xyz, xs_rpz, atol=2e-15)

        xss_xyz = cxyz.compute("x_ss")["x_ss"]
        xss_rpz = crpz.compute("x_ss")["x_ss"]
        np.testing.assert_allclose(xss_xyz, xss_rpz, atol=2e-15)

        xsss_xyz = cxyz.compute("x_sss")["x_sss"]
        xsss_rpz = crpz.compute("x_sss")["x_sss"]
        np.testing.assert_allclose(xsss_xyz, xsss_rpz, atol=2e-15)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of FourierPlanarCurve."""
        c = FourierPlanarCurve()

        r = c.get_coeffs(0)
        np.testing.assert_allclose(r, 2)
        c.set_coeffs(0, 3)
        np.testing.assert_allclose(c.r_n, [3])

        c.normal = [1, 2, 3]
        c.center = [3, 2, 1]
        np.testing.assert_allclose(np.linalg.norm(c.normal), 1)
        np.testing.assert_allclose(c.normal * np.linalg.norm(c.center), c.center[::-1])

        s = c.copy()
        assert s.equiv(c)

        c.change_resolution(5)
        with pytest.raises(ValueError):
            c.r_n = s.r_n

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking when creating FourierPlanarCurve."""
        c = FourierPlanarCurve()
        with pytest.raises(ValueError):
            c.center = [4]
        with pytest.raises(ValueError):
            c.normal = [4]
        with pytest.raises(AssertionError):
            _ = FourierPlanarCurve(r_n=[1], modes=[1, 2])

    @pytest.mark.unit
    def test_to_FourierPlanarCurve(self):
        """Test converting SplineXYZCurve to FourierPlanarCurve object."""
        npts = 1000
        N = 5

        # Create a SplineXYZCurve of a planar circle
        s = np.linspace(0, 2 * np.pi, npts)
        X = 2 * np.cos(s)
        Y = np.ones(npts)
        Z = 2 * np.sin(s)
        c = SplineXYZCurve(X=X, Y=Y, Z=Z)

        # Create a backwards SplineXYZCurve by flipping the coordinates
        c_backwards = SplineXYZCurve(X=np.flip(X), Y=np.flip(Y), Z=np.flip(Z))

        # Convert to FourierPlanarCurve
        c_planar = c.to_FourierPlanar(N=N, grid=npts, basis="xyz")
        c_backwards_planar = c_backwards.to_FourierPlanar(N=N, grid=npts, basis="xyz")

        grid = LinearGrid(N=20, endpoint=True)

        coords_spline = c.compute("x", grid=grid)["x"]
        coords_planar = c_planar.compute("x", grid=grid)["x"]
        coords_backwards_spline = c_backwards.compute("x", grid=grid)["x"]
        coords_backwards_planar = c_backwards_planar.compute("x", grid=grid)["x"]

        # Assertions for point positions
        np.testing.assert_allclose(coords_spline, coords_planar, atol=1e-10)
        np.testing.assert_allclose(
            coords_backwards_spline,
            coords_backwards_planar,
            atol=1e-10,
        )
        # backwards coordinate order is important for Biot-Savart current
        np.testing.assert_allclose(
            coords_planar, np.flip(coords_backwards_planar, axis=0), atol=1e-10
        )


class TestFourierXYCurve:
    """Tests for FourierXYCurve class."""

    @pytest.mark.unit
    def test_center(self):
        """Test center of curve."""
        c = FourierXYCurve(center=[5, 4, 3], X_n=[0.5, 2], Y_n=[1, 0], basis="xyz")
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [5, 4, 3]
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [-6, 5, 4]
        )
        c = FourierXYCurve(center=[5, 1, -2], X_n=[0.5, 2], Y_n=[1, 0], basis="rpz")
        np.testing.assert_allclose(
            c.compute("center", basis="rpz")["center"][0, :], [5, 1, -2]
        )

    @pytest.mark.unit
    def test_rotation(self):
        """Test rotation of planar curve."""
        cx = FourierXYCurve(center=[0, 0, 0], normal=[1, 0, 0], X_n=[0, 1], Y_n=[1, 0])
        cy = FourierXYCurve(center=[0, 0, 0], normal=[0, 1, 0], X_n=[0, 1], Y_n=[1, 0])
        cz = FourierXYCurve(center=[0, 0, 0], normal=[0, 0, 1], X_n=[0, 1], Y_n=[1, 0])
        datax = cx.compute("x", grid=20, basis="xyz")
        datay = cy.compute("x", grid=20, basis="xyz")
        dataz = cz.compute("x", grid=20, basis="xyz")
        np.testing.assert_allclose(datax["x"][:, 0], 0, atol=2e-16)  # only in Y-Z plane
        np.testing.assert_allclose(datay["x"][:, 1], 0, atol=2e-16)  # only in X-Z plane
        np.testing.assert_allclose(dataz["x"][:, 2], 0, atol=2e-16)  # only in X-Y plane

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        c = FourierXYCurve(modes=[-1, 1])
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], 2 * 2 * np.pi
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], 2 * 2 * np.pi
        )

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        c = FourierXYCurve()
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / 2)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / 2)

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        c = FourierXYCurve()
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-12
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-12
        )

    @pytest.mark.unit
    def test_frenet(self):
        """Test frenet-serret frame of circular curve."""
        c = FourierXYCurve()
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="xyz", grid=0
        )
        T, N, B = data["frenet_tangent"], data["frenet_normal"], data["frenet_binormal"]
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="xyz", grid=0
        )
        T, N, B = data["frenet_tangent"], data["frenet_normal"], data["frenet_binormal"]
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        c = FourierXYCurve()
        r, p, z = c.compute("x", grid=0, basis="rpz")["x"].T
        np.testing.assert_allclose(r, 12)
        np.testing.assert_allclose(p, 0)
        np.testing.assert_allclose(z, 0)
        dr, dp, dz = c.compute("x_sss", grid=0, basis="rpz")["x_sss"].T
        np.testing.assert_allclose(dr, 0, atol=1e-14)
        np.testing.assert_allclose(dp, 0, atol=1e-14)
        np.testing.assert_allclose(dz, 2)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        x, y, z = c.compute("x", grid=0, basis="xyz")["x"].T
        np.testing.assert_allclose(x, 1)
        np.testing.assert_allclose(y, -11)
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_basis(self):
        """Test xyz vs rpz basis."""
        cxyz = FourierXYCurve(center=[1, 1, 0], normal=[-1, 1, 0], basis="xyz")
        crpz = FourierXYCurve(
            center=[np.sqrt(2), np.pi / 4, 0], normal=[0, 1, 0], basis="rpz"
        )

        x_xyz = cxyz.compute("x")["x"]
        x_rpz = crpz.compute("x")["x"]
        np.testing.assert_allclose(x_xyz, x_rpz)

        xs_xyz = cxyz.compute("x_s")["x_s"]
        xs_rpz = crpz.compute("x_s")["x_s"]
        np.testing.assert_allclose(xs_xyz, xs_rpz, atol=2e-15)

        xss_xyz = cxyz.compute("x_ss")["x_ss"]
        xss_rpz = crpz.compute("x_ss")["x_ss"]
        np.testing.assert_allclose(xss_xyz, xss_rpz, atol=2e-15)

        xsss_xyz = cxyz.compute("x_sss")["x_sss"]
        xsss_rpz = crpz.compute("x_sss")["x_sss"]
        np.testing.assert_allclose(xsss_xyz, xsss_rpz, atol=2e-15)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of FourierXYCurve."""
        c = FourierXYCurve()

        X, Y = c.get_coeffs(1)
        np.testing.assert_allclose(X, 2)
        np.testing.assert_allclose(Y, 0)
        c.set_coeffs(1, 3, -2)
        np.testing.assert_allclose(c.X_n, [0, 3])
        np.testing.assert_allclose(c.Y_n, [2, -2])

        c.normal = [1, 2, 3]
        c.center = [3, 2, 1]
        np.testing.assert_allclose(np.linalg.norm(c.normal), 1)
        np.testing.assert_allclose(c.normal * np.linalg.norm(c.center), c.center[::-1])

        s = c.copy()
        assert s.equiv(c)

        c.change_resolution(5)
        with pytest.raises(ValueError):
            c.X_n = s.X_n

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking when creating FourierXYCurve."""
        c = FourierXYCurve()
        with pytest.raises(ValueError):
            c.center = [4]
        with pytest.raises(ValueError):
            c.normal = [4]
        with pytest.raises(AssertionError):
            _ = FourierXYCurve(X_n=[1], modes=[1, 2])

        with pytest.warns(UserWarning, match="Ignoring n=0 mode"):
            c0 = FourierXYCurve(X_n=[0, 1, 2], Y_n=[2, -1, 0])
        # check that curve is the same as default after n=0 mode is removed
        x = c.compute("x", grid=0)["x"]
        x0 = c0.compute("x", grid=0)["x"]
        np.testing.assert_allclose(x, x0)

    @pytest.mark.unit
    def test_to_FourierXYCurve_orientation(self):
        """Test converting SplineXYZCurve to FourierXYCurve object."""
        # specifically checking that orientation is preserved
        npts = 1000
        N = 5

        # Create a SplineXYZCurve of a planar circle
        s = np.linspace(0, 2 * np.pi, npts)
        X = 2 * np.cos(s)
        Y = np.ones(npts)
        Z = 2 * np.sin(s)
        c = SplineXYZCurve(X=X, Y=Y, Z=Z)

        # Create a backwards SplineXYZCurve by flipping the coordinates
        c_backwards = SplineXYZCurve(X=np.flip(X), Y=np.flip(Y), Z=np.flip(Z))

        # Convert to FourierXYCurve
        c_planar = c.to_FourierXY(N=N, grid=npts, basis="xyz")
        c_backwards_planar = c_backwards.to_FourierXY(N=N, grid=npts, basis="xyz")

        grid = LinearGrid(N=20, endpoint=True)

        coords_spline = c.compute("x", grid=grid)["x"]
        coords_planar = c_planar.compute("x", grid=grid)["x"]
        coords_backwards_spline = c_backwards.compute("x", grid=grid)["x"]
        coords_backwards_planar = c_backwards_planar.compute("x", grid=grid)["x"]

        # Assertions for point positions
        np.testing.assert_allclose(coords_spline, coords_planar, atol=1e-10)
        np.testing.assert_allclose(
            coords_backwards_spline,
            coords_backwards_planar,
            atol=1e-10,
        )
        # backwards coordinate order is important for Biot-Savart current
        np.testing.assert_allclose(
            coords_planar, np.flip(coords_backwards_planar, axis=0), atol=1e-10
        )

    @pytest.mark.unit
    def test_to_FourierXYCurve(self):
        """Test converting FourierRZCurve to FourierXYCurve."""
        # test different options for passing in s
        rz = FourierRZCurve(R_n=[0, 10, 0], Z_n=[-1, 0, 0])
        grid = LinearGrid(N=20, endpoint=False)
        xyz = rz.to_FourierXY(N=2, grid=grid, s=grid.nodes[:, 2])

        np.testing.assert_allclose(
            rz.compute("curvature", grid=grid)["curvature"],
            xyz.compute("curvature", grid=grid)["curvature"],
        )
        np.testing.assert_allclose(
            rz.compute("torsion", grid=grid)["torsion"],
            xyz.compute("torsion", grid=grid)["torsion"],
            atol=1e-16,
        )
        np.testing.assert_allclose(
            rz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
            atol=1e-16,
        )
        np.testing.assert_allclose(
            rz.compute("x", grid=grid, basis="xyz")["x"],
            xyz.compute("x", basis="xyz", grid=grid)["x"],
            atol=1e-12,
        )
        # same thing but pass in a closed grid
        grid = LinearGrid(N=20, endpoint=True)
        xyz = rz.to_FourierXY(N=2, grid=grid, s=grid.nodes[:, 2])

        np.testing.assert_allclose(
            rz.compute("curvature", grid=grid)["curvature"],
            xyz.compute("curvature", grid=grid)["curvature"],
        )
        np.testing.assert_allclose(
            rz.compute("torsion", grid=grid)["torsion"],
            xyz.compute("torsion", grid=grid)["torsion"],
            atol=1e-16,
        )
        np.testing.assert_allclose(
            rz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
        )
        np.testing.assert_allclose(
            rz.compute("x", grid=grid, basis="xyz")["x"],
            xyz.compute("x", basis="xyz", grid=grid)["x"],
            atol=1e-12,
        )

        # same thing but with arclength angle
        grid = LinearGrid(N=20, endpoint=False)
        xyz = rz.to_FourierXY(N=2, grid=grid, s="arclength")

        np.testing.assert_allclose(
            rz.compute("length", grid=grid)["length"],
            xyz.compute("length", grid=grid)["length"],
            rtol=1e-5,
        )

        # pass in non-monotonic s
        grid = LinearGrid(N=20, endpoint=False)
        s = grid.nodes[:, 2]
        s[-2] = s[-1]
        with pytest.raises(ValueError):
            xyz = rz.to_FourierXY(N=2, grid=grid, s=s)


class TestSplineXYZCurve:
    """Tests for SplineXYZCurve class."""

    @pytest.mark.unit
    def test_center(self):
        """Test center of curve."""
        c = SplineXYZCurve(
            X=np.array([5, 8, 7, 6]),
            Y=np.array([4, 2, 3, 1]),
            Z=np.array([-2, -1, 1, 0]),
        )
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [6.5, 2.5, -0.5]
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("center", basis="xyz")["center"][0, :], [-7.5, 3.5, 0.5]
        )

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        for method in [
            "nearest",
            "linear",
            "cubic",
            "cubic2",
            "catmull-rom",
            "monotonic",
            "cardinal",
        ]:
            R = 1
            phi = np.linspace(0, 2 * np.pi, 1001, endpoint=True)
            # if nearest method, cant give more than the knot pts or it will return
            # a length larger than the real one
            npts = (
                2000
                if method != ["nearest", "linear"]
                else phi + 0.1 * (phi[1] - phi[0])
            )
            # make sure that length error is less than what the error would be
            # if were simply missing one segment of a linear interpolation,
            #  to try to ensure we are not making that mistake
            atol = R * 2 * np.pi / npts if method not in ["nearest", "linear"] else 3e-3
            c = SplineXYZCurve(
                X=R * np.cos(phi),
                Y=R * np.sin(phi),
                Z=np.zeros_like(phi),
                method=method,
            )
            np.testing.assert_allclose(
                c.compute("length", grid=npts)["length"],
                R * 2 * np.pi,
                atol=atol,
                err_msg=f"Failed at {method}",
            )
            c.translate([1, 1, 1])
            c.rotate(angle=np.pi)
            c.flip([0, 1, 0])
            np.testing.assert_allclose(
                c.compute("length", grid=npts)["length"],
                R * 2 * np.pi,
                atol=atol,
                err_msg=f"Failed at {method}",
            )

            # make a simple circular curve with supplied knots as phi
            phi = np.linspace(0, 2 * np.pi, 201, endpoint=False)
            c = SplineXYZCurve(
                X=R * np.cos(phi),
                Y=R * np.sin(phi),
                Z=np.zeros_like(phi),
                knots=phi,
                method=method,
            )
            np.testing.assert_allclose(
                c.compute("length", grid=npts)["length"],
                R * 2 * np.pi,
                atol=atol,
                err_msg=f"Failed at {method}",
            )
            c.translate([1, 1, 1])
            c.rotate(angle=np.pi)
            c.flip([0, 1, 0])
            np.testing.assert_allclose(
                c.compute("length", grid=npts)["length"],
                R * 2 * np.pi,
                atol=atol,
                err_msg=f"Failed at {method}",
            )

            if method == "nearest":
                continue  # don't test changing the grid if nearest
                # since it will give wrong answers for
                # grids with more than the initial num of knots
            # check lengths when changing X,Y,Z from initial values
            # and from changing grids
            R = 1.1
            c.X = R * np.cos(phi)
            c.Y = R * np.sin(phi)
            c.Z = np.ones_like(phi)
            grid = LinearGrid(zeta=np.linspace(0, 2 * np.pi, npts, endpoint=False))
            np.testing.assert_allclose(
                c.compute("length", grid=grid)["length"],
                R * 2 * np.pi,
                atol=atol,
                err_msg=f"Failed at {method}",
            )
            np.testing.assert_allclose(
                c.compute("length", grid=None)["length"],
                R * 2 * np.pi,
                atol=9e-3,
                err_msg=f"Failed at {method}",
            )

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        # make a simple circular curve of radius 2
        R = 3
        phi = np.linspace(0, 2 * np.pi, 101, endpoint=False)
        c = SplineXYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))
        x, y, z = c.compute("x", grid=Grid(np.array([[0.0, 0.0, 0.0]])), basis="xyz")[
            "x"
        ].T
        np.testing.assert_allclose(x, R)
        np.testing.assert_allclose(y, 0, atol=1e-15)
        np.testing.assert_allclose(z, 0, atol=1e-15)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        r, p, z = c.compute("x", grid=Grid(np.array([[0.0, 0.0, 0.0]])), basis="rpz")[
            "x"
        ].T
        np.testing.assert_allclose(r, np.sqrt(1**2 + (R - 1) ** 2))
        np.testing.assert_allclose(p, np.arctan2(-(R - 1), 1))
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        # make a simple circular curve of radius 10
        R = 10
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=True)
        c = SplineXYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))
        np.testing.assert_allclose(
            c.compute("curvature", grid=10)["curvature"][1:-1], 1 / 10, atol=1e-3
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("curvature", grid=10)["curvature"][1:-1], 1 / 10, atol=1e-3
        )

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        # make a simple circular curve of radius 10
        R = 10
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=True)
        c = SplineXYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-12
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-12
        )

    @pytest.mark.unit
    def test_to_SplineXYZCurve(self):
        """Test converting FourierXYZCurve to SplineXYZCurve object."""
        npts = 4000
        # make a simple circular curve of radius 2
        R = 2
        c = FourierXYZCurve()
        c2 = c.to_SplineXYZ(grid=npts)

        np.testing.assert_allclose(
            c.compute("length", grid=npts)["length"], R * 2 * np.pi, atol=2e-3
        )
        np.testing.assert_allclose(
            c2.compute("length", grid=npts)["length"], R * 2 * np.pi, atol=2e-3
        )
        grid = LinearGrid(N=20, endpoint=False)
        coords1 = c.compute("x", grid=grid)["x"]
        coords2 = c2.compute("x", grid=grid)["x"]

        np.testing.assert_allclose(coords1, coords2, atol=1e-10)

    @pytest.mark.unit
    def test_asserts_and_errors(self):
        """Test error checking when creating or setting properties of SplineXYZCurve."""
        # make a simple circular curve of radius 2
        R = 2
        phi = np.linspace(0, 2 * np.pi, 101, endpoint=True)
        c = SplineXYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))

        # change number of knots, should raise error since is different than
        # existing knots
        phi = np.linspace(0, 2 * np.pi, 102, endpoint=True)
        with pytest.raises(ValueError):
            c.X = R * np.cos(phi)
        with pytest.raises(ValueError):
            c.Y = R * np.sin(phi)
        with pytest.raises(ValueError):
            c.Z = np.zeros_like(phi)

        # setter for knots
        with pytest.raises(ValueError):
            c.knots = np.linspace(0, 10, 10)
        knots = c.knots
        knots[-2] = knots[-1]  # make it non-monotonic
        with pytest.raises(ValueError):
            c.knots = knots
        with pytest.raises(ValueError):
            c.knots *= -1
        with pytest.raises(ValueError):
            c.knots += np.pi

        # setter for method
        with pytest.raises(ValueError):
            c.method = "not a valid method"

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of SplineXYZCurve."""
        # make a simple circular curve of radius 2
        R = 2
        phi = np.linspace(0, 2 * np.pi, 101, endpoint=True)
        c = SplineXYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))

        s = c.copy()
        assert s.equiv(c)

    @pytest.mark.unit
    def test_compute_ndarray_error(self):
        """Test raising TypeError if ndarray is passed in."""
        # make a simple circular curve of radius 2
        R = 2
        phi = np.linspace(0, 2 * np.pi, 101, endpoint=True)
        c = SplineXYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))
        with pytest.raises(TypeError):
            c.compute("length", grid=np.linspace(0, 1, 10))


class TestNurbsRPZCurve:
    """Tests for NurbsRPZCurve class."""

    @staticmethod
    def _example_curve(n=10, nfp=3, p=3):
        """Build a modestly-modulated stellarator-symmetric test curve."""
        phi = np.linspace(0, np.pi / nfp, n)
        R = 1.0 + 0.1 * np.cos(phi * nfp)
        Z = 0.1 * np.sin(phi * nfp)
        W = np.ones(n)
        return NurbsRPZCurve(
            R=R,
            phi=phi,
            Z=Z,
            W=W,
            n_ctrl_points=n,
            sym=True,
            nfp=nfp,
            degree=p,
            knot_parametrization="uniform",
        )

    @pytest.mark.unit
    def test_basis_derivs_match_scipy(self):
        """Test b_p_deriv3 basis functions/derivatives against scipy.BSpline.

        b_p_deriv3 is DESC's own (JAX, differentiable) Cox-de Boor implementation
        and shares no code with scipy -- agreement here is a real, independent
        check, not a tautology.
        """
        p = 3
        n_ctrl = 20
        knots = np.asarray(uniform_knots(n_ctrl - 1, p, domain=2 * np.pi))
        # stay away from the domain edges, where scipy's extrapolate=False
        # returns nan rather than wrapping
        s = np.linspace(0.3, 2 * np.pi - 0.3, 50)

        b, b_s, b_ss, b_sss = b_p_deriv3(s, p, knots)
        b, b_s, b_ss, b_sss = (np.asarray(x) for x in (b, b_s, b_ss, b_sss))

        n_basis = b.shape[1]
        for j in range(n_basis):
            c = np.zeros(n_basis)
            c[j] = 1.0
            spl = BSpline(knots, c, p, extrapolate=False)
            np.testing.assert_allclose(
                np.nan_to_num(spl(s)), b[:, j], atol=1e-12, err_msg=f"basis {j}"
            )
            np.testing.assert_allclose(
                np.nan_to_num(spl.derivative(1)(s)),
                b_s[:, j],
                atol=1e-8,
                err_msg=f"1st deriv, basis {j}",
            )
            np.testing.assert_allclose(
                np.nan_to_num(spl.derivative(2)(s)),
                b_ss[:, j],
                atol=1e-6,
                err_msg=f"2nd deriv, basis {j}",
            )
            np.testing.assert_allclose(
                np.nan_to_num(spl.derivative(3)(s)),
                b_sss[:, j],
                atol=1e-6,
                err_msg=f"3rd deriv, basis {j}",
            )

    @pytest.mark.unit
    def test_derivatives_match_independent_reconstruction(self):
        """Test x, x_s, x_ss, x_sss against an independent scipy-based rebuild.

        Rebuilds the same rational (NURBS) curve from scratch using scipy's
        BSpline derivatives and the quotient rule, entirely independent of
        DESC's own compute functions, then compares directly.
        """
        n, nfp, p = 10, 3, 3
        c = self._example_curve(n=n, nfp=nfp, p=p)

        data = c.compute(
            [
                "knots",
                "full_control_net",
                "full_weights",
                "s",
                "x",
                "x_s",
                "x_ss",
                "x_sss",
                "phi",
            ],
            grid=60,
        )
        knots = np.asarray(data["knots"])
        fcn = np.asarray(data["full_control_net"])  # already Cartesian
        fw = np.asarray(data["full_weights"])
        sq = np.asarray(data["s"])

        padded = np.concatenate([fcn[-p:], fcn, fcn[:p]], axis=0)
        padded_w = np.concatenate([fw[-p:], fw, fw[:p]])
        n_basis = padded.shape[0]

        # N^(d), D^(d) for d=0..3, built directly from scipy's own basis derivs
        Np = [np.zeros((len(sq), 3)) for _ in range(4)]
        Dp = [np.zeros(len(sq)) for _ in range(4)]
        for j in range(n_basis):
            cvec = np.zeros(n_basis)
            cvec[j] = 1.0
            spl = BSpline(knots, cvec, p, extrapolate=False)
            for d in range(4):
                bd = spl.derivative(d)(sq) if d > 0 else spl(sq)
                bd = np.nan_to_num(bd)
                Np[d] += padded_w[j] * bd[:, None] * padded[j]
                Dp[d] += padded_w[j] * bd

        N, N1, N2, N3 = Np
        D, D1, D2, D3 = Dp
        # quotient rule, applied recursively (C = N/D, Leibniz on N = C*D)
        cc = N / D[:, None]
        cs = (N1 - cc * D1[:, None]) / D[:, None]
        css = (N2 - 2 * cs * D1[:, None] - cc * D2[:, None]) / D[:, None]
        csss = (
            N3 - 3 * css * D1[:, None] - 3 * cs * D2[:, None] - cc * D3[:, None]
        ) / D[:, None]

        ph = np.asarray(data["phi"])
        np.testing.assert_allclose(rpz2xyz(np.asarray(data["x"])), cc, atol=1e-10)
        np.testing.assert_allclose(
            rpz2xyz_vec(np.asarray(data["x_s"]), phi=ph), cs, atol=1e-8
        )
        np.testing.assert_allclose(
            rpz2xyz_vec(np.asarray(data["x_ss"]), phi=ph), css, atol=1e-6
        )
        np.testing.assert_allclose(
            rpz2xyz_vec(np.asarray(data["x_sss"]), phi=ph), csss, atol=1e-4
        )

    @pytest.mark.unit
    def test_stellarator_symmetry(self):
        """Test R(phi)=R(2pi/nfp-phi), Z(phi)=-Z(2pi/nfp-phi) directly.

        This is the actual mirror relation the reflect-and-tile construction
        encodes, about phi=pi/nfp (not phi=0). Finds the curve's own native
        parameter s at each target phi via root-finding rather than assuming
        s and phi coincide -- s is fed directly into the spline with no
        root-solve, so it is *not* the same variable as the physical phi.
        """
        nfp = 3
        c = self._example_curve(nfp=nfp)
        axis = np.pi / nfp

        def x_at_phi(target, bracket):
            def phi_of_s(s):
                grid = LinearGrid(zeta=np.mod(s, 2 * np.pi))
                return c.compute("x", grid=grid)["x"][0, 1] - target

            s = brentq(phi_of_s, *bracket)
            return c.compute("x", grid=LinearGrid(zeta=np.mod(s, 2 * np.pi)))["x"][0]

        for delta in [0.1, 0.3, 0.5, 0.7]:
            xp = x_at_phi(axis + delta, (axis, axis + 1.2))
            xm = x_at_phi(axis - delta, (axis - 1.2, axis))
            np.testing.assert_allclose(xp[0], xm[0], atol=1e-6)  # R
            np.testing.assert_allclose(xp[2], -xm[2], atol=1e-6)  # Z

    @pytest.mark.unit
    def test_center(self):
        """Test center of a symmetric circular curve is exactly the origin.

        Equally-spaced points all the way around a full circle sum to
        exactly zero (roots-of-unity identity), so this is exact, not a
        tolerance-tuned approximation.
        """
        n, nfp = 10, 3
        phi = np.linspace(0, np.pi / nfp, n)
        c = NurbsRPZCurve(
            R=np.ones(n),
            phi=phi,
            Z=np.zeros(n),
            W=np.ones(n),
            n_ctrl_points=n,
            sym=True,
            nfp=nfp,
            degree=3,
        )
        np.testing.assert_allclose(
            c.compute("center", grid=60, basis="xyz")["center"][0, :],
            [0, 0, 0],
            atol=1e-10,
        )

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of a constant-R, Z=0 circular curve.

        Z=0 control points force the curve's own Z to be exactly zero
        everywhere (Z is linear in the control-point blend). R is *not*
        forced to exactly R0: the convex-hull property applies in Cartesian
        (X, Y, Z) space, and R=sqrt(X^2+Y^2) is a nonlinear function of that
        blend -- points on a circle are strictly convex, so any genuine
        blend of them is pulled strictly inside, giving R <= R0 (a real
        inequality, not floating-point slop).
        """
        n, nfp, R0 = 10, 3, 2.0
        phi = np.linspace(0, np.pi / nfp, n)
        c = NurbsRPZCurve(
            R=R0 * np.ones(n),
            phi=phi,
            Z=np.zeros(n),
            W=np.ones(n),
            n_ctrl_points=n,
            sym=True,
            nfp=nfp,
            degree=3,
        )
        x = c.compute("x", grid=60)["x"]
        np.testing.assert_allclose(x[:, 2], 0, atol=1e-10)
        assert np.all(x[:, 0] <= R0 + 1e-10)
        assert np.all(x[:, 0] > 0.9 * R0)  # sanity: not wildly off

    @pytest.mark.unit
    def test_curvature_torsion_length_circle(self):
        """Test curvature/torsion/length of a circular curve against analytic values."""
        # TODO: build a constant-R, Z=0 NurbsRPZCurve and check
        # curvature == 1/R, torsion == 0, length == 2*pi*R (with an atol
        # reflecting the B-spline's approximation of the true circle).
        pytest.skip("TODO: analytic circle curvature/torsion/length check")

    @pytest.mark.unit
    def test_frenet(self):
        """Test orthonormality of the Frenet-Serret frame."""
        # TODO: check |T|=|N|=|B|=1 and T.N=T.B=N.B=0 along the curve.
        pytest.skip("TODO: Frenet frame orthonormality check")

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking when creating/mutating NurbsRPZCurve."""
        # TODO: non-increasing phi raises ValueError, sym=False raises
        # NotImplementedError, mismatched R/phi/Z/W lengths raise errors.
        pytest.skip("TODO: constructor/assignment error checks")
