"""Tests for different Curve classes."""

import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierRZWindingSurfaceCurve,
    FourierXYZCurve,
    SplineXYZCurve,
)
from desc.grid import Grid, LinearGrid
from desc.io import InputReader, load


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


class TestRZWindingSurfaceCurve:
    """Tests for FourierRZWindingSurfaceCurve class."""

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        c = FourierRZWindingSurfaceCurve()
        # by default the curve is a poloidally closed curve at constant zeta
        a = 1  # default minor radius of the surface
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], a * 2 * np.pi
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("length", grid=20)["length"], a * 2 * np.pi
        )

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        c = FourierRZWindingSurfaceCurve()
        # by default the curve is a poloidally closed curve at constant zeta
        a = 1  # default minor radius of the surface
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / a)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute("curvature", grid=20)["curvature"], 1 / a)

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        c = FourierRZWindingSurfaceCurve()
        # by default the curve is a poloidally closed curve at constant zeta
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-16
        )
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(
            c.compute("torsion", grid=20)["torsion"], 0, atol=1e-16
        )

    @pytest.mark.unit
    def test_frenet(self):
        """Test frenet-serret frame of circular curve."""
        c = FourierRZWindingSurfaceCurve(secular_theta=0.0, secular_zeta=1.0)
        # the curve is a toroidally closed curve at constant theta
        data = c.compute(
            ["frenet_tangent", "frenet_normal", "frenet_binormal"], basis="rpz", grid=0
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
        c = FourierRZWindingSurfaceCurve(secular_theta=0.0, secular_zeta=1.0)
        # the curve is a toroidally closed curve at constant theta
        x, y, z = c.compute("x", grid=0, basis="xyz")["x"].T
        np.testing.assert_allclose(x, 11)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of FourierRZWindingSurfaceCurve."""
        c = FourierRZWindingSurfaceCurve(sym_theta="cos", sym_zeta="cos")

        theta_n, zeta_n = c.get_coeffs(0)
        np.testing.assert_allclose(theta_n, 0)
        np.testing.assert_allclose(zeta_n, 0)
        c.set_coeffs(0, 5, 1)
        np.testing.assert_allclose(c.theta_n, [5])
        np.testing.assert_allclose(c.zeta_n, [1])

        s = c.copy()
        assert s.equiv(c)

        c.change_resolution(5)
        assert c.N == 5
        c.set_coeffs(5, None, 2)
        np.testing.assert_allclose(c.theta_n, [5, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(c.zeta_n, [1, 0, 0, 0, 0, 2])

        with pytest.raises(ValueError):
            c.theta_n = s.theta_n
        with pytest.raises(ValueError):
            c.zeta_n = s.zeta_n

        c.name = "my curve"
        assert "my" in c.name
        assert c.name in str(c)
        assert "FourierRZWindingSurfaceCurve" in str(c)
        assert c.sym

        c.change_resolution(NFP=3)
        assert c.NFP == 3
        assert c.theta_basis.NFP == 3
        assert c.zeta_basis.NFP == 3

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking when creating FourierRZWindingSurfaceCurve."""
        with pytest.raises(ValueError):
            _ = FourierRZWindingSurfaceCurve(theta_n=[])
        with pytest.raises(AssertionError):
            _ = FourierRZWindingSurfaceCurve(theta_n=[1], modes_theta=[1, 2])
        with pytest.raises(AssertionError):
            _ = FourierRZWindingSurfaceCurve(zeta_n=[1], modes_zeta=[1, 2])

    @pytest.mark.unit
    def test_io(self, tmpdir_factory):
        """Test io for FourierRZWindingSurfaceCurve class."""
        c = FourierRZWindingSurfaceCurve(secular_theta=0.0, secular_zeta=1.0)
        tmpdir = tmpdir_factory.mktemp("curve_io_test")
        tmp_path = tmpdir.join("windingsurfacecurve.h5")
        c.save(tmp_path)
        c2 = load(tmp_path)
        assert c.equiv(c2)
        # the curve is a toroidally closed curve at constant theta
        # check it correctly computes coords after load
        x, y, z = c.compute("x", grid=0, basis="xyz")["x"].T
        np.testing.assert_allclose(x, 11)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)

    @pytest.mark.unit
    def test_curve_winding_surface_from_values(self):
        """Test from_values method for FourierRZWindingSurfaceCurve."""
        c = FourierRZWindingSurfaceCurve()
        # modular curve (secular term in zeta is 0, in theta is 1)
        theta = np.linspace(0, 2 * np.pi, endpoint=True)
        zeta = np.zeros_like(theta)
        curve = FourierRZWindingSurfaceCurve.from_values(theta, zeta, c.surface)
        assert curve.secular_theta == 1
        assert curve.secular_zeta == 0
        np.testing.assert_allclose(curve.theta_n, 0)
        np.testing.assert_allclose(curve.zeta_n, 0)
        # helical curve (secular term in zeta is 1, in theta is 1)
        zeta = theta
        curve = FourierRZWindingSurfaceCurve.from_values(theta, zeta, c.surface)
        assert curve.secular_theta == 1
        assert curve.secular_zeta == 1
        np.testing.assert_allclose(curve.theta_n, 0)
        np.testing.assert_allclose(curve.zeta_n, 0)
        # toroidal curve (secular term in zeta is 1, in theta is 0)
        zeta = np.linspace(0, 2 * np.pi, endpoint=True)
        theta = np.zeros_like(zeta)
        curve = FourierRZWindingSurfaceCurve.from_values(theta, zeta, c.surface)
        assert curve.secular_theta == 0
        assert curve.secular_zeta == 1
        np.testing.assert_allclose(curve.theta_n, 0)
        np.testing.assert_allclose(curve.zeta_n, 0)


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
