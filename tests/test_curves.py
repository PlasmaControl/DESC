"""Tests for different Curve classes."""

import numpy as np
import pytest

from desc.geometry import FourierPlanarCurve, FourierRZCurve, FourierXYZCurve, XYZCurve
from desc.grid import LinearGrid


class TestRZCurve:
    """Tests for FourierRZCurve class."""

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_length(grid=20), 10 * 2 * np.pi)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_length(grid=20), 10 * 2 * np.pi)

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 10)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 10)

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        c = FourierRZCurve()
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)

    @pytest.mark.unit
    def test_frenet(self):
        """Test frenet-seret frame of circular curve."""
        c = FourierRZCurve()
        c.grid = 0
        T, N, B = c.compute_frenet_frame(basis="rpz")
        np.testing.assert_allclose(T, np.array([[0, 1, 0]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 0, 1]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        c.grid = np.array([[0, 0, 0]])
        T, N, B = c.compute_frenet_frame(basis="xyz")
        np.testing.assert_allclose(T, np.array([[0, 1, 0]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 0, 1]]), atol=1e-12)

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        c = FourierRZCurve()
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 10)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        r, p, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="rpz").T
        np.testing.assert_allclose(r, np.sqrt(1**2 + 9**2))
        np.testing.assert_allclose(p, np.arctan2(-9, 1))
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of FourierRZCurve."""
        c = FourierRZCurve()
        grid = LinearGrid(M=2, N=2)
        c.grid = grid
        assert grid.eq(c.grid)

        R, Z = c.get_coeffs(0)
        np.testing.assert_allclose(R, 10)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 5, None)
        np.testing.assert_allclose(
            c.R_n,
            [
                5,
            ],
        )
        np.testing.assert_allclose(c.Z_n, [])

        s = c.copy()
        assert s.eq(c)

        c.change_resolution(5)
        assert c.N == 5
        c.set_coeffs(-1, None, 2)
        np.testing.assert_allclose(
            c.R_n,
            [5, 0, 0, 0, 0, 0],
        )
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

        c.NFP = 3
        assert c.NFP == 3
        assert c.R_basis.NFP == 3
        assert c.Z_basis.NFP == 3
        assert c.grid.NFP == 3

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking when creating FourierRZCurve."""
        with pytest.raises(ValueError):
            c = FourierRZCurve(R_n=[])
        c = FourierRZCurve()
        with pytest.raises(NotImplementedError):
            c.compute_coordinates(dt=4)
        with pytest.raises(TypeError):
            c.grid = [1, 2, 3]

    @pytest.mark.unit
    def test_to_FourierXYZCurve(self):
        """Test conversion to XYZCurve."""
        rz = FourierRZCurve(R_n=[0, 10, 1], Z_n=[-1, 0, 0])
        xyz = rz.to_FourierXYZCurve(N=2)

        np.testing.assert_allclose(
            rz.compute_curvature(), xyz.compute_curvature(grid=rz.grid)
        )
        np.testing.assert_allclose(
            rz.compute_torsion(), xyz.compute_torsion(grid=rz.grid)
        )
        np.testing.assert_allclose(
            rz.compute_length(), xyz.compute_length(grid=rz.grid)
        )
        np.testing.assert_allclose(
            rz.compute_coordinates(basis="rpz"),
            xyz.compute_coordinates(basis="rpz", grid=rz.grid),
            atol=1e-12,
        )


class TestFourierXYZCurve:
    """Tests for FourierXYZCurve class."""

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        c = FourierXYZCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        c = FourierXYZCurve(modes=[-1, 0, 1])
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)

    @pytest.mark.unit
    def test_frenet(self):
        """Test frenet-seret frame of circular curve."""
        c = FourierXYZCurve()
        c.grid = 0
        T, N, B = c.compute_frenet_frame(basis="rpz")
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        c.grid = np.array([0, 0, 0])
        T, N, B = c.compute_frenet_frame(basis="xyz")
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        c = FourierXYZCurve()
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 12)
        np.testing.assert_allclose(y, 0)
        np.testing.assert_allclose(z, 0)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        r, p, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="rpz").T
        np.testing.assert_allclose(r, np.sqrt(1**2 + 11**2))
        np.testing.assert_allclose(p, np.arctan2(-11, 1))
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_from_XYZCurve(self):
        """Test fitting FourierXYZCurve from XYZCurve object."""
        npts = 4000
        # make a simple circular curve of radius 2
        R = 2
        phi = np.linspace(0, 2 * np.pi, 1001, endpoint=True)
        c = XYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))
        c2 = FourierXYZCurve.from_XYZCurve(c, N=1, grid=100)
        c3 = c.to_FourierXYZCurve(N=1, grid=100)

        np.testing.assert_allclose(
            c.compute_length(grid=npts), R * 2 * np.pi, atol=2e-3
        )
        np.testing.assert_allclose(
            c2.compute_length(grid=npts), R * 2 * np.pi, atol=2e-3
        )
        np.testing.assert_allclose(
            c3.compute_length(grid=npts), R * 2 * np.pi, atol=2e-3
        )
        grid = LinearGrid(N=20, endpoint=True)
        coords1 = c.compute_coordinates(grid=grid)
        coords2 = c2.compute_coordinates(grid=grid)
        coords3 = c3.compute_coordinates(grid=grid)

        np.testing.assert_allclose(coords1, coords2, atol=1e-10)
        np.testing.assert_allclose(coords2, coords3)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of FourierXYZCurve."""
        c = FourierXYZCurve()
        grid = LinearGrid(M=2, N=2)
        c.grid = grid
        assert grid.eq(c.grid)

        X, Y, Z = c.get_coeffs(0)
        np.testing.assert_allclose(X, 10)
        np.testing.assert_allclose(Y, 0)
        np.testing.assert_allclose(Z, 0)
        c.set_coeffs(0, 5, 2, 3)
        np.testing.assert_allclose(c.X_n, [0, 5, 2])
        np.testing.assert_allclose(c.Y_n, [0, 2, 0])
        np.testing.assert_allclose(c.Z_n, [-2, 3, 0])

        s = c.copy()
        assert s.eq(c)

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
        c = FourierXYZCurve()
        with pytest.raises(KeyError):
            c.compute_coordinates(dt=4)
        with pytest.raises(TypeError):
            c.grid = [1, 2, 3]


class TestPlanarCurve:
    """Tests for FourierPlanarCurve class."""

    @pytest.mark.unit
    def test_length(self):
        """Test length of circular curve."""
        c = FourierPlanarCurve(modes=[0])
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_length(grid=20), 2 * 2 * np.pi)

    @pytest.mark.unit
    def test_curvature(self):
        """Test curvature of circular curve."""
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_curvature(grid=20), 1 / 2)

    @pytest.mark.unit
    def test_torsion(self):
        """Test torsion of circular curve."""
        c = FourierPlanarCurve()
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)
        c.translate([1, 1, 1])
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        np.testing.assert_allclose(c.compute_torsion(grid=20), 0)

    @pytest.mark.unit
    def test_frenet(self):
        """Test frenet-seret frame of circular curve."""
        c = FourierPlanarCurve()
        c.grid = 0
        T, N, B = c.compute_frenet_frame(basis="xyz")
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[-1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)
        c.rotate(angle=np.pi)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        c.grid = np.array([0, 0, 0])
        T, N, B = c.compute_frenet_frame(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz")
        np.testing.assert_allclose(T, np.array([[0, 0, -1]]), atol=1e-12)
        np.testing.assert_allclose(N, np.array([[1, 0, 0]]), atol=1e-12)
        np.testing.assert_allclose(B, np.array([[0, 1, 0]]), atol=1e-12)

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        c = FourierPlanarCurve()
        r, p, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="rpz").T
        np.testing.assert_allclose(r, 12)
        np.testing.assert_allclose(p, 0)
        np.testing.assert_allclose(z, 0)
        dr, dp, dz = c.compute_coordinates(
            grid=np.array([[0.0, 0.0, 0.0]]), dt=3, basis="rpz"
        ).T
        np.testing.assert_allclose(dr, 0)
        np.testing.assert_allclose(dp, 0)
        np.testing.assert_allclose(dz, 2)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, 1)
        np.testing.assert_allclose(y, -11)
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of FourierPlanarCurve."""
        c = FourierPlanarCurve()
        grid = LinearGrid(M=2, N=2)
        c.grid = grid
        assert grid.eq(c.grid)

        r = c.get_coeffs(0)
        np.testing.assert_allclose(r, 2)
        c.set_coeffs(0, 3)
        np.testing.assert_allclose(
            c.r_n,
            [
                3,
            ],
        )

        c.normal = [1, 2, 3]
        c.center = [3, 2, 1]
        np.testing.assert_allclose(np.linalg.norm(c.normal), 1)
        np.testing.assert_allclose(c.normal * np.linalg.norm(c.center), c.center[::-1])

        s = c.copy()
        assert s.eq(c)

        c.change_resolution(5)
        with pytest.raises(ValueError):
            c.r_n = s.r_n

    @pytest.mark.unit
    def test_asserts(self):
        """Test error checking when creating FourierPlanarCurve."""
        c = FourierPlanarCurve()
        with pytest.raises(NotImplementedError):
            c.compute_coordinates(dt=4)
        with pytest.raises(TypeError):
            c.grid = [1, 2, 3]
        with pytest.raises(ValueError):
            c.center = [4]
        with pytest.raises(ValueError):
            c.normal = [4]


class TestXYZCurve:
    """Tests for XYZCurve class."""

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
            "monotonic-0",
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
            c = XYZCurve(
                X=R * np.cos(phi),
                Y=R * np.sin(phi),
                Z=np.zeros_like(phi),
                method=method,
            )
            np.testing.assert_allclose(
                c.compute_length(grid=npts), R * 2 * np.pi, atol=atol
            )
            c.translate([1, 1, 1])
            c.rotate(angle=np.pi)
            c.flip([0, 1, 0])
            np.testing.assert_allclose(
                c.compute_length(grid=npts), R * 2 * np.pi, atol=atol
            )

            # make a simple circular curve with supplied knots as phi
            phi = np.linspace(0, 2 * np.pi, 201, endpoint=True)
            c = XYZCurve(
                X=R * np.cos(phi),
                Y=R * np.sin(phi),
                Z=np.zeros_like(phi),
                knots=phi,
                period=2 * np.pi,
                method=method,
            )
            np.testing.assert_allclose(
                c.compute_length(grid=npts), R * 2 * np.pi, atol=atol
            )
            c.translate([1, 1, 1])
            c.rotate(angle=np.pi)
            c.flip([0, 1, 0])
            np.testing.assert_allclose(
                c.compute_length(grid=npts), R * 2 * np.pi, atol=atol
            )

            if method == "nearest":
                continue  # don't test changing the grid if nearest
                # since it will give wrong answers for
                # grids with more than the initial num of knots
            # check lengths when changing X,Y,Z from intial values
            # and from changing grids
            R = 1.1
            c.X = R * np.cos(phi)
            c.Y = R * np.sin(phi)
            c.Z = np.ones_like(phi)
            np.testing.assert_allclose(
                c.compute_length(grid=npts), R * 2 * np.pi, atol=atol
            )

            c.grid = LinearGrid(
                L=0, M=0, zeta=np.linspace(0, 2 * np.pi, npts, endpoint=True)
            )
            np.testing.assert_allclose(
                c.compute_length(grid=None), R * 2 * np.pi, atol=atol
            )
            phi = c.grid.nodes[:, 2]
            c.grid = np.vstack((np.zeros_like(phi), np.zeros_like(phi), phi)).T
            np.testing.assert_allclose(
                c.compute_length(grid=None), R * 2 * np.pi, atol=atol
            )

    @pytest.mark.unit
    def test_coords(self):
        """Test lab frame coordinates of circular curve."""
        # make a simple circular curve of radius 2
        R = 3
        phi = np.linspace(0, 2 * np.pi, 101, endpoint=True)
        c = XYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))
        x, y, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="xyz").T
        np.testing.assert_allclose(x, R)
        np.testing.assert_allclose(y, 0, atol=1e-15)
        np.testing.assert_allclose(z, 0, atol=1e-15)
        c.rotate(angle=np.pi / 2)
        c.flip([0, 1, 0])
        c.translate([1, 1, 1])
        r, p, z = c.compute_coordinates(grid=np.array([[0.0, 0.0, 0.0]]), basis="rpz").T
        np.testing.assert_allclose(r, np.sqrt(1**2 + (R - 1) ** 2))
        np.testing.assert_allclose(p, np.arctan2(-(R - 1), 1))
        np.testing.assert_allclose(z, 1)

    @pytest.mark.unit
    def test_from_FourierXYZCurve(self):
        """Test converting FourierXYZCurve to XYZCurve object."""
        npts = 4000
        # make a simple circular curve of radius 2
        R = 2
        c = FourierXYZCurve()
        c2 = XYZCurve.from_FourierXYZCurve(c, grid=1000)
        c3 = c.to_XYZCurve(grid=1000)

        np.testing.assert_allclose(
            c.compute_length(grid=npts), R * 2 * np.pi, atol=2e-3
        )
        np.testing.assert_allclose(
            c2.compute_length(grid=npts), R * 2 * np.pi, atol=2e-3
        )
        np.testing.assert_allclose(
            c3.compute_length(grid=npts), R * 2 * np.pi, atol=2e-3
        )
        grid = LinearGrid(N=20, endpoint=True)
        coords1 = c.compute_coordinates(grid=grid)
        coords2 = c2.compute_coordinates(grid=grid)
        coords3 = c3.compute_coordinates(grid=grid)

        np.testing.assert_allclose(coords1, coords2, atol=1e-10)
        np.testing.assert_allclose(coords2, coords3)

    @pytest.mark.unit
    def test_asserts_and_errors(self):
        """Test error checking when creating or setting properties of XYZCurve."""
        # make a simple circular curve of radius 2
        R = 2
        phi = np.linspace(0, 2 * np.pi, 101, endpoint=True)
        c = XYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))
        with pytest.raises(TypeError):
            c.grid = [1, 2, 3]
        with pytest.raises(AssertionError):
            XYZCurve(
                X=R * np.cos(phi[:-1]),
                Y=R * np.sin(phi[:-1]),
                Z=np.zeros_like(phi[:-1]),
            )
        with pytest.raises(AssertionError):
            XYZCurve(
                X=R * np.cos(phi),
                Y=R * np.sin(phi),
                Z=np.zeros_like(phi),
                knots=phi,
                period=None,
            )
        # change number of knots, should raise error since is different than
        # existing knots
        phi = np.linspace(0, 2 * np.pi, 102, endpoint=True)
        with pytest.raises(ValueError):
            c.X = R * np.cos(phi)
        with pytest.raises(ValueError):
            c.Y = R * np.sin(phi)
        with pytest.raises(ValueError):
            c.Z = np.zeros_like(phi)

    @pytest.mark.unit
    def test_misc(self):
        """Test getting/setting misc attributes of XYZCurve."""
        # make a simple circular curve of radius 2
        R = 2
        phi = np.linspace(0, 2 * np.pi, 101, endpoint=True)
        c = XYZCurve(X=R * np.cos(phi), Y=R * np.sin(phi), Z=np.zeros_like(phi))
        grid = LinearGrid(M=2, N=2)
        c.grid = grid
        assert grid.eq(c.grid)

        s = c.copy()
        assert s.eq(c)
