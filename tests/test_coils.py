"""Tests for coils and coilsets."""

import shutil

import numpy as np
import pytest

from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    FourierRZCoil,
    FourierXYZCoil,
    MixedCoilSet,
    SplineXYZCoil,
)
from desc.compute import rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.examples import get
from desc.geometry import FourierRZCurve, FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.magnetic_fields import SumMagneticField, VerticalMagneticField


class TestCoil:
    """Tests for singular coil objects."""

    @pytest.mark.unit
    def test_biot_savart_all_coils(self):
        """Test biot-savart implementation against analytic formula."""
        coil_grid = LinearGrid(zeta=100, endpoint=False)

        R = 2
        y = 1
        I = 1e7

        By_true = 1e-7 * 2 * np.pi * R**2 * I / (y**2 + R**2) ** (3 / 2)
        Bz_true = 1e-7 * 2 * np.pi * R**2 * I / (y**2 + R**2) ** (3 / 2)

        B_true_xyz = np.atleast_2d([0, By_true, 0])
        grid_xyz = np.atleast_2d([10, y, 0])
        grid_rpz = xyz2rpz(grid_xyz)
        B_true_rpz_xy = xyz2rpz_vec(B_true_xyz, x=grid_xyz[:, 0], y=grid_xyz[:, 1])
        B_true_rpz_phi = xyz2rpz_vec(B_true_xyz, phi=grid_rpz[:, 1])

        # FourierXYZCoil
        coil = FourierXYZCoil(I)
        B_xyz = coil.compute_magnetic_field(
            grid_xyz, basis="xyz", source_grid=coil_grid
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
        By_true = 1e-7 * 2 * np.pi * R**2 * I / (y**2 + R**2) ** (3 / 2)
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
        By_true = 1e-7 * 2 * np.pi * R**2 * I / (y**2 + R**2) ** (3 / 2)
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
    def test_converting_coil_types(self):
        """Test conversions between coil representations."""
        s = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        coil1 = FourierRZCoil(1e6, [0, 10, 2], [-2, 0, 0])
        coil2 = coil1.to_FourierXYZ(s=s)
        coil3 = coil1.to_SplineXYZ(knots=s)
        grid = LinearGrid(zeta=s)
        x1 = coil1.compute("x", grid=grid, basis="xyz")["x"]
        x2 = coil2.compute("x", grid=grid, basis="xyz")["x"]
        x3 = coil3.compute("x", grid=grid, basis="xyz")["x"]
        B1 = coil1.compute_magnetic_field(
            np.zeros((1, 3)), source_grid=grid, basis="xyz"
        )
        B2 = coil2.compute_magnetic_field(
            np.zeros((1, 3)), source_grid=grid, basis="xyz"
        )
        B3 = coil3.compute_magnetic_field(
            np.zeros((1, 3)), source_grid=grid, basis="xyz"
        )
        np.testing.assert_allclose(x1, x2, atol=1e-12)
        np.testing.assert_allclose(x1, x3, atol=1e-12)
        np.testing.assert_allclose(B1, B2, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(B1, B3, rtol=1e-3, atol=1e-8)


class TestCoilSet:
    """Tests for sets of multiple coils."""

    @pytest.mark.unit
    def test_linspaced_linear(self):
        """Field from straight solenoid."""
        R = 10
        z = np.linspace(0, 10, 10)
        I = 1e7
        n = 10
        Bz_true = np.sum(1e-7 * 2 * np.pi * R**2 * I / (z**2 + R**2) ** (3 / 2))
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
        Bp_true = np.sum(1e-7 * 4 * np.pi * N * I / 2 / np.pi / R)
        B_true = np.array([0, Bp_true, 0])
        coil = FourierPlanarCoil()
        coil.current = I
        coils = CoilSet.linspaced_angular(coil, n=N)
        B_approx = coils.compute_magnetic_field(
            [10, 0, 0], basis="rpz", source_grid=32
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
        Bp_true = np.sum(1e-7 * 4 * np.pi * N * I / 2 / np.pi / R)
        B_true = np.array([0, Bp_true, 0])
        coil = FourierPlanarCoil(I)
        coils = CoilSet.linspaced_angular(coil, angle=np.pi / 2, n=N // 4)
        coils = MixedCoilSet.from_symmetry(coils, NFP=4)
        B_approx = coils.compute_magnetic_field(
            [10, 0, 0], basis="rpz", source_grid=32
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

    @pytest.mark.unit
    def test_symmetry_magnetic_field(self):
        """Tests that compute magnetic field is correct from symmetry."""
        eq = get("precise_QH")
        minor_radius = eq.compute("a")["a"]

        # initialize CoilSet with symmetry
        num_coils = 3  # number of unique coils per half field period
        grid = LinearGrid(rho=[0.0], M=0, zeta=2 * num_coils, NFP=eq.NFP * (eq.sym + 1))
        with pytest.warns(UserWarning):  # because eq.NFP != grid.NFP
            data_center = eq.axis.compute("x", grid=grid, basis="xyz")
            data_normal = eq.compute("e^zeta", grid=grid)
        centers = data_center["x"]
        normals = rpz2xyz_vec(data_normal["e^zeta"], phi=grid.nodes[:, 2])
        coils = []
        for k in range(1, 2 * num_coils + 1, 2):
            coil = FourierPlanarCoil(
                current=1e6,
                center=centers[k, :],
                normal=normals[k, :],
                r_n=[0, minor_radius + 0.5, 0],
            )
            coils.append(coil)
        sym_coilset = CoilSet(coils, NFP=eq.NFP, sym=eq.sym)

        # equivalent CoilSet without symmetry
        asym_coilset = CoilSet.from_symmetry(sym_coilset, NFP=eq.NFP, sym=eq.sym)

        # test that both coil sets compute the same field on the plasma surface
        grid = LinearGrid(rho=[1.0], M=eq.M_grid, N=eq.N_grid, NFP=1, sym=False)
        with pytest.warns(UserWarning):  # because eq.NFP != grid.NFP
            data = eq.compute(["phi", "R", "X", "Y", "Z"], grid)

        # test in (R, phi, Z) coordinates
        nodes_rpz = np.array([data["R"], data["phi"], data["Z"]]).T
        B_sym_rpz = sym_coilset.compute_magnetic_field(nodes_rpz, basis="rpz")
        B_asym_rpz = asym_coilset.compute_magnetic_field(nodes_rpz, basis="rpz")
        np.testing.assert_allclose(B_sym_rpz, B_asym_rpz, atol=1e-14)

        # test in (X, Y, Z) coordinates
        nodes_xyz = np.array([data["X"], data["Y"], data["Z"]]).T
        B_sym_xyz = sym_coilset.compute_magnetic_field(nodes_xyz, basis="xyz")
        B_asym_xyz = asym_coilset.compute_magnetic_field(nodes_xyz, basis="xyz")
        np.testing.assert_allclose(B_sym_xyz, B_asym_xyz, atol=1e-14)

    @pytest.mark.unit
    def test_properties(self):
        """Test getting/setting of CoilSet attributes."""
        coil = FourierPlanarCoil()
        coils = CoilSet.linspaced_linear(coil, n=4)
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
                    [12.5, 0, 0],
                    [13, 0, 0],
                    [13.5, 0, 0],
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
        coil2 = FourierPlanarCoil()
        coils2 = coils1 + [coil2]
        assert coils2[-1] is coil2
        coils2 = coils1 + MixedCoilSet([coil2, coil2])
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
    def test_coilset_convert(self):
        """Test converting coilsets between different representations."""
        grid = LinearGrid(N=20)
        coil1 = FourierXYZCoil(current=1e6)
        coil2 = coil1.to_SplineXYZ(grid=grid)

        coils1 = MixedCoilSet.linspaced_angular(coil1, n=12)
        coils2 = coils1.to_SplineXYZ(grid=grid)
        assert isinstance(coils2, MixedCoilSet)
        assert all(isinstance(coil, SplineXYZCoil) for coil in coils2)
        x1 = coils1.compute("x", grid=grid, basis="xyz")
        x2 = coils2.compute("x", grid=grid, basis="xyz")
        np.testing.assert_allclose(
            [xi["x"] for xi in x1], [xi["x"] for xi in x2], atol=1e-12
        )
        B1 = coils1.compute_magnetic_field(np.array([[10, 2, 1]]), source_grid=grid)
        B2 = coils2.compute_magnetic_field(np.array([[10, 2, 1]]), source_grid=grid)
        np.testing.assert_allclose(B1, B2, rtol=1e-2)

        coils3 = CoilSet.linspaced_angular(coil2, n=12)
        coils4 = coils3.to_FourierXYZ(grid=grid)
        assert isinstance(coils4, CoilSet)
        assert all(isinstance(coil, FourierXYZCoil) for coil in coils4)
        x3 = coils3.compute("x", grid=grid, basis="xyz")
        x4 = coils4.compute("x", grid=grid, basis="xyz")
        np.testing.assert_allclose(
            [xi["x"] for xi in x3], [xi["x"] for xi in x4], atol=1e-12
        )
        B3 = coils3.compute_magnetic_field(np.array([[10, 2, 1]]), source_grid=grid)
        B4 = coils4.compute_magnetic_field(np.array([[10, 2, 1]]), source_grid=grid)
        np.testing.assert_allclose(B3, B4, rtol=1e-2)


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
    coil = FourierRZCoil()
    assert "FourierRZCoil" in str(coil)
    assert "current=1" in str(coil)

    coils = CoilSet.linspaced_angular(coil, n=4)
    assert "CoilSet" in str(coils)
    assert "4 submembers" in str(coils)

    coils.name = "MyCoils"
    assert "MyCoils" in str(coils)
