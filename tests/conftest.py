"""Test fixtures for computing equilibria etc."""

import os

import h5py
import jax
import numpy as np
import pytest
from netCDF4 import Dataset

from desc.__main__ import main
from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    FourierRZCoil,
    FourierXYZCoil,
    MixedCoilSet,
    SplineXYZCoil,
)
from desc.compute import rpz2xyz_vec
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.examples import get
from desc.grid import LinearGrid
from desc.vmec import VMECIO


@pytest.fixture(scope="class", autouse=True)
def clear_caches_before():
    """Automatically run before each test to clear caches and reduce OOM issues."""
    jax.clear_caches()


@pytest.fixture(scope="session")
def TmpDir(tmpdir_factory):
    """Create a temporary directory to store testing files."""
    dir_path = tmpdir_factory.mktemp("test_results")
    return dir_path


@pytest.fixture(scope="session")
def SOLOVEV(tmpdir_factory):
    """Run SOLOVEV example."""
    input_path = ".//tests//inputs//SOLOVEV"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("SOLOVEV_out.h5")
    desc_nc_path = output_dir.join("SOLOVEV_out.nc")
    vmec_nc_path = ".//tests//inputs//wout_SOLOVEV.nc"
    booz_nc_path = output_dir.join("SOLOVEV_bx.nc")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running SOLOVEV test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "--numpy", "-vv"]
    main(args)

    SOLOVEV_out = {
        "input_path": input_path,
        "desc_h5_path": desc_h5_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
        "booz_nc_path": booz_nc_path,
    }
    return SOLOVEV_out


@pytest.fixture(scope="session")
def DSHAPE(tmpdir_factory):
    """Run DSHAPE fixed rotational transform example."""
    input_path = ".//tests//inputs//DSHAPE"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("DSHAPE_out.h5")
    desc_nc_path = output_dir.join("DSHAPE_out.nc")
    vmec_nc_path = ".//tests//inputs//wout_DSHAPE.nc"
    booz_nc_path = output_dir.join("DSHAPE_bx.nc")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE fixed rotational transform test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    main(args)

    DSHAPE_out = {
        "input_path": input_path,
        "desc_h5_path": desc_h5_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
        "booz_nc_path": booz_nc_path,
    }
    return DSHAPE_out


@pytest.fixture(scope="session")
def DSHAPE_current(tmpdir_factory):
    """Run DSHAPE fixed toroidal current example."""
    input_path = ".//tests//inputs//DSHAPE_current"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("DSHAPE_current_out.h5")
    desc_nc_path = output_dir.join("DSHAPE_current_out.nc")
    vmec_nc_path = ".//tests//inputs//wout_DSHAPE.nc"
    booz_nc_path = output_dir.join("DSHAPE_bx.nc")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE fixed toroidal current test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    main(args)

    DSHAPE_current_out = {
        "input_path": input_path,
        "desc_h5_path": desc_h5_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
        "booz_nc_path": booz_nc_path,
    }
    return DSHAPE_current_out


@pytest.fixture(scope="session")
def HELIOTRON(tmpdir_factory):
    """Run HELIOTRON fixed rotational transform example."""
    input_path = ".//tests//inputs//HELIOTRON"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("HELIOTRON_out.h5")
    desc_nc_path = output_dir.join("HELIOTRON_out.nc")
    vmec_nc_path = ".//tests//inputs//wout_HELIOTRON.nc"
    booz_nc_path = output_dir.join("HELIOTRON_bx.nc")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running HELIOTRON fixed rotational transform test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    main(args)

    HELIOTRON_out = {
        "input_path": input_path,
        "desc_h5_path": desc_h5_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
        "booz_nc_path": booz_nc_path,
    }
    return HELIOTRON_out


@pytest.fixture(scope="session")
def HELIOTRON_vac(tmpdir_factory):
    """Run HELIOTRON vacuum (vacuum) example."""
    input_path = ".//tests//inputs//HELIOTRON_vacuum"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("HELIOTRON_vacuum_out.h5")
    desc_nc_path = output_dir.join("HELIOTRON_vacuum_out.nc")
    vmec_nc_path = ".//tests//inputs//wout_HELIOTRON_vacuum.nc"
    booz_nc_path = output_dir.join("HELIOTRON_vacuum_bx.nc")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running HELIOTRON vacuum (vacuum) test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    with pytest.warns(UserWarning, match="Vacuum objective assumes 0 pressure"):
        main(args)

    HELIOTRON_vacuum_out = {
        "input_path": input_path,
        "desc_h5_path": desc_h5_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
        "booz_nc_path": booz_nc_path,
    }
    return HELIOTRON_vacuum_out


@pytest.fixture(scope="session")
def DummyStellarator(tmpdir_factory):
    """Create and save a dummy stellarator configuration for testing."""
    output_dir = tmpdir_factory.mktemp("result")
    output_path = output_dir.join("DummyStellarator.h5")

    inputs = {
        "sym": True,
        "NFP": 3,
        "Psi": 1.0,
        "L": 4,
        "M": 2,
        "N": 2,
        "pressure": np.array([[0, 1e4], [2, -2e4], [4, 1e4]]),
        "iota": np.array([[0, 0.5], [2, 0.5]]),
        "surface": np.array(
            [
                [0, 0, 0, 3, 0],
                [0, 1, 0, 1, 0],
                [0, -1, 0, 0, -1],
                [0, 1, 1, 0.3, 0],
                [0, -1, -1, 0.3, 0],
                [0, 1, -1, 0, -0.3],
                [0, -1, 1, 0, 0.3],
            ],
        ),
        "axis": np.array([[-1, 0, -0.2], [0, 3.4, 0], [1, 0.2, 0]]),
    }
    eq = Equilibrium(**inputs)
    eq.save(output_path)

    DummyStellarator_out = {"output_path": output_path}
    return DummyStellarator_out


@pytest.fixture(scope="session")
def DummyCoilSet(tmpdir_factory):
    """Create and save a dummy coil set for testing."""
    output_dir = tmpdir_factory.mktemp("result")
    output_path_sym = output_dir.join("DummyCoilSet_sym.h5")
    output_path_asym = output_dir.join("DummyCoilSet_asym.h5")

    eq = get("precise_QH")
    minor_radius = eq.compute("a")["a"]

    # CoilSet with symmetry
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
    coilset_sym = CoilSet(coils, NFP=eq.NFP, sym=eq.sym)
    coilset_sym.save(output_path_sym)

    # equivalent CoilSet without symmetry
    coilset_asym = CoilSet.from_symmetry(coilset_sym, NFP=eq.NFP, sym=eq.sym)
    coilset_asym.save(output_path_asym)

    DummyCoilSet_out = {
        "output_path_sym": output_path_sym,
        "output_path_asym": output_path_asym,
    }
    return DummyCoilSet_out


@pytest.fixture(scope="session")
def DummyMixedCoilSet(tmpdir_factory):
    """Create and save a dummy mixed coil set for testing."""
    output_dir = tmpdir_factory.mktemp("result")
    output_path = output_dir.join("DummyMixedCoilSet.h5")

    tf_coil = FourierPlanarCoil(current=3, center=[2, 0, 0], normal=[0, 1, 0], r_n=[1])
    tf_coil.rotate(angle=np.pi / 4)
    tf_coilset = CoilSet(tf_coil, NFP=2, sym=True)

    vf_coil = FourierRZCoil(current=-1, R_n=3, Z_n=-1)
    vf_coilset = CoilSet.linspaced_linear(
        vf_coil, displacement=[0, 0, 2], n=3, endpoint=True
    )
    xyz_coil = FourierXYZCoil(current=2)
    phi = 2 * np.pi * np.linspace(0, 1, 20, endpoint=True) ** 2
    spline_coil = SplineXYZCoil(
        current=1,
        X=np.cos(phi),
        Y=np.sin(phi),
        Z=np.zeros_like(phi),
        knots=np.linspace(0, 2 * np.pi, len(phi)),
    )
    full_coilset = MixedCoilSet(
        (tf_coilset, vf_coilset, xyz_coil, spline_coil), check_intersection=False
    )

    full_coilset.save(output_path)
    DummyMixedCoilSet_out = {"output_path": output_path}
    return DummyMixedCoilSet_out


@pytest.fixture(scope="session")
def writer_test_file(tmpdir_factory):
    """Create temporary output directory."""
    output_dir = tmpdir_factory.mktemp("writer")
    return output_dir.join("writer_test_file")


@pytest.fixture(scope="session")
def reader_test_file(tmpdir_factory):
    """Create temporary input directory."""
    input_dir = tmpdir_factory.mktemp("reader")
    filename = input_dir.join("reader_test_file")
    thedict = {"a": "a", "b": "b", "c": "c"}
    f = h5py.File(filename, "w")
    subgroup = "subgroup"
    g = f.create_group(subgroup)
    for key in thedict.keys():
        f.create_dataset(key, data=thedict[key])
        g.create_dataset(key, data=thedict[key])
    f.close()
    return filename


@pytest.fixture(scope="session")
def VMEC_save(SOLOVEV, tmpdir_factory):
    """Save an equilibrium in VMEC netcdf format for comparison."""
    vmec = Dataset(str(SOLOVEV["vmec_nc_path"]), mode="r")
    eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
    eq.change_resolution(M=vmec.variables["mpol"][:] - 1, N=vmec.variables["ntor"][:])
    VMECIO.save(
        eq, str(SOLOVEV["desc_nc_path"]), surfs=vmec.variables["ns"][:], verbose=0
    )
    desc = Dataset(str(SOLOVEV["desc_nc_path"]), mode="r")
    return vmec, desc
