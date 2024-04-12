"""Test fixtures for computing equilibria etc."""

import os

import h5py
import jax
import numpy as np
import pytest
from netCDF4 import Dataset

from desc.__main__ import main
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import FourierCurrentPotentialField, run_regcoil
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


@pytest.fixture(scope="session")
def regcoil_ellipse_and_axisym_surf():
    """Run regcoil for elliptical eq and axisymmetric surface."""
    eq = load("./tests/inputs/ellNFP4_init_smallish.h5")

    surf_winding = FourierRZToroidalSurface(
        R_lmn=np.array([0.7035, 0.0365]),
        Z_lmn=np.array([-0.0365]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
        sym=True,
        NFP=eq.NFP,
    )
    surface_current_field = FourierCurrentPotentialField.from_surface(
        surf_winding, M_Phi=8, N_Phi=8
    )
    fields, data = run_regcoil(
        surface_current_field,
        eq,
        eval_grid=LinearGrid(M=20, N=20, NFP=eq.NFP),
        source_grid=LinearGrid(M=40, N=40, NFP=1),
        alpha=np.append(np.array([0.0]), np.logspace(-30, -1, 30)),
        current_helicity=-1,
        vacuum=True,
    )
    surface_current_field = fields[0]
    return (data, surface_current_field, eq)


@pytest.fixture(scope="session")
def regcoil_ellipse_helical_coils():
    """Run regcoil for elliptical eq and surface."""
    eq = load("./tests/inputs/ellNFP4_init_smallish.h5")

    M_Phi = 8
    N_Phi = 8
    M_egrid = 30
    N_egrid = 30
    M_sgrid = 40
    N_sgrid = 50
    alpha = 1e-18

    surf_winding = FourierRZToroidalSurface(
        R_lmn=np.array([0.7035, 0.0365]),
        Z_lmn=np.array([-0.0365]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
        sym=True,
        NFP=eq.NFP,
    )
    surface_current_field = FourierCurrentPotentialField.from_surface(
        surf_winding, M_Phi=M_Phi, N_Phi=N_Phi
    )
    surface_current_field, data = run_regcoil(
        surface_current_field,
        eq,
        eval_grid=LinearGrid(M=M_egrid, N=N_egrid, NFP=eq.NFP, sym=True),
        source_grid=LinearGrid(M=M_sgrid, N=N_sgrid, NFP=eq.NFP),
        alpha=alpha,
        current_helicity=-2,
        vacuum=True,
    )

    return (data, surface_current_field, eq)


@pytest.fixture(scope="session")
def regcoil_ellipse_helical_coils_pos_helicity():
    """Run regcoil for elliptical eq and surface with positive helicity."""
    eq = load("./tests/inputs/ellNFP4_init_smallish.h5")

    M_Phi = 8
    N_Phi = 8
    M_egrid = 30
    N_egrid = 30
    M_sgrid = 50
    N_sgrid = 50
    alpha = 1e-18

    surf_winding = FourierRZToroidalSurface(
        R_lmn=np.array([0.7035, 0.0365]),
        Z_lmn=np.array([-0.0365]),
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[-1, 0]]),
        sym=True,
        NFP=eq.NFP,
    )
    surface_current_field = FourierCurrentPotentialField.from_surface(
        surf_winding, M_Phi=M_Phi, N_Phi=N_Phi
    )
    surface_current_field, data = run_regcoil(
        surface_current_field,
        eq,
        eval_grid=LinearGrid(M=M_egrid, N=N_egrid, NFP=eq.NFP, sym=True),
        source_grid=LinearGrid(M=M_sgrid, N=N_sgrid, NFP=eq.NFP),
        alpha=alpha,
        current_helicity=2,
        vacuum=True,
    )

    return (surface_current_field, data["chi^2_B"], eq)
