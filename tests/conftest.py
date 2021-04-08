import pytest
import subprocess
import os
import h5py
import numpy as np

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.__main__ import main


@pytest.fixture
def plot_eq():
    eq = EquilibriaFamily.load(load_from="./tests/inputs/SOLOVEV_output.h5")[-1]
    return eq


@pytest.fixture(scope="session")
def TmpDir(tmpdir_factory):
    """Create a temporary directory to store testing files."""
    dir_path = tmpdir_factory.mktemp("test_results")
    return dir_path


@pytest.fixture(scope="session")
def SOLOVEV(tmpdir_factory):
    """Run SOLOVEV example."""
    input_path = "examples//DESC//SOLOVEV"
    output_dir = tmpdir_factory.mktemp("result")
    output_path = output_dir.join("SOLOVEV_out")
    desc_nc_path = output_dir.join("SOLOVEV_out.nc")
    vmec_nc_path = "examples//VMEC//wout_SOLOVEV.nc"

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running SOLOVEV test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(output_path), input_filename, "--numpy"]
    main(args)

    SOLOVEV_out = {
        "input_path": input_path,
        "output_path": output_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
    }
    return SOLOVEV_out


@pytest.fixture(scope="session")
def DSHAPE(tmpdir_factory):
    """Run DSHAPE example."""
    input_path = "examples//DESC//DSHAPE"
    output_dir = tmpdir_factory.mktemp("result")
    output_path = output_dir.join("DSHAPE_out")
    desc_nc_path = output_dir.join("DSHAPE_out.nc")
    vmec_nc_path = "examples//VMEC//wout_DSHAPE_s256_M14_N0.nc"

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(output_path), input_filename]
    main(args)

    DSHAPE_out = {
        "input_path": input_path,
        "output_path": output_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
    }
    return DSHAPE_out


@pytest.fixture(scope="session")
def DummyStellarator(tmpdir_factory):
    """Create and save a dummy stellarator configuration for testing."""
    output_dir = tmpdir_factory.mktemp("result")
    output_path = output_dir.join("DummyStellarator")

    inputs = {
        "sym": True,
        "NFP": 1,
        "Psi": 1.0,
        "L": 2,
        "M": 2,
        "N": 1,
        "profiles": np.array([[0, 1e4, 0.5], [2, -2e4, 0.5], [4, 1e4, 0]]),
        "boundary": np.array(
            [
                [0, 0, 0, 3, 0],
                [0, 1, 0, 1, 0],
                [0, -1, 0, 0, 1],
                [0, 1, 1, 0.3, 0],
                [0, -1, -1, -0.3, 0],
                [0, 1, -1, -0.3, 0],
                [0, -1, 1, -0.3, 0],
            ],
        ),
        "bdry_mode": "lcfs",
        "objective": "force",
        "optimizer": "scipy-trf",
    }
    eq = Equilibrium(inputs=inputs)
    eq.build()
    eq.save(output_path)

    DummyStellarator_out = {
        "output_path": output_path,
    }
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


"""
def pytest_collection_modifyitems(items):
    for item in items:
        if "DSHAPE" in getattr(item, "fixturenames", ()):
            item.add_marker("slow")
"""
