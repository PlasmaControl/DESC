import pytest
import subprocess
import os
from desc.equilibrium import EquilibriaFamily


@pytest.fixture
def plot_eq():
    eq = EquilibriaFamily(load_from="./tests/inputs/SOLOVEV.h5")[-1]
    return eq


@pytest.fixture(scope="session")
def TmpDir(tmpdir_factory):
    """Creates a temporary directory to store testing files."""

    dir_path = tmpdir_factory.mktemp("test_results")
    return dir_path


@pytest.fixture(scope="session")
def SOLOVEV(tmpdir_factory):
    max_time = 2 * 60  # 2 minute max run time

    input_path = "examples//DESC//SOLOVEV"
    output_dir = tmpdir_factory.mktemp("result")
    output_path = output_dir.join("SOLOVEV_out")
    desc_nc_path = output_dir.join("SOLOVEV_out.nc")
    vmec_nc_path = "examples//VMEC//wout_SOLOVEV.nc"

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running SOLOVEV test")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    SOLOVEV_run = subprocess.run(
        ["python", "-m", "desc", "-o", str(output_path), input_filename],
        stdout=subprocess.PIPE,
        universal_newlines=True,
        timeout=max_time,
        cwd=exec_dir,
    )
    SOLOVEV_out = {
        "output": SOLOVEV_run,
        "input_path": input_path,
        "output_path": output_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
    }
    return SOLOVEV_out


@pytest.fixture(scope="session")
def DSHAPE(tmpdir_factory):
    max_time = 2 * 60  # 2 minute max run time

    input_path = "examples//DESC//DSHAPE"
    output_dir = tmpdir_factory.mktemp("result")
    output_path = output_dir.join("DSHAPE_out")
    desc_nc_path = output_dir.join("DSHAPE_out.nc")
    vmec_nc_path = "examples//VMEC//wout_DSHAPE.nc"

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE test")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    DSHAPE_run = subprocess.run(
        ["python", "-m", "desc", "-o", str(output_path), input_filename],
        stdout=subprocess.PIPE,
        universal_newlines=True,
        timeout=max_time,
        cwd=exec_dir,
    )
    SOLOVEV_out = {
        "output": DSHAPE_run,
        "input_path": input_path,
        "output_path": output_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
    }
    return SOLOVEV_out


"""
def pytest_collection_modifyitems(items):
    for item in items:
        if "DSHAPE" in getattr(item, "fixturenames", ()):
            item.add_marker("slow")
"""
