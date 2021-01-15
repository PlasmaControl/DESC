import pytest
import subprocess
import os


@pytest.fixture(scope="session")
def TmpDir(tmpdir_factory):
    """Creates a temporary directory to store testing files."""

    dir_path = tmpdir_factory.mktemp("test_results")
    return dir_path


@pytest.fixture(scope="session")
def SOLOVEV(tmpdir_factory):
    max_time = 2 * 60  # 2 minute max time for SOLOVEV run

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


def pytest_collection_modifyitems(items):
    for item in items:
        if "DSHAPE" in getattr(item, "fixturenames", ()):
            item.add_marker("slow")
