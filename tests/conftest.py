import pytest
import subprocess
import os


@pytest.fixture(scope="session")
def SOLOVEV(tmpdir_factory):
    max_time = 5 * 60  # 5 minute max time for SOLOVEV run

    output_dir = tmpdir_factory.mktemp("result")
    output_path = output_dir.join("SOLOVEV_out")
    input_path = "examples//DESC//SOLOVEV"
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
        "output_dir": output_dir,
        "output_path": output_path,
    }
    return SOLOVEV_out


def pytest_collection_modifyitems(items):
    for item in items:
        if "DSHAPE" in getattr(item, "fixturenames", ()):
            item.add_marker("slow")
