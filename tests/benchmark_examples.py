import pytest
import subprocess
import os
import h5py
import numpy as np
import time

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.__main__ import main


@pytest.mark.benchmark(min_rounds=1, max_time=300, disable_gc=True, warmup=False)
def test_SOLOVEV_run(SOLOVEV_dirs, benchmark):
    """Benchmark the SOLOVEV example"""
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, SOLOVEV_dirs["input_path"])

    print("Running SOLOVEV test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(SOLOVEV_dirs["desc_h5_path"]), input_filename, "--numpy", "-vv"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=1, max_time=500, disable_gc=True, warmup=False)
def test_DSHAPE_run(DSHAPE_dirs, benchmark):
    """Benchmark the DSHAPE example."""

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, DSHAPE_dirs["input_path"])

    print("Running DSHAPE test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(DSHAPE_dirs["desc_h5_path"]), input_filename, "-vv"]
    benchmark(main, args)

    return None
