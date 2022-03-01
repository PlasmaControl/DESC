from warnings import WarningMessage
import pytest
import subprocess
import os
import h5py
import numpy as np
import time
from desc.__main__ import main

from desc.grid import Grid, LinearGrid, ConcentricGrid
from desc.basis import (
    PowerSeries,
    FourierSeries,
    DoubleFourierSeries,
    ZernikePolynomial,
    FourierZernikeBasis,
)
from desc.transform import Transform


@pytest.mark.benchmark(
    min_rounds=1, max_time=50, disable_gc=False, warmup=True, warmup_iterations=50
)
def test_build_transform_fft_lowres(benchmark):
    """Tests how long it takes to build a transform (after it has already been compiled) for lowres"""

    def build():
        L = 5
        M = 5
        N = 5
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark(build)


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=True)
def test_build_transform_fft_midres(benchmark):
    """Tests how long it takes to build a transform (after it has already been compiled) for midres"""

    def build():
        L = 15
        M = 15
        N = 15
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark.pedantic(build, iterations=1, warmup_rounds=1, rounds=50)


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=True)
def test_build_transform_fft_highres(benchmark):
    """Tests how long it takes to build a transform (after it has already been compiled) for highres"""

    def build():
        L = 25
        M = 25
        N = 25
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark.pedantic(build, iterations=1, warmup_rounds=1, rounds=25)


@pytest.fixture(scope="session")
def TmpDir(tmpdir_factory):
    """Create a temporary directory to store testing files."""
    dir_path = tmpdir_factory.mktemp("test_results")
    return dir_path


@pytest.mark.benchmark(min_rounds=1, max_time=200, disable_gc=True, warmup=False)
def test_SOLOVEV_run(tmpdir_factory, benchmark):
    """Benchmark the SOLOVEV example"""
    input_path = "examples//DESC//SOLOVEV"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("SOLOVEV_out.h5")
    desc_nc_path = output_dir.join("SOLOVEV_out.nc")
    vmec_nc_path = "examples//VMEC//wout_SOLOVEV.nc"
    booz_nc_path = output_dir.join("SOLOVEV_bx.nc")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running SOLOVEV test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=1, max_time=300, disable_gc=True, warmup=False)
def test_DSHAPE_run(tmpdir_factory, benchmark):
    """Benchmark the DSHAPE example."""
    input_path = "examples//DESC//DSHAPE"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("DSHAPE_out.h5")
    desc_nc_path = output_dir.join("DSHAPE_out.nc")
    vmec_nc_path = "examples//VMEC//wout_DSHAPE.nc"
    booz_nc_path = output_dir.join("DSHAPE_bx.nc")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    benchmark(main, args)

    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=1, max_time=300, disable_gc=True, warmup=False)
def test_HELIOTRON_run(tmpdir_factory, benchmark):
    """Benchmark the HELIOTRON example."""
    input_path = "examples//DESC//HELIOTRON"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("HELIOTRON_out.h5")
    desc_nc_path = output_dir.join("HELIOTRON_out.nc")
    vmec_nc_path = "examples//VMEC//wout_HELIOTRON.nc"
    booz_nc_path = output_dir.join("HELIOTRON_bx.nc")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running HELIOTRON test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    benchmark(main, args)

    return None
