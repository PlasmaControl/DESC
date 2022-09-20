import os
import pytest

from desc.__main__ import main
from desc.grid import ConcentricGrid
from desc.basis import FourierZernikeBasis
from desc.transform import Transform


@pytest.mark.benchmark(
    min_rounds=1, max_time=50, disable_gc=False, warmup=True, warmup_iterations=50
)
def test_build_transform_fft_lowres(benchmark):
    """Test time to build a transform (after compilation) for low resolution."""

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
    """Test time to build a transform (after compilation) for mid resolution."""

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
    """Test time to build a transform (after compilation) for high resolution."""

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
    """Benchmark the SOLOVEV example."""
    input_path = ".//tests//benchmarks//SOLOVEV"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("SOLOVEV_out.h5")
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
    """Benchmark the DSHAPE fixed rotational transform example."""
    input_path = ".//tests//benchmarks//DSHAPE"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("DSHAPE_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE fixed rotational transform test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=1, max_time=300, disable_gc=True, warmup=False)
def test_DSHAPE_current_run(tmpdir_factory, benchmark):
    """Benchmark the DSHAPE fixed toroidal current example."""
    input_path = ".//tests//benchmarks//DSHAPE_current"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("DSHAPE_current_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE fixed toroidal current test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=1, max_time=300, disable_gc=True, warmup=False)
def test_HELIOTRON_run(tmpdir_factory, benchmark):
    """Benchmark the HELIOTRON fixed rotational transform example."""
    input_path = ".//tests//benchmarks//HELIOTRON"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("HELIOTRON_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running HELIOTRON fixed rotational transform test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=1, max_time=300, disable_gc=True, warmup=False)
def test_HELIOTRON_vacuum_run(tmpdir_factory, benchmark):
    """Benchmark the HELIOTRON vacuum (fixed current) example."""
    input_path = ".//tests//benchmarks//HELIOTRON_vacuum"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("HELIOTRON_vacuum_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running HELIOTRON vacuum (fixed current) test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    benchmark(main, args)
    return None
