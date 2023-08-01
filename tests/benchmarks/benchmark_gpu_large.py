"""Benchmarks for timing comparison on gpu backend."""

import os

import numpy as np
import pytest

import desc

desc.set_device("gpu")
from desc.__main__ import main
from desc.basis import FourierZernikeBasis
from desc.equilibrium import EquilibriaFamily
from desc.grid import ConcentricGrid
from desc.objectives import get_equilibrium_objective, get_fixed_boundary_constraints
from desc.perturbations import perturb
from desc.transform import Transform


@pytest.fixture(scope="session")
def TmpDir(tmpdir_factory):
    """Create a temporary directory to store testing files."""
    dir_path = tmpdir_factory.mktemp("test_results")
    return dir_path


@pytest.fixture(scope="session")
def SOLOVEV(tmpdir_factory):
    """Run SOLOVEV example."""
    input_path = "..//tests//inputs//SOLOVEV"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("SOLOVEV_out.h5")
    desc_nc_path = output_dir.join("SOLOVEV_out.nc")
    vmec_nc_path = "..//tests//inputs//wout_SOLOVEV.nc"
    booz_nc_path = output_dir.join("SOLOVEV_bx.nc")

    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running SOLOVEV test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "--numpy", "-vv", "-g"]
    main(args)

    SOLOVEV_out = {
        "input_path": input_path,
        "desc_h5_path": desc_h5_path,
        "desc_nc_path": desc_nc_path,
        "vmec_nc_path": vmec_nc_path,
        "booz_nc_path": booz_nc_path,
    }
    return SOLOVEV_out


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


@pytest.mark.benchmark(min_rounds=5, max_time=300, disable_gc=True, warmup=False)
def test_SOLOVEV_run(tmpdir_factory, benchmark):
    """Benchmark the SOLOVEV example."""
    input_path = ".//tests//inputs//SOLOVEV"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("SOLOVEV_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running SOLOVEV test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv", "-g"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=5, max_time=300, disable_gc=True, warmup=False)
def test_DSHAPE_run(tmpdir_factory, benchmark):
    """Benchmark the DSHAPE fixed rotational transform example."""
    input_path = ".//tests//inputs//DSHAPE"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("DSHAPE_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE fixed rotational transform test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv", "-g"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=5, max_time=300, disable_gc=True, warmup=False)
def test_DSHAPE_current_run(tmpdir_factory, benchmark):
    """Benchmark the DSHAPE fixed toroidal current example."""
    input_path = ".//tests//inputs//DSHAPE_current"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("DSHAPE_current_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running DSHAPE fixed toroidal current test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv", "-g"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=3, max_time=300, disable_gc=True, warmup=False)
def test_HELIOTRON_run(tmpdir_factory, benchmark):
    """Benchmark the HELIOTRON fixed rotational transform example."""
    input_path = ".//tests//inputs//HELIOTRON"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("HELIOTRON_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running HELIOTRON fixed rotational transform test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv", "-g"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=3, max_time=300, disable_gc=True, warmup=False)
def test_HELIOTRON_vacuum_run(tmpdir_factory, benchmark):
    """Benchmark the HELIOTRON vacuum (fixed current) example."""
    input_path = ".//tests//inputs//HELIOTRON_vacuum"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("HELIOTRON_vacuum_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running HELIOTRON vacuum (fixed current) test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv", "-g"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=3, max_time=300, disable_gc=True, warmup=False)
def test_ESTELL_run(tmpdir_factory, benchmark):
    """Benchmark the ESTELL example."""
    input_path = ".//examples//DESC//ESTELL"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("ESTELL_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running ESTELL test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv", "-g"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark(min_rounds=3, max_time=300, disable_gc=True, warmup=False)
def test_W7X_run(tmpdir_factory, benchmark):
    """Benchmark the W7X example."""
    input_path = ".//examples//DESC//W7-X"
    output_dir = tmpdir_factory.mktemp("result")
    desc_h5_path = output_dir.join("W7X_out.h5")
    cwd = os.path.dirname(__file__)
    exec_dir = os.path.join(cwd, "../..")
    input_filename = os.path.join(exec_dir, input_path)

    print("Running W7X test.")
    print("exec_dir=", exec_dir)
    print("cwd=", cwd)

    args = ["-o", str(desc_h5_path), input_filename, "-vv", "-g"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_perturb_1(SOLOVEV, benchmark):
    """Benchmark 1st order perturbations."""

    def setup():
        eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
        objective = get_equilibrium_objective(eq)
        constraints = get_fixed_boundary_constraints(eq)
        tr_ratio = [0.01, 0.25, 0.25]
        dp = np.zeros_like(eq.p_l)
        dp[np.array([0, 2])] = 8e3 * np.array([1, -1])
        deltas = {"p_l": dp}

        args = (
            eq,
            objective,
            constraints,
        )
        kwargs = {
            "deltas": deltas,
            "tr_ratio": tr_ratio,
            "order": 1,
            "verbose": 2,
            "copy": True,
        }
        return args, kwargs

    benchmark.pedantic(perturb, setup=setup, rounds=5, iterations=1)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_perturb_2(SOLOVEV, benchmark):
    """Benchmark 2nd order perturbations."""

    def setup():
        eq = EquilibriaFamily.load(load_from=str(SOLOVEV["desc_h5_path"]))[-1]
        objective = get_equilibrium_objective(eq)
        constraints = get_fixed_boundary_constraints(eq)
        tr_ratio = [0.01, 0.25, 0.25]
        dp = np.zeros_like(eq.p_l)
        dp[np.array([0, 2])] = 8e3 * np.array([1, -1])
        deltas = {"p_l": dp}

        args = (
            eq,
            objective,
            constraints,
        )
        kwargs = {
            "deltas": deltas,
            "tr_ratio": tr_ratio,
            "order": 2,
            "verbose": 2,
            "copy": True,
        }
        return args, kwargs

    benchmark.pedantic(perturb, setup=setup, rounds=5, iterations=1)
    return None
