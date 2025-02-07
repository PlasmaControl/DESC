"""Benchmarks for timing comparison."""

import os

import pytest

import desc
from desc.objectives import EffectiveRipple, ObjectiveFunction

desc.set_device("cpu")
from desc.__main__ import main


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

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
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

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
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

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
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

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
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

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
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

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
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

    args = ["-o", str(desc_h5_path), input_filename, "-vv"]
    benchmark(main, args)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_ripple_objective_2D(benchmark):
    """Benchmark computing objective for effective ripple."""
    eq = desc.examples.get("W7-X")
    eq.change_resolution(M=eq.M // 2, N=eq.N // 2)
    num_transit = 5
    objective = ObjectiveFunction(
        [
            EffectiveRipple(
                eq,
                num_transit=num_transit,
                num_well=10 * num_transit,
                num_quad=16,
            )
        ]
    )
    objective.build(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x, objective):
        objective.compute_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, objective), rounds=3, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_ripple_objective_1D(benchmark):
    """Benchmark computing objective for effective ripple."""
    eq = desc.examples.get("W7-X")
    eq.change_resolution(M=eq.M // 2, N=eq.N // 2)
    num_transit = 5
    objective = ObjectiveFunction(
        [
            EffectiveRipple(
                eq,
                num_transit=num_transit,
                num_well=10 * num_transit,
                num_quad=16,
                spline=True,
            )
        ]
    )
    objective.build(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x, objective):
        objective.compute_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, objective), rounds=3, iterations=1)
