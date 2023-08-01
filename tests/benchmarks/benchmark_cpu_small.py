"""Benchmarks for timing comparison on cpu (that are small enough to run on CI)."""

import numpy as np
import pytest

import desc

desc.set_device("cpu")
import desc.examples
from desc.basis import FourierZernikeBasis
from desc.equilibrium import Equilibrium
from desc.grid import ConcentricGrid
from desc.objectives import get_equilibrium_objective, get_fixed_boundary_constraints
from desc.perturbations import perturb
from desc.transform import Transform


@pytest.fixture(scope="session")
def TmpDir(tmpdir_factory):
    """Create a temporary directory to store testing files."""
    dir_path = tmpdir_factory.mktemp("test_results")
    return dir_path


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


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=True)
def test_equilibrium_init_lowres(benchmark):
    """Test time to create an equilibrium for low resolution."""

    def build():
        L = 5
        M = 5
        N = 5
        _ = Equilibrium(L=L, M=M, N=N)

    benchmark.pedantic(build, iterations=1, warmup_rounds=1, rounds=25)


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=True)
def test_equilibrium_init_medres(benchmark):
    """Test time to create an equilibrium for medium resolution."""

    def build():
        L = 15
        M = 15
        N = 15
        _ = Equilibrium(L=L, M=M, N=N)

    benchmark.pedantic(build, iterations=1, warmup_rounds=1, rounds=25)


@pytest.mark.benchmark(min_rounds=1, max_time=100, disable_gc=False, warmup=True)
def test_equilibrium_init_highres(benchmark):
    """Test time to create an equilibrium for high resolution."""

    def build():
        L = 25
        M = 25
        N = 25
        _ = Equilibrium(L=L, M=M, N=N)

    benchmark.pedantic(build, iterations=1, warmup_rounds=1, rounds=25)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compile_heliotron(benchmark):
    """Benchmark compiling objective."""

    def setup():
        eq = desc.examples.get("HELIOTRON")
        objective = get_equilibrium_objective(eq)
        args = (
            objective,
            eq,
        )
        kwargs = {}
        return args, kwargs

    def run(objective, eq):
        objective.compile()

    benchmark.pedantic(run, setup=setup, rounds=5, iterations=1)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compile_dshape_current(benchmark):
    """Benchmark compiling objective."""

    def setup():
        eq = desc.examples.get("DSHAPE_current")
        objective = get_equilibrium_objective(eq)
        args = (
            objective,
            eq,
        )
        kwargs = {}
        return args, kwargs

    def run(objective, eq):
        objective.compile()

    benchmark.pedantic(run, setup=setup, rounds=5, iterations=1)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compile_atf(benchmark):
    """Benchmark compiling objective."""

    def setup():
        eq = desc.examples.get("ATF")
        objective = get_equilibrium_objective(eq)
        args = (objective, eq)
        kwargs = {}
        return args, kwargs

    def run(objective, eq):
        objective.compile()

    benchmark.pedantic(run, setup=setup, rounds=5, iterations=1)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_heliotron(benchmark):
    """Benchmark computing objective."""
    eq = desc.examples.get("HELIOTRON")
    objective = get_equilibrium_objective(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x):
        objective.compute_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=10, iterations=10)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_dshape_current(benchmark):
    """Benchmark computing objective."""
    eq = desc.examples.get("DSHAPE_current")
    objective = get_equilibrium_objective(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x):
        objective.compute_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=10, iterations=10)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_atf(benchmark):
    """Benchmark computing objective."""
    eq = desc.examples.get("ATF")
    objective = get_equilibrium_objective(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x):
        objective.compute_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=10, iterations=10)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_jac_heliotron(benchmark):
    """Benchmark computing jacobian."""
    eq = desc.examples.get("HELIOTRON")
    objective = get_equilibrium_objective(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x):
        objective.jac_scaled(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=5, iterations=5)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_jac_dshape_current(benchmark):
    """Benchmark computing jacobian."""
    eq = desc.examples.get("DSHAPE_current")
    objective = get_equilibrium_objective(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x):
        objective.jac_scaled(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=5, iterations=5)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_jac_atf(benchmark):
    """Benchmark computing jacobian."""
    eq = desc.examples.get("ATF")
    objective = get_equilibrium_objective(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x):
        objective.jac_scaled(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=5, iterations=5)
    return None


@pytest.mark.slow
@pytest.mark.benchmark
def test_perturb_1(benchmark):
    """Benchmark 1st order perturbations."""

    def setup():
        eq = desc.examples.get("SOLOVEV")
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
def test_perturb_2(benchmark):
    """Benchmark 2nd order perturbations."""

    def setup():
        eq = desc.examples.get("SOLOVEV")
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
