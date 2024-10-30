"""Benchmarks for timing comparison on cpu (that are small enough to run on CI)."""

import jax
import numpy as np
import pytest

import desc

desc.set_device("cpu")
import desc.examples
from desc.basis import FourierZernikeBasis
from desc.equilibrium import Equilibrium
from desc.grid import ConcentricGrid, LinearGrid
from desc.magnetic_fields import ToroidalMagneticField
from desc.objectives import (
    BoundaryError,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.optimize import LinearConstraintProjection, ProximalProjection
from desc.perturbations import perturb
from desc.transform import Transform


@pytest.mark.benchmark()
def test_build_transform_fft_lowres(benchmark):
    """Test time to build a transform (after compilation) for low resolution."""

    def setup():
        jax.clear_caches()

    def build():
        L = 5
        M = 5
        N = 5
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=50)


@pytest.mark.benchmark()
def test_build_transform_fft_midres(benchmark):
    """Test time to build a transform (after compilation) for mid resolution."""

    def setup():
        jax.clear_caches()

    def build():
        L = 15
        M = 15
        N = 15
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=50)


@pytest.mark.benchmark()
def test_build_transform_fft_highres(benchmark):
    """Test time to build a transform (after compilation) for high resolution."""

    def setup():
        jax.clear_caches()

    def build():
        L = 25
        M = 25
        N = 25
        grid = ConcentricGrid(L=L, M=M, N=N)
        basis = FourierZernikeBasis(L=L, M=M, N=N)
        transf = Transform(grid, basis, method="fft", build=False)
        transf.build()

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=50)


@pytest.mark.benchmark()
def test_equilibrium_init_lowres(benchmark):
    """Test time to create an equilibrium for low resolution."""

    def setup():
        jax.clear_caches()

    def build():
        L = 5
        M = 5
        N = 5
        _ = Equilibrium(L=L, M=M, N=N)

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=50)


@pytest.mark.benchmark()
def test_equilibrium_init_medres(benchmark):
    """Test time to create an equilibrium for medium resolution."""

    def setup():
        jax.clear_caches()

    def build():
        L = 15
        M = 15
        N = 15
        _ = Equilibrium(L=L, M=M, N=N)

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=50)


@pytest.mark.benchmark()
def test_equilibrium_init_highres(benchmark):
    """Test time to create an equilibrium for high resolution."""

    def setup():
        jax.clear_caches()

    def build():
        L = 25
        M = 25
        N = 25
        _ = Equilibrium(L=L, M=M, N=N)

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=50)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compile_dshape_current(benchmark):
    """Benchmark compiling objective."""

    def setup():
        jax.clear_caches()
        eq = desc.examples.get("DSHAPE_current")
        objective = LinearConstraintProjection(
            get_equilibrium_objective(eq),
            ObjectiveFunction(
                maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
            ),
        )
        objective.build(eq)
        args = (
            objective,
            eq,
        )
        kwargs = {}
        return args, kwargs

    def run(objective, eq):
        objective.compile()

    benchmark.pedantic(run, setup=setup, rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compile_atf(benchmark):
    """Benchmark compiling objective."""

    def setup():
        jax.clear_caches()
        eq = desc.examples.get("ATF")
        objective = LinearConstraintProjection(
            get_equilibrium_objective(eq),
            ObjectiveFunction(
                maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
            ),
        )
        objective.build(eq)
        args = (objective, eq)
        kwargs = {}
        return args, kwargs

    def run(objective, eq):
        objective.compile()

    benchmark.pedantic(run, setup=setup, rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_dshape_current(benchmark):
    """Benchmark computing objective."""
    jax.clear_caches()
    eq = desc.examples.get("DSHAPE_current")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x, objective):
        objective.compute_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, objective), rounds=50, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_atf(benchmark):
    """Benchmark computing objective."""
    jax.clear_caches()
    eq = desc.examples.get("ATF")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x, objective):
        objective.compute_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, objective), rounds=50, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_jac_dshape_current(benchmark):
    """Benchmark computing jacobian."""
    jax.clear_caches()
    eq = desc.examples.get("DSHAPE_current")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x):
        objective.jac_scaled(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=15, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_jac_atf(benchmark):
    """Benchmark computing jacobian."""
    jax.clear_caches()
    eq = desc.examples.get("ATF")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(eq)
    objective.compile()
    x = objective.x(eq)

    def run(x):
        objective.jac_scaled(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=15, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_perturb_1(benchmark):
    """Benchmark 1st order perturbations."""

    def setup():
        jax.clear_caches()
        eq = desc.examples.get("SOLOVEV")
        objective = get_equilibrium_objective(eq)
        objective.build(eq)
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

    benchmark.pedantic(perturb, setup=setup, rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_perturb_2(benchmark):
    """Benchmark 2nd order perturbations."""

    def setup():
        jax.clear_caches()
        eq = desc.examples.get("SOLOVEV")
        objective = get_equilibrium_objective(eq)
        objective.build(eq)
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

    benchmark.pedantic(perturb, setup=setup, rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_proximal_jac_atf(benchmark):
    """Benchmark computing jacobian of constrained proximal projection."""
    jax.clear_caches()
    eq = desc.examples.get("ATF")
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.linspace(0.1, 1, 10))
    objective = ObjectiveFunction(QuasisymmetryTwoTerm(eq, grid=grid))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(objective, constraint, eq)
    prox.build()
    prox.compile()
    x = prox.x(eq)

    def run(x):
        prox.jac_scaled(x, prox.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=15, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_proximal_freeb_compute(benchmark):
    """Benchmark computing free boundary objective with proximal constraint."""
    jax.clear_caches()
    eq = desc.examples.get("ESTELL")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(6, 6, 6, 12, 12, 12)
    field = ToroidalMagneticField(1.0, 1.0)  # just a dummy field for benchmarking
    objective = ObjectiveFunction(BoundaryError(eq, field=field))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(objective, constraint, eq)
    obj = LinearConstraintProjection(
        prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
    )
    obj.build()
    obj.compile()
    x = obj.x(eq)

    def run(x):
        obj.compute_scaled_error(x, obj.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=50, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_proximal_freeb_jac(benchmark):
    """Benchmark computing free boundary jacobian with proximal constraint."""
    jax.clear_caches()
    eq = desc.examples.get("ESTELL")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(6, 6, 6, 12, 12, 12)
    field = ToroidalMagneticField(1.0, 1.0)  # just a dummy field for benchmarking
    objective = ObjectiveFunction(BoundaryError(eq, field=field))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(objective, constraint, eq)
    obj = LinearConstraintProjection(
        prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
    )
    obj.build()
    obj.compile()
    x = obj.x(eq)

    def run(x):
        obj.jac_scaled(x, prox.constants).block_until_ready()

    benchmark.pedantic(run, args=(x,), rounds=15, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_solve_fixed_iter(benchmark):
    """Benchmark running eq.solve for fixed iteration count."""
    jax.clear_caches()
    eq = desc.examples.get("ESTELL")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(6, 6, 6, 12, 12, 12)

    def run(eq):
        eq.solve(maxiter=20, ftol=0, xtol=0, gtol=0)

    benchmark.pedantic(run, args=(eq,), rounds=10, iterations=1)
