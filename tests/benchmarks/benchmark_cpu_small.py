"""Benchmarks for timing comparison on cpu (that are small enough to run on CI)."""

import numpy as np
import pytest

import desc

desc.set_device("cpu")
import desc.examples
from desc.backend import jax
from desc.basis import FourierZernikeBasis
from desc.equilibrium import Equilibrium
from desc.grid import ConcentricGrid, LinearGrid
from desc.magnetic_fields import ToroidalMagneticField
from desc.objectives import (
    BoundaryError,
    EffectiveRipple,
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

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=20)


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

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=20)


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

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=20)


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

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=20)


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

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=20)


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

    benchmark.pedantic(build, setup=setup, iterations=1, rounds=20)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compile_dshape_current(benchmark):
    """Benchmark compiling objective."""
    eq = desc.examples.get("DSHAPE_current")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(eq)

    def run(objective):
        jax.clear_caches()
        objective.compile()

    benchmark.pedantic(run, args=(objective,), rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compile_atf(benchmark):
    """Benchmark compiling objective."""
    eq = desc.examples.get("ATF")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(eq)

    def run(objective):
        jax.clear_caches()
        objective.compile()

    benchmark.pedantic(run, args=(objective,), rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_dshape_current(benchmark):
    """Benchmark computing objective."""
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

    benchmark.pedantic(run, args=(x, objective), rounds=100, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_atf(benchmark):
    """Benchmark computing objective."""
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

    benchmark.pedantic(run, args=(x, objective), rounds=100, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_jac_dshape_current(benchmark):
    """Benchmark computing jacobian."""
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
        objective.jac_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, objective), rounds=80, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_jac_atf(benchmark):
    """Benchmark computing jacobian."""
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
        objective.jac_scaled_error(x, objective.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, objective), rounds=20, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_perturb_1(benchmark):
    """Benchmark 1st order perturbations."""

    def setup():
        jax.clear_caches()
        eq = desc.examples.get("SOLOVEV")
        objective = get_equilibrium_objective(eq)
        objective.build()
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
        objective.build()
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
    eq = desc.examples.get("ATF")
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.linspace(0.1, 1, 10))
    objective = ObjectiveFunction(QuasisymmetryTwoTerm(eq, grid=grid))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(objective, constraint, eq)
    prox.build()
    x = prox.x(eq)
    prox.jac_scaled_error(x, prox.constants).block_until_ready()

    def run(x, prox):
        prox.jac_scaled_error(x, prox.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, prox), rounds=20, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_proximal_jac_atf_with_eq_update(benchmark):
    """Benchmark computing jacobian of constrained proximal projection."""
    # Compare with test_proximal_jac_atf, this test additionally benchmarks the
    # case where the equilibrium is updated before computing the jacobian.
    eq = desc.examples.get("ATF")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(12, 12, 4, 24, 24, 8)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.linspace(0.1, 1, 10))
    objective = ObjectiveFunction(QuasisymmetryTwoTerm(eq, grid=grid))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(
        objective,
        constraint,
        eq,
        perturb_options={"verbose": 3},
        solve_options={"verbose": 3, "maxiter": 0},
    )
    prox.build(verbose=3)
    x = prox.x(eq)
    # we change x slightly to profile solve/perturb equilibrium too
    # this one will compile everything inside the function
    x = x.at[0].add(np.random.rand() * 0.001)
    _ = prox.jac_scaled_error(x, prox.constants).block_until_ready()

    def run(x, prox):
        # we change x slightly to profile solve/perturb equilibrium too
        x = x.at[0].add(np.random.rand() * 0.001)
        prox.jac_scaled_error(x, prox.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, prox), rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_proximal_freeb_compute(benchmark):
    """Benchmark computing free boundary objective with proximal constraint."""
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
    x = obj.x(eq)
    obj.compute_scaled_error(x, obj.constants).block_until_ready()

    def run(x, obj):
        obj.compute_scaled_error(x, obj.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, obj), rounds=50, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_proximal_freeb_jac(benchmark):
    """Benchmark computing free boundary jacobian with proximal constraint."""
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
    x = obj.x(eq)
    obj.jac_scaled_error(x, prox.constants).block_until_ready()

    def run(x, obj, prox):
        obj.jac_scaled_error(x, prox.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, obj, prox), rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_solve_fixed_iter_compiled(benchmark):
    """Benchmark running eq.solve for fixed iteration count after compilation."""

    def setup():
        jax.clear_caches()
        eq = desc.examples.get("ESTELL")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(6, 6, 6, 12, 12, 12)
        eq.solve(maxiter=1, ftol=0, xtol=0, gtol=0)

        return (eq,), {}

    def run(eq):
        eq.solve(maxiter=20, ftol=0, xtol=0, gtol=0)

    benchmark.pedantic(run, setup=setup, rounds=5, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_solve_fixed_iter(benchmark):
    """Benchmark running eq.solve for fixed iteration count."""

    def setup():
        jax.clear_caches()
        eq = desc.examples.get("ESTELL")
        with pytest.warns(UserWarning, match="Reducing radial"):
            eq.change_resolution(6, 6, 6, 12, 12, 12)

        return (eq,), {}

    def run(eq):
        jax.clear_caches()
        eq.solve(maxiter=20, ftol=0, xtol=0, gtol=0)

    benchmark.pedantic(run, setup=setup, rounds=5, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_LinearConstraintProjection_build(benchmark):
    """Benchmark LinearConstraintProjection build."""
    eq = desc.examples.get("W7-X")
    obj = ObjectiveFunction(ForceBalance(eq))
    con = get_fixed_boundary_constraints(eq)
    con = maybe_add_self_consistency(eq, con)
    con = ObjectiveFunction(con)
    obj.build()
    con.build()

    def run(obj, con):
        jax.clear_caches()
        lc = LinearConstraintProjection(obj, con)
        lc.build()

    benchmark.pedantic(run, args=(obj, con), rounds=10, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_ripple(benchmark):
    """Benchmark computing objective for effective ripple."""
    _test_objective_ripple(benchmark, False, "compute_scaled_error")


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_compute_ripple_spline(benchmark):
    """Benchmark computing objective for effective ripple."""
    _test_objective_ripple(benchmark, True, "compute_scaled_error")


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_grad_ripple(benchmark):
    """Benchmark computing objective gradient for effective ripple."""
    _test_objective_ripple(benchmark, False, "jac_scaled_error")


@pytest.mark.slow
@pytest.mark.benchmark
def test_objective_grad_ripple_spline(benchmark):
    """Benchmark computing objective gradient for effective ripple."""
    _test_objective_ripple(benchmark, True, "jac_scaled_error")


def _test_objective_ripple(benchmark, spline, method):
    eq = desc.examples.get("W7-X")
    with pytest.warns(UserWarning, match="Reducing radial"):
        eq.change_resolution(L=eq.L // 2, M=eq.M // 2, N=eq.N // 2)
    num_transit = 20
    objective = ObjectiveFunction(
        [
            EffectiveRipple(
                eq,
                num_transit=num_transit,
                num_well=10 * num_transit,
                num_quad=16,
                spline=spline,
            )
        ]
    )
    constraint = ObjectiveFunction([ForceBalance(eq)])
    prox = ProximalProjection(objective, constraint, eq)
    prox.build(eq)
    x = prox.x(eq)
    _ = getattr(prox, method)(x, prox.constants).block_until_ready()

    def run(x, prox):
        getattr(prox, method)(x, prox.constants).block_until_ready()

    benchmark.pedantic(run, args=(x, prox), rounds=10, iterations=1)
