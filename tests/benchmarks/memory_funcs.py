"""Benchmark memory usage of various functions in DESC."""

import gc
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

import desc

if sys.argv[2] in ["GPU", "gpu"]:
    # Set the environment variable to use the GPU
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    from desc import set_device

    set_device("gpu")

import desc.examples
from desc.backend import jax
from desc.grid import LinearGrid
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


@pytest.mark.memory
def test_objective_jac_w7x():
    """Benchmark computing jacobian."""
    jax.clear_caches()
    gc.collect()
    eq = desc.examples.get("W7-X")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(verbose=0)
    x = objective.x(eq)

    for _ in range(3):
        _ = objective.jac_scaled_error(x, objective.constants).block_until_ready()


@pytest.mark.memory
def test_proximal_jac_w7x_with_eq_update():
    """Benchmark computing jacobian of constrained proximal projection."""
    jax.clear_caches()
    gc.collect()
    eq = desc.examples.get("W7-X")
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.linspace(0.1, 1, 10))
    objective = ObjectiveFunction(QuasisymmetryTwoTerm(eq, grid=grid))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(
        objective,
        constraint,
        eq,
        perturb_options={"verbose": 0},
        solve_options={
            "verbose": 0,
            "maxiter": 0,
            "solve_during_proximal_build": False,
        },
    )
    prox.build(verbose=0)
    x = prox.x(eq)
    for _ in range(3):
        # we change x slightly to profile solve/perturb equilibrium too
        # this one will compile everything inside the function
        x = x.at[0].add(np.random.rand() * 0.001)
        _ = prox.jac_scaled_error(x, prox.constants).block_until_ready()


@pytest.mark.memory
def test_proximal_freeb_jac():
    """Benchmark computing free boundary jacobian with proximal constraint."""
    jax.clear_caches()
    gc.collect()
    eq = desc.examples.get("ESTELL")
    res = 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(res, res, res, 2 * res, 2 * res, 2 * res)
    field = ToroidalMagneticField(1.0, 1.0)  # just a dummy field for benchmarking
    objective = ObjectiveFunction(BoundaryError(eq, field=field))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(
        objective, constraint, eq, solve_options={"solve_during_proximal_build": False}
    )
    obj = LinearConstraintProjection(
        prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
    )
    obj.build(verbose=0)
    x = obj.x(eq)
    for _ in range(3):
        _ = obj.jac_scaled_error(x, prox.constants).block_until_ready()


@pytest.mark.memory
def test_proximal_freeb_jac_batched():
    """Benchmark computing free boundary jacobian with proximal constraint."""
    jax.clear_caches()
    gc.collect()
    eq = desc.examples.get("ESTELL")
    res = 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(res, res, res, 2 * res, 2 * res, 2 * res)
    field = ToroidalMagneticField(1.0, 1.0)  # just a dummy field for benchmarking
    # TODO(1623): Once the memory issue is fixed, reduce the amount of chunking
    objective = ObjectiveFunction(
        BoundaryError(eq, field=field),
        deriv_mode="batched",
        jac_chunk_size=100,
    )
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(
        objective, constraint, eq, solve_options={"solve_during_proximal_build": False}
    )
    obj = LinearConstraintProjection(
        prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
    )
    obj.build(verbose=0)
    x = obj.x(eq)
    for _ in range(3):
        _ = obj.jac_scaled_error(x, prox.constants).block_until_ready()


@pytest.mark.memory
def test_proximal_freeb_jac_blocked():
    """Benchmark computing free boundary jacobian with proximal constraint."""
    jax.clear_caches()
    gc.collect()
    eq = desc.examples.get("ESTELL")
    res = 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(res, res, res, 2 * res, 2 * res, 2 * res)
    field = ToroidalMagneticField(1.0, 1.0)  # just a dummy field for benchmarking
    # TODO(1623): Once the memory issue is fixed, reduce the amount of chunking
    objective = ObjectiveFunction(
        BoundaryError(eq, field=field, jac_chunk_size=100),
        deriv_mode="blocked",
    )
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(
        objective, constraint, eq, solve_options={"solve_during_proximal_build": False}
    )
    obj = LinearConstraintProjection(
        prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
    )
    obj.build(verbose=0)
    x = obj.x(eq)
    for _ in range(3):
        _ = obj.jac_scaled_error(x, prox.constants).block_until_ready()


@pytest.mark.memory
def test_proximal_jac_ripple():
    """Benchmark computing objective jacobian for effective ripple."""
    _test_proximal_ripple(False, "jac_scaled_error")


@pytest.mark.memory
def test_proximal_jac_ripple_spline():
    """Benchmark computing objective jacobian for effective ripple."""
    _test_proximal_ripple(True, "jac_scaled_error")


@pytest.mark.memory
def _test_proximal_ripple(spline, method):
    jax.clear_caches()
    gc.collect()
    eq = desc.examples.get("HELIOTRON")
    res = 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(res, res, res, 2 * res, 2 * res, 2 * res)
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
    prox = ProximalProjection(
        objective, constraint, eq, solve_options={"solve_during_proximal_build": False}
    )
    prox.build(verbose=0)
    x = prox.x(eq)
    for _ in range(3):
        _ = getattr(prox, method)(x, prox.constants).block_until_ready()


@pytest.mark.memory
def test_eq_solve():
    """Benchmark equilibrium solve with 2 steps."""
    res = 12
    eq = desc.examples.get("precise_QA")
    eq.change_resolution(L=res, M=res, L_grid=2 * res, M_grid=2 * res)
    # this test is mostly for intermediate operations, so having a chunk size
    # of 100 will be fine to see their effect
    obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=100, deriv_mode="batched")
    obj.build(verbose=0)
    eq.solve(
        objective=obj,
        optimizer="lsq-exact",
        ftol=0,
        xtol=0,
        gtol=0,
        maxiter=2,
        verbose=0,
    )


if __name__ == "__main__":
    func = str(sys.argv[1])
    print(f"Running {func}...")

    # I know this is not the best way to do this, but just easy for now
    if func == "test_objective_jac_w7x":
        test_objective_jac_w7x()
    elif func == "test_proximal_jac_w7x_with_eq_update":
        test_proximal_jac_w7x_with_eq_update()
    elif func == "test_proximal_freeb_jac":
        test_proximal_freeb_jac()
    elif func == "test_proximal_freeb_jac_batched":
        test_proximal_freeb_jac_batched()
    elif func == "test_proximal_freeb_jac_blocked":
        test_proximal_freeb_jac_blocked()
    elif func == "test_proximal_jac_ripple":
        test_proximal_jac_ripple()
    elif func == "test_proximal_jac_ripple_spline":
        test_proximal_jac_ripple_spline()
    elif func == "test_eq_solve":
        test_eq_solve()
    else:
        print(f"Invalid function name {func}.")
