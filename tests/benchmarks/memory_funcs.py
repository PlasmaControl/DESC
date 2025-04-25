"""Benchmark memory usage of various functions in DESC."""

import gc
import os
import sys
import warnings

import numpy as np

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
from desc.perturbations import perturb


def test_objective_jac_atf():
    """Benchmark computing jacobian."""
    jax.clear_caches()
    eq = desc.examples.get("ATF")
    objective = LinearConstraintProjection(
        get_equilibrium_objective(eq),
        ObjectiveFunction(
            maybe_add_self_consistency(eq, get_fixed_boundary_constraints(eq)),
        ),
    )
    objective.build(verbose=0)
    x = objective.x(eq)

    for _ in range(3):
        objective.jac_scaled_error(x, objective.constants).block_until_ready()
        gc.collect()


def test_perturb_2():
    """Benchmark 2nd order perturbations."""
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
        "verbose": 0,
        "copy": True,
    }
    for _ in range(3):
        perturb(*args, **kwargs)
        gc.collect()


def test_proximal_jac_atf_with_eq_update():
    """Benchmark computing jacobian of constrained proximal projection."""
    jax.clear_caches()
    eq = desc.examples.get("ATF")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(12, 12, 4, 24, 24, 8)
    grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.linspace(0.1, 1, 10))
    objective = ObjectiveFunction(QuasisymmetryTwoTerm(eq, grid=grid))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(
        objective,
        constraint,
        eq,
        perturb_options={"verbose": 0},
        solve_options={"verbose": 0, "maxiter": 0},
    )
    prox.build(verbose=0)
    x = prox.x(eq)
    for _ in range(3):
        # we change x slightly to profile solve/perturb equilibrium too
        # this one will compile everything inside the function
        x = x.at[0].add(np.random.rand() * 0.001)
        _ = prox.jac_scaled_error(x, prox.constants).block_until_ready()
        gc.collect()


def test_proximal_freeb_jac():
    """Benchmark computing free boundary jacobian with proximal constraint."""
    jax.clear_caches()
    eq = desc.examples.get("ESTELL")
    res = 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq.change_resolution(res, res, res, 2 * res, 2 * res, 2 * res)
    field = ToroidalMagneticField(1.0, 1.0)  # just a dummy field for benchmarking
    objective = ObjectiveFunction(BoundaryError(eq, field=field))
    constraint = ObjectiveFunction(ForceBalance(eq))
    prox = ProximalProjection(objective, constraint, eq)
    obj = LinearConstraintProjection(
        prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
    )
    obj.build(verbose=0)
    x = obj.x(eq)
    for _ in range(3):
        obj.jac_scaled_error(x, prox.constants).block_until_ready()
        gc.collect()


def test_objective_jac_ripple():
    """Benchmark computing objective jacobian for effective ripple."""
    _test_objective_ripple(False, "jac_scaled_error")


def test_objective_jac_ripple_spline():
    """Benchmark computing objective jacobian for effective ripple."""
    _test_objective_ripple(True, "jac_scaled_error")


def _test_objective_ripple(spline, method):
    eq = desc.examples.get("W7-X")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
                jac_chunk_size=1,
            )
        ]
    )
    constraint = ObjectiveFunction([ForceBalance(eq)])
    prox = ProximalProjection(objective, constraint, eq)
    prox.build(verbose=0)
    x = prox.x(eq)
    for _ in range(3):
        _ = getattr(prox, method)(x, prox.constants).block_until_ready()
        gc.collect()


if __name__ == "__main__":
    func = str(sys.argv[1])
    print(f"Running {func}...")

    # I know this is not the best way to do this, but just easy for now
    if func == "test_objective_jac_atf":
        test_objective_jac_atf()
    elif func == "test_proximal_jac_atf_with_eq_update":
        test_proximal_jac_atf_with_eq_update()
    elif func == "test_perturb_2":
        test_perturb_2()
    elif func == "test_proximal_freeb_jac":
        test_proximal_freeb_jac()
    elif func == "test_objective_jac_ripple":
        test_objective_jac_ripple()
    elif func == "test_objective_jac_ripple_spline":
        test_objective_jac_ripple_spline()
    else:
        print(f"Invalid function name {func}.")
