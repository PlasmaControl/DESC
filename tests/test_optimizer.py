"""Tests for optimizers and Optimizer class."""

import numpy as np
import pytest
from numpy.random import default_rng
from scipy.optimize import BFGS, rosen, rosen_der

import desc.examples
from desc.backend import jit, jnp
from desc.derivatives import Derivative
from desc.equilibrium import Equilibrium
from desc.objectives import (
    Energy,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.objectives.objective_funs import _Objective
from desc.optimize import (
    LinearConstraintProjection,
    Optimizer,
    ProximalProjection,
    fmintr,
    lsqtr,
    optimizers,
    sgd,
)


@jit
def vector_fun(x, p):
    """Complicated-ish vector valued function for testing least squares."""
    a0 = x * p[0]
    a1 = jnp.exp(-(x**2) * p[1])
    a2 = jnp.cos(jnp.sin(x * p[2] - x**2 * p[3]))
    a3 = jnp.sum(
        jnp.array([(x + 2) ** -(i * 2) * pi ** (i + 1) for i, pi in enumerate(p[3:])]),
        axis=0,
    )
    return a0 + a1 + 3 * a2 + a3


A0 = 1
B0 = 2
C0 = -1
A1 = 4
B1 = 8
C1 = -1
SCALAR_FUN_SOLN = np.array(
    [
        (-B0 + np.sqrt(B0**2 - 4 * A0 * C0)) / (2 * A0),
        (-B1 + np.sqrt(B1**2 - 4 * A1 * C1)) / (2 * A1),
    ]
)


@jit
def scalar_fun(x):
    """Simple convex function for testing scalar minimization.

    Gradient is 2 uncoupled quadratic equations.
    """
    return (
        A0 / 2 * x[0] ** 2
        + A1 / 2 * x[1] ** 2
        + C0 * jnp.log(x[0] + B0 / A0)
        + C1 * jnp.log(x[1] + B1 / A1)
    )


scalar_grad = jit(Derivative(scalar_fun, mode="grad"))
scalar_hess = jit(Derivative(scalar_fun, mode="hess"))


class TestFmin:
    """Tests for scalar minimization routine."""

    @pytest.mark.unit
    def test_convex_full_hess_dogleg(self):
        """Test minimizing convex test function using dogleg method."""
        x0 = np.ones(2)

        out = fmintr(
            scalar_fun,
            x0,
            scalar_grad,
            scalar_hess,
            verbose=3,
            method="dogleg",
            x_scale="hess",
            ftol=0,
            xtol=0,
            gtol=1e-12,
            options={"ga_accept_threshold": 0},
        )
        np.testing.assert_allclose(out["x"], SCALAR_FUN_SOLN, atol=1e-8)

    @pytest.mark.unit
    def test_convex_full_hess_subspace(self):
        """Test minimizing rosenbrock function using subspace method with full hess."""
        x0 = np.ones(2)

        out = fmintr(
            scalar_fun,
            x0,
            scalar_grad,
            scalar_hess,
            verbose=1,
            method="subspace",
            x_scale="hess",
            ftol=0,
            xtol=0,
            gtol=1e-12,
            options={"ga_accept_threshold": 1},
        )
        np.testing.assert_allclose(out["x"], SCALAR_FUN_SOLN, atol=1e-8)

    @pytest.mark.slow
    @pytest.mark.unit
    def test_rosenbrock_bfgs_dogleg(self):
        """Test minimizing rosenbrock function using dogleg method with BFGS hess."""
        rando = default_rng(seed=3)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmintr(
            rosen,
            x0,
            rosen_der,
            hess="bfgs",
            verbose=1,
            method="dogleg",
            x_scale=1,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"ga_accept_threshold": 0},
        )
        np.testing.assert_allclose(out["x"], true_x)

    @pytest.mark.slow
    @pytest.mark.unit
    def test_rosenbrock_bfgs_subspace(self):
        """Test minimizing rosenbrock function using subspace method with BFGS hess."""
        rando = default_rng(seed=4)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmintr(
            rosen,
            x0,
            rosen_der,
            hess=BFGS(),
            verbose=1,
            method="subspace",
            x_scale=1,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"ga_accept_threshold": 0},
        )
        np.testing.assert_allclose(out["x"], true_x)


class TestSGD:
    """Tests for stochastic optimizers."""

    @pytest.mark.unit
    def test_sgd_convex(self):
        """Test minimizing convex test function using stochastic gradient descent."""
        x0 = np.ones(2)

        out = sgd(
            scalar_fun,
            x0,
            scalar_grad,
            verbose=3,
            ftol=0,
            xtol=0,
            gtol=1e-12,
            maxiter=2000,
        )
        np.testing.assert_allclose(out["x"], SCALAR_FUN_SOLN, atol=1e-4, rtol=1e-4)


class TestLSQTR:
    """Tests for least squares optimizer."""

    @pytest.mark.unit
    def test_lsqtr_exact(self):
        """Test minimizing least squares test function using exact trust region.

        Uses both "svd" and "cholesky" methods for factorizing jacobian.
        """
        p = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
        x = np.linspace(-1, 1, 100)
        y = vector_fun(x, p)

        def res(p):
            return vector_fun(x, p) - y

        rando = default_rng(seed=0)
        p0 = p + 0.25 * (rando.random(p.size) - 0.5)

        jac = Derivative(res, 0, "fwd")

        out = lsqtr(
            res,
            p0,
            jac,
            verbose=0,
            x_scale=1,
            tr_method="cho",
            options={"initial_trust_radius": 0.15, "max_trust_radius": 0.25},
        )
        np.testing.assert_allclose(out["x"], p)

        out = lsqtr(
            res,
            p0,
            jac,
            verbose=0,
            x_scale=1,
            tr_method="svd",
            options={"initial_trust_radius": 0.15, "max_trust_radius": 0.25},
        )
        np.testing.assert_allclose(out["x"], p)


@pytest.mark.unit
def test_no_iterations():
    """Make sure giving the correct answer works correctly."""
    np.random.seed(0)
    A = np.random.random((20, 10))
    b = np.random.random(20)
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]

    vecfun = lambda x: A @ x - b
    vecjac = Derivative(vecfun)

    fun = lambda x: np.sum(vecfun(x) ** 2)
    grad = Derivative(fun, 0, mode="grad")
    hess = Derivative(fun, 0, mode="hess")

    out1 = fmintr(fun, x0, grad, hess)
    out2 = lsqtr(vecfun, x0, vecjac)

    np.testing.assert_allclose(x0, out1["x"])
    np.testing.assert_allclose(x0, out2["x"])


@pytest.mark.unit
@pytest.mark.slow
def test_overstepping():
    """Test that equilibrium is NOT updated when final function value is worse.

    Previously, the optimizer would reach a point where no decrease was possible but
    due to noisy gradients it would keep trying until dx < xtol. However, the final
    step that it tried would be different from the final step accepted, and the
    wrong one would be returned as the "optimal" result. This test is to prevent that
    from happening.
    """

    class DummyObjective(_Objective):

        name = "Dummy"
        _print_value_fmt = "Dummy: {:.3e}"
        _units = "(Foo)"

        def build(self, eq, *args, **kwargs):

            # objective = just shift x by a lil bit
            self._x0 = (
                np.concatenate(
                    [
                        eq.R_lmn,
                        eq.Z_lmn,
                        eq.L_lmn,
                        eq.p_l,
                        eq.i_l,
                        eq.c_l,
                        np.atleast_1d(eq.Psi),
                    ]
                )
                + 1e-6
            )
            self._dim_f = self._x0.size
            self._check_dimensions()
            self._set_dimensions(eq)
            self._set_derivatives()
            self._built = True

        def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi):
            x = jnp.concatenate(
                [R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, jnp.atleast_1d(Psi)]
            )
            return x - self._x0

    np.random.seed(0)
    objective = ObjectiveFunction(DummyObjective(), use_jit=False)
    # make gradient super noisy so it stalls
    objective.jac = lambda x: objective._jac(x) + 1e2 * (
        np.random.random((objective._dim_f, x.size)) - 0.5
    )

    eq = desc.examples.get("DSHAPE")

    n = 10
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > n + 1, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > n + 1, :
    ]
    constraints = (
        ForceBalance(),
        FixBoundaryR(modes=R_modes),
        FixBoundaryZ(modes=Z_modes),
        FixPressure(),
        FixIota(),
        FixPsi(),
    )
    optimizer = Optimizer("lsq-exact")
    eq1, history = eq.optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=50,
        verbose=3,
        gtol=-1,  # disable gradient stopping
        ftol=-1,  # disable function stopping
        xtol=1e-3,
        copy=True,
        options={
            "initial_trust_radius": 0.5,
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0},
        },
    )

    x0 = objective.x(eq)
    x1 = objective.x(eq1)
    # expect it to try more than 1 step
    assert len(history["alltr"]) > 1
    # but all steps increase cost so expect original x at the end
    np.testing.assert_allclose(x0, x1, rtol=1e-14, atol=1e-14)


@pytest.mark.unit
@pytest.mark.slow
def test_maxiter_1_and_0_solve():
    """Test that solves with maxiter 1 and 0 terminate correctly."""
    # correctly meaning they terminate, instead of looping infinitely
    constraints = (
        FixBoundaryR(fixed_boundary=True),
        FixBoundaryZ(fixed_boundary=True),
        FixPressure(),
        FixIota(),
        FixPsi(),
    )
    objectives = ForceBalance()
    obj = ObjectiveFunction(objectives)
    eq = desc.examples.get("SOLOVEV")
    for opt in ["lsq-exact", "dogleg-bfgs"]:
        eq, result = eq.solve(
            maxiter=1, constraints=constraints, objective=obj, optimizer=opt
        )
        assert result["nfev"] <= 2
    for opt in ["lsq-exact", "dogleg-bfgs"]:
        eq, result = eq.solve(
            maxiter=0, constraints=constraints, objective=obj, optimizer=opt
        )
        assert result["nfev"] <= 1


@pytest.mark.unit
@pytest.mark.slow
def test_scipy_fail_message():
    """Test that scipy fail message does not cause an error (see PR #434)."""
    constraints = (
        FixBoundaryR(fixed_boundary=True),
        FixBoundaryZ(fixed_boundary=True),
        FixPressure(),
        FixCurrent(),
        FixPsi(),
    )
    objectives = ForceBalance()
    obj = ObjectiveFunction(objectives)
    eq = Equilibrium()
    # should fail on maxiter, and should NOT throw an error
    for opt in ["scipy-trf"]:
        eq, result = eq.solve(
            maxiter=3,
            constraints=constraints,
            objective=obj,
            optimizer=opt,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        assert "Maximum number of iterations has been exceeded" in result["message"]
    objectives = Energy()
    obj = ObjectiveFunction(objectives)
    for opt in ["scipy-trust-exact"]:
        eq, result = eq.solve(
            maxiter=3,
            constraints=constraints,
            objective=obj,
            optimizer=opt,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        assert "Maximum number of iterations has been exceeded" in result["message"]


def test_not_implemented_error():
    """Test NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Optimizer("not-a-method")


@pytest.mark.unit
def test_wrappers():
    """Tests for using wrapped objectives."""
    eq = desc.examples.get("SOLOVEV")
    con = (
        FixBoundaryR(fixed_boundary=True),
        FixBoundaryZ(fixed_boundary=True),
        FixIota(),
        FixPressure(),
        FixPsi(),
    )
    con_nl = (ForceBalance(),)
    obj = ForceBalance()
    with pytest.raises(AssertionError):
        _ = LinearConstraintProjection(obj, con)
    with pytest.raises(ValueError):
        _ = LinearConstraintProjection(ObjectiveFunction(obj), con + con_nl)
    ob = LinearConstraintProjection(ObjectiveFunction(obj), con, eq=eq)
    assert ob.built

    con = (
        FixBoundaryR(fixed_boundary=True),
        FixBoundaryZ(fixed_boundary=True),
        FixIota(),
        FixPressure(),
        FixPsi(),
    )
    con_nl = (ForceBalance(),)
    obj = ForceBalance()
    with pytest.raises(AssertionError):
        _ = ProximalProjection(obj, con[0])
    with pytest.raises(AssertionError):
        _ = ProximalProjection(ObjectiveFunction(con[0]), con[1])
    with pytest.raises(ValueError):
        _ = ProximalProjection(ObjectiveFunction(con[0]), ObjectiveFunction(con[1]))
    with pytest.raises(ValueError):
        _ = ProximalProjection(
            ObjectiveFunction(con[0]), ObjectiveFunction(con + con_nl)
        )
    ob = ProximalProjection(ObjectiveFunction(con[0]), ObjectiveFunction(con_nl), eq=eq)
    assert ob.built


def test_all_optimizers():
    """Just tests that the optimizers run without error, eg tests for the wrappers."""
    eq = desc.examples.get("SOLOVEV")
    fobj = ObjectiveFunction(ForceBalance())
    eobj = ObjectiveFunction(Energy())
    fobj.build(eq)
    eobj.build(eq)
    constraints = (
        FixBoundaryR(fixed_boundary=True),
        FixBoundaryZ(fixed_boundary=True),
        FixIota(),
        FixPressure(),
        FixPsi(),
    )

    for opt in optimizers:
        print("TESTING ", opt)
        if optimizers[opt]["scalar"]:
            obj = eobj
        else:
            obj = fobj
        eq.solve(objective=obj, constraints=constraints, optimizer=opt, maxiter=5)
