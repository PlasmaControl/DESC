import numpy as np
import pytest

from desc.backend import jnp
from scipy.optimize import BFGS
from desc.optimize import fmintr, lsqtr, Optimizer
from desc.optimize.utils import make_spd, chol_U_update
from scipy.optimize import rosen, rosen_der, rosen_hess
from desc.derivatives import Derivative
from numpy.random import default_rng
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    FixBoundaryR,
    FixBoundaryZ,
    FixPressure,
    FixIota,
    FixPsi,
)
from desc.objectives.objective_funs import _Objective
import desc.examples


def vector_fun(x, p):
    """Complicated-ish vector valued function for testing least squares."""
    a0 = x * p[0]
    a1 = jnp.exp(-(x ** 2) * p[1])
    a2 = jnp.cos(jnp.sin(x * p[2] - x ** 2 * p[3]))
    a3 = jnp.sum(
        jnp.array([(x + 2) ** -(i * 2) * pi ** (i + 1) for i, pi in enumerate(p[3:])]),
        axis=0,
    )
    return a0 + a1 + 3 * a2 + a3


def scalar_fun(x):
    """Simple convex function for testing scalar minimization."""
    return x[0] ** 2 + x[1] ** 2 - jnp.log(x[0] + 2) + 10 - jnp.log(x[1] + 2)


scalar_grad = Derivative(scalar_fun, mode="grad")
scalar_hess = Derivative(scalar_fun, mode="hess")


class TestFmin:
    """Tests for scalar minimization routine."""
    
    @pytest.mark.unit
    def test_convex_full_hess_dogleg(self):
        """Test minimizing convex test function using dogleg method."""
        rando = default_rng(seed=2)

        x0 = 10 * rando.random(2)

        out = fmintr(
            scalar_fun,
            x0,
            scalar_grad,
            scalar_hess,
            verbose=1,
            method="dogleg",
            x_scale="hess",
            ftol=0,
            xtol=0,
            gtol=1e-12,
            options={"ga_accept_threshold": 0},
        )
        assert out["success"] is True
        np.testing.assert_allclose(scalar_grad(out["x"]), 0, atol=1e-12)

    @pytest.mark.unit
    def test_convex_full_hess_subspace(self):
        """Test minimizing convex test function using subspace method."""
        rando = default_rng(seed=2)

        x0 = rando.random(2)

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
        assert out["success"] is True
        np.testing.assert_allclose(scalar_grad(out["x"]), 0, atol=1e-12)

    @pytest.mark.slow
    @pytest.mark.unit
    def test_rosenbrock_bfgs_dogleg(self):
        """Test minimizing rosenbrock function using dogleg method with BFGS hessian."""
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
        """Test minimizing rosenbrock function using subspace method with BFGS hessian."""
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


class TestLSQTR:
    """Tests for least squares optimizer."""

    @pytest.mark.unit
    def test_lsqtr_exact(self):
        """Test minimizing least squares test function using svd and cholesky methods."""
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
    """Make sure giving the correct answer works correctly"""

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
    """Test that equilibrium is correctly NOT updated when final function value is worse.

    Previously, the optimizer would reach a point where no decrease was possible but
    due to noisy gradients it would keep trying until dx < xtol. However, the final
    step that it tried would be different from the final step accepted, and the
    wrong one would be returned as the "optimal" result. This test is to prevent that
    from happening.
    """

    class DummyObjective(_Objective):

        name = "Dummy"
        _print_value_fmt = "Dummy: {:.3e}"

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
