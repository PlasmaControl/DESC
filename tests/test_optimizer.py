"""Tests for optimizers and Optimizer class."""

import numpy as np
import pytest
from numpy.random import default_rng
from scipy.optimize import (
    BFGS,
    NonlinearConstraint,
    least_squares,
    minimize,
    rosen,
    rosen_der,
)

import desc.examples
from desc.backend import jit, jnp
from desc.derivatives import Derivative
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.objectives import (
    AspectRatio,
    Energy,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixIota,
    FixParameter,
    FixPressure,
    FixPsi,
    ForceBalance,
    GenericObjective,
    MagneticWell,
    MeanCurvature,
    ObjectiveFunction,
    PlasmaVesselDistance,
    Volume,
    get_fixed_boundary_constraints,
)
from desc.objectives.objective_funs import _Objective
from desc.optimize import (
    LinearConstraintProjection,
    Optimizer,
    ProximalProjection,
    fmin_auglag,
    fmintr,
    lsq_auglag,
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
            x_scale="hess",
            ftol=0,
            xtol=0,
            gtol=1e-12,
            options={"tr_method": "dogleg"},
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
            verbose=3,
            x_scale="hess",
            ftol=0,
            xtol=0,
            gtol=1e-12,
            options={"tr_method": "subspace"},
        )
        np.testing.assert_allclose(out["x"], SCALAR_FUN_SOLN, atol=1e-8)

    @pytest.mark.unit
    def test_convex_full_hess_exact(self):
        """Test minimizing rosenbrock function using exact method with full hess."""
        x0 = np.ones(2)

        out = fmintr(
            scalar_fun,
            x0,
            scalar_grad,
            scalar_hess,
            verbose=3,
            x_scale="hess",
            ftol=0,
            xtol=0,
            gtol=1e-12,
            options={"tr_method": "exact"},
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
            verbose=3,
            x_scale="hess",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"tr_method": "dogleg"},
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
            verbose=3,
            x_scale=1,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"tr_method": "subspace"},
        )
        np.testing.assert_allclose(out["x"], true_x)

    @pytest.mark.slow
    @pytest.mark.unit
    def test_rosenbrock_bfgs_exact(self):
        """Test minimizing rosenbrock function using exact method with BFGS hess."""
        rando = default_rng(seed=4)

        x0 = rando.random(7)
        true_x = np.ones(7)
        out = fmintr(
            rosen,
            x0,
            rosen_der,
            hess=BFGS(),
            verbose=3,
            x_scale=1,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            options={"tr_method": "exact"},
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
            verbose=3,
            x_scale=1,
            options={
                "initial_trust_radius": 0.15,
                "max_trust_radius": 0.25,
                "tr_method": "cho",
            },
        )
        np.testing.assert_allclose(out["x"], p)

        out = lsqtr(
            res,
            p0,
            jac,
            verbose=3,
            x_scale=1,
            options={
                "initial_trust_radius": 0.15,
                "max_trust_radius": 0.25,
                "tr_method": "svd",
            },
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
@pytest.mark.optimize
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

        def build(self, *args, **kwargs):
            eq = self.things[0]
            # objective = just shift x by a lil bit
            self._x0 = {key: val + 1e-6 for key, val in eq.params_dict.items()}
            self._dim_f = sum(np.asarray(x).size for x in self._x0.values())
            super().build()

        def compute(self, params, constants=None):
            x = jnp.concatenate(
                [jnp.atleast_1d(params[arg] - self._x0[arg]) for arg in params]
            )
            return x

    eq = desc.examples.get("DSHAPE")

    np.random.seed(0)
    objective = ObjectiveFunction(DummyObjective(things=eq), use_jit=False)
    # make gradient super noisy so it stalls
    objective.jac_scaled = lambda x, *args: objective._jac_scaled(x) + 1e2 * (
        np.random.random((objective._dim_f, x.size)) - 0.5
    )

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
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixIota(eq=eq),
        FixPsi(eq=eq),
    )
    optimizer = Optimizer("proximal-lsq-exact")
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
@pytest.mark.solve
def test_maxiter_1_and_0_solve():
    """Test that solves with maxiter 1 and 0 terminate correctly."""
    # correctly meaning they terminate, instead of looping infinitely
    eq = desc.examples.get("SOLOVEV")
    constraints = (
        FixBoundaryR(eq=eq),
        FixBoundaryZ(eq=eq),
        FixPressure(eq=eq),
        FixIota(eq=eq),
        FixPsi(eq=eq),
    )
    objectives = ForceBalance(eq=eq)
    obj = ObjectiveFunction(objectives)
    for opt in ["lsq-exact", "fmintr-bfgs"]:
        eq, result = eq.solve(
            maxiter=1, constraints=constraints, objective=obj, optimizer=opt, verbose=3
        )
        assert result["nit"] == 1
    for opt in ["lsq-exact", "fmintr-bfgs"]:
        eq, result = eq.solve(
            maxiter=0, constraints=constraints, objective=obj, optimizer=opt, verbose=3
        )
        assert result["nit"] == 0


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.solve
def test_scipy_fail_message():
    """Test that scipy fail message does not cause an error (see PR #434)."""
    eq = Equilibrium(M=3)
    constraints = (
        FixBoundaryR(eq=eq),
        FixBoundaryZ(eq=eq),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )
    objectives = ForceBalance(eq=eq)
    obj = ObjectiveFunction(objectives)

    # should fail on maxiter, and should NOT throw an error
    for opt in ["scipy-trf"]:
        eq, result = eq.solve(
            maxiter=3,
            verbose=3,
            constraints=constraints,
            objective=obj,
            optimizer=opt,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        assert "Maximum number of iterations has been exceeded" in result["message"]
    eq.set_initial_guess()
    objectives = Energy(eq=eq)
    obj = ObjectiveFunction(objectives)
    for opt in ["scipy-trust-exact"]:
        eq, result = eq.solve(
            maxiter=3,
            verbose=3,
            constraints=constraints,
            objective=obj,
            optimizer=opt,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        assert "Maximum number of iterations has been exceeded" in result["message"]


@pytest.mark.unit
def test_not_implemented_error():
    """Test NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Optimizer("not-a-method")


@pytest.mark.unit
def test_wrappers():
    """Tests for using wrapped objectives."""
    eq = desc.examples.get("SOLOVEV")
    con = (
        FixBoundaryR(eq=eq),
        FixBoundaryZ(eq=eq),
        FixIota(eq=eq),
        FixPressure(eq=eq),
        FixPsi(eq=eq),
    )
    con_nl = (ForceBalance(eq=eq),)
    obj = ForceBalance(eq=eq)
    with pytest.raises(AssertionError):
        _ = LinearConstraintProjection(obj, con)
    with pytest.raises(ValueError):
        _ = LinearConstraintProjection(ObjectiveFunction(obj), con + con_nl)
    ob = LinearConstraintProjection(ObjectiveFunction(obj), con)
    ob.build()
    assert ob.built

    np.testing.assert_allclose(
        ob.compute_scaled(ob.x(eq)), obj.compute_scaled(*obj.xs(eq))
    )
    np.testing.assert_allclose(
        ob.compute_unscaled(ob.x(eq)), obj.compute_unscaled(*obj.xs(eq))
    )
    np.testing.assert_allclose(ob.target_scaled, obj.target / obj.normalization)
    np.testing.assert_allclose(ob.bounds_scaled[0], obj.target / obj.normalization)
    np.testing.assert_allclose(ob.bounds_scaled[1], obj.target / obj.normalization)
    np.testing.assert_allclose(ob.weights, obj.weight)

    con = (
        FixBoundaryR(eq=eq),
        FixBoundaryZ(eq=eq),
        FixIota(eq=eq),
        FixPressure(eq=eq),
        FixPsi(eq=eq),
    )
    con_nl = (ForceBalance(eq=eq),)
    obj = ForceBalance(eq=eq)
    with pytest.raises(AssertionError):
        _ = ProximalProjection(obj, con[0], eq=eq)
    with pytest.raises(AssertionError):
        _ = ProximalProjection(ObjectiveFunction(con[0]), con[1], eq=eq)
    with pytest.raises(ValueError):
        _ = ProximalProjection(
            ObjectiveFunction(con[0]), ObjectiveFunction(con[1]), eq=eq
        )
    with pytest.raises(ValueError):
        _ = ProximalProjection(
            ObjectiveFunction(con[0]), ObjectiveFunction(con + con_nl), eq=eq
        )
    ob = ProximalProjection(ObjectiveFunction(con[0]), ObjectiveFunction(con_nl), eq=eq)
    ob.build()
    assert ob.built

    np.testing.assert_allclose(
        ob.compute_scaled(ob.x(eq)), con[0].compute_scaled(*con[0].xs(eq))
    )
    np.testing.assert_allclose(
        ob.compute_unscaled(ob.x(eq)), con[0].compute_unscaled(*con[0].xs(eq))
    )
    np.testing.assert_allclose(ob.target_scaled, con[0].target / con[0].normalization)
    np.testing.assert_allclose(
        ob.bounds_scaled[0], con[0].target / con[0].normalization
    )
    np.testing.assert_allclose(
        ob.bounds_scaled[1], con[0].target / con[0].normalization
    )
    np.testing.assert_allclose(ob.weights, con[0].weight)


@pytest.mark.unit
@pytest.mark.slow
def test_all_optimizers():
    """Just tests that the optimizers run without error, eg tests for the wrappers."""
    eqf = desc.examples.get("SOLOVEV")
    eqe = eqf.copy()
    fobj = ObjectiveFunction(ForceBalance(eq=eqf))
    eobj = ObjectiveFunction(Energy(eq=eqe))
    fobj.build()
    eobj.build()
    econ = get_fixed_boundary_constraints(eq=eqe)
    fcon = get_fixed_boundary_constraints(eq=eqf)
    options = {"sgd": {"alpha": 1e-4}}

    for opt in optimizers:
        print("TESTING ", opt)
        if optimizers[opt]["scalar"]:
            obj = eobj
            eq = eqe
            con = econ
        else:
            obj = fobj
            eq = eqf
            con = fcon
        eq.solve(
            objective=obj,
            constraints=con,
            optimizer=opt,
            copy=True,
            verbose=3,
            maxiter=5,
            options=options.get(opt, None),
        )


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.optimize
def test_scipy_constrained_solve():
    """Tests that the scipy constrained optimizer does something.

    This isn't that great of a test, since trust-constr and SLSQP don't work well on
    badly scaled problems like ours. Also usually you'd need to run for way longer,
    since stopping them early might return a point worse than you started with...
    """
    eq = desc.examples.get("DSHAPE")
    # increase pressure so no longer in force balance
    eq.p_l *= 1.1

    constraints = (
        FixBoundaryR(eq=eq, modes=[0, 0, 0]),  # fix specified major axis position
        FixBoundaryZ(eq=eq),  # fix Z shape but not R
        FixPressure(eq=eq),  # fix pressure profile
        FixIota(eq=eq),  # fix rotational transform profile
        FixPsi(eq=eq),  # fix total toroidal magnetic flux
    )
    # some random constraints to keep the shape from getting wacky
    V = eq.compute("V")["V"]
    Vbounds = (0.95 * V, 1.05 * V)
    AR = eq.compute("R0/a")["R0/a"]
    ARbounds = (0.95 * AR, 1.05 * AR)
    H = eq.compute(
        "curvature_H_rho",
        grid=LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym),
    )["curvature_H_rho"]
    Hbounds = ((1 - 0.05 * np.sign(H)) * H, (1 + 0.05 * np.sign(H)) * abs(H))
    constraints += (
        Volume(eq=eq, bounds=Vbounds),
        AspectRatio(eq=eq, bounds=ARbounds),
        MeanCurvature(eq=eq, bounds=Hbounds),
    )
    obj = ObjectiveFunction(ForceBalance(eq=eq))
    eq2, result = eq.optimize(
        objective=obj,
        constraints=constraints,
        optimizer="scipy-trust-constr",
        maxiter=50,
        verbose=1,
        x_scale="auto",
        copy=True,
        options={
            "disp": 1,
            "verbose": 3,
            "initial_barrier_parameter": 1e-4,
        },
    )
    V2 = eq2.compute("V")["V"]
    AR2 = eq2.compute("R0/a")["R0/a"]
    H2 = eq2.compute(
        "curvature_H_rho",
        grid=LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym),
    )["curvature_H_rho"]

    assert ARbounds[0] < AR2 < ARbounds[1]
    assert Vbounds[0] < V2 < Vbounds[1]
    assert np.all(Hbounds[0] < H2)
    assert np.all(H2 < Hbounds[1])
    assert eq2.is_nested()


@pytest.mark.unit
@pytest.mark.solve
def test_solve_with_x_scale():
    """Make sure we can manually specify x_scale when solving/optimizing."""
    # basically just tests that it runs without error
    with pytest.warns(UserWarning, match="pressure profile is not an even"):
        eq = Equilibrium(L=2, M=2, N=2, pressure=np.array([1000, -2000, 1000]))
    scale = jnp.concatenate(
        [
            (abs(eq.R_basis.modes[:, :2]).sum(axis=1) + 1),
            (abs(eq.Z_basis.modes[:, :2]).sum(axis=1) + 1),
            (abs(eq.L_basis.modes[:, :2]).sum(axis=1) + 1),
            jnp.ones(
                eq.p_l.size
                + eq.c_l.size
                + eq.Ra_n.size
                + eq.Za_n.size
                + eq.Rb_lmn.size
                + eq.Zb_lmn.size
                + 1
            ),
        ]
    )
    eq.solve(x_scale=scale)
    assert eq.is_nested()


@pytest.mark.unit
def test_bounded_optimization():
    """Test that our bounded optimizers are as good as scipy."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
    x = np.linspace(-1, 1, 100)
    y = vector_fun(x, p)

    def fun(p):
        return vector_fun(x, p) - y

    rando = default_rng(seed=5)
    p0 = p + 0.25 * (rando.random(p.size) - 0.5)

    jac = Derivative(fun, 0, "fwd")

    def sfun(x):
        f = fun(x)
        return 1 / 2 * f.dot(f)

    def grad(x):
        f = fun(x)
        J = jac(x)
        return f.dot(J)

    def hess(x):
        J = jac(x)
        return J.T @ J

    bounds = (2, np.inf)
    p0 = np.clip(p0, *bounds)

    out1 = lsqtr(
        fun,
        p0,
        jac,
        bounds=bounds,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-8,
        verbose=3,
        x_scale=1,
        options={"tr_method": "svd"},
    )
    out2 = fmintr(
        sfun,
        p0,
        grad,
        hess,
        bounds=bounds,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-8,
        verbose=3,
        x_scale=1,
    )
    out3 = least_squares(
        fun,
        p0,
        jac,
        bounds=bounds,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-8,
        verbose=2,
        x_scale=1,
    )

    np.testing.assert_allclose(out1["x"], out3["x"], rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(out2["x"], out3["x"], rtol=1e-06, atol=1e-06)


@pytest.mark.unit
def test_auglag():
    """Test that our augmented lagrangian works as well as scipy for convex problems."""
    rng = default_rng(12)

    n = 15  # number of variables
    m = 10  # number of constraints
    p = 7  # number of primals
    Qs = []
    gs = []
    for i in range(m + p):
        A = 0.5 - rng.random((n, n))
        Qs.append(A.T @ A)
        gs.append(0.5 - rng.random(n))

    @jit
    def vecfun(x):
        y0 = x @ Qs[0] + gs[0]
        y1 = x @ Qs[1] + gs[1]
        y = jnp.concatenate([y0, y1**2])
        return y

    @jit
    def fun(x):
        y = vecfun(x)
        return 1 / 2 * jnp.dot(y, y)

    grad = jit(Derivative(fun, mode="grad"))
    hess = jit(Derivative(fun, mode="hess"))
    jac = jit(Derivative(vecfun, mode="fwd"))

    @jit
    def con(x):
        cs = []
        for i in range(m):
            cs.append(x @ Qs[p + i] @ x + gs[p + i] @ x)
        return jnp.array(cs)

    conjac = jit(Derivative(con, mode="fwd"))
    conhess = jit(Derivative(lambda x, v: v @ con(x), mode="hess"))

    constraint = NonlinearConstraint(con, -np.inf, 0, conjac, conhess)
    x0 = rng.random(n)

    out1 = fmin_auglag(
        fun,
        x0,
        grad,
        hess=hess,
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint,
        args=(),
        x_scale="auto",
        ftol=0,
        xtol=1e-6,
        gtol=1e-6,
        ctol=1e-6,
        verbose=3,
        maxiter=None,
        options={"initial_multipliers": "least_squares"},
    )
    print(out1["active_mask"])
    out2 = lsq_auglag(
        vecfun,
        x0,
        jac,
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint,
        args=(),
        x_scale="auto",
        ftol=0,
        xtol=1e-6,
        gtol=1e-6,
        ctol=1e-6,
        verbose=3,
        maxiter=None,
        options={"initial_multipliers": "least_squares", "tr_method": "cho"},
    )

    out3 = minimize(
        fun,
        x0,
        jac=grad,
        hess=hess,
        constraints=constraint,
        method="trust-constr",
        options={"verbose": 3, "maxiter": 1000},
    )
    out4 = fmin_auglag(
        fun,
        x0,
        grad,
        hess="bfgs",
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint,
        args=(),
        x_scale="auto",
        ftol=0,
        xtol=1e-6,
        gtol=1e-6,
        ctol=1e-6,
        verbose=3,
        maxiter=None,
        options={"initial_multipliers": "least_squares"},
    )

    np.testing.assert_allclose(out1["x"], out3["x"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out2["x"], out3["x"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out4["x"], out3["x"], rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.optimize
def test_constrained_AL_lsq():
    """Tests that the least squares augmented Lagrangian optimizer does something."""
    eq = desc.examples.get("SOLOVEV")

    constraints = (
        FixBoundaryR(eq=eq, modes=[0, 0, 0]),  # fix specified major axis position
        FixPressure(eq=eq),
        FixParameter(
            eq, "i_l", bounds=(eq.i_l * 0.9, eq.i_l * 1.1)
        ),  # linear inequality
        FixPsi(eq=eq, bounds=(eq.Psi * 0.99, eq.Psi * 1.01)),  # linear inequality
    )
    # some random constraints to keep the shape from getting wacky
    V = eq.compute("V")["V"]
    Vbounds = (0.95 * V, 1.05 * V)
    AR = eq.compute("R0/a")["R0/a"]
    ARbounds = (0.95 * AR, 1.05 * AR)
    constraints += (
        Volume(eq=eq, bounds=Vbounds),
        AspectRatio(eq=eq, bounds=ARbounds),
        MagneticWell(eq=eq, bounds=(0, jnp.inf)),
        ForceBalance(eq=eq, bounds=(-1e-3, 1e-3), normalize_target=False),
    )
    H = eq.compute(
        "curvature_H_rho",
        grid=LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym),
    )["curvature_H_rho"]
    obj = ObjectiveFunction(MeanCurvature(eq=eq, target=H))
    ctol = 1e-4
    eq2, result = eq.optimize(
        objective=obj,
        constraints=constraints,
        optimizer="lsq-auglag",
        maxiter=500,
        verbose=3,
        ctol=ctol,
        x_scale="auto",
        copy=True,
        options={},
    )
    V2 = eq2.compute("V")["V"]
    AR2 = eq2.compute("R0/a")["R0/a"]
    Dwell = constraints[-2].compute_scaled(*constraints[-2].xs(eq2))
    assert (ARbounds[0] - ctol) < AR2 < (ARbounds[1] + ctol)
    assert (Vbounds[0] - ctol) < V2 < (Vbounds[1] + ctol)
    assert (0.99 * eq.Psi - ctol) <= eq2.Psi <= (1.01 * eq.Psi + ctol)
    np.testing.assert_array_less((0.9 * eq.i_l - ctol), eq2.i_l)
    np.testing.assert_array_less(eq2.i_l, (1.1 * eq.i_l + ctol))
    assert eq2.is_nested()
    np.testing.assert_array_less(-Dwell, ctol)


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.optimize
def test_constrained_AL_scalar():
    """Tests that the augmented Lagrangian constrained optimizer does something."""
    eq = desc.examples.get("SOLOVEV")

    constraints = (
        FixBoundaryR(eq=eq, modes=[0, 0, 0]),  # fix specified major axis position
        FixBoundaryZ(eq=eq),  # fix Z shape but not R
        FixPressure(eq=eq),  # fix pressure profile
        FixIota(eq=eq),  # fix rotational transform profile
        FixPsi(eq=eq),  # fix total toroidal magnetic flux
    )
    V = eq.compute("V")["V"]
    AR = eq.compute("R0/a")["R0/a"]
    constraints += (
        Volume(eq=eq, target=V),
        AspectRatio(eq=eq, target=AR),
        MagneticWell(eq=eq, bounds=(0, jnp.inf)),
        ForceBalance(eq=eq, bounds=(-1e-3, 1e-3), normalize_target=False),
    )
    # Dummy objective to return 0, we just want a feasible solution.
    obj = ObjectiveFunction(GenericObjective("0", eq=eq))
    ctol = 1e-4
    eq2, result = eq.optimize(
        objective=obj,
        constraints=constraints,
        optimizer="fmin-auglag",
        maxiter=1000,
        verbose=3,
        ctol=ctol,
        x_scale="auto",
        copy=True,
        options={},
    )
    V2 = eq2.compute("V")["V"]
    AR2 = eq2.compute("R0/a")["R0/a"]
    Dwell = constraints[-2].compute_scaled(*constraints[-2].xs(eq2))
    np.testing.assert_allclose(AR, AR2, atol=ctol, rtol=ctol)
    np.testing.assert_allclose(V, V2, atol=ctol, rtol=ctol)
    assert eq2.is_nested()
    np.testing.assert_array_less(-Dwell, ctol)


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.optimize
def test_proximal_with_PlasmaVesselDistance():
    """Tests that the proximal projection works with fixed surface distance obj."""
    eq = desc.examples.get("SOLOVEV")

    constraints = (
        ForceBalance(eq=eq),
        FixPressure(eq=eq),  # fix pressure profile
        FixIota(eq=eq),  # fix rotational transform profile
        FixPsi(eq=eq),  # fix total toroidal magnetic flux
    )
    # circular surface
    a = 2
    R0 = 4
    surf = FourierRZToroidalSurface(
        R_lmn=[R0, a],
        Z_lmn=[0.0, -a],
        modes_R=np.array([[0, 0], [1, 0]]),
        modes_Z=np.array([[0, 0], [-1, 0]]),
        sym=True,
        NFP=eq.NFP,
    )

    grid = LinearGrid(M=eq.M, N=0, NFP=eq.NFP)
    obj = PlasmaVesselDistance(
        surface=surf, eq=eq, target=0.5, plasma_grid=grid, surface_fixed=True
    )
    objective = ObjectiveFunction((obj,))

    optimizer = Optimizer("proximal-lsq-exact")
    eq, result = optimizer.optimize(
        eq,
        objective,
        constraints,
        verbose=3,
        maxiter=3,
    )

    # make sure it also works if proximal is given multiple objects in things
    obj = PlasmaVesselDistance(
        surface=surf, eq=eq, target=0.5, plasma_grid=grid, surface_fixed=False
    )
    objective = ObjectiveFunction((obj,))
    (eq, surf), result = optimizer.optimize(
        (eq, surf),
        objective,
        constraints,
        verbose=3,
        maxiter=3,
    )
