"""Globalized and stochastic optimization drivers for DESC.

DESC's built-in optimizers (``lsq-exact``, ``proximal-lsq-exact``, ...) are local
trust-region methods: they converge to a local minimum near the initial guess.
Stage-two coil optimization in particular is strongly multimodal (coil-coil and
plasma-coil distance penalties, curvature bounds, discrete symmetries), so the
minimum found depends on the starting point. This module adds global search
strategies on top of DESC's local solvers:

* :func:`multistart` - random perturbations of an initial guess, each followed by a
  local solve.
* :func:`basin_hopping` - perturb the incumbent local minimum, re-solve locally,
  accept/reject with a Metropolis rule on ``log(cost)``, and adapt the step size.
* :func:`cma_es` - covariance matrix adaptation evolution strategy on the raw
  objective (population evaluated in a single vmapped call), followed by local
  polishing of the best candidates.
* :func:`annealed_noisy_gradient` - Adam with annealed injected noise (a stochastic
  gradient Langevin-type method) that can climb out of shallow basins, followed by
  local polishing of the best iterates.

All of them work through :class:`ProjectedProblem`, which builds the objective and
the linear-constraint projection once, so that the JIT-compiled residual and
Jacobian are reused by every local solve. Without that, each restart would pay the
objective build + compile cost again.

Only problems whose constraints are all linear (``FixCoilCurrent``,
``FixParameters``, ``FixSumCoilCurrent``, ...) are handled by
:class:`ProjectedProblem`. That covers the usual stage-two problem. For stage one
(``proximal-lsq-exact``, where the equilibrium is re-solved at every step) see
``stage1_global.py``, which uses the same perturbation machinery.
"""

import time
import warnings

import jax
import numpy as np

from desc.backend import jnp
from desc.objectives import ObjectiveFunction
from desc.objectives.objective_funs import _Objective
from desc.optimizable import OptimizableCollection
from desc.optimize import LinearConstraintProjection, Optimizer

# ---------------------------------------------------------------------------
# Perturbation scales
# ---------------------------------------------------------------------------

# parameter name -> attribute name of the basis holding the mode numbers. Mode
# numbers are rows (l, m, n) of ``basis.modes``.
_SPECTRAL = {
    "X_n": "X_basis",  # FourierXYZCurve
    "Y_n": "Y_basis",
    "Z_n": "Z_basis",  # FourierXYZCurve and FourierRZCurve
    "R_n": "R_basis",  # FourierRZCurve
    "r_n": "r_basis",  # FourierPlanarCurve
    "R_lmn": "R_basis",  # FourierRZToroidalSurface
    "Z_lmn": "Z_basis",
}
# for an Equilibrium only the boundary is perturbed; the interior has to be
# re-solved for force balance afterwards.
_EQ_SPECTRAL = {"Rb_lmn": "R_basis", "Zb_lmn": "Z_basis"}


def _mode_order(modes):
    """Max-norm order of each (l, m, n) mode."""
    return np.max(np.abs(np.atleast_2d(modes)), axis=1).astype(float)


def _leaf_scales(thing, length_scale, alpha, current_scale, angle_scale, overrides):
    """Per-parameter perturbation scales and mode orders for a single object."""
    from desc.equilibrium import Equilibrium

    scales, orders = {}, {}
    is_eq = isinstance(thing, Equilibrium)
    for name in thing.optimizable_params:
        val = np.atleast_1d(np.asarray(getattr(thing, name)))
        order = np.zeros(val.size)
        if name in overrides:
            ov = overrides[name]
            s = np.broadcast_to(ov(thing) if callable(ov) else ov, val.shape)
        elif is_eq and name in _EQ_SPECTRAL:
            modes = getattr(thing.surface, _EQ_SPECTRAL[name]).modes
            order = _mode_order(modes)
            s = length_scale * np.exp(-alpha * order)
        elif (not is_eq) and name in _SPECTRAL and hasattr(thing, _SPECTRAL[name]):
            modes = getattr(thing, _SPECTRAL[name]).modes
            order = _mode_order(modes)
            s = length_scale * np.exp(-alpha * order)
        elif name == "current":
            s = np.full(val.shape, current_scale)
        elif name == "center":  # FourierPlanarCoil
            s = np.full(val.shape, length_scale)
        elif name == "normal":  # FourierPlanarCoil (unit-ish vector)
            s = np.full(val.shape, angle_scale)
        else:  # rotmat, shift, profiles, Psi, interior eq coefficients, ...
            s = np.zeros(val.shape)
        scales[name] = np.array(s, dtype=float)
        orders[name] = order
    return scales, orders


def _tree_scales(thing, *args):
    if isinstance(thing, OptimizableCollection):
        out = [_tree_scales(t, *args) for t in thing]
        return [o[0] for o in out], [o[1] for o in out]
    return _leaf_scales(thing, *args)


def _mean_abs_current(thing):
    if isinstance(thing, OptimizableCollection):
        cur = [_mean_abs_current(t) for t in thing]
        cur = [c for c in cur if c > 0]
        return float(np.mean(cur)) if cur else 0.0
    if "current" in getattr(thing, "optimizable_params", []):
        return float(np.mean(np.abs(np.asarray(thing.current))))
    return 0.0


def perturbation_scales(
    things,
    length_scale,
    alpha=0.5,
    current_rel=0.1,
    angle_scale=0.1,
    overrides=None,
):
    """Build a vector of perturbation scales aligned with ``objective.x(*things)``.

    Fourier coefficients of curves/surfaces get ``length_scale * exp(-alpha * k)``,
    where ``k = max(|l|,|m|,|n|)`` is the mode order, so random perturbations are
    smooth geometric deformations instead of high-frequency noise. Coil currents get
    ``current_rel * mean(|I|)``, planar-coil normals ``angle_scale``. Parameters
    with no sensible default (``rotmat``, ``shift``, profiles, ...) get 0 and are
    not perturbed. Any parameter can be overridden by name via ``overrides`` (a
    float, an array, or a callable ``f(thing) -> array``).

    Returns
    -------
    scales, orders : ndarray
        Perturbation scale and mode order of every entry of the full state vector.
    """
    overrides = {} if overrides is None else overrides
    scale_list, order_list = [], []
    for thing in things:
        current_scale = current_rel * _mean_abs_current(thing)
        s, o = _tree_scales(
            thing, length_scale, alpha, current_scale, angle_scale, overrides
        )
        scale_list.append(np.asarray(thing.pack_params(s)))
        order_list.append(np.asarray(thing.pack_params(o)))
    return np.concatenate(scale_list), np.concatenate(order_list)


# ---------------------------------------------------------------------------
# Deflation
# ---------------------------------------------------------------------------


class ScaledDeflation(_Objective):
    """Sum-reduction deflation penalty with centers that can change without recompiling.

    For known states ``c_i`` (e.g. local minima already found) the penalty is::

        M(x) = sum_i a_i * d_i(x)^(-p),   d_i = rms((x - c_i) / s)

    where ``s`` are the perturbation scales (so distances are measured in the same
    mode-aware units as the random steps), and the residual is the hinge
    ``w * max(0, M(x) - r^(-p))``. It is zero outside balls of radius ``r`` around
    every center and grows without bound inside them, i.e. it enforces the deflation
    constraint ``||x - c_i|| >= r`` of Panici et al. (2026), eq. (2.5), in its
    sum-reduction form (their eq. 4.4, with zero shift), as a least-squares penalty.

    Centers, activity flags ``a_i``, weight ``w`` and radius ``r`` are stored as
    array attributes, i.e. as dynamic pytree leaves of the objective: changing them
    between local solves does not trigger recompilation. With all ``a_i = 0`` (or
    ``w = 0``) the term is exactly zero and the problem is the undeflated one.
    """

    _scalar = True
    _units = "(~)"
    _print_value_fmt = "Deflation penalty: "
    _coordinates = ""
    _static_attrs = _Objective._static_attrs + ["_n_slots", "_power", "_reduction"]

    def __init__(
        self, thing, scales, n_slots=8, power=2, reduction="sum", name="scaled deflation"
    ):
        self._n_slots = int(n_slots)
        self._power = float(power)
        # "sum": eq. (4.4) of the deflation paper; "max": only the nearest center
        # counts, so the excluded region stays a union of balls however many centers
        self._reduction = reduction
        s = np.asarray(scales, dtype=float)
        self._inv_scale = jnp.asarray(np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0))
        self._n_eff = jnp.asarray(float(max(np.count_nonzero(s), 1)))
        self._centers = jnp.zeros((self._n_slots, s.size))
        self._active = jnp.zeros(self._n_slots)
        self._weight = jnp.asarray(0.0)
        self._radius = jnp.asarray(1.0)
        super().__init__(
            things=thing, target=0, weight=1, normalize=False, normalize_target=False,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        self._dim_f = 1
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        x = self.things[0].pack_params(params)
        diff = (x[None, :] - self._centers) * self._inv_scale[None, :]
        d2 = jnp.sum(diff**2, axis=1) / self._n_eff + 1e-12
        Mi = self._active * d2 ** (-self._power / 2)
        M = jnp.max(Mi) if self._reduction == "max" else jnp.sum(Mi)
        excess = jnp.maximum(M - self._radius ** (-self._power), 0.0)
        return jnp.atleast_1d(self._weight * excess)


# ---------------------------------------------------------------------------
# Problem wrapper
# ---------------------------------------------------------------------------


class ProjectedProblem:
    """A DESC problem (objective + linear constraints) built and compiled once.

    Parameters
    ----------
    things : Optimizable or list
        Things being optimized (e.g. a CoilSet). Must match ``objective.things``.
    objective : ObjectiveFunction
        Objective to minimize, in least-squares form.
    constraints : tuple of linear _Objective
        Linear constraints (e.g. ``FixCoilCurrent``).
    local_method : str
        DESC optimizer used for local solves, e.g. ``"lsq-exact"``.
    local_kwargs : dict
        Default keyword arguments for local solves (``maxiter``, ``ftol``, ...).
    length_scale, alpha, current_rel, angle_scale, overrides
        Passed to :func:`perturbation_scales`.
    deflation_slots : int
        If > 0, a :class:`ScaledDeflation` term with this many centers is appended to
        the objective (inactive until :meth:`set_deflation` is called). Requires a
        single optimizable thing.
    free_curve_frame : bool
        DESC's ``Optimizer.optimize`` adds self-consistency constraints for every
        optimized thing; for curves these fix ``shift`` and ``rotmat``, which are
        redundant with the Fourier coefficients. The default (False) does the same.
        True leaves them free, which reproduces the first two rounds of stage-two
        runs (``results/stage2``); equilibria always get their constraints.
    """

    def __init__(
        self,
        things,
        objective,
        constraints,
        local_method="lsq-exact",
        local_kwargs=None,
        length_scale=0.05,
        alpha=0.5,
        current_rel=0.1,
        angle_scale=0.1,
        overrides=None,
        verbose=1,
        deflation_slots=0,
        deflation_reduction="sum",
        free_curve_frame=False,
    ):
        self.things = list(things) if isinstance(things, (list, tuple)) else [things]
        self.verbose = verbose
        t0 = time.time()
        # things in the order ObjectiveFunction uses (first appearance)
        obj_things = []
        for o in objective.objectives:
            for t in o.things:
                if not any(t is u for u in obj_things):
                    obj_things.append(t)
        self.scales, self.orders = perturbation_scales(
            obj_things,
            length_scale,
            alpha=alpha,
            current_rel=current_rel,
            angle_scale=angle_scale,
            overrides=overrides,
        )
        self.deflation = None
        if deflation_slots:
            if len(obj_things) != 1:
                raise ValueError("deflation needs a single optimizable thing")
            self.deflation = ScaledDeflation(
                obj_things[0], self.scales, deflation_slots, reduction=deflation_reduction
            )
            objective = ObjectiveFunction(
                tuple(objective.objectives) + (self.deflation,)
            )
        if not objective.built:
            objective.build(verbose=0)
        # Self-consistency constraints, as Optimizer.optimize adds them. Equilibria
        # always need them (boundary vs. interior coefficients, axis, lambda gauge),
        # or e.g. Rb_lmn would stay frozen while R_lmn changes. For curves they fix
        # the redundant ``shift`` and ``rotmat`` unless ``free_curve_frame``.
        from desc.equilibrium import Equilibrium
        from desc.objectives.getters import maybe_add_self_consistency

        constraints = tuple(constraints)
        for t in obj_things:
            if isinstance(t, Equilibrium) or not free_curve_frame:
                constraints = maybe_add_self_consistency(t, constraints)
        constraint = ObjectiveFunction(tuple(constraints))
        constraint.build(verbose=0)
        self.objective = objective
        if not all(a is b for a, b in zip(objective.things, obj_things)):
            raise RuntimeError("unexpected ordering of optimizable things")
        self.proj = LinearConstraintProjection(objective, constraint)
        self.proj.build(verbose=0)
        self.optimizer = Optimizer(local_method)
        self.local_kwargs = dict(maxiter=100, ftol=1e-4, xtol=1e-6, gtol=1e-8)
        self.local_kwargs.update({} if local_kwargs is None else local_kwargs)
        self._fun = lambda z: self.proj.compute_scaled_error(z, self.proj.constants)
        self._fun_batch = jax.jit(jax.vmap(self._fun))
        self._val_and_grad = jax.jit(
            jax.value_and_grad(lambda z: 0.5 * jnp.sum(self._fun(z) ** 2))
        )
        self.nfev = 0
        self.ngev = 0
        self.nlocal = 0
        self.t_start = time.time()
        if verbose:
            print(
                f"ProjectedProblem built in {time.time() - t0:.1f}s: "
                f"dim x = {self.x_full().size}, "
                f"dim z = {np.asarray(self.project(self.x_full())).size}, "
                f"dim f = {self.proj.dim_f}, "
                f"perturbed entries = {np.count_nonzero(self.scales)}"
            )

    # -- state handling ------------------------------------------------------
    def x_full(self):
        """Full parameter vector of the things in their current state."""
        return np.asarray(self.objective.x(*self.objective.things))

    def set_x_full(self, x):
        """Write a full parameter vector back into the things."""
        params = self.objective.unpack_state(jnp.asarray(x), per_objective=False)
        for t, p in zip(self.objective.things, params):
            t.params_dict = p

    def project(self, x):
        return self.proj.project(jnp.asarray(x))

    def recover(self, z):
        return np.asarray(self.proj.recover(jnp.asarray(z)))

    def feasible(self, x):
        """Orthogonally project a full vector onto the linear constraints."""
        return self.recover(self.project(x))

    # -- evaluations -----------------------------------------------------------
    def cost(self, x):
        """0.5 * ||f||^2 of the (projected) full vector x; inf if not finite."""
        self.nfev += 1
        c = float(0.5 * jnp.sum(self._fun(self.project(x)) ** 2))
        return c if np.isfinite(c) else np.inf

    def cost_batch(self, X):
        """Costs of a batch of full vectors, evaluated in one vmapped call."""
        Z = jax.vmap(self.proj.project)(jnp.asarray(X))
        self.nfev += len(X)
        c = np.asarray(0.5 * jnp.sum(self._fun_batch(Z) ** 2, axis=1))
        return np.where(np.isfinite(c), c, np.inf)

    def value_and_grad_z(self, z):
        self.nfev += 1
        self.ngev += 1
        c, g = self._val_and_grad(jnp.asarray(z))
        return float(c), np.asarray(g)

    def perturb(self, x, sigma, rng, max_order=None):
        """Random smooth perturbation ``x + sigma * scales * N(0, 1)``."""
        s = self.scales.copy()
        if max_order is not None:
            s[self.orders > max_order] = 0.0
        return x + sigma * s * rng.standard_normal(x.size)

    def distance(self, x, y):
        """RMS distance between full vectors in units of the perturbation scales."""
        s = self.scales
        m = s > 0
        return float(np.sqrt(np.mean(((np.asarray(x) - np.asarray(y))[m] / s[m]) ** 2)))

    # -- deflation -------------------------------------------------------------
    def set_deflation(self, centers, radius, weight):
        """Deflate around ``centers`` (full vectors): penalty inside rms-radius ``radius``.

        At most ``deflation_slots`` centers are used (the first ones given).
        """
        d = self.deflation
        if d is None:
            raise ValueError("problem was built without deflation_slots")
        k = min(len(centers), d._n_slots)
        C = np.zeros(d._centers.shape)
        a = np.zeros(d._n_slots)
        if k:
            C[:k] = np.asarray(centers[:k])
            a[:k] = 1.0
        d._centers = jnp.asarray(C)
        d._active = jnp.asarray(a)
        d._radius = jnp.asarray(float(radius))
        d._weight = jnp.asarray(float(weight))

    def clear_deflation(self):
        if self.deflation is not None:
            self.deflation._active = jnp.zeros(self.deflation._n_slots)
            self.deflation._weight = jnp.asarray(0.0)

    # -- local solve -------------------------------------------------------------
    def local_solve(self, x, **kwargs):
        """Run the local DESC optimizer from full vector ``x``.

        Returns a dict with the optimized full vector ``x``, its ``cost`` and solver
        statistics. Things are left in the optimized state.
        """
        opts = dict(self.local_kwargs)
        opts.update(kwargs)
        self.set_x_full(self.feasible(x))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, res = self.optimizer.optimize(
                    self.objective.things,
                    objective=self.proj,
                    verbose=0,
                    copy=False,
                    **opts,
                )
                xn = self.x_full()
                cost = self.cost(xn)
                nit, nfev, njev = res["nit"], res["nfev"], res.get("njev", 0)
            except Exception as e:  # e.g. NaNs from intersecting coils
                if self.verbose:
                    print(f"  local solve failed: {e!r}")
                xn, cost, nit, nfev, njev = np.asarray(x), np.inf, 0, 0, 0
        self.nlocal += 1
        self.nfev += nfev
        self.ngev += njev
        return dict(
            x=xn, cost=cost, nit=nit, nfev=nfev, njev=njev, time=time.time() - t0
        )

    def elapsed(self):
        return time.time() - self.t_start


class GlobalResult(dict):
    """Result of a global search: ``x``, ``cost``, ``history`` and ``trace``."""

    __getattr__ = dict.__getitem__


def _record(problem, trace, best_cost, label=""):
    trace.append(
        dict(
            time=problem.elapsed(),
            nlocal=problem.nlocal,
            nfev=problem.nfev,
            ngev=problem.ngev,
            best_cost=best_cost,
            label=label,
        )
    )


def _finish(problem, best_x, best_cost, history, trace, method, t0):
    problem.set_x_full(best_x)
    return GlobalResult(
        x=best_x,
        cost=best_cost,
        history=history,
        trace=trace,
        method=method,
        wall_time=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Global strategies
# ---------------------------------------------------------------------------


def multistart(
    problem,
    n_starts=10,
    sigma=1.0,
    x0=None,
    seed=0,
    include_x0=True,
    max_order=None,
    time_budget=np.inf,
    **local_kwargs,
):
    """Independent local solves from random perturbations of ``x0``.

    Parameters
    ----------
    problem : ProjectedProblem
    n_starts : int
        Number of local solves (including the unperturbed one if ``include_x0``).
    sigma : float
        Perturbation size in units of ``problem.scales``.
    x0 : ndarray, optional
        Initial guess (full vector). Defaults to the current state of the things.
    max_order : int, optional
        Only perturb modes with order <= max_order.
    time_budget : float
        Stop starting new local solves after this many seconds.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    x0 = problem.x_full() if x0 is None else np.asarray(x0)
    history, trace = [], []
    best_x, best_cost = x0, np.inf
    for i in range(n_starts):
        if time.time() - t0 > time_budget:
            break
        xs = x0 if (i == 0 and include_x0) else problem.perturb(x0, sigma, rng, max_order)
        res = problem.local_solve(xs, **local_kwargs)
        history.append(dict(start=i, cost=res["cost"], nit=res["nit"], x=res["x"]))
        if res["cost"] < best_cost:
            best_x, best_cost = res["x"], res["cost"]
        _record(problem, trace, best_cost, f"start {i}")
        if problem.verbose:
            print(
                f"[multistart] start {i:3d}: cost {res['cost']:.4e} "
                f"(nit {res['nit']:3d}, {res['time']:.0f}s)  best {best_cost:.4e}"
            )
    return _finish(problem, best_x, best_cost, history, trace, "multistart", t0)


def basin_hopping(
    problem,
    n_hops=20,
    sigma=1.0,
    x0=None,
    seed=0,
    temperature=0.0,
    adapt_interval=4,
    target_accept=0.4,
    adapt_factor=0.75,
    sigma_bounds=(0.05, 5.0),
    max_order=None,
    restart_after=None,
    time_budget=np.inf,
    **local_kwargs,
):
    """Basin hopping with adaptive step size.

    Each hop perturbs the current local minimum by ``sigma * scales * N(0,1)`` and
    re-solves locally. The new minimum is accepted if it is better, or with
    probability ``exp(-(log c_new - log c_cur) / temperature)`` otherwise
    (``temperature=0`` gives monotonic basin hopping). Working in ``log(cost)`` makes
    the temperature independent of the overall scale of the objective. Every
    ``adapt_interval`` hops the step size is increased if the acceptance rate is above
    ``target_accept`` and decreased otherwise. If ``restart_after`` consecutive hops
    fail to improve the best, the walker restarts from the best point.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    x0 = problem.x_full() if x0 is None else np.asarray(x0)
    history, trace = [], []
    res = problem.local_solve(x0, **local_kwargs)
    x_cur, c_cur = res["x"], res["cost"]
    best_x, best_cost = x_cur, c_cur
    history.append(dict(hop=0, cost=c_cur, accepted=True, sigma=sigma, x=x_cur))
    _record(problem, trace, best_cost, "hop 0")
    if problem.verbose:
        print(f"[basin hopping] hop   0: cost {c_cur:.4e}")
    n_acc, n_since_best = 0, 0
    for k in range(1, n_hops + 1):
        if time.time() - t0 > time_budget:
            break
        xs = problem.perturb(x_cur, sigma, rng, max_order)
        res = problem.local_solve(xs, **local_kwargs)
        c_new = res["cost"]
        if c_new < c_cur:
            accept = True
        elif temperature > 0 and np.isfinite(c_new):
            accept = rng.random() < np.exp(-(np.log(c_new) - np.log(c_cur)) / temperature)
        else:
            accept = False
        if accept:
            x_cur, c_cur = res["x"], c_new
            n_acc += 1
        if c_new < best_cost:
            best_x, best_cost = res["x"], c_new
            n_since_best = 0
        else:
            n_since_best += 1
        history.append(dict(hop=k, cost=c_new, accepted=accept, sigma=sigma, x=res["x"]))
        _record(problem, trace, best_cost, f"hop {k}")
        if problem.verbose:
            print(
                f"[basin hopping] hop {k:3d}: sigma {sigma:.3f} cost {c_new:.4e} "
                f"{'ACC' if accept else 'rej'}  current {c_cur:.4e} best {best_cost:.4e}"
            )
        if adapt_interval and k % adapt_interval == 0:
            rate = n_acc / adapt_interval
            sigma = sigma / adapt_factor if rate > target_accept else sigma * adapt_factor
            sigma = float(np.clip(sigma, *sigma_bounds))
            n_acc = 0
        if restart_after and n_since_best >= restart_after:
            x_cur, c_cur = best_x, best_cost
            n_since_best = 0
    return _finish(problem, best_x, best_cost, history, trace, "basin_hopping", t0)


def pool_search(
    problem,
    x0=None,
    seed=0,
    n_hops=10**6,
    pool_size=4,
    sigma=None,
    sigma_bounds=(0.05, 3.0),
    adapt_interval=4,
    target_success=0.3,
    adapt_factor=0.8,
    rank_temperature=1.0,
    deflate=True,
    defl_radius_rel=3.0,
    defl_strength=1.0,
    phase1_iters=20,
    race_factor=3.0,
    dup_radius=0.05,
    time_budget=np.inf,
    **local_kwargs,
):
    """Deflated elite-pool basin hopping with racing.

    Improves on :func:`basin_hopping` in four ways, motivated by the stage-two runs
    (a single walker's result varied a lot between seeds, and most hops landed in much
    worse basins) and by the deflation paper (Panici et al. 2026):

    1. **Elite pool.** The ``pool_size`` best *distinct* minima are kept, and each hop
       starts from a member chosen with probability ``~ exp(-rank / rank_temperature)``.
       Several good basins are explored at once instead of one walker's.
    2. **Deflated escape.** With ``deflate=True`` the kick is small and the first
       ``phase1_iters`` iterations of the local solve are deflated around the parent
       and the other pool members (:class:`ScaledDeflation`, radius
       ``defl_radius_rel * sigma``). Deflation, not a large random jump, pushes the
       iterate out of known basins, so it lands in a *neighboring* basin.
    3. **Racing.** After phase 1, a hop whose (undeflated) cost exceeds
       ``race_factor`` times the worst pool member is abandoned, saving the rest of
       the solve.
    4. **Polish.** Surviving hops are polished without deflation, so every pool member
       is a true local minimum of the original problem. Results closer than
       ``dup_radius`` (rms, in perturbation-scale units) to a pool member are
       treated as the same minimum.

    The kick size ``sigma`` adapts every ``adapt_interval`` hops toward a pool-entry
    rate of ``target_success``.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if deflate and problem.deflation is None:
        raise ValueError("deflate=True needs ProjectedProblem(..., deflation_slots>0)")
    sigma = (0.3 if deflate else 1.0) if sigma is None else sigma
    x0 = problem.x_full() if x0 is None else np.asarray(x0)
    history, trace = [], []
    problem.clear_deflation()
    res = problem.local_solve(x0, **local_kwargs)
    pool = [(res["cost"], res["x"])]
    recent = []  # recently found minima, also deflated
    history.append(dict(hop=0, cost=res["cost"], x=res["x"], kind="start"))
    _record(problem, trace, res["cost"], "hop 0")
    if problem.verbose:
        print(f"[pool] hop   0: cost {res['cost']:.4e}")
    n_succ = 0
    stats = dict(abandoned=0, duplicates=0, inserted=0)
    maxiter = dict(problem.local_kwargs, **local_kwargs).get("maxiter", 100)
    for k in range(1, n_hops + 1):
        if time.time() - t0 > time_budget:
            break
        pool.sort(key=lambda p: p[0])
        ranks = np.arange(len(pool))
        w = np.exp(-ranks / rank_temperature)
        ip = rng.choice(len(pool), p=w / w.sum())
        c_par, x_par = pool[ip]
        xs = problem.perturb(x_par, sigma, rng)
        # phase 1: (deflated) partial solve
        if deflate:
            centers = [x_par] + [p[1] for i, p in enumerate(pool) if i != ip] + recent
            radius = defl_radius_rel * sigma
            weight = defl_strength * np.sqrt(2 * max(c_par, 1e-30)) * radius**2
            problem.set_deflation(centers, radius=radius, weight=weight)
        r1 = problem.local_solve(xs, **dict(local_kwargs, maxiter=phase1_iters))
        problem.clear_deflation()
        c1 = problem.cost(r1["x"]) if deflate else r1["cost"]
        c_ref = pool[-1][0] if len(pool) >= pool_size else max(p[0] for p in pool)
        entry = dict(hop=k, parent_rank=int(ip), sigma=sigma, phase1_cost=c1)
        if not np.isfinite(c1) or c1 > race_factor * c_ref:
            stats["abandoned"] += 1
            entry.update(kind="abandoned", cost=c1)
            history.append(entry)
            _record(problem, trace, pool[0][0], f"hop {k}")
            if problem.verbose:
                print(
                    f"[pool] hop {k:3d}: sigma {sigma:.3f} parent #{ip} "
                    f"phase-1 cost {c1:.3e} -> abandoned   best {pool[0][0]:.4e}"
                )
            succ = False
        else:
            r2 = problem.local_solve(r1["x"], **dict(local_kwargs, maxiter=maxiter))
            c, x = r2["cost"], r2["x"]
            dists = [problem.distance(x, p[1]) for p in pool]
            jmin = int(np.argmin(dists))
            succ, kind = False, "worse"
            if dists[jmin] < dup_radius:
                stats["duplicates"] += 1
                kind = "duplicate"
                if c < pool[jmin][0]:
                    pool[jmin] = (c, x)
                    succ = jmin == 0
            elif len(pool) < pool_size or c < pool[-1][0]:
                if len(pool) >= pool_size:
                    pool.pop()
                pool.append((c, x))
                stats["inserted"] += 1
                succ, kind = True, "inserted"
            recent = ([x] + recent)[: max(0, (problem.deflation._n_slots if deflate else 0) - pool_size)]
            pool.sort(key=lambda p: p[0])
            entry.update(
                kind=kind, cost=c, x=x, dist_parent=problem.distance(x, x_par),
                nit=r1["nit"] + r2["nit"],
            )
            history.append(entry)
            _record(problem, trace, pool[0][0], f"hop {k}")
            if problem.verbose:
                print(
                    f"[pool] hop {k:3d}: sigma {sigma:.3f} parent #{ip} ({c_par:.3e}) "
                    f"phase-1 {c1:.3e} -> {c:.4e} {kind:9s} d_parent "
                    f"{entry['dist_parent']:.2f}  best {pool[0][0]:.4e}"
                )
        n_succ += succ
        if adapt_interval and k % adapt_interval == 0:
            rate = n_succ / adapt_interval
            sigma = sigma / adapt_factor if rate > target_success else sigma * adapt_factor
            sigma = float(np.clip(sigma, *sigma_bounds))
            n_succ = 0
    best_cost, best_x = pool[0]
    res = _finish(problem, best_x, best_cost, history, trace, "pool_search", t0)
    res["stats"] = stats
    res["pool_costs"] = [float(p[0]) for p in pool]
    return res


def cma_es(
    problem,
    sigma0=0.5,
    x0=None,
    seed=0,
    popsize=None,
    max_evals=5000,
    max_order=None,
    n_polish=3,
    time_budget=np.inf,
    cma_options=None,
    **local_kwargs,
):
    """CMA-ES on the raw objective, then local polishing of the best candidates.

    The search variables are ``y`` with ``x = x0 + scales * y`` (restricted to entries
    with nonzero scale and, optionally, mode order <= ``max_order``), so the initial
    search distribution is the same smooth random perturbation used by the other
    methods. Each generation is evaluated in one vmapped call. Afterwards the best
    ``n_polish`` distinct points (best-ever, final mean, and best of the last
    generations) are refined by the local solver.

    Requires the ``cma`` package (``pip install cma``).
    """
    import cma

    t0 = time.time()
    x0 = problem.x_full() if x0 is None else np.asarray(x0)
    mask = problem.scales > 0
    if max_order is not None:
        mask &= problem.orders <= max_order
    idx = np.where(mask)[0]
    S = problem.scales[idx]

    def to_x(y):
        x = x0.copy()
        x[idx] += S * y
        return x

    opts = {"seed": seed + 1, "maxfevals": max_evals, "verbose": -9}
    if popsize is not None:
        opts["popsize"] = popsize
    opts.update({} if cma_options is None else cma_options)
    es = cma.CMAEvolutionStrategy(np.zeros(idx.size), sigma0, opts)
    history, trace = [], []
    best_y, best_cost = np.zeros(idx.size), problem.cost(x0)
    gen_bests = []
    gen = 0
    while not es.stop() and time.time() - t0 < time_budget:
        Y = es.ask()
        costs = problem.cost_batch(np.stack([to_x(y) for y in Y]))
        # CMA-ES is rank based, log keeps the numbers readable and inf handled
        es.tell(Y, [float(np.log(c)) if np.isfinite(c) else 1e300 for c in costs])
        i = int(np.argmin(costs))
        gen_bests.append((costs[i], np.array(Y[i])))
        if costs[i] < best_cost:
            best_cost, best_y = costs[i], np.array(Y[i])
        gen += 1
        if gen % 10 == 0:
            _record(problem, trace, best_cost, f"gen {gen}")
        if problem.verbose and gen % 25 == 0:
            print(
                f"[cma-es] gen {gen:4d} evals {es.countevals:6d}: "
                f"gen best {costs[i]:.4e}  best {best_cost:.4e}  sigma {es.sigma:.3f}"
            )
    history.append(dict(stage="cma", cost=best_cost, evals=es.countevals, gens=gen))
    if problem.verbose:
        print(f"[cma-es] finished after {es.countevals} evals: best raw {best_cost:.4e}")

    # candidates for polishing: best ever, final mean, best of recent generations
    cands = [best_y, np.asarray(es.mean)]
    for c, y in sorted(gen_bests[-50:], key=lambda t: t[0]):
        if len(cands) >= n_polish:
            break
        if all(np.linalg.norm(y - yc) > 1e-3 * (1 + np.linalg.norm(yc)) for yc in cands):
            cands.append(y)
    best_x, best_polished = to_x(best_y), np.inf
    for j, y in enumerate(cands[:n_polish]):
        res = problem.local_solve(to_x(y), **local_kwargs)
        history.append(dict(stage="polish", cand=j, cost=res["cost"], x=res["x"]))
        if res["cost"] < best_polished:
            best_x, best_polished = res["x"], res["cost"]
        _record(problem, trace, min(best_polished, best_cost), f"polish {j}")
        if problem.verbose:
            print(f"[cma-es] polish {j}: cost {res['cost']:.4e}")
    if best_cost < best_polished:  # raw CMA point better than all polished ones
        best_x, best_polished = to_x(best_y), best_cost
    return _finish(problem, best_x, best_polished, history, trace, "cma_es", t0)


def annealed_noisy_gradient(
    problem,
    n_cycles=1000,
    steps_per_cycle=200,
    lr=0.02,
    temperature0=0.3,
    x0=None,
    seed=0,
    beta1=0.9,
    beta2=0.999,
    time_budget=np.inf,
    **local_kwargs,
):
    """Langevin-driven basin hopping: annealed noisy-Adam excursions + local polish.

    Each cycle starts from the incumbent local minimum and runs ``steps_per_cycle``
    Adam steps in the (constraint-projected) variables, scaled by
    ``problem.scales``. Every step adds smooth Langevin noise of amplitude
    ``sqrt(2 * lr * T_k)`` (in units of ``problem.scales``). The temperature is
    annealed as ``T_k = temperature0 * (1 - k / K)^2``, so it reaches zero at the end
    of the cycle. Early in the cycle the noise lets the iterate cross low barriers,
    while the gradient keeps it in low-cost regions. The excursion therefore follows
    the landscape instead of being an isotropic jump as in :func:`basin_hopping`. The
    end point is polished with the local solver and accepted if it improves on the
    incumbent (a stochastic-gradient Langevin / simulated-annealing hybrid).

    Pass a local minimum as ``x0`` (e.g. the result of ``problem.local_solve``): a
    first-order method is slow on the initial large descent, which the local
    least-squares solver does much faster.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    x_inc = problem.x_full() if x0 is None else np.asarray(x0)
    c_inc = problem.cost(x_inc)
    history, trace = [], []
    _record(problem, trace, c_inc, "start")
    if problem.verbose:
        print(f"[noisy-grad] start: cost {c_inc:.4e}")
    for cyc in range(1, n_cycles + 1):
        if time.time() - t0 > time_budget:
            break
        z = np.asarray(problem.project(x_inc))
        # per-coordinate scale in z: project the scale vector itself
        zs = np.abs(np.asarray(problem.project(x_inc + problem.scales)) - z)
        zs = np.where(zs > 0, zs, np.median(zs[zs > 0]) if np.any(zs > 0) else 1.0)
        m = np.zeros_like(z)
        v = np.zeros_like(z)
        c = c_inc
        for k in range(steps_per_cycle):
            c, g = problem.value_and_grad_z(z)
            if not np.isfinite(c):
                break
            gs = g * zs  # gradient w.r.t. the scaled variables z / zs
            m = beta1 * m + (1 - beta1) * gs
            v = beta2 * v + (1 - beta2) * gs**2
            mh = m / (1 - beta1 ** (k + 1))
            vh = v / (1 - beta2 ** (k + 1))
            z = z - lr * zs * mh / (np.sqrt(vh) + 1e-12)
            T = temperature0 * (1 - k / steps_per_cycle) ** 2
            if T > 0:
                x_cur = problem.recover(z)
                xn = problem.perturb(x_cur, np.sqrt(2 * lr * T), rng)
                z = np.asarray(problem.project(xn))
        if not np.isfinite(c):
            history.append(dict(cycle=cyc, raw=np.inf, cost=np.inf, accepted=False))
            continue
        res = problem.local_solve(problem.recover(z), **local_kwargs)
        accept = res["cost"] < c_inc
        if accept:
            x_inc, c_inc = res["x"], res["cost"]
        history.append(
            dict(cycle=cyc, raw=float(c), cost=res["cost"], accepted=accept, x=res["x"])
        )
        _record(problem, trace, c_inc, f"cycle {cyc}")
        if problem.verbose:
            print(
                f"[noisy-grad] cycle {cyc:3d}: end of excursion {c:.4e} -> polished "
                f"{res['cost']:.4e} {'ACC' if accept else 'rej'}  best {c_inc:.4e}"
            )
    return _finish(problem, x_inc, c_inc, history, trace, "annealed_noisy_gradient", t0)
