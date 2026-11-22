"""Global search for NAE-constrained equilibria (the DESC deflation tutorial problem).

Problem
-------
The first section of ``docs/notebooks/tutorials/deflation.ipynb``, which follows
Panici et al. (2026), "Deflation techniques for stellarator equilibrium and
optimization": global equilibria constrained to the axis and O(rho) behaviour of the
first-order precise-QA near-axis (pyQSC) solution, at the tutorial's resolution
(L = M = 5, N = 6, initial surfaces from the NAE at r = 0.25). Minimizing force
balance subject to the linear NAE constraints is under-constrained, so it has many
local minima: distinct global equilibria with the same near-axis behaviour. As in
the tutorial and the paper, each NAE-constrained solution is followed by a
conventional fixed-boundary solve to confirm that it is an equilibrium.

Methods
-------
deflation
    The tutorial's Algorithm 1: from the same initial guess every time, a deflated
    solve (DESC ``DeflationOperator`` wrapping force balance, M*f with p = 2,
    sigma = 500, deflating the boundary coefficients of every solution found so far),
    then an NAE-constrained solve, then a fixed-boundary solve. The tutorial's
    default product reduction overflows float64 after ~55 deflated states (each
    factor is >= sigma = 500), after which every iteration returns the first
    solution again.
deflation_sum
    The same with the paper's sum-reduction operator, eq. (4.4)
    (``multiple_deflation_type="sum"``, single shift), which does not overflow.
multistart
    Random smooth perturbations of the initial guess (outer-shape Zernike modes only,
    so the near-axis behaviour is untouched), then the NAE-constrained solve and the
    fixed-boundary check.
defl_multistart
    As multistart, but the first 15 iterations are deflated around every NAE
    solution found so far (:class:`desc_global.ScaledDeflation`, nearest-center
    form), so a start cannot fall back into a known solution.
qs_pool
    Deflated pool search on a physics figure of merit: parents are drawn from the
    five valid solutions with the lowest quasisymmetry error (Boozer f_B averaged over
    rho = 0.5, 0.75, 1), perturbed slightly, deflated away from every known solution
    and re-solved. It looks for *better* NAE-constrained equilibria, not just more.

Every method uses the same NAE-solve settings as the tutorial (ftol 1e-3, xtol 1e-7,
50 iterations), the same fixed-boundary check and the same metrics. Results are
written after every solution to ``results/nae/<method>_seed<seed>.json`` (+ ``.npz``
with the boundary coefficients).
"""

import argparse
import json
import os
import sys
import time
import warnings

import jax

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/jax-cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np  # noqa: E402
from qsc import Qsc  # noqa: E402

from desc.compute import get_transforms  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402
from desc.objectives import (  # noqa: E402
    DeflationOperator,
    ForceBalance,
    ObjectiveFunction,
    get_NAE_constraints,
)
from desc.vmec_utils import ptolemy_linear_transform  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from desc_global import ProjectedProblem  # noqa: E402

OUTDIR = os.path.join(HERE, "results", "nae")
NAE_SOLVE = dict(ftol=1e-3, xtol=1e-7, maxiter=50)  # as in the tutorial
FB_SOLVE = dict(ftol=1e-3, maxiter=50)
RHO = np.array([0.25, 0.5, 0.75, 1.0])


def setup():
    """The tutorial's near-axis solution and the global equilibrium fitted to it.

    Returns the first-order precise-QA ``Qsc`` object and an ``Equilibrium`` built
    from it at r = 0.25 (L = M = 5, N = 6), the initial guess of every method.
    """
    qsc_eq = Qsc.from_paper("precise QA", order=1)
    eq_fit = Equilibrium.from_near_axis(qsc_eq, r=0.25, L=5, M=5, N=6, ntheta=12)
    return qsc_eq, eq_fit


def shape_scales(ell, alpha):
    """Perturb only Zernike modes that vanish to O(rho^2) at the axis.

    Modes with l = |m| >= 2 (rho^|m| cos/sin(m theta) shapes) and the rho^2 (l=2, m=0)
    mode, scaled ``ell * exp(-alpha * max(|m|, |n|))``. Perturbing all interior
    modes frequently produces non-nested surfaces that the local solve cannot fix.
    """

    def f(e, name):
        """Scales of ``e``'s R_lmn or Z_lmn: nonzero only for the shape modes."""
        modes = getattr(e, name[0] + "_basis").modes
        l, m, n = np.abs(modes).T
        s = ell * np.exp(-alpha * np.maximum(m, n))
        return np.where(((l == m) & (m >= 2)) | ((l == 2) & (m == 0)), s, 0.0)

    return {"R_lmn": lambda e: f(e, "R_lmn"), "Z_lmn": lambda e: f(e, "Z_lmn")}


def force_error(eq):
    """Volume-averaged force error, normalized by the magnetic pressure gradient."""
    g = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    d = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=g)
    return float(d["<|F|>_vol"] / d["<|grad(|B|^2)|/2mu0>_vol"])


def metrics(eq):
    """Force error, iota, aspect ratio, Boozer QS error f_B(rho), magnetic well."""
    out = dict(force=force_error(eq))
    g = LinearGrid(rho=np.array([0.05, 1.0]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    d = eq.compute(["iota", "R0/a", "magnetic well"], grid=g)
    io = np.asarray(d["iota"]).reshape(2, -1).mean(axis=1)
    well = np.asarray(d["magnetic well"]).reshape(2, -1).mean(axis=1)
    out.update(iota_axis=float(io[0]), iota_edge=float(io[1]),
               aspect=float(d["R0/a"]), well_edge=float(well[1]))
    grid = LinearGrid(M=2 * eq.M_grid, N=2 * eq.N_grid, NFP=eq.NFP, rho=RHO)
    M_booz, N_booz = 2 * eq.M, 2 * eq.N
    tr = get_transforms("f_B_normalized", obj=eq, grid=grid, M_booz=M_booz, N_booz=N_booz)
    matrix, _, idx = ptolemy_linear_transform(
        tr["B"].basis.modes, helicity=(1, 0), NFP=tr["B"].basis.NFP
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dq = eq.compute(["f_B_normalized"], grid=grid, M_booz=M_booz, N_booz=N_booz,
                        helicity=(1, 0), matrix=matrix, idx=idx)
    fB = np.linalg.norm(np.asarray(dq["f_B_normalized"]), axis=-1)
    out["f_B"] = [float(v) for v in fB]
    out["qs"] = float(np.mean(fB[1:]))  # outer half: rho = 0.5, 0.75, 1
    return out


class Recorder:
    """Collects solutions and checkpoints them to disk after each one."""

    def __init__(self, method, seed, args):
        """Start the clock for run ``<method>_seed<seed>``; ``args`` are saved too."""
        self.t0 = time.time()
        self.tag = f"{method}_seed{seed}"
        self.sols, self.bdry = [], []
        self.args = args

    def add(self, eq_nae, eq_fb, label, extra=None):
        """Record one solution: metrics of the fixed-boundary re-solve ``eq_fb``.

        A solution is valid if both the NAE-constrained solve and the
        fixed-boundary solve have force error below 1 % and finite QS error.
        Returns the record (dict) after checkpointing everything to disk.
        """
        rec = dict(time=time.time() - self.t0, label=label, force_nae=force_error(eq_nae))
        rec.update(metrics(eq_fb))
        rec["valid"] = bool(rec["force_nae"] < 1e-2 and rec["force"] < 1e-2
                            and np.isfinite(rec["qs"]))
        rec.update(extra or {})
        self.sols.append(rec)
        self.bdry.append(np.r_[eq_fb.Rb_lmn, eq_fb.Zb_lmn])
        # distance to the nearest earlier solution (boundary coefficients, L2)
        if len(self.bdry) > 1:
            B = np.array(self.bdry[:-1])
            rec["nearest"] = float(np.min(np.linalg.norm(B - self.bdry[-1], axis=1)))
        print(f"[{rec['time']:6.0f}s] {label}: force {rec['force_nae']:.1e}/{rec['force']:.1e} "
              f"iota {rec['iota_edge']:.4f} A {rec['aspect']:.3f} f_B(1) {rec['f_B'][-1]:.2e} "
              f"qs {rec['qs']:.2e} well {rec['well_edge']:.2e} "
              f"nearest {rec.get('nearest', np.nan):.4f} {'ok' if rec['valid'] else 'INVALID'}",
              flush=True)
        self.save()
        return rec

    def save(self):
        """Write all records (JSON) and boundary coefficients (NPZ)."""
        os.makedirs(OUTDIR, exist_ok=True)
        with open(os.path.join(OUTDIR, self.tag + ".json"), "w") as f:
            json.dump(dict(args=vars(self.args), wall_time=time.time() - self.t0,
                           solutions=self.sols), f, indent=1)
        np.savez(os.path.join(OUTDIR, self.tag + ".npz"), bdry=np.array(self.bdry))


def fixed_boundary(eq):
    """Copy of ``eq`` re-solved as a conventional fixed-boundary equilibrium."""
    eqf = eq.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eqf.solve(objective=ObjectiveFunction(ForceBalance(eqf)), verbose=0, **FB_SOLVE)
    return eqf


def run_deflation(args, qsc_eq, eq_fit, rec, reduction="prod"):
    """The tutorial's Algorithm 1 (``reduction="sum"``: eq. 4.4 of the paper)."""
    eqs = [None] * 400  # padded so the deflation objective is not recompiled
    i = 0
    while time.time() - rec.t0 < args.budget:
        eq = eq_fit.copy()
        cons = get_NAE_constraints(eq, qsc_eq, order=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if i > 0:
                dobj = ObjectiveFunction(DeflationOperator(
                    thing=eq, things_to_deflate=eqs, objective=ForceBalance(eq),
                    power=2, sigma=500, params_to_deflate_with={"Rb_lmn": True, "Zb_lmn": True},
                    multiple_deflation_type=reduction, single_shift=reduction == "sum",
                ))
                eq.solve(objective=dobj, constraints=cons, verbose=0, ftol=1e-3,
                         maxiter=15, xtol=1e-6)
            eq.solve(objective=ObjectiveFunction(ForceBalance(eq)), constraints=cons,
                     verbose=0, **NAE_SOLVE)
        eqf = fixed_boundary(eq)
        rec.add(eq, eqf, f"deflation {i}")
        eqs[i] = eqf.copy()
        i += 1


def make_problem(args, qsc_eq, eq_fit, slots):
    """A :class:`ProjectedProblem` for force balance under the NAE constraints.

    Only the shape modes of :func:`shape_scales` are perturbed. With ``slots > 0``
    it carries a nearest-center (``"max"``) deflation term with that many centers.
    Returns the equilibrium being optimized and the problem.
    """
    eq = eq_fit.copy()
    p = ProjectedProblem(
        eq, ObjectiveFunction(ForceBalance(eq)), get_NAE_constraints(eq, qsc_eq, order=1),
        local_kwargs=NAE_SOLVE, length_scale=0.0,
        overrides=shape_scales(args.length_scale, 0.5),
        deflation_slots=slots, deflation_reduction="max", verbose=1,
    )
    return eq, p


def run_multistart(args, qsc_eq, eq_fit, rec, deflate):
    """Multistart from the initial guess (``deflate``: first 15 iterations deflated)."""
    rng = np.random.default_rng(args.seed)
    eq, p = make_problem(args, qsc_eq, eq_fit, 400 if deflate else 0)
    x0 = p.x_full()
    centers = []
    i = 0
    while time.time() - rec.t0 < args.budget:
        xs = x0 if i == 0 else p.perturb(x0, args.sigma, rng)
        if deflate and centers:
            p.set_deflation(centers[::-1], radius=args.defl_radius, weight=1.0)
            xs = p.local_solve(xs, maxiter=15, xtol=1e-6)["x"]
            p.clear_deflation()
        r = p.local_solve(xs)
        centers.append(r["x"])
        p.set_x_full(r["x"])
        rec.add(eq, fixed_boundary(eq), f"start {i}")
        i += 1


def run_qs_pool(args, qsc_eq, eq_fit, rec, pool_size=5):
    """QS-guided deflated pool search.

    Parents are the ``pool_size`` valid solutions with the lowest QS error, chosen
    with weights ``exp(-rank)``. Each hop perturbs a parent, runs 15 iterations
    deflated around every solution found so far, then the full NAE solve.
    """
    rng = np.random.default_rng(args.seed)
    eq, p = make_problem(args, qsc_eq, eq_fit, 400)
    x0 = p.x_full()
    centers, pool = [], []  # pool: (qs, x) of valid solutions
    i = 0
    while time.time() - rec.t0 < args.budget:
        if i == 0:
            xs, parent = x0, None
        else:
            pool.sort(key=lambda t: t[0])
            src = pool if pool else [(np.inf, x0)]
            w = np.exp(-np.arange(len(src)))
            parent = rng.choice(len(src), p=w / w.sum())
            xs = p.perturb(src[parent][1], args.sigma, rng)
            p.set_deflation(centers[::-1], radius=args.defl_radius, weight=1.0)
            xs = p.local_solve(xs, maxiter=15, xtol=1e-6)["x"]
            p.clear_deflation()
        r = p.local_solve(xs)
        centers.append(r["x"])
        p.set_x_full(r["x"])
        s = rec.add(eq, fixed_boundary(eq), f"hop {i}",
                    extra=dict(parent=None if parent is None else int(parent)))
        if s["valid"] and (len(pool) < pool_size or s["qs"] < pool[-1][0]):
            pool = sorted(pool + [(s["qs"], r["x"])], key=lambda t: t[0])[:pool_size]
        i += 1


def main():
    """Run one method for ``--budget`` seconds, recording every solution."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["deflation", "deflation_sum", "multistart", "defl_multistart",
                             "qs_pool"])
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--length-scale", type=float, default=0.02)
    ap.add_argument("--defl-radius", type=float, default=None)
    args = ap.parse_args()
    if args.sigma is None:
        args.sigma = {"multistart": 1.0, "defl_multistart": 0.5, "qs_pool": 0.5}.get(args.method, 0.0)
    if args.defl_radius is None:
        args.defl_radius = {"defl_multistart": 1.0, "qs_pool": 0.5}.get(args.method, 0.0)
    qsc_eq, eq_fit = setup()
    rec = Recorder(args.method, args.seed, args)
    if args.method == "deflation":
        run_deflation(args, qsc_eq, eq_fit, rec)
    elif args.method == "deflation_sum":
        run_deflation(args, qsc_eq, eq_fit, rec, reduction="sum")
    elif args.method == "multistart":
        run_multistart(args, qsc_eq, eq_fit, rec, deflate=False)
    elif args.method == "defl_multistart":
        run_multistart(args, qsc_eq, eq_fit, rec, deflate=True)
    else:
        run_qs_pool(args, qsc_eq, eq_fit, rec)
    rec.save()
    v = [s for s in rec.sols if s["valid"]]
    print(json.dumps(dict(n=len(rec.sols), valid=len(v),
                          best_qs=min((s["qs"] for s in v), default=None))))


if __name__ == "__main__":
    main()
