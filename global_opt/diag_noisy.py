"""Diagnose the annealed noisy gradient's excursions on the stage-two coil problem.

From one fixed start (a stored local minimum), run ``--n`` independent excursions
with the given hyperparameters, polish each with ``lsq-exact``, and record:

* the cost at the end of the excursion and after polishing,
* how far the excursion travelled (RMS distance in perturbation-scale units, over
  the perturbed Fourier coefficients and currents), for the excursion end point and
  the polished minimum,
* how much of the travel came from the Adam drift and how much from the noise.

Usage::

    python diag_noisy.py --start results/stage2_fixed/noisy_seed0_coilset.h5 \\
        --lr 0.02 --T0 0.3 --beta1 0.9 --steps 200 --n 8 [--free-frame]

Results go to ``results/diag_noisy/<frame>_lr<lr>_T<T0>_b<beta1>_s<steps>.json``.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from desc.io import load  # noqa: E402
from desc_global import ProjectedProblem  # noqa: E402
from stage2_precise_QA import setup  # noqa: E402


def excursion(problem, x0, lr, T0, beta1, steps, rng, beta2=0.999):
    """One annealed noisy-Adam excursion, as in ``annealed_noisy_gradient``.

    Also returns the accumulated drift (Adam) and noise displacements, in full x.
    """
    z = np.asarray(problem.project(x0))
    zs = np.abs(np.asarray(problem.project(x0 + problem.scales)) - z)
    zs = np.where(zs > 0, zs, np.median(zs[zs > 0]))
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    drift = np.zeros_like(x0)
    noise = np.zeros_like(x0)
    c = np.nan
    for k in range(steps):
        c, g = problem.value_and_grad_z(z)
        if not np.isfinite(c):
            break
        gs = g * zs
        m = beta1 * m + (1 - beta1) * gs
        v = beta2 * v + (1 - beta2) * gs**2
        mh = m / (1 - beta1 ** (k + 1))
        vh = v / (1 - beta2 ** (k + 1))
        x_before = problem.recover(z)
        z = z - lr * zs * mh / (np.sqrt(vh) + 1e-12)
        x_cur = problem.recover(z)
        drift += x_cur - x_before
        T = T0 * (1 - k / steps) ** 2
        if T > 0:
            xn = problem.perturb(x_cur, np.sqrt(2 * lr * T), rng)
            z = np.asarray(problem.project(xn))
            noise += problem.recover(z) - x_cur
    c_end, _ = problem.value_and_grad_z(z)
    return problem.recover(z), float(c_end), drift, noise


def rms(problem, dx):
    """RMS of a displacement in perturbation-scale units (perturbed entries only)."""
    s = problem.scales
    msk = s > 0
    return float(np.sqrt(np.mean((dx[msk] / s[msk]) ** 2)))


def main():
    """Run the excursions for one setting and write the JSON summary."""
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--free-frame", action="store_true")
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--T0", type=float, default=0.3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    eq, coilset, obj, cons = setup()
    problem = ProjectedProblem(
        coilset,
        obj,
        cons,
        local_kwargs=dict(maxiter=100, ftol=1e-4, xtol=1e-6),
        length_scale=0.05,
        alpha=0.5,
        current_rel=0.1,
        free_curve_frame=args.free_frame,
        verbose=1,
    )
    # build with the circular coils first, as the driver does (objective
    # normalizations are set at build time), then load the start coils
    for c, d in zip(coilset, load(os.path.join(HERE, args.start)).params_dict):
        c.params_dict = d
    x0 = problem.x_full()
    c0 = problem.cost(x0)
    print(f"start cost {c0:.4f}", flush=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n):
        t0 = time.time()
        xe, ce, drift, noise = excursion(
            problem, x0, args.lr, args.T0, args.beta1, args.steps, rng
        )
        t1 = time.time()
        res = problem.local_solve(xe)
        row = dict(
            i=i,
            raw=ce,
            polished=res["cost"],
            d_end=rms(problem, xe - x0),
            d_pol=rms(problem, res["x"] - x0),
            d_drift=rms(problem, drift),
            d_noise=rms(problem, noise),
            t_exc=t1 - t0,
            t_pol=res["time"],
        )
        rows.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}),
              flush=True)
    pol = np.array([r["polished"] for r in rows])
    summary = dict(
        args=vars(args),
        start_cost=c0,
        rows=rows,
        better=int(np.sum(pol < c0)),
        median_polished=float(np.median(pol)),
        best_polished=float(np.min(pol)),
        median_d_end=float(np.median([r["d_end"] for r in rows])),
        median_d_pol=float(np.median([r["d_pol"] for r in rows])),
    )
    out = os.path.join(HERE, "results", "diag_noisy")
    os.makedirs(out, exist_ok=True)
    name = (f"{'free' if args.free_frame else 'fixed'}_lr{args.lr}_T{args.T0}_b{args.beta1}"
            f"_s{args.steps}{args.tag}.json")
    json.dump(summary, open(os.path.join(out, name), "w"), indent=1)
    print("SUMMARY", name, {k: v for k, v in summary.items() if k not in ("rows", "args")})


if __name__ == "__main__":
    main()
