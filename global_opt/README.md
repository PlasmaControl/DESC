# Global / stochastic optimization for DESC stage-one and stage-two problems

DESC's optimizers (`lsq-exact`, `proximal-lsq-exact`, ...) are local trust-region
methods: they converge to the local minimum nearest the initial guess. This folder
adds global search strategies on top of them, and tests them on the optimization
examples that ship with DESC:

* **Stage two:** the first problem of
  `docs/notebooks/tutorials/coil_stage_two_optimization.ipynb`. That's 3 unique
  `FourierXYZCoil`s (N = 10) for the vacuum precise QA equilibrium, starting from
  circular coils, with the tutorial's exact objective, weights and constraints.
* **Stage one:** `desc/examples/precise_QA.py` (QS two-term + aspect ratio + iota,
  continuation over boundary modes with `proximal-lsq-exact`). It runs at reduced
  resolution (M = N = 6, boundary modes up to 3) so that many local solves fit in a
  CPU budget.
* **NAE-constrained equilibria:** the first problem of
  `docs/notebooks/tutorials/deflation.ipynb` (precise-QA near-axis constraints).
  It has many solutions and follows the deflation paper (Panici et al. 2026).

A second round used what the first round showed, together with the deflation
paper, to build a better method: **deflated pool search**. It was tested on the
coil problem, and its ideas were applied to the NAE-constrained problem.

All runs were made on 4 CPU cores (no GPU) with DESC master at commit `d644e62`.

> **Correction (21 November 2026).** Earlier versions of this README compared the
> stage-two results with a tutorial result of 197.9 and reported an 84×
> improvement. That baseline was wrong. My wrapper left each coil's rotation
> matrix and shift free, while DESC's `Optimizer` fixes them (`FixCurveRotation`,
> `FixCurveShift`). They are redundant with the Fourier coefficients, so they don't
> change which coils can be reached. They do change the local solver's path, and
> from circular coils the free-frame solve stops in a much worse minimum. With
> them fixed, now the default, the tutorial's single solve ends at **31.0**. That
> matches the notebook's own output (`4.516e+06 --> 3.100e+01`). Every stage-two
> number in this README now uses the fixed frame; the improvement is about 14×,
> not 84×. The free-frame runs are kept in `results/stage2/` and summarized in
> [Earlier runs with the coil frame free](#earlier-runs-with-the-coil-frame-free).

## Results in brief

* **Stage two: global search finds coils about 14× better than the tutorial's.**
  The tutorial's single `lsq-exact` solve ends at cost **31.0**. The same solve
  run to tolerances of 1e-10 also ends at 31.0, so it's a genuine local minimum,
  not an unconverged solve. Deflated pool search reached **2.21 (14× lower)** in
  20 minutes on 3 CPU cores, with ⟨|B·n|⟩/⟨|B|⟩ = 1.5e-4 instead of 3.7e-4.
* **Deflated pool search is the most consistent method.** It combines a pool of
  the best minima, small kicks with a deflated escape phase, and early
  abandonment of poor hops. Its three seeds ended at 2.96, 2.21 and 2.51. Basin
  hopping ended at 5.91, 5.74 and 2.44, and the annealed noisy gradient at 8.8,
  16.3 and 23.5. Without deflation, the same pool search ended at 6.7 on both
  seeds.
* **Not just looser coils.** The better coils are longer (up to 11.4 m against
  7.8 m) and closer together (5–6 cm against 10.5 cm), which the tutorial's
  bounds allow. Bounding the coil–coil distance and coil length at the tutorial
  result's own values (0.105 m, 7.8 m), a single solve ends at
  31.0. Deflated pool search reaches **9.10 (3.4× lower)** within those
  bounds, and basin hopping 14.24.
* **NAE-constrained equilibria: different tools for different goals.** The
  tutorial's deflation spreads its solutions widest, but it stalls after about 55
  because its product-form operator overflows float64. A QS-guided deflated pool
  finds NAE-constrained equilibria with 2–2.5 times lower outer-volume QS error
  (best 3.9e-3 against 9.6e-3), and its whole population is 2.4 times better at
  the median.
* **Stage one: smaller but real gains.** Against the example's continuation
  result (2.02e-4), multistart and basin hopping from scratch reach 1.39–1.43e-4
  (~30 % lower). Basin hopping started from the example's result reaches
  **1.16e-4 (43 % lower)**. The gain is almost entirely in the QS error; aspect
  ratio and ι stay on target.
* **Use deflated pool search, not CMA-ES.** In the earlier free-frame runs, CMA-ES
  did worst of all the global methods (54.9, or 35.8 restricted to |n| ≤ 2
  modes, while the others reached 2.4–16). The methods built on DESC's
  exact-Jacobian local solver win.

## Stage two: precise QA coils (tutorial problem)

Every global run had the same 20-minute wall-clock budget on the same 3 cores.
Each coil's rotation and shift are fixed, as `Optimizer.optimize` does. The
objective is the tutorial's, unchanged: cost = ½‖weighted residuals‖². In the
better minima it's almost entirely the quadratic-flux term. "Local solves"
counts calls to `lsq-exact` (15–40 s each); a pool-search hop makes two, the
deflated phase and the polish.

| Method | Seed | Best cost | ⟨\|B·n\|⟩/⟨\|B\|⟩ | Min coil-coil [m] | Longest coil [m] | Local solves |
|---|---|---|---|---|---|---|
| Tutorial (single `lsq-exact`) | – | 31.00 | 3.66e-4 | 0.105 | 7.79 | 1 |
| Same start, tolerances 1e-10 | – | 31.00 | 3.66e-4 | 0.105 | 7.79 | 1 |
| Basin hopping | 0 / 1 / 2 | 5.91 / 5.74 / 2.44 | 2.3e-4 / 2.1e-4 / 1.6e-4 | 0.052 / 0.051 / 0.051 | 11.2 / 10.5 / 11.1 | 39 / 37 / 32 |
| Annealed noisy gradient | 0 / 1 / 2 | 8.79 / 16.27 / 23.45 | 2.3e-4 / 2.8e-4 / 3.4e-4 | 0.077 / 0.051 / 0.057 | 8.7 / 8.3 / 8.1 | 20 / 21 / 28 |
| Pool search without deflation (ablation) | 0 / 1 | 6.67 / 6.72 | 2.5e-4 / 2.0e-4 | 0.049 / 0.070 | 11.6 / 9.0 | 62 / 53 |
| **Deflated pool search** | 0 / 1 / 2 | 2.96 / **2.21** / 2.51 | 1.5e-4 / **1.5e-4** / 1.5e-4 | 0.063 / 0.057 / 0.063 | 10.3 / 11.4 / 10.6 | 70 / 60 / 49 |

Full per-term breakdown, curvature, plasma-coil distance and timings are in
[`results/summary.md`](results/summary.md).

![Stage two convergence](figures/stage2_fixed_convergence.png)

![Stage two local minima](figures/stage2_fixed_local_minima.png)

![Stage two B.n and coils](figures/stage2_fixed_Bn_and_coils.png)

What these show:

* **The landscape is rugged.** Basin hopping's 105 local minima span 2.4 to
  5800, and only 59 % beat the tutorial's 31.0. A single solve from circular
  coils lands in one of these minima, more or less at random.
* **Good minima cluster.** Restarting near a good minimum beats restarting
  far from it. Every one of pool search's 83 polished minima beat the tutorial
  (median 4.1), against a median of 18.9 for basin hopping, whose kicks are
  larger.
* **One run is one draw, except for pool search.** Basin hopping ended between
  2.44 and 5.91 depending on the seed, and the noisy gradient between 8.8 and
  23.5. Pool search's three seeds all ended between 2.21 and 2.96.
* **The noisy gradient did not hold up.** With the frame free it was the most
  consistent first-round method (4.4, 5.7 and 9.0). With the frame fixed, its
  seeds ended at 8.8, 16.3 and 23.5, and only 32 % of its minima beat the
  tutorial. I haven't isolated why. Its Adam excursions move through all the
  projected variables, and with the frame free those included 36 redundant ones;
  that is one candidate.

### Same problem with bounds matched to the tutorial result

The better minima lengthen the coils (up to 11.4 m against 7.8 m) and push the
coil-coil distance down to the tutorial's bound (5–6 cm against 10.5 cm). The
tutorial's objective allows both, but you may not want them. `--variant matched`
sets those bounds to the tutorial result's own values: coil-coil distance
≥ 0.105 m and every coil ≤ 7.8 m. The tutorial's result therefore stays
feasible, with the same cost.

| Method | Seed | Best cost | ⟨\|B·n\|⟩/⟨\|B\|⟩ | Min coil-coil [m] | Coil lengths [m] | Local solves |
|---|---|---|---|---|---|---|
| Single `lsq-exact` (tutorial procedure) | – | 31.00 | 3.66e-4 | 0.105 | 7.79, 6.24, 6.12 | 1 |
| Basin hopping | 0 | 14.24 | 2.56e-4 | 0.105 | 7.81, 6.64, 6.80 | 32 |
| **Deflated pool search** | 0 | 9.10 | 1.92e-4 | 0.105 | 7.81, 7.16, 7.33 | 68 |

![Stage two, matched bounds, convergence](figures/stage2_fixed_matched_convergence.png)

![Stage two, matched bounds, B.n and coils](figures/stage2_fixed_matched_Bn_and_coils.png)

Within the tutorial result's own spacing and length, deflated pool search still
reaches **9.10**, 3.4 times lower than the single solve (31.0), and basin hopping
reaches 14.24. ⟨|B·n|⟩/⟨|B|⟩ falls 1.9× (3.7e-4 to 1.9e-4). So the gain shrinks
from 14× to 3.4× once the coils can't grow longer or closer, but a real gain
remains. The best coils keep the tutorial's overall layout. The bounds are soft
penalties, as in the tutorial, so the minima sit just past them (0.1047 m instead
of 0.105 m, 7.81 m instead of 7.8 m). Raise those terms' weights if they must hold
exactly.

## Round two: deflated pool search

### What the first round and the deflation paper suggested

The first round ran with the coil frame free (see the correction above), so its
numbers differ, but the lessons hold with the frame fixed.

* **Seed scatter.** One basin-hopping walker ended at 12.4 or 2.4 depending on
  the seed. A single walker is one draw.
* **Most local solves are wasted, but rarely on repeats.** A run's median
  minimum was 2.4–17 times worse than its best, while exact repeats were rare
  (0–2 per run). Avoiding known minima matters less than avoiding bad basins.
* **Deflation can replace big jumps.** Panici et al. (2026) push a solve away
  from known solutions with a deflation operator. Applied to hops, deflation
  can move a solve out of the parent's basin after only a small kick. Plain
  basin hopping needs kicks large enough to leave the basin, and those often land
  somewhere much worse.

### The method: `pool_search`

Each hop:

1. **Choose a parent** from a pool of the 4 best distinct minima, with
   probability ∝ exp(−rank).
2. **Kick it slightly**: σ starts at 0.3 perturbation-scale units and adapts
   toward a 30 % rate of entering the pool. Then run the first 20 iterations of
   `lsq-exact` **deflated** around the parent, the other pool members and the 4
   most recent minima. The penalty is `ScaledDeflation`: zero outside an RMS
   radius of 3σ around every center, growing without bound inside it. This is the
   sum-reduction deflation constraint of the paper, eq. (4.4), applied as a
   least-squares penalty.
3. **Race**: if the undeflated cost after that phase is more than 3 times the
   worst pool member, abandon the hop.
4. **Polish** the survivors without deflation, so every pool member is a true
   local minimum of the original problem.

`ScaledDeflation` keeps its centers, weight and radius as array attributes. DESC
objectives are pytrees, so these are dynamic leaves: changing them between solves
does not recompile anything, and with zero weight the term contributes exactly
nothing.

### Results

| Run | Seed | Hops | Abandoned early | Polished | Entered pool | Best cost |
|---|---|---|---|---|---|---|
| Pool search without deflation | 0 / 1 | 37 / 29 | 13 / 6 | 24 / 23 | 16 / 14 | 6.67 / 6.72 |
| **Deflated pool search** | 0 / 1 / 2 | 38 / 31 / 24 | 7 / 3 / 0 | 31 / 28 / 24 | 21 / 17 / 15 | 2.96 / 2.21 / 2.51 |
| Deflated pool search, matched bounds | 0 | 34 | 1 | 33 | 16 | 9.10 |

* **Consistent.** All three seeds ended between 2.21 and 2.96, with
  ⟨|B·n|⟩/⟨|B|⟩ = 1.5e-4. The worst seed beats every seed of the noisy gradient
  and of the ablation, and two of basin hopping's three.
* **Deflation is what makes it work.** With deflation off (and the larger
  basin-hopping kicks), the same pool-and-racing search ended at 6.7 on both
  seeds. Deflated hops started from kicks of only 0.3–0.6 scale units, yet
  landed a median 2.8 units from their parent: deflation, not a random jump,
  carried them into a neighboring basin.
* **Racing saves solves.** After the first 20 iterations the cost is usually
  close to the polished value (median 0.1–5 % off), so the race is a reliable
  early test. 0–7 of 24–38 hops per run were abandoned there, and none of the 83
  polished hops returned to a pool member.

### Earlier runs with the coil frame free

Rounds one and two ran with each coil's rotation matrix and shift free (36 extra,
redundant unknowns). From circular coils, the single solve then stops at 197.9
instead of 31.0. The global methods were less affected, because they restart
from their own minima:

| Method | Frame free: each seed | Worst seed | Frame fixed: each seed | Worst seed |
|---|---|---|---|---|
| Tutorial (single `lsq-exact`) | 197.9 | – | 31.0 | – |
| Multistart | 16.43, 14.10 | 16.43 | not rerun | – |
| CMA-ES + polish | 54.88 | 54.88 | not rerun | – |
| CMA-ES on \|n\| ≤ 2 modes + polish | 35.77 | 35.77 | not rerun | – |
| Basin hopping | 12.38, 2.37, 2.91 | 12.38 | 5.91, 5.74, 2.44 | 5.91 |
| Annealed noisy gradient | 4.41, 5.68, 8.97 | 8.97 | 8.79, 16.27, 23.45 | 23.45 |
| Pool search without deflation | 10.99, 4.44 | 10.99 | 6.67, 6.72 | 6.72 |
| **Deflated pool search** | 2.37, 3.55, 2.51 | 3.55 | 2.96, 2.21, 2.51 | 2.96 |

* **What still holds:** deflated pool search was the best and most consistent
  method in both settings. Basin hopping scatters with the seed in both, and
  deflation helps pool search in both.
* **What changed:** the size of the gain over the tutorial (14× rather than 84×)
  and the noisy gradient's standing (see above).
* **The "tight" variant was based on the wrong baseline.** Its bounds (coil-coil
  distance ≥ 0.14 m, coils ≤ 7 m) were the free-frame result's own values. The
  real tutorial result violates them (0.105 m, 7.79 m), so that variant does not
  check what it was meant to. `--variant matched` replaces it. For the record,
  with the frame free the tight variant gave: single solve 197.9, multistart
  70.0, noisy gradient 68.5, basin hopping 54.5, deflated pool search 29.1.

The free-frame summaries and coilsets are in `results/stage2/`; rerun them with
`--free-frame`. Their figures are `figures/stage2_*.png` without `_fixed`.

## NAE-constrained equilibria (the deflation tutorial problem)

**Problem.** This is the first section of
`docs/notebooks/tutorials/deflation.ipynb`, which follows the deflation paper.
The global equilibrium is constrained to the axis and O(ρ) behaviour of the
first-order precise-QA near-axis solution (pyQSC), at the tutorial's resolution:
L = M = 5, N = 6, initial surfaces from the NAE at r = 0.25.

Minimizing force balance under these linear constraints is under-constrained, so
there are many solutions. Every candidate is re-solved as a fixed-boundary
equilibrium, as in the tutorial, and counted as **valid** if both force errors are
below 1 %.

The NAE guarantees quasisymmetry only near the axis. The outer surfaces differ
between solutions, so the figure of merit is the Boozer QS error f_B averaged over
ρ = 0.5, 0.75 and 1: "better" means a lower value.

**Methods.** `nae_global.py` runs each method for 30 minutes on one CPU core:

| Method | What it does |
|---|---|
| Deflation, tutorial | The tutorial's Algorithm 1: deflated solve from the same initial guess each time (`DeflationOperator`, M·f, p = 2, σ = 500, product over all solutions, on the boundary coefficients), then the NAE solve, then the fixed-boundary solve |
| Deflation, sum form | The same with the paper's sum-reduction operator, eq. (4.4) |
| Multistart | Smooth random perturbations of the initial guess, then the NAE solve. Only Zernike modes vanishing as ρ² or faster at the axis are perturbed: perturbing all modes produced non-nested surfaces that never converged |
| Deflated multistart | Multistart whose first 15 iterations are deflated around every solution found so far (`ScaledDeflation`, nearest-center form) |
| QS-guided deflated pool | Pool search on the QS figure of merit: kick one of the 5 best-QS solutions, deflate around every known solution, re-solve |

**Results.** Distinct means boundary coefficients more than δ apart (L2 norm, m)
from every solution counted before.

| Method | Seed | Valid | Distinct, δ = 0.02 / 0.04 | Best QS error | Median QS error | ι edge range |
|---|---|---|---|---|---|---|
| Deflation, tutorial | – | 68 | 54 / **39** | 9.6e-3 | 2.4e-2 | **0.378–0.473** |
| Deflation, sum form | – | 58 | 18 / 6 | 9.3e-3 | 2.0e-2 | 0.421–0.460 |
| Multistart | 0 / 1 | 75 / 73 | 70 / 32–36 | 7.1e-3 / 8.2e-3 | 1.7e-2 | 0.355–0.434 |
| Deflated multistart | 0 / 1 | 68 / 67 | 64 / 17–22 | 4.2e-3 / 5.4e-3 | 1.3–1.5e-2 | 0.372–0.440 |
| QS-guided deflated pool | 0 / 1 | 69 / 67 | 45–50 / 10 | **4.8e-3 / 3.9e-3** | **1.0e-2** | 0.398–0.433 |

![NAE-constrained equilibria found and their best quasisymmetry](figures/nae_search.png)

![Every valid NAE-constrained equilibrium](figures/nae_diversity.png)

* **The tutorial's deflation spreads solutions widest** (39 distinct at
  δ = 0.04, ι at the edge from 0.38 to 0.47). But it **stops finding new
  solutions after about 55**. Every factor of its product-form operator is at
  least σ = 500, so with ~55 deflated states the squared deflated cost passes
  10³⁰⁸ and overflows float64. The deflated solve then fails, and every later
  iteration returns the first solution again. Computing the product in log space
  would fix this. So would the sum form, but at σ = 500 that form deflates too
  weakly here, and most of its solutions were near-duplicates (18 distinct of
  58 at δ = 0.02).
* **Random starts find more solutions per minute.** Shape-perturbed multistart
  found 73–75 valid equilibria per half hour (70 distinct at δ = 0.02), but
  closer together than deflation's.
* **The QS-guided pool finds better equilibria.** Its best solutions have
  2–2.5 times lower outer-volume QS error than the tutorial deflation's best
  (3.9e-3 and 4.8e-3 against 9.6e-3), and its whole population is 2.4 times
  better at the median (1.0e-2 against 2.4e-2). As designed, it searches near
  good solutions, so its solutions are less spread out (10 distinct at
  δ = 0.04).
* **What to use when.** To map the family of equilibria, use deflation, with the
  overflow fixed. To pick the best equilibrium for a later stage-one
  optimization, use the QS-guided pool, or deflated multistart for a balance of
  spread and quality.

## Stage one: precise QA boundary (reduced resolution)

Each run used 1 CPU core and a budget of at most 1 hour (the last column is wall time).
Each local optimization is a `proximal-lsq-exact` solve, which re-solves the
equilibrium; one takes 1.5–7 min (a full k = 1..3 continuation).

| Method | Best cost | QS term | Aspect ratio | ι (edge) | Local optimizations | Wall [s] |
|---|---|---|---|---|---|---|
| Example (continuation k = 1, 2, 3) | 2.02e-4 | 1.97e-4 | 6.000 | 0.420 | 1 | 636 |
| Multistart (perturbed initial boundary + continuation) | 1.43e-4 | 1.36e-4 | 6.000 | 0.419 | 11 | 3650 |
| Basin hopping (runs the example, then hops) | 1.39e-4 | 1.38e-4 | 6.000 | 0.420 | 29 | 3696 |
| Basin hopping from the example's saved result (exploratory, seed 1) | **1.16e-4** | 1.13e-4 | 6.000 | 0.419 | 21 | 2541 |

![Stage one convergence](figures/stage1_convergence.png)

![Stage one local minima](figures/stage1_local_minima.png)

The gains are tens of percent, not orders of magnitude, so stage one's landscape
looks much less rugged at this resolution. Global search is also costlier per
step there: each local solve takes minutes. Large boundary perturbations
sometimes make the equilibrium solve diverge (the open triangles in the figure).
Those trials are rejected automatically, but they waste time, so keep
`--sigma`/`--length-scale` moderate.

## Caveats

* **Coil frame.** The stage-two results above fix each coil's rotation and
  shift, as DESC's `Optimizer` does. Rounds one and two left them free; see
  [Earlier runs with the coil frame free](#earlier-runs-with-the-coil-frame-free).
* **The NAE runs used one core each,** at the tutorial's low resolution, with
  one or two seeds. Their seed-0 runs ran alongside stage-two runs, and the
  seed-1 runs and the sum-form run ran four at a time. Rates are comparable to
  within about 10 %.

* **CPU only.** All timings are from 3–4 CPU cores. On a GPU each local solve is
  faster, so more of them fit in the same budget. The ranking of the methods
  should not change.
* **Few seeds.** With the frame fixed, basin hopping, the noisy gradient and
  pool search ran 3 seeds each, the ablation 2 and the matched-bounds runs 1.
  Multistart and CMA-ES ran only with the frame free. Stage one ran 1 seed. The
  robust conclusions are "every global method ≫ single solve" and "deflated pool
  search is the most consistent"; finer rankings are within seed scatter.
* **Curvature bound everywhere.** Every stage-two solution, including the
  tutorial's, has its maximum curvature at the bound (2.0, normalized).
* **Reduced stage-one resolution.** Stage one ran at M = N = 6 with boundary
  modes up to 3, instead of M = N = 8 and 4 continuation steps. The minima at
  full resolution may differ.
* **A failed first design.** The first version of the noisy gradient (pure
  annealed Langevin dynamics from the circular coils) failed, ending at 208 (with
  the frame free). It
  is kept in `results/stage2/superseded/` for reference. The current version
  starts each excursion from the incumbent local minimum and always finishes with
  a local solve.

## Using this on your own problems

The tools work for any DESC problem whose constraints are linear: coil
optimization, fixed-boundary optimization with `lsq-exact`, and so on:

```python
from desc_global import ProjectedProblem, basin_hopping, multistart, annealed_noisy_gradient

problem = ProjectedProblem(
    coilset, objective, constraints,
    local_kwargs=dict(maxiter=100, ftol=1e-4),
    length_scale=0.05,   # ~10% of a typical coil radius, in meters
    alpha=0.5,           # smoothness: scale of mode n is length_scale * exp(-alpha * n)
    current_rel=0.1,
)
res0 = problem.local_solve(problem.x_full())           # the usual single solve
result = basin_hopping(problem, x0=res0["x"], sigma=1.0, temperature=0.5,
                       restart_after=6, time_budget=3600)
# or the round-two method (build the problem with deflation_slots=8):
# result = pool_search(problem, x0=problem.x_full(), time_budget=3600)
# or: annealed_noisy_gradient(problem, x0=res0["x"], temperature0=0.3, time_budget=3600)
coilset.save("best_coils.h5")                           # coilset now holds the best found
```

* With many Fourier modes, keep perturbations smooth. For example, for a coilset
  with 20 coils × 151 modes each, use a larger `alpha` or
  `max_order=...` (perturb only |n| ≤ max_order). Otherwise each hop is mostly
  high-frequency noise that the local solve has to undo.
* Multistart and independent basin-hopping seeds are embarrassingly parallel:
  run one process per GPU or core and keep the best.
* `ProjectedProblem` fixes each curve's rotation and shift, as DESC's
  `Optimizer` does. Pass `free_curve_frame=True` to leave them free, as in the
  first two rounds here; that changed where a single solve stopped by a factor
  of six.
* After a big improvement, check the per-term breakdown. In the tutorial
  objective the better minima come partly from longer, closer-packed coils.
  Decide whether that's acceptable, and tighten the bounds or raise the weights
  if not.
* For equilibrium (stage-one / free-boundary) problems, `ForceBalance` needs
  `ProximalProjection`, which `ProjectedProblem` does not wrap. Follow
  `stage1_precise_QA.py` instead: perturb with `Equilibrium.perturb` and re-run
  `eq.optimize`.

## Files

| File | What it is |
|---|---|
| `desc_global.py` | Library: `ProjectedProblem` wrapper plus `multistart`, `basin_hopping`, `cma_es`, `annealed_noisy_gradient` |
| `stage2_precise_QA.py` | Stage-two driver (tutorial problem), one method per call |
| `run_stage2_all.sh` | Runs stage-two methods in sequence with the same budget and cores (`BUDGET`, `SEED`, `METHODS`, `EXTRA="--variant matched"`) |
| `stage1_precise_QA.py` | Stage-one driver (precise QA example): baseline / multistart / basin hopping |
| `run_stage1_all.sh` | Same for stage one |
| `nae_global.py` | NAE-constrained equilibria (deflation tutorial problem): tutorial deflation, sum-form deflation, multistart, deflated multistart, QS-guided pool |
| `analyze.py` | Builds `results/summary.md` and `figures/` |
| `results/stage2_fixed/*.json`, `*_coilset.h5` | Stage two with the coil frame fixed (the results above): per-run summaries (cost trace, every local minimum found, metrics) and the best coilsets; `*_matched_*` = matched-bounds variant |
| `results/stage2/*.json`, `*_coilset.h5` | The same for the earlier free-frame runs (rounds one and two); `*_tight_*` = tight-bounds variant |
| `results/stage1/*.json`, `*_eq.h5` | Same for stage one, with the best equilibria |
| `results/nae/*.json`, `*.npz` | Every NAE-constrained equilibrium found: metrics (force error, ι, aspect ratio, Boozer f_B(ρ), magnetic well) and boundary coefficients |
| `results/*/logs/` | Run logs (not committed) |
| `results/stage2/superseded/` | First (failed) noisy-gradient design, kept for reference |
| `figures/stage2_fixed_*.png` | Stage-two figures above; `figures/stage2_*.png` without `_fixed` are the free-frame ones |

## How the global search works

### Build once, solve many times

Each restart has to reuse the JIT-compiled residual and Jacobian, otherwise every
local solve pays the objective build + compile cost again. `ProjectedProblem`
builds the `ObjectiveFunction` and the `LinearConstraintProjection` once.
`Optimizer.optimize` accepts that prebuilt projection directly, so every local solve
is just "write a new state into the coils, call `lsq-exact`".

```python
from desc_global import ProjectedProblem, multistart, basin_hopping

problem = ProjectedProblem(
    coilset, objective, constraints,          # same objects you'd pass to Optimizer
    local_method="lsq-exact",
    local_kwargs=dict(maxiter=100, ftol=1e-4),
    length_scale=0.05,                        # perturbation size for Fourier coeffs [m]
    alpha=0.5,                                # spectral decay exp(-alpha * |n|)
    current_rel=0.1,                          # perturbation of currents, relative
)
result = multistart(problem, n_starts=30, sigma=1.0, time_budget=1200)
coilset  # now holds the best coils found; result.cost, result.history, result.trace
```

### Perturbations are smooth, mode-aware and constraint-consistent

`perturbation_scales` gives each Fourier coefficient of a curve or surface a scale
`length_scale * exp(-alpha * k)`, where `k` is the mode order. Random perturbations
are therefore smooth geometric deformations, not high-frequency noise. Currents get
`current_rel * mean|I|`. `rotmat`, `shift`, profiles and the equilibrium interior
are not perturbed. Any parameter can be overridden by name. Every trial point is
orthogonally projected onto the linear constraints, so `FixCoilCurrent`,
`FixParameters`, etc. stay satisfied.

### Strategies

| Function | Idea | Notes |
|---|---|---|
| `multistart` | Independent local solves from random perturbations of the initial guess | Embarrassingly parallel; also maps out the distribution of local minima |
| `basin_hopping` | Perturb the incumbent minimum, solve locally, Metropolis accept on `log(cost)`, adapt step size towards ~40 % acceptance, restart from the best after repeated failures | Exploits the structure near good minima |
| `cma_es` | CMA-ES on the raw objective (each generation is evaluated in one `vmap`-ed call), then `lsq-exact` polishing of the best candidates | Derivative-free; best used around an existing local minimum |
| `pool_search` | Pool of the best distinct minima; small kicks with a deflated first phase (`ScaledDeflation`), early abandonment of poor hops, undeflated polish | Round-two method; the most consistent on the coil problem |
| `annealed_noisy_gradient` | Langevin-driven basin hopping: from the incumbent minimum, run Adam in the projected variables with smooth noise annealed to 0 over the cycle, then polish with `lsq-exact` and keep the result if it is better | The excursions follow low-cost valleys instead of jumping isotropically; consistent with the coil frame free, much less so with it fixed |

The stage-one driver uses the same idea at the level of `Equilibrium.optimize`. The
boundary is perturbed with `Equilibrium.perturb` (modes up to the current k, major
radius untouched), and `ProximalProjection` re-solves force balance before each
local optimization.

## Reproducing

```bash
pip install -e . cma            # from the DESC repo root
cd global_opt
# stage two, coil frame fixed (results/stage2_fixed): 20 min per method on 3 cores
SEED=0 METHODS="baseline converged basin noisy pool pool_nodefl" ./run_stage2_all.sh
SEED=1 METHODS="basin noisy pool pool_nodefl" ./run_stage2_all.sh
SEED=2 METHODS="basin noisy pool" ./run_stage2_all.sh
EXTRA="--variant matched" LOGTAG=_matched METHODS="baseline pool basin" ./run_stage2_all.sh
# earlier free-frame runs (results/stage2): add --free-frame
EXTRA="--free-frame" ./run_stage2_all.sh                   # seed 0, round-one methods
EXTRA="--free-frame" SEED=1 METHODS="multistart basin noisy" ./run_stage2_all.sh
python stage2_precise_QA.py --free-frame --method cma --max-order 2 --tag _lowmode
EXTRA="--free-frame --variant tight" LOGTAG=_tight METHODS="baseline multistart basin noisy pool" ./run_stage2_all.sh
# stage one: 1 core, 1 h per method
python stage1_precise_QA.py --method baseline
python stage1_precise_QA.py --method multistart --budget 3600
python stage1_precise_QA.py --method basin --budget 3600
python stage1_precise_QA.py --method basin --budget 2400 --seed 1 --tag _explore \
    --init results/stage1/baseline_M6_k3_seed0_eq.h5
# NAE-constrained equilibria: 30 min on one core each
for m in deflation deflation_sum multistart defl_multistart qs_pool; do
  taskset -c 3 python nae_global.py --method $m --budget 1800 --seed 0
done
python analyze.py --fields
```

Runs write their JSON summary and best result when they finish. The stage-one
driver also checkpoints after every improvement.
