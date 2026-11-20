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

All runs were made on 4 CPU cores (no GPU) with DESC master at commit `d644e62`.

RESULTS_PLACEHOLDER

## Files

| File | What it is |
|---|---|
| `desc_global.py` | Library: `ProjectedProblem` wrapper plus `multistart`, `basin_hopping`, `cma_es`, `annealed_noisy_gradient` |
| `stage2_precise_QA.py` | Stage-two driver (tutorial problem), one method per call |
| `run_stage2_all.sh` | Runs stage-two methods in sequence with the same budget and cores (`BUDGET`, `SEED`, `METHODS`, `EXTRA="--variant tight"`) |
| `stage1_precise_QA.py` | Stage-one driver (precise QA example): baseline / multistart / basin hopping |
| `run_stage1_all.sh` | Same for stage one |
| `analyze.py` | Builds `results/summary.md` and `figures/` |
| `results/stage2/*.json`, `*_coilset.h5` | Per-run summaries (cost trace, every local minimum found, metrics) and the best coilsets; `*_tight_*` = tight-bounds variant |
| `results/stage1/*.json`, `*_eq.h5` | Same for stage one, with the best equilibria |
| `results/*/logs/` | Run logs (not committed) |
| `results/stage2/superseded/` | First (failed) noisy-gradient design, kept for reference |

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
| `annealed_noisy_gradient` | Langevin-driven basin hopping: from the incumbent minimum, run Adam in the projected variables with smooth noise annealed to 0 over the cycle, then polish with `lsq-exact` and keep the result if it is better | The excursions follow low-cost valleys instead of jumping isotropically; most consistent method in the stage-two tests |

The stage-one driver uses the same idea at the level of `Equilibrium.optimize`. The
boundary is perturbed with `Equilibrium.perturb` (modes up to the current k, major
radius untouched), and `ProximalProjection` re-solves force balance before each
local optimization.

## Reproducing

```bash
pip install -e /path/to/DESC cma
cd global_opt
# stage two: 20 min per method on 3 cores
./run_stage2_all.sh                                        # seed 0, all methods
SEED=1 METHODS="multistart basin noisy" ./run_stage2_all.sh
python stage2_precise_QA.py --method cma --max-order 2 --tag _lowmode
EXTRA="--variant tight" LOGTAG=_tight METHODS="baseline multistart basin noisy" ./run_stage2_all.sh
# stage one: 1 core, 1 h per method
python stage1_precise_QA.py --method baseline
python stage1_precise_QA.py --method multistart --budget 3600
python stage1_precise_QA.py --method basin --budget 3600
python stage1_precise_QA.py --method basin --budget 2400 --seed 1 --tag _explore \
    --init results/stage1/baseline_M6_k3_seed0_eq.h5
python analyze.py --fields
```

Runs write their JSON summary and best result when they finish. The stage-one
driver also checkpoints after every improvement.
