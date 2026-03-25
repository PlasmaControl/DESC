# DESC GPU Profiling Design

## Goal

One-time profiling of 4 DESC optimization cases to answer three questions:

1. **Step anatomy**: Where does wall time go in an optimization step? (Jacobian vs objective eval vs linear solve vs Python overhead)
2. **GPU utilization**: Is the GPU actually busy, or mostly idle between kernel launches?
3. **Objective cost breakdown**: For each case, which objectives/Jacobians are the most expensive?

These answers determine which speedup strategies (block-sparse Jacobian, analytical Jacobians, CPU vs GPU, etc.) apply to which optimization types.

## Cases

| # | Type | Optimizer | Source script |
|---|------|-----------|---------------|
| 1 | Coil opt | `lsq-auglag` | `finding_better_basins/run_baseline.py` |
| 2 | Coil opt | `lsq-exact` | Same setup, swap optimizer |
| 3 | Free boundary eq | `proximal-lsq-exact` | `dynamic-accessibility/.../free_boundary_half-slew_lower-res.py` |
| 4 | Single-stage | `proximal-lsq-exact` | `dynamic-accessibility/.../free_boundary_half-slew_eq-QS-divLeg-...-lower-res.py` |

## Lightweight Settings

Each case runs at reduced resolution -- just enough for 3-5 optimizer steps post-warmup. Not seeking convergence, just representative per-step timing.

- Coil opt: use `lightweight=True` config, `maxiter=5`
- Free boundary: reduce `jac_chunk_size`, `maxiter_boundary=5`, smaller grids
- Single-stage: `LIGHTWEIGHT=True` (already supported in config), `maxiter=5`

## Profiling Method

### Layer 1: Nsight Systems (hardware-level)

Answers questions 1 and 2.

**Execution:**
```bash
nsys profile --output=profiling/case_N --force-overwrite true python profiling/case_N.py
```

**Analysis:**
```bash
nsys export --type=sqlite profiling/case_N.nsys-rep
```

Then parse the SQLite database programmatically to extract:
- Total GPU kernel execution time vs total wall time (GPU utilization %)
- Kernel launch count and average launch overhead
- Top-10 kernels by cumulative duration
- CUDA memcpy time (host-to-device, device-to-host)
- Gaps between kernels (idle time)

**Output**: Summary table per case with GPU utilization %, kernel launch overhead %, memcpy overhead %.

### Layer 2: JAX Profiler (operation-level)

Answers question 1 with DESC-level granularity.

**Instrumentation** (via monkey-patching, not modifying DESC source):
- Wrap `optimizer.optimize()` with `jax.profiler.trace()`
- Add `jax.profiler.TraceAnnotation` context managers around:
  - `ObjectiveFunction.jac_scaled` / `jvp_scaled` (Jacobian computation)
  - `ObjectiveFunction.compute_scaled` (objective evaluation)
  - Trust-region subproblem solve (within `lsqtr`)
  - `LinearConstraintProjection._jac` (constraint Jacobian)

**Analysis**: Parse the TensorBoard protobuf trace to extract time per annotated region per optimization step.

**Output**: Per-step breakdown: % Jacobian, % objective eval, % linear solve, % other (Python/overhead).

### Layer 3: Per-objective Jacobian breakdown

Answers question 3 directly.

**Method**: After building the `ObjectiveFunction` for each case:
1. Warmup: call each sub-objective's `jvp_scaled` once (triggers JIT compilation)
2. Time: call each sub-objective's `jvp_scaled` 3-5 times with `block_until_ready()`, take median
3. Also time the full `ObjectiveFunction.jvp_scaled` for comparison (sum of parts vs whole)

**Output**: Per-case table showing each objective's Jacobian cost as % of total.

## Deliverables

### Scripts (in `profiling/` directory)

- `case_1_coil_auglag.py` -- Lightweight coil opt with lsq-auglag
- `case_2_coil_exact.py` -- Same setup with lsq-exact
- `case_3_free_boundary.py` -- Lightweight free boundary eq
- `case_4_single_stage.py` -- Lightweight single-stage
- `analyze_nsight.py` -- Parses nsys SQLite exports, prints summary tables
- `analyze_jax_trace.py` -- Parses TensorBoard traces, prints step breakdown
- `run_all.sh` -- Runs all 4 cases through both profiling layers

### Report

A single summary printed to stdout by the analysis scripts:

```
=== DESC GPU Profiling Report ===

Case 1: Coil opt (lsq-auglag)
  GPU utilization: XX%
  Step breakdown: Jacobian XX% | Obj eval XX% | Linear solve XX% | Other XX%
  Jacobian by objective:
    QuadraticFlux:      XX% (XXms)
    CoilSetMinDistance:  XX% (XXms)
    CoilLength:          XX% (XXms)
    FieldNormalError:    XX% (XXms)

Case 2: Coil opt (lsq-exact)
  ...

Case 3: Free boundary (proximal-lsq-exact)
  ...

Case 4: Single-stage (proximal-lsq-exact)
  ...

=== Recommendations ===
- ...
```

## Constraints

- All profiling scripts are self-contained and reference data files by absolute path
- No modifications to DESC source code -- all instrumentation via monkey-patching
- Scripts are one-time use; delete after report is generated
- Must run on Perlmutter GPU nodes (A100)
- `nsys` and `ncu` are available at `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/`

## Non-goals

- Not building a reusable profiling framework
- Not profiling compilation time (only steady-state per-step cost)
- Not profiling memory usage (separate investigation if needed)
- Not modifying any optimization parameters or DESC internals
