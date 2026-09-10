# Generalized toroidal angle (PR #2282) — review response plan

Tracks the `f0uriest` review comments on
[PlasmaControl/DESC#2282](https://github.com/PlasmaControl/DESC/pull/2282)
("Generalized toroidal angle (omega): re-derived against current master")
and the plan for addressing them. One commit per item, ordered easy to hard.
The `OMEGA_IS_0` / bounce-integral item is last on purpose — it is the item
with the largest blast radius (it touches live physics results for `Gamma_c`
and the effective-ripple objective), so it should land once everything
simpler is out of the way and the test suite is a stable baseline to compare
against.

## Background

`omega` is a periodic stream function that decouples the DESC computational
toroidal angle `zeta` from the lab-frame cylindrical angle `phi`:

```
phi(rho, theta, zeta) = zeta + omega(rho, theta, zeta)
```

stored spectrally as `W_lmn` (surfaces/equilibria) or `W_n` (axis curves).
Default `omega = 0` recovers old behavior exactly (`zeta = phi`).

## Review comments (25, all from `f0uriest`, 2026-09-09)

| # | File:line | Comment | Theme |
|---|-----------|---------|-------|
| 1 | [desc/compute/_geometry.py:21](../../desc/compute/_geometry.py#L21) | Multiple ways to compute `A(zeta)`; should measure divergence and either enforce one method when omega≠0, or restrict some quantities to phi=const planes | design decision |
| 2 | [desc/equilibrium/coords.py:132](../../desc/equilibrium/coords.py#L132) | Remove `OMEGA_IS_0` constant — no longer true that omega is identically zero everywhere | OMEGA_IS_0 |
| 3 | [desc/equilibrium/coords.py:132](../../desc/equilibrium/coords.py#L132) (reply) | Add a shortcut for mapping to PEST coordinates when omega≠0, common use case | OMEGA_IS_0 |
| 4 | [desc/equilibrium/equilibrium.py:159](../../desc/equilibrium/equilibrium.py#L159) | Prefer `Lw/Mw/Nw` over `Lz/Mz/Nz` (W is used elsewhere for omega) | rename |
| 5 | [desc/equilibrium/equilibrium.py:223](../../desc/equilibrium/equilibrium.py#L223) | Don't want `_io_attrs_optional_`; prefer the existing `_set_up`-fills-default + warns pattern | io_attrs |
| 6 | [desc/equilibrium/equilibrium.py:347](../../desc/equilibrium/equilibrium.py#L347) | Shouldn't every surface have an `Mz` attribute? | consistency |
| 7 | [desc/equilibrium/equilibrium.py:388](../../desc/equilibrium/equilibrium.py#L388) | Non-symmetric `W_basis` still carries the (0,0,0) mode; isn't there a "full without constant" symmetry option from `@ddudt`? | symmetry/gauge |
| 8 | [desc/equilibrium/equilibrium.py:764](../../desc/equilibrium/equilibrium.py#L764) (reply to #6) | "see previous comment" | consistency |
| 9 | [desc/equilibrium/equilibrium.py:1719](../../desc/equilibrium/equilibrium.py#L1719) | Surface setter grows the eq's omega resolution to match an incoming higher-res surface, but truncates R/Z resolution instead — inconsistent, intentional? | consistency |
| 10 | [desc/equilibrium/equilibrium.py:2394](../../desc/equilibrium/equilibrium.py#L2394) | `resolution_summary` only prints omega resolution when nonzero; print always for completeness | cleanup |
| 11 | [desc/geometry/curve.py:42](../../desc/geometry/curve.py#L42) | Class docstring intro line should be updated too (still says "in terms of toroidal angle phi" only) | cleanup |
| 12 | [desc/geometry/curve.py:89](../../desc/geometry/curve.py#L89) (reply to #5) | "see previous comment" | io_attrs |
| 13 | [desc/geometry/curve.py:128](../../desc/geometry/curve.py#L128) | `modes_R`/`modes_Z` not cast to int like `modes_W` is — shouldn't they all be ints? | cleanup |
| 14 | [desc/geometry/curve.py:161](../../desc/geometry/curve.py#L161) (reply to #13) | Alternative to #7: allow the constant mode but fix it to zero via gauge constraint, like lambda | symmetry/gauge |
| 15 | [desc/io/hdf5_io.py:125](../../desc/io/hdf5_io.py#L125) | "Not crazy about this" (same `_io_attrs_optional_` objection) | io_attrs |
| 16 | [desc/objectives/getters.py:381](../../desc/objectives/getters.py#L381) | Cleaner to always give a `W_basis` and always fix the extra coefficients, vs. the current conditional gating | simplification |
| 17 | [desc/objectives/linear_objectives.py:743](../../desc/objectives/linear_objectives.py#L743) | Outdated TODO wording ("yell at Dario" joke → should name `@YigitElma`) | cleanup |
| 18 | [desc/objectives/linear_objectives.py:1346](../../desc/objectives/linear_objectives.py#L1346) | Gauge freedom is any `c(rho, theta)`, not just `c(rho)` — `FixOmegaGauge` under-constrains | gauge (physics) |
| 19 | [desc/objectives/linear_objectives.py:1395](../../desc/objectives/linear_objectives.py#L1395) | Check whether `FixOmegaInterior` is still needed once the gauge fix (#18) is correct | gauge (physics) |
| 20 | [desc/objectives/linear_objectives.py:1399](../../desc/objectives/linear_objectives.py#L1399) (reply to #19) | The self-consistency/interior-fix split affects conditioning of the linear constraint projection; fixing an orthogonal sum may be better than individual DOFs | gauge (physics) |
| 21 | [desc/plotting.py:1566](../../desc/plotting.py#L1566) | `_phi_to_zeta_bisect` converges logarithmically; bracketed Newton would be faster and still branch-safe | numerics |
| 22 | [desc/plotting.py:1598](../../desc/plotting.py#L1598) (reply to #21) | Consider folding into `map_coordinates` generally, but worried about per-call overhead; maybe box-constrain the existing Newton solve instead | numerics |
| 23 | [desc/plotting.py:1656](../../desc/plotting.py#L1656) | Question: shouldn't a "failed" inversion just be `phi + 2*pi*k` (same point), not actually outside the plasma? | needs a reply, not code |
| 24 | [desc/plotting.py:2105](../../desc/plotting.py#L2105) | "Not sure this comment is useful" | cleanup |
| 25 | [tests/test_generalized_toroidal.py:337](../../tests/test_generalized_toroidal.py#L337) | Already covered by `test_transform`/`test_basis`; add finite-difference-vs-analytic derivative checks instead, like `test_magnetic_field_derivatives` in `test_compute_funs.py` | tests |

## Investigation: `OMEGA_IS_0` and the bounce integrals

Two use sites, not one:

- [coords.py:132](../../desc/equilibrium/coords.py#L132) — `map_coordinates`'s closed-form
  PEST inversion shortcut. `OMEGA_IS_0` is hardcoded `True` and only ever
  `and`ed with a real per-equilibrium check (`eq.W_basis` empty or absent) —
  removing it here is a pure no-op today.
- [coords.py:562](../../desc/equilibrium/coords.py#L562), inside
  `_map_poloidal_coordinates` — the `alpha -> theta` root-find that
  underlies every bounce-integral field-line grid. This one is load-bearing:
  it's called directly by `GammaC.compute` (`Gamma_c`,
  [_fast_ion.py:215](../../desc/objectives/_fast_ion.py#L215)), the
  effective-ripple objective
  ([_neoclassical.py:191](../../desc/objectives/_neoclassical.py#L191)), and
  `Bounce1D/2D`'s own coordinate mapping
  ([bounce_integral.py:608](../../desc/integrals/bounce_integral.py#L608)).
  Naively deleting the `OMEGA_IS_0` name breaks all three with a
  `NameError`, regardless of whether the equilibrium in question has any
  omega at all — a much bigger regression than site 1.

  It's also not a real per-equilibrium guard today: it's a global "not
  implemented anywhere" flag hardcoded `True`, so an equilibrium that
  already carries nonzero `W_lmn` currently gets **silently wrong** bounce
  integrals from this function (`omega` is hardcoded to `0` right after the
  check) rather than an error.

**The fix is not a `NotImplementedError` gate** (blocks every proxy —
`Gamma_c`, effective ripple, anything using `Bounce1D/2D` — for any
generalized-toroidal equilibrium; unacceptable). The correct generalization:

- `zeta` is a *given* grid coordinate in this function, not solved for —
  only `theta` is unknown. Substituting `phi = zeta + omega(rho,theta,zeta)`
  into `alpha = theta_PEST - iota*phi`, `theta_PEST = theta + lambda(...)`:
  `alpha + iota*zeta = theta + lambda(theta,zeta) - iota*omega(theta,zeta)`.
  The RHS precompute `varepsilon = alpha + iota*zeta` is **unchanged** — no
  omega dependence. Only the root residual gains a second spectral term:
  `t + lambda(t,zeta) - iota*omega(t,zeta) - varepsilon = 0`.
  **Still one vectorized 1D root-find**, not two — this is exactly what the
  existing code comment ("Root finding for θₖ such that
  θₖ + (λ−ιω)(ρ,θₖ,ζ) - ε = 0") already anticipated, and `_partial_sum`
  already has the signature for it
  (`_partial_sum(lmbda, L_lmn, omega, W_lmn, iota)`,
  [coords.py:410](../../desc/equilibrium/coords.py#L410)) — it's called
  with `None, None, None` at
  [coords.py:559](../../desc/equilibrium/coords.py#L559) today and just
  needs wiring.
- The `inbasis="vartheta"` branch *is* genuinely self-referential
  (`varepsilon = poloidal - iota*omega` needs `omega(theta,zeta)` before
  `theta` is known) and would need an alternating fixed-point solve, the
  same pattern as `_pest_phi_to_zeta`/`_phi_to_zeta_bisect` in
  `plotting.py`. It doesn't block anything: its only caller
  ([bounce_integral.py:588](../../desc/integrals/bounce_integral.py#L588),
  `name == "lambda"`) is already opt-in gated behind
  `ignore_lambda_guard=True` and isn't the default path. Leave it
  `errorif`'d for omega≠0 for now; implement later if/when needed.
- Bounce integral quadrature itself (`Bounce1D`/`Bounce2D`) needs **no**
  change: along a field line, `dl/B = dphi/(B.grad(phi)) = dzeta/(B.grad(zeta))`,
  so integrating in `zeta` is already correct as long as the metric
  quantities it consumes from `eq.compute(...)` include omega — which they
  already do elsewhere in this PR. Confirmed zero `OMEGA_IS_0` references
  in `bounce_integral.py`/`_bounce_utils.py`.

## Implementation order (easy → hard, one commit each)

1. **[cleanup] TODO wording** (#17) — rename the poincare-stuff joke to
   reference `@YigitElma` or a real tracking note.
2. **[cleanup] `curve.py` docstring** (#11) — update the class summary line.
3. **[cleanup] `curve.py` int casting** (#13) — `dtype=int` on `modes_R`/`modes_Z`
   to match `modes_W`.
4. **[cleanup] `plotting.py:2105` comment** (#24) — delete or clarify.
5. **[cleanup] `resolution_summary` always prints omega line** (#10).
6. **[reply, no code] `plotting.py:1656` branch-jump question** (#23) —
   post concrete failing examples on the thread.
7. **[rename] `Lz/Mz/Nz` → `Lw/Mw/Nw`** (#4) across `surface.py`, `curve.py`,
   `equilibrium.py`, `continuation.py`, `perturbations.py`,
   `initial_guess.py`, `test_generalized_toroidal.py`. (Verified other `Nz`
   hits in `vmec.py`/`tests/utils.py`/`test_profiles.py` are an unrelated
   grid zeta-count and must NOT be touched.)
8. **[consistency] `Mz`/`Nz` attribute presence + surface-setter growth
   asymmetry** (#6, #8, #9) — decide and apply one consistent rule for how
   `Equilibrium.surface = new_surface` reconciles resolution, for R/Z and
   omega alike.
9. **[io_attrs] Remove `_io_attrs_optional_`** (#5, #12, #15) — delete the
   mechanism from `hdf5_io.py`, `equilibrium.py`, `curve.py`; rely on
   existing `_set_up` defaulting + warning.
10. **[simplification] Always attach `W_basis`, drop `has_omega` gating**
    (#16) in `getters.py` and related call sites.
11. **[symmetry/gauge] "Full without constant" option or fix-to-zero gauge**
    (#7, #14) for the `W_basis`/axis `(0,0,0)` mode.
12. **[gauge, physics] `FixOmegaGauge` should fix all `n=0` modes, not just
    `(m=0,n=0)`** (#18) — the substantive correctness fix. Re-derive DOF
    counts against `BoundaryWSelfConsistency`/`FixOmegaInterior` afterward.
13. **[gauge, physics] Re-examine `FixOmegaInterior` necessity and the
    self-consistency/interior-fix split** (#19, #20) — numerical experiment
    (constraint Jacobian conditioning) before code changes.
14. **[numerics] Bracketed Newton for `_phi_to_zeta_bisect`** (#21, #22).
15. **[design] `A(zeta)` convention in `_geometry.py`** (#1) — benchmark,
    decide, document, then implement.
16. **[tests] Finite-difference derivative checks for omega-dependent
    `_compute` functions** (#25) — do after the compute functions this
    tests are stable from steps above.
17. **[OMEGA_IS_0, hardest, physics-critical] Wire omega into
    `_map_poloidal_coordinates`/`_partial_sum`** (#2, #3) as derived above;
    thread `W_lmn`/omega `Transform` through `_fast_ion.py`,
    `_neoclassical.py`, `bounce_integral.py`; delete `OMEGA_IS_0` from
    `backend.py`. Validate `Gamma_c`/effective-ripple on a nonzero-`W_lmn`
    equilibrium (e.g. the racetrack case) against a `phi`-parameterized
    cross-check using `_phi_to_zeta_bisect`, to catch a sign/Jacobian error
    in the new `-iota*omega` term.

Steps 1-6 have no interaction with each other or with anything below; they
can land in any order within that block. Steps 7 onward are ordered by
dependency and risk — each assumes the previous steps' tests are green.
