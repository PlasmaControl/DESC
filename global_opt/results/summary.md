## Stage two: tutorial objective

| Method | Seed | Best cost | Quadratic-flux term | <\|B·n\|>/<\|B\|> | Min coil-coil [m] | Min plasma-coil [m] | Max curvature (norm.) | Coil lengths [m] | Local solves | Wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| Tutorial (single lsq-exact) | 0 | 197.88 | 197.87 | 9.16e-04 | 0.144 | 0.329 | 2.002 | 6.97, 5.42, 5.52 | 1 | 50 |
| Same start, tight tolerances | 0 | 197.87 | 197.86 | 9.16e-04 | 0.144 | 0.329 | 2.002 | 6.97, 5.42, 5.52 | 1 | 68 |
| Multistart | 0 | 16.43 | 16.43 | 3.91e-04 | 0.051 | 0.368 | 2.000 | 10.79, 6.33, 6.02 | 33 | 1225 |
| Multistart | 1 | 14.10 | 14.10 | 2.87e-04 | 0.096 | 0.367 | 2.000 | 8.72, 6.59, 6.21 | 31 | 1246 |
| Basin hopping | 0 | 12.38 | 11.86 | 3.15e-04 | 0.050 | 0.392 | 2.003 | 11.48, 7.58, 7.57 | 31 | 1228 |
| Basin hopping | 1 | 2.37 | 2.37 | 1.57e-04 | 0.051 | 0.331 | 2.000 | 12.27, 9.26, 7.96 | 33 | 1220 |
| Basin hopping | 2 | 2.91 | 2.91 | 1.73e-04 | 0.053 | 0.382 | 2.000 | 11.76, 7.51, 6.86 | 23 | 1251 |
| Annealed noisy gradient | 0 | 4.41 | 4.41 | 1.95e-04 | 0.059 | 0.374 | 2.000 | 10.10, 6.82, 6.65 | 17 | 1364 |
| Annealed noisy gradient | 1 | 5.68 | 5.68 | 1.88e-04 | 0.073 | 0.358 | 2.000 | 9.49, 7.10, 7.25 | 13 | 1209 |
| Annealed noisy gradient | 2 | 8.97 | 8.97 | 2.36e-04 | 0.051 | 0.379 | 2.000 | 9.04, 6.73, 6.81 | 22 | 1330 |
| CMA-ES + polish | 0 | 54.88 | 54.87 | 1.02e-03 | 0.051 | 0.420 | 2.000 | 12.36, 9.26, 8.25 | 4 | 1096 |
| CMA-ES (\|n\| <= 2 subspace) + polish | 0 | 35.77 | 35.52 | 5.50e-04 | 0.050 | 0.334 | 1.997 | 10.26, 6.64, 6.36 | 4 | 1380 |
| Pool search without deflation (ablation) | 0 | 10.99 | 10.98 | 2.59e-04 | 0.051 | 0.359 | 2.000 | 11.37, 7.40, 8.18 | 42 | 1214 |
| Pool search without deflation (ablation) | 1 | 4.44 | 4.44 | 1.92e-04 | 0.052 | 0.340 | 2.000 | 10.58, 7.67, 7.12 | 34 | 1235 |
| Deflated pool search (new) | 0 | 2.37 | 2.37 | 1.60e-04 | 0.052 | 0.358 | 2.000 | 11.85, 7.35, 7.72 | 44 | 1211 |
| Deflated pool search (new) | 1 | 3.55 | 3.45 | 1.83e-04 | 0.050 | 0.369 | 2.000 | 13.88, 8.81, 7.80 | 81 | 1225 |
| Deflated pool search (new) | 2 | 2.51 | 2.51 | 1.76e-04 | 0.052 | 0.327 | 2.000 | 13.23, 7.26, 7.43 | 49 | 1233 |

## Stage two: pool search bookkeeping

| Method | Seed | Hops | Abandoned early | Polished | Entered pool | Duplicates | Best |
|---|---|---|---|---|---|---|---|
| Pool search without deflation (ablation) | 0 | 23 | 5 | 18 | 10 | 0 | 10.99 |
| Pool search without deflation (ablation) | 1 | 19 | 5 | 14 | 12 | 0 | 4.44 |
| Deflated pool search (new) | 0 | 25 | 7 | 18 | 13 | 0 | 2.37 |
| Deflated pool search (new) | 1 | 44 | 8 | 36 | 15 | 0 | 3.55 |
| Deflated pool search (new) | 2 | 26 | 4 | 22 | 16 | 0 | 2.51 |
| Deflated pool search (new), tight bounds | 0 | 32 | 9 | 23 | 17 | 0 | 29.07 |

## Stage two: tight coil-coil distance (>= 0.14 m) and length (<= 7 m)

| Method | Seed | Best cost | Quadratic-flux term | <\|B·n\|>/<\|B\|> | Min coil-coil [m] | Min plasma-coil [m] | Max curvature (norm.) | Coil lengths [m] | Local solves | Wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| Tutorial (single lsq-exact), tight bounds | 0 | 197.88 | 197.87 | 9.16e-04 | 0.144 | 0.329 | 2.002 | 6.97, 5.42, 5.52 | 1 | 57 |
| Multistart, tight bounds | 0 | 69.96 | 67.22 | 4.92e-04 | 0.138 | 0.330 | 2.000 | 7.06, 6.05, 6.94 | 32 | 1211 |
| Basin hopping, tight bounds | 0 | 54.48 | 51.98 | 4.18e-04 | 0.137 | 0.329 | 2.000 | 7.05, 6.22, 6.96 | 31 | 1216 |
| Annealed noisy gradient, tight bounds | 0 | 68.49 | 65.87 | 4.66e-04 | 0.137 | 0.332 | 2.000 | 7.05, 6.37, 6.18 | 17 | 1224 |
| Deflated pool search (new), tight bounds | 0 | 29.07 | 28.34 | 2.71e-04 | 0.138 | 0.317 | 2.000 | 7.01, 7.00, 7.00 | 56 | 1213 |

## Stage one

| Method | Seed | Best cost | QS term | Aspect term | iota term | Aspect ratio | iota (edge) | Local optimizations | Wall [s] |
|---|---|---|---|---|---|---|---|---|---|
| Example (continuation k = 1..3) | 0 | 2.019e-04 | 1.974e-04 | 2.0e-09 | 4.5e-06 | 6.000 | 0.420 | 1 | 636 |
| Multistart (perturbed initial boundary) | 0 | 1.425e-04 | 1.357e-04 | 1.4e-10 | 6.7e-06 | 6.000 | 0.419 | 11 | 3650 |
| Basin hopping | 0 | 1.392e-04 | 1.378e-04 | 4.7e-10 | 1.4e-06 | 6.000 | 0.420 | 29 | 3696 |
| Basin hopping from example result (1 core) | 1 | 1.158e-04 | 1.131e-04 | 6.4e-10 | 2.7e-06 | 6.000 | 0.419 | 21 | 2541 |

## Stage two with the coil frame fixed (rotmat, shift; DESC default)

| Method | Seed | Best cost | Quadratic-flux term | <\|B·n\|>/<\|B\|> | Min coil-coil [m] | Min plasma-coil [m] | Max curvature (norm.) | Coil lengths [m] | Local solves | Wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| Tutorial (single lsq-exact) | 0 | 31.00 | 31.00 | 3.66e-04 | 0.105 | 0.353 | 2.000 | 7.79, 6.24, 6.12 | 1 | 78 |
| Same start, tight tolerances | 0 | 31.00 | 31.00 | 3.66e-04 | 0.105 | 0.353 | 2.000 | 7.79, 6.24, 6.12 | 1 | 79 |
| Basin hopping | 0 | 5.91 | 5.91 | 2.30e-04 | 0.052 | 0.377 | 2.000 | 11.19, 6.67, 6.97 | 39 | 1210 |
| Basin hopping | 1 | 5.74 | 5.74 | 2.14e-04 | 0.051 | 0.373 | 2.000 | 10.45, 7.68, 7.11 | 37 | 1219 |
| Basin hopping | 2 | 2.44 | 2.41 | 1.56e-04 | 0.051 | 0.324 | 2.000 | 11.12, 7.16, 7.54 | 32 | 1228 |
| Annealed noisy gradient | 0 | 8.79 | 8.79 | 2.31e-04 | 0.077 | 0.360 | 2.000 | 8.71, 6.65, 6.59 | 20 | 1228 |
| Annealed noisy gradient | 1 | 16.27 | 16.27 | 2.83e-04 | 0.051 | 0.379 | 2.001 | 8.33, 6.60, 6.56 | 21 | 1221 |
| Annealed noisy gradient | 2 | 23.45 | 23.45 | 3.44e-04 | 0.057 | 0.372 | 2.001 | 8.08, 6.27, 6.16 | 28 | 1260 |
| Pool search without deflation (ablation) | 0 | 6.67 | 6.26 | 2.46e-04 | 0.049 | 0.395 | 2.000 | 11.63, 7.74, 7.52 | 62 | 1222 |
| Pool search without deflation (ablation) | 1 | 6.72 | 6.72 | 2.02e-04 | 0.070 | 0.355 | 2.001 | 8.99, 6.97, 6.95 | 53 | 1230 |
| Deflated pool search (new) | 0 | 2.96 | 2.96 | 1.50e-04 | 0.063 | 0.364 | 2.000 | 10.26, 7.61, 7.50 | 70 | 1221 |
| Deflated pool search (new) | 1 | 2.21 | 2.21 | 1.49e-04 | 0.057 | 0.345 | 2.000 | 11.40, 7.45, 7.87 | 60 | 1240 |
| Deflated pool search (new) | 2 | 2.51 | 2.51 | 1.51e-04 | 0.063 | 0.349 | 2.000 | 10.59, 7.20, 7.42 | 49 | 1238 |

### Free vs. fixed coil frame

| Method | Coil frame | Seed 0 | Seed 1 | Seed 2 | Worst seed |
|---|---|---|---|---|---|
| Tutorial (single lsq-exact) | free (rounds 1-2) | 197.88 | – | – | – |
| Tutorial (single lsq-exact) | fixed (DESC default) | 31.00 | – | – | – |
| Multistart | free (rounds 1-2) | 16.43 | 14.10 | – | 16.43 |
| Basin hopping | free (rounds 1-2) | 12.38 | 2.37 | 2.91 | 12.38 |
| Basin hopping | fixed (DESC default) | 5.91 | 5.74 | 2.44 | 5.91 |
| Annealed noisy gradient | free (rounds 1-2) | 4.41 | 5.68 | 8.97 | 8.97 |
| Annealed noisy gradient | fixed (DESC default) | 8.79 | 16.27 | 23.45 | 23.45 |
| Pool search without deflation (ablation) | free (rounds 1-2) | 10.99 | 4.44 | – | 10.99 |
| Pool search without deflation (ablation) | fixed (DESC default) | 6.67 | 6.72 | – | 6.72 |
| Deflated pool search (new) | free (rounds 1-2) | 2.37 | 3.55 | 2.51 | 3.55 |
| Deflated pool search (new) | fixed (DESC default) | 2.96 | 2.21 | 2.51 | 2.96 |

### Pool search bookkeeping (fixed frame)

| Method | Seed | Hops | Abandoned early | Polished | Entered pool | Duplicates | Best |
|---|---|---|---|---|---|---|---|
| Pool search without deflation (ablation) | 0 | 37 | 13 | 24 | 16 | 0 | 6.67 |
| Pool search without deflation (ablation) | 1 | 29 | 6 | 23 | 14 | 0 | 6.72 |
| Deflated pool search (new) | 0 | 38 | 7 | 31 | 21 | 0 | 2.96 |
| Deflated pool search (new) | 1 | 31 | 3 | 28 | 17 | 0 | 2.21 |
| Deflated pool search (new) | 2 | 24 | 0 | 24 | 15 | 0 | 2.51 |
| Deflated pool search (new), matched bounds | 0 | 34 | 1 | 33 | 16 | 0 | 9.10 |

## Stage two, fixed frame, matched bounds (coil-coil >= 0.105 m, length <= 7.8 m)

| Method | Seed | Best cost | Quadratic-flux term | <\|B·n\|>/<\|B\|> | Min coil-coil [m] | Min plasma-coil [m] | Max curvature (norm.) | Coil lengths [m] | Local solves | Wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| Tutorial (single lsq-exact), matched bounds | 0 | 31.00 | 31.00 | 3.66e-04 | 0.105 | 0.353 | 2.000 | 7.79, 6.24, 6.12 | 1 | 69 |
| Basin hopping, matched bounds | 0 | 14.24 | 14.14 | 2.56e-04 | 0.105 | 0.348 | 2.000 | 7.81, 6.64, 6.80 | 32 | 1253 |
| Deflated pool search (new), matched bounds | 0 | 9.10 | 9.06 | 1.92e-04 | 0.105 | 0.336 | 2.000 | 7.81, 7.16, 7.33 | 68 | 1218 |

## NAE-constrained equilibria (deflation tutorial problem)

| Method | Seed | Solutions | Valid | Distinct (δ = 0.01 / 0.02 / 0.04 m) | s per solution | Best QS error | Median QS error | ι edge range | Best f_B(ρ=1) |
|---|---|---|---|---|---|---|---|---|---|
| Deflation (tutorial Algorithm 1) | 0 | 69 | 68 | 55 / 54 / 39 | 26 | 9.58e-03 | 2.38e-02 | 0.378–0.473 | 1.71e-02 |
| Deflation, sum-reduction (paper eq. 4.4) | 0 | 58 | 58 | 38 / 18 / 6 | 31 | 9.30e-03 | 1.98e-02 | 0.421–0.460 | 1.62e-02 |
| Multistart (shape perturbations) | 0 | 91 | 75 | 75 / 70 / 32 | 20 | 7.14e-03 | 1.68e-02 | 0.355–0.434 | 1.26e-02 |
| Multistart (shape perturbations) | 1 | 88 | 73 | 73 / 70 / 36 | 20 | 8.15e-03 | 1.76e-02 | 0.375–0.433 | 1.37e-02 |
| Deflated multistart (new) | 0 | 68 | 68 | 68 / 64 / 22 | 27 | 4.24e-03 | 1.46e-02 | 0.386–0.440 | 6.99e-03 |
| Deflated multistart (new) | 1 | 68 | 67 | 67 / 64 / 17 | 27 | 5.36e-03 | 1.32e-02 | 0.372–0.436 | 8.71e-03 |
| QS-guided deflated pool (new) | 0 | 69 | 69 | 68 / 50 / 10 | 26 | 4.75e-03 | 1.02e-02 | 0.398–0.433 | 8.19e-03 |
| QS-guided deflated pool (new) | 1 | 67 | 67 | 65 / 45 / 10 | 27 | 3.91e-03 | 9.79e-03 | 0.400–0.433 | 6.72e-03 |
