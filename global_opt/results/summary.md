## Stage two: tutorial objective

| Method | Seed | Best cost | Quadratic-flux term | <\|B·n\|>/<\|B\|> | Min coil-coil [m] | Min plasma-coil [m] | Max curvature (norm.) | Coil lengths [m] | Local solves | Wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| Tutorial (single lsq-exact) | 0 | 197.88 | 197.87 | 9.16e-04 | 0.144 | 0.329 | 2.002 | 6.97, 5.42, 5.52 | 1 | 50 |
| Same start, tight tolerances | 0 | 197.87 | 197.86 | 9.16e-04 | 0.144 | 0.329 | 2.002 | 6.97, 5.42, 5.52 | 1 | 68 |
| Multistart | 0 | 16.43 | 16.43 | 3.91e-04 | 0.051 | 0.368 | 2.000 | 10.79, 6.33, 6.02 | 33 | 1225 |
| Multistart | 1 | 14.10 | 14.10 | 2.87e-04 | 0.096 | 0.367 | 2.000 | 8.72, 6.59, 6.21 | 31 | 1246 |
| Basin hopping | 0 | 12.38 | 11.86 | 3.15e-04 | 0.050 | 0.392 | 2.003 | 11.48, 7.58, 7.57 | 31 | 1228 |
| Basin hopping | 1 | 2.37 | 2.37 | 1.57e-04 | 0.051 | 0.331 | 2.000 | 12.27, 9.26, 7.96 | 33 | 1220 |
| Annealed noisy gradient | 0 | 4.41 | 4.41 | 1.95e-04 | 0.059 | 0.374 | 2.000 | 10.10, 6.82, 6.65 | 17 | 1364 |
| Annealed noisy gradient | 1 | 5.68 | 5.68 | 1.88e-04 | 0.073 | 0.358 | 2.000 | 9.49, 7.10, 7.25 | 13 | 1209 |
| CMA-ES + polish | 0 | 54.88 | 54.87 | 1.02e-03 | 0.051 | 0.420 | 2.000 | 12.36, 9.26, 8.25 | 4 | 1096 |
| CMA-ES (\|n\| <= 2 subspace) + polish | 0 | 35.77 | 35.52 | 5.50e-04 | 0.050 | 0.334 | 1.997 | 10.26, 6.64, 6.36 | 4 | 1380 |

## Stage two: tight coil-coil distance (>= 0.14 m) and length (<= 7 m)

| Method | Seed | Best cost | Quadratic-flux term | <\|B·n\|>/<\|B\|> | Min coil-coil [m] | Min plasma-coil [m] | Max curvature (norm.) | Coil lengths [m] | Local solves | Wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| Tutorial (single lsq-exact) | 0 | 197.88 | 197.87 | 9.16e-04 | 0.144 | 0.329 | 2.002 | 6.97, 5.42, 5.52 | 1 | 57 |
| Multistart | 0 | 69.96 | 67.22 | 4.92e-04 | 0.138 | 0.330 | 2.000 | 7.06, 6.05, 6.94 | 32 | 1211 |
| Basin hopping | 0 | 54.48 | 51.98 | 4.18e-04 | 0.137 | 0.329 | 2.000 | 7.05, 6.22, 6.96 | 31 | 1216 |
| Annealed noisy gradient | 0 | 68.49 | 65.87 | 4.66e-04 | 0.137 | 0.332 | 2.000 | 7.05, 6.37, 6.18 | 17 | 1224 |

## Stage one

| Method | Seed | Best cost | QS term | Aspect term | iota term | Aspect ratio | iota (edge) | Local optimizations | Wall [s] |
|---|---|---|---|---|---|---|---|---|---|
| Example (continuation k = 1..3) | 0 | 2.019e-04 | 1.974e-04 | 2.0e-09 | 4.5e-06 | 6.000 | 0.420 | 1 | 636 |
| Multistart (perturbed initial boundary) | 0 | 1.425e-04 | 1.357e-04 | 1.4e-10 | 6.7e-06 | 6.000 | 0.419 | 11 | 3650 |
| Basin hopping | 0 | 1.392e-04 | 1.378e-04 | 4.7e-10 | 1.4e-06 | 6.000 | 0.420 | 29 | 3696 |
| Basin hopping from example result (1 core) | 1 | 1.158e-04 | 1.131e-04 | 6.4e-10 | 2.7e-06 | 6.000 | 0.419 | 21 | 2541 |
