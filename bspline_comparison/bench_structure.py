"""Time a full DSHAPE spline solve with and without the radial structure.

Usage: python bench_structure.py L [iterations] [method]

Without ``method`` every method is run in its own process (so that the peak memory,
``ru_maxrss``, belongs to that method alone) and the results are written to
``bench_structure_L{L}.json``. Methods:

* ``dense``: dense Jacobian (one JVP per column), QR trust region step
* ``grouped``: grouped-column Jacobian expanded to a dense matrix, QR step
* ``banded``: grouped-column Jacobian kept compressed, block tridiagonal Cholesky
"""

import json
import resource
import subprocess
import sys
import time
import warnings


warnings.filterwarnings("ignore")

import desc.examples  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import QuadratureGrid  # noqa: E402
from desc.optimize._constraint_wrappers import LinearConstraintProjection  # noqa: E402

METHODS = {
    "dense": ("dense Jacobian, QR", False, "qr"),
    "grouped": ("grouped Jacobian, QR", True, "qr"),
    "banded": ("compressed Jacobian, block Cholesky", True, "banded"),
}
Ls = int(sys.argv[1])
nit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
if len(sys.argv) <= 3:
    rows = []
    for m in METHODS:
        out = subprocess.run(
            [sys.executable, __file__, str(Ls), str(nit), m],
            capture_output=True,
            text=True,
        )
        lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
        if out.returncode or not lines:
            print(m, "failed with code", out.returncode, out.stderr[-2000:])
            continue
        rows.append(json.loads(lines[-1]))
        print(lines[-1], flush=True)
    json.dump(rows, open(f"bench_structure_L{Ls}.json", "w"), indent=1)
    sys.exit()

ref = desc.examples.get("DSHAPE")
grid = QuadratureGrid(2 * ref.L_grid, 2 * ref.M_grid, 0)
orig = LinearConstraintProjection._setup_structure


def no_structure(self, rescale_only=False):
    orig(self, rescale_only)
    self._structure_jac = False


label, structured, tr = METHODS[sys.argv[3]]
LinearConstraintProjection._setup_structure = orig if structured else no_structure
eq = Equilibrium(
    Psi=ref.Psi,
    sym=True,
    M=ref.M,
    N=0,
    M_grid=ref.M_grid,
    N_grid=0,
    surface=ref.surface.copy(),
    pressure=ref.pressure,
    iota=ref.iota,
    L=Ls,
    L_grid=4 * Ls,
    radial_basis="bspline",
    spline_axis_power="full",
)
# first short solve compiles everything, the second one is timed
eq.solve(maxiter=2, verbose=0, options={"tr_method": tr}, copy=True)
t0 = time.time()
_, res = eq.solve(
    ftol=0, xtol=0, gtol=0, maxiter=nit, verbose=0, options={"tr_method": tr}
)
t = time.time() - t0
d = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=grid)
row = dict(
    L=Ls,
    method=label,
    modes=int(eq.R_basis.num_modes),
    iterations=int(res["nit"]),
    time_s=t,
    s_per_iteration=t / max(int(res["nit"]), 1),
    F_norm=float(d["<|F|>_vol"] / d["<|grad(|B|^2)|/2mu0>_vol"]),
    # peak resident memory of this process (Linux reports kB)
    peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
)
print(json.dumps(row), flush=True)
