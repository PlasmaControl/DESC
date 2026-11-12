"""Time a full DSHAPE spline solve with and without the radial structure.

Usage: python bench_structure.py L [iterations]
"""

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import desc.examples  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import QuadratureGrid  # noqa: E402
from desc.optimize._constraint_wrappers import LinearConstraintProjection  # noqa: E402

Ls = int(sys.argv[1])
nit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ref = desc.examples.get("DSHAPE")
grid = QuadratureGrid(2 * ref.L_grid, 2 * ref.M_grid, 0)
orig = LinearConstraintProjection._setup_structure


def no_structure(self, rescale_only=False):
    orig(self, rescale_only)
    self._structure_jac = False


rows = []
for label, structured, tr in [
    ("dense Jacobian, QR", False, "qr"),
    ("grouped Jacobian, QR", True, "qr"),
    ("grouped Jacobian, banded", True, "banded"),
]:
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
    rows.append(
        dict(
            L=Ls,
            method=label,
            modes=int(eq.R_basis.num_modes),
            iterations=int(res["nit"]),
            time_s=t,
            s_per_iteration=t / max(int(res["nit"]), 1),
            F_norm=float(d["<|F|>_vol"] / d["<|grad(|B|^2)|/2mu0>_vol"]),
        )
    )
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(f"bench_structure_L{Ls}.json", "w"), indent=1)
