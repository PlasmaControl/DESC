"""Variations of the B-spline solve: collocation points, knots, rho^m prefactor.

Usage:
    python experiments.py NAME LABEL key=value ...   run one spline configuration
    python experiments.py NAME --diagnose LABEL FILE  diagnostics of a saved equilibrium

Keys (defaults in brackets): L [DOF matched], nq [as compare.py] Gauss points per
radial interval, degree [5], axis_power [0] (int or "full"), first [1.0] size of the
first radial interval relative to uniform, maxiter [200], tr [qr].

Results are appended to experiments_NAME.json, profiles go to exp_NAME_LABEL.npz.
"""

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import desc.examples  # noqa: E402
import desc.io  # noqa: E402
from desc.basis import BSplineDoubleFourierBasis  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.equilibrium.coords import get_rtz_grid  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402
from desc.integrals import surface_averages  # noqa: E402
from desc.objectives import ForceBalance, ObjectiveFunction  # noqa: E402

name, label = sys.argv[1], sys.argv[2]
ref = desc.examples.get(name)

qgrid = QuadratureGrid(2 * ref.L_grid, 2 * ref.M_grid, 2 * ref.N_grid, ref.NFP)
rho = np.linspace(0.02, 1, 50)
fgrid = LinearGrid(rho=rho, M=2 * ref.M_grid, N=2 * ref.N_grid, NFP=ref.NFP)
g_rho = np.linspace(0.1, 1, 10)
g_theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
g_zeta = np.linspace(0, 2 * np.pi / ref.NFP, 6 if ref.N else 1, endpoint=False)


def diagnostics(eq):
    """Force error, profiles and flux surface shapes on grids common to all runs."""
    d = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=qgrid)
    f = eq.compute(["|F|", "iota", "current", "sqrt(g)"], grid=fgrid)
    ggrid = get_rtz_grid(eq, g_rho, g_theta, g_zeta, "rvp", jitable=False)
    g = eq.compute(["R", "Z"], grid=ggrid)
    norm = float(d["<|grad(|B|^2)|/2mu0>_vol"])
    axis = eq.compute(["R", "Z"], grid=LinearGrid(rho=0.0, NFP=eq.NFP))
    return dict(
        F_norm=float(d["<|F|>_vol"]) / norm,
        axis_R=float(axis["R"][0]),
        F_rho=np.asarray(surface_averages(fgrid, f["|F|"], f["sqrt(g)"], expand_out=False))
        / norm,
        iota=np.asarray(fgrid.compress(f["iota"])),
        current=np.asarray(fgrid.compress(f["current"])),
        R=np.asarray(g["R"]),
        Z=np.asarray(g["Z"]),
    )


_ref_cache = f"exp_{name}_reference.npz"
if os.path.exists(_ref_cache):
    dref = dict(np.load(_ref_cache))
    dref["F_norm"], dref["axis_R"] = float(dref["F_norm"]), float(dref["axis_R"])
else:
    dref = diagnostics(ref)
    np.savez(_ref_cache, rho=rho, **dref)


def summarize(d, extra):
    """Scalar summary of a run relative to the stock example, saved to disk."""
    # iota is the free profile when the current is prescribed, and vice versa
    free = "iota" if ref.iota is None else "current"
    scale = np.abs(dref[free]).max()
    row = dict(
        label=label,
        F_norm=d["F_norm"],
        dR_max=float(np.abs(d["R"] - dref["R"]).max()),
        dZ_max=float(np.abs(d["Z"] - dref["Z"]).max()),
        daxis_R=d["axis_R"] - dref["axis_R"],
        free_profile=free,
        dprofile_max_rel=float(np.abs(d[free] - dref[free]).max() / scale),
        **extra,
    )
    np.savez(f"exp_{name}_{label}.npz", rho=rho, **d)
    path = f"experiments_{name}.json"
    rows = json.load(open(path)) if os.path.exists(path) else []
    rows = [r for r in rows if r["label"] != label] + [row]
    json.dump(rows, open(path, "w"), indent=1)
    print(json.dumps(row), flush=True)


if sys.argv[3] == "--diagnose":
    eq = desc.io.load(sys.argv[4])
    summarize(
        diagnostics(eq),
        dict(basis=eq.radial_basis, L=int(eq.L), modes=int(eq.R_basis.num_modes)),
    )
    sys.exit()

opts = dict(a.split("=") for a in sys.argv[3:])
degree = int(opts.get("degree", 5))
axis_power = opts.get("axis_power", "0")
axis_power = axis_power if axis_power == "full" else int(axis_power)
first = float(opts.get("first", 1.0))
maxiter = int(opts.get("maxiter", 200))
tr = opts.get("tr", "qr")
sym = "cos" if ref.sym else False


def n_modes(Ls):
    return BSplineDoubleFourierBasis(
        Ls, ref.M, ref.N, ref.NFP, sym, degree, axis_power=axis_power
    ).num_modes


if "L" in opts:
    Ls = int(opts["L"])
else:  # closest number of modes to the Zernike basis of the example
    cands = np.arange(1, 40)
    Ls = int(cands[np.argmin([abs(n_modes(c) - ref.R_basis.num_modes) for c in cands])])
nq = int(opts.get("nq", np.ceil((ref.L_grid / 2 + 1) / Ls)))
# first interval is `first` times the uniform width, the others share the rest
knots = np.concatenate([[0.0], np.linspace(first / Ls, 1, Ls)]) if Ls > 1 else None

eq = Equilibrium(
    Psi=ref.Psi,
    NFP=ref.NFP,
    sym=ref.sym,
    M=ref.M,
    N=ref.N,
    M_grid=ref.M_grid,
    N_grid=ref.N_grid,
    surface=ref.surface.copy(),
    pressure=ref.pressure,
    iota=ref.iota,
    current=ref.current,
    L=Ls,
    L_grid=nq * Ls,
    radial_basis="bspline",
    spline_degree=degree,
    spline_knots=knots,
    spline_axis_power=axis_power,
)
obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=600)
obj.build(verbose=0)
nodes = obj.objectives[0].constants["transforms"]["grid"].num_nodes
t0 = time.time()
_, res = eq.solve(
    objective=obj,
    ftol=1e-8,
    xtol=1e-12,
    gtol=1e-12,
    maxiter=maxiter,
    verbose=2,
    options={"tr_method": tr},
)
t = time.time() - t0
eq.save(f"eq_{name}_{label}.h5")
summarize(
    diagnostics(eq),
    dict(
        basis="bspline",
        L=Ls,
        nq=nq,
        degree=degree,
        axis_power=str(axis_power),
        first=first,
        tr=tr,
        modes=int(eq.R_basis.num_modes),
        grid_nodes=int(nodes),
        solve_time_s=t,
        iterations=int(res["nit"]),
        message=str(res["message"]).split("\n")[-1],
    ),
)
