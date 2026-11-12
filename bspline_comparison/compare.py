"""Compare Fourier-Zernike and B-spline x double-Fourier equilibrium solves.

Usage: python compare.py NAME [spline_L] [maxiter]

Both equilibria use the boundary, profiles, M, N and angular grid resolution of the
DESC example NAME, start from the same default (scaled boundary) initial guess, and
are solved with the same ``eq.solve`` call. Results go to results_NAME.json/.npz.
"""

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import desc.examples  # noqa: E402
from desc.equilibrium import Equilibrium  # noqa: E402
from desc.equilibrium.coords import get_rtz_grid  # noqa: E402
from desc.grid import LinearGrid, QuadratureGrid  # noqa: E402

name = sys.argv[1]
spline_L = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "auto" else None
maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 200
degree = 5

ref = desc.examples.get(name)
common = dict(
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
)


def n_spline_modes(Ls):
    from desc.basis import BSplineDoubleFourierBasis

    return BSplineDoubleFourierBasis(
        Ls, ref.M, ref.N, ref.NFP, "cos" if ref.sym else False, degree
    ).num_modes


if spline_L is None:  # match number of degrees of freedom as closely as possible
    cands = np.arange(1, 40)
    spline_L = int(
        cands[np.argmin([abs(n_spline_modes(c) - ref.R_basis.num_modes) for c in cands])]
    )
# same number of collocation radii as the concentric grid (L_grid/2 + 1), rounded up
nq = int(np.ceil((ref.L_grid / 2 + 1) / spline_L))

eqs = {
    "zernike": Equilibrium(
        L=ref.L, L_grid=ref.L_grid, spectral_indexing=ref.spectral_indexing, **common
    ),
    "bspline": Equilibrium(
        L=spline_L,
        L_grid=nq * spline_L,
        radial_basis="bspline",
        spline_degree=degree,
        **common,
    ),
}

# common diagnostic grids
qgrid = QuadratureGrid(2 * ref.L_grid, 2 * ref.M_grid, 2 * ref.N_grid, ref.NFP)
rho = np.linspace(0.02, 1, 50)
fgrid = LinearGrid(rho=rho, M=2 * ref.M_grid, N=2 * ref.N_grid, NFP=ref.NFP)
# flux surface shapes are compared at fixed straight field line angle theta_PEST,
# since the computational poloidal angle is not unique in the plasma interior
g_rho = np.linspace(0.1, 1, 10)
g_theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
g_zeta = np.linspace(0, 2 * np.pi / ref.NFP, 6 if ref.N else 1, endpoint=False)


def diagnostics(eq):
    d = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol", "W"], grid=qgrid)
    f = eq.compute(["|F|", "iota", "sqrt(g)"], grid=fgrid)
    from desc.integrals import surface_averages

    Frho = surface_averages(fgrid, f["|F|"], f["sqrt(g)"], expand_out=False)
    ggrid = get_rtz_grid(eq, g_rho, g_theta, g_zeta, "rvp", jitable=False)
    g = eq.compute(["R", "Z"], grid=ggrid)
    norm = float(d["<|grad(|B|^2)|/2mu0>_vol"])
    return dict(
        F_vol=float(d["<|F|>_vol"]),
        F_norm=float(d["<|F|>_vol"]) / norm,
        W=float(d["W"]),
        F_rho=np.asarray(Frho) / norm,
        iota=np.asarray(fgrid.compress(f["iota"])),
        R=np.asarray(g["R"]),
        Z=np.asarray(g["Z"]),
        axis_R=float(eq.compute("R", grid=LinearGrid(rho=0.0))["R"][0]),
    )


out = {"name": name, "spline_L": spline_L, "spline_degree": degree}
arrs = {"rho": rho}
dref = diagnostics(ref)
out["reference"] = {k: v for k, v in dref.items() if np.isscalar(v)}
out["reference"]["num_modes"] = int(ref.R_basis.num_modes)
for k in ["F_rho", "iota", "R", "Z"]:
    arrs["reference_" + k] = dref[k]

for key, eq in eqs.items():
    from desc.objectives import ForceBalance, ObjectiveFunction

    obj = ObjectiveFunction(ForceBalance(eq))
    obj.build(verbose=0)
    nodes = obj.objectives[0].constants["transforms"]["grid"].num_nodes
    d0 = diagnostics(eq)
    t0 = time.time()
    _, res = eq.solve(
        objective=obj,
        ftol=1e-8,
        xtol=1e-12,
        gtol=1e-12,
        maxiter=maxiter,
        verbose=1,
    )
    t = time.time() - t0
    d = diagnostics(eq)
    eq.save(f"eq_{name}_{key}.h5")
    out[key] = {k: v for k, v in d.items() if np.isscalar(v)}
    out[key].update(
        L=int(eq.L),
        num_modes_R=int(eq.R_basis.num_modes),
        num_modes_total=int(
            eq.R_basis.num_modes + eq.Z_basis.num_modes + eq.L_basis.num_modes
        ),
        grid_nodes=int(nodes),
        F_norm_initial=d0["F_norm"],
        solve_time_s=t,
        iterations=int(res["nit"]),
        message=str(res["message"]),
        nested=bool(eq.is_nested()),
        dR_max=float(np.abs(d["R"] - dref["R"]).max()),
        dZ_max=float(np.abs(d["Z"] - dref["Z"]).max()),
        dR_rms=float(np.sqrt(np.mean((d["R"] - dref["R"]) ** 2))),
        dZ_rms=float(np.sqrt(np.mean((d["Z"] - dref["Z"]) ** 2))),
        diota_max=float(np.abs(d["iota"] - dref["iota"]).max()),
        dW_rel=float(abs(d["W"] - dref["W"]) / abs(dref["W"])),
    )
    for k in ["F_rho", "iota", "R", "Z"]:
        arrs[key + "_" + k] = d[k]
    print(key, json.dumps(out[key], indent=1), flush=True)

out["minor_radius"] = float(ref.compute("a")["a"])
json.dump(out, open(f"results_{name}.json", "w"), indent=1)
np.savez(f"results_{name}.npz", **arrs)
