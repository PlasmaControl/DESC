"""Radial convergence of the B-spline basis on DSHAPE (force error vs number of intervals)."""
import json, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import desc.examples
from desc.equilibrium import Equilibrium
from desc.grid import QuadratureGrid, LinearGrid
from desc.integrals import surface_averages

ref = desc.examples.get("DSHAPE")
qgrid = QuadratureGrid(2 * ref.L_grid, 2 * ref.M_grid, 0, 1)
rho = np.linspace(0.02, 1, 50)
fgrid = LinearGrid(rho=rho, M=2 * ref.M_grid, N=0)
rows = []
prof = {"rho": rho}
for degree, Ls in [(5, 4), (5, 8), (5, 16), (5, 32), (3, 16), (7, 8), (7, 16)]:
    eq = Equilibrium(Psi=ref.Psi, NFP=1, sym=True, M=ref.M, N=0, M_grid=ref.M_grid, N_grid=0,
                     surface=ref.surface.copy(), pressure=ref.pressure, iota=ref.iota,
                     L=Ls, L_grid=max(4, degree - 1) * Ls, radial_basis="bspline", spline_degree=degree)
    t0 = time.time()
    _, res = eq.solve(ftol=1e-8, xtol=1e-12, gtol=1e-12, maxiter=400, verbose=0)
    t = time.time() - t0
    d = eq.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"], grid=qgrid)
    f = eq.compute(["|F|", "sqrt(g)"], grid=fgrid)
    norm = float(d["<|grad(|B|^2)|/2mu0>_vol"])
    prof[f"p{degree}_L{Ls}"] = np.asarray(surface_averages(fgrid, f["|F|"], f["sqrt(g)"], expand_out=False)) / norm
    rows.append(dict(degree=degree, L=Ls, modes=int(eq.R_basis.num_modes), F_norm=float(d["<|F|>_vol"]) / norm,
                     time=t, nit=int(res["nit"]), msg=str(res["message"]).split("\n")[-1]))
    print(rows[-1], flush=True)
json.dump(rows, open("scan_DSHAPE.json", "w"), indent=1)
np.savez("scan_DSHAPE.npz", **prof)
