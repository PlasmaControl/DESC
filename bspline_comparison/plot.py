"""Plot the Zernike vs B-spline comparison. Usage: python plot.py NAME [NAME ...]."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {"reference": "#8a8a85", "zernike": "#2a78d6", "bspline": "#eb6834"}
LABEL = {
    "reference": "stock example",
    "zernike": "Fourier-Zernike re-solve",
    "bspline": "B-spline x Fourier",
}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5df",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8a85",
        "lines.linewidth": 2,
    }
)

names = sys.argv[1:]
fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 8.2), squeeze=False)
for j, name in enumerate(names):
    d = np.load(f"results_{name}.npz")
    info = json.load(open(f"results_{name}.json"))
    ax = axes[0, j]
    for key in ["reference", "zernike", "bspline"]:
        lab = LABEL[key]
        if key != "reference":
            lab += f" ({info[key]['num_modes_R']} R modes)"
        ax.semilogy(d["rho"], d[key + "_F_rho"], color=C[key], label=lab,
                    ls="--" if key == "reference" else "-")
    ax.set_title(f"{name}: normalized force error", loc="left")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\langle|F|\rangle_{fsa}\ /\ \langle|\nabla B^2/2\mu_0|\rangle_{vol}$")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1, j]
    nz = 6 if info.get("has_N", d["zernike_R"].size > 480) else 1
    for key in ["zernike", "bspline"]:
        # nodes are ordered zeta (slowest), rho, theta (fastest) on the tensor grid
        R = d[key + "_R"].reshape(nz, 10, 48)[0]
        Z = d[key + "_Z"].reshape(nz, 10, 48)[0]
        for i in range(R.shape[0]):
            ax.plot(np.append(R[i], R[i, 0]), np.append(Z[i], Z[i, 0]), color=C[key],
                    lw=1.6 if key == "zernike" else 0.9,
                    ls="-" if key == "zernike" else "--",
                    label=LABEL[key] if i == 0 else None)
    ax.set_aspect("equal")
    ax.set_title(r"flux surfaces at $\zeta=0$", loc="left")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=2)
fig.tight_layout()
fig.savefig("comparison.png", dpi=200)

if os.path.exists("scan_DSHAPE.json"):
    rows = json.load(open("scan_DSHAPE.json"))
    z = json.load(open("results_DSHAPE.json"))["zernike"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cols = {3: "#1baf7a", 5: "#eb6834", 7: "#6250d6"}
    for p in sorted({r["degree"] for r in rows}):
        rr = sorted([r for r in rows if r["degree"] == p], key=lambda r: r["modes"])
        ax.loglog([r["modes"] for r in rr], [r["F_norm"] for r in rr], "o-",
                  color=cols[p], ms=7, label=f"B-spline, degree {p}")
    ax.loglog([z["num_modes_R"]], [z["F_norm"]], "s", color=C["zernike"], ms=9,
              label="Fourier-Zernike (L=26)")
    ax.set_xlabel("number of R modes")
    ax.set_ylabel("normalized volume-averaged force error")
    ax.set_title("DSHAPE: radial refinement of the spline basis", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("scan_DSHAPE.png", dpi=200)
