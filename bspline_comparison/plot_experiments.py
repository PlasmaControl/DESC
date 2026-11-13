"""Force error and flux surfaces for runs of experiments.py.

Usage: python plot_experiments.py NAME:label,label ...
The first label of each case is drawn as the reference for the flux surfaces.
Writes comparison_experiments.png, or the file given with -o FILE.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_OUT = "comparison_experiments.png"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c4359c", "#8a6d1f"]
NICE = {
    "zernike": "Fourier-Zernike re-solve",
    "bspline_base": "B-spline, first attempt",
    "apfull": "B-spline, rho^m prefactor",
    "apfull_nq8": "B-spline, rho^m prefactor",
    "p7_apfull": "B-spline, prefactor, degree 7",
    "p3_L3": "B-spline, degree 3, 3 intervals",
    "p3_L4": "B-spline, degree 3, 4 intervals",
    "p3_L6": "B-spline, degree 3, 6 intervals",
    "p3_L8": "B-spline, degree 3, 8 intervals",
    "p3_L12": "B-spline, degree 3, 12 intervals",
    "p7_apfull_nq8": "B-spline, prefactor, degree 7",
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

args = sys.argv[1:]
out = DEFAULT_OUT
if "-o" in args:  # output file name
    i = args.index("-o")
    out = args[i + 1]
    del args[i : i + 2]
cases = [a.split(":") for a in args]
fig, axes = plt.subplots(2, len(cases), figsize=(4.8 * len(cases), 8.4), squeeze=False)
for j, (name, labels) in enumerate(cases):
    labels = labels.split(",")
    rows = {r["label"]: r for r in json.load(open(f"experiments_{name}.json"))}
    ref = np.load(f"exp_{name}_reference.npz")
    ax = axes[0, j]
    ax.semilogy(ref["rho"], ref["F_rho"], color="#8a8a85", ls="--", label="stock example")
    for c, lab in zip(COLORS, labels):
        d = np.load(f"exp_{name}_{lab}.npz")
        ax.semilogy(
            d["rho"], d["F_rho"], color=c,
            label=f"{NICE.get(lab, lab)} ({rows[lab]['modes']} R modes)",
        )
    ax.set_title(f"{name.split('@')[0]}: normalized force error", loc="left")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\langle|F|\rangle_{fsa}\ /\ \langle|\nabla B^2/2\mu_0|\rangle_{vol}$")
    ax.legend(frameon=False, fontsize=7.5, loc="best")

    # flux surfaces at zeta=0 and fixed straight field line angle: first vs last label
    ax = axes[1, j]
    for lab, c, lw, ls in [(labels[0], COLORS[0], 1.6, "-"), (labels[-1], "#eb6834", 0.9, "--")]:
        d = np.load(f"exp_{name}_{lab}.npz")
        nz = d["R"].size // 480
        R = d["R"].reshape(nz, 10, 48)[0]
        Z = d["Z"].reshape(nz, 10, 48)[0]
        for i in range(10):
            ax.plot(np.append(R[i], R[i, 0]), np.append(Z[i], Z[i, 0]), color=c, lw=lw,
                    ls=ls, label=NICE.get(lab, lab) if i == 0 else None)
    ax.set_aspect("equal")
    ax.set_title(r"flux surfaces at $\zeta=0$", loc="left")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=1)
fig.tight_layout()
fig.savefig(out, dpi=200)
