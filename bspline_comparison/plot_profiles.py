"""Plot the free profile (iota or toroidal current) of each case against the stock example.

Usage: python plot_profiles.py [-o FILE] NAME:label,label ...
(labels from experiments_NAME.json; writes profiles.png or FILE)
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_OUT = "profiles.png"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c4359c", "#8a6d1f"]
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
NICE = {
    "zernike": "Fourier-Zernike re-solve",
    "bspline_base": "B-spline, first attempt",
    "apfull": "B-spline, rho^m prefactor",
    "apfull_nq6": "B-spline, rho^m prefactor",
    "apfull_nq8": "B-spline, rho^m prefactor",
    "p7_apfull": "B-spline, prefactor, degree 7",
    "p3_L3": "B-spline, degree 3, 3 intervals",
    "p3_L4": "B-spline, degree 3, 4 intervals",
    "p3_L6": "B-spline, degree 3, 6 intervals",
    "p3_L8": "B-spline, degree 3, 8 intervals",
    "p3_L12": "B-spline, degree 3, 12 intervals",
    "p7_apfull_nq6": "B-spline, prefactor, degree 7",
    "p7_apfull_nq8": "B-spline, prefactor, degree 7",
}

args = sys.argv[1:]
out = DEFAULT_OUT
if "-o" in args:  # output file name
    i = args.index("-o")
    out = args[i + 1]
    del args[i : i + 2]
cases = [a.split(":") for a in args]
fig, axes = plt.subplots(2, len(cases), figsize=(4.8 * len(cases), 7), squeeze=False)
for j, (name, labels) in enumerate(cases):
    ref = np.load(f"exp_{name}_reference.npz")
    rho = ref["rho"]
    # the free profile is the one that is not prescribed: current if iota is given
    free = json.load(open(f"experiments_{name}.json"))[0]["free_profile"]
    unit = 1e-3 if free == "current" else 1.0
    ylabel = "toroidal current (kA)" if free == "current" else r"rotational transform $\iota$"
    top, bot = axes[0, j], axes[1, j]
    top.plot(rho, ref[free] * unit, color="#8a8a85", ls="--", label="stock example")
    for c, lab in zip(COLORS, labels.split(",")):
        d = np.load(f"exp_{name}_{lab}.npz")
        nice = NICE.get(lab, "B-spline, " + lab.replace("_", " "))
        top.plot(rho, d[free] * unit, color=c, label=nice)
        bot.semilogy(rho, np.abs(d[free] - ref[free]) * unit + 1e-300, color=c, label=nice)
    top.set_title(f"{name.split(chr(64))[0]}: {ylabel}", loc="left", fontsize=10)
    top.set_xlabel(r"$\rho$")
    bot.set_title("absolute difference from stock example", loc="left", fontsize=10)
    bot.set_xlabel(r"$\rho$")
    bot.set_ylabel("kA" if free == "current" else r"$|\Delta\iota|$")
    top.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=200)
