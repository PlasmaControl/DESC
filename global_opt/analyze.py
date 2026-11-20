"""Summarize and plot the stage-one and stage-two global optimization runs.

Usage::

    python analyze.py            # tables + convergence / local-minima plots
    python analyze.py --fields   # also plot B.n maps and coil shapes (needs DESC)
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

# categorical slots in fixed order (validated reference palette), baseline neutral
COLORS = {
    "multistart": "#2a78d6",
    "basin": "#eb6834",
    "noisy": "#1baf7a",
    "cma": "#eda100",
    "cma_lowmode": "#e87ba4",
    "basin_explore": "#eb6834",
}
BASELINE_COLOR = "#7a7974"
LABELS = {
    "baseline": "Tutorial (single lsq-exact)",
    "converged": "Same start, tight tolerances",
    "multistart": "Multistart",
    "basin": "Basin hopping",
    "noisy": "Annealed noisy gradient",
    "cma": "CMA-ES + polish",
    "cma_lowmode": "CMA-ES (\\|n\\| <= 2 subspace) + polish",
    "basin_explore": "Basin hopping from example result (1 core)",
}
STAGE1_LABELS = {
    "baseline": "Example (continuation k = 1..3)",
    "multistart": "Multistart (perturbed initial boundary)",
    "basin": "Basin hopping",
    "basin_explore": "Basin hopping from example result (1 core)",
}
ORDER = ["baseline", "converged", "multistart", "basin", "noisy", "cma", "cma_lowmode"]
GLOBAL = ["multistart", "basin", "noisy", "cma", "cma_lowmode"]
TIGHT = ["baseline_tight", "multistart_tight", "basin_tight", "noisy_tight"]
TIGHT_GLOBAL = TIGHT[1:]


def _base(key):
    return key[: -len("_tight")] if key.endswith("_tight") else key


def label(key):
    return LABELS[_base(key)].replace("\\|", "|")


def color(key):
    return COLORS[_base(key)]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)


def load(stage):
    runs = {}
    for f in sorted(glob.glob(os.path.join(RES, stage, "*.json"))):
        d = json.load(open(f))
        if "smoke" in f:
            continue
        runs.setdefault(d["method"] + d.get("tag", ""), []).append(d)
    return runs


# ---------------------------------------------------------------------------
# stage two
# ---------------------------------------------------------------------------


def stage2_table(runs, keys=ORDER):
    rows = [
        "| Method | Seed | Best cost | Quadratic-flux term | <\\|B·n\\|>/<\\|B\\|> "
        "| Min coil-coil [m] | Min plasma-coil [m] | Max curvature (norm.) "
        "| Coil lengths [m] | Local solves | Wall [s] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in keys:
        for d in runs.get(m, []):
            x = d["metrics"]
            rows.append(
                f"| {LABELS[_base(m)]} | {d['seed']} | {d['best_cost']:.2f} "
                f"| {x['terms']['quadratic flux']:.2f} | {x['Bn_over_B']:.2e} "
                f"| {x['min_coil_coil_dist']:.3f} | {x['min_plasma_coil_dist']:.3f} "
                f"| {x['max_curvature_normalized']:.3f} "
                f"| {', '.join(f'{v:.2f}' for v in x['coil_lengths'])} "
                f"| {d['n_local_solves']} | {d['wall_time']:.0f} |"
            )
    return "\n".join(rows)


def plot_convergence(
    runs,
    fname,
    baseline_cost,
    ylabel,
    title,
    keys=GLOBAL,
    base_label="tutorial result",
    legend_loc="lower left",
):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    _style(ax)
    ax.axhline(baseline_cost, color=BASELINE_COLOR, lw=1.5, ls=(0, (4, 3)))
    ax.text(
        0.99,
        baseline_cost,
        " " + base_label,
        color="#52514e",
        fontsize=8.5,
        va="bottom",
        ha="right",
        transform=ax.get_yaxis_transform(),
    )
    for m in keys:
        for i, d in enumerate(runs.get(m, [])):
            tr = [t for t in d["trace"] if np.isfinite(t["best_cost"])]
            if not tr:
                continue
            t = [tt["time"] / 60 for tt in tr]
            c = [tt["best_cost"] for tt in tr]
            ax.step(
                t,
                c,
                where="post",
                color=color(m),
                lw=2,
                alpha=1.0 if i == 0 else 0.55,
                label=(label(m) if i == 0 else None),
            )
            ax.plot(t[-1], c[-1], "o", ms=5, color=color(m), mec="white", mew=1.2)
    if any(len(runs.get(m, [])) > 1 for m in keys):
        ax.plot([], [], color=BASELINE_COLOR, alpha=0.55, lw=2, label="lighter shade: seed 1")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock time [min]", color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    if legend_loc == "upper right":
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1, 0.92))
    else:
        ax.legend(frameon=False, fontsize=8.5, loc=legend_loc)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_minima(
    runs, fname, baseline_cost, xlabel, title, methods, labels=None, xmax=None
):
    """Strip plot of every local minimum; values above xmax are drawn at the edge."""
    fig, ax = plt.subplots(figsize=(7.6, max(2.4, 0.9 + 0.55 * len(methods))), dpi=150)
    _style(ax)
    ax.grid(False, axis="y")
    rng = np.random.default_rng(0)
    n_clipped = 0
    for j, m in enumerate(methods):
        vals = np.concatenate(
            [np.asarray(d["local_minima"], float) for d in runs.get(m, [])] or [[]]
        )
        vals = vals[np.isfinite(vals)]
        y = j + 0.18 * rng.uniform(-1, 1, vals.size)
        inside = vals <= xmax if xmax is not None else np.ones(vals.size, bool)
        ax.scatter(
            vals[inside], y[inside], s=22, color=color(m), edgecolor="white", lw=0.8, zorder=3
        )
        if (~inside).any():
            n_clipped += (~inside).sum()
            ax.scatter(
                np.full((~inside).sum(), xmax),
                y[~inside],
                s=26,
                marker=">",
                facecolor="none",
                edgecolor=color(m),
                lw=1.0,
                zorder=3,
            )
        if vals.size:
            ax.text(
                1.01,
                j,
                f"best {vals.min():.3g}",
                fontsize=8.5,
                color="#52514e",
                va="center",
                transform=ax.get_yaxis_transform(),
            )
    ax.axvline(baseline_cost, color=BASELINE_COLOR, lw=1.5, ls=(0, (4, 3)))
    ax.text(
        baseline_cost,
        len(methods) - 0.45,
        " single local solve",
        fontsize=8,
        color="#52514e",
        va="bottom",
    )
    labels = labels or {}
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(
        [labels.get(m, label(m)) for m in methods], fontsize=9, color="#0b0b0b"
    )
    ax.set_ylim(-0.6, len(methods) - 0.1)
    ax.set_xscale("log")
    if xmax is not None:
        ax.set_xlim(right=xmax * 1.8)
        xlabel += f"   (open triangles: > {xmax:g}, equilibrium solve diverged)"
    ax.set_xlabel(xlabel, color="#52514e", fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def stage2_fields(runs, variant="tutorial"):
    """B.n on the plasma boundary + coil shapes: tutorial-style result vs best found."""
    import sys

    sys.path.insert(0, HERE)
    from desc.grid import LinearGrid
    from desc.io import load as dload
    from stage2_precise_QA import setup

    sfx = "" if variant == "tutorial" else "_" + variant
    keys = GLOBAL if variant == "tutorial" else TIGHT_GLOBAL
    eq, _, _, _ = setup()
    base_run = runs["baseline" + sfx][0]
    base = dload(os.path.join(RES, "stage2", f"baseline{sfx}_seed0_coilset.h5"))
    best_run = min((d for m in keys for d in runs.get(m, [])), key=lambda d: d["best_cost"])
    best_key = best_run["method"] + best_run.get("tag", "")
    best = dload(os.path.join(RES, "stage2", f"{best_key}_seed{best_run['seed']}_coilset.h5"))
    grid = LinearGrid(M=60, N=60, NFP=1, endpoint=True)
    data = []
    for cs in (base, best):
        Bn, _ = cs.compute_Bnormal(eq.surface, eval_grid=grid)
        data.append(np.asarray(Bn).reshape(grid.num_zeta, grid.num_theta))
    vmax = max(np.abs(d).max() for d in data)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)
    th = grid.nodes[grid.unique_theta_idx, 1]
    ze = grid.nodes[grid.unique_zeta_idx, 2]
    titles = [
        f"Tutorial-style single solve (cost {base_run['best_cost']:.1f})",
        f"Best found: {label(best_key)}, seed {best_run['seed']} "
        f"(cost {best_run['best_cost']:.1f})",
    ]
    for ax, d, t in zip(axs[:2], data, titles):
        im = ax.pcolormesh(ze, th, d.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlabel(r"$\zeta$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(t, fontsize=10, loc="left")
    fig.colorbar(im, ax=axs[:2], label=r"$B\cdot n$ [T]", shrink=0.9)
    ax = axs[2]
    cg = LinearGrid(N=100, endpoint=True)
    for cs, col, lab in (
        (base, BASELINE_COLOR, "single solve"),
        (best, color(best_key), "best found"),
    ):
        # all coils including the stellarator-symmetric / field-period copies
        xyz = np.asarray(cs._compute_position(grid=cg, basis="xyz"))
        for k, xi in enumerate(xyz):
            ax.plot(xi[:, 0], xi[:, 1], color=col, lw=1.2, label=lab if k == 0 else None)
    pg = LinearGrid(rho=np.array([1.0]), theta=np.array([0.0, np.pi]), N=128, NFP=1)
    eqd = eq.compute(["X", "Y"], grid=pg)
    for th in (0.0, np.pi):
        sel = np.isclose(pg.nodes[:, 1], th)
        ax.plot(
            np.append(eqd["X"][sel], eqd["X"][sel][0]),
            np.append(eqd["Y"][sel], eqd["Y"][sel][0]),
            color="#0b0b0b",
            lw=0.8,
            ls=":",
            label="plasma (inboard/outboard)" if th == 0 else None,
        )
    ax.set_aspect("equal")
    ax.set_title("Coils, top view", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)
    ax.axis("off")
    fig.savefig(os.path.join(FIG, f"stage2{sfx}_Bn_and_coils.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# stage one
# ---------------------------------------------------------------------------


def stage1_table(runs):
    rows = [
        "| Method | Seed | Best cost | QS term | Aspect term | iota term "
        "| Aspect ratio | iota (edge) | Local optimizations | Wall [s] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in ["baseline", "multistart", "basin", "basin_explore"]:
        for d in runs.get(m, []):
            b = d["best"]
            rows.append(
                f"| {STAGE1_LABELS[m]} | {d['seed']} | {b['cost']:.3e} | {b['qs']:.3e} "
                f"| {b['aspect']:.1e} | {b['iota']:.1e} | {b['aspect_ratio']:.3f} "
                f"| {b['iota_edge']:.3f} | {len(d['local_minima'])} | {d['wall_time']:.0f} |"
            )
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fields", action="store_true")
    args = p.parse_args()
    os.makedirs(FIG, exist_ok=True)
    out = []
    r2 = load("stage2")
    if r2:
        base = r2["baseline"][0]["best_cost"]
        out += ["## Stage two: tutorial objective", "", stage2_table(r2), ""]
        plot_convergence(
            r2,
            os.path.join(FIG, "stage2_convergence.png"),
            base,
            "best objective found",
            "Stage two (precise QA coils): best cost vs. time",
        )
        plot_minima(
            r2,
            os.path.join(FIG, "stage2_local_minima.png"),
            base,
            "objective value of each local minimum (log)",
            "Stage two: local minima reached by each method",
            [m for m in GLOBAL if m in r2],
        )
        if args.fields:
            stage2_fields(r2)
        if "baseline_tight" in r2:
            base_t = r2["baseline_tight"][0]["best_cost"]
            out += [
                "## Stage two: tight coil-coil distance (>= 0.14 m) and length (<= 7 m)",
                "",
                stage2_table(r2, TIGHT),
                "",
            ]
            keys = [m for m in TIGHT_GLOBAL if m in r2]
            plot_convergence(
                r2,
                os.path.join(FIG, "stage2_tight_convergence.png"),
                base_t,
                "best objective found",
                "Stage two, tight distance/length bounds: best cost vs. time",
                keys=keys,
                base_label="single lsq-exact solve",
                legend_loc="upper right",
            )
            plot_minima(
                r2,
                os.path.join(FIG, "stage2_tight_local_minima.png"),
                base_t,
                "objective value of each local minimum (log)",
                "Stage two, tight bounds: local minima reached by each method",
                keys,
            )
            if args.fields and keys:
                stage2_fields(r2, "tight")
    r1 = load("stage1")
    if r1:
        out += ["## Stage one", "", stage1_table(r1), ""]
        if "baseline" in r1:
            base1 = r1["baseline"][0]["best"]["cost"]
            keys = [m for m in ["multistart", "basin"] if m in r1]
            plot_convergence(
                r1,
                os.path.join(FIG, "stage1_convergence.png"),
                base1,
                "best objective found",
                "Stage one (precise QA boundary, M = N = 6): best cost vs. time",
                keys=keys,
                base_label="example result",
                legend_loc="upper right",
            )
            plot_minima(
                r1,
                os.path.join(FIG, "stage1_local_minima.png"),
                base1,
                "objective value of each local minimum (log)",
                "Stage one: local minima reached by each method",
                [m for m in ["multistart", "basin", "basin_explore"] if m in r1],
                labels=STAGE1_LABELS,
                xmax=1.0,
            )
    text = "\n".join(out)
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
