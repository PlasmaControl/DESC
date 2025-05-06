"""Load and compare memory usage of the master and PR branches."""

import os
import pickle

import matplotlib.pyplot as plt

if os.path.exists("master.pickle") and os.path.exists("pr.pickle"):
    with open("master.pickle", "rb") as f:
        data_master = pickle.load(f)
    with open("pr.pickle", "rb") as f:
        data_pr = pickle.load(f)

    # ---------- plot ----------
    num_tests = len(data_master.keys())
    fig, axes = plt.subplots(num_tests, 1, figsize=(12, 6 * num_tests), sharex=False)

    for i, (name, ax) in enumerate(zip(data_master.keys(), axes)):
        ax.plot(data_pr[name]["t"], data_pr[name]["mem"], "r", label="PR", lw=3)
        ax.plot(
            data_master[name]["t"],
            data_master[name]["mem"],
            "--b",
            label="Master",
            lw=1,
        )
        ax.set_title(name)
        ax.set_ylabel("Memory Usage [MB]")
        max_time = max(data_master[name]["t"][-1], data_pr[name]["t"][-1]) + 0.5
        ax.set_xlabel("Time [s]")
        ax.set_xlim([0, max_time])
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    png = "compare.png"
    plt.savefig(png, dpi=150)

    # ---------- commit message ----------
    msg = "### Memory benchmark result\n\n```diff\n"
    msg += (
        f"| {'Test Name':^38} | {'%Δ':^12} | {'Master (MB)':^18} | "
        + f"{'PR (MB)':^18} | {'Δ (MB)':^12} | {'Time PR (s)':^18} | "
        + f"{'Time Master (s)':^18} |\n"
    )
    msg += f"| {'-'*38} | {'-'*12} | {'-'*18} | {'-'*18} | {'-'*12} |"
    msg += f" {'-'*18} | {'-'*18} |\n"
    for i, name in enumerate(data_master.keys()):
        peak_pr = data_pr[name]["mem"].max()
        peak_ma = data_master[name]["mem"].max()
        delta = peak_pr - peak_ma
        # only show color if the delta is significant
        color = " " if abs((delta / peak_ma) * 100) < 10 else "-" if delta >= 0 else "+"
        percent_change = f"{(delta / peak_ma) * 100:.2f}" + " %"
        delta = f"{delta:.2f}"
        time_pr = data_pr[name]["t"][-1]
        time_ma = data_master[name]["t"][-1]
        msg += (
            f"{color} {name:<38} | {percent_change:^12} | {peak_ma:^18.3e} | "
            + f"{peak_pr:^18.3e} | {delta:^12} | {time_pr:^18.2f} | "
            + f"{time_ma:^18.2f} |\n"
        )

    msg += "```"
    msg += (
        "\n\nFor the memory plots, go to the summary of `Memory Benchmarks` "
        + "workflow and download the artifact.\n"
    )

    with open("commit_msg.txt", "w") as fh:
        fh.write(msg)
