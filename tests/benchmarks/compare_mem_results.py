"""Load and compare memory usage of the master and PR branches."""

import os
import pickle

import matplotlib.pyplot as plt

with open("master2.pickle", "rb") as f:
    data_master = pickle.load(f)
with open("pr2.pickle", "rb") as f:
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
PNG = "compare.png"
plt.savefig(PNG, dpi=100)

# environment variable passed from workflow
run_id = os.environ.get("GITHUB_RUN_ID")
repo = os.environ.get("GITHUB_REPOSITORY")

# GitHub artifact URL (raw links are not stable unless served via Pages)
image_url = f"https://github.com/{repo}/actions/runs/{run_id}/artifacts"

# ---------- commit message ----------
msg = "### Memory benchmark result\n\n```diff\n"
msg += (
    f"| {'Test Name':^38} | {'%Δ':^12} | {'Master (MB)':^18} | "
    + "{'PR (MB)':^18} | {'Δ (MB)':^12} |\n"
)
msg += f"| {'-'*38} | {'-'*12} | {'-'*18} | {'-'*18} | {'-'*12} |\n"
for i, name in enumerate(data_master.keys()):
    peak_pr = data_pr[name]["mem"].max()
    peak_ma = data_master[name]["mem"].max()
    delta = peak_pr - peak_ma
    sign = "+" if delta >= 0 else "-"
    # only show color if the delta is significant
    color = " " if abs((delta / peak_ma) * 100) < 7 else "-" if delta >= 0 else "+"
    percent_change = sign + f"{abs((delta / peak_ma) * 100):.2f}" + " %"
    delta = sign + f"{abs(delta):.2f}"
    msg += (
        f"{color} {name:<38} | {percent_change:^12} | {peak_ma:^18.1f} | "
        + f"{peak_pr:^18.1f} | {delta:^12} |\n"
    )
msg += "```"
msg += (
    "\n\nFor the memory plots, go to the summary of `Memory Benchmarks` "
    + "workflow and download the artifact!\n"
)


with open("commit_msg.txt", "w") as fh:
    fh.write(msg)
