"""Reads benchmark times and computes differences to comment on PR."""

import json
import os

import numpy as np

cwd = os.getcwd()

data = {}
master_idx = 0
latest_idx = 0
commit_ind = 0
for diret in os.walk(cwd + "/compare_results"):
    files = diret[2]
    timing_file_exists = False

    for filename in files:
        if filename.find("json") != -1:  # check if json output file is present
            try:
                filepath = os.path.join(diret[0], filename)
                with open(filepath) as f:
                    print(filepath)
                    curr_data = json.load(f)
                    commit_id = curr_data["commit_info"]["id"][0:7]
                    data[commit_id] = curr_data
                    if filepath.find("master") != -1:
                        master_idx = commit_ind
                    elif filepath.find("Latest_Commit") != -1:
                        latest_idx = commit_ind
                    commit_ind += 1
            except Exception as e:
                print(e)
                continue


# need arrays of size [ num benchmarks x num commits ]
# one for mean one for stddev
# number of benchmark cases
num_benchmarks = len(data[list(data.keys())[0]]["benchmarks"])
num_commits = len(list(data.keys()))
times = np.zeros([num_benchmarks, num_commits])
stddevs = np.zeros([num_benchmarks, num_commits])
commit_ids = []
test_names = [None] * num_benchmarks

for id_num, commit_id in enumerate(data.keys()):
    commit_ids.append(commit_id)
    for i, test in enumerate(data[commit_id]["benchmarks"]):
        t_mean = test["stats"]["median"]
        t_stddev = test["stats"]["iqr"]
        times[i, id_num] = t_mean
        stddevs[i, id_num] = t_stddev
        test_names[i] = test["name"]


# we say a slowdown/speedup has occurred if the mean time difference is greater than
# n_sigma * (stdev of time delta)
significance = 3  # n_sigmas of normal distribution, ie z score of 3
colors = [" "] * num_benchmarks  # g if faster, w if similar, r if slower
delta_times_ms = times[:, latest_idx] - times[:, master_idx]
delta_stds_ms = np.sqrt(stddevs[:, latest_idx] ** 2 + stddevs[:, master_idx] ** 2)
delta_times_pct = delta_times_ms / times[:, master_idx] * 100
delta_stds_pct = delta_stds_ms / times[:, master_idx] * 100
for i, (pct, spct) in enumerate(zip(delta_times_pct, delta_stds_pct)):
    if pct > 0 and pct > significance * spct:
        colors[i] = "-"  # this will make the line red
    elif pct < 0 and -pct > significance * spct:
        colors[i] = "+"  # this makes text green
    else:
        pass

# now make the commit message, save as a txt file
# benchmark_name dt(%) dt(s) t_new(s) t_old(s)
print(latest_idx)
print(master_idx)
commit_msg_lines = [
    "```diff",
    f"| {'benchmark_name':^38} | {'dt(%)':^22} | {'dt(s)':^22} |"
    + f" {'t_new(s)':^22} | {'t_old(s)':^22} | ",
    f"| {'-'*38} | {'-'*22} | {'-'*22} | {'-'*22} | {'-'*22} |",
]

for i, (dt, dpct, sdt, sdpct) in enumerate(
    zip(delta_times_ms, delta_times_pct, delta_stds_ms, delta_stds_pct)
):

    line = f"{colors[i]:>1}{test_names[i]:<39} |"
    line += f" {f'{dpct:+6.2f} +/- {sdpct:4.2f}':^22} |"
    line += f" {f'{dt:+.2e} +/- {sdt:.2e}':^22} |"
    line += f" {f'{times[i, latest_idx]:.2e} +/- {stddevs[i, latest_idx]:.1e}':^22} |"
    line += f" {f'{times[i, master_idx]:.2e} +/- {stddevs[i, master_idx]:.1e}':^22} |"

    commit_msg_lines.append(line)

commit_msg_lines.append("```")
commit_msg_lines = [line + "\n" for line in commit_msg_lines]
print("".join(commit_msg_lines))
# write out commit msg

fname = "commit_msg.txt"
with open(fname, "w+") as f:
    f.writelines(commit_msg_lines)
