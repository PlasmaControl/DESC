"""Reads benchmark times and computes differences to comment on PR."""

import json
import os

import numpy as np

cwd = os.getcwd()

data = {}
master_idx = []
latest_idx = []
commit_ind = 0
folder_names = []

for root1, dirs1, files1 in os.walk(cwd):
    for dir_name in dirs1:
        if dir_name == "compare_results" or dir_name.startswith("benchmark_artifact"):
            print("Including folder: " + dir_name)
            # "compare_results" is the folder containing the benchmark results from this
            # job "benchmark_artifact" is the folder containing the benchmark results
            # from other jobs if in future we change the Python version of the
            # benchmarks, we will need to update this
            # "/Linux-CPython-<PYTHON_VERSION>-64bit"
            files2walk = (
                os.walk(cwd + "/" + dir_name)
                if dir_name == "compare_results"
                else os.walk(cwd + "/" + dir_name + "/Linux-CPython-3.12-64bit")
            )
            for root, dirs, files in files2walk:
                for filename in files:
                    if (
                        filename.find("json") != -1
                    ):  # check if json output file is present
                        try:
                            filepath = os.path.join(root, filename)
                            with open(filepath) as f:
                                curr_data = json.load(f)
                                commit_id = curr_data["commit_info"]["id"][0:7]
                                data[commit_ind] = curr_data["benchmarks"]
                                if filepath.find("master") != -1:
                                    master_idx.append(commit_ind)
                                elif filepath.find("Latest_Commit") != -1:
                                    latest_idx.append(commit_ind)
                                commit_ind += 1
                        except Exception as e:
                            print(e)
                            continue

# need arrays of size [ num benchmarks x num commits ]
# one for mean one for stddev
# number of benchmark cases
num_benchmarks = 0
# sum number of benchmarks splitted into different jobs
for split in master_idx:
    num_benchmarks += len(data[split])
num_commits = 2

times = np.zeros([num_benchmarks, num_commits])
stddevs = np.zeros([num_benchmarks, num_commits])
commit_ids = []
test_names = [None] * num_benchmarks

id_num = 0
for i in master_idx:
    for test in data[i]:
        t_mean = test["stats"]["median"]
        t_stddev = test["stats"]["iqr"]
        times[id_num, 0] = t_mean
        stddevs[id_num, 0] = t_stddev
        test_names[id_num] = test["name"]
        id_num = id_num + 1

id_num = 0
for i in latest_idx:
    for test in data[i]:
        t_mean = test["stats"]["median"]
        t_stddev = test["stats"]["iqr"]
        times[id_num, 1] = t_mean
        stddevs[id_num, 1] = t_stddev
        test_names[id_num] = test["name"]
        id_num = id_num + 1

# we say a slowdown/speedup has occurred if the mean time difference is greater than
# n_sigma * (stdev of time delta)
significance = 3  # n_sigmas of normal distribution, ie z score of 3
colors = [" "] * num_benchmarks  # g if faster, w if similar, r if slower
delta_times_ms = times[:, 1] - times[:, 0]
delta_stds_ms = np.sqrt(stddevs[:, 1] ** 2 + stddevs[:, 0] ** 2)
delta_times_pct = delta_times_ms / times[:, 0] * 100
delta_stds_pct = delta_stds_ms / times[:, 0] * 100
for i, (pct, spct) in enumerate(zip(delta_times_pct, delta_stds_pct)):
    if pct > 0 and pct > significance * spct:
        colors[i] = "-"  # this will make the line red
    elif pct < 0 and -pct > significance * spct:
        colors[i] = "+"  # this makes text green
    else:
        pass

# now make the commit message, save as a txt file
# benchmark_name dt(%) dt(s) t_new(s) t_old(s)
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
    line += f" {f'{times[i, 1]:.2e} +/- {stddevs[i, 1]:.1e}':^22} |"
    line += f" {f'{times[i, 0]:.2e} +/- {stddevs[i, 0]:.1e}':^22} |"

    commit_msg_lines.append(line)

commit_msg_lines.append("```")
commit_msg_lines = [line + "\n" for line in commit_msg_lines]
print("".join(commit_msg_lines))
# write out commit msg

fname = "commit_msg.txt"
with open(fname, "w+") as f:
    f.writelines(commit_msg_lines)
