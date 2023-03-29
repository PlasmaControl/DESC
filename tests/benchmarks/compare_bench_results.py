"""Reads benchmark times and computes differences to comment on PR."""
import json
import os

import numpy as np

cwd = os.getcwd()

data = {}
master_commit_index = 0
latest_commit_index = 0
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
                        master_commit_index = commit_ind
                    elif filepath.find("Latest_Commit") != -1:
                        latest_commit_index = commit_ind
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
        t_mean = test["stats"]["mean"]
        t_stddev = test["stats"]["stddev"]
        times[i, id_num] = t_mean
        stddevs[i, id_num] = t_stddev
        test_names[i] = test["name"]


colors = [""] * num_benchmarks  # g if faster, w if similar, r if slower
delta_times_ms = times[:, latest_commit_index] - times[:, master_commit_index]
delta_stds_ms = np.sqrt(
    stddevs[:, latest_commit_index] ** 2 + stddevs[:, master_commit_index] ** 2
)
delta_times_pct = delta_times_ms / times[:, master_commit_index] * 100
delta_stds_pct = delta_stds_ms / times[:, master_commit_index] * 100
for i, (pct, spct) in enumerate(zip(delta_times_pct, delta_stds_pct)):
    if pct > 0 and pct > spct:
        colors[i] = "-"  # this will make the line red
    elif pct < 0 and -pct > spct:
        colors[i] = "+"  # this makes text green
    else:
        pass

# now make the commit message, save as a txt file
# benchmark_name dt(%) dt(s) t_new(s) t_old(s)
print(latest_commit_index)
print(master_commit_index)
commit_msg_lines = [
    "```diff",
    f"| {'benchmark_name':^38} | {'dt(%)':^19} | {'dt(s)':^19} | {'t_new(s)':^18} | {'t_old(s)':^18} | ",  # noqa: E501
    "| ------------------------------- | ------------------- | ------------------- | ------------------ | ------------------ |",  # noqa: E501
]

for i, (dt, dpct, sdt, sdpct) in enumerate(
    zip(delta_times_ms, delta_times_pct, delta_stds_ms, delta_stds_pct)
):

    line = f"{colors[i]:>1}{test_names[i]:<38} | "
    line += f"{dpct:+4.2f}+/-{sdpct:4.2f} | "
    line += f"{dt:+.2e}+/-{sdt:.1e} | "
    line += f"{times[i, latest_commit_index]:.2e}+/-{stddevs[i, latest_commit_index]:.1e} | "  # noqa: E501
    line += (
        f"{times[i, master_commit_index]:.2e}+/-{stddevs[i, master_commit_index]:.1e} |"
    )

    commit_msg_lines.append(line)

commit_msg_lines.append("```")
commit_msg_lines = [line + "\n" for line in commit_msg_lines]
print("".join(commit_msg_lines))
# write out commit msg

fname = "commit_msg.txt"
with open(fname, "w+") as f:
    f.writelines(commit_msg_lines)
