"""Reads benchmarks stored in .benchmarks folder and saves a commit msg with the comparison between the latest commit and the master branch"""
import json
import numpy as np
import os

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
num_benchmarks = len(
    data[list(data.keys())[0]]["benchmarks"]
)  # number of benchmark cases
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
delta_times_pct = delta_times_ms / times[:, master_commit_index] * 100
for i, pct in enumerate(delta_times_pct):
    if pct > 20:
        colors[i] = "-"  # this will make the line red
    elif pct < -20:
        colors[i] = "+"  # this makes text green
    else:
        pass

# now make the commit message, save as a txt file
# benchmark_name dt(%) dt(ms) t_new(ms) t_old(ms)
print(latest_commit_index)
print(master_commit_index)
commit_msg_lines = []

commit_msg_lines.append("```diff\n")

commit_msg_lines.append(
    f'{"benchmark_name":^30}\t{"dt(%)":>15}\t{"dt(ms)":>15}\t{"t_new(ms)":>15}\t{"t_old(ms)":>15}\n'
)
for i, (dt, dpct) in enumerate(zip(delta_times_ms, delta_times_pct)):

    line = f"{colors[i]:>2}{test_names[i]:<36}\t{dpct:8.8}\t{dt:8.8}\t{times[i, latest_commit_index]:8.4}\t{times[i, master_commit_index]:8.4}\n"

    commit_msg_lines.append(line)

commit_msg_lines.append("```")
# write out commit msg

fname = "commit_msg.txt"
with open(fname, "w+") as f:
    f.writelines(commit_msg_lines)
