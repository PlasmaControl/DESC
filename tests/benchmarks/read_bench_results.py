"""Reads benchmarks stored in .benchmarks folder and saves a .png of the times for each benchmark case vs. commit id in the Figures folder"""
import matplotlib.pyplot as plt
import json
import numpy as np
import os

cwd = os.getcwd()

data = {}

for diret in os.walk(cwd + "/.benchmarks"):
    files = diret[2]
    timing_file_exists = False

    for filename in files:
        if filename.find("json") != -1:  # check if json output file is present
            try:
                filepath = os.path.join(diret[0], filename)
                with open(filepath) as f:
                    curr_data = json.load(f)
                    commit_id = curr_data["commit_info"]["id"][0:7]
                    data[commit_id] = curr_data

            except Exception as e:
                print(e)
                continue
## pytest benchmark_transform.py --benchmark-autosave

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
for i in range(num_benchmarks):
    plt.figure(i)
    plt.errorbar(range(len(commit_ids)), times[i, :], yerr=stddevs[i, :], marker="o")
    plt.xticks(range(len(commit_ids)), commit_ids)
    plt.title(test_names[i])
    plt.ylabel("Time (s)")
    plt.xlabel("Commit ID")
    plt.savefig(f"Figures/benchmark_{test_names[i]}.png")
