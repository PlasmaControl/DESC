"""Benchmark memory usage of various functions."""

import gc
import pickle
import subprocess
import sys
import threading
import time

import numpy as np
import psutil


def monitor_ram(proc, interval, ram_usage, timestamps):
    """Sample system RAM until *proc* finishes."""
    while proc.poll() is None:  # child still running?
        info = psutil.virtual_memory()
        used_mb = (info.total - info.available) / 1024 / 1024
        ram_usage.append(used_mb)
        timestamps.append(time.time())
        time.sleep(interval)

    # keep watching for an extra second
    end = time.time() + 1.0
    while time.time() < end:
        info = psutil.virtual_memory()
        ram_usage.append((info.total - info.available) / 1024 / 1024)
        timestamps.append(time.time())
        time.sleep(interval)


def monitor_vram(proc, interval, vram_usage, timestamps):
    """Sample total GPU memory until *proc* finishes."""
    while proc.poll() is None:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        vram_usage.append(int(out.split()[0]))
        timestamps.append(time.time())
        time.sleep(interval)

    # keep watching for an extra second
    end = time.time() + 2.0
    while time.time() < end:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        vram_usage.append(int(out.split()[0]))
        timestamps.append(time.time())
        time.sleep(interval)


if __name__ == "__main__":
    mode = "CPU"  # "CPU" or "GPU"
    interval = 0.1  # seconds between samples

    data = {}

    funs = [
        "test_objective_jac_atf",
        "test_proximal_jac_atf_with_eq_update",
        "test_perturb_2",
        "test_proximal_freeb_jac",
        "test_objective_jac_ripple",
        "test_objective_jac_ripple_spline",
    ]

    for i in range(len(funs)):
        mem = []
        t = []
        gc.collect()
        # start the sampler thread
        # launch the script to be profiled
        child = subprocess.Popen(["python", "memory_funcs.py", funs[i], mode])
        target = monitor_vram if mode == "GPU" else monitor_ram
        sampler = threading.Thread(
            target=target,
            args=(child, interval, mem, t),
            daemon=True,
        )
        sampler.start()

        # wait until the child exits, then join the sampler
        child.wait()
        sampler.join()
        data[funs[i]] = {}
        data[funs[i]]["mem"] = np.array(mem) - min(mem)
        data[funs[i]]["t"] = np.array(t) - t[0]  # to start at 0

    branch = sys.argv[1]  # master or pr
    with open(f"{branch}2.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
