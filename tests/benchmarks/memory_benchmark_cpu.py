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
    while proc.poll() is None:  # check if child still running
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
    while proc.poll() is None:  # check if child still running
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
    end = time.time() + 1.0
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

    # list of functions to benchmark
    # CI benchmark will be more valuable if the tests use at least couple GBs
    funs = [
        "test_objective_jac_w7x",
        "test_proximal_jac_w7x_with_eq_update",
        "test_proximal_freeb_jac",
        "test_proximal_freeb_jac_blocked",
        "test_proximal_freeb_jac_batched",
        "test_proximal_jac_ripple",
        "test_proximal_jac_ripple_spline",
        "test_eq_solve",
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

        # check if one of the processes failed
        if child.returncode != 0:
            print(
                f"ERROR: Subprocess for function {funs[i]} failed with "
                f"exit code {child.returncode}"
            )
            # Raising an exception will cause the main script to fail,
            # making the overall GitHub Actions job fail.
            raise subprocess.CalledProcessError(
                returncode=child.returncode, cmd=funs[i], output=None
            )
        # save the data
        # make sure memory usage is 0 somewhere and t starts at 0
        data[funs[i]] = {}
        data[funs[i]]["mem"] = np.array(mem) - min(mem)
        data[funs[i]]["t"] = np.array(t) - t[0]
        print(
            f"{funs[i]} used max {max(data[funs[i]]['mem']):.2f} MB and took "
            + f"{data[funs[i]]['t'][-1]:.2f} s",
            flush=True,
        )
        # wait a bit before starting the next function to release memory
        # just in case child process didn't yet.
        time.sleep(2)

    # save the data to a pickle file
    branch = sys.argv[1]  # master or pr
    with open(f"{branch}.pickle", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
