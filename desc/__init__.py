"""DESC: a 3D MHD equilibrium solver and stellarator optimization suite."""

import os
import platform
import re
import subprocess
import warnings
from typing import NamedTuple

import colorama
from termcolor import colored

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

colorama.init()

_BANNER = r"""
 ____  ____  _____   ___
|  _ \| ___|/  ___|/ ___|
| | \ | |_  | (__ | |
| | | |  _| \___ \| |
| |_/ | |__  ___) | |___
|____/|____||____/ \____|

"""

BANNER = colored(_BANNER, "magenta")


class Device(NamedTuple):
    """Helper class to represent data about CPU/GPU."""

    name: str
    kind: str
    mem: float
    default: bool


config = {"devices": [], "backend": None, "dtype": None}

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def _get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


def set_cpu_count(n):
    """Set the number of CPUs visible to JAX.

    By default, JAX sees the whole CPU as a single device, regardless of the number of
    cores or threads. It then uses multiple cores and threads for lower level
    parallelism within individual operations.

    Alternatively, you can force JAX to expose a given number of "virtual" CPUs that
    can then be used manually for higher level parallelism (as in at the level of
    multiple objective functions.)

    Parameters
    ----------
    n : int
        Number of virtual CPUs for high level parallelism.

    Notes
    -----
    This function must be called before importing anything else from DESC or JAX,
    and before calling ``desc.set_device``, otherwise it will have no effect.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        [f"--xla_force_host_platform_device_count={n}"] + xla_flags
    )


def set_device(kind="cpu"):
    """Sets the default device type to use for computation.

    Respects environment variable CUDA_VISIBLE_DEVICES for selecting from multiple
    available GPUs

    Parameters
    ----------
    kind : {``'cpu'``, ``'gpu'``}
        whether to use CPU or GPU.

    Notes
    -----
    This function must be called before importing anything else from DESC or JAX,
    otherwise it will have no effect.

    """
    os.environ["JAX_PLATFORM_NAME"] = kind

    import psutil

    cpu_mem = psutil.virtual_memory().available / 1024**2  # RAM in MB
    # see if we're in a slurm job, in which case report allocated memory
    ntask = os.environ.get("SLURM_NTASKS", 1)
    cpupertask = os.environ.get("SLURM_CPU_PER_TASK", 1)
    mempercpu = os.environ.get("SLURM_MEM_PER_CPU", cpu_mem)  # in MB
    cpu_avail_mem = ntask * cpupertask * mempercpu / 1024  # put into GB
    try:
        import jax

        jax_cpu = jax.devices("cpu")
        n = len(jax_cpu)
    except ModuleNotFoundError:
        jax = None
        n = 1
    cpu = Device(
        _get_processor_name().strip() + f" (x{n})", "CPU", cpu_avail_mem, kind == "cpu"
    )
    config["devices"].append(cpu)

    import nvgpu

    try:
        gpus = nvgpu.gpu_info()
    except FileNotFoundError:
        gpus = []
    if (len(gpus) == 0) or (jax is None):
        if kind == "gpu":
            warnings.warn(colored("No GPU found, falling back to CPU", "yellow"))
            set_device(kind="cpu")
        return

    for i, (jaxgpu, gpuinfo) in enumerate(zip(jax.devices("gpu"), gpus)):
        mem = (gpuinfo["mem_total"] - gpuinfo["mem_used"]) / 1024  # in GB
        name = gpuinfo["type"] + " (id={})".format(gpuinfo["index"])
        device = Device(name, "GPU", mem, (i == 0) and (kind == "gpu"))
        config["devices"].append(device)
