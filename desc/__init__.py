"""DESC: a 3D MHD equilibrium solver and stellarator optimization suite."""

import importlib
import os
import platform
import re
import subprocess
import sys
import warnings

import colorama
import psutil
from termcolor import colored

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

colorama.init()


__all__ = [
    "basis",
    "coils",
    "compute",
    "continuation",
    "derivatives",
    "equilibrium",
    "examples",
    "geometry",
    "grid",
    "io",
    "magnetic_fields",
    "objectives",
    "optimize",
    "particles",
    "perturbations",
    "plotting",
    "profiles",
    "random",
    "transform",
    "vmec",
]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_BANNER = r"""
 ____  ____  _____   ___
|  _ \| ___|/  ___|/ ___|
| | \ | |_  | (__ | |
| | | |  _| \___ \| |
| |_/ | |__  ___) | |___
|____/|____||____/ \____|

"""

BANNER = colored(_BANNER, "magenta")

# mpi-cuda = True is not supported yet
config = {
    "devices": None,
    "avail_mems": None,
    "kind": None,
    "num_device": None,
    # Set to True if CUDA-aware MPI is installed (not tested)
    "mpi-cuda": False,
    # Suppress the warning in `desc.backend.safe_transfer_to_device`
    "SUPPRESS_GPU_MEMORY_WARNING": False,
}


def _get_processor_name():
    """Get the processor name of the current system."""
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
                return re.sub(pattern=".*model name.*:", repl="", string=line, count=1)
    return "CPU"


def _set_cpu_count(n):
    """Divide 1 physical CPU into multiple virtual CPUs.

    By default, JAX sees the whole CPU as a single device, regardless of the number of
    cores or threads. It then uses multiple cores and threads for lower level
    parallelism within individual operations.

    Alternatively, you can force JAX to expose a given number of "virtual" CPUs that
    can then be used manually for higher level parallelism (as in at the level of
    multiple objective functions.)

    This function is mainly for testing on CI purposes of the parallelism in DESC.
    It won't use multiple CPUs even if there are multiple CPUs available on the
    machine. It will just divide the first CPU into multiple virtual CPUs.

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


def set_device(kind="cpu", gpuid=None, num_device=1, mpi=None):  # noqa: C901
    """Sets the device to use for computation.

    Only affects JAX, the GPUs visible to non-JAX codes are not changed. Must be called
    before importing JAX or anything else from DESC.

    Parameters
    ----------
    kind : {``'cpu'``, ``'gpu'``, ``'tpu'``}
        Which device to use.
    gpuid : int, optional
        Index of the GPU to use among the visible ones. If ``None``, uses all visible
        GPUs.
    num_device : int, optional
        Number of devices to use. For `cpu`, this is the number of nodes.
        For `gpu`, this is equal to the number of GPUs connected to a single node.
        Default is 1.
    mpi : MPI object, optional
        MPI communicator. Used to get distinct CPU information for multi-node
        jobs where each rank runs on different node. Communicator is not used
        if the backend is ``'gpu'``. Supplying communicator doesn't
        change the computations, it can only change the output of
        ``desc.backend.print_backend_info()``.

    """
    if kind not in ["cpu", "gpu", "tpu"]:
        raise ValueError(f"kind must be 'cpu', 'gpu' or 'tpu', got {kind}")
    if "jax" in sys.modules and config["kind"] not in [None, kind]:
        warnings.warn(
            f"JAX has already been initialized on {config['kind']}, switching "
            f"to {kind} will have no effect. Call set_device before importing "
            "anything else from DESC."
        )

    config["kind"] = kind
    config["num_device"] = num_device

    cpu_mem = psutil.virtual_memory().available / 1024**3  # RAM in GB
    cpu_info = _get_processor_name()
    config["cpu_info"] = f"{cpu_info} CPU"
    config["cpu_mem"] = cpu_mem

    if kind == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        if num_device == 1:
            config["devices"] = [f"{cpu_info} CPU"]
            config["avail_mems"] = [cpu_mem]
        else:
            if mpi is None:
                warnings.warn(
                    "To get the full list of CPUs, provide the MPI communicator.",
                    UserWarning,
                )
                # return the same device multiple times
                cpu_names = [f"{i} {cpu_info}" for i in range(num_device)]
            else:
                comm = mpi.COMM_WORLD
                rank = comm.Get_rank()
                cpu_name = f"{rank} {cpu_info}"
                cpu_names = comm.allgather(cpu_name)
            config["devices"] = cpu_names
            # This memory is not individual but the total memory
            config["avail_mems"] = [cpu_mem] * num_device
    else:
        os.environ.pop("JAX_PLATFORMS", None)
    if kind == "gpu":
        # so that the ids assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # pynvml namespace is exposed through nvidia-ml-py
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlMemory_v2,
            nvmlShutdown,
        )

        nvmlInit()
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            visible = [i for i in visible if i.strip()]
            visible = visible or [str(i) for i in range(nvmlDeviceGetCount())]
            if gpuid is not None and num_device != 1:
                # TODO: implement multiple GPU selection
                raise ValueError("Cannot specify `gpuid` when requesting multiple GPUs")
            ids = list(range(num_device)) if gpuid is None else [int(gpuid)]
            if not 0 <= min(ids) <= max(ids) < len(visible):
                raise ValueError(
                    f"Requested GPUs {ids} are out of range for the visible GPUs "
                    f"{visible}"
                )
            if gpuid is not None or num_device > 1:
                os.environ["JAX_CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ids))
            config["devices"], config["avail_mems"] = [], []
            for i in ids:
                # physical id of the GPU JAX will use
                idx = int(visible[i])
                handle = nvmlDeviceGetHandleByIndex(idx)
                # Use nvmlMemory_v2 to account for system-reserved memory
                mem = nvmlDeviceGetMemoryInfo(handle, version=nvmlMemory_v2)
                config["devices"].append(f"{nvmlDeviceGetName(handle)} (id={idx})")
                config["avail_mems"].append((mem.total - mem.used) / 1024**3)
        finally:
            nvmlShutdown()
    if kind == "tpu":
        config["devices"] = ["TPU"]
