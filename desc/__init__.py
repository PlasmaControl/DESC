"""DESC: a 3D MHD equilibrium solver and stellarator optimization suite."""

import importlib
import os
import sys
import warnings

import colorama
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
    "set_device",
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


config = {"device": None, "avail_mem": None, "kind": None}


def _read_cgroup_value(directory, name, stat=False):
    """Integer in a cgroup file (or a field of memory.stat), None if absent."""
    try:
        with open(os.path.join(directory, "memory.stat" if stat else name)) as f:
            if not stat:
                value = f.read().strip()
                return None if value == "max" else int(value)
            for line in f:
                key, _, value = line.partition(" ")
                if key == name:
                    return int(value)
    except (OSError, ValueError):
        pass
    return None


def _cgroup_available_memory(proc="/proc/self/cgroup", root="/sys/fs/cgroup"):
    """Memory left under the cgroup memory limits of this process, in bytes.

    Batch systems (eg Slurm, PBS, HTCondor) and containers limit the memory of a job
    with a cgroup, which ``psutil.virtual_memory`` does not see: it reports the
    memory of the whole machine. Every limit on the process's cgroup and on its
    parents counts, against the usage of that cgroup without its inactive page
    cache, which the kernel can reclaim. Both cgroup v2 and the memory controller
    of cgroup v1 are read.

    Parameters
    ----------
    proc : str
        The file listing the cgroups of the process.
    root : str
        Mount point of the cgroup filesystem.

    Returns
    -------
    available : float
        Bytes left under the tightest limit, or ``inf`` if there is none.

    """
    try:
        with open(proc) as f:
            lines = f.read().splitlines()
    except OSError:
        return float("inf")
    available = float("inf")
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":  # cgroup v2
            base = root
            names = ("memory.max", "memory.current", "inactive_file")
        elif "memory" in controllers.split(","):  # cgroup v1 memory controller
            base = os.path.join(root, "memory")
            if not os.path.isdir(base):  # mounted together with other controllers
                base = os.path.join(root, controllers)
            names = ("memory.limit_in_bytes", "memory.usage_in_bytes")
            names += ("total_inactive_file",)
        else:
            continue
        base = os.path.normpath(base)
        directory = os.path.normpath(base + "/" + path.lstrip("/"))
        while directory.startswith(base):
            limit = _read_cgroup_value(directory, names[0])
            usage = _read_cgroup_value(directory, names[1])
            # cgroup v1 reports no limit as a number close to 2**63
            if limit is not None and limit < 2**60 and usage is not None:
                inactive = _read_cgroup_value(directory, names[2], stat=True) or 0
                available = min(available, max(limit - (usage - inactive), 0))
            if directory == base:
                break
            directory = os.path.dirname(directory)
    return available


def set_device(kind="cpu", gpuid=None):
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
    if kind == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        import psutil

        config["device"] = "CPU"
        # in a batch job or a container, its cgroup memory limit applies as well
        config["avail_mem"] = (
            min(psutil.virtual_memory().available, _cgroup_available_memory()) / 1024**3
        )
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
            if gpuid is not None:
                if not 0 <= int(gpuid) < len(visible):
                    raise ValueError(
                        f"gpuid {gpuid} is out of range for the visible GPUs {visible}"
                    )
                os.environ["JAX_CUDA_VISIBLE_DEVICES"] = str(int(gpuid))
            # physical id of the first GPU JAX will use
            idx = int(visible[0 if gpuid is None else int(gpuid)])
            handle = nvmlDeviceGetHandleByIndex(idx)
            # Use nvmlMemory_v2 to account for system-reserved memory
            mem = nvmlDeviceGetMemoryInfo(handle, version=nvmlMemory_v2)
            config["device"] = f"{nvmlDeviceGetName(handle)} (id={idx})"
            config["avail_mem"] = (mem.total - mem.used) / 1024**3
        finally:
            nvmlShutdown()
    if kind == "tpu":
        config["device"] = "TPU"
