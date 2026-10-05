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
        config["avail_mem"] = psutil.virtual_memory().available / 1024**3
    else:
        os.environ.pop("JAX_PLATFORMS", None)
    if kind == "gpu":
        # so that the ids assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        visible = [i for i in visible if i.strip()]
        if gpuid is not None:
            if visible and not 0 <= int(gpuid) < len(visible):
                raise ValueError(
                    f"gpuid {gpuid} is out of range for "
                    f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
                )
            os.environ["JAX_CUDA_VISIBLE_DEVICES"] = str(int(gpuid))
        # pynvml namespace is exposed through nvidia-ml-py
        from pynvml import (
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlMemory_v2,
            nvmlShutdown,
        )

        # physical id of the first GPU JAX will use
        idx = 0 if gpuid is None else int(gpuid)
        idx = int(visible[idx]) if visible else idx
        nvmlInit()
        try:
            handle = nvmlDeviceGetHandleByIndex(idx)
            # Use nvmlMemory_v2 to account for system-reserved memory
            mem = nvmlDeviceGetMemoryInfo(handle, version=nvmlMemory_v2)
            config["device"] = f"{nvmlDeviceGetName(handle)} (id={idx})"
            config["avail_mem"] = (mem.total - mem.used) / 1024**3
        finally:
            nvmlShutdown()
