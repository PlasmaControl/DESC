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

    This only sets environment variables that JAX reads when it initializes, so it
    must be called before importing JAX or anything else from DESC.

    Parameters
    ----------
    kind : {``'cpu'``, ``'gpu'``, ``'tpu'``}
        Which device to use. On CPU, the accelerators are left untouched so that
        other codes can use them.
    gpuid : int, optional
        Index of the GPU to use, in ``nvidia-smi`` ordering, or an index into
        ``CUDA_VISIBLE_DEVICES`` if that is set. If ``None``, JAX will use every
        visible GPU, which is usually what you want on a cluster where the scheduler
        already assigned the GPUs.

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
    else:
        os.environ.pop("JAX_PLATFORMS", None)
        if kind == "gpu" and gpuid is not None:
            # so that the ids assigned by CUDA match those from nvidia-smi
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            visible = os.environ.get("JAX_CUDA_VISIBLE_DEVICES", "").split(",")
            visible = [i for i in visible if i]
            os.environ["JAX_CUDA_VISIBLE_DEVICES"] = (
                visible[int(gpuid)] if visible else str(int(gpuid))
            )
