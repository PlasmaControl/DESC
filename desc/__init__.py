"""DESC: a 3D MHD equilibrium solver and stellarator optimization suite."""

import importlib
import os
import platform
import re
import subprocess
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


config = {"devices": None, "avail_mem": None, "kind": None, "num_device": None}


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
                return re.sub(".*model name.*:", "", line, 1)
    return ""


def _set_cpu_count(n):
    """Set the number of CPUs visible to JAX.

    By default, JAX sees the whole CPU as a single device, regardless of the number of
    cores or threads. It then uses multiple cores and threads for lower level
    parallelism within individual operations.

    Alternatively, you can force JAX to expose a given number of "virtual" CPUs that
    can then be used manually for higher level parallelism (as in at the level of
    multiple objective functions.)

    This function is mainly for testing on CI purposes of the parallelism in DESC.

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


def set_device(kind="cpu", gpuid=None, num_device=1):  # noqa: C901
    """Sets the device to use for computation.

    If kind==``'gpu'`` and a gpuid is specified, uses the specified GPU. If
    gpuid==``None`` or a wrong GPU id is given, checks available GPUs and selects the
    one with the most available memory.
    Respects environment variable CUDA_VISIBLE_DEVICES for selecting from multiple
    available GPUs.

    Notes
    -----
    This function must be called before importing anything else from DESC or JAX,
    otherwise it will have no effect.

    Parameters
    ----------
    kind : {``'cpu'``, ``'gpu'``}
        whether to use CPU or GPU.
    gpuid : int, optional
        GPU id to use. Default is None. Supported only when num_device is 1.
    num_device : int
        number of devices to use. Default is 1.

    """
    config["kind"] = kind
    config["num_device"] = num_device

    cpu_mem = psutil.virtual_memory().available / 1024**3  # RAM in GB
    cpu_info = _get_processor_name()
    config["cpu_info"] = f"{cpu_info} CPU"
    config["cpu_mem"] = cpu_mem

    if kind == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if num_device == 1:
            config["devices"] = [f"{cpu_info} CPU"]
            config["avail_mems"] = [cpu_mem]
        else:
            try:
                import jax

                jax_cpu = jax.devices("cpu")
                assert len(jax_cpu) == num_device
                config["devices"] = [f"{dev}" for dev in jax_cpu]
                config["avail_mems"] = [cpu_mem for _ in range(num_device)]
            except ModuleNotFoundError:
                raise ValueError(
                    "JAX not installed. Please install JAX to use multiple CPUs."
                    "Alternatively, set num_device=1 to use a single CPU."
                )

    elif kind == "gpu":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        import nvgpu

        try:
            devices = nvgpu.gpu_info()
        except FileNotFoundError:
            devices = []
        if len(devices) == 0:
            warnings.warn(colored("No GPU found, falling back to CPU", "yellow"))
            set_device(kind="cpu")
            return

        gpu_ids = [dev["index"] for dev in devices]
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_ids = [
                s for s in re.findall(r"\b\d+\b", os.environ["CUDA_VISIBLE_DEVICES"])
            ]
            gpu_ids = [i for i in cuda_ids if i in gpu_ids]
        if len(gpu_ids) == 0:
            warnings.warn(
                colored(
                    f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} did "
                    "not match any physical GPU "
                    f"(id={[dev['index'] for dev in devices]}), falling back to CPU",
                    "yellow",
                )
            )
            set_device(kind="cpu")
            return

        devices = [dev for dev in devices if dev["index"] in gpu_ids]
        memories = {dev["index"]: dev["mem_total"] - dev["mem_used"] for dev in devices}

        if num_device == 1:
            if gpuid is not None:
                if str(gpuid) in gpu_ids:
                    selected_gpu = next(
                        dev for dev in devices if dev["index"] == str(gpuid)
                    )
                else:
                    warnings.warn(
                        colored(
                            f"Specified gpuid {gpuid} not found, selecting GPU with "
                            "most memory",
                            "yellow",
                        )
                    )
            else:
                selected_gpu = max(
                    devices, key=lambda dev: dev["mem_total"] - dev["mem_used"]
                )
            devices = [selected_gpu]

        else:
            if num_device > len(devices):
                raise ValueError(
                    f"Requested {num_device} GPUs, but only {len(devices)} available"
                )
            if gpuid is not None:
                # TODO: implement multiple GPU selection
                raise ValueError("Cannot specify `gpuid` when requesting multiple GPUs")

        config["avail_mems"] = [
            memories[dev["index"]] / 1024 for dev in devices[:num_device]
        ]  # in GB
        config["devices"] = [
            f"{dev['type']} (id={dev['index']})" for dev in devices[:num_device]
        ]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(dev["index"]) for dev in devices[:num_device]
        )
