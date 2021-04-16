import colorama
import os
import re
import warnings
from termcolor import colored
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

colorama.init()

_BANNER = """
 ____  ____  _____   ___ 
|  _ \| ___|/  ___|/ ___|
| | \ | |_  | (__ | |
| | | |  _| \___ \| |
| |_/ | |__  ___) | |___ 
|____/|____||____/ \____|
                         
"""

BANNER = colored(_BANNER, "magenta")


config = {"device": None, "avail_mem": None}


def set_device(kind="cpu"):
    """Sets the device to use for computation.

    If kind==``'gpu'``, checks available GPUs and selects the one with the most
    available memory.
    Respects environment variable CUDA_VISIBLE_DEVICES for selecting from multiple
    available GPUs

    Parameters
    ----------
    kind : {``'cpu'``, ``'gpu'``}
        whether to use CPU or GPU.

    """
    if kind == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import psutil

        cpu_mem = psutil.virtual_memory().available / 1024 ** 3  # RAM in GB
        config["device"] = "CPU"
        config["avail_mem"] = cpu_mem

    if kind == "gpu":
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        import nvgpu

        devices = nvgpu.gpu_info()
        if len(devices) == 0:
            warnings.warn(colored("No GPU found, falling back to CPU", "yellow"))
            set_device(kind="cpu")
            return

        maxmem = 0
        selected_gpu = None
        gpu_ids = [dev["index"] for dev in devices]
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            cuda_ids = [
                s for s in re.findall(r"\b\d+\b", os.environ["CUDA_VISIBLE_DEVICES"])
            ]
            # check that the visible devices actually exist and are gpus
            gpu_ids = [i for i in cuda_ids if i in gpu_ids]
        if len(gpu_ids) == 0:
            # cuda visible devices = '' -> don't use any gpu
            warnings.warn(
                colored(
                    "CUDA_VISIBLE_DEVICES={} did not match any physical GPU (id={}), falling back to CPU".format(
                        os.environ["CUDA_VISIBLE_DEVICES"],
                        [dev["index"] for dev in devices],
                    ),
                    "yellow",
                )
            )
            set_device(kind="cpu")
            return
        devices = [dev for dev in devices if dev["index"] in gpu_ids]
        for dev in devices:
            mem = dev["mem_total"] - dev["mem_used"]
            if mem > maxmem:
                maxmem = mem
                selected_gpu = dev
        config["device"] = selected_gpu["type"] + " (id={})".format(dev["index"])
        config["avail_mem"] = (
            selected_gpu["mem_total"] - selected_gpu["mem_used"]
        ) / 1024  # in GB
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])
