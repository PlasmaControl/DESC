import colorama
import os
import re
import logging
from logging import NullHandler
import warnings
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


config = {"device": None, "avail_mem": None, "kind": None}


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
    config["kind"] = kind
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

        try:
            devices = nvgpu.gpu_info()
        except FileNotFoundError:
            devices = []
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
        config["device"] = selected_gpu["type"] + " (id={})".format(
            selected_gpu["index"]
        )
        config["avail_mem"] = (
            selected_gpu["mem_total"] - selected_gpu["mem_used"]
        ) / 1024  # in GB
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])

logging.getLogger(__name__).addHandler(NullHandler())

def add_stderr_logger(level: int = logging.INFO) -> logging.StreamHandler:
    """
    Helper for quickly adding a StreamHandler to the base Python logger. 
    Defaults to DEBUG logging level. In increasing order of severity, the
    base Python logging options are DEBUG, INFO, WARNING, ERROR, and 
    CRITICAL, with NOTSET being used to silence all logging.
    """

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler

def add_stdout_logger(level: int = logging.ERROR) -> logging.StreamHandler:
    """
    Helper for quickly adding a StreamHandler to the base Python logger. 
    Defaults to DEBUG logging level. In increasing order of severity, the
    base Python logging options are DEBUG, INFO, WARNING, ERROR, and 
    CRITICAL, with NOTSET being used to silence all logging.
    """

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stdout logging handler to logger: %s", __name__)
    return handler

def add_file_logger(level: int = logging.DEBUG, filename: str = "desc.log") -> logging.FileHandler:
    """
    Helper for quickly adding a FileHandler to the base Python logger. Defaults
    to DEBUG logging level, and desc.log as the filename.  In increasing order
    of severity, the base Python logging options are DEBUG, INFO, 
    WARNING, ERROR, and CRITICAL, with NOTSET being used to silence logging.
    """
    
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler
