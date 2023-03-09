"""DESC: a 3D MHD equilibrium solver and stellarator optimization suite."""

import logging
import logging.handlers
import os
import re
import sys
import warnings
from logging.handlers import RotatingFileHandler

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

        cpu_mem = psutil.virtual_memory().available / 1024**3  # RAM in GB
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
                    (
                        "CUDA_VISIBLE_DEVICES=%s " % os.environ["CUDA_VISIBLE_DEVICES"]
                        + "did not match any physical GPU "
                        + "(id=%s), falling back to CPU"
                        % [dev["index"] for dev in devices]
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
        config["device"] = selected_gpu["type"] + " (id=%s)" % selected_gpu["index"]
        config["avail_mem"] = (
            selected_gpu["mem_total"] - selected_gpu["mem_used"]
        ) / 1024  # in GB
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])


logging.getLogger().addHandler(logging.NullHandler())
# DESC automatically configures it's own logger, not the root logger, so we give the
# root logger a NullHandler.  DESC's logger config can be changed with the following
# functions, or with flags passed to any input-reading desc function as desired, with
# the 'quiet' (-q), or 'disable_logging' (-d) flags to prevent messages propogating
# into your own logging solution.  'disable logging' will toggle logfile logging off,
# and the 'quiet' command will turn off console logging.


def stop_logfile_logging():
    """Quickly stops logfile logging to DESC logger.

    Returns
    -------
        bool :
            Returns True if logfile logger was successfully turned off.
    """
    logger = logging.getLogger("DESC_logger")
    if logger.handlers is not None:
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel("CRITICAL")
                return True
            else:
                warnings.warn(
                    colored("No DESC logfile logger has been found.", "yellow")
                )
    else:
        warnings.warn(colored("No DESC logger has been found.", "yellow"))
        return False


def set_logfile_logging(logfile_level="DEBUG", logfile_file="desc.log"):
    """Quickly adds a logfile handler to the DESC logger.

    Parameters
    ----------
    logfile_file : str
        path to, and filename of, logfile to write output to
    logfile_level : str
        level of logging to logfile; "DEBUG", "INFO", WARNING", "ERROR" or
        "CRITICAL" are accepted values, in increasing order of severity. DESC
        only uses "DEBUG" and "INFO" currently.

    Returns
    -------
    logger : Logger object
        Returns Logger object if successfully configured
    """
    # Creates logger that accepts DEBUG level logs and up
    logger = logging.getLogger("DESC_logger")
    logger.setLevel(logfile_level)

    if logger.hasHandlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                logger.removeHandler(handler)

    # Create file handler
    logfile_handler = logging.handlers.RotatingFileHandler(
        logfile_file, maxBytes=5 * 1024 * 1024, backupCount=1, mode="w"
    )
    try:
        logfile_handler.setLevel(logfile_level)
    except ValueError:
        warnings.warn(
            colored(
                "Failed to set log file log level: invalid level: '%s'" % logfile_level,
                "yellow",
            )
            + "use 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', or an integer"
            + "value from 0 to 50"
        )
        return

    # Set log formatting
    logfile_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d :: "
        + "%(name)s :: %(levelname)s ::  "
        + "File: %(module)s  Func: %(funcName)s  Line: %(lineno)s  ::  %(message)s ",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Assign formatter to handler
    logfile_handler.setFormatter(logfile_formatter)

    # Assign handler to root logger
    logger.addHandler(logfile_handler)
    logging.captureWarnings(True)

    return logger


def set_console_logging(console_log_level="INFO", console_log_output="stdout"):
    """Quickly adds console handlers to python's root logger.

    Parameters
    ----------
    console_log_level : str
        level of logging to console; "DEBUG", "INFO", WARNING", "ERROR" or
        "CRITICAL" are accepted values, in increasing order of severity
    console_log_output : str
        output logging to console with either "stdout" or "stderr"
    Returns
    -------
    logger : Logger object
        Returns Logger object if successfully configured
    """
    # Create logger that accepts DEBUG level logs and up
    logger = logging.getLogger("DESC_logger")
    logger.setLevel("DEBUG")
    if logger.hasHandlers():
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

    # Create console handler
    console_log_output = console_log_output.lower()
    if console_log_output == "stdout":
        console_handler = logging.StreamHandler(sys.stdout)
    elif console_log_output == "stderr":
        console_handler = logging.StreamHandler(sys.stderr)
    else:
        print("Console logging setup failed: invalid output: '%s'" % console_log_output)
        return False

    # Set console log level
    try:
        console_handler.setLevel(console_log_level.upper())
    except ValueError:
        print("Console logging setup failed: invalid level: '%s'" % console_log_level)
        return

    # Set log formatting
    console_formatter = logging.Formatter(
        "%(name)s :: %(levelname)s :: "
        + "File: %(module)s Func: %(funcName)s Line: %(lineno)s ::  %(message)s ",
    )

    console_handler.setFormatter(console_formatter)

    # Assign handlers to logger
    logger.addHandler(console_handler)
    logging.captureWarnings(True)

    return logger


logger = set_logfile_logging()
logger = set_console_logging()

del RotatingFileHandler
