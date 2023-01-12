"""DESC: a 3D MHD equilibrium solver and stellarator optimization suite."""

import os
import re
import sys
import warnings
import logging
import logging.handlers
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
                        "CUDA_VISIBLE_DEVICES={} ".format(
                            os.environ["CUDA_VISIBLE_DEVICES"]
                        )
                        + "did not match any physical GPU "
                        + "(id={}), falling back to CPU".format(
                            [dev["index"] for dev in devices]
                        )
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


logging.getLogger().addHandler( logging.NullHandler())
# Standard practice is to give the logging module a NullHandler such that no 
# output is generated without user intending it.  However, as DESC evolves we
# have a logger automatically set up to save crash data- can be turned 
# off with a call to set_logfile_logging(log_activated = False)

def stop_logfile_logging():
    """Quickly stops logging to specified file if it is being handled by a
    rotating file handler.   Non-global version of the stop_logfile_logging
    function from __init__.py for internal use.

    Returns
    -------
        bool :
            Returns True if logfile logger was successfully turned off.
    """
    logger = logging.getLogger("DESC_logger")
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            logger.removeHandler(handler)
            return True
    warnings.warn(colored("No DESC logger has been found.", "yellow"))
    return False


def set_logfile_logging(logfile_level = "DEBUG", logfile_file = "desc.log"):

    """Quickly adds a logfile handler to the root logger.  Non-global version 
    of the set_logfile_logging function from __init__.py for internal use.
    
    Arguments allow basic configuration of the logger, but this is not meant to
    be a replacement for setting up logging when using DESC in the context of a
    larger project- primarily meant for debugging.  Selecting a lower level of 
    logging- e.g. "INFO"- will print logs of "INFO" level or higher- "WARNING",
    "ERROR" and "CRITICAL" logs would be displayed in the same location as well.
    Accepts numeric inputs as well- levels "DEBUG", "INFO", "WARNING", "ERROR",
    "CRITICAL" correspond to 10, 20, 30, 40, and 50 respectively.
    
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
    bool : 
        Returns True if logging is set up, or stopped successfully.
    """

    #Creates(or overwrites) logger that accepts DEBUG level logs and up
    logfile_logger = logging.getLogger("DESC_logger")
    logfile_logger.setLevel(logfile_level)

    # Create file handler
    logfile_handler = logging.handlers.RotatingFileHandler(
                    logfile_file, 
                    maxBytes=5*1024*1024, 
                    backupCount=1, mode='w')
    try: 
        logfile_handler.setLevel(logfile_level)
    except:
        warnings.warn(
            colored(
                "Failed to set log file log level: invalid level: '%s'" 
                    % logfile_level),
                "yellow"
                )     
        return False

    # Set log formatting
    logfile_formatter = logging.Formatter(
        "%(asctime)s  ::  " + "%(name)s  ::  %(levelname)s ::  " +
        "File: %(module)s  Func: %(funcName)s  Line: %(lineno)s  :: %(message)s ",
        datefmt="%Y-%m-%d %H:%M:%S.uuu")

    # Assign formatter to handler
    logfile_handler.setFormatter(logfile_formatter)

    # Assign handler to root logger
    logfile_logger.addHandler(logfile_handler)

    return True


def set_console_logging(console_log_level = "INFO", console_log_output = "stdout"):
    """Quickly adds console handlers to python's root logger.  Non-global version 
    of the set_console_logging function from __init__.py for internal use.

    Arguments allow basic configuration of the logger, but this is not meant to
    be a replacement for setting up logging when using DESC in the context of a
    larger project- primarily meant for debugging.  Selecting a lower level of 
    logging- e.g. "INFO"- will print logs of "INFO" level or higher- "WARNING",
    "ERROR" and "CRITICAL" logs would be displayed in the same location as well.
    Accepts numeric inputs as well- levels "DEBUG", "INFO", "WARNING", "ERROR",
    "CRITICAL" correspond to 10, 20, 30, 40, and 50 respectively.  DESC only 
    uses DEBUG and INFO currently.
    
    Parameters
    ----------
    console_log_level : str
        level of logging to console; "DEBUG", "INFO", WARNING", "ERROR" or 
        "CRITICAL" are accepted values, in increasing order of severity
    console_log_output : str
        output logging to console with either "stdout" or "stderr"
    Returns
    -------
    bool : 
        Returns True if logging setup successfully
    """
           
    #Create logger that accepts DEBUG level logs and up
    logger = logging.getLogger("DESC_logger")
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    console_log_output = console_log_output.lower()
    if (console_log_output == "stdout"):
        console_handler = logging.StreamHandler(sys.stdout)
    elif (console_log_output == "stderr"):
        console_handler = logging.StreamHandler(sys.stderr)
    else:
        print("Console logging setup failed: invalid output: '%s'"%console_log_output)
        return False
    
    # Set console log level
    try:
        console_handler.setLevel(console_log_level.upper())
    except:
        print("Console logging setup failed: invalid level: '%s'"%console_log_level)
        return False

    # Set log formatting
    console_formatter = logging.Formatter(
        "%(asctime)s  ::  " + "%(name)s  ::  %(levelname)s ::  " +
        "File: %(module)s  Func: %(funcName)s  Line: %(lineno)s  :: %(message)s ",
        datefmt="%Y-%m-%d %H:%M:%S.uuu")

    console_handler.setFormatter(console_formatter)

    # Assign handlers to logger
    logger.addHandler(console_handler)

    return True

set_logfile_logging()
set_console_logging()

del RotatingFileHandler
