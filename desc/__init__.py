from cmath import log
import colorama
import os
import re
import logging
from logging import NullHandler
import sys
import warnings
from termcolor import colored
# from ._version import get_versions

# __version__ = get_versions()["version"]
# del get_versions

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

def setup_logging(console_log_output = "stdout", console_log_level = "INFO", logfile_file = "desc.log", logfile_level = "DEBUG"):
	"""Creates a python logger and handlers to output to console and logfiles.

    Arguments allow basic configuration of the logger, but this is not meant to
    be a replacement for setting up logging when using DESC in the context of a
    larger project- primarily meant for debugging.  Selecting a lower level of 
    logging- e.g. "INFO"- will print logs of "INFO" level or higher- "WARNING",
    "ERROR" and "CRITICAL" logs would be displayed as well.
    
	Parameters
	----------
	console_log_output : str
		output logging to console with either "stdout" or "stderr"
	console_log_level : str
		level of logging to console; "DEBUG", "INFO", WARNING", "ERROR" or 
		"CRITICAL" are accepted values, in increasing order of severity
	logfile_file : str
		path to, and filename of, logfile to write output to
	logfile_level : str
		level of logging to logfile; "DEBUG", "INFO", WARNING", "ERROR" or 
		"CRITICAL" are accepted values, in increasing order of severity
	Returns
	-------
	bool : 
		Returns True if logging setup successfully
	"""
	
	#Create logger that accepts DEBUG level logs and up
	logger = logging.getLogger("desc_auto_logger")
	logger.setLevel(logging.DEBUG)
	
	# Create console handler
	console_log_output = console_log_output.lower()
	if (console_log_output == "stdout"):
		console_handler = logging.StreamHandler(sys.stdout)
	elif (console_log_output == "stderr"):
		console_handler = logging.StreamHandler(sys.stderr)
	else:
		print("Failed to set console output: invalid output: '%s'" % console_log_output)
		return False
	
	# Set console log level
	try:
		console_handler.setLevel(console_log_level.upper()) # only accepts uppercase level names
	except:
		print("Failed to set console log level: invalid level: '%s'" % console_log_level)
		return False
	console_handler.setFormatter(formatter)

	# Create file handler
	try:
		logfile_handler = logging.FileHandler(logfile_file)
	except Exception as exception:
		print("Failed to set up log file: %s" % str(exception))
		return False
	#Set logfile log level
	try: logfile_handler.setLevel(logfile_level.lower())
	except:
		print("Failed to set log file log level: invalid level: '%s'" % logfile_level)
		return False

	# Set log formatting
	formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")
	console_handler.setFormatter(formatter)
	logfile_handler.setFormatter(formatter)

	# Assign handlers to logger
	logger.addHandler(console_handler)
	logger.addHandler(logfile_handler)

	# Test messages
	logging.debug("Debug message")
	logging.info("Info message")
	logging.warning("Warning message")
	logging.error("Error message")
	logging.critical("Critical message")

	# Success returns True
	return True

del NullHandler
