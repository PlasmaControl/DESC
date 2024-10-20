"""Functions and classes for reading and writing DESC data."""

# InputReader lives outside this module for import ordering reasons, so we can
# import InputReader in __main__ without importing optimizable_io which imports JAX
# stuff potentially before we've set the GPU correctly.
# We include a link to it here for backwards compatibility
from desc.input_reader import InputReader

from .ascii_io import read_ascii, write_ascii
from .hdf5_io import hdf5Reader, hdf5Writer
from .optimizable_io import IOAble, load
from .pickle_io import PickleReader, PickleWriter

__all__ = ["InputReader", "load"]
