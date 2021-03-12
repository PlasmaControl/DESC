import os
from abc import ABC
from termcolor import colored
from .pickle_io import PickleReader, PickleWriter
from .hdf5_io import hdf5Reader, hdf5Writer


class IOAble(ABC):
    """Abstract Base Class for savable and loadable objects."""

    def _init_from_file_(self, load_from=None, file_format=None, obj_lib=None):
        """Initialize from file.

        Parameters
        ----------
        load_from : str or path-like or file instance (Default self.load_from)
            file to initialize from
        file_format : str (Default self._file_format_)
            file format of file initializing from

        """
        if load_from is None:
            raise RuntimeError(
                colored(
                    "_init_from_file_ should only be called when load_from is given",
                    "red",
                )
            )

        if file_format is None and isinstance(load_from, (str, os.PathLike)):
            name = str(load_from)
            if name.endswith(".h5") or name.endswith(".hdf5"):
                file_format = "hdf5"
            elif name.endswith(".pkl") or name.endswith(".pickle"):
                file_format = "pickle"
            else:
                raise RuntimeError(
                    colored(
                        "could not infer file format from file name, it should be provided as file_format",
                        "red",
                    )
                )

        reader = reader_factory(load_from, file_format)
        reader.read_obj(self, obj_lib=obj_lib)

    def save(self, file_name, file_format=None, file_mode="w"):
        """Save the object.

        Parameters
        ----------
        file_name : str file path OR file instance
            location to save object
        file_format : str (Default hdf5)
            format of save file. Only used if file_name is a file path
        file_mode : str (Default w - overwrite)
            mode for save file. Only used if file_name is a file path

        """
        if file_format is None:
            if isinstance(file_name, (str, os.PathLike)):
                name = str(file_name)
                if name.endswith(".h5") or name.endswith(".hdf5"):
                    file_format = "hdf5"
                elif name.endswith(".pkl") or name.endswith(".pickle"):
                    file_format = "pickle"
                else:
                    file_format = "hdf5"
            else:
                file_format = "hdf5"

        writer = writer_factory(file_name, file_format=file_format, file_mode=file_mode)
        writer.write_obj(self)
        writer.close()


def reader_factory(load_from, file_format):
    """Select and return instance of appropriate reader class for given file format.

    Parameters
    ----------
    load_from : str or file instance
        file path or instance from which to read
    file_format : str
        format of file to be read

    Returns
    -------
    Reader instance

    """
    if file_format == "hdf5":
        reader = hdf5Reader(load_from)
    elif file_format == "pickle":
        reader = PickleReader(load_from)
    else:
        raise NotImplementedError(
            "Format '{}' has not been implemented.".format(file_format)
        )
    return reader


def writer_factory(file_name, file_format, file_mode="w"):
    """Select and return instance of appropriate reader class for given file format.

    Parameters
    ----------
    load_from : str or file instance
        file path or instance from which to read
    file_format : str
        format of file to be read

    Returns
    -------
    Reader instance

    """
    if file_format == "hdf5":
        writer = hdf5Writer(file_name, file_mode)
    elif file_format == "pickle":
        writer = PickleWriter(file_name, file_mode)
    else:
        raise NotImplementedError(
            "Format '{}' has not been implemented.".format(file_format)
        )
    return writer
