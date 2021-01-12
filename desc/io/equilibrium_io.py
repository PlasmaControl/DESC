from abc import ABC
from termcolor import colored
from .pickle_io import PickleReader, PickleWriter
from .hdf5_io import hdf5Reader, hdf5Writer


class IOAble(ABC):
    """Abstract Base Class for savable and loadable objects."""

    def _init_from_file_(
        self, load_from=None, file_format: str = None, obj_lib=None
    ) -> None:
        """Initialize from file.

        Parameters
        ----------
        load_from : str file path OR file instance (Default self.load_from)
            file to initialize from
        file_format : str (Default self._file_format_)
            file format of file initializing from

        Returns
        -------
        None

        """
        if load_from is None:
            raise RuntimeError(
                colored(
                    "_init_from_file_ should only be called when load_from is given",
                    "red",
                )
            )

        if file_format is None:
            raise RuntimeError(
                colored(
                    "file_format argument must be included when loading from file",
                    "red",
                )
            )

        reader = reader_factory(load_from, file_format)
        reader.read_obj(self, obj_lib=obj_lib)
        return None

    def save(self, file_name, file_format="hdf5", file_mode="w"):
        """Save the object.

        Parameters
        ----------
        file_name : str file path OR file instance
            location to save object
        file_format : str (Default hdf5)
            format of save file. Only used if file_name is a file path
        file_mode : str (Default w - overwrite)
            mode for save file. Only used if file_name is a file path

        Returns
        -------
        None

        """
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
