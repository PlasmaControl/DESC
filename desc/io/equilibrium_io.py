import os
from abc import ABC
from termcolor import colored
from .pickle_io import PickleReader, PickleWriter
from .hdf5_io import hdf5Reader, hdf5Writer


class IOAble(ABC):
    """Abstract Base Class for savable and loadable objects.

    Objects inheriting from this class can be saved and loaded via hdf5 or pickle.
    To save properly, each object should have an attribute `_io_attrs_` which
    is a list of strings of the object attributes or properties that should be
    saved and loaded. If any of these attributes are custom types (ie, not
    standard python containers or numpy arrays), an attribute called
    `_object_lib_` should also be defined. `_object_lib_` is a dictionary,
    where the keys are the class names of any custom types, and the values are
    instances of the class.

    For saved objects to be loaded correctly, the __init__ method of any custom
    types being saved should only assign attributes that are listed in `_io_attrs_`.
    Other attributes or other initialization should be done in a separate
    `set_up()` method that can be called during __init__. The loading process
    will involve creating an empty object, bypassing init, then setting any `_io_attrs_`
    of the object, then calling `_set_up()` without any arguments, if it exists.

    """

    @classmethod
    def load(cls, load_from, file_format=None, obj_lib=None):
        """Initialize from file.

        Parameters
        ----------
        load_from : str or path-like or file instance
            file to initialize from
        file_format : {``'hdf5'``, ``'pickle'``} (Default: infer from file name)
            file format of file initializing from

        """

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
        self = cls.__new__(cls)  # create a blank object bypassing init
        reader = reader_factory(load_from, file_format)
        reader.read_obj(self, obj_lib=obj_lib)

        # to set other secondary stuff that wasnt saved possibly:
        if hasattr(self, "_set_up"):
            self._set_up()

        return self

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
