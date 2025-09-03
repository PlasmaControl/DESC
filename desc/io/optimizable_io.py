"""Functions and methods for saving and loading equilibria and other objects."""

import copy
import os
import pickle
import pydoc
from abc import ABC, ABCMeta

import h5py
import numpy as np
from termcolor import colored

from desc.backend import register_pytree_node
from desc.utils import equals

from .hdf5_io import hdf5Reader, hdf5Writer
from .pickle_io import PickleReader, PickleWriter


def load(load_from, file_format=None):
    """Load any DESC object from previously saved file.

    Parameters
    ----------
    load_from : str or path-like or file instance
        file to initialize from
    file_format : {``'hdf5'``, ``'pickle'``} (Default: infer from file name)
        file format of file initializing from

    Returns
    -------
    obj :
        The object saved in the file

    """
    if file_format is None and isinstance(load_from, (str, os.PathLike)):
        name = str(load_from)
        load_from = os.path.expanduser(load_from)
        if name.endswith(".h5") or name.endswith(".hdf5"):
            file_format = "hdf5"
        elif name.endswith(".pkl") or name.endswith(".pickle"):
            file_format = "pickle"
        else:
            raise RuntimeError(
                colored(
                    (
                        "could not infer file format from file name, "
                        + "it should be provided as file_format"
                    ),
                    "red",
                )
            )

    if file_format == "pickle":
        with open(load_from, "rb") as f:
            obj = pickle.load(f)
    elif file_format == "hdf5":
        with h5py.File(load_from, "r") as f:
            if "__class__" in f.keys():
                cls_name = f["__class__"][()].decode("utf-8")
                cls = pydoc.locate(cls_name)
                obj = cls.__new__(cls)
                reader = reader_factory(load_from, file_format)
                reader.read_obj(obj)
                reader.close()
            else:
                raise ValueError(
                    "Could not load from {}, no __class__ attribute found".format(
                        load_from
                    )
                )
    else:
        raise ValueError("Unknown file format: {}".format(file_format))
    # to set other secondary stuff that wasn't saved possibly:
    if hasattr(obj, "_set_up"):
        obj._set_up()
    return obj


def _make_hashable(x):
    # turn unhashable ndarray of ints into a hashable tuple
    if isinstance(x, list):
        return [_make_hashable(y) for y in x]
    if isinstance(x, tuple):
        return tuple([_make_hashable(y) for y in x])
    if isinstance(x, dict):
        return {key: _make_hashable(val) for key, val in x.items()}
    if hasattr(x, "shape"):
        return ("ndarray", x.shape, tuple(x.flatten()))
    return x


def _unmake_hashable(x):
    # turn tuple of ints and shape to ndarray
    if isinstance(x, tuple) and x[0] == "ndarray":
        return np.array(x[2]).reshape(x[1])
    if isinstance(x, list):
        return [_unmake_hashable(y) for y in x]
    if isinstance(x, tuple):
        return tuple([_unmake_hashable(y) for y in x])
    if isinstance(x, dict):
        return {key: _unmake_hashable(val) for key, val in x.items()}
    return x


# this gets used as a metaclass, to ensure that all of the subclasses that
# inherit from IOAble get properly registered with JAX.
# subclasses can define their own tree_flatten and tree_unflatten methods to override
# default behavior
class _AutoRegisterPytree(type):
    def __init__(cls, *args, **kwargs):
        def _generic_tree_flatten(obj):
            """Convert DESC objects to JAX pytrees."""
            if hasattr(obj, "tree_flatten"):
                # use subclass method
                return obj.tree_flatten()

            # in jax parlance, "children" of a pytree are things like arrays etc
            # that get traced and can change. "aux_data" is metadata that is assumed
            # static and must be hashable. By default we assume floating point arrays
            # are children, and int/bool arrays are metadata that should be static
            children = {}
            aux_data = []

            static_attrs = getattr(obj, "_static_attrs", [])

            for key, val in obj.__dict__.items():
                if key in static_attrs:
                    aux_data += [(key, _make_hashable(val))]
                else:
                    children[key] = val

            return ((children,), aux_data)

        def _generic_tree_unflatten(aux_data, children):
            """Recreate a DESC object from JAX pytree."""
            if hasattr(cls, "tree_unflatten"):
                # use subclass method
                return cls.tree_unflatten(aux_data, children)

            obj = cls.__new__(cls)
            obj.__dict__.update(children[0])
            for kv in aux_data:
                setattr(obj, kv[0], _unmake_hashable(kv[1]))
            return obj

        register_pytree_node(cls, _generic_tree_flatten, _generic_tree_unflatten)
        super().__init__(*args, **kwargs)


# need this for inheritance to work correctly between the metaclass and ABC
# https://stackoverflow.com/questions/57349105/python-abc-inheritance-with-specified-metaclass
class _CombinedMeta(_AutoRegisterPytree, ABCMeta):
    pass


class IOAble(ABC, metaclass=_CombinedMeta):
    """Abstract Base Class for savable and loadable objects.

    Objects inheriting from this class can be saved and loaded via hdf5 or pickle.
    To save properly, each object should have an attribute ``_io_attrs_`` which
    is a list of strings of the object attributes or properties that should be
    saved and loaded.

    For saved objects to be loaded correctly, the ``__init__`` method of any custom
    types being saved should only assign attributes that are listed in ``_io_attrs_``.
    Other attributes or other initialization should be done in a separate
    ``set_up()`` method that can be called during ``__init__``. The loading process
    will involve creating an empty object, bypassing init, then setting any
    ``_io_attrs_`` of the object, then calling ``_set_up()`` without any arguments,
    if it exists.

    """

    @classmethod
    def load(cls, load_from, file_format=None):
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
                        (
                            "could not infer file format from file name, "
                            + "it should be provided as file_format"
                        ),
                        "red",
                    )
                )
        if isinstance(load_from, (str, os.PathLike)):  # load from top level of file
            self = load(load_from, file_format)
        else:  # being called from within a nested object
            self = cls.__new__(cls)  # create a blank object bypassing init
            reader = reader_factory(load_from, file_format)
            reader.read_obj(self)

            # to set other secondary stuff that wasn't saved possibly:
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
                file_name = os.path.expanduser(file_name)
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

    def __getstate__(self):
        """Helper method for working with pickle io."""
        if hasattr(self, "_io_attrs_"):
            return {
                attr: val
                for attr, val in self.__dict__.items()
                if attr in self._io_attrs_
            }
        return self.__dict__

    def __setstate__(self, state):
        """Helper method for working with pickle io."""
        self.__dict__.update(state)
        if hasattr(self, "_set_up"):
            self._set_up()

    def equiv(self, other):
        """Compare equivalence between DESC objects.

        Two objects are considered equivalent if they will be saved and loaded
        with the same data, (ie, they have the same data "where it counts",
        specifically, they have the same _io_attrs_)

        Parameters
        ----------
        other
            object to compare to

        Returns
        -------
        equiv : bool
            whether this and other are equivalent
        """
        if self.__class__ != other.__class__:
            return False
        if hasattr(self, "_io_attrs_"):
            dict1 = {
                key: val for key, val in self.__dict__.items() if key in self._io_attrs_
            }
            dict2 = {
                key: val
                for key, val in other.__dict__.items()
                if key in self._io_attrs_
            }
        else:
            dict1 = self.__dict__
            dict2 = other.__dict__
        return equals(dict1, dict2)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                setattr(result, k, copy.deepcopy(v, memo))
            except TypeError:
                setattr(result, k, copy.copy(v))
        return result

    def copy(self, deepcopy=True):
        """Return a (deep)copy of this object."""
        if deepcopy:
            new = copy.deepcopy(self)
        else:
            new = copy.copy(self)
        return new


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
