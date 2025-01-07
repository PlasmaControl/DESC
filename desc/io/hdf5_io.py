"""Classes for reading and writing HDF5 files."""

import numbers
import pydoc
import warnings

import h5py
import numpy as np
from termcolor import colored

from .core_io import IO, Reader, Writer


def fullname(o):
    """Find where an object is defined."""
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


class hdf5IO(IO):
    """Class to wrap ABC IO for hdf5 file format."""

    def __init__(self):
        """Initialize hdf5IO instance."""
        self._file_types_ = (h5py._hl.group.Group, h5py._hl.files.File)
        self._file_format_ = "hdf5"
        super().__init__()

    def open_file(self, file_name, file_mode):
        """Open hdf5 file.

        Parameters
        ----------
        file_name : str
            path to file to open
        file_mode : str
            mode used when opening file

        Returns
        -------
        hdf5 file instance

        """
        return h5py.File(file_name, file_mode)

    def sub(self, name):
        """Create subgroup or return if already exists.

        Parameters
        ----------
        name : str
            name of subgroup

        Returns
        -------
        sub : subgroup instance

        """
        try:
            return self.base.create_group(name)
        except ValueError:
            return self.base[name]
        except KeyError:
            raise RuntimeError("Cannot create sub in reader.")

    def groups(self, where=None):
        """Find groups in location given by 'where'.

        Parameters
        ----------
        where : None or file instance

        Returns
        -------
        groups : list

        """
        loc = self.resolve_where(where)
        return list(loc.keys())


class hdf5Reader(hdf5IO, Reader):
    """Class specifying a Reader with hdf5IO."""

    def __init__(self, target):
        """Initialize hdf5Reader class.

        Parameters
        ----------
        target : str or file instance
            Path to file OR file instance to be read.

        """
        self.target = target
        self.file_mode = "r"
        super().__init__()

    def _decode_attr(self, loc, attr):

        if isinstance(loc[attr][()], bytes):
            s = loc[attr][()].decode("utf-8")
        else:
            s = loc[attr][()]

        # isinstance check to avoid comparing strings with numpy arrays
        if isinstance(s, str) and s == "None":
            return None
        return s

    def read_obj(self, obj, where=None):
        """Read object from file in group specified by where argument.

        Parameters
        ----------
        obj : python object instance
            object must have _io_attrs_ attribute to have attributes read and loaded
        where : None or file instance
            specifies where to read obj from

        """
        loc = self.resolve_where(where)
        for attr in obj._io_attrs_:
            if attr not in loc.keys():
                warnings.warn(
                    colored(
                        "\n"
                        f"The object attribute '{attr}' was not loaded from the file.\n"
                        "This is likely because the file containing "
                        f"'{obj.__class__.__name__}' was created before '{attr}' "
                        f"became an attribute of objects of class '{obj.__class__}'.\n"
                        "The user may verify that a default value has been set.\n"
                        "This warning will persist until the file is saved with the "
                        "new object.\n"
                        "\n"
                        "Note to developers: Add 'def _set_up(self)' as a method to "
                        f"class '{obj.__class__}'\n"
                        "(or the superclass where this new attribute is assigned) that "
                        f"assigns a value to '{attr}'.\n"
                        "This method is called automatically when a file is loaded.\n"
                        "Recall that the testing suite will fail on warnings.",
                        "yellow",
                    ),
                    RuntimeWarning,
                )
                continue
            if isinstance(loc[attr], h5py.Dataset):
                s = self._decode_attr(loc, attr)
                if not isinstance(s, str) or s != "__class__":
                    setattr(obj, attr, s)
            elif isinstance(loc[attr], h5py.Group):
                if "__class__" not in loc[attr].keys():
                    warnings.warn(
                        "Could not load attribute '{}', no class name found.".format(
                            attr
                        ),
                        RuntimeWarning,
                    )
                    continue

                cls_name = loc[attr]["__class__"][()].decode("utf-8")
                if cls_name == "list":
                    setattr(obj, attr, self.read_list(where=loc[attr]))
                elif cls_name == "dict":
                    setattr(obj, attr, self.read_dict(where=loc[attr]))
                else:
                    # use importlib to import the correct class
                    cls = pydoc.locate(cls_name)
                    if cls is not None:
                        setattr(
                            obj,
                            attr,
                            cls.load(
                                load_from=loc[attr],
                                file_format=self._file_format_,
                            ),
                        )
                    else:
                        warnings.warn(
                            "Class '{}' could not be imported.".format(cls_name),
                            RuntimeWarning,
                        )
                        continue

    def read_dict(self, where=None):
        """Read dictionary from file in group specified by where argument.

        Parameters
        ----------
        where : None or file instance
            specifies where to read dict from

        """
        thedict = {}
        loc = self.resolve_where(where)
        for key in loc.keys():
            if isinstance(loc[key], h5py.Dataset) and key != "__class__":
                s = self._decode_attr(loc, key)
                thedict[key] = s

            elif isinstance(loc[key], h5py.Group):
                if "__class__" not in loc[key].keys():
                    warnings.warn(
                        "Could not load attribute '{}', no class name found.".format(
                            key
                        ),
                        RuntimeWarning,
                    )
                    continue
                cls_name = loc[key]["__class__"][()].decode("utf-8")
                if cls_name == "list":
                    thedict[key] = self.read_list(where=loc[key])
                elif cls_name == "dict":
                    thedict[key] = self.read_dict(where=loc[key])
                else:
                    # use importlib to import the correct class
                    cls = pydoc.locate(cls_name)
                    if cls is not None:
                        thedict[key] = cls.load(
                            load_from=loc[key],
                            file_format=self._file_format_,
                        )
                    else:
                        warnings.warn(
                            "Class '{}' could not be imported.".format(cls_name),
                            RuntimeWarning,
                        )
                        continue

        return thedict

    def read_list(self, where=None):
        """Read list from file in group specified by where argument.

        Parameters
        ----------
        where : None or file instance
            specifies where to read dict from

        """
        thelist = []
        loc = self.resolve_where(where)
        i = 0
        while str(i) in loc.keys():
            if isinstance(loc[str(i)], h5py.Dataset):
                s = self._decode_attr(loc, str(i))
                if not isinstance(s, str) or s != "__class__":
                    thelist.append(s)
            elif isinstance(loc[str(i)], h5py.Group):
                if "__class__" not in loc[str(i)].keys():
                    warnings.warn(
                        "Could not load attribute '{}', no class name found.".format(
                            str(i)
                        ),
                        RuntimeWarning,
                    )
                    continue
                cls_name = loc[str(i)]["__class__"][()].decode("utf-8")
                if cls_name == "list":
                    thelist.append(self.read_list(where=loc[str(i)]))
                elif cls_name == "dict":
                    thelist.append(self.read_dict(where=loc[str(i)]))
                else:
                    # use importlib to import the correct class
                    cls = pydoc.locate(cls_name)
                    if cls is not None:
                        thelist.append(
                            cls.load(
                                load_from=loc[str(i)],
                                file_format=self._file_format_,
                            ),
                        )
                    else:
                        warnings.warn(
                            "Class '{}' could not be imported.".format(cls_name),
                            RuntimeWarning,
                        )
                        continue
            i += 1

        return thelist


class hdf5Writer(hdf5IO, Writer):
    """Class specifying a writer with hdf5IO."""

    def __init__(self, target, file_mode="w"):
        """Initialize hdf5Writer class.

        Parameters
        ----------
        target : str or file instance
            path OR file instance to write to
        file_mode : str
            mode used when opening file.

        """
        self.target = target
        self.file_mode = file_mode
        super().__init__()

    def write_obj(self, obj, where=None):
        """Write object to file in group specified by where argument.

        Parameters
        ----------
        obj : python object instance
            object must have _io_attrs_ attribute to have attributes read and loaded
        where : None or file instance
            specifies where to write obj to

        """
        loc = self.resolve_where(where)

        def isarray(x):
            return hasattr(x, "shape") and hasattr(x, "dtype")

        # save name of object class
        loc.create_dataset("__class__", data=fullname(obj))
        from desc import __version__

        loc.create_dataset("__version__", data=__version__)
        for attr in obj._io_attrs_:
            try:
                data = getattr(obj, attr)
            except AttributeError:
                warnings.warn(
                    "Save attribute '{}' was not saved as it does "
                    "not exist.".format(attr),
                    RuntimeWarning,
                )
                continue
            if data is None:
                data = "None"
            if isarray(data):
                data = np.asarray(data)  # convert jax arrs to np

            if (
                isarray(data)
                or isinstance(data, numbers.Number)
                or isinstance(data, str)
            ):
                compression = "gzip" if isarray(data) and data.size > 1 else None
                loc.create_dataset(attr, data=data, compression=compression)
            elif isinstance(data, dict):
                group = loc.create_group(attr)
                self.write_dict(data, where=group)
            elif isinstance(data, (list, tuple)):
                group = loc.create_group(attr)
                self.write_list(data, where=group)
            else:
                from .optimizable_io import IOAble

                if isinstance(data, IOAble):
                    group = loc.create_group(attr)
                    data.save(group)
                else:
                    raise TypeError(
                        f"don't know how to save attribute {attr} of type {type(data)}"
                    )

    def write_dict(self, thedict, where=None):
        """Write dictionary to file in group specified by where argument.

        Parameters
        ----------
        thedict : dictionary
            dictionary to write to file
        where : None or file instance
            specifies where to write dict to

        """
        loc = self.resolve_where(where)
        loc.create_dataset("__class__", data="dict")
        for key in thedict.keys():
            if isinstance(thedict[key], list):
                group = loc.create_group(key)
                self.write_list(thedict[key], where=group)
            elif isinstance(thedict[key], dict):
                group = loc.create_group(key)
                self.write_dict(thedict[key], where=group)
            else:
                try:
                    data = thedict[key]
                    compression = (
                        "gzip"
                        if isinstance(data, np.ndarray) and np.asarray(data).size > 1
                        else None
                    )
                    loc.create_dataset(key, data=data, compression=compression)
                except TypeError:
                    group = loc.create_group(key)
                    self.write_obj(thedict[key], group)

    def write_list(self, thelist, where=None):
        """Write list to file in group specified by where argument.

        Parameters
        ----------
        thelist : list
            list to write to file
        where : None or file instance
            specifies where to write list to

        """
        loc = self.resolve_where(where)
        loc.create_dataset("__class__", data="list")
        for i in range(len(thelist)):
            if isinstance(thelist[i], list):
                subloc = loc.create_group(str(i))
                self.write_list(thelist[i], where=subloc)
            elif isinstance(thelist[i], dict):
                subloc = loc.create_group(str(i))
                self.write_dict(thelist[i], where=subloc)
            else:
                try:
                    data = thelist[i]
                    compression = (
                        "gzip"
                        if isinstance(data, np.ndarray) and np.asarray(data).size > 1
                        else None
                    )
                    loc.create_dataset(str(i), data=data, compression=compression)
                except TypeError:
                    subloc = loc.create_group(str(i))
                    self.write_obj(thelist[i], where=subloc)
