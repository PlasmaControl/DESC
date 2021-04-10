import warnings
import numpy as np
import h5py
from .core_io import IO, Reader, Writer


class hdf5IO(IO):
    """Class to wrap ABC IO for hdf5 file format."""

    def __init__(self):
        """Initialize hdf5IO instance"""
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
        """Finds groups in location given by 'where'.

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
        self.obj_lib = {}
        super().__init__()

    def read_obj(self, obj, where=None, obj_lib=None):
        """Read object from file in group specified by where argument.

        Parameters
        ----------
        obj : python object instance
            object must have _io_attrs_ attribute to have attributes read and loaded
        where : None or file insance
            specifies where to read obj from

        """
        if obj_lib is not None:
            self.obj_lib.update(obj_lib)
        if hasattr(obj, "_object_lib_"):
            self.obj_lib.update(obj._object_lib_)

        loc = self.resolve_where(where)
        for attr in obj._io_attrs_:
            if attr not in loc.keys():
                warnings.warn(
                    "Save attribute '{}' was not loaded.".format(attr), RuntimeWarning
                )
                continue
            if isinstance(loc[attr], h5py.Dataset):
                setattr(obj, attr, loc[attr][()])
            elif isinstance(loc[attr], h5py.Group):
                if "name" not in loc[attr].keys():
                    warnings.warn(
                        "Could not load attribute '{}', no name found.".format(attr),
                        RuntimeWarning,
                    )
                    continue
                name = loc[attr]["name"][()]
                if name == "list":
                    setattr(obj, attr, self.read_list(where=loc[attr]))
                elif name == "dict":
                    setattr(obj, attr, self.read_dict(where=loc[attr]))
                else:
                    if name not in self.obj_lib:
                        warnings.warn(
                            "No object_lib '{}'.".format(name), RuntimeWarning
                        )
                        continue

                    # initialized an object from object_lib
                    setattr(
                        obj,
                        attr,
                        self.obj_lib[name].load(
                            load_from=loc[attr],
                            file_format=self._file_format_,
                            obj_lib=self.obj_lib,
                        ),
                    )

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
            if isinstance(loc[key], h5py.Dataset) and key != "name":
                thedict[key] = loc[key][()]
            elif isinstance(loc[key], h5py.Group):
                if "name" not in loc[key].keys():
                    warnings.warn(
                        "Could not load attribute '{}', no name found.".format(key),
                        RuntimeWarning,
                    )
                    continue
                name = loc[key]["name"][()]
                if name == "list":
                    thedict[name] = self.read_list(where=loc[key])
                elif name == "dict":
                    thedict[key] = self.read_dict(where=loc[key])
                else:
                    if name not in self.obj_lib:
                        warnings.warn(
                            "No object_lib '{}'.".format(name), RuntimeWarning
                        )
                        continue

                    # initialized an object from object_lib
                    thedict[key] = self.obj_lib[name].load(
                        load_from=loc[key],
                        file_format=self._file_format_,
                        obj_lib=self.obj_lib,
                    )
        return thedict

    def read_list(self, where=None):
        """Read list from file in group specified by where argument.

        Parameters
        ----------
        where : None or file instance
            specifies wehre to read dict from

        """
        thelist = []
        loc = self.resolve_where(where)
        i = 0
        while str(i) in loc.keys():
            if isinstance(loc[str(i)], h5py.Dataset):
                thelist.append(loc[str(i)][()])
            elif isinstance(loc[str(i)], h5py.Group):
                if "name" not in loc[str(i)].keys():
                    warnings.warn(
                        "Could not load attribute '{}', no name found.".format(str(i)),
                        RuntimeWarning,
                    )
                    continue
                name = loc[str(i)]["name"][()]
                if name == "list":
                    thelist.append(self.read_list(where=loc[key]))
                elif name == "dict":
                    thelist.append(self.read_dict(where=loc[key]))
                else:
                    if name not in self.obj_lib:
                        warnings.warn(
                            "No object_lib '{}'.".format(name), RuntimeWarning
                        )
                        continue

                    # initialized an object from object_lib
                    thelist.append(
                        self.obj_lib[name].load(
                            load_from=loc[str(i)],
                            file_format=self._file_format_,
                            obj_lib=self.obj_lib,
                        )
                    )
                i += 1

        return thelist


class hdf5Writer(hdf5IO, Writer):
    """Class specifying a writer with hdf5IO."""

    def __init__(self, target, file_mode="w"):
        """Initializes hdf5Writer class.

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
        where : None or file insance
            specifies where to write obj to

        """
        loc = self.resolve_where(where)

        # save name of object class
        loc.create_dataset("name", data=type(obj).__name__)
        for attr in obj._io_attrs_:
            try:
                data = getattr(obj, attr)
                compression = (
                    "gzip"
                    if isinstance(data, np.ndarray) and np.asarray(data).size > 1
                    else None
                )
                loc.create_dataset(attr, data=data, compression=compression)
            except AttributeError:
                warnings.warn(
                    "Save attribute '{}' was not saved as it does "
                    "not exist.".format(attr),
                    RuntimeWarning,
                )
            except TypeError:
                theattr = getattr(obj, attr)
                if isinstance(theattr, dict):
                    group = loc.create_group(attr)
                    self.write_dict(theattr, where=group)
                elif isinstance(theattr, list):
                    group = loc.create_group(attr)
                    self.write_list(theattr, where=group)
                else:
                    try:

                        group = loc.create_group(attr)
                        sub_obj = getattr(obj, attr)
                        sub_obj.save(group)
                    except AttributeError:
                        warnings.warn(
                            "Could not save object '{}'.".format(attr), RuntimeWarning
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
        loc.create_dataset("name", data="dict")
        for key in thedict.keys():
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
        loc.create_dataset("name", data="list")
        for i in range(len(thelist)):
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
