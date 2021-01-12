import warnings
import h5py
from .core_io import IO, Reader, Writer


class hdf5IO(IO):
    """Class to wrap ABC IO for hdf5 file format."""

    def __init__(self):
        """Initialize hdf5IO instance.

        Parameters
        ----------

        Returns
        -------
        None

        """
        self._file_types_ = [h5py._hl.group.Group, h5py._hl.files.File]
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

        Returns
        -------
        None

        """
        self.target = target
        self.file_mode = "r"
        super().__init__()

    def read_obj(self, obj, where=None, obj_lib=None):
        """Read object from file in group specified by where argument.

        Parameters
        ----------
        obj : python object instance
            object must have _io_attrs_ attribute to have attributes read and loaded
        where : None or file insance
            specifies where to read obj from

        Returns
        -------
        None

        """
        if obj_lib is not None:
            self.obj_lib = obj_lib
        elif hasattr(self, "obj_lib"):
            pass
        elif hasattr(obj, "_object_lib_"):
            self.obj_lib = obj._object_lib_
        else:
            pass
        loc = self.resolve_where(where)
        for attr in obj._io_attrs_:
            try:
                setattr(obj, attr, loc[attr][()])
            except KeyError:
                warnings.warn(
                    "Save attribute '{}' was not loaded.".format(attr), RuntimeWarning
                )
            except AttributeError:
                try:
                    if "name" in loc[attr].keys():
                        theattr = loc[attr]["name"][()]
                        if theattr == "list":
                            setattr(obj, attr, self.read_list(where=loc[attr]))
                        elif theattr == "dict":
                            setattr(obj, attr, self.read_dict(where=loc[attr]))
                        else:
                            try:
                                # initialized an object from object_lib
                                # print('setting attribute', attr, 'as an ', theattr)
                                setattr(
                                    obj,
                                    attr,
                                    self.obj_lib[theattr](
                                        load_from=loc[attr],
                                        file_format=self._file_format_,
                                        obj_lib=self.obj_lib,
                                    ),
                                )
                            except KeyError:
                                warnings.warn(
                                    "No object_lib '{}'.".format(attr), RuntimeWarning
                                )
                    else:
                        warnings.warn(
                            "Could not load attribute '{}'.".format(attr),
                            RuntimeWarning,
                        )
                except AttributeError:
                    warnings.warn(
                        "Could not set attribute '{}'.".format(attr), RuntimeWarning
                    )
        return None

    def read_dict(self, thedict=None, where=None):
        """Read dictionary from file in group specified by where argument.

        Parameters
        ----------
        thedict : dictionary (Default None)
            dictionary to update from the file
        where : None or file instance
            specifies where to read dict from

        Returns
        -------
        None

        """
        ret = False
        if thedict is None:
            thedict = {}
            ret = True
        loc = self.resolve_where(where)
        for key in loc.keys():
            try:
                thedict[key] = loc[key][()]
            except AttributeError:
                if "name" in loc[key].keys():
                    theattr = loc[key]["name"][()]
                    if theattr == "list":
                        thedict[theattr] = self.read_list(where=loc[key])
                    elif theattr == "dict":
                        thedict[theattr] = self.read_dict(where=loc[key])
                    else:
                        try:
                            # initialized an object from object_lib
                            thedict[theattr] = self.obj_lib[theattr](
                                load_from=loc[key],
                                file_format=self._file_format_,
                                obj_lib=self.obj_lib,
                            )
                        except KeyError:
                            warnings.warn(
                                "Could not load attribute '{}'.".format(key),
                                RuntimeWarning,
                            )
                else:
                    warnings.warn(
                        "Could not load attribute '{}'.".format(key), RuntimeWarning
                    )
        if ret:
            return thedict
        else:
            return None

    def read_list(self, thelist=None, where=None):
        """Read list from file in group specified by where argument.

        Parameters
        ----------
        thelist : list (Default None)
            list to update from the file
        where : None or file instance
            specifies wehre to read dict from

        Returns
        -------
        None

        """
        ret = False
        if thelist is None:
            thelist = []
            ret = True
        loc = self.resolve_where(where)
        i = 0
        while str(i) in loc.keys():
            try:
                thelist.append(loc[str(i)][()])
            except AttributeError:
                if "name" in loc[str(i)].keys():
                    theattr = loc[str(i)]["name"][()]
                    # print('loading a ', theattr, 'from list') #debug
                    if theattr == "list":
                        thelist.append(self.read_list(where=theattr))
                    elif theattr == "dict":
                        thelist.append(self.read_dict(where=theattr))
                    else:
                        try:
                            # initialized an object from object_lib
                            thelist.append(
                                self.obj_lib[theattr](
                                    load_from=loc[str(i)],
                                    file_format=self._file_format_,
                                    obj_lib=self.obj_lib,
                                )
                            )
                        except KeyError:
                            warnings.warn(
                                "Could not load list index '{}'.".format(i),
                                RuntimeWarning,
                            )
                else:
                    warnings.warn(
                        "Could not load list index '{}'.".format(i), RuntimeWarning
                    )
            i += 1
        if ret:
            return thelist
        else:
            return None


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

        Returns
        -------
        None

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

        Returns
        -------
        None

        """
        loc = self.resolve_where(where)
        # save name of object class
        loc.create_dataset("name", data=type(obj).__name__)
        for attr in obj._io_attrs_:
            try:
                loc.create_dataset(attr, data=getattr(obj, attr))
            except AttributeError:
                warnings.warn(
                    "Save attribute '{}' was not saved as it does "
                    "not exist.".format(attr),
                    RuntimeWarning,
                )
            except TypeError:
                theattr = getattr(obj, attr)
                if type(theattr) is dict:
                    self.write_dict(theattr, where=self.sub(attr))
                elif type(theattr) is list:
                    self.write_list(theattr, where=self.sub(attr))
                else:
                    try:
                        group = loc.create_group(attr)
                        sub_obj = getattr(obj, attr)
                        sub_obj.save(group)
                    except AttributeError:
                        warnings.warn(
                            "Could not save object '{}'.".format(attr), RuntimeWarning
                        )
        return None

    def write_dict(self, thedict, where=None):
        """Write dictionary to file in group specified by where argument.

        Parameters
        ----------
        thedict : dictionary
            dictionary to write to file
        where : None or file instance
            specifies where to write dict to

        Returns
        -------
        None

        """
        loc = self.resolve_where(where)
        loc.create_dataset("name", data="dict")
        for key in thedict.keys():
            try:
                loc.create_dataset(key, data=thedict[key])
            except TypeError:
                self.write_obj(thedict[key], loc)
        return None

    def write_list(self, thelist, where=None):
        """Write list to file in group specified by where argument.

        Parameters
        ----------
        thelist : list
            list to write to file
        where : None or file instance
            specifies where to write list to

        Returns
        -------
        None

        """
        loc = self.resolve_where(where)
        loc.create_dataset("name", data="list")
        for i in range(len(thelist)):
            try:
                loc.create_dataset(str(i), data=thelist[i])
            except TypeError:
                subloc = loc.create_group(str(i))
                self.write_obj(thelist[i], where=subloc)
        return None
