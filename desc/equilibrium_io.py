import re
import pathlib
import warnings
import h5py
import numpy as np
from datetime import datetime
import warnings
from abc import ABC, abstractmethod
import io
import pickle

from desc.backend import TextColors

def output_to_file(fname, equil):
    """Prints the equilibrium solution to a text file

    Parameters
    ----------
    fname : str or path-like
        filename of output file.
    equil : dict
        dictionary of equilibrium parameters.

    Returns
    -------

    """

    cR = equil['cR']
    cZ = equil['cZ']
    cL = equil['cL']
    bdryR = equil['bdryR']
    bdryZ = equil['bdryZ']
    cP = equil['cP']
    cI = equil['cI']
    Psi_lcfs = equil['Psi_lcfs']
    NFP = equil['NFP']
    zern_idx = equil['zern_idx']
    lambda_idx = equil['lambda_idx']
    bdry_idx = equil['bdry_idx']

    # open file
    file = open(fname, 'w+')
    file.seek(0)

    # scaling factors
    file.write('NFP = {:3d}\n'.format(NFP))
    file.write('Psi = {:16.8E}\n'.format(Psi_lcfs))

    # boundary paramters
    nbdry = len(bdry_idx)
    file.write('Nbdry = {:3d}\n'.format(nbdry))
    for k in range(nbdry):
        file.write('m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n'.format(
            int(bdry_idx[k, 0]), int(bdry_idx[k, 1]), bdryR[k], bdryZ[k]))

    # profile coefficients
    nprof = max(cP.size, cI.size)
    file.write('Nprof = {:3d}\n'.format(nprof))
    for k in range(nprof):
        if k >= cP.size:
            file.write(
                'l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k, 0, cI[k]))
        elif k >= cI.size:
            file.write(
                'l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k, cP[k], 0))
        else:
            file.write(
                'l: {:3d} cP = {:16.8E} cI = {:16.8E}\n'.format(k, cP[k], cI[k]))

    # R & Z Fourier-Zernike coefficients
    nRZ = len(zern_idx)
    file.write('NRZ = {:5d}\n'.format(nRZ))
    for k, lmn in enumerate(zern_idx):
        file.write('l: {:3d} m: {:3d} n: {:3d} cR = {:16.8E} cZ = {:16.8E}\n'.format(
            lmn[0], lmn[1], lmn[2], cR[k], cZ[k]))

    # lambda Fourier coefficients
    nL = len(lambda_idx)
    file.write('NL = {:5d}\n'.format(nL))
    for k, mn in enumerate(lambda_idx):
        file.write('m: {:3d} n: {:3d} cL = {:16.8E}\n'.format(
            mn[0], mn[1], cL[k]))

    # close file
    file.truncate()
    file.close()

    return None


def read_desc(filename):
    """reads a previously generated DESC ascii output file

    Parameters
    ----------
    filename : str or path-like
        path to file to read

    Returns
    -------
    equil : dict
        dictionary of equilibrium parameters.

    """

    equil = {}
    f = open(filename, 'r')
    lines = list(f)
    equil['NFP'] = int(lines[0].strip('\n').split()[-1])
    equil['Psi_lcfs'] = float(lines[1].strip('\n').split()[-1])
    lines = lines[2:]

    Nbdry = int(lines[0].strip('\n').split()[-1])
    equil['bdry_idx'] = np.zeros((Nbdry, 2), dtype=int)
    equil['bdryR'] = np.zeros(Nbdry)
    equil['bdryZ'] = np.zeros(Nbdry)
    for i in range(Nbdry):
        equil['bdry_idx'][i, 0] = int(lines[i+1].strip('\n').split()[1])
        equil['bdry_idx'][i, 1] = int(lines[i+1].strip('\n').split()[3])
        equil['bdryR'][i] = float(lines[i+1].strip('\n').split()[6])
        equil['bdryZ'][i] = float(lines[i+1].strip('\n').split()[9])
    lines = lines[Nbdry+1:]

    Nprof = int(lines[0].strip('\n').split()[-1])
    equil['cP'] = np.zeros(Nprof)
    equil['cI'] = np.zeros(Nprof)
    for i in range(Nprof):
        equil['cP'][i] = float(lines[i+1].strip('\n').split()[4])
        equil['cI'][i] = float(lines[i+1].strip('\n').split()[7])
    lines = lines[Nprof+1:]

    NRZ = int(lines[0].strip('\n').split()[-1])
    equil['zern_idx'] = np.zeros((NRZ, 3), dtype=int)
    equil['cR'] = np.zeros(NRZ)
    equil['cZ'] = np.zeros(NRZ)
    for i in range(NRZ):
        equil['zern_idx'][i, 0] = int(lines[i+1].strip('\n').split()[1])
        equil['zern_idx'][i, 1] = int(lines[i+1].strip('\n').split()[3])
        equil['zern_idx'][i, 2] = int(lines[i+1].strip('\n').split()[5])
        equil['cR'][i] = float(lines[i+1].strip('\n').split()[8])
        equil['cZ'][i] = float(lines[i+1].strip('\n').split()[11])
    lines = lines[NRZ+1:]

    NL = int(lines[0].strip('\n').split()[-1])
    equil['lambda_idx'] = np.zeros((NL, 2), dtype=int)
    equil['cL'] = np.zeros(NL)
    for i in range(NL):
        equil['lambda_idx'][i, 0] = int(lines[i+1].strip('\n').split()[1])
        equil['lambda_idx'][i, 1] = int(lines[i+1].strip('\n').split()[3])
        equil['cL'][i] = float(lines[i+1].strip('\n').split()[6])
    lines = lines[NL+1:]

    return equil

class IOAble(ABC):
    """Abstract Base Class for savable and loadable objects."""

    def _init_from_file_(self, load_from=None, file_format:str=None, obj_lib=None) -> None:
        """Initialize from file.

        Parameters
        __________
        load_from : str file path OR file instance (Default self.load_from)
            file to initialize from
        file_format : str (Default self._file_format_)
            file format of file initializing from

        Returns
        _______
        None

        """
        if load_from is None:
            load_from = self.load_from
        if file_format is None:
            file_format = self._file_format_
        reader = reader_factory(load_from, file_format)
        reader.read_obj(self, obj_lib=obj_lib)
        return None

    def save(self, save_to, file_format='hdf5', file_mode='w'):
        """Save the object.

        Parameters
        __________
        save_to : str file path OR file instance
            location to save object
        file_format : str (Default hdf5)
            format of save file. Only used if save_to is a file path
        file_mode : str (Default w - overwrite)
            mode for save file. Only used if save_to is a file path

        Returns
        _______
        None

        """
        writer = writer_factory(save_to, file_format=file_format,
                file_mode=file_mode)
        writer.write_obj(self)
        writer.close()

class IO(ABC):
    """Abstract Base Class (ABC) for readers and writers."""

    def __init__(self):
        """Initalize ABC IO.

        Parameters
        __________

        Returns
        _______

        None

        """
        self.resolve_base()

    def __del__(self):
        """Close file upon garbage colleciton or explicit deletion with del function.

        Parameters
        __________

        Returns
        _______
        None

        """
        self.close()

    def close(self):
        """Close file if initialized with class instance.

        Parameters
        __________

        Returns
        _______
        None

        """
        if self._close_base_:
            self.base.close()
            self._close_base_ = False
        return None

    def resolve_base(self):
        """Set base attribute.

        Base is target if target is a file instance of type given by
        _file_types_ attribute. _close_base_ is False.

        Base is a runtime-initialized file if target is a string file path.
        _close_base_ is True.

        Parameters
        __________

        Returns
        _______
        None

        """
        if self.check_type(self.target):
            self.base = self.target
            self._close_base_ = False
        elif type(self.target) is str:
            self.base = self.open_file(self.target, self.file_mode)
            self._close_base_ = True
        else:
            raise SyntaxError('save_to of type {} is not a filename or file '
                'instance.'.format(type(self.target)))

    def resolve_where(self, where):
        """Find where 'where' points and check if it's a readable type.

        Parameters
        __________
        where : None or file with type found in _file_types_ attribute

        Returns
        _______
        if where is None:
            base attribute
        if where is file with type foundin _file_types_
            where

        """
        if where is None:
            loc = self.base
        elif self.check_type(where):
            loc = where
        else:
            raise SyntaxError("where '{}' is not a readable type.".format(where))
        return loc

    @abstractmethod
    def open_file(self, file_name, file_mode):
        pass

    def check_type(self, obj):
        if type(obj) in self._file_types_:
            return True
        else:
            return False


class hdf5IO(IO):
    """Class to wrap ABC IO for hdf5 file format."""
    def __init__(self):
        """Initialize hdf5IO instance.

        Parameters
        __________

        Returns
        _______
        None

        """
        self._file_types_ = [h5py._hl.group.Group, h5py._hl.files.File]
        self._file_format_ = 'hdf5'
        super().__init__()

    def open_file(self, file_name, file_mode):
        """Open hdf5 file.

        Parameters
        __________
        file_name : str
            path to file to open
        file_mode : str
            mode used when opening file

        Returns
        _______
        hdf5 file instance

        """
        return h5py.File(file_name, file_mode)

    def sub(self, name):
        """Create subgroup or return if already exists.

        Parameters
        __________
        name : str
            name of subgroup

        Returns
        _______
        sub : subgroup instance

        """
        try:
            return self.base.create_group(name)
        except ValueError:
            return self.base[name]
        except KeyError:
            raise RuntimeError('Cannot create sub in reader.')

    def groups(self, where=None):
        """Finds groups in location given by 'where'.

        Parameters
        __________
        where : None or file instance

        Returns
        _______
        groups : list

        """
        loc = self.resolve_where(where)
        return list(loc.keys())

class PickleIO(IO):
    """Class to wrap ABC IO for pickle file format. """
    def __init__(self):
        """Initialize PickleIO instance.

        Parameters
        __________

        Returns
        _______
        None

        """
        self._file_types_ = [io.BufferedWriter]
        self._file_format_ = 'pickle'
        super().__init__()

    def open_file(self, file_name, file_mode):
        """Open file containing pickled object.

        Parameters
        __________
        file_name : str
            path to file to open
        file_mode : str
            mode used when opening file. Binary flag automatically added if missing.

        Returns
        binary file instance

        """
        if file_mode[-1] != 'b':
            file_mode += 'b'
        return open(file_name, file_mode)

class Reader(ABC):
    """ABC for all readers."""
    @abstractmethod
    def read_obj(self, obj, where=None):
        pass

    @abstractmethod
    def read_dict(self, thedict, where=None):
        pass

class Writer(ABC):
    """ABC for all writers."""
    @abstractmethod
    def write_obj(self, obj, where=None):
        pass

    @abstractmethod
    def write_dict(self, thedict, where=None):
        pass

class hdf5Reader(hdf5IO,Reader):
    """Class specifying a Reader with hdf5IO."""
    def __init__(self, target):
        """Initialize hdf5Reader class.

        Parameters
        __________
        target : str or file instance
            Path to file OR file instance to be read.

        Returns
        _______
        None

        """
        self.target = target
        self.file_mode = 'r'
        super().__init__()

    def read_obj(self, obj, where=None, obj_lib=None):
        """Read object from file in group specified by where argument.

        Parameters
        __________
        obj : python object instance
            object must have _save_attrs_ attribute to have attributes read and loaded
        where : None or file insance
            specifies where to read obj from

        Returns
        _______
        None

        """
        if obj_lib is not None:
            self.obj_lib = obj_lib
        elif hasattr(self, 'obj_lib'):
            pass
        elif hasattr(obj, '_object_lib_'):
            self.obj_lib = obj._object_lib_
        else:
            pass
        loc = self.resolve_where(where)
        for attr in obj._save_attrs_:
            try:
                setattr(obj, attr, loc[attr][()])
            except KeyError:
                warnings.warn("Save attribute '{}' was not loaded.".format(attr),
                        RuntimeWarning)
            except AttributeError:
                try:
                    if 'name' in loc[attr].keys():
                        theattr = loc[attr]['name'][()]
                        if theattr == 'list':
                            setattr(obj, attr, self.read_list(where=loc[attr]))
                        elif theattr == 'dict':
                            setattr(obj, attr, self.read_dict(where=loc[attr]))
                        else:
                            try:
                                #initialized an object from object_lib
                                #print('setting attribute', attr, 'as an ', theattr)
                                setattr(obj, attr, self.obj_lib[theattr](load_from=loc[attr],
                                    file_format=self._file_format_, obj_lib=self.obj_lib))
                            except KeyError:
                                warnings.warn("No object_lib  '{}'.".format(attr),
                                        RuntimeWarning)
                    else:
                        warnings.warn("Could not load attribute '{}'.".format(attr),
                                RuntimeWarning)
                except AttributeError:
                    warnings.warn("Could not set attribute '{}'.".format(attr),
                                RuntimeWarning)
                #    theattr = loc[attr][()]
                #    print('for attr', attr, 'theattr is', theattr, 'with object', obj)
                #    if type(theattr) is np.bool_:
                #        print('converting bool')
                #        newattr = bool(theattr)
                #        print('new type is', type(newattr))
                #        setattr(obj, attr, newattr)
                #    else:
                #        raise NotImplementedError("Data of type '{}' has not "
                #            "been made compatible with loading.".format(type(loc[attr][()])))
        return None

    def read_dict(self, thedict=None, where=None):
        """Read dictionary from file in group specified by where argument.

        Parameters
        __________
        thedict : dictionary (Default None)
            dictionary to update from the file
        where : None or file instance
            specifies where to read dict from

        Returns
        _______
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
                if 'name' in loc[key].keys():
                    theattr = loc[key]['name'][()]
                    if theattr == 'list':
                        thedict[theattr] = self.read_list(where=loc[key])
                    elif theattr == 'dict':
                        thedict[theattr] = self.read_dict(where=loc[key])
                    else:
                        try:
                            #initialized an object from object_lib
                            thedict[theattr] = self.obj_lib[theattr](load_from=loc[key],
                                    file_format=self._file_format_, obj_lib=self.obj_lib)
                        except KeyError:
                            warnings.warn("Could not load attribute '{}'.".format(key),
                                    RuntimeWarning)
                else:
                    warnings.warn("Could not load attribute '{}'.".format(key),
                            RuntimeWarning)
        if ret:
            return thedict
        else:
            return None

    def read_list(self, thelist=None, where=None):
        """Read list from file in group specified by where argument.

        Parameters
        __________
        thelist : list (Default None)
            list to update from the file
        where : None or file instance
            specifies wehre to read dict from

        Returns
        _______
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
                if 'name' in loc[str(i)].keys():
                    theattr = loc[str(i)]['name'][()]
                    #print('loading a ', theattr, 'from list') #debug
                    if theattr == 'list':
                        thelist.append(self.read_list(where=theattr))
                    elif theattr == 'dict':
                        thelist.append(self.read_dict(where=theattr))
                    else:
                        try:
                            #initialized an object from object_lib
                            thelist.append(self.obj_lib[theattr](load_from=loc[str(i)],
                                file_format=self._file_format_, obj_lib=self.obj_lib))
                        except KeyError:
                            warnings.warn("Could not load list index '{}'.".format(i),
                                    RuntimeWarning)
                else:
                    warnings.warn("Could not load list index '{}'.".format(i),
                            RuntimeWarning)
            i += 1
        if ret:
            return thelist
        else:
            return None

class PickleReader(PickleIO,Reader):
    """Class specifying a reader with PickleIO."""
    def __init__(self, target):
        """Initialize hdf5Reader class.

        Parameters
        __________
        target : str or file instance
            Path to file OR file instance to be read.

        Returns
        _______
        None

        """
        self.target = target
        self.file_mode = 'r'
        super().__init__()

    def read_obj(self, obj=None, where=None):
        """Read object from file in group specified by where argument.

        Parameters
        __________
        obj : python object instance
            object must have _save_attrs_ attribute to have attributes read and loaded
        where : None or file insance
            specifies where to read obj from

        Returns
        _______
        None

        """
        loc = self.resolve_where(where)
        if obj is None:
            return pickle.load(loc)
        else:
            obj = pickle.load(loc)

    def read_dict(self, thedict, where=None):
        """Read dictionary from file in group specified by where argument.

        Parameters
        __________
        thedict : dictionary
            dictionary to update from the file
        where : None of file instance
            specifies where to read dict from

        Returns
        _______
        None

        """
        loc = self.resolve_where(where)
        thedict.update(pickle.load(loc))
        return None

class hdf5Writer(hdf5IO,Writer):
    """Class specifying a writer with hdf5IO."""
    def __init__(self, target, file_mode='w'):
        """Initializes hdf5Writer class.

        Parameters
        __________
        target : str or file instance
            path OR file instance to write to
        file_mode : str
            mode used when opening file.

        Returns
        _______
        None

        """
        self.target = target
        self.file_mode = file_mode
        super().__init__()

    def write_obj(self, obj, where=None):
        """Write object to file in group specified by where argument.

        Parameters
        __________
        obj : python object instance
            object must have _save_attrs_ attribute to have attributes read and loaded
        where : None or file insance
            specifies where to write obj to

        Returns
        _______
        None

        """
        loc = self.resolve_where(where)
        #save name of object class
        loc.create_dataset('name', data=type(obj).__name__)
        for attr in obj._save_attrs_:
            try:
                #print(attr) #debugging
                loc.create_dataset(attr, data=getattr(obj, attr))
            except AttributeError:
                warnings.warn("Save attribute '{}' was not saved as it does "
                        "not exist.".format(attr), RuntimeWarning)
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
                        warnings.warn("Could not save object '{}'.".format(attr),
                                RuntimeWarning)
        return None

    def write_dict(self, thedict, where=None):
        """Write dictionary to file in group specified by where argument.

        Parameters
        __________
        thedict : dictionary
            dictionary to write to file
        where : None or file instance
            specifies where to write dict to

        Returns
        _______
        None

        """
        loc = self.resolve_where(where)
        loc.create_dataset('name', data='dict')
        for key in thedict.keys():
            try:
                loc.create_dataset(key, data=thedict[key])
            except TypeError:
                self.write_obj(thedict[key], loc)
        return None

    def write_list(self, thelist, where=None):
        """Write list to file in group specified by where argument.

        Parameters
        __________
        thelist : list
            list to write to file
        where : None or file instance
            specifies where to write list to

        Returns
        _______
        None

        """
        loc = self.resolve_where(where)
        loc.create_dataset('name', data='list')
        for i in range(len(thelist)):
            try:
                loc.create_dataset(str(i), data=thelist[i])
            except TypeError:
                subloc = loc.create_group(str(i))
                self.write_obj(thelist[i], where=subloc)
        return None

class PickleWriter(PickleIO,Writer):
    """Class specifying a writer with PickleIO."""
    def __init__(self, target, file_mode='w'):
        """Initializes PickleWriter class.

        Parameters
        __________
        target : str or file instance
            path OR file instance to write to
        file_mode : str
            mode used when opening file.

        Returns
        _______
        None

        """
        self.target = target
        self.file_mode = file_mode
        super().__init__()

    def write_obj(self, obj, where=None):
        """Write object to file in group specified by where argument.

        Parameters
        __________
        obj : python object instance
            object must have _save_attrs_ attribute to have attributes read and loaded
        where : None or file insance
            specifies where to write obj to

        Returns
        _______
        None

        """
        loc = self.resolve_where(where)
        pickle.dump(obj, loc)
        return None

    def write_dict(self, thedict, where=None):
        """Write dictionary to file in group specified by where argument.

        Parameters
        __________
        thedict : dictionary
            dictionary to update from the file
        where : None of file instance
            specifies where to write dict to

        Returns
        _______
        None

        """
        if type(thedict) is not dict:
            raise TypeError('Object provided is not a dictionary.')
        self.write_object(thedict, where=where)
        return None

def reader_factory(load_from, file_format):
    """Select and return instance of appropriate reader class for given file format.

    Parameters
    __________
    load_from : str or file instance
        file path or instance from which to read
    file_format : str
        format of file to be read

    Returns
    _______
    Reader instance

    """
    if file_format == 'hdf5':
        reader = hdf5Reader(load_from)
    elif file_format == 'pickle':
        reader = PickleReader(load_from)
    else:
        raise NotImplementedError("Format '{}' has not been implemented.".format(file_format))
    return reader

def writer_factory(save_to, file_format, file_mode='w'):
    """Select and return instance of appropriate reader class for given file format.

    Parameters
    __________
    load_from : str or file instance
        file path or instance from which to read
    file_format : str
        format of file to be read

    Returns
    _______
    Reader instance

    """
    if file_format == 'hdf5':
        writer = hdf5Writer(save_to, file_mode)
    elif file_format == 'pickle':
        writer = PickleWriter(save_to, file_mode)
    else:
        raise NotImplementedError("Format '{}' has not been implemented.".format(file_format))
    return writer

def write_hdf5(obj, save_to, file_mode='w'):
    """Writes attributes of obj from obj._save_attrs_ list to an hdf5 file.

    Parameters
    __________
    obj: object to save
        must have _save_attrs_ list attribute. Otherwise AttributeError raised.
    save_loc : str or path-like; hdf5 file or group
        file or group to write to. If str or path-like, file is created. If
        hdf5 file or group instance, datasets are created there.
    file_mode='w': str
        hdf5 file mode. Default is 'w'.
    """
    # check save_loc is an accepted type
    save_to_type = type(save_to)
    if save_to_type is h5py._hl.group.Group or save_to_type is h5py._hl.files.File:
        file_group = save_to
        close = False
    elif save_to_type is str:
        file_group = h5py.File(save_to, file_mode)
        close = True
    else:
        raise SyntaxError('save_to of type {} is not a filename or hdf5 '
            'file or group.'.format(save_to_type))

    # save to file or group
    for attr in obj._save_attrs_:
        file_group.create_dataset(attr, data=getattr(obj, attr))

    # close file if created
    if close:
        file_group.close()

    return None

def write_desc_h5(filename, equilibrium):
    """Writes a DESC equilibrium to a hdf5 format binary file

    Parameters
    ----------
    filename : str or path-like
        file to write to. If it doesn't exist,
        it is created.
    equilibrium : dict
        dictionary of equilibrium parameters.

    Returns
    -------

    """

    f = h5py.File(filename, 'w')
    equil = f.create_group('equilibrium')
    for key, val in equilibrium.items():
        equil.create_dataset(key, data=val)
    equil['zern_idx'].attrs.create('column_labels', ['l', 'm', 'n'])
    equil['bdry_idx'].attrs.create('column_labels', ['m', 'n'])
    equil['lambda_idx'].attrs.create('column_labels', ['m', 'n'])
    f.close()


class Checkpoint():
    """Class for periodically saving equilibria during solution

    Parameters
    ----------
    filename : str or path-like
        file to write to. If it does not exist,
        it will be created
    write_ascii : bool
        Whether to also write ascii files. By default,
        only an hdf5 file is created and appended with each new solution.
        If write_ascii is True, additional files will be written, each with
        the same base filename but appeneded with _0, _1,...

    Returns
    -------
    checkpointer: Checkpoint
        object with methods to periodically save solutions
    """

    def __init__(self, filename, write_ascii=False):

        self.filename = str(pathlib.Path(filename).resolve())
        if self.filename.endswith('.h5'):
            self.base_file = self.filename[:-3]
        elif self.filename.endswith('.hdf5'):
            self.base_file = self.filename[:-5]
        else:
            self.base_file = self.filename
            self.filename += '.h5'

        self.f = h5py.File(self.filename, 'w')
        _ = self.f.create_group('iterations')
        _ = self.f.create_group('final').create_group('equilibrium')
        self.write_ascii = write_ascii

    def write_iteration(self, equilibrium, iter_num, inputs=None, update_final=True):
        """Write an equilibrium to the checkpoint file

        Parameters
        ----------
        equilibrium : dict
            equilibrium to write
        iter_num : int
            iteration number
        inputs : dict, optional
             dictionary of input parameters to the solver (Default value = None)
        update_final : bool
            whether to update the 'final' equilibrium
            with this entry (Default value = True)

        Returns
        -------

        """
        iter_str = str(iter_num)
        if iter_str not in self.f['iterations']:
            self.f['iterations'].create_group(iter_str)
        if 'equilibrium' not in self.f['iterations'][iter_str]:
            self.f['iterations'][iter_str].create_group('equilibrium')
        for key, val in equilibrium.items():
            self.f['iterations'][iter_str]['equilibrium'][key] = val

        self.f['iterations'][iter_str]['equilibrium']['zern_idx'].attrs.create(
            'column_labels', ['l', 'm', 'n'])
        self.f['iterations'][iter_str]['equilibrium']['bdry_idx'].attrs.create(
            'column_labels', ['m', 'n'])
        self.f['iterations'][iter_str]['equilibrium']['lambda_idx'].attrs.create(
            'column_labels', ['m', 'n'])

        if self.write_ascii:
            fname = self.base_file + '_' + str(iter_str) + '.out'
            output_to_file(fname, equilibrium)

        if inputs is not None:
            arrays = ['Mpol', 'Ntor', 'Mnodes', 'Nnodes', 'bdry_ratio', 'pres_ratio',
                      'zeta_ratio', 'errr_ratio', 'pert_order', 'ftol', 'xtol', 'gtol', 'nfev']
            if 'inputs' not in self.f['iterations'][iter_str]:
                self.f['iterations'][iter_str].create_group('inputs')
            for key, val in inputs.items():
                if key in arrays and isinstance(iter_num, int):
                    val = val[iter_num-1]
                self.f['iterations'][iter_str]['inputs'][key] = val

        if update_final:
            if 'final' in self.f:
                del self.f['final']
            self.f['final'] = self.f['iterations'][iter_str]

    def close(self):
        """Close the checkpointing file"""
        self.f.close()


def vmec_to_desc_input(vmec_fname, desc_fname):
    """Converts a VMEC input file to an equivalent DESC input file

    Parameters
    ----------
    vmec_fname : str or path-like
        filename of VMEC input file
    desc_fname : str or path-like
        filename of DESC input file. If it already exists it is overwritten.

    Returns
    -------

    """

    # file objects
    vmec_file = open(vmec_fname, 'r')
    desc_file = open(desc_fname, 'w')

    desc_file.seek(0)
    now = datetime.now()
    date = now.strftime('%m/%d/%Y')
    time = now.strftime('%H:%M:%S')
    desc_file.write('# This DESC input file was auto generated from the VMEC input file\n# {}\n# on {} at {}.\n\n'
                    .format(vmec_fname, date, time))

    num_form = r'[-+]?\ *\d*\.?\d*(?:[Ee]\ *[-+]?\ *\d+)?'
    Ntor = 99

    pres_scale = 1.0
    cP = np.array([0.0])
    cI = np.array([0.0])
    axis = np.array([[0, 0, 0.0]])
    bdry = np.array([[0, 0, 0.0, 0.0]])

    for line in vmec_file:
        comment = line.find('!')
        command = (line.strip()+' ')[0:comment]

        # global parameters
        if re.search(r'LRFP\s*=\s*T', command, re.IGNORECASE):
            warnings.warn(
                TextColors.WARNING + 'Using poloidal flux instead of toroidal flux!' + TextColors.ENDC)
        match = re.search('LASYM\s*=\s*[TF]', command, re.IGNORECASE)
        if match:
            if re.search(r'T', match.group(0), re.IGNORECASE):
                desc_file.write('stell_sym \t=   0\n')
            else:
                desc_file.write('stell_sym \t=   1\n')
        match = re.search(r'NFP\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            desc_file.write('NFP\t\t\t= {:3d}\n'.format(numbers[0]))
        match = re.search(r'PHIEDGE\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            desc_file.write('Psi_lcfs\t= {:16.8E}\n'.format(numbers[0]))
        match = re.search(r'MPOL\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            desc_file.write('Mpol\t\t= {:3d}\n'.format(numbers[0]))
        match = re.search(r'NTOR\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [int(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            desc_file.write('Ntor\t\t= {:3d}\n'.format(numbers[0]))
            Ntor = numbers[0]

        # pressure profile
        match = re.search(r'bPMASS_TYPE\s*=\s*\w*', command, re.IGNORECASE)
        if match:
            if not re.search(r'\bpower_series\b', match.group(0), re.IGNORECASE):
                warnings.warn(
                    TextColors.WARNING + 'Pressure is not a power series!' + TextColors.ENDC)
        match = re.search(r'GAMMA\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            if numbers[0] != 0:
                warnings.warn(TextColors.WARNING +
                              'GAMMA is not 0.0' + TextColors.ENDC)
        match = re.search(r'BLOAT\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            if numbers[0] != 1:
                warnings.warn(TextColors.WARNING +
                              'BLOAT is not 1.0' + TextColors.ENDC)
        match = re.search(r'SPRES_PED\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            if numbers[0] != 1:
                warnings.warn(TextColors.WARNING +
                              'SPRES_PED is not 1.0' + TextColors.ENDC)
        match = re.search(r'PRES_SCALE\s*=\s*'+num_form,
                          command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            pres_scale = numbers[0]
        match = re.search(r'AM\s*=(\s*'+num_form+')*', command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            for k in range(len(numbers)):
                l = 2*k
                if cP.size < l+1:
                    cP = np.pad(cP, (0, l+1-cP.size), mode='constant')
                cP[l] = numbers[k]

        # rotational transform
        match = re.search(r'NCURR\s*=(\s*'+num_form+')*',
                          command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            if numbers[0] != 0:
                warnings.warn(
                    TextColors.WARNING + 'Not using rotational transform!' + TextColors.ENDC)
        if re.search(r'\bPIOTA_TYPE\b', command, re.IGNORECASE):
            if not re.search(r'\bpower_series\b', command, re.IGNORECASE):
                warnings.warn(TextColors.WARNING +
                              'Iota is not a power series!' + TextColors.ENDC)
        match = re.search(r'AI\s*=(\s*'+num_form+')*', command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            for k in range(len(numbers)):
                l = 2*k
                if cI.size < l+1:
                    cI = np.pad(cI, (0, l+1-cI.size), mode='constant')
                cI[l] = numbers[k]

        # magnetic axis
        match = re.search(r'RAXIS\s*=(\s*'+num_form+')*',
                          command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            for k in range(len(numbers)):
                if k > Ntor:
                    l = -k+Ntor+1
                else:
                    l = k
                idx = np.where(axis[:, 0] == l)[0]
                if np.size(idx) > 0:
                    axis[idx[0], 1] = numbers[k]
                else:
                    axis = np.pad(axis, ((0, 1), (0, 0)), mode='constant')
                    axis[-1, :] = np.array([l, numbers[k], 0.0])
        match = re.search(r'ZAXIS\s*=(\s*'+num_form+')*',
                          command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            for k in range(len(numbers)):
                if k > Ntor:
                    l = k-Ntor-1
                else:
                    l = -k
                idx = np.where(axis[:, 0] == l)[0]
                if np.size(idx) > 0:
                    axis[idx[0], 2] = numbers[k]
                else:
                    axis = np.pad(axis, ((0, 1), (0, 0)), mode='constant')
                    axis[-1, :] = np.array([l, 0.0, numbers[k]])

        # boundary shape
        # RBS*sin(m*t-n*p) = RBS*sin(m*t)*cos(n*p) - RBS*cos(m*t)*sin(n*p)
        match = re.search(r'RBS\(\s*'+num_form+'\s*,\s*'+num_form +
                          '\s*\)\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            n = int(numbers[0])
            m = int(numbers[1])
            n_sgn = np.sign(np.array([n]))[0]
            n *= n_sgn
            if np.sign(m) < 0:
                warnings.warn(TextColors.WARNING +
                              'm is negative!' + TextColors.ENDC)
            RBS = numbers[2]
            if m != 0:
                m_idx = np.where(bdry[:, 0] == -m)[0]
                n_idx = np.where(bdry[:, 1] == n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]], 2] = RBS
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                    bdry[-1, :] = np.array([-m, n, RBS, 0.0])
            if n != 0:
                m_idx = np.where(bdry[:, 0] == m)[0]
                n_idx = np.where(bdry[:, 1] == -n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]], 2] = -n_sgn*RBS
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                    bdry[-1, :] = np.array([m, -n, -n_sgn*RBS, 0.0])
        # RBC*cos(m*t-n*p) = RBC*cos(m*t)*cos(n*p) + RBC*sin(m*t)*sin(n*p)
        match = re.search(r'RBC\(\s*'+num_form+'\s*,\s*'+num_form +
                          '\s*\)\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            n = int(numbers[0])
            m = int(numbers[1])
            n_sgn = np.sign(np.array([n]))[0]
            n *= n_sgn
            if np.sign(m) < 0:
                warnings.warn(TextColors.WARNING +
                              'm is negative!' + TextColors.ENDC)
            RBC = numbers[2]
            m_idx = np.where(bdry[:, 0] == m)[0]
            n_idx = np.where(bdry[:, 1] == n)[0]
            idx = np.where(np.isin(m_idx, n_idx))[0]
            if np.size(idx) > 0:
                bdry[m_idx[idx[0]], 2] = RBC
            else:
                bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                bdry[-1, :] = np.array([m, n, RBC, 0.0])
            if m != 0 and n != 0:
                m_idx = np.where(bdry[:, 0] == -m)[0]
                n_idx = np.where(bdry[:, 1] == -n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]], 2] = n_sgn*RBC
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                    bdry[-1, :] = np.array([-m, -n, n_sgn*RBC, 0.0])
        # ZBS*sin(m*t-n*p) = ZBS*sin(m*t)*cos(n*p) - ZBS*cos(m*t)*sin(n*p)
        match = re.search(r'ZBS\(\s*'+num_form+'\s*,\s*'+num_form +
                          '\s*\)\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            n = int(numbers[0])
            m = int(numbers[1])
            n_sgn = np.sign(np.array([n]))[0]
            n *= n_sgn
            if np.sign(m) < 0:
                warnings.warn(TextColors.WARNING +
                              'm is negative!' + TextColors.ENDC)
            ZBS = numbers[2]
            if m != 0:
                m_idx = np.where(bdry[:, 0] == -m)[0]
                n_idx = np.where(bdry[:, 1] == n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]], 3] = ZBS
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                    bdry[-1, :] = np.array([-m, n, 0.0, ZBS])
            if n != 0:
                m_idx = np.where(bdry[:, 0] == m)[0]
                n_idx = np.where(bdry[:, 1] == -n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]], 3] = -n_sgn*ZBS
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                    bdry[-1, :] = np.array([m, -n, 0.0, -n_sgn*ZBS])
        # ZBC*cos(m*t-n*p) = ZBC*cos(m*t)*cos(n*p) + ZBC*sin(m*t)*sin(n*p)
        match = re.search(r'ZBC\(\s*'+num_form+'\s*,\s*'+num_form +
                          '\s*\)\s*=\s*'+num_form, command, re.IGNORECASE)
        if match:
            numbers = [float(x) for x in re.findall(
                num_form, match.group(0)) if re.search(r'\d', x)]
            n = int(numbers[0])
            m = int(numbers[1])
            n_sgn = np.sign(np.array([n]))[0]
            n *= n_sgn
            if np.sign(m) < 0:
                warnings.warn(TextColors.WARNING +
                              'm is negative!' + TextColors.ENDC)
            ZBC = numbers[2]
            m_idx = np.where(bdry[:, 0] == m)[0]
            n_idx = np.where(bdry[:, 1] == n)[0]
            idx = np.where(np.isin(m_idx, n_idx))[0]
            if np.size(idx) > 0:
                bdry[m_idx[idx[0]], 3] = ZBC
            else:
                bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                bdry[-1, :] = np.array([m, n, 0.0, ZBC])
            if m != 0 and n != 0:
                m_idx = np.where(bdry[:, 0] == -m)[0]
                n_idx = np.where(bdry[:, 1] == -n)[0]
                idx = np.where(np.isin(m_idx, n_idx))[0]
                if np.size(idx) > 0:
                    bdry[m_idx[idx[0]], 3] = n_sgn*ZBC
                else:
                    bdry = np.pad(bdry, ((0, 1), (0, 0)), mode='constant')
                    bdry[-1, :] = np.array([-m, -n, 0.0, n_sgn*ZBC])

        # catch multi-line inputs
        match = re.search(r'=', command)
        if not match:
            numbers = [float(x) for x in re.findall(
                num_form, command) if re.search(r'\d', x)]
            if len(numbers) > 0:
                raise IOError(
                    TextColors.FAIL + 'Cannot handle multi-line VMEC inputs!' + TextColors.ENDC)

    cP *= pres_scale
    desc_file.write('\n')
    desc_file.write('# pressure and rotational transform profiles\n')
    for k in range(max(cP.size, cI.size)):
        if k >= cP.size:
            desc_file.write(
                'l: {:3d}\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k, 0.0, cI[k]))
        elif k >= cI.size:
            desc_file.write(
                'l: {:3d}\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k, cP[k], 0.0))
        else:
            desc_file.write(
                'l: {:3d}\tcP = {:16.8E}\tcI = {:16.8E}\n'.format(k, cP[k], cI[k]))

    desc_file.write('\n')
    desc_file.write('# magnetic axis initial guess\n')
    for k in range(np.shape(axis)[0]):
        desc_file.write('n: {:3d}\taR = {:16.8E}\taZ = {:16.8E}\n'.format(
            int(axis[k, 0]), axis[k, 1], axis[k, 2]))

    desc_file.write('\n')
    desc_file.write('# fixed-boundary surface shape\n')
    for k in range(np.shape(bdry)[0]):
        desc_file.write('m: {:3d}\tn: {:3d}\tbR = {:16.8E}\tbZ = {:16.8E}\n'.format(
            int(bdry[k, 0]), int(bdry[k, 1]), bdry[k, 2], bdry[k, 3]))

    desc_file.truncate()

    # close files
    vmec_file.close()
    desc_file.close()
