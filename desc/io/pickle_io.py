import io
import pickle
from .core_io import IO, Reader, Writer


class PickleIO(IO):
    """Class to wrap ABC IO for pickle file format. """

    def __init__(self):
        """Initialize PickleIO instance"""
        self._file_types_ = io.BufferedWriter
        self._file_format_ = "pickle"
        super().__init__()

    def open_file(self, file_name, file_mode):
        """Open file containing pickled object.

        Parameters
        ----------
        file_name : str
            path to file to open
        file_mode : str
            mode used when opening file. Binary flag automatically added if missing.

        Returns
        -------
        binary file instance

        """
        if file_mode[-1] != "b":
            file_mode += "b"
        return open(file_name, file_mode)


class PickleReader(PickleIO, Reader):
    """Class specifying a reader with PickleIO."""

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

    def read_obj(self, obj=None, where=None):
        """Read object from file in group specified by where argument.

        Parameters
        ----------
        obj : python object instance
            object must have _io_attrs_ attribute to have attributes read and loaded
        where : None or file insance
            specifies where to read obj from

        """
        loc = self.resolve_where(where)
        if obj is None:
            return pickle.load(loc)
        else:
            obj = pickle.load(loc)

    def read_dict(self, thedict, where=None):
        """Read dictionary from file in group specified by where argument.

        Parameters
        ----------
        thedict : dictionary
            dictionary to update from the file
        where : None of file instance
            specifies where to read dict from

        """
        loc = self.resolve_where(where)
        thedict.update(pickle.load(loc))


class PickleWriter(PickleIO, Writer):
    """Class specifying a writer with PickleIO."""

    def __init__(self, target, file_mode="w"):
        """Initializes PickleWriter class.

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
        pickle.dump(obj, loc)

    def write_dict(self, thedict, where=None):
        """Write dictionary to file in group specified by where argument.

        Parameters
        ----------
        thedict : dictionary
            dictionary to update from the file
        where : None of file instance
            specifies where to write dict to

        """
        if not isinstance(thedict, dict):
            raise TypeError("Object provided is not a dictionary.")
        self.write_object(thedict, where=where)
