"""Functions and methods for saving and loading equilibria and other objects."""

import copy
import os
import pickle
import pydoc
import types
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


def _unjittable(x):
    # strings and functions can't be args to jitted functions, and ints/bools are pretty
    # much always flags or array sizes which also need to be a compile time constant
    if isinstance(x, (list, tuple)):
        return any([_unjittable(y) for y in x])
    if isinstance(x, dict):
        return any([_unjittable(y) for y in x.values()])
    if hasattr(x, "dtype"):
        return np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.bool_)
    return isinstance(x, (str, types.FunctionType, bool, int, np.int_))


def _make_hashable(x):
    # turn unhashable ndarray of ints into a hashable tuple
    if hasattr(x, "shape"):
        return ("ndarray", x.shape, tuple(x.flatten()))
    return x


def _unmake_hashable(x):
    # turn tuple of ints and shape to ndarray
    if isinstance(x, tuple) and x[0] == "ndarray":
        return np.array(x[2]).reshape(x[1])
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

            # this allows classes to override the default static/dynamic stuff
            # if they need certain floats to be static or ints to by dynamic etc.
            static_attrs = getattr(obj, "_static_attrs", [])
            dynamic_attrs = getattr(obj, "_dynamic_attrs", [])
            assert set(static_attrs).isdisjoint(set(dynamic_attrs))

            for key, val in obj.__dict__.items():
                if key in static_attrs:
                    aux_data += [(key, _make_hashable(val))]
                elif key in dynamic_attrs:
                    children[key] = val
                elif _unjittable(val):
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
    To save properly, each object should have an attribute `_io_attrs_` which
    is a list of strings of the object attributes or properties that should be
    saved and loaded.

    For saved objects to be loaded correctly, the __init__ method of any custom
    types being saved should only assign attributes that are listed in `_io_attrs_`.
    Other attributes or other initialization should be done in a separate
    `set_up()` method that can be called during __init__. The loading process
    will involve creating an empty object, bypassing init, then setting any `_io_attrs_`
    of the object, then calling `_set_up()` without any arguments, if it exists.

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


def get_driver():
    """Initialize a webdriver for use in uploading to the database."""
    from selenium import webdriver

    try:
        options = webdriver.FirefoxOptions()
        options.headless = True
        return webdriver.Firefox(options=options)
    except:  # noqa: E722
        pass

    try:
        options = webdriver.ChromeOptions()
        options.headless = True
        return webdriver.Chrome(options=options)
    except:  # noqa: E722
        pass

    try:
        options = webdriver.EdgeOptions()
        options.use_chromium = True
        options.add_argument("headless")
        return webdriver.Edge(options=options)
    except:  # noqa: E722
        pass

    try:
        return webdriver.Safari()
    except:  # noqa: E722
        print(
            "Failed to initialize any webdriver! Consider installing "
            + "Chrome, Safari, Firefox, or Edge."
        )

    # If no browser was successfully initialized, return None
    return None


def load_to_database(
    filename,
    configid,
    user,
    deviceid=None,
    description=None,
    provenance=None,
    inputfilename=None,
    config_class=None,
    current=True,
    initialization_method="surface",
    copy=False,
):
    """Load a DESC equilibrium and upload it to the database.

    Parameters
    ----------
    filename : str
        file path of the output file without .h5 extension
    configid : str
        unique identifier for the configuration
    initialization_method : str (Default: 'surface')
        method used to initialize the equilibrium
    user :
        user who created the equilibrium (must have an account on the database)

    """
    import zipfile

    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    zip_upload_button_id = "zipToUpload"
    csv_upload_button_id = "descToUpload"
    cfg_upload_button_id = "configToUpload"
    confirm_button_id = "confirmDesc"

    # Zip the files
    zip_filename = filename + ".zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(filename + ".h5")
        if os.path.exists(filename + "_input.txt"):
            zipf.write(filename + "_input.txt")
        elif os.path.exists("auto_generated_" + filename + "_input.txt"):
            zipf.write("auto_generated_" + filename + "_input.txt")
        elif inputfilename is not None:
            if os.path.exists(inputfilename):
                zipf.write(inputfilename)

    csv_filename = "desc_runs.csv"
    config_csv_filename = "configurations.csv"
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        print(f"Previous {csv_filename} has been deleted.")
    if os.path.exists(config_csv_filename):
        os.remove(config_csv_filename)
        print(f"Previous {config_csv_filename} has been deleted.\n")

    print("Creating desc_runs.csv and configurations.csv...\n")
    desc_to_csv(
        filename + ".h5",  # output filename
        name=configid,  # some string descriptive name, not necessarily unique
        provenance=provenance,
        description=description,
        inputfilename=inputfilename,
        current=current,
        deviceid=deviceid,
        config_class=config_class,
        user_updated=user,
        user_created=user,
        initialization_method=initialization_method,
    )

    driver = get_driver()
    driver.get("https://ye2698.mycpanel.princeton.edu/import-page/")

    try:
        # Upload the zip file
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, zip_upload_button_id))
        )
        file_input.send_keys(os.path.abspath(zip_filename))

        # Upload the csv file for desc_runs
        file_input2 = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, csv_upload_button_id))
        )
        file_input2.send_keys(os.path.abspath(csv_filename))

        # Upload the csv file for configurations
        file_input3 = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, cfg_upload_button_id))
        )
        file_input3.send_keys(os.path.abspath(config_csv_filename))

        # Confirm the upload
        confirm_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, confirm_button_id))
        )
        confirm_button.click()

        # Wait for the messageContainer div to contain text
        WebDriverWait(driver, 10).until(
            lambda driver: driver.find_element(By.ID, "messageContainer").text.strip()
            != ""
        )

        # Extract and print the message
        message_element = driver.find_element(By.ID, "messageContainer")
        message = message_element.text
        print(message)
    except:  # noqa: E722
        # Extract and print the message
        message_element = driver.find_element(By.ID, "messageContainer")
        message = message_element.text
        print(message)

    finally:
        # Clean up resources
        driver.quit()
        if not copy:
            os.remove(zip_filename)
            os.remove(csv_filename)
            os.remove(config_csv_filename)


def desc_to_csv(  # noqa
    eq,
    current=True,
    name=None,
    provenance=None,
    description=None,
    inputfilename=None,
    initialization_method="surface",
    user_created=None,
    user_updated=None,
    **kwargs,
):
    """Save DESC output file as a csv with relevant information.

    Args
    ----
        eq (Equilibrium or str): DESC equilibrium to save or path to .h5 of
         DESC equilibrium to save
        current (bool): True if the equilibrium was solved with fixed current or not
            if False, was solved with fixed iota
        name (str) : name of configuration (and desc run)
        provenance (str): where this configuration (and desc run) came from, e.g.
            DESC github repo
        description (str): description of the configuration (and desc run)
        inputfilename (str): name of the input file corresponding to this
            configuration (and desc run)
        initialization_method (str): how the DESC equilibrium solution was initialized
            one of "surface", "NAE", or the name of a .nc or .h5 file
            corresponding to a VMEC (if .nc) or DESC (if .h5) solution

    Kwargs
    ------
        date_created (str): when the DESC run was created, defaults to current day
        publicationid (str): unique ID for a publication which this DESC output file is
            associated with.
        deviceid (str): unique ID for a device/concept which this configuration
         is associated with.
        config_class (str): class of configuration i.e. quasisymmetry (QA, QH, QP)
            or omnigenity (QI, OT, OH) or axisymmetry (AS).
            Defaults to None for a stellarator
            and (AS) for a tokamak
            #TODO: can we attempt to automatically detect this for QS configs?
            maybe with a threshold on low QS, then if passes that, classify
            based on largest Boozer mode? can add a flag to the table like
            "automatically labelled class" if this occurs
            to be transparent about source of the class if it was not a human

    Returns
    -------
        None
    """
    import csv
    from datetime import date

    from desc.equilibrium import EquilibriaFamily
    from desc.grid import LinearGrid
    from desc.vmec_utils import ptolemy_identity_rev, zernike_to_fourier

    # data dicts for each table
    data_desc_runs = {}
    data_configurations = {}

    desc_runs_csv_name = "desc_runs.csv"
    configurations_csv_name = "configurations.csv"

    if isinstance(eq, str):
        data_desc_runs["outputfile"] = os.path.basename(eq)
        reader = hdf5Reader(eq)
        version = reader.read_dict()["__version__"]
        eq = load(eq)

    elif isinstance(eq, EquilibriaFamily):
        eq = eq[-1]  # just use last equilibrium
        runid = None  # TODO: is this needed, the equilibriaFamily being an input?
        savename = kwargs.pop("savename", f"DESC_eq_runid_{runid}.h5")
        data_desc_runs["outputfile"] = savename
        from desc import __version__ as version
    if isinstance(eq, EquilibriaFamily):
        eq = eq[-1]

    ############ DESC_runs Data Table ############
    # FIXME: what to do for these?
    if name is not None:
        data_desc_runs["configid"] = name
    if provenance is not None:
        data_desc_runs["provenance"] = provenance
    if description is not None:
        data_desc_runs["description"] = description

    data_desc_runs[
        "version"
    ] = version  # this is basically redundant with git commit I think
    data_desc_runs[
        "git_commit"
    ] = version  # this is basically redundant with git commit I think
    if inputfilename is not None:
        data_desc_runs["inputfilename"] = inputfilename

    data_desc_runs["initialization_method"] = initialization_method

    data_desc_runs["l_rad"] = eq.L
    data_desc_runs["l_grid"] = eq.L_grid
    data_desc_runs["m_pol"] = eq.M
    data_desc_runs["m_grid"] = eq.M_grid
    data_desc_runs["n_tor"] = eq.N
    data_desc_runs["n_grid"] = eq.N_grid

    data_desc_runs[
        "bdry_ratio"
    ] = 1.0  # this is not a equilibrium property, so should not be saved

    # save profiles

    rho = np.linspace(0, 1.0, 11, endpoint=True)
    rho_grid = LinearGrid(rho=rho, M=0, N=0, NFP=eq.NFP)
    data_desc_runs["profile_rho"] = rho
    rho_dense = np.linspace(0, 1.0, 101, endpoint=True)
    rho_grid_dense = LinearGrid(rho=rho_dense, M=0, N=0, NFP=eq.NFP)

    rho_grid.nodes[0, 0] = 1e-12  # bc we dont have axis limit right now
    rho_grid_dense.nodes[0, 0] = 1e-12  # bc we dont have axis limit right now

    if eq.iota and not current:
        data_desc_runs["iota_profile"] = eq.iota(rho)  # sohuld name differently
        data_desc_runs["iota_max"] = np.max(eq.iota(rho_dense))

        data_desc_runs["iota_min"] = np.min(eq.iota(rho_dense))

        data_desc_runs["current_profile"] = round(
            eq.compute("current", grid=rho_grid)["current"], ndigits=14
        )  # round to make sure any 0s are actually zero
        data_configurations["current_specification"] = "iota"
        data_desc_runs["current_specification"] = "iota"
    elif eq.current and current:
        data_desc_runs["current_profile"] = eq.current(rho)
        data_desc_runs["iota_profile"] = round(
            eq.compute("iota", grid=rho_grid)["iota"], ndigits=14
        )
        data_desc_runs["iota_max"] = np.max(
            eq.compute("iota", grid=rho_grid_dense)["iota"]
        )
        data_desc_runs["iota_min"] = np.min(
            eq.compute("iota", grid=rho_grid_dense)["iota"]
        )
        data_configurations["current_specification"] = "net enclosed current"
        data_desc_runs["current_specification"] = "net enclosed current"
    Dmerc = eq.compute("D_Mercier", grid=rho_grid)["D_Mercier"]
    data_desc_runs["D_Mercier_max"] = np.max(Dmerc)
    data_desc_runs["D_Mercier_min"] = np.min(Dmerc)
    data_desc_runs["D_Mercier"] = Dmerc

    data_desc_runs["iota_min"] = np.min(eq.compute("iota", grid=rho_grid_dense)["iota"])
    data_desc_runs["pressure_profile"] = eq.pressure(rho)
    data_desc_runs["pressure_max"] = np.max(eq.pressure(rho_dense))
    data_desc_runs["pressure_min"] = np.min(eq.pressure(rho_dense))

    today = date.today()
    data_desc_runs["date_created"] = kwargs.get("date_created", today)
    data_desc_runs["date_updated"] = kwargs.get("date_updated", today)
    if user_created is not None:
        data_desc_runs["user_created"] = user_created
    if user_updated is not None:
        data_desc_runs["user_updated"] = user_updated
    # FIXME: publicationid should exist in the database

    ############ configuration Data Table ############
    data_configurations["configid"] = name  # FIXME what should this be? how to hash?
    data_configurations["name"] = name
    data_configurations["NFP"] = eq.NFP
    data_configurations["stell_sym"] = bool(eq.sym)

    if kwargs.get("deviceid", None) is not None:
        data_configurations["deviceid"] = kwargs.get("deviceid", None)

    # FIXME: Defaults for these?
    if provenance is not None:
        data_configurations["provenance"] = provenance
    if description is not None:
        data_configurations["description"] = description

    data_configurations["toroidal_flux"] = eq.Psi
    data_configurations["aspect_ratio"] = eq.compute("R0/a")["R0/a"]
    data_configurations["minor_radius"] = eq.compute("a")["a"]
    data_configurations["major_radius"] = eq.compute("R0")["R0"]
    data_configurations["volume"] = eq.compute("V")["V"]
    data_configurations["volume_averaged_B"] = eq.compute("<|B|>_vol")["<|B|>_vol"]
    data_configurations["volume_averaged_beta"] = eq.compute("<beta>_vol")["<beta>_vol"]
    data_configurations["total_toroidal_current"] = float(
        f'{eq.compute("current")["current"][-1]:1.2e}'
    )
    position_data = eq.compute(["R", "Z", "a_major/a_minor"])
    data_configurations["R_excursion"] = float(
        f'{np.max(position_data["R"])-np.min(position_data["R"]):1.4e}'
    )
    data_configurations["Z_excursion"] = float(
        f'{np.max(position_data["Z"])-np.min(position_data["Z"]):1.4e}'
    )
    data_configurations["average_elongation"] = float(
        f'{np.mean(position_data["a_major/a_minor"]):1.4e}'
    )
    if kwargs.get("config_class", None) is not None:
        data_configurations["class"] = kwargs.get("config_class", None)
    if eq.N == 0:  # is axisymmetric
        data_configurations["class"] = "AS"

    # surface geometry
    # currently saving as VMEC format but I'd prefer if we could do DESC format...

    r1 = np.ones_like(eq.R_lmn)
    r1[eq.R_basis.modes[:, 1] < 0] *= -1
    m, n, x_mn = zernike_to_fourier(
        r1 * eq.R_lmn, basis=eq.R_basis, rho=np.array([1.0])
    )
    xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)

    data_configurations["m"] = xm
    data_configurations["n"] = xn

    data_configurations["RBC"] = c[0, :]
    if not eq.sym:
        data_configurations["RBS"] = s[0, :]
    else:
        data_configurations["RBS"] = np.zeros_like(c)
    # Z
    z1 = np.ones_like(eq.Z_lmn)
    z1[eq.Z_basis.modes[:, 1] < 0] *= -1
    m, n, x_mn = zernike_to_fourier(
        z1 * eq.Z_lmn, basis=eq.Z_basis, rho=np.array([1.0])
    )
    xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
    data_configurations["ZBS"] = s
    if not eq.sym:
        data_configurations["ZBC"] = c
    else:
        data_configurations["ZBC"] = np.zeros_like(s)

    # profiles
    # TODO: make dict of different classes of Profile and
    # the corresponding type of profile, to support more than just
    # power series
    data_configurations["pressure_profile_type"] = "power_series"
    data_configurations["pressure_profile_data1"] = eq.pressure.basis.modes[
        :, 0
    ]  # these are the mode numbers
    data_configurations[
        "pressure_profile_data2"
    ] = eq.pressure.params  # these are the coefficients

    if eq.current:
        data_configurations["current_profile_type"] = "power_series"
        data_configurations["current_profile_data1"] = eq.current.basis.modes[
            :, 0
        ]  # these are the mode numbers
        data_configurations[
            "current_profile_data2"
        ] = eq.current.params  # these are the coefficients

    elif eq.iota:
        data_configurations["iota_profile_type"] = "power_series"
        data_configurations["iota_profile_data1"] = eq.iota.basis.modes[
            :, 0
        ]  # these are the mode numbers
        data_configurations[
            "iota_profile_data2"
        ] = eq.iota.params  # these are the coefficients

    data_configurations["date_created"] = kwargs.get("date_created", today)
    data_configurations["date_updated"] = kwargs.get("date_updated", today)
    if user_created is not None:
        data_configurations["user_created"] = user_created
    if user_updated is not None:
        data_configurations["user_updated"] = user_updated

    csv_columns_desc_runs = list(data_desc_runs.keys())
    csv_columns_desc_runs.sort()
    desc_runs_csv_exists = os.path.isfile(desc_runs_csv_name)

    try:
        with open(desc_runs_csv_name, "a+") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns_desc_runs)
            if not desc_runs_csv_exists:
                writer.writeheader()  # only need header if file did not exist already
            writer.writerow(data_desc_runs)
    except OSError:
        print("I/O error")
    csv_columns_configurations = list(data_configurations.keys())
    csv_columns_configurations.sort()

    configurations_csv_exists = os.path.isfile(configurations_csv_name)
    try:
        with open(configurations_csv_name, "a+") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns_configurations)
            if not configurations_csv_exists:
                writer.writeheader()  # only need header if file did not exist already
            writer.writerow(data_configurations)
    except OSError:
        print("I/O error")

    return None
