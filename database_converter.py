"""Functions to convert DESC or VMEC output files into .csv files for the Datbase."""

# flake8: noqa
import csv
import os
from datetime import date

import numpy as np

from desc.equilibrium import EquilibriaFamily
from desc.grid import LinearGrid
from desc.io import load
from desc.io.hdf5_io import hdf5Reader
from desc.vmec_utils import ptolemy_identity_rev, zernike_to_fourier


def get_config_hash(eq):  # or take in the data from a DESC eq?
    """Take in a DESC equilibrium and return a unique hash for the configuration."""
    # what to do? first 10 bdry modes and their strengths?
    # name and a number?
    return None


# TODO: add threshold to truncate at what amplitude surface Fourier coefficient
# that it is working

# TODO: make arrays stored in one line

# TODO: either make separate utilities for desc_runs csv and configurations csv,
# or have the utility somehow check for if the configuration exists already,
# or add configid to the arguments so that if it
# is passed in, a new row in configuration
# won't be created and instead the configid will be used, which points to
# an existing configuration in the database


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
    data_desc_runs["configid"] = name  # FIXME what should this be? how to hash?

    # FIXME: Defaults for these?
    data_desc_runs["provenance"] = provenance
    data_desc_runs["description"] = description

    data_desc_runs[
        "version"
    ] = version  # this is basically redundant with git commit I think
    data_desc_runs[
        "git_commit"
    ] = version  # this is basically redundant with git commit I think
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
    # data_desc_runs["publicationid"] = kwargs.get("publicationid", None)
    # FIXME: publicationid should exist in the database

    ############ configuration Data Table ############
    data_configurations["configid"] = name  # FIXME what should this be? how to hash?
    data_configurations["name"] = name
    data_configurations["NFP"] = eq.NFP
    data_configurations["stell_sym"] = bool(eq.sym)

    data_configurations["deviceid"] = kwargs.get("deviceid", None)
    data_configurations["hashkey"] = get_config_hash(eq)  # FIXME: what is this?

    # FIXME: Defaults for these?
    data_configurations["provenance"] = provenance
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
    # data_configurations["average_elongation"] = float(
    #     f'{position_data["a_major/a_minor"]:1.4e}'
    # )
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
        data_configurations["iota_profile_type"] = None
        data_configurations["iota_profile_data1"] = None
        data_configurations["iota_profile_data2"] = None

    elif eq.iota:
        data_configurations["iota_profile_type"] = "power_series"
        data_configurations["iota_profile_data1"] = eq.iota.basis.modes[
            :, 0
        ]  # these are the mode numbers
        data_configurations[
            "iota_profile_data2"
        ] = eq.iota.params  # these are the coefficients
        data_configurations["current_profile_type"] = None
        data_configurations["current_profile_data1"] = None
        data_configurations["current_profile_data2"] = None

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


def device_or_concept_to_csv(  # noqa
    name,
    device_class=None,
    NFP=None,
    description=None,
    stell_sym=False,
    deviceid=None,
):
    """Save DESC as a csv with relevant information.

    Args
    ----
        eq (Equilibrium or str): DESC equilibrium to save or path to .h5 of
         DESC equilibrium to save
        device_class (str): class of device i.e. quasisymmetry (QA, QH, QP)
            or omnigenity (QI, OT, OH) or axisymmetry (AS).
        NFP (int): (Nominal) number of field periods for the device/concept
        description (str): description of the device/concept
        stell_sym (bool): (Nominal) stellarator symmetry of the device
            (stellarator symmetry defined as R(theta, zeta) = R(-theta,-zeta)
            and Z(theta, zeta) = -Z(-theta,-zeta))
        deviceid (str): unique identifier for this device

    Returns
    -------
        None
    """
    # data dicts for each table
    devices_and_concepts = {}

    devices_csv_name = "devices_and_concepts.csv"

    devices_and_concepts["name"] = name
    devices_and_concepts["class"] = device_class

    devices_and_concepts["NFP"] = NFP
    devices_and_concepts["stell_sym"] = bool(stell_sym)

    devices_and_concepts["description"] = description
    devices_and_concepts["deviceid"] = deviceid

    csv_columns_desc_runs = list(devices_and_concepts.keys())
    csv_columns_desc_runs.sort()
    desc_runs_csv_exists = os.path.isfile(devices_csv_name)

    try:
        with open(devices_csv_name, "a+") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns_desc_runs)
            if not desc_runs_csv_exists:
                writer.writeheader()  # only need header if file did not exist already
            writer.writerow(devices_and_concepts)
    except OSError:
        print("I/O error")

    return None
