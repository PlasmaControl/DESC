"""Functions to convert DESC or VMEC output files into .csv files for the Datbase."""
import csv
import os
from datetime import date

import numpy as np

from desc.equilibrium import EquilibriaFamily
from desc.grid import LinearGrid
from desc.io import load
from desc.io.hdf5_io import hdf5Reader
from desc.vmec_utils import ptolemy_identity_rev, zernike_to_fourier


def get_DESC_runid(eq):  # or take in the data from a DESC eq?
    """Take in a DESC equilibrium and return a unique hash for the run."""
    return None


def get_config_hash(eq):  # or take in the data from a DESC eq?
    """Take in a DESC equilibrium and return a unique hash for the configuration."""
    # what to do? first 10 bdry modes and their strengths?
    return None


def desc_to_csv(  # noqa
    eq,
    current=True,
    **kwargs,
):
    """Save DESC as a csv with relevant information.

    Args
    ----
        eq (Equilibrium or str): DESC equilibrium to save or path to .h5 of
         DESC equilibrium to save
        current: bool, if the equilibrium was solved with fixed current or not
            if False, was solved with fixed iota

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
    data_desc_runs[
        "desc_run_ID"
    ] = None  # FIXME what should this be? how to hash? commit ID?
    data_desc_runs["configid"] = kwargs.get(
        "name", None
    )  # FIXME what should this be? how to hash?

    # FIXME: Defaults for these?
    data_desc_runs["provenance"] = kwargs.pop("provenance", None)
    data_desc_runs["description"] = kwargs.pop("description", None)

    data_desc_runs[
        "version"
    ] = version  # this is basically redundant with git commit I think
    data_desc_runs[
        "git_commit"
    ] = version  # this is basically redundant with git commit I think
    data_desc_runs["inputfilename"] = kwargs.pop("inputfilename", None)

    data_desc_runs["initialization_method"] = kwargs.pop(
        "initialization_method", "surface"
    )

    data_desc_runs["l_rad"] = eq.L
    data_desc_runs["l_grid"] = eq.L_grid
    data_desc_runs["m_pol"] = eq.M
    data_desc_runs["m_grid"] = eq.M_grid
    data_desc_runs["n_tor"] = eq.N
    data_desc_runs["N_grid"] = eq.N_grid

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

    data_desc_runs["pressure_profile"] = eq.pressure(rho)
    data_desc_runs["pressure_max"] = np.max(eq.pressure(rho_dense))
    data_desc_runs["pressure_min"] = np.min(eq.pressure(rho_dense))

    today = date.today()
    data_desc_runs["date_created"] = kwargs.get("date_created", today)
    data_desc_runs["user_created"] = kwargs.get(
        "user_created", None
    )  # FIXME: what is this?
    data_desc_runs["date_updated"] = kwargs.get("date_updated", today)
    data_desc_runs["user_updated"] = kwargs.get(
        "user_updated", None
    )  # FIXME: what is this?
    data_desc_runs["publicationid"] = kwargs.get("publicationid", None)

    ############ configuration Data Table ############
    data_configurations["configid"] = kwargs.get(
        "name", None
    )  # FIXME what should this be? how to hash?
    data_configurations["name"] = kwargs.get("name", None)
    data_configurations["NFP"] = eq.NFP
    data_configurations["stell_sym"] = bool(eq.sym)

    data_configurations["deviceid"] = kwargs.get("deviceid", None)
    data_configurations["hashkey"] = get_config_hash(eq)  # FIXME: what is this?

    # FIXME: Defaults for these?
    data_configurations["provenance"] = kwargs.pop("provenance", None)
    data_configurations["description"] = kwargs.pop("description", None)

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
    data_configurations["user_created"] = kwargs.get(
        "user_created", None
    )  # FIXME: what is this?
    data_configurations["date_updated"] = kwargs.get("date_updated", today)
    data_configurations["user_updated"] = kwargs.get(
        "user_updated", None
    )  # FIXME: what is this?

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
