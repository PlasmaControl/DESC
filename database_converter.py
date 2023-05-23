"""Functions to convert DESC or VMEC output files into .csv files for the Datbase."""

from datetime import date

import numpy as np

from desc.equilibrium import EquilibriaFamily
from desc.grid import LinearGrid
from desc.io import load
from desc.io.hdf5_io import hdf5Reader
from desc.vmec_utils import (
    fourier_to_zernike,
    ptolemy_identity_fwd,
    ptolemy_identity_rev,
    zernike_to_fourier,
)


def get_DESC_runid(eq):  # or take in the data from a DESC eq?
    """Take in a DESC equilibrium and return a unique hash for the run.

    Args:
    ----
        eq (_type_): _description_
    """
    return None


def get_config_hash(eq):  # or take in the data from a DESC eq?
    """Take in a DESC equilibrium and return a unique hash for the configuration.

    Args:
    ----
        eq (_type_): _description_
    """

    # what to do? first 10 bdry modes and their strengths?

    return None


def desc_to_csv(
    eq,
    filename=None,
    inputfilename=None,
    publications=None,
    version=None,
    **kwargs,
):
    """Save DESC as a csv with relevant information.

    Args
    ----
        eq (Equilibrium or str): DESC equilibrium to save or path to .h5 of DESC equilibrium to save
        filename: name of csv file to save to
        inputfilename (str): Path to DESC input file or script used to create the DESC equilibrium

        filename: name of csv file to save to


    Returns
    -------
        None
    """
    # data dicts for each table
    data_desc_runs = {}
    data_configurations = {}

    if isinstance(eq, str):
        data_desc_runs["outputfile"] = eq
        eq = load(eq)
        reader = hdf5Reader(eq)
        version = reader.read_dict()["__version__"]
    elif isinstance(eq, EquilibriaFamily):
        eq = eq[-1]  # just use last equilibrium
        runid = None
        savename = kwargs.pop("savename", f"DESC_eq_runid_{runid}.h5")
        data_desc_runs["outputfile"] = savename
        from desc import __version__ as version

    ############ DESC_runs Data Table ############
    # FIXME: what to do for these?
    data_desc_runs[
        "desc_run_ID"
    ] = None  # FIXME what should this be? how to hash? commit ID?
    data_desc_runs["configid"] = kwargs.get(
        "configid", None
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
    if inputfilename:
        data_desc_runs["inputfilename"] = inputfilename
    else:
        pass  # make an input file? though this would not necessarily correlate to the same exact equilibrium

    data_desc_runs["initialization_method"] = kwargs.pop(
        "initialization_method", None
    )  # we don't usually store this information in the DESC equilibrium

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
    # TODO: note which ones were fixed for the solve? or only include what was fixed in the solve?

    rho = np.linspace(0, 1.0, 10, endpoint=True)
    if eq.iota:
        data_desc_runs["iota_profile"] = eq.iota(rho)
    else:
        data_desc_runs["current_profile"] = eq.current(rho)

    data_desc_runs["pressure_profile"] = eq.pressure(rho)

    today = date.today()
    data_desc_runs["date_created"] = kwargs.get("date_created", today)
    data_desc_runs["user_created"] = kwargs.get(
        "user_created", None
    )  # FIXME: what is this?
    data_desc_runs["date_updated"] = kwargs.get("date_updated", today)
    data_desc_runs["user_updated"] = kwargs.get(
        "user_updated", None
    )  # FIXME: what is this?
    data_desc_runs["publicationid"] = kwargs.get(
        "publicationid", None
    )  # FIXME: what is this?

    ############ configuration Data Table ############
    data_configurations["configid"] = kwargs.get(
        "configid", None
    )  # FIXME what should this be? how to hash?
    data_configurations["name"] = kwargs.get("name", None)
    data_configurations["deviceid"] = kwargs.get("name", None)
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
    data_configurations["total_toroidal_current"] = eq.compute("current")["current"][-1]

    # surface geometry
    # currently saving as VMEC format but I'd prefer if we could do DESC format...

    xm = np.tile(np.linspace(0, M, M + 1), (2 * N + 1, 1)).T.flatten()[
        -file.dimensions["mn_mode"].size :
    ]

    data_configurations["RBC"] = eq.surface.R_lmn

    r1 = np.ones_like(eq.R_lmn)
    r1[eq.R_basis.modes[:, 1] < 0] *= -1
    m, n, x_mn = zernike_to_fourier(
        r1 * eq.R_lmn, basis=eq.R_basis, rho=np.array([1.0])
    )
    xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
    data_configurations["RBC"] = c
    if not eq.sym:
        data_configurations["RBS"] = s
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

    return None


# TODO: other fields?
# provenance could be used to include if the file was based off a
# NAE equilibrium, but maybe if this is supposed to be more
# common we could also include the NAE parameters...

# Aza would ideally like one CSV per input
# can take multiple inputs at a time
# then just output to a few csvs

# one CVS per table


# CSV
# headers of columsn are the different IDs etc
# each row is a different input (so this fxn will append to these CSVs)
# one CSV for desc_runs
# one CSV for configurations

# if there are attributes I think should be added, I can add these here

# use this for saving https://www.tutorialspoint.com/How-to-save-a-Python-Dictionary-to-CSV-file

# or maybe this one https://www.geeksforgeeks.org/how-to-append-a-new-row-to-an-existing-csv-file/
# check that the file exists, if not do first way, else open it and append a row with this second way
"""from csv import writer

# List that we want to add as a new row
List = [6, 'William', 5532, 1, 'UAE']

# Open our existing CSV file in append mode
# Create a file object for this file
with open('event.csv', 'a') as f_object:

    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(List)

    # Close the file object
    f_object.close()
"""
# so make fieldnames from list(dict.keys()) probably to make it easiest on me
