"""Functions to convert DESC or VMEC output files into .csv files for the Datbase."""
import csv
import os
from datetime import date

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from simsopt.mhd.vmec import Vmec

# TODO: add threshold to truncate at what amplitude surface Fourier coefficient
# that it is working

# TODO: make arrays stored in one line

# TODO: either make separate utilities for desc_runs csv and configurations csv,
# or have the utility somehow check for if the configuration exists already,
# or add configid to the arguments so that if it
# is passed in, a new row in configuration
# won't be created and instead the configid will be used, which points to
# an existing configuration in the database


def vmec_to_csv(  # noqa
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
    """Save VMEC output file as a csv with relevant information.

    Parameters
    ----------
        eq : str
            VMEC equilibrium to save or path to .h5 of VMEC equilibrium to save
        current : bool
            True if the equilibrium was solved with fixed current or not if False,
            was solved with fixed iota
        name : str
            name of configuration (and VMEC run)
        provenance : str
            where this configuration (and VMEC run) came from, e.g. VMEC github repo
        description : str
            description of the configuration (and VMEC run)
        inputfilename : str
            name of the input file corresponding to this configuration (and VMEC run)
        initialization_method : str
            how the VMEC equilibrium solution was initialized
            one of "surface", "NAE", or the name of a .nc or .h5 file
            corresponding to a VMEC (if .nc) or DESC (if .h5) solution

    Kwargs
    ------
        date_created : str
            when the VMEC run was created, defaults to current day
        publicationid : str
            unique ID for a publication which this VMEC output file is associated with.
        deviceid : str
            unique ID for a device/concept which this configuration is associated with.
        config_class : str
            class of configuration i.e. quasisymmetry (QA, QH, QP)
            or omnigenity (QI, OT, OH) or axisymmetry (AS).
            Defaults to None for a stellarator and (AS) for a tokamak
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
    data_vmec_runs = {}
    data_configurations = {}

    vmec_runs_csv_name = "vmec_runs.csv"
    configurations_csv_name = "configurations.csv"

    if isinstance(eq, str):
        file_name = eq
        data_vmec_runs["outputfile"] = file_name
        vmec = Vmec(file_name)
        vmec_wout = vmec.wout
        version = vmec_wout.version_
        eq = vmec_wout
    elif isinstance(eq, Vmec):
        # Assuming that the equilibrium had been run, and thus wout is not empty
        vmec = eq
        eq = vmec.wout
    else:
        raise TypeError("Wrong VMEC file or object was passed!")

    ############ VMEC_runs Data Table ############
    data_vmec_runs["configid"] = name
    if provenance is not None:
        data_vmec_runs["provenance"] = provenance
    if description is not None:
        data_vmec_runs["description"] = description

    data_vmec_runs["version"] = version  # version property in wout
    data_vmec_runs["git_commit"] = version  # not sure how this is standarised for vmec
    if inputfilename is not None:
        data_vmec_runs["inputfilename"] = inputfilename

    data_vmec_runs[
        "initialization_method"
    ] = initialization_method  # Not sure what this is

    data_vmec_runs["ns"] = eq.ns
    data_vmec_runs["mpol"] = eq.mpol
    data_vmec_runs["ntor"] = eq.ntor

    # save profiles

    s_full = eq.phi / eq.phi[-1]
    data_vmec_runs["profile_s"] = s_full
    s_dense = np.linspace(0.0, 1.0, 101)

    # This is how iota is computed in vmec_splines in simsopt
    iota = InterpolatedUnivariateSpline(vmec.s_half_grid, eq.iotas[1:])
    data_vmec_runs["iota_profile"] = iota(s_full)  # sohuld name differently

    iota_dense = iota(s_dense)
    data_vmec_runs["iota_max"] = np.max(iota_dense)
    data_vmec_runs["iota_min"] = np.min(iota_dense)

    # Not sure what current is wanted here: vmec outputs many.
    # I guess is the profile derived from ac
    data_vmec_runs["ac"] = eq.ac
    data_vmec_runs["current_profile"] = np.polyval(
        eq.ac[::-1], s_full
    )  # round to make sure any 0s are actually zero
    if current:
        data_configurations["current_specification"] = "net enclosed current"
    else:
        data_configurations["current_specification"] = "iota"

    # It seems that DMerc is not being read or computed properly from the in/from wout
    Dmerc = eq.DMerc
    data_vmec_runs["D_Mercier_max"] = np.max(Dmerc)
    data_vmec_runs["D_Mercier_min"] = np.min(Dmerc)
    data_vmec_runs["D_Mercier"] = Dmerc

    # This is how iota is computed in vmec_splines in simsopt
    pressure = InterpolatedUnivariateSpline(vmec.s_half_grid, eq.pres[1:])
    data_vmec_runs["pressure_profile"] = pressure(s_full)
    pressure_dense = pressure(s_dense)
    data_vmec_runs["pressure_max"] = np.max(pressure_dense)
    data_vmec_runs["pressure_min"] = np.min(pressure_dense)

    today = date.today()
    data_vmec_runs["date_created"] = kwargs.get("date_created", today)
    data_vmec_runs["date_updated"] = kwargs.get("date_updated", today)
    if kwargs.get("publicationid", None) is not None:
        data_vmec_runs["publicationid"] = kwargs.get("publicationid", None)
    if user_created is not None:
        data_vmec_runs["user_created"] = user_created
    if user_updated is not None:
        data_vmec_runs["user_updated"] = user_updated

    ############ configuration Data Table ############
    data_configurations["configid"] = name  # FIXME what should this be? how to hash?
    data_configurations["name"] = name
    data_configurations["NFP"] = eq.nfp
    data_configurations["stell_sym"] = not eq.lasym

    if kwargs.get("deviceid", None) is not None:
        data_configurations["deviceid"] = kwargs.get("deviceid", None)

    if provenance is not None:
        data_configurations["provenance"] = provenance
    if description is not None:
        data_configurations["description"] = description

    data_configurations["toroidal_flux"] = eq.phi[-1]
    data_configurations["aspect_ratio"] = eq.aspect
    data_configurations["minor_radius"] = eq.Aminor_p
    data_configurations["major_radius"] = eq.Rmajor_p
    data_configurations["volume"] = eq.volume
    data_configurations["volume_averaged_B"] = eq.volavgB
    data_configurations[
        "volume_averaged_beta"
    ] = (
        eq.betatotal
    )  # FIXME: I am assuming betatot is this (there are various beta outputs in vmec)
    data_configurations["total_toroidal_current"] = eq.ctor

    theta = np.linspace(0, 1.0, 101) * 2 * np.pi
    phi = np.linspace(0, 1.0, 101) * 2 * np.pi
    [theta_grid, phi_grid] = np.meshgrid(theta, phi)
    xm = eq.xm
    xn = eq.xn
    angle = (
        xm[:, None, None] * theta_grid[None, :, :]
        - xn[:, None, None] * phi_grid[None, :, :]
    )
    rmnc = eq.rmnc[:, -1]
    zmns = eq.zmns[:, -1]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    R = np.einsum("i,ikl->ikl", rmnc, cosangle)
    Z = np.einsum("i,ikl->ikl", zmns, sinangle)
    if eq.lasym:
        rmns = eq.rmns[:, -1]
        zmnc = eq.zmnc[:, -1]
        R += np.einsum("i,ikl->ikl", rmns, sinangle)
        Z += np.einsum("i,ikl->ikl", zmnc, cosangle)
    data_configurations["R_excursion"] = float(f"{np.max(R)-np.min(R):1.4e}")
    data_configurations["Z_excursion"] = float(f"{np.max(Z)-np.min(Z):1.4e}")

    # Not sure how you are computing the average elongation: all the R & Z info is above

    data_configurations["class"] = kwargs.get("config_class", None)
    if eq.ntor == 0:  # is axisymmetric
        data_configurations["class"] = "AS"

    # surface geometry
    # currently saving as VMEC format but I'd prefer if we could do DESC format...

    data_configurations["m"] = xm
    data_configurations["n"] = xn

    data_configurations["RBC"] = rmnc
    if eq.lasym:
        data_configurations["RBS"] = rmns
    else:
        data_configurations["RBS"] = np.zeros_like(rmnc)
    # Z
    data_configurations["ZBS"] = zmns
    if eq.lasym:
        data_configurations["ZBC"] = zmnc
    else:
        data_configurations["ZBC"] = np.zeros_like(zmns)

    # profiles
    # TODO: make dict of different classes of Profile and
    # the corresponding type of profile, to support more than just
    # power series
    data_configurations["pressure_profile_type"] = "power_series"
    data_configurations["pressure_profile_data"] = eq.am  # these are the coefficients

    if current:
        data_configurations["current_profile_type"] = "power_series"
        data_configurations["current_profile_data"] = eq.ac
        data_configurations["iota_profile_type"] = None
        data_configurations["iota_profile_data"] = None
    else:
        data_configurations["iota_profile_type"] = "power_series"
        data_configurations["iota_profile_data"] = eq.ai
        data_configurations["current_profile_type"] = None
        data_configurations["current_profile_data"] = None

    data_configurations["date_created"] = kwargs.get("date_created", today)
    data_configurations["date_updated"] = kwargs.get("date_updated", today)
    if user_created is not None:
        data_configurations["user_created"] = user_created
    if user_updated is not None:
        data_configurations["user_updated"] = user_updated

    csv_columns_desc_runs = list(data_vmec_runs.keys())
    csv_columns_desc_runs.sort()
    desc_runs_csv_exists = os.path.isfile(vmec_runs_csv_name)

    try:
        with open(vmec_runs_csv_name, "a+") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns_desc_runs)
            if not desc_runs_csv_exists:
                writer.writeheader()  # only need header if file did not exist already
            writer.writerow(data_vmec_runs)
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
    """Save concept as a csv with relevant information.

    Args
    ----
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

    csv_columns_runs = list(devices_and_concepts.keys())
    csv_columns_runs.sort()
    runs_csv_exists = os.path.isfile(devices_csv_name)

    try:
        with open(devices_csv_name, "a+") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns_runs)
            if not runs_csv_exists:
                writer.writeheader()  # only need header if file did not exist already
            writer.writerow(devices_and_concepts)
    except OSError:
        print("I/O error")

    return None
