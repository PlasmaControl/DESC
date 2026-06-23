"""Formatting into BMW and FIELDLINES-style formats for precomputed arrays."""

import os

import h5py
import numpy as np
from netCDF4 import Dataset

from desc.utils import errorif, rpz2xyz, rpz2xyz_vec


def write_bmw_file(
    path,
    B,
    Rmin,
    Rmax,
    Zmin,
    Zmax,
    nR,
    nZ,
    nphi,
    source_data={},
    source_grid=None,
    NFP=1,
    A=None,
    series=3,
):
    """
    Save an array of magnetic field values in the same format as BMW.

    Parameters
    ----------
    path : str
        The filepath to save the magnetic field. Ends with .h5.
    B : array-like, shape (nR*nphi*nZ,3)
        Magnetic field values, assumed to be reshapeable to (nR, nphi, nZ, 3)
    Rmin, Rmax, Zmin, Zmax : float, optional
        Bounds for the R, phi, and Z coordinates of the desired evaluation points
    nR, nZ, nphi : int, optional
        Desired number of evaluation points in the radial, vertical, and toroidal
        directions.
    source_data : dict , optional
        Dictionary containing data used to evaluate the magnetic field and vector
        potential. If supplied, contains the following keys: ["J", "R", "phi", "Z"]
    source_grid : _Grid , optional
        Grid object used to discretize source_data integral. Must be supplied if
        source_data is supplied.
    NFP : int, optional
        Number of field periods
    A : array-like, shape (nR*nphi*nZ,3), optional
        Vector potential values, assumed to be reshapeable to (nR, nphi, nZ, 3)
    series: int, optional
        The BMW series tag that is saved.

    Notes
    -----
    This file accepts A and B as reshapeable to (R,phi,Z) indexing, but
    are later transposed to (phi, Z, R), and likewise J is transposed to be
    indexed by (v,u,s)~(zeta,theta,rho) instead of DESC's ordering of
    (zeta, rho, theta).

    Reference links
    ---------------
    https://ornl-fusion.github.io/Stellarator-Tools-Docs/result_file_bmw.html

    """
    errorif(
        (source_grid is None) and ("J" in source_data.keys()),
        msg="If you wish to save the current density used to evaluate B,"
        "please also include the source grid on which it was calculated.",
    )

    #
    path = os.path.expanduser(path)

    # Reshape magnetic field on grid
    new_shape = (nR, nphi, nZ)
    transpose = (1, 2, 0)  # (R,phi,Z) --> (phi,R,Z)
    B_R = B[:, 0].reshape(new_shape).transpose(transpose)
    B_phi = B[:, 1].reshape(new_shape).transpose(transpose)
    B_Z = B[:, 2].reshape(new_shape).transpose(transpose)

    # Reshape magnetic vector potential on grid
    if A is not None:
        A_R = A[:, 0].reshape(new_shape).transpose(transpose)
        A_phi = A[:, 1].reshape(new_shape).transpose(transpose)
        A_Z = A[:, 2].reshape(new_shape).transpose(transpose)

    # Write BMW-style file
    file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")

    # Create RPZ dimensions
    file.createDimension("r", nR)
    file.createDimension("z", nZ)
    file.createDimension("phi", nphi)

    # Create variables to denote parameterization
    _series = file.createVariable("series", np.int32)
    _series[:] = series
    nfp = file.createVariable("nfp", np.int32)
    nfp[:] = NFP
    rmin = file.createVariable("rmin", np.float64)
    rmin[:] = Rmin
    rmax = file.createVariable("rmax", np.float64)
    rmax[:] = Rmax
    zmin = file.createVariable("zmin", np.float64)
    zmin[:] = Zmin
    zmax = file.createVariable("zmax", np.float64)
    zmax[:] = Zmax

    rpz = ("phi", "z", "r")
    # If given, create a variable for the vector potential
    if A is not None:
        ar = file.createVariable("ar_grid", np.float64, rpz)
        ar[:] = A_R
        ap = file.createVariable("ap_grid", np.float64, rpz)
        ap[:] = A_phi
        az = file.createVariable("az_grid", np.float64, rpz)
        az[:] = A_Z

    # Create a variable for the magnetic field
    br = file.createVariable("br_grid", np.float64, rpz)
    br[:] = B_R
    bp = file.createVariable("bp_grid", np.float64, rpz)
    bp[:] = B_phi
    bz = file.createVariable("bz_grid", np.float64, rpz)
    bz[:] = B_Z

    # If given, create a variable for the source grid data
    if "J" in source_data.keys():
        num_s = source_grid.num_rho
        num_u = source_grid.num_theta
        num_v = source_grid.num_zeta
        file.createDimension("s", num_s)
        file.createDimension("u", num_u)
        file.createDimension("v", num_v)

        # DESC grids order nodes as (zeta, rho, theta) ~ (v,s,u)
        # BMW outputs J and P as (v,u,s), so we need to swap the last two dimensions
        new_shape = (
            source_grid.num_zeta,
            source_grid.num_rho,
            source_grid.num_theta,
            3,
        )
        transpose = (0, 2, 1, 3)

        source_rpz = np.stack(
            (source_data["R"], source_data["phi"], source_data["Z"]), axis=-1
        )
        source_xyz = rpz2xyz(source_rpz).reshape(new_shape).transpose(transpose)
        X = source_xyz[..., 0]
        Y = source_xyz[..., 1]
        Z = source_xyz[..., 2]

        # Create a variable for the XYZ coordinates of the source grid
        rtz = ("v", "u", "s")
        px_grid = file.createVariable("px_grid", np.float64, rtz)
        px_grid[:] = X
        py_grid = file.createVariable("py_grid", np.float64, rtz)
        py_grid[:] = Y
        pz_grid = file.createVariable("pz_grid", np.float64, rtz)
        pz_grid[:] = Z

        # Create a variable for the current density used to compute B
        J = rpz2xyz_vec(source_data["J"], phi=source_data["phi"])
        J = J.reshape(new_shape).transpose(transpose)
        jx_grid = file.createVariable("jx_grid", np.float64, rtz)
        jx_grid[:] = J[..., 0]
        jy_grid = file.createVariable("jy_grid", np.float64, rtz)
        jy_grid[:] = J[..., 1]
        jz_grid = file.createVariable("jz_grid", np.float64, rtz)
        jz_grid[:] = J[..., 2]

    file.close()


def write_fieldlines_file(
    path,
    B,
    Rmin,
    Rmax,
    Zmin,
    Zmax,
    phi_min,
    phi_max,
    nR,
    nZ,
    nphi,
    pressure=None,
):
    """
    Save an array of magnetic field values as a FIELDLINES-style file.

    Parameters
    ----------
    path : str
        The filepath to save the magnetic field. Ends with .h5.
    B : array-like, shape (nR*nphi*nZ,3)
        Magnetic field values, assumed to be reshapeable to (nR, nphi, nZ, 3)
    Rmin, Rmax, Zmin, Zmax, phimin, phimax : float, optional
        Bounds for the R, phi, and Z coordinates of the desired evaluation points
    nR, nZ, nphi : int, optional
        Desired number of evaluation points in the radial, vertical, and toroidal
        directions.
    pressure : array-like, shape (nR*nphi*nZ), optional
        Pressure values at the same evaluation points as B.

    Notes
    -----
    FIELDLINES requires a minimum and maximum toroidal angle to be supplied, and
    the endpoint is included. Older versions of FIELDLINES normalize B_R and B_Z
    as B_i*R/B_PHI, for i=R,Z, but the most recent version returns B_R and B_Z
    in units of T.

    Reference links
    ---------------
    https://princetonuniversity.github.io/STELLOPT/FIELDLINES.html#output-data-format
    https://github.com/PrincetonUniversity/STELLOPT/blob/7a03761db9c408902669b3a34823ef2f3d6dba0e/pySTEL/libstell/fieldlines.py
    https://github.com/PrincetonUniversity/STELLOPT/commit/30bce7c339e66fa1ea1e1be4043fe8a19bdae400
    https://www.mathworks.com/matlabcentral/fileexchange/54931-read_fieldlines
    """
    # Reshape magnetic field on grid
    B = B.reshape(nR, nphi, nZ, 3)
    lpres = pressure is not None
    if lpres:
        pressure = pressure.reshape(nR, nphi, nZ)
    else:
        pressure = np.zeros((nR, nphi, nZ))

    save_data = [
        {
            "key": "B_PHI",
            "description": np.bytes_(b"Toroidal Field (BPHI)"),
            "value": B[..., 1],
        },
        {
            "key": "B_R",
            "description": np.bytes_(b"Radial Fieldline Eq. (BR)"),
            "value": B[..., 0],
        },
        {
            "key": "B_Z",
            "description": np.bytes_(b"Vertical Fieldline Eq. (BZ)"),
            "value": B[..., 2],
        },
        {
            "key": "PRES",
            "description": np.bytes_(b"Plasma Pressure (PRES)"),
            "value": pressure,
        },
        {
            "key": "VERSION",
            "description": np.bytes_(b"Version Number"),
            "value": np.array([1.79999995]),
        },
        {
            "key": "ladvanced",
            "description": np.bytes_(b"Advanced Grid Flag"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "laxis_i",
            "description": np.bytes_(b"Axis calc"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lcoil",
            "description": np.bytes_(b"Coil input"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "leqdsk",
            "description": np.bytes_(b"EQDSK input"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lhint",
            "description": np.bytes_(b"HINT input"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lmgrid",
            "description": np.bytes_(b"MGRID input"),
            "value": np.array([1], dtype=np.int32),
        },
        {
            "key": "lmu",
            "description": np.bytes_(b"Diffusion Logical"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lpies",
            "description": np.bytes_(b"PIES input"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lpres",
            "description": np.bytes_(b"Pressure output"),
            "value": np.array([lpres], dtype=np.int32),
        },
        {
            "key": "lreverse",
            "description": np.bytes_(b"VMEC input"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lspec",
            "description": np.bytes_(b"SPEC input"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lvac",
            "description": np.bytes_(b"Vacuum calc"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lvessel",
            "description": np.bytes_(b"Vessel input"),
            "value": np.array([0], dtype=np.int32),
        },
        {
            "key": "lvmec",
            "description": np.bytes_(b"VMEC input"),
            "value": np.array([1], dtype=np.int32),
        },
        {
            "key": "nphi",
            "description": np.bytes_(b"Number of Toroidal Gridpoints"),
            "value": np.array([nphi], dtype=np.int32),
        },
        {
            "key": "nr",
            "description": np.bytes_(b"Number of Radial Gridpoints"),
            "value": np.array([nR], dtype=np.int32),
        },
        {
            "key": "nz",
            "description": np.bytes_(b"Number of Vertical Gridpoints"),
            "value": np.array([nZ], dtype=np.int32),
        },
        {
            "key": "phiaxis",
            "description": np.bytes_(b"Toroidal Axis [rad]"),
            "value": np.linspace(phi_min, phi_max, nphi, endpoint=True),
        },
        {
            "key": "raxis",
            "description": np.bytes_(b"Radial Axis [m]"),
            "value": np.linspace(Rmin, Rmax, nR, endpoint=True),
        },
        {
            "key": "zaxis",
            "description": np.bytes_(b"Vertical Axis [m]"),
            "value": np.linspace(Zmin, Zmax, nZ, endpoint=True),
        },
    ]
    f = h5py.File(path, "w")

    def add_ds(key, value, description):
        if value.ndim > 1:
            # FIELDLINES is written in Fortran using R,phi,Z coordinates,
            # but since arrays are stored in column-major order in Fortran,
            # it's tranposed when loaded into python, where arrays are stored
            # in row-major order. See the first Github link
            T = tuple(np.flip(np.arange(value.ndim)))
            value = value.transpose(T)
        dset = f.create_dataset(key, data=value)
        dset.attrs[f"{key}_description"] = description

    for ds in save_data:
        add_ds(ds["key"], np.asarray(ds["value"]), ds["description"])

    f.close()
