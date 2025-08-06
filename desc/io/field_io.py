import os
import numpy as np
from netCDF4 import Dataset
from desc.utils import rpz2xyz, rpz2xyz_vec, errorif


def save_bmw_format(
    path,
    B,
    Rmin,
    Rmax,
    Zmin,
    Zmax,
    source_data={},
    source_grid=None,
    nR=101,
    nZ=101,
    nphi=90,
    NFP=1,
    A=None,
    series=3,
):
    """
    BMW format: A and B indexed by (phi, Z, R) and J indexed by (v,u,s)~(zeta,theta,rho).
    https://ornl-fusion.github.io/Stellarator-Tools-Docs/result_file_bmw.html

    """
    errorif(
        (source_grid is None) and ("J" in source_data.keys()),
        msg="If you wish to save the current density used to evaluate B," \
        "please also include the source grid on which it was calculated.",
    )

    # 
    path = os.path.expanduser(path)

    # Reshape magnetic field on grid
    new_shape = (nR,nphi,nZ)
    transpose = (1,2,0) # (R,phi,Z) --> (phi,R,Z)
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
        new_shape = (source_grid.num_zeta, source_grid.num_rho, source_grid.num_theta, 3)
        transpose = (0,2,1,3)
        
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
