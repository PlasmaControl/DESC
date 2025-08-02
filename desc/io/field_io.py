import os
import numpy as np
from netCDF4 import Dataset
from desc.compute.geom_utils import rpz2xyz, rpz2xyz_vec


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
    if source_grid is None:
        num_rho = 33
        num_theta = 129
        num_zeta = 129
    else:
        num_rho = source_grid.num_rho
        num_theta = source_grid.num_theta
        num_zeta = source_grid.num_zeta

    # MAKE SURE ORDERING/DIMS IS CORRECT AND ALIGNED WITH BMW
    path = os.path.expanduser(path)

    # cylindrical coordinates grid
    B_R = B[:, 0].reshape(nphi, nZ, nR)
    B_phi = B[:, 1].reshape(nphi, nZ, nR)
    B_Z = B[:, 2].reshape(nphi, nZ, nR)

    # evaluate magnetic vector potential on grid
    if A is not None:
        A_R = A[:, 0].reshape(nphi, nZ, nR)
        A_phi = A[:, 1].reshape(nphi, nZ, nR)
        A_Z = A[:, 2].reshape(nphi, nZ, nR)
    else:
        A_R = None

    # write BMW-style file
    file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")

    # dimensions
    file.createDimension("r", nR)
    file.createDimension("z", nZ)
    file.createDimension("phi", nphi)
    file.createDimension("s", num_rho)
    file.createDimension("u", num_theta)
    file.createDimension("v", num_zeta)

    # variables
    _series = file.createVariable("series", np.int64)
    _series[:] = series
    nfp = file.createVariable("nfp", np.int64)
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
    if A is not None:
        ar = file.createVariable("ar_grid", np.float64, rpz)
        ar[:] = A_R
        ap = file.createVariable("ap_grid", np.float64, rpz)
        ap[:] = A_phi
        az = file.createVariable("az_grid", np.float64, rpz)
        az[:] = A_Z

    br = file.createVariable("br_grid", np.float64, rpz)
    br[:] = B_R
    bp = file.createVariable("bp_grid", np.float64, rpz)
    bp[:] = B_phi
    bz = file.createVariable("bz_grid", np.float64, rpz)
    bz[:] = B_Z

    if "J" in source_data.keys():
        source_rpz = np.vstack((source_data["R"], source_data["phi"], source_data["Z"])).T
        source_xyz = rpz2xyz(source_rpz)
        X = source_xyz[:, 0]
        Y = source_xyz[:, 1]
        Z = source_xyz[:, 2]

        rtz = ("v", "u", "s")
        px_grid = file.createVariable("px_grid", np.float64, rtz)
        px_grid[:] = X
        py_grid = file.createVariable("py_grid", np.float64, rtz)
        py_grid[:] = Y
        pz_grid = file.createVariable("pz_grid", np.float64, rtz)
        pz_grid[:] = Z

        J = rpz2xyz_vec(source_data["J"],phi=source_data["phi"])
        jx_grid = file.createVariable("jx_grid", np.float64, rtz)
        jx_grid[:] = J[:,0]
        jy_grid = file.createVariable("jy_grid", np.float64, rtz)
        jy_grid[:] = J[:,1]
        jz_grid = file.createVariable("jz_grid", np.float64, rtz)
        jz_grid[:] = J[:,2]
    
    file.close()
