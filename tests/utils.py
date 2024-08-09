"""Functions needed by other tests for computing differences between equilibria."""

import os
import warnings

import numpy as np
from shapely.geometry import Polygon

from desc.grid import Grid, LinearGrid
from desc.vmec import VMECIO


def compute_coords(equil, Nr=10, Nt=8, Nz=None):
    """Computes coordinate values from a given equilibrium."""
    if Nz is None and equil.N == 0:
        Nz = 1
    elif Nz is None:
        Nz = 6

    num_theta = 1000
    num_rho = 1000

    # flux surfaces to plot
    rr = np.linspace(1, 0, Nr, endpoint=False)[::-1]
    rt = np.linspace(0, 2 * np.pi, num_theta)
    rz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
    r_grid = LinearGrid(rho=rr, theta=rt, zeta=rz, NFP=equil.NFP)

    # straight field-line angles to plot
    tr = np.linspace(0, 1, num_rho)
    tt = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
    tz = np.linspace(0, 2 * np.pi / equil.NFP, Nz, endpoint=False)
    t_grid = LinearGrid(rho=tr, theta=tt, zeta=tz, NFP=equil.NFP)

    # Note: theta* (also known as vartheta) is the poloidal straight field-line
    # angle in PEST-like flux coordinates

    # find theta angles corresponding to desired theta* angles
    v_grid = Grid(
        equil.map_coordinates(t_grid.nodes, inbasis=("rho", "theta_PEST", "zeta"))
    )
    r_coords = equil.compute(["R", "Z"], grid=r_grid)
    v_coords = equil.compute(["R", "Z"], grid=v_grid)

    # rho contours
    Rr1 = r_coords["R"].reshape(
        (r_grid.num_theta, r_grid.num_rho, r_grid.num_zeta), order="F"
    )
    Rr1 = np.swapaxes(Rr1, 0, 1)
    Zr1 = r_coords["Z"].reshape(
        (r_grid.num_theta, r_grid.num_rho, r_grid.num_zeta), order="F"
    )
    Zr1 = np.swapaxes(Zr1, 0, 1)

    # vartheta contours
    Rv1 = v_coords["R"].reshape(
        (t_grid.num_theta, t_grid.num_rho, t_grid.num_zeta), order="F"
    )
    Rv1 = np.swapaxes(Rv1, 0, 1)
    Zv1 = v_coords["Z"].reshape(
        (t_grid.num_theta, t_grid.num_rho, t_grid.num_zeta), order="F"
    )
    Zv1 = np.swapaxes(Zv1, 0, 1)

    return Rr1, Zr1, Rv1, Zv1


def area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2):
    """Compute area difference between coordinate curves.

    Parameters
    ----------
    args : ndarray
        R and Z coordinates of constant rho (r) or vartheta (v) contours.
        Arrays should be indexed as [rho,theta,zeta]

    Returns
    -------
    area_rho : ndarray, shape(Nz, Nr)
        normalized area difference of rho contours, computed as the symmetric
        difference divided by the intersection
    area_theta : ndarray, shape(Nt, Nz)
        normalized area difference between vartheta contours, computed as the area
        of the polygon created by closing the two vartheta contours divided by the
        perimeter squared
    """
    assert Rr1.shape == Rr2.shape == Zr1.shape == Zr2.shape
    assert Rv1.shape == Rv2.shape == Zv1.shape == Zv2.shape

    poly_r1 = np.array(
        [
            [Polygon(np.array([R, Z]).T) for R, Z in zip(Rr1[:, :, i], Zr1[:, :, i])]
            for i in range(Rr1.shape[2])
        ]
    )
    poly_r2 = np.array(
        [
            [Polygon(np.array([R, Z]).T) for R, Z in zip(Rr2[:, :, i], Zr2[:, :, i])]
            for i in range(Rr2.shape[2])
        ]
    )
    poly_v = np.array(
        [
            [
                Polygon(np.array([R, Z]).T)
                for R, Z in zip(
                    np.hstack([Rv1[:, :, i].T, Rv2[::-1, :, i].T]),
                    np.hstack([Zv1[:, :, i].T, Zv2[::-1, :, i].T]),
                )
            ]
            for i in range(Rv1.shape[2])
        ]
    )

    diff_rho = np.array(
        [
            poly1.symmetric_difference(poly2).area
            for poly1, poly2 in zip(poly_r1.flat, poly_r2.flat)
        ]
    ).reshape((Rr1.shape[2], Rr1.shape[0]))
    # for some reason shapely sometimes throws a warning here on CI but not locally,
    # see https://github.com/shapely/shapely/issues/1345
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in intersection"
        )
        intersect_rho = np.array(
            [
                poly1.intersection(poly2).area
                for poly1, poly2 in zip(poly_r1.flat, poly_r2.flat)
            ]
        ).reshape((Rr1.shape[2], Rr1.shape[0]))
    area_rho = np.where(
        diff_rho > 0, diff_rho / np.where(intersect_rho != 0, intersect_rho, 1), 0
    )
    area_theta = np.array(
        [poly.area / (poly.length) ** 2 for poly in poly_v.flat]
    ).reshape((Rv1.shape[1], Rv1.shape[2]))
    return area_rho, area_theta


def area_difference_vmec(equil, vmec_data, Nr=10, Nt=8, Nz=None, **kwargs):
    """Compute average normalized area difference between VMEC and DESC equilibria.

    Parameters
    ----------
    equil : Equilibrium
        desc equilibrium to compare
    vmec_data : dict
        dictionary of vmec outputs
    Nr : int, optional
        number of radial surfaces to average over
    Nt : int, optional
        number of vartheta contours to compare
    Nz : int, optional
        Number of zeta planes to compare. If None, use 1 plane for axisymmetric cases
        or 6 for non-axisymmetric.

    Returns
    -------
    area_rho : ndarray, shape(Nz, Nr)
        normalized area difference of rho contours, computed as the symmetric
        difference divided by the intersection
    area_theta : ndarray, shape(Nt, Nz)
        normalized area difference between vartheta contours, computed as the area
        of the polygon created by closing the two vartheta contours divided by the
        perimeter squared

    """
    # 1e-3 tolerance seems reasonable for testing, similar to comparison by eye
    if isinstance(vmec_data, (str, os.PathLike)):
        vmec_data = VMECIO.read_vmec_output(vmec_data)

    signgs = vmec_data["signgs"]
    coords = VMECIO.compute_coord_surfaces(equil, vmec_data, Nr, Nt, Nz, **kwargs)

    Rr1 = coords["Rr_desc"]
    Zr1 = coords["Zr_desc"]
    Rv1 = coords["Rv_desc"]
    Zv1 = coords["Zv_desc"]
    Rr2 = coords["Rr_vmec"]
    Zr2 = coords["Zr_vmec"]
    # need to reverse the order of these due to different sign conventions for theta
    Rv2 = coords["Rv_vmec"][::signgs]
    Zv2 = coords["Zv_vmec"][::signgs]
    area_rho, area_theta = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
    return area_rho, area_theta


def area_difference_desc(eq1, eq2, Nr=10, Nt=8, Nz=None):
    """Compute average normalized area difference between two DESC equilibria.

    Parameters
    ----------
    eq1, eq2 : Equilibrium
        desc equilibria to compare
    Nr : int, optional
        Number of radial surfaces to average over
    Nt : int, optional
        Number of vartheta contours to compare
    Nz : int, optional
        Number of zeta planes to compare. If None, use 1 plane for axisymmetric cases
        or 6 for non-axisymmetric.

    Returns
    -------
    area_rho : ndarray, shape(Nr, Nz)
        normalized area difference of rho contours, computed as the symmetric
        difference divided by the intersection
    area_theta : ndarray, shape(Nt, Nz)
        normalized area difference between vartheta contours, computed as the area
        of the polygon created by closing the two vartheta contours divided by the
        perimeter squared

    """
    Rr1, Zr1, Rv1, Zv1 = compute_coords(eq1, Nr=Nr, Nt=Nt, Nz=Nz)
    Rr2, Zr2, Rv2, Zv2 = compute_coords(eq2, Nr=Nr, Nt=Nt, Nz=Nz)

    area_rho, area_theta = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
    return area_rho, area_theta
