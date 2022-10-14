"""Functions needed by other tests for computing differences between equilibria."""

import os

import numpy as np
from shapely.geometry import Polygon

from desc.grid import Grid, LinearGrid
from desc.vmec import VMECIO


def compute_coords(equil, check_all_zeta=False):
    """Computes coordinate values from a given equilibrium."""
    if equil.N == 0 and not check_all_zeta:
        Nz = 1
    else:
        Nz = 6

    Nr = 10
    Nt = 8
    num_theta = 1000
    num_rho = 1000

    # flux surfaces to plot
    rr = np.linspace(0, 1, Nr)
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
    v_grid = Grid(equil.compute_theta_coords(t_grid.nodes))
    r_coords = equil.compute("R", r_grid)
    v_coords = equil.compute("Z", v_grid)

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
    area_rho : ndarray, shape(Nr, Nz)
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


def area_difference_vmec(equil, vmec_data, Nr=10, Nt=8, **kwargs):
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
    # 1e-3 tolerance seems reasonable for testing, similar to comparison by eye
    if isinstance(vmec_data, (str, os.PathLike)):
        vmec_data = VMECIO.read_vmec_output(vmec_data)

    coords = VMECIO.compute_coord_surfaces(equil, vmec_data, Nr, Nt, **kwargs)

    Rr1 = coords["Rr_desc"]
    Zr1 = coords["Zr_desc"]
    Rv1 = coords["Rv_desc"]
    Zv1 = coords["Zv_desc"]
    Rr2 = coords["Rr_vmec"]
    Zr2 = coords["Zr_vmec"]
    Rv2 = coords["Rv_vmec"]
    Zv2 = coords["Zv_vmec"]
    area_rho, area_theta = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
    return area_rho, area_theta


def area_difference_desc(eq1, eq2, Nr=10, Nt=8, **kwargs):
    """Compute average normalized area difference between two DESC equilibria.

    Parameters
    ----------
    eq1, eq2 : Equilibrium
        desc equilibria to compare
    Nr : int, optional
        number of radial surfaces to average over
    Nt : int, optional
        number of vartheta contours to compare

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
    Rr1, Zr1, Rv1, Zv1 = compute_coords(eq1)
    Rr2, Zr2, Rv2, Zv2 = compute_coords(eq2)

    area_rho, area_theta = area_difference(Rr1, Rr2, Zr1, Zr2, Rv1, Rv2, Zv1, Zv2)
    return area_rho, area_theta
