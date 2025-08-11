#import numpy as np
import os

import sys
import functools
import pickle

import jax
import jax.numpy as jnp
from jax import jit, jacfwd

#from netCDF4 import Dataset
#import h5py

from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid

from desc.transform import Transform
from desc.derivatives import Derivative
from desc.geometry import FourierRZToroidalSurface

from desc.magnetic_fields._core import biot_savart_general

from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.utils import flatten_list, cross, dot

from desc.magnetic_fields import FourierCurrentPotentialField


def _compute_magnetic_field_from_Current(Kgrid,
                                         K_at_grid, 
                                         #surface, 
                                         #data,
                                         x_surf,
                                         jac_surf,
                                         #eq,
                                         #Bgrid,
                                         coords,
                                         basis="rpz"):
    
    """Compute magnetic field at a set of points.

    Parameters
    ----------
    K_at_grid : ndarray, shape (num_nodes,3)
        Surface current evaluated at points on a grid, which you want to calculate
        B from, should be in cartesian ("xyz") or cylindrical ("rpz") specifiec
        by "basis" argument
    surface : FourierRZToroidalSurface
        surface object upon which the surface current K_at_grid lies
    coords : array-like shape(N,3) or Grid
        cylindrical or cartesian coordinates to evlauate B at
    grid : Grid,
        source grid upon which to evaluate the surface current density K
    basis : {"rpz", "xyz"}
        basis for input coordinates and returned magnetic field

    Returns
    -------
    field : ndarray, shape(N,3)
        magnetic field at specified points

    """

    #Bdata = eq.compute(["R","phi","Z","n_rho"], grid = Bgrid)
    #coords = jnp.vstack([Bdata["R"],Bdata["phi"],Bdata["Z"]]).T
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=coords[:, 0], y=coords[:, 1])
    
    surface_grid = Kgrid

    # compute and store grid quantities
    # needed for integration
    # TODO: does this have to be xyz, or can it be computed in rpz as well?
    #data = surface.compute(["x", "|e_theta x e_zeta|"], grid=surface_grid, basis="xyz")

    #_rs = xyz2rpz(data["x"])
    _rs = x_surf#xyz2rpz(x_surf)
    _K = K_at_grid

    # surface element, must divide by NFP to remove the NFP multiple on the
    # surface grid weights, as we account for that when doing the for loop
    # over NFP
    #_dV = surface_grid.weights * data["|e_theta x e_zeta|"] / surface_grid.NFP
    _dV = surface_grid.weights * jac_surf / surface_grid.NFP
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi = ( surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        # new coords are just old R,Z at a new phi (bc of discrete NFP symmetry)
        rs = jnp.vstack( (_rs[:, 0], phi, _rs[:, 2]) ).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general( coords, rs, K, _dV, )
        f += fj
        return f
    
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(coords))
    
    if basis == "rpz":
        B = xyz2rpz_vec(B, x=coords[:, 0], y=coords[:, 1])
        
    return B