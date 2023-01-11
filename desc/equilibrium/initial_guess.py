"""Functions for computing initial guesses for coordinate surfaces."""

import os

import numpy as np

from desc.backend import jnp
from desc.basis import zernike_radial
from desc.geometry import FourierRZCurve, Surface
from desc.grid import Grid
from desc.io import load
from desc.transform import Transform
from desc.utils import copy_coeffs


def set_initial_guess(eq, *args):  # noqa: C901 - FIXME: simplify this
    """Set the initial guess for the flux surfaces, eg R_lmn, Z_lmn, L_lmn.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to initialize
    args :
        either:
          - No arguments, in which case eq.surface will be scaled for the guess.
          - Another Surface object, which will be scaled to generate the guess.
            Optionally a Curve object may also be supplied for the magnetic axis.
          - Another Equilibrium, whose flux surfaces will be used.
          - File path to a VMEC or DESC equilibrium, which will be loaded and used.
          - Grid and 2-3 ndarrays, specifying the flux surface locations (R, Z, and
            optionally lambda) at fixed flux coordinates. All arrays should have the
            same length. Optionally, an ndarray of shape(k,3) may be passed instead
            of a grid.

    Examples
    --------
    Use existing equil.surface and scales down for guess:

    >>> equil.set_initial_guess()

    Use supplied Surface and scales down for guess. Assumes axis is centroid
    of user supplied surface:

    >>> equil.set_initial_guess(surface)

    Optionally, an interior surface may be scaled by giving the surface a
    flux label:

    >>> surf = FourierRZToroidalSurface(rho=0.7)
    >>> equil.set_initial_guess(surf)

    Use supplied Surface and a supplied Curve for axis and scales between
    them for guess:

    >>> equil.set_initial_guess(surface, curve)

    Use the flux surfaces from an existing Equilibrium:

    >>> equil.set_initial_guess(equil2)

    Use flux surfaces from existing Equilibrium or VMEC output stored on disk:

    >>> equil.set_initial_guess(path_to_saved_DESC_or_VMEC_output)

    Use flux surfaces specified by points:
    nodes should either be a Grid or an ndarray, shape(k,3) giving the locations
    in rho, theta, zeta coordinates. R, Z, and optionally lambda should be
    array-like, shape(k,) giving the corresponding real space coordinates

    >>> equil.set_initial_guess(nodes, R, Z, lambda)

    """
    nargs = len(args)
    if nargs > 4:
        raise ValueError(
            "set_initial_guess should be called with 4 or fewer arguments."
        )
    if nargs == 0 or nargs == 1 and args[0] is None:
        if hasattr(eq, "_surface"):
            # use whatever surface is already assigned
            if hasattr(eq, "_axis"):
                axisR = np.array([eq._axis.R_basis.modes[:, -1], eq._axis.R_n]).T
                axisZ = np.array([eq._axis.Z_basis.modes[:, -1], eq._axis.Z_n]).T
            else:
                axisR = None
                axisZ = None
            coord = eq.surface.rho if hasattr(eq.surface, "rho") else None
            eq.R_lmn = _initial_guess_surface(
                eq.R_basis,
                eq.Rb_lmn,
                eq.surface.R_basis,
                axisR,
                "lcfs",
                coord,
            )
            eq.Z_lmn = _initial_guess_surface(
                eq.Z_basis,
                eq.Zb_lmn,
                eq.surface.Z_basis,
                axisZ,
                "lcfs",
                coord,
            )
        else:
            raise ValueError(
                "set_initial_guess called with no arguments, "
                + "but no surface is assigned."
            )
    else:  # nargs > 0
        if isinstance(args[0], Surface):
            surface = args[0]
            if nargs > 1:
                if isinstance(args[1], FourierRZCurve):
                    axis = args[1]
                    axisR = np.array([axis.R_basis.modes[:, -1], axis.R_n]).T
                    axisZ = np.array([axis.Z_basis.modes[:, -1], axis.Z_n]).T
                else:
                    raise TypeError(
                        "Don't know how to initialize from object type {}".format(
                            type(args[1])
                        )
                    )
            else:
                axisR = None
                axisZ = None
            coord = surface.rho if hasattr(surface, "rho") else None
            eq.R_lmn = _initial_guess_surface(
                eq.R_basis,
                surface.R_lmn,
                surface.R_basis,
                axisR,
                coord=coord,
            )
            eq.Z_lmn = _initial_guess_surface(
                eq.Z_basis,
                surface.Z_lmn,
                surface.Z_basis,
                axisZ,
                coord=coord,
            )
        elif type(args[0]) is type(eq):
            eq1 = args[0]
            if nargs > 1:
                raise ValueError(
                    "set_initial_guess got unknown additional argument {}.".format(
                        args[1]
                    )
                )
            eq.R_lmn = copy_coeffs(eq1.R_lmn, eq1.R_basis.modes, eq.R_basis.modes)
            eq.Z_lmn = copy_coeffs(eq1.Z_lmn, eq1.Z_basis.modes, eq.Z_basis.modes)
            eq.L_lmn = copy_coeffs(eq1.L_lmn, eq1.L_basis.modes, eq.L_basis.modes)
        elif isinstance(args[0], (str, os.PathLike)):
            # from file
            path = args[0]
            file_format = None
            if nargs > 1:
                if isinstance(args[1], str):
                    file_format = args[1]
                else:
                    raise ValueError(
                        "set_initial_guess got unknown additional argument "
                        + "{}.".format(args[1])
                    )
            try:  # is it desc?
                eq1 = load(path, file_format)
            except:  # noqa: E722
                try:  # maybe its vmec
                    from desc.vmec import VMECIO

                    eq1 = VMECIO.load(path)
                except:  # noqa: E722
                    raise ValueError(
                        "Could not load equilibrium from path {}, ".format(path)
                        + "please make sure it is a valid DESC or VMEC equilibrium."
                    )
            if not type(eq1) is type(eq):
                if hasattr(eq1, "equilibria"):  # it's a family!
                    eq1 = eq1[-1]
                else:
                    raise TypeError(
                        "Cannot initialize equilibrium from loaded object of type "
                        + "{}".format(type(eq1))
                    )
            eq.R_lmn = copy_coeffs(eq1.R_lmn, eq1.R_basis.modes, eq.R_basis.modes)
            eq.Z_lmn = copy_coeffs(eq1.Z_lmn, eq1.Z_basis.modes, eq.Z_basis.modes)
            eq.L_lmn = copy_coeffs(eq1.L_lmn, eq1.L_basis.modes, eq.L_basis.modes)

        elif nargs > 2:  # assume we got nodes and ndarray of points
            grid = args[0]
            R = args[1]
            eq.R_lmn = _initial_guess_points(grid, R, eq.R_basis)
            Z = args[2]
            eq.Z_lmn = _initial_guess_points(grid, Z, eq.Z_basis)
            if nargs > 3:
                lmbda = args[3]
                eq.L_lmn = _initial_guess_points(grid, lmbda, eq.L_basis)
            else:
                eq.L_lmn = jnp.zeros(eq.L_basis.num_modes)

        else:
            raise ValueError("Can't initialize equilibrium from args {}.".format(args))
    return eq


def _initial_guess_surface(x_basis, b_lmn, b_basis, axis=None, mode=None, coord=None):
    """Create an initial guess from boundary coefficients and a magnetic axis guess.

    Parameters
    ----------
    x_basis : FourierZernikeBais
        basis of the flux surfaces (for R, Z, or Lambda).
    b_lmn : ndarray, shape(b_basis.num_modes,)
        vector of boundary coefficients associated with b_basis.
    b_basis : Basis
        basis of the boundary surface (for Rb or Zb)
    axis : ndarray, shape(num_modes,2)
        coefficients of the magnetic axis. axis[i, :] = [n, x0].
        Only used for 'lcfs' boundary mode. Defaults to m=0 modes of boundary
    mode : str
        One of 'lcfs', 'poincare'.
        Whether the boundary condition is specified by the last closed flux surface
        (rho=1) or the Poincare section (zeta=0).
    coord : float or None
        Surface label (ie, rho, zeta, etc.) for supplied surface.

    Returns
    -------
    x_lmn : ndarray
        vector of flux surface coefficients associated with x_basis.

    """
    x_lmn = np.zeros((x_basis.num_modes,))
    if mode is None:
        # auto-detect based on mode numbers
        if np.all(b_basis.modes[:, 0] == 0):
            mode = "lcfs"
        elif np.all(b_basis.modes[:, 2] == 0):
            mode = "poincare"
        else:
            raise ValueError("Surface should have either l=0 or n=0")
    if mode == "lcfs":
        if coord is None:
            coord = 1.0
        if axis is None:
            axidx = np.where(b_basis.modes[:, 1] == 0)[0]
            axis = np.array([b_basis.modes[axidx, 2], b_lmn[axidx]]).T
        for k, (l, m, n) in enumerate(b_basis.modes):
            scale = zernike_radial(coord, abs(m), m)
            # index of basis mode with lowest radial power (l = |m|)
            idx0 = np.where((x_basis.modes == [np.abs(m), m, n]).all(axis=1))[0]
            if m == 0:  # magnetic axis only affects m=0 modes
                # index of basis mode with second lowest radial power (l = |m| + 2)
                idx2 = np.where((x_basis.modes == [np.abs(m) + 2, m, n]).all(axis=1))[0]
                ax = np.where(axis[:, 0] == n)[0]
                if ax.size:
                    a_n = axis[ax[0], 1]  # use provided axis guess
                else:
                    a_n = b_lmn[k]  # use boundary centroid as axis
                x_lmn[idx0] = (b_lmn[k] + a_n) / 2 / scale
                x_lmn[idx2] = (b_lmn[k] - a_n) / 2 / scale
            else:
                x_lmn[idx0] = b_lmn[k] / scale

    elif mode == "poincare":
        for k, (l, m, n) in enumerate(b_basis.modes):
            idx = np.where((x_basis.modes == [l, m, n]).all(axis=1))[0]
            x_lmn[idx] = b_lmn[k]

    else:
        raise ValueError("Boundary mode should be either 'lcfs' or 'poincare'.")

    return x_lmn


def _initial_guess_points(nodes, x, x_basis):
    """Create an initial guess based on locations of flux surfaces in real space.

    Parameters
    ----------
    nodes : Grid or ndarray, shape(k,3)
        Locations in flux coordinates where real space coordinates are given.
    x : ndarray, shape(k,)
        R, Z or lambda values at specified nodes.
    x_basis : Basis
        Spectral basis for x (R, Z or lambda)

    Returns
    -------
    x_lmn : ndarray
        Vector of flux surface coefficients associated with x_basis.

    """
    if not isinstance(nodes, Grid):
        nodes = Grid(nodes, sort=False)
    transform = Transform(nodes, x_basis, build=False, build_pinv=True)
    x_lmn = transform.fit(x)
    return x_lmn
