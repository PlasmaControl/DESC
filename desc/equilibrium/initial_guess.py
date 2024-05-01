"""Functions for computing initial guesses for coordinate surfaces."""

import os
import warnings

import numpy as np

from desc.backend import fori_loop, jit, jnp, put
from desc.basis import zernike_radial
from desc.geometry import FourierRZCurve, Surface
from desc.grid import Grid, _Grid
from desc.io import load
from desc.objectives import (
    FixThetaSFL,
    GoodCoordinates,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
)
from desc.transform import Transform
from desc.utils import copy_coeffs, warnif


def set_initial_guess(  # noqa: C901 - FIXME: simplify
    eq, *args, ensure_nested=True, lcfs_surface=True
):
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
    ensure_nested : bool
        If True, and the default initial guess does not produce nested surfaces,
        run a small optimization problem to attempt to refine initial guess to improve
        coordinate mapping.

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
        if hasattr(eq, "_surface") and lcfs_surface:
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
                coord,
            )
            eq.Z_lmn = _initial_guess_surface(
                eq.Z_basis,
                eq.Zb_lmn,
                eq.surface.Z_basis,
                axisZ,
                coord,
            )
        elif hasattr(eq, "_xsection") and not lcfs_surface:
            eq.R_lmn = _initial_guess_surface(
                eq.R_basis,
                eq.Rp_lmn,
                eq.xsection.R_basis,
            )
            eq.Z_lmn = _initial_guess_surface(
                eq.Z_basis,
                eq.Zp_lmn,
                eq.xsection.Z_basis,
            )
            eq.L_lmn = _initial_guess_surface(
                eq.L_basis,
                eq.Lp_lmn,
                eq.xsection.L_basis,
            )
        else:
            raise ValueError(
                "set_initial_guess called with no arguments, "
                + "but no surface or cross-section is assigned."
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
            eq.Ra_n = copy_coeffs(
                eq1.Ra_n, eq1.axis.R_basis.modes, eq.axis.R_basis.modes
            )
            eq.Za_n = copy_coeffs(
                eq1.Za_n, eq1.axis.Z_basis.modes, eq.axis.Z_basis.modes
            )
            eq.xsection = eq1.xsection

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
            eq.Ra_n = copy_coeffs(
                eq1.Ra_n, eq1.axis.R_basis.modes, eq.axis.R_basis.modes
            )
            eq.Za_n = copy_coeffs(
                eq1.Za_n, eq1.axis.Z_basis.modes, eq.axis.Z_basis.modes
            )

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

    if ensure_nested and not eq.is_nested():
        warnings.warn(
            "Surfaces from initial guess are not nested, attempting to refine "
            + "coordinates. This may take a few moments."
        )
        obj = ObjectiveFunction(GoodCoordinates(eq))
        constraints = get_fixed_boundary_constraints(eq) + (FixThetaSFL(eq),)
        eq.solve(
            objective=obj,
            constraints=constraints,
            ftol=0,
            xtol=0,
            gtol=1e-8,
            verbose=0,
            optimizer="fmintr-bfgs",
        )
        warnif(
            not eq.is_nested(),
            UserWarning,
            "Surfaces still not nested after refinement. This is possibly because "
            + "the boundary contains self-intersections or other singularities, or "
            + "because the refinement requires more iterations. You may need to "
            + "manually adjust the initial guess or do further refinement using the "
            + "GoodCoordinates objective.",
        )

    return eq


def _initial_guess_surface(x_basis, b_lmn, b_basis, axis=None, coord=None):
    """Create an initial guess from boundary coefficients and a magnetic axis guess.

    Parameters
    ----------
    x_basis : FourierZernikeBasis
        basis of the flux surfaces (for R, Z, or Lambda).
    b_lmn : ndarray, shape(b_basis.num_modes,)
        vector of boundary coefficients associated with b_basis.
    b_basis : Basis
        basis of the boundary surface (for Rb or Zb)
    axis : ndarray, shape(num_modes,2)
        coefficients of the magnetic axis. axis[i, :] = [n, x0].
    coord : float or None
        Surface label (ie, rho, zeta, etc.) for supplied surface.

    Returns
    -------
    x_lmn : ndarray
        vector of flux surface coefficients associated with x_basis.

    """
    b_modes = jnp.asarray(b_basis.modes)
    x_modes = jnp.asarray(x_basis.modes)
    b_lmn = jnp.asarray(b_lmn)
    x_lmn = jnp.zeros((x_basis.num_modes,))

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
            axis = jnp.array([b_basis.modes[axidx, 2], b_lmn[axidx]]).T

        # first do all the m != 0 modes, easiest since no special logic needed
        def body(k, x_lmn):
            l, m, n = b_modes[k]
            scale = scales[k]
            # index of basis mode with lowest radial power (l = |m|)
            mask0 = (x_modes == jnp.array([abs(m), m, n])).all(axis=1)
            x_lmn = jnp.where(mask0, b_lmn[k] / scale, x_lmn)
            return x_lmn

        # Get scale values for all modes
        m_values = b_modes[:, 1]
        scales = zernike_radial(coord, abs(m_values), m_values)
        x_lmn = fori_loop(0, b_basis.num_modes, body, x_lmn)

        # now overwrite stuff to deal with the axis
        scale = zernike_radial(coord, 0, 0)
        for k, (l, m, n) in enumerate(b_basis.modes):
            if m != 0:
                continue
            # index of basis mode with lowest radial power (l = |m|)
            idx0 = np.where((x_basis.modes == [abs(m), m, n]).all(axis=1))[0]
            # index of basis mode with second lowest radial power (l = |m| + 2)
            idx2 = np.where((x_basis.modes == [abs(m) + 2, m, n]).all(axis=1))[0]
            ax = np.where(axis[:, 0] == n)[0]
            if ax.size:
                a_n = axis[ax[0], 1]  # use provided axis guess
            else:
                a_n = b_lmn[k]  # use boundary centroid as axis
            x_lmn = jit(put)(x_lmn, idx0, (b_lmn[k] + a_n) / 2 / scale)
            x_lmn = jit(put)(x_lmn, idx2, (b_lmn[k] - a_n) / 2 / scale)

    elif mode == "poincare":

        def body(k, x_lmn):
            l, m, n = b_modes[k]
            mask0 = (x_modes == jnp.array([l, m, n])).all(axis=1)
            x_lmn = jnp.where(mask0, b_lmn[k], x_lmn)
            return x_lmn

        x_lmn = fori_loop(0, b_basis.num_modes, body, x_lmn)

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
    if not isinstance(nodes, _Grid):
        nodes = Grid(nodes, sort=False)
    transform = Transform(nodes, x_basis, build=False, build_pinv=True)
    x_lmn = transform.fit(x)
    return x_lmn
