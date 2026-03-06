"""Functions for computing initial guesses for coordinate surfaces."""

import os
import warnings

import numpy as np
from map2disc import BCM

from desc.backend import fori_loop, jit, jnp, put
from desc.basis import FourierZernikeBasis, zernike_radial
from desc.geometry import FourierRZCurve, Surface
from desc.grid import Grid, LinearGrid, _Grid
from desc.io import load
from desc.objectives import (
    FixThetaSFL,
    GoodCoordinates,
    ObjectiveFunction,
    get_fixed_axis_constraints,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import factorize_linear_constraints
from desc.transform import Transform
from desc.utils import copy_coeffs, warnif


def set_initial_guess(eq, *args, ensure_nested=True):  # noqa: C901
    """Set the initial guess for the flux surfaces, eg R_lmn, Z_lmn, L_lmn.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to initialize
    args :
        either:
          - No arguments, in which case eq.surface will be scaled down to eq.axis
          as the initial guess, and eq.L_lmn will be set to zero.
          - Another Surface object, which will be scaled down to an axis to generate
            the guess. Optionally a Curve object may also be supplied for the magnetic
            axis, if not supplied then the Surface object's `get_axis` method will be
            used to find the axis from the surface, and eq.L_lmn will be set to zero.
          - Another Equilibrium, whose flux surfaces and lambda will be used.
          - File path to a VMEC or DESC equilibrium, which will be loaded and used.
          - Grid and 2-3 ndarrays, specifying the flux surface locations (R, Z, and
            optionally lambda) at fixed flux coordinates. All arrays should have the
            same length. Optionally, an ndarray of shape(k,3) may be passed instead
            of a grid. If lambda is not passed, it will be set to zero.
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
                coord=coord,
            )
            eq.Z_lmn = _initial_guess_surface(
                eq.Z_basis,
                eq.Zb_lmn,
                eq.surface.Z_basis,
                axisZ,
                coord=coord,
            )
            eq.L_lmn = np.zeros_like(eq.L_lmn)
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
            eq.L_lmn = np.zeros_like(eq.L_lmn)
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
        try:
            Rlmn, Zlmn = _babin_init_Zernike_only(eq, eq.L)

            eq.R_lmn = Rlmn
            eq.Z_lmn = Zlmn
            eq.axis = eq.get_axis()
            # enforce the R_lmn Z_lmn to match the original surface, as
            # right now there may be a mismatch, through applying
            # the linear constraints
            constraints = get_fixed_axis_constraints(
                profiles=False, eq=eq
            ) + get_fixed_boundary_constraints(eq=eq)
            constraints = maybe_add_self_consistency(eq, constraints)
            objective = ObjectiveFunction(constraints)
            objective.build(verbose=0)
            _, _, _, _, _, _, project, recover, *_ = factorize_linear_constraints(
                objective, objective
            )
            args = objective.unpack_state(recover(project(objective.x(eq))), False)[0]
            eq.params_dict = args
        except (
            Exception
        ) as e:  # if Babin init fails, just try to refine the existing guess
            print(
                "Unable to use map2disc method to obtain nested initial guess, got "
                "error: ",
                e,
            )
            print(
                "Falling back to performing optimization using GoodCoordinates"
                " objective following work of Tecchiolli et al."
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


def _initial_guess_surface(x_basis, b_lmn, b_basis, axis=None, mode=None, coord=None):
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
    b_modes = jnp.asarray(b_basis.modes)
    x_modes = jnp.asarray(x_basis.modes)
    b_lmn = jnp.asarray(b_lmn)
    x_lmn = jnp.zeros((x_basis.num_modes,))

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


def _boundary_cut(surface, zeta):
    """Given surface and zeta, return function curve(theta) which yields R(t),Z(t)."""

    def curve(theta):
        theta = np.asarray(theta)
        nodes = np.vstack([np.ones_like(theta), theta, zeta * np.ones_like(theta)]).T
        grid = Grid(nodes=nodes, NFP=surface.NFP, jitable=True)
        # Must use Grid, as cannot let LinearGrid re-sort the nodes
        # in case map2disc needs to use a different curve orientation
        # (which it does, as it needs left-handed coordinate system
        # with theta increasing CCW)
        data = surface.compute(["R", "Z"], grid=grid)
        return np.array(data["R"]), np.array(data["Z"])

    return curve


def _babin_init_Zernike_only(eq, nrho):
    """Use Babin's map2disc package to get initially nested mapping.

    Assumes equilibrium is right-handed (positive jacobian),
    and returns R_lmn Z_lmn in the same right-handed convention.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to initialize
    nrho : int
        Number of radial grid points to evaluate mapping at. These
        will then be fit to form the output Fourier-Zernike coefficients

    Returns
    -------
    R_lmn : ndarray
        Nested R_lmn coefficients
    Z_lmn : ndarray
        Nested Z_lmn coefficients

    """
    surface = eq.surface
    L = eq.L
    M = eq.M
    N = eq.N
    rho1d_out = np.linspace(0, 1, nrho * 2)
    N_out = N
    zeta1d_out = np.linspace(0, 2 * np.pi / eq.NFP, 2 * N_out + 1, endpoint=False)

    # solve BCM
    MZernike = M
    zeta_cut = zeta1d_out
    theta1d = np.linspace(0, 2 * np.pi, 2 * M + 1, endpoint=False)
    theta1d += 0.5 * (theta1d[1] - theta1d[0])
    zeta1d = np.linspace(0, 2 * np.pi / eq.NFP, 2 * N + 1, endpoint=False)
    zeta1d += 0.5 * (zeta1d[1] - zeta1d[0])

    all_bcm = {}
    for z, zeta in enumerate(zeta_cut):
        curve = _boundary_cut(surface, zeta)
        bcm = BCM(curve, MZernike)
        bcm.solve("interpolate")
        all_bcm[z] = bcm

    grid = LinearGrid(zeta=zeta_cut, NFP=eq.NFP)
    all_cx = np.zeros((all_bcm[0].cx.shape[0], zeta_cut.size))
    all_cy = np.zeros((all_bcm[0].cy.shape[0], zeta_cut.size))
    for z, zeta in enumerate(zeta_cut):
        all_cx[:, z] = all_bcm[z].cx
        all_cy[:, z] = all_bcm[z].cy

    # simplest way to obtain the 3D Fourier-Zernike coeffs: just evaluate in real space
    # and re-fit, that way no need to worry about mode orderings
    grid = LinearGrid(rho=rho1d_out, M=M, zeta=zeta_cut, NFP=eq.NFP)
    Rbasis = FourierZernikeBasis(
        L=L,
        M=surface.M,
        N=surface.N,
        NFP=surface.NFP,
        sym={True: "cos", False: False}[surface.sym],
        spectral_indexing=eq.spectral_indexing,
    )
    Rtransform_3d = Transform(grid=grid, basis=Rbasis, build_pinv=True)
    Zbasis = FourierZernikeBasis(
        L=L,
        M=surface.M,
        N=surface.N,
        NFP=surface.NFP,
        sym={True: "sin", False: False}[surface.sym],
        spectral_indexing=eq.spectral_indexing,
    )
    Ztransform_3d = Transform(grid=grid, basis=Zbasis, build_pinv=True)

    Rout = np.zeros(grid.nodes.shape[0])
    Zout = np.zeros(grid.nodes.shape[0])
    Rout = grid.meshgrid_reshape(Rout, "rtz")
    Zout = grid.meshgrid_reshape(Zout, "rtz")
    rhos_grid = grid.meshgrid_reshape(grid.nodes[:, 0], "rtz")
    thetas_grid = grid.meshgrid_reshape(grid.nodes[:, 1], "rtz")

    curve = _boundary_cut(surface, 0.0)  # not used in evaluation of cx,cy
    bcm = BCM(curve, MZernike)
    for z, zeta in enumerate(zeta1d_out):
        bcm.cx = all_cx[:, z]
        bcm.cy = all_cy[:, z]
        Rout[:, :, z], Zout[:, :, z] = bcm.eval_rt(
            rhos_grid[:, :, z].squeeze(), thetas_grid[:, :, z].squeeze()
        )
    R_lmn = Rtransform_3d.fit(grid.meshgrid_flatten(Rout, "rtz"))
    Z_lmn = Ztransform_3d.fit(grid.meshgrid_flatten(Zout, "rtz"))

    # TODO: smarter way would be to take the existing Zernike coeffs as fxn of phi and
    # fit those coefficients to immediately get the 3D FourierZernike coefficients.
    # Need to know what the mode orderings are though for cx and cy to use this
    # approach, which I've not dug into yet as the other method works fine.

    # map2disc outputs in left-handed coordinates, so we need to flip
    # the sign of theta to get back to right-handed coords
    rone = np.ones_like(R_lmn)
    rone[eq.R_basis.modes[:, 1] < 0] *= -1
    R_lmn *= rone

    zone = np.ones_like(Z_lmn)
    zone[eq.Z_basis.modes[:, 1] < 0] *= -1
    Z_lmn *= zone

    return (
        R_lmn,
        Z_lmn,
    )
