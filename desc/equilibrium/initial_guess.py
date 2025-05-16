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
from desc.vmec_utils import ptolemy_identity_rev
from map2disc import BCM


def set_initial_guess(eq, *args, ensure_nested=True):  # noqa: C901
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


def convert_bdry(eq):
    # convert boundary in desc representation lmn to vmecs rbmnc/zbmns
    eps = 1e-15  # to filter very low values after ptolemy's identity
    M, N, _, RBC = ptolemy_identity_rev(
        eq.surface.R_basis.modes[:, 1],
        eq.surface.R_basis.modes[:, 2],
        eq.surface.R_lmn,
    )
    _, _, ZBS, _ = ptolemy_identity_rev(
        eq.surface.Z_basis.modes[:, 1],
        eq.surface.Z_basis.modes[:, 2],
        eq.surface.Z_lmn,
    )
    ZBS = ZBS[0]
    RBC = RBC[0]
    ZBS[np.abs(ZBS) < eps] = 0.0
    RBC[np.abs(RBC) < eps] = 0.0
    max_m = max(np.max(M[RBC != 0]), np.max(M[ZBS != 0]))
    max_n = max(np.max(np.abs(N[ZBS != 0])), np.max(np.abs(N[RBC != 0])))

    M_bdry = np.arange(max_m + 1)
    N_bdry = np.arange(max_n * 2 + 1) - max_n

    RBC_out = RBC[np.logical_and(M <= max_m, np.abs(N) <= max_n)]
    ZBS_out = ZBS[np.logical_and(M <= max_m, np.abs(N) <= max_n)]
    RBC_out = np.concatenate([np.zeros(max_n), RBC_out]).reshape(
        max_m + 1, 2 * max_n + 1
    )
    ZBS_out = np.concatenate([np.zeros(max_n), ZBS_out]).reshape(
        max_m + 1, 2 * max_n + 1
    )

    return M_bdry, N_bdry, RBC_out, ZBS_out


def mn_to_tz(theta1d, zeta1d, m1d, n1d, xc2d=None, xs2d=None):
    """
    evaluate a 2d fourier series x(theta,zeta) = sum_mn (xc_mn cos(m*theta-n*zeta)+ xs_mn sin(m*theta-n*zeta) ) on a meshgrid of theta1d X zeta1d
    making use of the tensor product to speed up the evaluation
    Input:
    theta1d : 1d array of angle in 2pi in poloidal direction
    zeta1d  : 1d array of angle in 2pi in toroidal direction
    m1d     : 1d array of integer poloidal mode numbers, size must be equal to first dimension of xc2d, xs2d
    n1d     : 1d array of integer toroidal mode numbers, size must be equal to second dimension of xc2d, xs2d
    xc2d    : 2d array of coefficients m,n for cos(m*theta-n*zeta),  dimension is of len(m1d) X  len(n1d)
    xs2d    : 2d array of coefficients m,n for sin(m*theta-n*zeta),  dimension is of len(m1d) X  len(n1d)
    Output:
    x2d     : 2d array of values of the 2d fourier series, dimension is of len(theta1d) X len(zeta1d)
    """
    assert (xc2d is not None) or (xs2d is not None)
    if xc2d is not None:
        assert xc2d.shape == (len(m1d), len(n1d))
    if xs2d is not None:
        assert xs2d.shape == (len(m1d), len(n1d))

    sin_mt = np.sin(np.outer(theta1d, m1d))
    cos_mt = np.cos(np.outer(theta1d, m1d))
    sin_nz = np.sin(np.outer(n1d, zeta1d))
    cos_nz = np.cos(np.outer(n1d, zeta1d))

    Lt = len(theta1d)
    Lz = len(zeta1d)
    Lm = len(m1d)
    Ln = len(n1d)
    #  xout= (  cos_mt @ xc2d @ cos_nz + sin_mt @ xc2d @ sin_nz
    #         + sin_mt @ xs2d @ cos_nz - cos_mt @ xs2d @ sin_nz )
    # performance, number of multiplications:
    # - first bracket m*theta then apply n*zeta: [1,2]*Lt*Lm*Ln + Lt*Ln*Lz = Lt*Ln*([1,2]*Lm+Lz)
    # - first bracket n*zeta then apply m*theta: [1,2]*Lm*Ln*Lz + Lt*Lm*Lz = Lz*Lm*([1,2]*Ln+Lt)
    onetwo = (xc2d is not None) * 1 + (xs2d is not None) * 1
    mfirst = Lt * Ln * (onetwo * Lm + Lz) < Lz * Lm * (onetwo * Ln + Lt)
    if xc2d is not None:
        if xs2d is None:  # only xc2d
            if mfirst:
                xout = (cos_mt @ xc2d) @ cos_nz + (sin_mt @ xc2d) @ sin_nz
            else:
                xout = cos_mt @ (xc2d @ cos_nz) + sin_mt @ (xc2d @ sin_nz)
        else:  # xc2d and xs2d
            if mfirst:
                xout = (cos_mt @ xc2d + sin_mt @ xs2d) @ cos_nz + (
                        sin_mt @ xc2d - cos_mt @ xs2d
                ) @ sin_nz
            else:
                xout = cos_mt @ (xc2d @ cos_nz - xs2d @ sin_nz) + sin_mt @ (
                        xc2d @ sin_nz + xs2d @ cos_nz
                )
    else:  # only xs2d
        if mfirst:
            xout = (sin_mt @ xs2d) @ cos_nz - (cos_mt @ xs2d) @ sin_nz
        else:
            xout = sin_mt @ (xs2d @ cos_nz) - cos_mt @ (xs2d @ sin_nz)

    return xout


def tz_to_mn(theta1d, zeta1d, m1d, n1d, x2d=None, sincos=None, testmass=True):
    """
    project a 2d data given on a meshgrid of theta1d X zeta1d to a 2d real fourier series:
        xc_mn = 1/norm_mn * sum_tz (x(theta,zeta)* cos(m*theta-n*zeta)
    or
        xs_mn = 1/norm_mn *sum_tz (x(theta,zeta) sin(m*theta-n*zeta)
    making use of the tensor product to speed up the evaluation
    Input:
    theta1d : 1d array of angle in 2pi in poloidal direction: careful, this is now an integration, so it has to be equidistant without the periodic endpoint, [0,2pi[
    zeta1d  : 1d array of angle in 2pi in toroidal direction  careful, this is now an integration, so it has to be equidistant [0,2pi[
    m1d     : 1d array of integer poloidal mode numbers, size equals first dimension of xc2d, xs2d
    n1d     : 1d array of integer toroidal mode numbers, size equals second dimension of xc2d, xs2d
    x2d    : 2d array of values on the meshgrid (theta1d) X (zeta1d)
    sincos : either sine,cosine or both to be projected!
    testmass : check if mass matrix (int(cos_m1,cos_m2)  , int(cos_n1,cos_n2)) is diagonal with the chosen poloidal / toroidal points
    Output:
    xc2d    : 2d array of coefficients m,n for cos(m*theta-n*zeta),  dimension is of len(m1d) X  len(n1d) (single output if sincos="cos" or first output when sincos="sincos")
    xs2d    : 2d array of coefficients m,n for sin(m*theta-n*zeta),  dimension is of len(m1d) X  len(n1d) (single output if sincos="sin" or second output if sincos="sincos")
    """
    assert sincos == "sin" or sincos == "cos" or sincos == "sincos"
    assert x2d is not None
    sin_mt = np.sin(np.outer(m1d, theta1d))
    cos_mt = np.cos(np.outer(m1d, theta1d))
    sin_nz = np.sin(np.outer(zeta1d, n1d))
    cos_nz = np.cos(np.outer(zeta1d, n1d))

    Lt = len(theta1d)
    Lz = len(zeta1d)
    Lm = len(m1d)
    Ln = len(n1d)

    bases = {}
    bases["sin_mt"] = {"mat": sin_mt, "mode1d": m1d}
    bases["cos_mt"] = {"mat": cos_mt, "mode1d": m1d}
    bases["sin_nz"] = {"mat": sin_nz.T, "mode1d": n1d}
    bases["cos_nz"] = {"mat": cos_nz.T, "mode1d": n1d}

    # check mass matrix to be diagonal, extract the diagonal
    for key, base in bases.items():
        mass = base["mat"] @ base["mat"].T
        # check mass matrix,  equal absolute mode numbers => non-zero entry in mass matrix (includes diagonal)
        filter = (
                         np.abs(base["mode1d"][:, None]) - np.abs(base["mode1d"][None, :]) == 0
                 ) * 1
        maxerr = np.amax(np.abs(mass - mass * filter))
        assert (
                maxerr < 1e-10
        ), f" {key} mass matrix not diagonal, maxerr = {maxerr} > 1e-10"

    # divide by norm of cosine / sine of each mode m,n (use cosine since)
    # 1/norm, from sum_Lt sum_lz
    sdiag = np.ones((Lm, Ln)) / (0.5 * Lt * Lz)
    # set m=0,n=0 mode to half
    sdiag[np.where(m1d == 0), np.where(n1d == 0)] *= 0.5
    # set zero m=0,n<0 modes!
    filter_m0_nneg = ((m1d[:, None] == 0) * (n1d[None, :] < 0)) * (-1) + 1
    sdiag *= filter_m0_nneg

    #  xc2d=   cos_mt @ x2D @ cos_nz + sin_mt @ x2d @ sin_nz
    #  xs2d=   sin_mt @ x2d @ cos_nz - cos_mt @ x2d @ sin_nz )

    xtz_cos_nz = x2d @ cos_nz
    xtz_sin_nz = x2d @ sin_nz
    if sincos == "cos":
        xc2d = (cos_mt @ xtz_cos_nz + sin_mt @ xtz_sin_nz) * sdiag
        return xc2d
    if sincos == "sin":
        xs2d = (sin_mt @ xtz_cos_nz - cos_mt @ xtz_sin_nz) * sdiag
        return xs2d
    if sincos == "sincos":
        xc2d = (cos_mt @ xtz_cos_nz + sin_mt @ xtz_sin_nz) * sdiag
        xs2d = (sin_mt @ xtz_cos_nz - cos_mt @ xtz_sin_nz) * sdiag
        return xc2d, xs2d


def boundary_cut(rbc, zbs, zeta):
    M, N = rbc.shape[0] - 1, (rbc.shape[1] - 1) // 2
    assert rbc.shape == zbs.shape == (M + 1, 2 * N + 1)  # m = 0..M, n = -N..N

    # assert np.isnan(rbc).sum() == 0
    # assert np.isnan(zbs).sum() == 0
    # mg,ng=np.meshgrid(np.arange(M + 1), np.arange(-N, N + 1), indexing="ij")

    def curve(theta):
        theta = np.asarray(theta)
        test = True

        Rslow, Zslow = np.zeros((2, theta.size))
        for m in range(M + 1):
            for n in range(-N, N + 1):
                if m == 0 and n < 0:
                    continue
                Rslow += rbc[m, n + N] * np.cos(m * theta - n * zeta)
                Zslow += zbs[m, n + N] * np.sin(m * theta - n * zeta)

        # R, Z = np.zeros((2, theta.size))

        # R  = np.cos(np.outer(theta,mg.flatten()) - np.outer(zeta,ng.flatten())) @ rbc.flatten()
        # Z  = np.sin(np.outer(theta,mg.flatten()) - np.outer(zeta,ng.flatten())) @ zbs.flatten()
        # if(test):
        #    assert(np.allclose(R, Rslow))
        #    assert(np.allclose(Z, Zslow))

        return Rslow, Zslow

    return curve


def real_dft_mat(x_in, x_out, nfp=1, modes=None, deriv=0):
    """
    Flexible Direct Fourier Transform for real data
    takes an input array of equidistant points in [0,2pi/nfp[ (exclude endpoint!),
    evaluate the discrete fourier transform with the given 1d mode vector (all >=0) using the input points x_in, then evaluate the inverse transform (or its derivative deriv>0) on the output points x_out anywhere...
    len(x_in) must be > 2*max(modes)
    output is the matrix that transforms real function to real function [derivative]:
     f^deriv(x_out) = Mat f(x_in) (can then be used to do 2d transforms with matmul!)

    nfp is the number of field periods, default 1 (int), all modes are multiples of nfp

    """
    if modes is None:
        modes = np.arange((len(x_in) - 1) // 2 + 1)  # all modes up to Nyquist
    assert (
            np.abs(x_in[-1] + (x_in[1] - x_in[0]) - x_in[0] - 2 * np.pi / nfp) < 1.0e-8
    ), "x_in must be equidistant in [0,2pi/nfp["
    assert np.all(modes >= 0), "modes must be positive"
    zeromode = np.where(modes == 0)
    assert len(zeromode) <= 1, "only one zero mode allowed"
    maxmode = np.amax(modes)
    assert (
            len(x_in) > 2 * maxmode
    ), f"number of sampling points ({len(x_in)}) > 2*maxmodenumber/nfp ({maxmode})"
    # matrix for forward transform
    modes_forward = np.exp(1j * nfp * (modes[:, None] * x_in[None, :]))
    mass_re = modes_forward.real @ modes_forward.real.T
    mass_im = modes_forward.imag @ modes_forward.imag.T
    diag_re = np.copy(np.diag(mass_re))
    diag_im = np.copy(np.diag(mass_im))

    assert np.all(
        np.abs(mass_re - np.diag(diag_re)) < 1.0e-8
    ), "massre must be diagonal"
    assert np.all(
        np.abs(mass_im - np.diag(diag_im)) < 1.0e-8
    ), "massim must be diagonal"
    diag_im[zeromode] = 1  # imag (=sin) is zero at zero mode
    assert np.all(diag_re > 0.0)
    assert np.all(diag_im > 0.0)

    # inverse mass matrix applied (for real and imag)
    modes_forward_mod = (
            np.diag(1 / diag_re) @ modes_forward.real
            + np.diag(1j / diag_im) @ modes_forward.imag
    )
    if deriv == 0:
        modes_back = np.exp(-1j * nfp * (modes[None, :] * x_out[:, None]))
    else:
        modes_back = (-1j * nfp * modes[None, :]) ** deriv * np.exp(
            -1j * nfp * (modes[None, :] * x_out[:, None])
        )
    Mat = (modes_back @ modes_forward_mod).real
    return Mat


def babin_init(rbc, zbs, M, N, nrho, m_out, n_out):
    # what about nfp?
    rbc = np.array(rbc)
    zbs = np.array(zbs)
    # example for output of the  full solution (VMEC):
    rho1d_out = np.linspace(0, 1, nrho) ** 2  # in rho^2
    M_out = m_out
    m1d_out = np.arange(0, M_out + 1)
    theta1d_out = np.linspace(0, 2 * np.pi, 2 * M_out + 1, endpoint=False)
    N_out = n_out
    n1d_out = np.arange(-N_out, N_out + 1)
    zeta1d_out = np.linspace(0, 2 * np.pi, 2 * N_out + 1, endpoint=False)

    # filter out poloidal and toroidal modes:
    # M_filter = M set on 11.24 because mass test of tz_to_mn fails otherwise
    M_filter = M
    m1d_filter = np.arange(0, M_filter + 1)
    # theta1d_filter = np.linspace(0, 2 * np.pi, 2 * M_filter + 1, endpoint=False)
    N_filter = N
    n1d_filter = np.arange(-N_filter, N_filter + 1)
    m1d = np.arange(0, M + 1)
    n1d = np.arange(-N, N + 1)

    # solve BCM
    cuts = 2 * N_filter + 1
    MZernike = M_filter
    zeta_cut = np.linspace(
        0, 2 * np.pi, cuts, endpoint=False
    )  # only half-period due to stellarator symmetry
    theta1d = np.linspace(0, 2 * np.pi, 2 * M + 1, endpoint=False)
    theta1d += 0.5 * (theta1d[1] - theta1d[0])
    zeta1d = np.linspace(0, 2 * np.pi, 2 * N + 1, endpoint=False)
    zeta1d += 0.5 * (zeta1d[1] - zeta1d[0])
    # to real
    Rout = mn_to_tz(theta1d, zeta1d, m1d, n1d, xc2d=rbc)
    Zout = mn_to_tz(theta1d, zeta1d, m1d, n1d, xs2d=zbs)
    rbc_filter = tz_to_mn(
        theta1d, zeta1d, m1d_filter, n1d_filter, x2d=Rout, sincos="cos"
    )
    zbs_filter = tz_to_mn(
        theta1d, zeta1d, m1d_filter, n1d_filter, x2d=Zout, sincos="sin"
    )

    all_bcm = {}
    for z, zeta in enumerate(zeta_cut):
        curve = boundary_cut(rbc_filter, zbs_filter, zeta)
        bcm = BCM(curve, MZernike)
        bcm.solve("interpolate")
        all_bcm[z] = bcm

    all_cx = np.zeros((all_bcm[0].cx.shape[0], cuts))
    all_cy = np.zeros((all_bcm[0].cy.shape[0], cuts))
    for z, zeta in enumerate(zeta_cut):
        all_cx[:, z] = all_bcm[z].cx
        all_cy[:, z] = all_bcm[z].cy

    f_up = real_dft_mat(zeta_cut, zeta1d_out)

    # using all_cx from above
    new_cx = all_cx @ f_up.T
    new_cy = all_cy @ f_up.T

    # use bcm for evaluation of zernike polynomial (coefficients cx, cy)
    Rout = np.zeros((nrho, 2 * M_out + 1, 2 * N_out + 1))
    Zout = np.zeros((nrho, 2 * M_out + 1, 2 * N_out + 1))
    curve = boundary_cut(rbc_filter, zbs_filter, 0.0)  # not used in evaluation of cx,cy
    bcm = BCM(curve, MZernike)
    scaledjac = 0 * zeta1d_out - 1
    for z, zeta in enumerate(zeta1d_out):
        bcm.cx = new_cx[:, z]
        bcm.cy = new_cy[:, z]
        Rout[:, :, z], Zout[:, :, z] = bcm.eval_rt_1d(rho1d_out, theta1d_out)
        # check jacobian
        jac = bcm.eval_Jac_rt_1d(rho1d_out, theta1d_out)
        scaledjac[z] = np.amin(jac) / np.amax(jac)

    assert np.all(
        scaledjac > 1e-4
    ), f"Problem with invertibility, scaled 2d jacobian in cross-sections is <1.0e-4... min(scaledjac(zeta))={np.amin(scaledjac)}, max(scaledjac(zeta))={np.amax(scaledjac)}"

    rbc_out = np.zeros((nrho, M_out + 1, 2 * N_out + 1))
    zbs_out = np.zeros((nrho, M_out + 1, 2 * N_out + 1))
    for i in range(nrho):
        rbc_out[i, :, :] = tz_to_mn(
            theta1d_out, zeta1d_out, m1d_out, n1d_out, x2d=Rout[i, :, :], sincos="cos"
        )
        zbs_out[i, :, :] = tz_to_mn(
            theta1d_out, zeta1d_out, m1d_out, n1d_out, x2d=Zout[i, :, :], sincos="sin"
        )

    xm_out = m1d_out[:, None] * 1 + n1d_out[None, :] * 0
    xn_out = m1d_out[:, None] * 0 + n1d_out[None, :] * 1

    # vmec output, filter out m=0 n<0 and flatten modes
    filtm, filtn = np.where(~((m1d_out[:, None] == 0) * (n1d_out[None, :] < 0)))

    return (
        xm_out[filtm, filtn],
        xn_out[filtm, filtn],
        rbc_out[:, filtm, filtn],
        zbs_out[:, filtm, filtn],
        rho1d_out,
    )


def babin_initial_guess(eq):
    print("Using babin init")
    n_babin_init = 2
    M_bdry, N_bdry, rbc_bdry, zbs_bdry = convert_bdry(eq)
    xm_init, xn_init, rmnc, zmns, rhos = babin_init(
        rbc_bdry, zbs_bdry, np.max(M_bdry), np.max(N_bdry), 7, 0, n_babin_init
    )
    ramnc = rmnc[0]
    zamns = zmns[0]

    # babin_axis = FourierRZCurve(
    #     R_n=ramnc[0],
    #     Z_n=zamns[0],
    #     modes_R=np.arange(0, n_babin_init + 1),
    #     modes_Z=np.arange(0, n_babin_init + 1),
    #     sym=True,
    #     NFP=model_eq.NFP,
    # )
    rax = np.concatenate([-np.zeros_like(ramnc)[1:][::-1], ramnc])
    zax = np.concatenate([-zamns[1:][::-1], np.zeros_like(zamns)])
    nax = len(ramnc) - 1
    nax = np.arange(-nax, nax + 1)
    # inputs["axis"] = np.vstack([nax, rax, zax]).T

    # how to set the new axis?
    eq.axis = np.vstack([nax, rax, zax]).T
    # eq = Equilibrium(
    #     Psi=eq.Psi,
    #     NFP=eq.NFP,
    #     M=eq.M,
    #     N=eq.N,
    #     L=eq.L,
    #     L_grid=eq.L_grid,
    #     M_grid=eq.M_grid,
    #     N_grid=eq.N_grid,
    #     pressure=eq.pressure,
    #     iota=eq.iota,
    #     current=eq.current,
    #     surface=eq.surface,
    #     sym=eq.sym,
    #     axis=np.vstack([nax, rax, zax]).T,
    # )

    return eq


if __name__ == "__main__":
    import desc

    p = "/Users/tthun/prog/git/mhdinn/data/raw/equilibria/iota_prescribed/toks/HirshManDSHAPE_beta3/output_DSHAPE.h5"
    eq = babin_initial_guess(desc.io.load(p))
