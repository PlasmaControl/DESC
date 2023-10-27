"""Utility functions for dealing with older versions of DESC."""

import warnings

import numpy as np

from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.utils import errorif


def ensure_positive_jacobian(eq):
    """Convert an Equilibrium to have a positive coordinate Jacobian.

    Parameters
    ----------
    eq : Equilibrium or iterable of Equilibrium
        Equilibria to convert to right-handed coordinate system.

    Returns
    -------
    eq : Equilibrium or iterable of Equilibrium
        Same as input, but with coefficients adjusted to give right-handed coordinates.

    """
    # maybe it's iterable:
    if hasattr(eq, "__len__"):
        for e in eq:
            ensure_positive_jacobian(e)
        return eq

    sign = np.sign(eq.compute("sqrt(g)", grid=Grid(np.array([[1, 0, 0]])))["sqrt(g)"])
    errorif(
        sign == 0,
        ValueError,
        "sqrt(g) == 0, are you sure this Equilibrium is not degenerate?",
    )
    if sign < 0:
        warnings.warn(
            "Left handed coordinates detected, switching sign of theta."
            + " To avoid this warning in the future, switch the sign of all"
            + " modes with m<0 and iota/current profile."
        )

        if eq.iota is not None:
            eq.i_l *= -1
        else:
            eq.c_l *= -1

        rone = np.ones_like(eq.R_lmn)
        rone[eq.R_basis.modes[:, 1] < 0] *= -1
        eq.R_lmn *= rone

        zone = np.ones_like(eq.Z_lmn)
        zone[eq.Z_basis.modes[:, 1] < 0] *= -1
        eq.Z_lmn *= zone

        lone = np.ones_like(eq.L_lmn)
        lone[eq.L_basis.modes[:, 1] >= 0] *= -1
        eq.L_lmn *= lone

        eq.axis = eq.get_axis()
        eq.surface = eq.get_surface_at(rho=1)

    sign = np.sign(eq.compute("sqrt(g)", grid=Grid(np.array([[1, 0, 0]])))["sqrt(g)"])
    assert sign == 1
    return eq


def flip_helicity(eq):
    """Change the sign of the helicity of an Equilibrium.

    Parameters
    ----------
    eq : Equilibrium or iterable of Equilibrium
        Equilibria to flip the helicity of.

    Returns
    -------
    eq : Equilibrium or iterable of Equilibrium
        Same as input, but with the opposite sign helicity.

    """
    # maybe it's iterable:
    if hasattr(eq, "__len__"):
        for e in eq:
            flip_helicity(e)
        return eq

    if eq.iota is not None:
        eq.i_l *= -1
    else:
        eq.c_l *= -1

    rone = np.ones_like(eq.R_lmn)
    rone[eq.R_basis.modes[:, 2] < 0] *= -1
    eq.R_lmn *= rone

    zone = np.ones_like(eq.Z_lmn)
    zone[eq.Z_basis.modes[:, 2] < 0] *= -1
    eq.Z_lmn *= zone

    lone = np.ones_like(eq.L_lmn)
    lone[eq.L_basis.modes[:, 2] < 0] *= -1
    eq.L_lmn *= lone

    eq.axis = eq.get_axis()
    eq.surface = eq.get_surface_at(rho=1)

    return eq


def rescale(eq, R0=None, B0=None, verbose=0):
    """Rescale an Equilibrium to major radius R and magnetic field strength on axis B.

    Assumes the aspect ratio is held constant?

    Parameters
    ----------
    eq : Equilibrium or iterable of Equilibrium
        Equilibria to rescale.
    R0 : float
        Desired major radius. If None, no change (default).
    B0 : float
        Desired magnetic field strength on axis. If None, no change (default).
    verbose : int
        Level of output.

    Returns
    -------
    eq : Equilibrium or iterable of Equilibrium
        Same as input, but rescaled to the desired major radius and field strength.

    """
    # maybe it's iterable:
    if hasattr(eq, "__len__"):
        for e in eq:
            rescale(e, R0, B0)
        return eq

    # compute actual major radius
    grid_R = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    R = eq.compute("R0", grid=grid_R)["R0"]
    R0 = R0 or R

    # compute actual |B| on axis
    grid_B = LinearGrid(N=eq.N_grid, NFP=eq.NFP, rho=0)
    B = np.mean(eq.compute("|B|", grid=grid_B)["|B|"])
    B0 = B0 or B

    # scaling factor = desired / actual
    cR = R0 / R
    cB = B0 / B
    if verbose:
        print("Major radius scaling factor:   {:.2f}".format(cR))
        print("Magnetic field scaling factor: {:.2f}".format(cB))

    # scale flux surfaces
    eq.R_lmn *= cR
    eq.Z_lmn *= cR
    eq.Psi *= cR**2 * cB

    # scale pressure profile
    if eq.pressure is not None:
        eq.p_l *= cB**2
    else:
        eq.ne_l *= cB
        eq.Te_l *= cB
        eq.Ti_l *= cB

    # scale current profile
    if eq.current is not None:
        eq.c_l *= cR * cB

    # check new major radius
    grid_R = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    R = eq.compute("R0", grid=grid_R)["R0"]
    # TODO: assert R == R0

    # check new |B| on axis
    grid_B = LinearGrid(N=eq.N_grid, NFP=eq.NFP, rho=0)
    B = np.mean(eq.compute("|B|", grid=grid_B)["|B|"])
    # TODO: assert B == B0

    return eq
