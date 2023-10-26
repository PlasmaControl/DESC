"""Utility functions for dealing with older versions of DESC."""

import warnings

import numpy as np

from desc.grid import Grid
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
