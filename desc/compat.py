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


def rescale(eq, L=("R0", None), B=("B0", None), verbose=0):
    """Rescale an Equilibrium in size L and magnetic field strength B.

    Parameters
    ----------
    eq : Equilibrium or iterable of Equilibrium
        Equilibria to rescale.
    L : tuple, (str, float)
        First element is a string denoting the length to scale. One of:
        {"R0", "a", "V"} for major radius, minor radius, or volume.
        Second element is a float denoting the desired size. Default is no scaling.
    B : tuple, (str, float)
        First element is a string denoting the magnetic field strength to scale. One of:
        {"B0", "<|B|>", "B_max"} for B on axis, volume averaged, or maximum on the LCFS.
        Second element is a float denoting the desired field. Default is no scaling.
    verbose : int
        Level of output.

    Returns
    -------
    eq : Equilibrium or iterable of Equilibrium
        Same as input, but rescaled to the desired size and magnetic field strength.

    """
    # maybe it's iterable:
    if hasattr(eq, "__len__"):
        for e in eq:
            rescale(e, L, B)
        return eq

    assert len(L) == 2
    assert len(B) == 2

    L_key = L[0]
    L_new = L[1]
    B_key = B[0]
    B_new = B[1]

    L_keys = ["R0", "a", "V"]
    B_keys = ["B0", "<|B|>_vol", "B_max"]

    if L_key not in L_keys:
        raise ValueError("Size scale L must be one of {{'R0', 'a', 'V'}}, got " + L_key)
    if B_key not in B_keys:
        raise ValueError(
            "Field strength scale B must be one of {{'B0', '<|B|>_vol', 'B_max'}}, got "
            + B_key
        )

    # size scaling
    grid_L = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data_L = eq.compute(L_keys, grid=grid_L)
    L_old = data_L[L_key]
    L_new = L_new or L_old
    cL = L_new / L_old
    cL = cL ** (1 / 3) if L_key == "V" else cL  # V = 2 π^2 R0 a^2

    # field scaling
    if B_key == "B0":
        grid_B = LinearGrid(N=eq.N_grid, NFP=eq.NFP, rho=0)
        data_B = eq.compute("|B|", grid=grid_B)
        B_old = np.mean(data_B["|B|"])
    elif B_key == "<|B|>_vol":
        grid_B = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        data_B = eq.compute("<|B|>_vol", grid=grid_B)
        B_old = data_B["<|B|>_vol"]
    elif B_key == "B_max":
        grid_B = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=1)
        data_B = eq.compute("|B|", grid=grid_B)
        B_old = np.max(data_B["|B|"])
    B_new = B_new or B_old
    cB = B_new / B_old

    # scaling factor = desired / actual
    if verbose:
        print("Size scaling factor:  {:.2f}".format(cL))
        print("Field scaling factor: {:.2f}".format(cB))

    # scale flux surfaces
    eq.R_lmn *= cL
    eq.Z_lmn *= cL
    eq.Psi *= cL**2 * cB

    # scale pressure profile
    if eq.pressure is not None:
        eq.p_l *= cB**2
    else:
        eq.ne_l *= cB
        eq.Te_l *= cB
        eq.Ti_l *= cB

    # scale current profile
    if eq.current is not None:
        eq.c_l *= cL * cB

    # boundary & axis
    eq.axis = eq.get_axis()
    eq.surface = eq.get_surface_at(rho=1)

    return eq
