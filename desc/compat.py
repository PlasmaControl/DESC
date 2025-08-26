"""Utility functions for dealing with older versions of DESC."""

import warnings

import numpy as np

from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.profiles import FourierZernikeProfile, PowerSeriesProfile, SplineProfile
from desc.utils import errorif, setdefault, warnif


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

        sign = np.sign(
            eq.compute("sqrt(g)", grid=Grid(np.array([[1, 0, 0]])))["sqrt(g)"]
        )
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


def flip_theta(eq):
    """Change the gauge freedom of the poloidal angle of an Equilibrium.

    Equivalent to redefining theta_new = theta_old + π

    Parameters
    ----------
    eq : Equilibrium or iterable of Equilibrium
        Equilibria to redefine the poloidal angle of.

    Returns
    -------
    eq : Equilibrium or iterable of Equilibrium
        Same as input, but with the poloidal angle redefined.

    """
    # maybe it's iterable:
    if hasattr(eq, "__len__"):
        for e in eq:
            flip_theta(e)
        return eq

    rone = np.ones_like(eq.R_lmn)
    rone[eq.R_basis.modes[:, 1] % 2 == 1] *= -1
    eq.R_lmn *= rone

    zone = np.ones_like(eq.Z_lmn)
    zone[eq.Z_basis.modes[:, 1] % 2 == 1] *= -1
    eq.Z_lmn *= zone

    lone = np.ones_like(eq.L_lmn)
    lone[eq.L_basis.modes[:, 1] % 2 == 1] *= -1
    eq.L_lmn *= lone

    eq.axis = eq.get_axis()
    eq.surface = eq.get_surface_at(rho=1)

    return eq


def rescale(
    eq, L=("R0", None), B=("B0", None), scale_pressure=True, copy=False, verbose=0
):
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
        {"B0", "<B>", "B_max"} for B on axis, volume averaged, or maximum on the LCFS.
        Second element is a float denoting the desired field. Default is no scaling.
    scale_pressure : bool, optional
        Whether or not to scale the pressure profile to maintain force balance.
    copy : bool, optional
        Whether to rescale the original equilibrium (default) or a copy.
    verbose : int
        Level of output.

    Returns
    -------
    eq : Equilibrium or iterable of Equilibrium
        Same as input, but rescaled to the desired size and magnetic field strength.
        The profiles will now be ``ScaledProfile` classes, with the ``scale`` being
        the rescaling parameter computed in this function.

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
    B_keys = ["B0", "<B>", "B_max"]

    if L_key not in L_keys:
        raise ValueError("Size scale L must be one of {{'R0', 'a', 'V'}}, got " + L_key)
    if B_key not in B_keys:
        raise ValueError(
            "Field strength scale B must be one of {{'B0', '<B>', 'B_max'}}, got "
            + B_key
        )

    if copy:
        eq = eq.copy()

    # size scaling
    grid_L = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data_L = eq.compute(L_keys, grid=grid_L)
    L_old = data_L[L_key]
    L_new = L_new or L_old
    cL = L_new / L_old
    cL = cL ** (1 / 3) if L_key == "V" else cL  # V = 2 π^2 R0 a^2
    cL = float(cL.squeeze())
    # field scaling
    if B_key == "B0":
        grid_B = LinearGrid(N=eq.N_grid, NFP=eq.NFP, rho=0)
        data_B = eq.compute("|B|", grid=grid_B)
        B_old = np.mean(data_B["|B|"])
    elif B_key == "<B>":
        grid_B = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        data_B = eq.compute("<|B|>_vol", grid=grid_B)
        B_old = data_B["<|B|>_vol"]
    elif B_key == "B_max":
        grid_B = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=1)
        data_B = eq.compute("|B|", grid=grid_B)
        B_old = np.max(data_B["|B|"])
    B_new = B_new or B_old
    cB = B_new / B_old
    cB = float(cB.squeeze())

    # scaling factor = desired / actual
    if verbose:
        print("Size scaling factor:  {:.2f}".format(cL))
        print("Field scaling factor: {:.2f}".format(cB))

    # scale flux surfaces
    eq.R_lmn *= cL
    eq.Z_lmn *= cL
    eq.Psi *= cL**2 * cB

    # scale pressure profile
    if scale_pressure:
        if eq.pressure is not None:
            eq.pressure *= cB**2
        else:
            eq.electron_density *= cB
            eq.electron_temperature *= cB
            eq.ion_temperature *= cB

    # scale current profile
    if eq.current is not None:
        eq.current *= cL * cB

    # boundary & axis
    eq.axis = eq.get_axis()
    eq.surface = eq.get_surface_at(rho=1)

    return eq


def rotate_zeta(eq, angle, copy=False):
    """Rotate the equilibrium about the toroidal direction.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to rotate.
    angle : float
        Angle to rotate the equilibrium in radians. The actual physical rotation
        is by angle radians. Any rotation that is not a multiple of pi/NFP will
        break the symmetry of a stellarator symmetric equilibrium.
    copy : bool, optional
        Whether to update the existing equilibrium or make a copy (Default).

    Returns
    -------
    eq_rotated : Equilibrium
        Equilibrium rotated about the toroidal direction
    """
    eq_rotated = eq.copy() if copy else eq
    # We will apply the rotation in NFP domain
    angle = angle * eq.NFP
    # Check if the angle is a multiple of pi/NFP
    kpi = np.isclose(angle % np.pi, 0, 1e-8, 1e-8) or np.isclose(
        angle % np.pi, np.pi, 1e-8, 1e-8
    )
    if eq.sym and not kpi and eq.N != 0:
        warnings.warn(
            "Rotating a stellarator symmetric equilibrium by an angle "
            "that is not a multiple of pi/NFP will break the symmetry. "
            "Changing the symmetry to False to rotate the equilibrium."
        )
        eq_rotated.change_resolution(sym=0)

    def _get_new_coeffs(fun):
        if fun == "R":
            f_lmn = np.array(eq_rotated.R_lmn)
            basis = eq_rotated.R_basis
        elif fun == "Z":
            f_lmn = np.array(eq_rotated.Z_lmn)
            basis = eq_rotated.Z_basis
        elif fun == "L":
            f_lmn = np.array(eq_rotated.L_lmn)
            basis = eq_rotated.L_basis
        else:
            raise ValueError("fun must be 'R', 'Z' or 'L'")

        new_coeffs = f_lmn.copy()
        for i, (l, m, n) in enumerate(basis.modes):
            id_sin = basis.get_idx(L=l, M=m, N=-n, error=False)
            v_sin = np.sin(np.abs(n) * angle)
            v_cos = np.cos(np.abs(n) * angle)
            c_sin = f_lmn[id_sin] if isinstance(id_sin, int) else 0
            if n >= 0:
                new_coeffs[i] = f_lmn[i] * v_cos + c_sin * v_sin
            elif n < 0:
                new_coeffs[i] = f_lmn[i] * v_cos - c_sin * v_sin
        return new_coeffs

    eq_rotated.R_lmn = _get_new_coeffs(fun="R")
    eq_rotated.Z_lmn = _get_new_coeffs(fun="Z")
    eq_rotated.L_lmn = _get_new_coeffs(fun="L")

    eq_rotated.surface = eq_rotated.get_surface_at(rho=1.0)
    eq_rotated.axis = eq_rotated.get_axis()

    return eq_rotated


def contract_equilibrium(
    eq, inner_rho, contract_profiles=True, profile_num_points=None, copy=True
):
    """Contract an equilibrium so that an inner flux surface is the new boundary.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to contract.
    inner_rho: float
        rho value (<1) to contract the Equilibrium to.
    contract_profiles :  bool
        Whether or not to contract the profiles.
        If True, the new profile's value at ``rho=1.0`` will be the same as the old
        profile's value at ``rho=inner_rho``, i.e. in physical space, the new
        profile is the same as the old profile. If the profile is a
        ``PowerSeriesProfile`` or ``SplineProfile``, the same profile type will be
        returned. If not one of these two classes, a ``SplineProfile`` will be returned
        as other profile classes cannot be safely contracted.
        If False, the new profile will have the same functional form
        as the old profile, with no rescaling performed. This means the new equilibrium
        has a physically different profile than the original equilibrium.
    profile_num_points : int
        Number of points to use when fitting or re-scaling the profiles. Defaults to
        ``eq.L_grid``
    copy : bool
        Whether or not to return a copy or to modify the original equilibrium.

    Returns
    -------
    eq_inner: New Equilibrium object, contracted from the old one such that
        eq.pressure(rho=inner_rho) = eq_inner.pressure(rho=1), and
        eq_inner LCFS = eq's rho=inner_rho surface.
        Note that this will not be in force balance, and so must be re-solved.

    """
    errorif(
        not (inner_rho < 1 and inner_rho > 0),
        ValueError,
        f"Surface must be in the range 0 < inner_rho < 1, instead got {inner_rho}.",
    )
    profile_num_points = setdefault(profile_num_points, eq.L_grid)

    def scale_profile(profile, rho):
        errorif(
            isinstance(profile, FourierZernikeProfile),
            ValueError,
            "contract_equilibrium does not support FourierZernikeProfile",
        )
        if profile is None:
            return profile
        is_power_series = isinstance(profile, PowerSeriesProfile)
        if contract_profiles and is_power_series:
            # only PowerSeriesProfile both
            # a) has a from_values
            # b) can safely use that from_values to represent a
            #    subset of itself.
            x = np.linspace(0, 1, profile_num_points)
            grid = LinearGrid(rho=x / rho)
            y = profile.compute(grid)
            return profile.from_values(x=x, y=y)
        elif contract_profiles:
            warnif(
                not isinstance(profile, SplineProfile),
                UserWarning,
                f"{profile} is not a PowerSeriesProfile or SplineProfile,"
                " so cannot safely contract using the same profile type."
                "Falling back to fitting the values with a SplineProfile",
            )
            x = np.linspace(0, 1, profile_num_points)
            grid = LinearGrid(rho=x / rho)
            y = profile.compute(grid)
            return SplineProfile(knots=x, values=y)
        else:  # don't do any scaling of the profile
            return profile

    # create new profiles for contracted equilibrium
    pressure = scale_profile(eq.pressure, inner_rho)
    iota = scale_profile(eq.iota, inner_rho)
    current = scale_profile(eq.current, inner_rho)
    electron_density = scale_profile(eq.electron_density, inner_rho)
    electron_temperature = scale_profile(eq.electron_temperature, inner_rho)
    ion_temperature = scale_profile(eq.ion_temperature, inner_rho)
    atomic_number = scale_profile(eq.atomic_number, inner_rho)
    anisotropy = scale_profile(eq.anisotropy, inner_rho)

    surf_inner = eq.get_surface_at(rho=inner_rho)
    surf_inner.rho = 1.0
    from .equilibrium import Equilibrium

    eq_inner = Equilibrium(
        surface=surf_inner,
        pressure=pressure,
        iota=iota,
        current=current,
        electron_density=electron_density,
        electron_temperature=electron_temperature,
        ion_temperature=ion_temperature,
        atomic_number=atomic_number,
        anisotropy=anisotropy,
        Psi=(
            eq.Psi * inner_rho**2
        ),  # flux (in Webers) within the new last closed flux surface
        NFP=eq.NFP,
        L=eq.L,
        M=eq.M,
        N=eq.N,
        L_grid=eq.L_grid,
        M_grid=eq.M_grid,
        N_grid=eq.N_grid,
        sym=eq.sym,
        ensure_nested=False,  # we fit the surfaces later so don't check now
    )
    inner_grid = LinearGrid(
        rho=np.linspace(0, inner_rho, eq.L_grid * 2),
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP,
        axis=True,
    )
    inner_data = eq.compute(["R", "Z", "lambda"], grid=inner_grid)
    nodes = inner_grid.nodes
    nodes[:, 0] = nodes[:, 0] / inner_rho
    eq_inner.set_initial_guess(
        nodes, inner_data["R"], inner_data["Z"], inner_data["lambda"]
    )
    if not copy:  # overwrite the original eq
        eq.__dict__.update(eq_inner.__dict__)
        return eq
    return eq_inner
