"""Utility functions for parsing inputs to Equilibrium."""

import numbers

import numpy as np

from desc.backend import jnp
from desc.geometry import (
    FourierRZCurve,
    FourierRZToroidalSurface,
    Surface,
    ZernikeRZToroidalSection,
)
from desc.grid import LinearGrid
from desc.profiles import PowerSeriesProfile, SplineProfile, _Profile
from desc.utils import errorif, warnif


def parse_profile(prof, name="", **kwargs):
    """Parse an object into a Profile.

    Parameters
    ----------
    prof : numeric, ndarray, Profile
        Object to parse. Profile objects are returned unaltered. Scalars are treated
        as a constant profile. Arrays of shape(k,) are treated as coefficients of
        power series, arrays of shape (k,2) are treated as mode numbers and coefficients
        of a power series. If None, returns None
    name : str, optional
        Name to assign to profile if prof is numeric or ndarray
    kwargs: any
        Additional keyword arguments are passed through to Profile constructor if prof
        is numeric or ndarray.

    Returns
    -------
    prof : Profile or None
        Profile object, or None if input is None

    Raises
    ------
    TypeError
        If the object cannot be parsed as a Profile
    """
    if isinstance(prof, _Profile):
        return prof
    if isinstance(prof, numbers.Number) or (
        isinstance(prof, (np.ndarray, jnp.ndarray)) and prof.ndim == 1
    ):
        return PowerSeriesProfile(params=prof, name=name, **kwargs)
    if (
        isinstance(prof, (np.ndarray, jnp.ndarray))
        and prof.ndim == 2
        and prof.shape[1] == 2
    ):
        return PowerSeriesProfile(
            modes=prof[:, 0], params=prof[:, 1], name=name, **kwargs
        )
    if prof is None:
        return None
    raise TypeError(f"Got unknown {name} profile {prof}")


def parse_surface(surface, NFP=1, sym=True, spectral_indexing="ansi"):
    """Parse surface input into Surface object.

    Parameters
    ----------
    surface : Surface, ndarray, None
        Surface to parse.
    NFP : int
        Number of field periods of the Equilibrium.
    sym : bool
        Stellarator symmetry of the Equilibrium.
    spectral_indexing : {"ansi", "fringe"}
        Spectral indexing scheme of the Equilibrium.

    Returns
    -------
    surface : Surface
        Parsed surface object, either FourierRZToroidalSurface or
        ZernikeRZToroidalSection.
    bdry_mode : str
        Either "lcfs" or "poincare"
    """
    if isinstance(surface, Surface):
        surface = surface
    elif surface is None:
        surface = FourierRZToroidalSurface(NFP=NFP, sym=sym)
    elif isinstance(surface, (np.ndarray, jnp.ndarray)):
        if np.all(surface[:, 0] == 0):
            surface = FourierRZToroidalSurface(
                surface[:, 3],
                surface[:, 4],
                surface[:, 1:3].astype(int),
                surface[:, 1:3].astype(int),
                NFP,
                sym,
                check_orientation=False,
            )
        elif np.all(surface[:, 2] == 0):
            surface = ZernikeRZToroidalSection(
                surface[:, 3],
                surface[:, 4],
                surface[:, :2].astype(int),
                surface[:, :2].astype(int),
                spectral_indexing,
                sym,
            )
        else:
            raise ValueError("boundary should either have l=0 or n=0")
    else:
        raise TypeError("Got unknown surface type {}".format(surface))

    if isinstance(surface, FourierRZToroidalSurface):
        bdry_mode = "lcfs"
    if isinstance(surface, ZernikeRZToroidalSection):
        bdry_mode = "poincare"
    return surface, bdry_mode


def parse_axis(axis, NFP=1, sym=True, surface=None):
    """Parse axis input into Curve object.

    Parameters
    ----------
    axis : Curve, ndarray, None
        Axis to parse.
    NFP : int
        Number of field periods of the Equilibrium.
    sym : bool
        Stellarator symmetry of the Equilibrium.

    Returns
    -------
    axis : Curve
        Parsed axis object.
    """
    if isinstance(axis, FourierRZCurve):
        axis = axis
    elif isinstance(axis, (np.ndarray, jnp.ndarray)):
        axis = FourierRZCurve(
            axis[:, 1],
            axis[:, 2],
            axis[:, 0].astype(int),
            axis[:, 0].astype(int),
            NFP=NFP,
            sym=sym,
            name="axis",
        )
    elif axis is None:  # use the center of surface
        # TODO (#1384): make this method of surface, surface.get_axis()?
        if isinstance(surface, FourierRZToroidalSurface):
            axis = FourierRZCurve(
                R_n=surface.R_lmn[np.where(surface.R_basis.modes[:, 1] == 0)],
                Z_n=surface.Z_lmn[np.where(surface.Z_basis.modes[:, 1] == 0)],
                modes_R=surface.R_basis.modes[
                    np.where(surface.R_basis.modes[:, 1] == 0)[0], -1
                ],
                modes_Z=surface.Z_basis.modes[
                    np.where(surface.Z_basis.modes[:, 1] == 0)[0], -1
                ],
                NFP=NFP,
            )
        elif isinstance(surface, ZernikeRZToroidalSection):
            # TODO (#782): include m=0 l!=0 modes
            axis = FourierRZCurve(
                R_n=surface.R_lmn[
                    np.where(
                        (surface.R_basis.modes[:, 0] == 0)
                        & (surface.R_basis.modes[:, 1] == 0)
                    )
                ].sum(),
                Z_n=surface.Z_lmn[
                    np.where(
                        (surface.Z_basis.modes[:, 0] == 0)
                        & (surface.Z_basis.modes[:, 1] == 0)
                    )
                ].sum(),
                modes_R=[0],
                modes_Z=[0],
                NFP=NFP,
            )
    else:
        raise TypeError("Got unknown axis type {}".format(axis))
    return axis


def contract_equilibrium(eq, inner_rho, contract_profiles=True, copy=True):
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

    def scale_profile(profile, rho):
        if profile is None:
            return profile
        is_power_series = isinstance(profile, PowerSeriesProfile)
        if contract_profiles and is_power_series:
            # only PowerSeriesProfile both
            # a) has a from_values
            # b) can safely use that from_values to represent a
            #    subset of itself.
            x = np.linspace(0, 1, eq.L_grid)
            grid = LinearGrid(rho=x / rho)
            y = profile.compute(grid)
            return profile.from_values(x=x, y=y)
        elif contract_profiles:
            warnif(
                not isinstance(profile, SplineProfile),
                UserWarning,
                f"{profile} is not a PowerSeriesProfile or SplineProfile,"
                " so cannot safely contract using the same profile type."
                "falling back to fitting the values with a SplineProfile",
            )
            x = np.linspace(0, 1, eq.L_grid)
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
        Psi=float(
            eq.compute("Psi", grid=LinearGrid(rho=inner_rho, NFP=eq.NFP))["Psi"][0]
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
        eq = eq_inner
        return eq
    return eq_inner
