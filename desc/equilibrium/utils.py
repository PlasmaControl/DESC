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
from desc.profiles import PowerSeriesProfile, _Profile
from desc.utils import warnif


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


def ensure_consistent_profile_eq_resolution(profile, eq, name=""):
    """Ensure that the profile resolution is consistent with the Equilibrium resolution.

    Parameters
    ----------
    profile : Profile
        Profile to check.
    eq : Equilibrium
        Equilibrium to check against.
    name : str, optional
        Name of the profile.

    Returns
    -------
    profile : Profile
        Profile with updated resolution if necessary.
    """
    if profile is None:
        return None
    if hasattr(profile, "change_resolution"):
        profile.change_resolution(max(profile.basis.L, eq.L))

    warnif(
        isinstance(profile, PowerSeriesProfile) and profile.sym != "even",
        msg=f"{name} profile is not an even power series.",
    )

    return profile


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

    Returns
    -------
    surface : FourierRZToroidalSurface
        Parsed surface object.
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
        else:
            raise ValueError("boundary surface should have l=0")
    else:
        raise TypeError("Got unknown surface type {}".format(surface))

    return surface


def parse_axis(axis, NFP=1, sym=True, surface=None, xsection=None):
    """Parse axis input into Curve object.

    Parameters
    ----------
    axis : Curve, ndarray, None
        Axis to parse.
    NFP : int
        Number of field periods of the Equilibrium.
    sym : bool
        Stellarator symmetry of the Equilibrium.
    surface: FourierRZToroidalSurface
        Last closed flux surface to get axis from
    xsection: ZernikeRZToroidalSection
        Poincare cross-section at given zeta toroidal angle
        If supplied, axis will be axisymmetic

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
        if isinstance(surface, FourierRZToroidalSurface):
            axis = surface.get_axis()
        elif isinstance(xsection, ZernikeRZToroidalSection):
            axis = xsection.get_axis()
            # ZernikeRZToroidalSection doesn't have NFP, so we need to
            # set it manually
            axis._NFP = NFP
    else:
        raise TypeError("Got unknown axis type {}".format(axis))
    return axis


def parse_section(xsection=None, surface=None, sym=True, spectral_indexing="ansi"):
    """Parse section input into ZernikeRZToroidalSection object.

    Parameters
    ----------
    xsection : ZernikeRZToroidalSection, None
        Poincare surface object to parse.
    surface: FourierRZToroidalSurface
        Last closed flux surface to get section from. Only used for resolution.
        The proper coefficients will be set after set_initial_guess.
    sym : bool
        Stellarator symmetry of the Equilibrium.
    spectral_indexing : {"ansi", "fringe"}
        Spectral indexing scheme of the Equilibrium.

    Returns
    -------
    xsection : ZernikeRZToroidalSection
        Parsed Poincare section object.
    """
    if isinstance(xsection, ZernikeRZToroidalSection):
        _xsection = xsection
    elif isinstance(xsection, (np.ndarray, jnp.ndarray)):
        raise NotImplementedError(
            "ZernikeRZToroidalSection from input file not implemented"
        )
    else:
        _xsection = ZernikeRZToroidalSection(
            L=surface.M, M=surface.M, sym=sym, spectral_indexing=spectral_indexing
        )
    return _xsection
