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
from desc.profiles import PowerSeriesProfile, _Profile


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
            grid = LinearGrid(rho=1, theta=2, zeta=surface.N * 2, NFP=surface.NFP)
            data = surface.compute(["R", "Z"], grid=grid)
            R = data["R"]
            Z = data["Z"]
            Rout = R[::2]
            Rin = R[1::2]
            Zout = Z[::2]
            Zin = Z[1::2]
            # TODO: depending on the beta value, shift the mid point to the outside
            Rmid = (Rout + Rin) / 2
            Zmid = (Zout + Zin) / 2
            phis = jnp.linspace(
                0, 2 * np.pi / surface.NFP, surface.N * 2, endpoint=False
            )
            axis = FourierRZCurve.from_values(
                jnp.vstack([Rmid, phis, Zmid]).T, N=surface.N, NFP=surface.NFP
            )
        elif isinstance(surface, ZernikeRZToroidalSection):
            # TODO (#782): include m=0 l!=0 modes
            grid = LinearGrid(rho=1, theta=2, zeta=surface.N * 2)
            data = surface.compute(["R", "Z"], grid=grid)
            R = data["R"]
            Z = data["Z"]
            Rout = R[::2]
            Rin = R[1::2]
            Zout = Z[::2]
            Zin = Z[1::2]
            Rmid = (Rout + Rin) / 2
            Zmid = (Zout + Zin) / 2
            phis = jnp.zeros_like(Rmid)
            axis = FourierRZCurve.from_values(
                jnp.vstack([Rmid, phis, Zmid]).T, N=0, NFP=surface.NFP
            )
    else:
        raise TypeError("Got unknown axis type {}".format(axis))
    return axis
