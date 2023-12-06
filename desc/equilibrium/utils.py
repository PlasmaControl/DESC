"""Utility functions for parsing inputs to Equilibrium."""

import numbers

import numpy as np

from desc.backend import jnp
from desc.basis import FourierZernike_to_FourierZernike_no_N_modes
from desc.equilibrium import Equilibrium
from desc.geometry import (
    FourierRZCurve,
    FourierRZToroidalSurface,
    Surface,
    ZernikeRZToroidalSection,
)
from desc.profiles import PowerSeriesProfile, _Profile
from desc.utils import isnonnegint


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
            NFP=NFP,
            sym=sym,
            name="axis",
        )
    elif axis is None:  # use the center of surface
        # TODO: make this method of surface, surface.get_axis()?
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
            # FIXME: include m=0 l!=0 modes
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


def _assert_nonnegint(x, name=""):
    assert (x is None) or isnonnegint(
        x
    ), f"{name} should be a non-negative integer or None, got {x}"


def set_poincare_equilibrium(eq, zeta=0):
    """Sets the equilibrium for solving Poincare BC problem.

    Parameters
    ----------
    eq : Equilibrium
        Some equilibrium to be used for creating Poincare equilibrium
    zeta : float (optional)
        Zeta angle at which the Poincare section will be fixed
        Only 0 and Pi is supported for now

    Returns
    -------
    eq_poincare : Equilibrium
        Equilibrium object to be used for Poincare BC problem
    """
    surface = eq.get_surface_at(zeta=zeta / eq.NFP)
    Lb_lmn, Lb_basis = FourierZernike_to_FourierZernike_no_N_modes(
        eq.L_lmn, eq.L_basis, zeta
    )

    eq_poincare = Equilibrium(
        surface=surface,
        pressure=eq.pressure,
        iota=eq.iota,
        Psi=eq.Psi,  # flux (in Webers) within the last closed flux surface
        NFP=eq.NFP,  # number of field periods
        L=eq.L,  # radial spectral resolution
        M=eq.M,  # poloidal spectral resolution
        N=eq.N,  # toroidal spectral resolution
        L_grid=eq.L_grid,  # real space radial resolution, slightly oversampled
        M_grid=eq.M_grid,  # real space poloidal resolution, slightly oversampled
        N_grid=eq.N_grid,  # real space toroidal resolution
        sym=True,  # explicitly enforce stellarator symmetry
        bdry_mode="poincare",
        spectral_indexing=eq._spectral_indexing,
    )
    eq_poincare.L_lmn = (
        Lb_lmn  # initialize the poincare eq with the lambda of the original eq
    )
    eq_poincare.axis = eq_poincare.get_axis()
    return eq_poincare
