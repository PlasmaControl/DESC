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
from desc.utils import errorif


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


def scale_profile(profile, inner_rho):
    """Return a new profile whose rho=1 val is value of the old profile at inner_rho.

    Args
    ----
        profile (Profile): Profile to scale
        inner_rho (float): rho to take as the new LCFS rho for the new profile
    Returns
    -------
        new_profile (Profile): new profile whose last value is the inner_rho value of
        the original profile
    """
    params_list = []
    modes_list = []
    if isinstance(profile, PowerSeriesProfile):
        for coeff, mode in zip(profile.params, profile.basis.modes):
            l = mode[0]
            params_list.append(coeff * inner_rho**l)
            modes_list.append(l)
        new_profile = PowerSeriesProfile(params=params_list, modes=modes_list)
    elif isinstance(profile, SplineProfile):
        n_knots = profile.params.size
        new_knots = np.linspace(0, 1, n_knots, endpoint=True)
        new_vals = profile(np.linspace(0, inner_rho, n_knots, endpoint=True))
        new_profile = SplineProfile(
            values=new_vals, knots=new_knots, method=profile._method
        )
    else:
        raise NotImplementedError(
            f"Expected PowerSeriesProfile or SplineProfile, got {type(profile)}"
        )
    return new_profile


def contract_equilibrium(eq, inner_rho):
    """Create a new equilibrium by using an inner surface of the passed-in equilibrium.

    Args
    ----
        eq (Equilibrium): Equilibrium to contract.
        inner_rho (float): rho value (<1) to contract the Equilibrium to

    Returns
    -------
        eq_inner: New Equilibrium object, contracted from the old one such that
            eq.pressure(rho=inner_rho) = eq_inner.pressure(rho=1), and
            eq_inner LCFS = eq's rho=inner_rho surface.
            Note that this will not be in force balance, and so must be re-solved.
    """
    errorif(not (inner_rho < 1 and inner_rho > 0), ValueError, "inner_rho should be <1")
    # create new profiles for contracted equilibrium
    # pressure
    pressure = scale_profile(eq.pressure, inner_rho)
    current = None
    iota = None
    if eq.iota is not None:
        iota = scale_profile(eq.iota, inner_rho)
    elif eq.current is not None:
        current = scale_profile(eq.current, inner_rho)

    surf_inner = eq.get_surface_at(rho=inner_rho)
    surf_inner.rho = 1.0
    from .equilibrium import Equilibrium

    eq_inner = Equilibrium(
        surface=surf_inner,
        pressure=pressure,
        iota=iota,
        current=current,
        Psi=eq.Psi
        * inner_rho**2,  # flux (in Webers) within the last closed flux surface
        NFP=eq.NFP,
        L=eq.L,
        M=eq.M,
        N=eq.N,
        L_grid=eq.L_grid,
        M_grid=eq.M_grid,
        N_grid=eq.N_grid,
        sym=eq.sym,
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

    return eq_inner
