"""Utility functions for parsing inputs to Equilibrium."""

import numbers

import numpy as np

from desc.backend import jnp
from desc.equilibrium import Equilibrium
from desc.profiles import PowerSeriesProfile, Profile, SplineProfile


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
    if isinstance(prof, Profile):
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
        n_knots = profile.params.size()
        new_knots = np.linspace(0, 1, n_knots, endpoint=True)
        new_vals = profile(np.linspace(0, inner_rho, endpoint=True))
        new_profile = SplineProfile(params=new_vals, knots=new_knots)
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
            eq_inner LCFS = eq's rho=inner_rho surface
    """
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
        bdry_mode=eq.bdry_mode,
    )
    return eq_inner
