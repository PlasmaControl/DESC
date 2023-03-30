"""Utility functions for parsing inputs to Equilibrium."""

import numbers

import numpy as np

from desc.backend import jnp
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
