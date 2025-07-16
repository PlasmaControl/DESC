"""Module for getting precomputed example equilibria."""

import os

import desc.io
from desc.backend import execute_on_cpu
from desc.equilibrium import EquilibriaFamily


def listall():
    """Return a list of examples that DESC has."""
    here = os.path.abspath(os.path.dirname(__file__))
    h5s = [f for f in os.listdir(here) if f.endswith(".h5")]
    names_stripped = [f.replace("_output.h5", "") for f in h5s]
    return names_stripped


@execute_on_cpu
def get(name, data=None):
    """Get example equilibria and data.

    Returns a solved equilibrium or selected attributes for one of several examples.

    A full list of valid names can be found with ``desc.examples.listall()``

    Parameters
    ----------
    name : str (case insensitive)
        Name of the example equilibrium to load from, should be one from
        ``desc.examples.listall()``.
    data : {None, "all", "boundary", "pressure", "iota", "current"}
        Data to return. None returns the final solved equilibrium. "all" returns the
        intermediate solutions from the continuation method as an EquilibriaFamily.
        "boundary" returns a representation for the last closed flux surface.
        "pressure", "iota", and "current" return the profile objects.

    Returns
    -------
    data : varies
        Data requested, see "data" argument for more details.

    """
    assert data in {None, "all", "boundary", "pressure", "iota", "current"}
    here = os.path.abspath(os.path.dirname(__file__))
    h5s = [f for f in os.listdir(here) if f.endswith(".h5")]
    h5s_lower = [f.lower() for f in h5s]
    filename = name.lower() + "_output.h5"
    try:
        idx = h5s_lower.index(filename)
    except ValueError as e:
        raise ValueError(
            "example {} not found, should be one of {}".format(name, listall())
        ) from e
    path = here + "/" + h5s[idx]
    assert os.path.exists(path)

    eqf = desc.io.load(path)
    if not isinstance(eqf, EquilibriaFamily):
        eqf = EquilibriaFamily(eqf)

    if data is None:
        return eqf[-1]
    if data == "all":
        return eqf
    if data == "boundary":
        return eqf[-1].get_surface_at(rho=1)
    if data == "pressure":
        return eqf[-1].pressure
    if data == "iota":
        return eqf[-1].iota
    if data == "current":
        return eqf[-1].current
