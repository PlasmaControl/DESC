"""Module for getting precomputed example equilibria."""

import os

import desc.io


def get(name, data=None):
    """Get example equilibria and data.

    Returns a solved equilibrium or selected attributes for one of several examples.

    Examples include: SOLOVEV, DSHAPE, DSHAPE_current, HELIOTRON, ATF, ESTELL,
    WISTELL-A, ARIES-CS, QAS, NCSX, W7-X

    Parameters
    ----------
    name : str
        Name of the example equilibrium to load from, should be one from list above.
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
    here = os.path.abspath(os.path.dirname(__file__)) + "/"
    path = here + name + "_output.h5"
    if os.path.exists(path):
        eqf = desc.io.load(path)
    else:
        raise ValueError("example {} not found".format(path))
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
