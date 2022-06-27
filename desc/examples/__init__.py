import os
import desc.io


def get(name, which=None):
    """Get example equilibria and data

    Returns a solved equilibrium or selected attributes for one of several examples.

    current examples include: dshape, solovev, atf, heliotron

    Parameters
    ----------
    name : str
        name of the example equilibrium to load from, should be one from list above
    which : {None, "all", "boundary", "pressure", "iota"}
        what data to return. None returns the final solved equilibrium. "all" returns
        the intermediate solutions from the continuation method as an EquilibriaFamily.
        "boundary" returns a representation for the LCFS. "pressure" and "iota" return
        profile objects.

    Returns
    -------
    data : varies
        data requested, see "which" argument for more details

    """

    assert which in {None, "all", "boundary", "pressure", "iota"}
    here = os.path.abspath(os.path.dirname(__file__)) + "/"
    path = here + name.upper() + "_output.h5"
    if os.path.exists(path):
        eqf = desc.io.load(path)
    else:
        raise ValueError("example {} not found".format(path))
    if which is None:
        return eqf[-1]
    if which == "all":
        return eqf
    if which == "boundary":
        return eqf[-1].get_surface_at(rho=1)
    if which == "pressure":
        return eqf[-1].pressure
    if which == "iota":
        return eqf[-1].iota
