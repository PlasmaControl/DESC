"""Functions for computing field and plasma quantities from an equilibrium.

All compute functions take the following arguments:

Parameters
----------
params : dict of ndarray
    Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
transforms : dict of Transform
    Transforms for R, Z, lambda, etc
profiles : dict of Profile
    Profile objects for pressure, iota, current, etc
data : dict of ndarray
    Data computed so far, generally output from other compute functions
kwargs : dict
    Other arguments needed by some functions, such as helicity

Returns
-------
data : dict of ndarray
    Dictionary of ndarray, shape(num_nodes,) of computed quantities.
    Keys are of the form 'X_y' meaning the derivative of X wrt y.

"""

# just need to import all the submodules here to register everything in the
# data_index

from . import (
    _basis_vectors,
    _bootstrap,
    _core,
    _equil,
    _field,
    _geometry,
    _metric,
    _profiles,
    _qs,
    _stability,
)
from .data_index import data_index
from .utils import (
    arg_order,
    compute,
    get_data_deps,
    get_derivs,
    get_params,
    get_profiles,
    get_transforms,
    profile_names,
)


# rather than having to recursively compute the full dependencies every time we
# compute something, it's easier to just do it once for all quantities when we first
# import the compute module.
def _build_data_index():
    for key in data_index.keys():
        full = {
            "data": get_data_deps(key, False),
            "transforms": get_derivs(key, False),
            "params": get_params(key, False),
            "profiles": get_profiles(key, False),
        }
        data_index[key]["full_dependencies"] = full

        full_and_axis_data = get_data_deps(key, True)
        if len(full["data"]) >= len(full_and_axis_data):
            # Then this quantity lacks dependencies that would only be required
            # to evaluate this quantity's limit at the magnetic axis.
            full_and_axis = full
        else:
            full_and_axis = {
                "data": full_and_axis_data,
                "transforms": get_derivs(key, True),
                "params": get_params(key, True),
                "profiles": get_profiles(key, True),
            }
            for _key, val in full_and_axis.items():
                if len(full[_key]) >= len(val):
                    # one is a deep copy of the other; dereference to save memory
                    full_and_axis[_key] = full[_key]
        data_index[key]["full_and_axis_dependencies"] = full_and_axis


_build_data_index()
