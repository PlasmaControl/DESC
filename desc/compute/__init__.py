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


# Rather than having to recursively compute the full dependencies every time we
# compute something, it's easier to just do it once for all quantities when we first
# import the compute module.
def _build_data_index():
    for key in data_index.keys():
        full = {
            "data": get_data_deps(key, has_axis=False),
            "transforms": get_derivs(key, has_axis=False),
            "params": get_params(key, has_axis=False),
            "profiles": get_profiles(key, has_axis=False),
        }
        data_index[key]["full_dependencies"] = full

        full_with_axis_data = get_data_deps(key, has_axis=True)
        if len(full["data"]) >= len(full_with_axis_data):
            # Then this quantity and all its dependencies do not need anything
            # extra to evaluate its limit at the magnetic axis.
            # The dependencies in the `full` dictionary and the `full_with_axis`
            # dictionary will be identical, so we assign the same reference to
            # avoid storing a copy.
            full_with_axis = full
        else:
            full_with_axis = {
                "data": full_with_axis_data,
                "transforms": get_derivs(key, has_axis=True),
                "params": get_params(key, has_axis=True),
                "profiles": get_profiles(key, has_axis=True),
            }
            for _key, val in full_with_axis.items():
                if full[_key] == val:
                    # Nothing extra was needed to evaluate this quantity's limit.
                    # One is a copy of the other; dereference to save memory.
                    full_with_axis[_key] = full[_key]
        data_index[key]["full_with_axis_dependencies"] = full_with_axis


_build_data_index()
