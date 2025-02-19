"""Functions for computing field and plasma quantities from an equilibrium.

All compute functions take the following arguments:

Parameters
----------
params : dict of ndarray
    Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc.
transforms : dict of Transform
    Transforms for R, Z, lambda, etc.
profiles : dict of Profile
    Profile objects for pressure, iota, current, etc.
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
    _curve,
    _equil,
    _field,
    _geometry,
    _metric,
    _neoclassical,
    _old,
    _omnigenity,
    _profiles,
    _stability,
    _surface,
    _turbulence,
)
from .data_index import all_kwargs, allowed_kwargs, data_index
from .geom_utils import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from .utils import (
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

    for p in data_index:
        for key in data_index[p]:
            full = {
                "data": get_data_deps(key, p, has_axis=False, basis="rpz"),
                "transforms": get_derivs(key, p, has_axis=False, basis="rpz"),
                "params": get_params(key, p, has_axis=False, basis="rpz"),
                "profiles": get_profiles(key, p, has_axis=False, basis="rpz"),
            }
            data_index[p][key]["full_dependencies"] = full

            full_with_axis_data = get_data_deps(key, p, has_axis=True)
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
                    "transforms": get_derivs(key, p, has_axis=True, basis="rpz"),
                    "params": get_params(key, p, has_axis=True, basis="rpz"),
                    "profiles": get_profiles(key, p, has_axis=True, basis="rpz"),
                }
                for _key, val in full_with_axis.items():
                    if full[_key] == val:
                        # Nothing extra was needed to evaluate this quantity's limit.
                        # One is a copy of the other; dereference to save memory.
                        full_with_axis[_key] = full[_key]
            data_index[p][key]["full_with_axis_dependencies"] = full_with_axis


_build_data_index()


def set_tier(name, p):
    """Determine how deep in the dependency tree a given name is.

    tier of 0 means no dependencies on other data,
    tier of 1 means it depends on only tier 0 stuff,
    tier of 2 means it depends on tier 0 and tier 1, etc etc.

    Designed such that if you compute things in the order determined by tiers,
    all dependencies will always be computed in the correct order.
    """
    if "tier" in data_index[p][name]:
        return
    if len(data_index[p][name]["full_with_axis_dependencies"]["data"]) == 0:
        data_index[p][name]["tier"] = 0
    else:
        thistier = 0
        for name1 in data_index[p][name]["full_with_axis_dependencies"]["data"]:
            set_tier(name1, p)
            thistier = max(thistier, data_index[p][name1]["tier"])
        data_index[p][name]["tier"] = thistier + 1


for par in data_index.keys():
    for name in data_index[par]:
        set_tier(name, par)
