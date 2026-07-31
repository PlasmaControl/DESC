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

import numpy as np

from ..utils import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from . import (
    _basis_vectors,
    _bootstrap,
    _core,
    _curve,
    _equil,
    _fast_ion,
    _field,
    _geometry,
    _metric,
    _neoclassical,
    _old,
    _omnigenity,
    _profiles,
    _stability,
    _surface,
)
from .data_index import _topological_order, all_kwargs, allowed_kwargs, data_index
from .utils import (
    compute,
    get_data_deps,
    get_derivs,
    get_params,
    get_profiles,
    get_transforms,
    profile_names,
)

# just need to import all the submodules here to register everything in the
# data_index


# Rather than having to recursively compute the full dependencies every time we
# compute something, it's easier to just do it once for all quantities when we first
# import the compute module.
def _build_data_index():  # noqa: C901
    """For each quantity in data_index, build the full set of dependencies.

    This function first performs a sort of the quantities such that the ones with
    no dependencies come first, then the ones that depend only on those, etc. Then
    it iterates through the quantities in that order, building the full dependency
    set for each one by taking the union of its direct dependencies and the full
    dependencies of those dependencies, which have already been computed by the time
    we get to this quantity.

    The first sorting is important to avoid deep recursion when building the full
    dependency sets. Since the first elements of the order have no dependencies,
    we can build their full dependency sets by simple union operation.

    Note: This function is originally written by Claude Code and reviewed by Yigit
    Gunsur Elmacioglu.
    """

    def _collect_deps(p, all_deps):
        """Collect transforms, params, profiles from a list of dependency keys.

        For each key in the data_index, we call this function with full set of
        dependencies of that key, and it collects the transforms, params, and
        profiles needed by all those dependencies in a single pass.
        """
        transforms = {}
        params = []
        profiles = []
        for k in all_deps:
            k_deps = data_index[p][k]["dependencies"]
            for tkey, tval in k_deps["transforms"].items():
                if tkey not in transforms:
                    transforms[tkey] = []
                transforms[tkey] += tval
            params += k_deps["params"]
            profiles += k_deps["profiles"]
        transforms = {k: np.unique(v, axis=0).tolist() for k, v in transforms.items()}
        profiles = sorted(set(profiles))
        return transforms, params, profiles

    for p in data_index:
        # --- Step 1: Topological sort via iterative Depth-First Search ---
        # We need to process quantities with no dependencies before the
        # quantities that depend on them. This way, when we process key K,
        # all of K's dependencies already have their full_dependencies and
        # full_with_axis_dependencies cached, and we can build K's full
        # dependency set with a simple set union instead of deep recursion.
        order = []
        visited = set()
        for start in data_index[p]:
            if start in visited:
                continue
            stack = [(start, False)]
            while stack:
                node, processed = stack.pop()
                if processed:
                    if node not in visited:
                        visited.add(node)
                        order.append(node)
                    continue
                if node in visited:
                    continue
                # Mark for post-processing after all deps are visited.
                stack.append((node, True))
                node_deps = data_index[p][node]["dependencies"]
                for dep in node_deps["data"]:
                    if dep not in visited:
                        stack.append((dep, False))
                for dep in node_deps["axis_limit_data"]:
                    if dep not in visited:
                        stack.append((dep, False))

        # --- Step 2: Build full dependency sets incrementally ---
        # Because we iterate in topological order, every dependency of the
        # current key already has its full_dependencies cached. So the full
        # transitive data deps of key K is just:
        #   union of (each direct dep D) + (D's already-cached full data deps)
        # No recursion needed — O(number of direct deps) per key.

        # The deps are stored in topological order (not alphabetical) so that
        # iterating over them computes quantities in valid dependency order.
        topo_index = {key: i for i, key in enumerate(order)}
        _topological_order[p] = topo_index
        for key in order:
            d = data_index[p][key]
            deps_info = d["dependencies"]
            direct_data = deps_info["data"]

            # Full data deps without axis limit contributions.
            # Apply union operation
            full_data_set = set()
            for dep in direct_data:
                full_data_set.add(dep)
                full_data_set.update(data_index[p][dep]["full_dependencies"]["data"])
            deps_no_axis = sorted(full_data_set, key=topo_index.__getitem__)

            transforms, params, profiles = _collect_deps(p, [key] + deps_no_axis)
            full = {
                "data": deps_no_axis,
                "transforms": transforms,
                "params": params,
                "profiles": profiles,
            }
            # Cache now so later keys can use it.
            d["full_dependencies"] = full

            # Full data deps including axis limit data contributions.
            # axis_limit_data lists extra quantities needed to evaluate limits
            # at the magnetic axis; these are only relevant when has_axis=True.
            axis_limit_data = deps_info["axis_limit_data"]
            full_data_axis_set = set()
            for dep in direct_data:
                full_data_axis_set.add(dep)
                full_data_axis_set.update(
                    data_index[p][dep]["full_with_axis_dependencies"]["data"]
                )
            for dep in axis_limit_data:
                full_data_axis_set.add(dep)
                full_data_axis_set.update(
                    data_index[p][dep]["full_with_axis_dependencies"]["data"]
                )
            deps_with_axis = sorted(full_data_axis_set, key=topo_index.__getitem__)

            if len(deps_no_axis) >= len(deps_with_axis):
                # This quantity and all its dependencies do not need anything
                # extra to evaluate its limit at the magnetic axis.
                # Assign the same reference to avoid storing a copy.
                full_with_axis = full
            else:
                transforms_a, params_a, profiles_a = _collect_deps(
                    p, [key] + deps_with_axis
                )
                full_with_axis = {
                    "data": deps_with_axis,
                    "transforms": transforms_a,
                    "params": params_a,
                    "profiles": profiles_a,
                }
                # transforms, params, and profiles can be the same for both full and
                # full_with_axis, so check if they are and if so, dereference the copy
                # to save memory.
                for _key, val in full_with_axis.items():
                    if full[_key] == val:
                        # Nothing extra was needed for this field.
                        # Dereference the copy to save memory.
                        full_with_axis[_key] = full[_key]

            d["full_with_axis_dependencies"] = full_with_axis


_build_data_index()
