"""Functions for flux surface averages and vector algebra operations."""

import copy
import warnings

import numpy as np
from termcolor import colored

from desc.backend import jnp
from desc.grid import ConcentricGrid

from .data_index import data_index

# defines the order in which objective arguments get concatenated into the state vector
arg_order = ("R_lmn", "Z_lmn", "L_lmn", "p_l", "i_l", "c_l", "Psi", "Rb_lmn", "Zb_lmn")


def _sort_args(args):
    return [arg for arg in arg_order if arg in args]


def compute(names, params, transforms, profiles, data=None, **kwargs):
    """Compute the quantity given by name on grid.

    Parameters
    ----------
    names : str or array-like of str
        Name(s) of the quantity(s) to compute.
    grid : Grid, optional
        Grid of coordinates to evaluate at. Defaults to the quadrature grid.
    params : dict of ndarray
        Parameters from the equilibrium, such as R_lmn, Z_lmn, i_l, p_l, etc
        Defaults to attributes of self.
    transforms : dict of Transform
        Transforms for R, Z, lambda, etc. Default is to build from grid
    profiles : dict of Profile
        Profile objects for pressure, iota, current, etc. Defaults to attributes
        of self
    data : dict of ndarray
        Data computed so far, generally output from other compute functions

    Returns
    -------
    data : dict of ndarray
        Computed quantity and intermediate variables.

    """
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name not in data_index:
            raise ValueError("Unrecognized value '{}'.".format(name))
    allowed_kwargs = {"helicity", "M_booz", "N_booz", "gamma"}
    bad_kwargs = set(kwargs.keys()).difference(allowed_kwargs)
    if len(bad_kwargs) > 0:
        raise ValueError(f"Unrecognized argument(s): {bad_kwargs}")

    for name in names:
        assert _has_params(name, params), f"Don't have params to compute {name}"
        assert _has_profiles(name, profiles), f"Don't have profiles to compute {name}"
        assert _has_transforms(
            name, transforms
        ), f"Don't have transforms to compute {name}"

    if data is None:
        data = {}

    data = _compute(
        names,
        params=params,
        transforms=transforms,
        profiles=profiles,
        data=data,
        **kwargs,
    )
    return data


def _compute(names, params, transforms, profiles, data=None, **kwargs):
    """Same as above but without checking inputs for faster recursion."""
    for name in names:
        if name in data:
            continue
        if has_dependencies(name, params, transforms, profiles, data):
            data = data_index[name]["fun"](params, transforms, profiles, data, **kwargs)
        else:
            data = _compute(
                data_index[name]["dependencies"]["data"],
                params=params,
                transforms=transforms,
                profiles=profiles,
                data=data,
                **kwargs,
            )
            data = data_index[name]["fun"](params, transforms, profiles, data, **kwargs)
    return data


def get_data_deps(keys):
    """Get list of data keys needed to compute a given quantity.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index

    Returns
    -------
    deps : list of str
        Names of quantities needed to compute key
    """
    keys = [keys] if isinstance(keys, str) else keys

    def _get_deps_1_key(key):
        if "full_dependencies" in data_index[key]:
            return data_index[key]["full_dependencies"]["data"]
        deps = data_index[key]["dependencies"]["data"]
        if len(deps) == 0:
            return deps
        out = deps.copy()
        for dep in deps:
            out += _get_deps_1_key(dep)
        return sorted(list(set(out)))

    out = []
    for key in keys:
        out += _get_deps_1_key(key)
    return sorted(list(set(out)))


def get_derivs(keys):
    """Get dict of derivative orders needed to compute a given quantity.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index

    Returns
    -------
    derivs : dict of list of int
        Orders of derivatives needed to compute key.
        Keys for R, Z, L, etc
    """
    keys = [keys] if isinstance(keys, str) else keys

    def _get_derivs_1_key(key):
        if "full_dependencies" in data_index[key]:
            return data_index[key]["full_dependencies"]["transforms"]
        deps = [key] + get_data_deps(key)
        derivs = {}
        for dep in deps:
            for key, val in data_index[dep]["dependencies"]["transforms"].items():
                if key not in derivs:
                    derivs[key] = []
                derivs[key] += val
        return derivs

    derivs = {}
    for key in keys:
        derivs1 = _get_derivs_1_key(key)
        for key1, val in derivs1.items():
            if key1 not in derivs:
                derivs[key1] = []
            derivs[key1] += val
    return {key: np.unique(val, axis=0).tolist() for key, val in derivs.items()}


def get_profiles(keys, eq=None, grid=None, **kwargs):
    """Get profiles needed to compute a given quantity on a given grid.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    eq : Equilibrium
        Equilibrium to compute quantity for.
    grid : Grid
        Grid to compute quantity on

    Returns
    -------
    profiles : list of str or dict of Profile
        Profiles needed to compute key.
        if eq is None, returns a list of the names of profiles needed
        otherwise, returns a dict of Profiles
        Keys for pressure, iota, etc.
    """
    keys = [keys] if isinstance(keys, str) else keys
    deps = list(keys) + get_data_deps(keys)
    profs = []
    for key in deps:
        profs += data_index[key]["dependencies"]["profiles"]
    # kludge for now to always get all profiles until we break up compute funs
    profs += ["iota", "current"]
    profs = sorted(list(set(profs)))
    if eq is None:
        return profs
    # need to use copy here because profile may be None
    profiles = {name: copy.deepcopy(getattr(eq, name)) for name in profs}
    if grid is None:
        return profiles
    for val in profiles.values():
        if val is not None:
            val.grid = grid
    return profiles


def get_params(keys, eq=None, **kwargs):
    """Get parameters needed to compute a given quantity.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    eq : Equilibrium
        Equilibrium to compute quantity for.

    Returns
    -------
    profiles : list of str or dict of ndarray
        Parameters needed to compute key.
        If eq is None, returns a list of the names of params needed
        otherwise, returns a dict of ndarray with keys for R_lmn, Z_lmn, etc.
    """
    keys = [keys] if isinstance(keys, str) else keys
    deps = list(keys) + get_data_deps(keys)
    params = []
    for key in deps:
        params += data_index[key]["dependencies"]["params"]
    params = _sort_args(list(set(params)))
    if eq is None:
        return params
    params = {name: np.atleast_1d(getattr(eq, name)).copy() for name in params}
    return params


def get_transforms(keys, eq, grid, **kwargs):
    """Get transforms needed to compute a given quantity on a given grid.

    Parameters
    ----------
    keys : str or array-like of str
        Name of the desired quantity from the data index
    eq : Equilibrium
        Equilibrium to compute quantity for.
    grid : Grid
        Grid to compute quantity on

    Returns
    -------
    transforms : dict of Transform
        Transforms needed to compute key.
        Keys for R, Z, L, etc

    """
    from desc.basis import DoubleFourierSeries
    from desc.transform import Transform

    keys = [keys] if isinstance(keys, str) else keys
    derivs = get_derivs(keys)
    transforms = {"grid": grid}
    for c in ["R", "L", "Z"]:
        if c in derivs:
            transforms[c] = Transform(
                grid, getattr(eq, c + "_basis"), derivs=derivs[c], build=True
            )
    if "B" in derivs:
        transforms["B"] = Transform(
            grid,
            DoubleFourierSeries(
                M=kwargs.get("M_booz", 2 * eq.M),
                N=kwargs.get("N_booz", 2 * eq.N),
                NFP=eq.NFP,
                sym=eq.R_basis.sym,
            ),
            derivs=derivs["B"],
            build=True,
            build_pinv=True,
        )
    if "w" in derivs:
        transforms["w"] = Transform(
            grid,
            DoubleFourierSeries(
                M=kwargs.get("M_booz", 2 * eq.M),
                N=kwargs.get("N_booz", 2 * eq.N),
                NFP=eq.NFP,
                sym=eq.Z_basis.sym,
            ),
            derivs=derivs["w"],
            build=True,
            build_pinv=True,
        )
    return transforms


def has_dependencies(qty, params, transforms, profiles, data):
    """Determine if we have the ingredients needed to compute qty.

    Parameters
    ----------
    qty : str
        Name of something from the data index.
    params : dict of ndarray
        Dictionary of parameters we have.
    transforms : dict of Transform
        Dictionary of transforms we have.
    profiles : dict of Profile
        Dictionary of profiles we have.
    data : dict of ndarray
        Dictionary of what we've computed so far.

    Returns
    -------
    has_dependencies : bool
        Whether we have what we need.
    """
    return (
        _has_data(qty, data)
        and _has_params(qty, params)
        and _has_profiles(qty, profiles)
        and _has_transforms(qty, transforms)
    )


def _has_data(qty, data):
    if qty in data:  # don't compute something that's already been computed
        return False
    deps = data_index[qty]["dependencies"]["data"]
    return all(d in data for d in deps)


def _has_params(qty, params):
    deps = data_index[qty]["dependencies"]["params"]
    return all(d in params for d in deps)


def _has_profiles(qty, profiles):
    deps = data_index[qty]["dependencies"]["profiles"]
    return all(d in profiles for d in deps)


def _has_transforms(qty, transforms):
    flags = {}
    derivs = data_index[qty]["dependencies"]["transforms"]
    for key in ["R", "Z", "L", "w", "B"]:
        if key not in derivs:
            flags[key] = True
        elif key not in transforms:
            return False
        else:
            flags[key] = np.array(
                [d in transforms[key].derivatives.tolist() for d in derivs[key]]
            ).all()

    return all(flags.values())


def dot(a, b, axis=-1):
    """Batched vector dot product.

    Parameters
    ----------
    a : array-like
        First array of vectors.
    b : array-like
        Second array of vectors.
    axis : int
        Axis along which vectors are stored.

    Returns
    -------
    y : array-like
        y = sum(a*b, axis=axis)

    """
    return jnp.sum(a * b, axis=axis, keepdims=False)


def cross(a, b, axis=-1):
    """Batched vector cross product.

    Parameters
    ----------
    a : array-like
        First array of vectors.
    b : array-like
        Second array of vectors.
    axis : int
        Axis along which vectors are stored.

    Returns
    -------
    y : array-like
        y = a x b

    """
    return jnp.cross(a, b, axis=axis)


def _get_grid_surface(grid, surface_label):
    """Return grid quantities associated with the given surface label.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, theta, or zeta.

    Returns
    -------
    nodes : ndarray
        The column in the grid corresponding to this surface_label's nodes.
    unique_idx : ndarray
        The indices of the unique values of the surface_label in grid.nodes.
    ds : ndarray
        The differential elements (dtheta * dzeta for rho surface).
    max_surface_val : float
        The supremum of this surface_label.

    """
    assert surface_label in {"rho", "theta", "zeta"}
    if surface_label == "rho":
        nodes = grid.nodes[:, 0]
        unique_idx = grid.unique_rho_idx
        ds = grid.spacing[:, 1:].prod(axis=1)
        max_surface_val = 1
    elif surface_label == "theta":
        nodes = grid.nodes[:, 1]
        unique_idx = grid.unique_theta_idx
        ds = grid.spacing[:, [0, 2]].prod(axis=1)
        max_surface_val = 2 * jnp.pi
    else:
        nodes = grid.nodes[:, 2]
        unique_idx = grid.unique_zeta_idx
        ds = grid.spacing[:, :2].prod(axis=1)
        max_surface_val = 2 * jnp.pi

    return nodes, unique_idx, ds, max_surface_val


def compress(grid, x, surface_label="rho"):
    """Compress x by returning only the elements at unique surface_label indices.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        The array to compress.
        Should usually represent a surface function (a function constant over a surface)
        in an array that matches the grid's pattern.
    surface_label : str
        The surface label of rho, theta, or zeta.

    Returns
    -------
    compress_x : ndarray
        x[grid.unique_surface_label_indices]
        This array will be sorted such that the
            first element corresponds to the value associated with the smallest surface
            last element  corresponds to the value associated with the largest surface

    """
    assert surface_label in {"rho", "theta", "zeta"}
    assert len(x) == grid.num_nodes
    if surface_label == "rho":
        return x[grid.unique_rho_idx]
    if surface_label == "theta":
        return x[grid.unique_theta_idx]
    if surface_label == "zeta":
        return x[grid.unique_zeta_idx]


def expand(grid, x, surface_label="rho"):
    """Expand x by duplicating elements to match the grid's pattern.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        Stores the values of a surface function (a function constant over a surface)
        for all unique surfaces of the specified label on the grid.
        - len(x) should be grid.num_surface_label
        - x should be sorted such that the
            first element corresponds to the value associated with the smallest surface
            last element  corresponds to the value associated with the largest surface
    surface_label : str
        The surface label of rho, theta, or zeta.

    Returns
    -------
    expand_x : ndarray
        X expanded to match the grid's pattern.

    """
    assert surface_label in {"rho", "theta", "zeta"}
    if surface_label == "rho":
        assert len(x) == grid.num_rho
        return x[grid.inverse_rho_idx]
    if surface_label == "theta":
        assert len(x) == grid.num_theta
        return x[grid.inverse_theta_idx]
    if surface_label == "zeta":
        assert len(x) == grid.num_zeta
        return x[grid.inverse_zeta_idx]


def surface_integrals(grid, q=jnp.array([1]), surface_label="rho", max_surface=False):
    """Compute the surface integral of a quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
    max_surface : bool
        If True, only computes the surface integral on the flux surface with the
        maximum radial coordinate (as opposed to on all flux surfaces).

    Returns
    -------
    integrals : ndarray
        Surface integrals of q over each surface in grid.

    """
    if surface_label == "theta" and isinstance(grid, ConcentricGrid):
        warnings.warn(
            colored(
                "Integrals over constant theta surfaces are poorly defined for "
                + "ConcentricGrid.",
                "yellow",
            )
        )

    q = jnp.atleast_1d(q)
    nodes, unique_idx, ds, max_surface_val = _get_grid_surface(grid, surface_label)

    if max_surface:
        max_rho = grid.nodes[grid.unique_rho_idx[-1], 0]
        idx = np.nonzero(grid.nodes[:, 0] == max_rho)[0]
        q = q[idx]
        nodes = nodes[idx]
        unique_idx = (unique_idx / grid.num_rho).astype(int)
        ds = ds[idx] / grid.spacing[idx, 0] / max_rho

    # Separate nodes into bins with boundaries at unique values of the surface label.
    # This groups nodes with identical surface label values.
    # Each is assigned a weight of their contribution to the integral.
    # The elements of each bin are summed, performing the integration.
    bins = jnp.append(nodes[unique_idx], max_surface_val)
    integrals = jnp.histogram(nodes, bins=bins, weights=ds * q)[0]
    return expand(grid, integrals, surface_label)


def surface_averages(
    grid, q, sqrt_g=jnp.array([1]), surface_label="rho", denominator=None
):
    """Compute the surface average of a quantity for all surfaces in the grid.

    Notes
    -----
        Implements the flux-surface average formula given by equation 4.9.11 in
        W.D. D'haeseleer et al. (1991) doi:10.1007/978-3-642-75595-8.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to average.
    sqrt_g : ndarray
        Coordinate system Jacobian determinant; see data_index["sqrt(g)"].
    surface_label : str
        The surface label of rho, theta, or zeta to compute the average over.
    denominator : ndarray
        Volume enclosed by the surfaces, derivative wrt radial coordinate.
        Sign must match sqrt_g.
        This can be supplied to avoid redundant computations.

    Returns
    -------
    averages : ndarray
        Surface averages of q over each surface in grid.

    """
    q = jnp.atleast_1d(q)
    sqrt_g = jnp.atleast_1d(sqrt_g)

    if denominator is None:
        if sqrt_g.size == 1:
            denominator = (
                4 * jnp.pi**2 if surface_label == "rho" else 2 * jnp.pi
            ) * sqrt_g
        else:
            denominator = surface_integrals(grid, sqrt_g, surface_label)

    averages = surface_integrals(grid, sqrt_g * q, surface_label) / denominator
    return averages
