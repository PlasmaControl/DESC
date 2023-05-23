"""Functions for flux surface averages and vector algebra operations."""

import copy
import warnings

import numpy as np
from termcolor import colored

from desc.backend import fori_loop, jnp, put
from desc.grid import ConcentricGrid, LinearGrid

from .data_index import data_index

# defines the order in which objective arguments get concatenated into the state vector
arg_order = (
    "R_lmn",
    "Z_lmn",
    "L_lmn",
    "p_l",
    "i_l",
    "c_l",
    "Psi",
    "Te_l",
    "ne_l",
    "Ti_l",
    "Zeff_l",
    "Rb_lmn",
    "Zb_lmn",
)
# map from profile name to equilibrium parameter name
profile_names = {
    "pressure": "p_l",
    "iota": "i_l",
    "current": "c_l",
    "electron_temperature": "Te_l",
    "electron_density": "ne_l",
    "ion_temperature": "Ti_l",
    "atomic_number": "Zeff_l",
}


def _sort_args(args):
    return [arg for arg in arg_order if arg in args]


def compute(names, params, transforms, profiles, data=None, **kwargs):
    """Compute the quantity given by name on grid.

    Parameters
    ----------
    names : str or array-like of str
        Name(s) of the quantity(s) to compute.
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
    unique_size : ndarray
        The number of the unique values of the surface_label.
    inverse_idx : ndarray
        Indexing array to go from unique values to full grid.
    spacing : ndarray
        The relevant columns of grid.spacing.
    has_endpoint_dupe : bool
        Whether this surface label's nodes have a duplicate at the endpoint
        of a periodic domain. (e.g. a node at 0 and 2π).

    """
    assert surface_label in {"rho", "theta", "zeta"}
    if surface_label == "rho":
        unique_size = grid.num_rho
        inverse_idx = grid.inverse_rho_idx
        spacing = grid.spacing[:, 1:]
        has_endpoint_dupe = False
    elif surface_label == "theta":
        unique_size = grid.num_theta
        inverse_idx = grid.inverse_theta_idx
        spacing = grid.spacing[:, [0, 2]]
        has_endpoint_dupe = (
            grid.nodes[grid.unique_theta_idx[0], 1] == 0
            and grid.nodes[grid.unique_theta_idx[-1], 1] == 2 * np.pi
        )
    else:
        unique_size = grid.num_zeta
        inverse_idx = grid.inverse_zeta_idx
        spacing = grid.spacing[:, :2]
        has_endpoint_dupe = (
            grid.nodes[grid.unique_zeta_idx[0], 2] == 0
            and grid.nodes[grid.unique_zeta_idx[-1], 2] == 2 * np.pi / grid.NFP
        )
    return unique_size, inverse_idx, spacing, has_endpoint_dupe


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


def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    """Cumulatively integrate y(x) using the composite trapezoidal rule.

    Taken from SciPy, but changed NumPy references to JAX.NumPy:
        https://github.com/scipy/scipy/blob/v1.10.1/scipy/integrate/_quadrature.py

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        Typically, this value should be 0. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    """
    y = jnp.asarray(y)
    if x is None:
        d = dx
    else:
        x = jnp.asarray(x)
        if x.ndim == 1:
            d = jnp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the " "same as y.")
        else:
            d = jnp.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError(
                "If given, length of x along axis must be the " "same as y."
            )

    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = jnp.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if not jnp.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = jnp.concatenate(
            [jnp.full(shape, initial, dtype=res.dtype), res], axis=axis
        )

    return res


def line_integrals(
    grid, q=jnp.array([1.0]), line_label="theta", fix_surface=("rho", 1.0)
):
    """Compute the line integral of a quantity over curves covering fix_surface.

    As an example, by specifying the combination of line_label="theta" and
    fix_surface=("rho", 1.0), the intention is to integrate along the outermost
    perimeter of a particular zeta surface (toroidal cross-section), for each
    zeta surface in the grid.

    Notes
    -----
        It is assumed that the integration curve has length 1 when the line
        label is rho and length 2π when the line label is theta or zeta.
        You may want to multiply q by the line length Jacobian.

        Correctness is not guaranteed on grids with duplicate nodes.
        An attempt to print a warning is made if the given grid has duplicate
        nodes and is one of the predefined grid types
        (Linear, Concentric, Quadrature).
        If the grid is custom, no attempt is made to warn.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
    line_label : str
        The coordinate curve of rho, theta, or zeta to compute the integration over.
        To clarify, a theta (poloidal) curve is the intersection of a
        rho (flux) surface and zeta (toroidal) surface.
    fix_surface : str, float
        A tuple of the form: label, value.
        fix_surface label should differ from line_label.
        By default, fix_surface is chosen to be the flux surface at rho=1.

    Returns
    -------
    integrals : ndarray
        Line integrals of q over curves covering the given surface.

    """
    assert (
        line_label != fix_surface[0]
    ), "There is no valid use for this combination of inputs."
    assert line_label == "theta" or not isinstance(
        grid, ConcentricGrid
    ), "ConcentricGrid should only be used for theta line integrals."
    if isinstance(grid, LinearGrid) and grid.endpoint:
        warnings.warn(
            colored(
                "Correctness not guaranteed on grids with duplicate nodes.", "yellow"
            )
        )
    # Generate a new quantity q_prime which is zero everywhere
    # except on the fixed surface, on which q_prime takes the value of q.
    # Then forward the computation to surface_integrals.
    # The differential element of the line integral, denoted dl,
    # should correspond to the line label's spacing.
    # The differential element of the surface integral is
    # ds = dl * fix_surface_dl, so we scale q_prime by 1 / fix_surface_dl.
    labels = {"rho": 0, "theta": 1, "zeta": 2}
    column_id = labels[fix_surface[0]]
    mask = grid.nodes[:, column_id] == fix_surface[1]
    q_prime = (mask * jnp.atleast_1d(q).T / grid.spacing[:, column_id]).T
    (surface_label,) = labels.keys() - {line_label, fix_surface[0]}
    return surface_integrals(grid, q_prime, surface_label)


def surface_integrals(grid, q=jnp.array([1.0]), surface_label="rho"):
    """Compute the surface integral of a quantity for all surfaces in the grid.

    Notes
    -----
        It is assumed that the integration surface has area 4π^2 when the
        surface label is rho and area 2π when the surface label is theta or
        zeta. You may want to multiply q by the surface area Jacobian.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
    surface_label : str
        The surface label of rho, theta, or zeta to compute the integration over.

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
    unique_size, inverse_idx, spacing, has_endpoint_dupe = _get_grid_surface(
        grid, surface_label
    )

    # Todo: define masks as a sparse matrix?
    # The ith row of masks is True only at the indices which correspond to the
    # ith surface. The integral over the ith surface is the dot product of the
    # ith row vector and the vector of integrands of all surfaces.
    masks = inverse_idx == jnp.arange(unique_size)[:, jnp.newaxis]
    if has_endpoint_dupe:
        masks = put(masks, jnp.asarray([0, -1]), masks[0] | masks[-1])
        # Imagine a torus cross-section at zeta=π.
        # A grid with a duplicate zeta=π node has 2 of those cross-sections.
        #     In grid.py, we multiply by 1/n the areas of surfaces with
        # duplicity n. This prevents the area of that surface from being
        # double-counted, as surfaces with the same node value are combined
        # into 1 integral, which sums their areas. Thus, if the zeta=π
        # cross-section has duplicity 2, we ensure that the area on the zeta=π
        # surface will have the correct total area of π+π = 2π.
        #     An edge case exists if the duplicate surface has nodes with
        # different values for the surface label, which only occurs when
        # has_endpoint_dupe is true. If has_endpoint_dupe is true, this grid
        # has a duplicate surface at surface_label=0 and
        # surface_label=max surface value. Although the modulo of these values
        # are equal, their numeric values are not, so the integration
        # would treat them as different surfaces. We solve this issue by
        # combining the indices corresponding to the integrands of the duplicated
        # surface, so that the duplicate surface is treated as one, like in the
        # previous paragraph.

    integrands = (spacing.prod(axis=1) * jnp.nan_to_num(q).T).T
    assert integrands.ndim <= 3  # otherwise replace @ with jnp.tensordot.
    # `integrands` (and `q`) has shape (g.size, f.size, v.size), where
    #     g is the grid function depending on the integration variables
    #     f is a function which may be independent of the integration variables
    #     v is the vector of components of f.
    # The intention is to integrate `integrands` which is a
    #     vector-valued            (with v.size components)
    #     function-valued          (with image size of f.size)
    #     function over the grid   (with domain size of g.size = grid.num_nodes)
    # over each surface in the grid.
    #
    # The distinction between f and v is semantic.
    # We may alternatively consider an `integrands` of shape (g.size, f.size) to
    # represent a vector-valued (with f.size components) function over the grid.
    # Likewise, we may alternatively consider an `integrands` of shape
    # (g.size, v.size) to represent a function-valued (with image size v.size)
    # function over the grid. When `integrands` has dimension one, it is a
    # scalar function over the grid. That is, a
    #     vector-valued            (with 1 component),
    #     function-valued          (with image size of 1)
    #     function over the grid   (with domain size of g.size = grid.num_nodes)
    #
    # The integration is performed by applying `masks`, the surface
    # integral operator, to `integrands`. This operator hits the matrix formed
    # by the last two dimensions of `integrands`, for every element along the
    # previous dimension of `integrands`. Therefore, when `integrands` has three
    # dimensions, the second must hold g. We may choose which of the first and
    # third dimensions hold f and v. The choice below transposes `integrands` to
    # shape (v.size, g.size, f.size). As we expect f.size >> v.size, the
    # integration is potentially faster since numpy likely optimizes large
    # matrix products.
    axis_to_move = (integrands.ndim == 3) * 2  # interpolates (3, 2, 1) ↦ (2, 0, 0)
    integrals = jnp.moveaxis(
        masks @ jnp.moveaxis(integrands, axis_to_move, 0), 0, axis_to_move
    )
    return expand(grid, integrals, surface_label)


def surface_averages(
    grid, q, sqrt_g=jnp.array([1.0]), surface_label="rho", denominator=None
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
        Volume enclosed by the surfaces, derivative wrt the surface label.
        This can optionally be supplied to avoid redundant computations.

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

    averages = (
        surface_integrals(grid, (sqrt_g * q.T).T, surface_label).T / denominator
    ).T
    return averages


def surface_max(grid, x, surface_label="rho"):
    """Get the max of x for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        Quantity to find max.
    surface_label : str
        The surface label of rho, theta, or zeta to compute max over.

    Returns
    -------
    maxs : ndarray
        Maximum of x over each surface in grid.

    """
    return -surface_min(grid, -x, surface_label)


def surface_min(grid, x, surface_label="rho"):
    """Get the min of x for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        Quantity to find min.
    surface_label : str
        The surface label of rho, theta, or zeta to compute min over.

    Returns
    -------
    mins : ndarray
        Minimum of x over each surface in grid.

    """
    unique_size, inverse_idx, _, _ = _get_grid_surface(grid, surface_label)
    inverse_idx = jnp.asarray(inverse_idx)
    x = jnp.asarray(x)
    mins = jnp.full(unique_size, jnp.inf)

    def body(i, mins):
        mins = put(mins, inverse_idx[i], jnp.minimum(x[i], mins[inverse_idx[i]]))
        return mins

    mins = fori_loop(0, inverse_idx.size, body, mins)
    # The above implementation was benchmarked to be more efficient, after jit
    # compilation, than the alternative given in the two lines below.
    # masks = inverse_idx == jnp.arange(unique_size)[:, jnp.newaxis]  # noqa: E501,E800
    # mins = jnp.amin(x[jnp.newaxis, :], axis=1, initial=jnp.inf, where=masks)  # noqa: E501,E800
    return expand(grid, mins, surface_label)
