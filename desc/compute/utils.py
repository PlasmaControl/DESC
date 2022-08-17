import numpy as np

from desc.backend import jnp
from .data_index import data_index


def check_derivs(key, R_transform=None, Z_transform=None, L_transform=None):
    """Check if Transforms can compute required derivatives of R, Z, lambda.

    Parameters
    ----------
    key : str
        Key indicating a quantity from data_index.
    R_transform : Transform, optional
        Transforms R_lmn coefficients to real space.
    Z_transform : Transform, optional
        Transforms Z_lmn coefficients to real space.
    L_transform : Transform, optional
        Transforms L_lmn coefficients to real space.

    Returns
    -------
    flag : bool
        True if the Transforms can compute requested derivatives, False otherwise.

    """
    if "R_derivs" not in data_index[key]:
        R_flag = True
        Z_flag = True
    else:
        R_flag = np.array(
            [d in R_transform.derivatives.tolist() for d in data_index[key]["R_derivs"]]
        ).all()
        Z_flag = np.array(
            [d in Z_transform.derivatives.tolist() for d in data_index[key]["R_derivs"]]
        ).all()

    if "L_derivs" not in data_index[key]:
        L_flag = True
    else:
        L_flag = np.array(
            [d in L_transform.derivatives.tolist() for d in data_index[key]["L_derivs"]]
        ).all()

    return R_flag and Z_flag and L_flag


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

    """
    assert surface_label in {"rho", "theta", "zeta"}
    if surface_label == "rho":
        nodes = grid.nodes[:, 0]
        unique_idx = grid.unique_rho_idx
        ds = grid.spacing[:, 1:].prod(axis=1)
    elif surface_label == "theta":
        nodes = grid.nodes[:, 1]
        unique_idx = grid.unique_theta_idx
        ds = grid.spacing[:, [0, 2]].prod(axis=1)
    else:
        nodes = grid.nodes[:, 2]
        unique_idx = grid.unique_zeta_idx
        ds = grid.spacing[:, :2].prod(axis=1)

    return nodes, unique_idx, ds


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


def surface_integrals(grid, q=1, surface_label="rho", match_grid=False):
    """Compute the surface integral of a quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
    match_grid : bool
        Whether to expand the result to match the dimension of the grid.
        If False (default), the result is a single value for each surface in the grid.
        If True, the result has repeated values to match the number of grid nodes.

    Returns
    -------
    integrals : ndarray
        Surface integrals of q over each surface in grid.

    """
    nodes, unique_idx, ds = _get_grid_surface(grid, surface_label)
    max_surface_val = 1 if surface_label == "rho" else 2 * jnp.pi
    bins = jnp.append(nodes[unique_idx], max_surface_val)
    integrals = jnp.histogram(nodes, bins=bins, weights=ds * q)[0]

    return expand(grid, integrals, surface_label) if match_grid else integrals


def surface_averages(
    grid,
    q,
    sqrt_g=jnp.array([1]),
    surface_label="rho",
    match_grid=False,
    denominator=None,
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
    match_grid : bool
        Whether to expand the result to match the dimension of the grid.
        If False (default), the result is a single value for each surface in the grid.
        If True, the result has repeated values to match the number of grid nodes.
    denominator : ndarray
        Volume over the other two coordinates besides surface_label.
        This can be supplied to avoid redundant computations.

    Returns
    -------
    averages : ndarray
        Surface averages of q over each surface in grid.

    """
    if denominator is None:
        if sqrt_g.size == 1:
            denominator = (
                4 * jnp.pi ** 2 if surface_label == "rho" else 2 * jnp.pi
            ) * sqrt_g
        else:
            denominator = surface_integrals(grid, sqrt_g, surface_label)
    averages = surface_integrals(grid, sqrt_g * q, surface_label) / denominator
    return expand(grid, averages, surface_label) if match_grid else averages
