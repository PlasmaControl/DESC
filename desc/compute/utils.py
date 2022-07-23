import numpy as np
import os

from desc.backend import jnp


def _get_proper_surface(grid, surface_label):
    """
    Returns grid quantities associated with the given surface label.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.

    Returns
    -------
    :rtype: (ndarray, ndarray, float, ndarray)
    surface_label_nodes : ndarray
        The column in the grid corresponding to this surface_label's nodes.
    unique_indices : ndarray
        The indices of the unique values of the surface_label in grid.nodes.
    upper_bound : float
        The supremum of the set of all values of the surface_label.
    ds : ndarray
        The surface differential element (dtheta * dzeta for rho surface)
    """

    if surface_label == "rho":
        surface_label_nodes = grid.nodes[:, 0]
        unique_indices = grid.unique_rho_indices
        upper_bound = 1
        # NFP bug, grid.weights is the value we expect from dtheta * dzeta if rho is constant.
        ds = grid.weights if grid.num_rho == 1 else grid.spacing[:, 1:].prod(axis=1)
    elif surface_label == "theta":
        surface_label_nodes = grid.nodes[:, 1]
        unique_indices = grid.unique_theta_indices
        upper_bound = 2 * jnp.pi
        ds = (
            grid.weights
            if grid.num_theta == 1
            else grid.spacing[:, [0, 2]].prod(axis=1)
        )
        raise ValueError("Not implemented yet.")
    elif surface_label == "zeta":
        surface_label_nodes = grid.nodes[:, 2]
        unique_indices = grid.unique_zeta_indices
        upper_bound = 2 * jnp.pi
        ds = grid.weights if grid.num_zeta == 1 else grid.spacing[:, :2].prod(axis=1)
    else:
        raise ValueError("Surface label must be 'rho', 'theta', or 'zeta'.")

    return surface_label_nodes, unique_indices, upper_bound, ds


def _expand(grid, x, surface_label="rho"):
    """
    Expand the array x by duplicating elements in x to match the grid's pattern.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        Stores the values of some surface function - (a function whose output is constant over a surface) -
        for all unique surfaces of the specified label on the grid.
        len(x) should be grid.num_ of the surface label. x should be sorted such that x[0] corresponds to the
        value associated with the smallest surface value on the grid and x[-1] the largest.
    surface_label : str
        The surface label of rho, theta, or zeta.

    Returns
    -------
    ndarray
        An array that matches the grid pattern.
    """

    # assert len(x) == grid.num_ of the surface_label
    # TODO: confirm this is standard way to get whether user has jax
    no_jax = os.environ.get("DESC_BACKEND") == "numpy"
    number_nodes_zeta_surface = len(grid.nodes) // grid.num_zeta

    if surface_label == "rho":
        # first repeat x[i], x at each rho surface, to become x at each theta node at each rho surface
        # the difference between unique indices, when sorted, is the number of theta nodes to repeat for each rho value
        # then tile x to repeat its pattern on a single zeta surface over all zeta surfaces.
        theta_node_repeats = jnp.diff(
            grid.unique_rho_indices, append=number_nodes_zeta_surface
        )
        if no_jax:
            return np.tile(np.repeat(x, repeats=theta_node_repeats), reps=grid.num_zeta)
        return jnp.tile(
            jnp.repeat(
                x,
                repeats=theta_node_repeats,
                total_repeat_length=number_nodes_zeta_surface,
            ),
            reps=grid.num_zeta,
        )

    if surface_label == "theta":
        raise ValueError("Not implemented yet.")

    if surface_label == "zeta":
        if no_jax:
            return np.repeat(x, repeats=number_nodes_zeta_surface)
        return jnp.repeat(
            x, repeats=number_nodes_zeta_surface, total_repeat_length=len(grid.nodes)
        )

    raise ValueError("Surface label must be 'rho', 'theta', or 'zeta'.")


def surface_integrals(grid, integrands, surface_label="rho", match_grid=False):
    """
    Bulk surface integral function.
    Computes the surface integral of the specified quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    integrands : ndarray
        Quantity to integrate.
        Should not include the surface differential element ds (dtheta * dzeta for rho surface).
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
        Defaults to the flux surface label rho.
    match_grid : bool
        False to return an array which assigns every surface integral to a single element of the array.
        This array will have length = number of unique surfaces in the grid.
        True to return an array which assigns every surface integral to all indices in grid.nodes which
        are associated with that surface.
        This array will match the grid's pattern and have length = len(grid.nodes).

    Returns
    -------
    ndarray
        Surface integrals of integrand over each surface in grid.
    """

    surface_label_nodes, unique_indices, upper_bound, ds = _get_proper_surface(
        grid, surface_label
    )

    # DESIRED ALGORITHM
    # surfaces = dict()
    # collect collocation node indices for each surface_label surface
    # for index_in_grid_column, surface_label_value in enumerate(surface_label_nodes):
    #     surfaces.setdefault(surface_label_value, list()).append(index_in_grid_column)
    # integration over non-contiguous elements
    # for i, surface_indices in enumerate(surfaces.values()):
    #     integrals[i] = (ds * integrands)[surface_indices].sum()

    # NO LOOP IMPLEMENTATION
    # Separate collocation nodes into bins with boundaries at unique values of the surface label.
    # This groups nodes with identical surface label values.
    # Each is assigned a weight of their contribution to the integral.
    # The elements of each bin are summed, performing the integration.
    bins = jnp.append(surface_label_nodes[unique_indices], upper_bound)
    # assert bins is sorted. satisfied by grid being sorted at the time the unique indices are stored.
    integrals = jnp.histogram(surface_label_nodes, bins, weights=ds * integrands)[0]
    return _expand(grid, integrals, surface_label) if match_grid else integrals


def surface_averages(
    grid, q, sqrtg, surface_label="rho", match_grid=False, denominator=None
):
    """
    Bulk surface average function.
    Computes the surface average of the specified quantity for all surfaces in the grid.
    See D'haeseleer flux coordinates eq. 4.9.11.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to average.
    sqrtg : ndarray
        Magnitude of the 3d jacobian determinant, data["sqrt(g)"].
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
        Defaults to the flux surface label rho.
    match_grid : bool
        False to return an array which assigns every surface integral to a single element of the array.
        This array will have length = number of unique surfaces in the grid.
        True to return an array which assigns every surface integral to all indices in grid.nodes which
        are associated with that surface.
        This array will match the grid's pattern and have length = len(grid.nodes).
    denominator : ndarray
        The denominator of a surface average does not depend on the quantity q.
        Multiple calls to surface_averages() on the same surface label will recompute this quantity.
        Some users may prefer to cache the denominator externally and supply it to avoid this.

    Returns
    -------
    ndarray
        Surface averages of the given quantity, q, over each surface in grid.
    """
    numerator = surface_integrals(grid, sqrtg * q, surface_label)
    if denominator is None:
        denominator = surface_integrals(grid, sqrtg, surface_label)
    averages = numerator / denominator
    return _expand(grid, averages, surface_label) if match_grid else averages
