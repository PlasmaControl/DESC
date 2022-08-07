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
        The differential elements (dtheta * dzeta for rho surface)
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
        raise NotImplementedError("theta not implemented yet")
    elif surface_label == "zeta":
        surface_label_nodes = grid.nodes[:, 2]
        unique_indices = grid.unique_zeta_indices
        upper_bound = 2 * jnp.pi
        ds = grid.weights if grid.num_zeta == 1 else grid.spacing[:, :2].prod(axis=1)
    else:
        raise ValueError("Surface label must be 'rho', 'theta', or 'zeta'.")

    return surface_label_nodes, unique_indices, upper_bound, ds


def compress(grid, x, surface_label="rho"):
    """
    Compress the array x by returning only the elements in x at unique surface_label indices.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        The array to compress.
    surface_label : str
        The surface label of rho, theta, or zeta.

    Returns
    -------
    ndarray
        x[grid.unique_surface_label_indices].
    """
    if surface_label == "rho":
        return x[grid.unique_rho_indices]
    if surface_label == "theta":
        return x[grid.unique_theta_indices]
    if surface_label == "zeta":
        return x[grid.unique_zeta_indices]


def expand(grid, x, surface_label="rho"):
    """
    Expand the array x by duplicating elements in x to match the grid's pattern.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        Stores the values of some surface function - (a function whose output is constant over a surface) -
        for all unique surfaces of the specified label on the grid.
        len(x) should be grid.num_surface_label. x should be sorted such that x[0] corresponds to the
        value associated with the smallest surface value on the grid and x[-1] the largest.
    surface_label : str
        The surface label of rho, theta, or zeta.

    Returns
    -------
    ndarray
        An array that matches the grid pattern.
    """

    # TODO: confirm this is the standard way to get whether user has jax
    #   and that desc.backend.jnp is an alias for numpy when user doesn't have jax
    no_jax = os.environ.get("DESC_BACKEND") == "numpy"
    number_nodes_zeta_surface = len(grid.nodes) // grid.num_zeta

    if surface_label == "rho":
        assert len(x) == grid.num_rho
        # First duplicate each x[i] over every theta node of the same rho surface.
        # The difference between unique rho indices is the number of theta nodes at each rho surface.
        # Next, tile the result to repeat the pattern on a single zeta surface over all zeta surfaces.
        theta_node_counts = jnp.diff(
            grid.unique_rho_indices, append=number_nodes_zeta_surface
        )
        zeta_surface = (
            jnp.repeat(x, repeats=theta_node_counts)
            if no_jax
            else jnp.repeat(
                x,
                repeats=theta_node_counts,
                total_repeat_length=number_nodes_zeta_surface,
            )
        )
        return jnp.tile(zeta_surface, reps=grid.num_zeta)

    if surface_label == "theta":
        assert len(x) == grid.num_theta
        raise NotImplementedError("theta not implemented yet")

    if surface_label == "zeta":
        assert len(x) == grid.num_zeta
        if no_jax:
            return jnp.repeat(x, repeats=number_nodes_zeta_surface)
        return jnp.repeat(
            x, repeats=number_nodes_zeta_surface, total_repeat_length=len(grid.nodes)
        )

    raise ValueError("Surface label must be 'rho', 'theta', or 'zeta'.")


def surface_integrals(grid, integrands=1, surface_label="rho", match_grid=False):
    """
    Bulk surface integral function.
    Computes the surface integral of the specified quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    integrands : ndarray
        Quantity to integrate.
        Should not include the differential elements (dtheta * dzeta for rho surface).
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
        Defaults to the flux surface label rho.
    match_grid : bool
        False to return a compressed array which assigns every surface integral to a single element.
        This array will have length = number of unique surfaces in the grid.
        True to return an expanded array which assigns every surface integral to all indices in
        grid.nodes which are associated with that surface.
        This array will match the grid's pattern and have length = len(grid.nodes).

    Returns
    -------
    integrals : ndarray
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
    return expand(grid, integrals, surface_label) if match_grid else integrals


def surface_averages(
    grid, q, sqrtg=1, surface_label="rho", match_grid=False, denominator=None
):
    """
    Bulk surface average function.
    Computes the surface average of the specified quantity for all surfaces in the grid.
    See D'haeseleer flux coordinates eq. 4.9.11.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
        Due to the symmetry / NFP bugs, the grid should temporarily be limited to pass this assertion:
            assert (grid.sym is False) and (grid.NFP == 1 or grid.num_surface_label == 1)
    q : ndarray
        Quantity to average.
    sqrtg : ndarray
        Magnitude of the 3d jacobian determinant, data["sqrt(g)"]. Defaults to 1.
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
        The denominator in the surface average is independent of the quantity q.
        To avoid recomputing it, some users may prefer to cache and supply it.
        When the sqrt(g) factor is included in the average, the denominator is d(volume)/d(surface label).
        When the sqrt(g) factor is not included, the denominator is the unweighted surface area.
        See D'haeseleer flux coordinates eq. 4.9.11.

    Returns
    -------
    averages : ndarray
        Surface averages of the given quantity, q, over each surface in grid.
    """
    if denominator is None:
        denominator = surface_integrals(grid, sqrtg, surface_label)
    averages = surface_integrals(grid, sqrtg * q, surface_label) / denominator
    return expand(grid, averages, surface_label) if match_grid else averages


def enclosed_volumes(grid, data, dr=0, match_grid=False):
    """
    Returns enclosed positive volume derivatives wrt rho of each rho surface in grid.

    Parameters
    ----------
    grid : Grid, LinearGrid, ConcentricGrid, QuadratureGrid
        Collocation grid containing the nodes to evaluate at.
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of covariant basis vectors and toroidal coords.
        Keys are of the form 'X_y' meaning the derivative of X wrt y.
    dr : int
        Enclosed volume derivative order to return.
    match_grid : bool
        False to return an array which assigns every surface integral to a single element of the array.
        This array will have length = number of unique surfaces in the grid.
        True to return an array which assigns every surface integral to all indices in grid.nodes which
        are associated with that surface.
        This array will match the grid's pattern and have length = len(grid.nodes).

    Returns
    -------
    ndarray
        Derivative wrt rho of specified order of volume enclosed by flux surface.
    """
    if dr == 0:
        from desc.compute import cross

        # data["V"] is the total volume, not the volume enclosed by the flux surface.
        # enclosed volume is computed using divergence theorem:
        # volume integral(div [0, 0, Z] = 1) = surface integral([0, 0, Z] dot ds)
        sqrtg_e_sup_rho = cross(data["e_theta"], data["e_zeta"])
        V = jnp.abs(surface_integrals(grid, sqrtg_e_sup_rho[:, 2] * data["Z"]))
        return expand(grid, V) if match_grid else V
    if dr == 1:
        # See D'haeseleer flux coordinates eq. 4.9.10 for dv/d(flux label).
        # Intuitively, this formula makes sense because
        # V = integral(dr dt dz * sqrt(g)) and differentiating wrt rho removes
        # the dr integral. What remains is a surface integral with a 3D jacobian.
        # This is the volume added by a thin shell of constant rho.
        dv_drho = surface_integrals(grid, jnp.abs(data["sqrt(g)"]))
        return expand(grid, dv_drho) if match_grid else dv_drho
    if dr == 2:
        d2v_drho2 = surface_integrals(grid, jnp.abs(data["sqrt(g)_r"]))
        return expand(grid, d2v_drho2) if match_grid else d2v_drho2
