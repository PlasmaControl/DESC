import warnings

from desc.backend import jnp
from desc.grid import LinearGrid


def _get_proper_surface(grid, surface_label):
    """Returns grid quantities associated with the given surface label.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, theta, or zeta.

    Returns
    -------
    :rtype: (ndarray, ndarray, ndarray)
    surface_label_nodes : ndarray
        The column in the grid corresponding to this surface_label's nodes.
    unique_idx : ndarray
        The indices of the unique values of the surface_label in grid.nodes.
    ds : ndarray
        The differential elements (dtheta * dzeta for rho surface).
    """
    assert surface_label in {"rho", "theta", "zeta"}
    if surface_label == "rho":
        surface_label_nodes = grid.nodes[:, 0]
        unique_idx = grid.unique_rho_idx
        ds = grid.spacing[:, 1:].prod(axis=1)
    elif surface_label == "theta":
        surface_label_nodes = grid.nodes[:, 1]
        unique_idx = grid.unique_theta_idx
        ds = grid.spacing[:, [0, 2]].prod(axis=1)
    else:
        surface_label_nodes = grid.nodes[:, 2]
        unique_idx = grid.unique_zeta_idx
        ds = grid.spacing[:, :2].prod(axis=1)

    return surface_label_nodes, unique_idx, ds


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
    ndarray
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
    ndarray
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
    """Bulk surface integral function.

    Computes the surface integral of the specified quantity for all surfaces in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
        Note for surface_label="theta" LinearGrid will have better accuracy than QuadratureGrid.
    match_grid : bool
        False to return a compressed array which assigns every surface integral to a single element.
        This array will have length = number of unique surfaces in the grid.
        True to return an expanded array which assigns every surface integral to all indices in
        grid.nodes which are associated with that surface.
        This array will match the grid's pattern and have length = grid.num_nodes.

    Returns
    -------
    integrals : ndarray
        Surface integrals of q over each surface in grid.
    """
    if surface_label == "theta" and not isinstance(grid, LinearGrid):
        warnings.warn(
            "Nonlinear grids may have bad accuracy for theta surface computations.",
            RuntimeWarning,
        )
    surface_label_nodes, unique_idx, ds = _get_proper_surface(grid, surface_label)

    # DESIRED ALGORITHM
    # surfaces = dict()
    # collect collocation node indices for each surface_label surface
    # for index_in_grid_column, surface_label_value in enumerate(surface_label_nodes):
    #     surfaces.setdefault(surface_label_value, list()).append(index_in_grid_column)
    # integration over non-contiguous elements
    # for i, e in enumerate(sorted(surfaces.items())):
    #     _, surface_indices = e
    #     integrals[i] = (ds * q)[surface_indices].sum()

    # NO LOOP IMPLEMENTATION
    # Separate collocation nodes into bins with boundaries at unique values of the surface label.
    # This groups nodes with identical surface label values.
    # Each is assigned a weight of their contribution to the integral.
    # The elements of each bin are summed, performing the integration.

    surface_label_supremum = 7  # anything >= 1 for rho or 2pi for theta/zeta works
    bins = jnp.append(surface_label_nodes[unique_idx], surface_label_supremum)
    integrals = jnp.histogram(surface_label_nodes, bins, weights=ds * q)[0]
    return expand(grid, integrals, surface_label) if match_grid else integrals


def surface_averages(
    grid, q, sqrtg=1, surface_label="rho", match_grid=False, denominator=None
):
    """Bulk surface average function.

    Computes the surface average of the specified quantity for all surfaces in the grid.
    See D'haeseleer flux coordinates eq. 4.9.11.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to average.
    sqrtg : ndarray
        Magnitude of the 3D jacobian determinant, data["sqrt(g)"].
    surface_label : str
        The surface label of rho, theta, or zeta to compute integration over.
        Note for surface_label="theta" LinearGrid will have better accuracy than QuadratureGrid.
    match_grid : bool
        False to return an array which assigns every surface integral to a single element of the array.
        This array will have length = number of unique surfaces in the grid.
        True to return an array which assigns every surface integral to all indices in grid.nodes which
        are associated with that surface.
        This array will match the grid's pattern and have length = grid.num_nodes.
    denominator : ndarray
        The denominator in the surface average is independent of the quantity q.
        To avoid recomputing it for surface averages that include the sqrt(g) factor,
        some users may prefer to cache and supply the denominator.
        See D'haeseleer flux coordinates eq. 4.9.11.

    Returns
    -------
    averages : ndarray
        Surface averages of the given quantity, q, over each surface in grid.
    """
    if denominator is None:
        if isinstance(sqrtg, int) and sqrtg == 1:
            # shortcut to avoid unnecessary computation
            denominator = 4 * jnp.pi ** 2 if surface_label == "rho" else 2 * jnp.pi
        else:
            denominator = surface_integrals(grid, sqrtg, surface_label)
    averages = surface_integrals(grid, sqrtg * q, surface_label) / denominator
    return expand(grid, averages, surface_label) if match_grid else averages


def enclosed_volumes(grid, data, dr=0, match_grid=False):
    """Derivatives wrt rho of the positive volume enclosed by each rho surface in grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    data : dict
        Dictionary of ndarray, shape(num_nodes,) of covariant basis vectors and toroidal coords.
            - dr = 0 requires 'e_theta', 'e_zeta', and 'Z'
            - dr = 1 requires 'sqrt(g)'
            - dr = 2 requires 'sqrt(g)_r'
    dr : int
        Derivative order.
    match_grid : bool
        False to return an array which assigns every surface integral to a single element of the array.
        This array will have length = number of unique surfaces in the grid.
        True to return an array which assigns every surface integral to all indices in grid.nodes which
        are associated with that surface.
        This array will match the grid's pattern and have length = grid.num_nodes.

    Returns
    -------
    ndarray
        Derivative wrt rho of specified order of positive volume enclosed by flux surface.
    """
    if dr == 0:
        # data["V"] is the total volume, not the volume enclosed by the flux surface.
        # enclosed volume is computed using divergence theorem:
        # volume integral(div [0, 0, Z]) = surface integral(ds dot [0, 0, Z])
        sqrtg_times_e_sup_rho = cross(data["e_theta"], data["e_zeta"])
        V = jnp.abs(surface_integrals(grid, sqrtg_times_e_sup_rho[:, 2] * data["Z"]))
        return expand(grid, V) if match_grid else V
    if dr == 1:
        # See D'haeseleer flux coordinates eq. 4.9.10 for dv/d(flux label).
        # Intuitively, this formula makes sense because
        # V = integral(dr dt dz * sqrt(g)) and differentiating wrt rho removes the dr integral.
        # What remains is a surface integral with a 3D jacobian.
        # This is the volume added by a thin shell of constant rho.
        dv_drho = surface_integrals(grid, jnp.abs(data["sqrt(g)"]))
        return expand(grid, dv_drho) if match_grid else dv_drho
    if dr == 2:
        d2v_drho2 = surface_integrals(grid, jnp.abs(data["sqrt(g)_r"]))
        return expand(grid, d2v_drho2) if match_grid else d2v_drho2


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
