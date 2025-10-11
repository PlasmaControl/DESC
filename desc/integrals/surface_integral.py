"""Surface integrals of non-singular functions.

If you would like to view a detailed tutorial for use of these functions, see
https://desc-docs.readthedocs.io/en/latest/notebooks/dev_guide/grid.html.
"""

from desc.backend import cond, fori_loop, jnp, put
from desc.grid import ConcentricGrid, LinearGrid
from desc.utils import errorif, warnif

# TODO (#1389): Make the surface integral stuff objects with a callable method instead
#       of returning functions. Would make simpler, allow users to actually see the
#       docstrings of the methods, and less bookkeeping to default to more
#       efficient methods on tensor product grids.


def _get_grid_surface(grid, surface_label):
    """Return grid quantities associated with the given surface label.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, poloidal, or zeta.

    Returns
    -------
    unique_size : int
        The number of the unique values of the surface_label.
    inverse_idx : ndarray
        Indexing array to go from unique values to full grid.
    spacing : ndarray
        The relevant columns of grid.spacing.
    has_endpoint_dupe : bool
        Whether this surface label's nodes have a duplicate at the endpoint
        of a periodic domain. (e.g. a node at 0 and 2π).
    has_idx : bool
        Whether the grid knows the number of unique nodes and inverse idx.

    """
    assert surface_label in {"rho", "poloidal", "zeta"}
    if surface_label == "rho":
        spacing = grid.spacing[:, 1:]
        has_endpoint_dupe = False
    elif surface_label == "poloidal":
        spacing = grid.spacing[:, [0, 2]]
        has_endpoint_dupe = isinstance(grid, LinearGrid) and grid._poloidal_endpoint
    else:
        spacing = grid.spacing[:, :2]
        has_endpoint_dupe = isinstance(grid, LinearGrid) and grid._toroidal_endpoint
    has_idx = hasattr(grid, f"num_{surface_label}") and hasattr(
        grid, f"_inverse_{surface_label}_idx"
    )
    unique_size = getattr(grid, f"num_{surface_label}", -1)
    inverse_idx = getattr(grid, f"_inverse_{surface_label}_idx", jnp.array([]))
    return unique_size, inverse_idx, spacing, has_endpoint_dupe, has_idx


def line_integrals(
    grid,
    q=jnp.array([1.0]),
    line_label="poloidal",
    fix_surface=("rho", 1.0),
    expand_out=True,
    tol=1e-14,
):
    """Compute line integrals over curves covering the given surface.

    As an example, by specifying the combination of ``line_label="poloidal"`` and
    ``fix_surface=("rho", 1.0)``, the intention is to integrate along the
    outermost perimeter of a particular zeta surface (toroidal cross-section),
    for each zeta surface in the grid.

    Notes
    -----
        It is assumed that the integration curve has length 1 when the line
        label is rho and length 2π when the line label is theta or zeta.
        You may want to multiply the input by the line length Jacobian.

        The grid must have nodes on the specified surface in ``fix_surface``.

        Correctness is not guaranteed on grids with duplicate nodes.
        An attempt to print a warning is made if the given grid has duplicate
        nodes and is one of the predefined grid types
        (``Linear``, ``Concentric``, ``Quadrature``).
        If the grid is custom, no attempt is made to warn.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
        The first dimension of the array should have size ``grid.num_nodes``.
        When ``q`` is n-dimensional, the intention is to integrate,
        over the domain parameterized by rho, poloidal, and zeta,
        an n-dimensional function over the previously mentioned domain.
    line_label : str
        The coordinate curve to compute the integration over.
        To clarify, a theta (poloidal) curve is the intersection of a
        rho surface (flux surface) and zeta (toroidal) surface.
    fix_surface : (str, float)
        A tuple of the form: label, value.
        ``fix_surface`` label should differ from ``line_label``.
        By default, ``fix_surface`` is chosen to be the flux surface at rho=1.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    integrals : ndarray
        Line integrals of the input over curves covering the given surface.
        By default, the returned array has the same shape as the input.

    """
    line_label = grid.get_label(line_label)
    fix_label = grid.get_label(fix_surface[0])
    errorif(
        line_label == fix_label,
        msg="There is no valid use for this combination of inputs.",
    )
    errorif(
        line_label != "poloidal" and isinstance(grid, ConcentricGrid),
        msg="ConcentricGrid should only be used for poloidal line integrals.",
    )
    warnif(
        isinstance(grid, LinearGrid) and grid.endpoint,
        msg="Correctness not guaranteed on grids with duplicate nodes.",
    )
    # Generate a new quantity q_prime which is zero everywhere
    # except on the fixed surface, on which q_prime takes the value of q.
    # Then forward the computation to surface_integrals().
    # The differential element of the line integral, denoted dl,
    # should correspond to the line label's spacing.
    # The differential element of the surface integral is
    # ds = dl * fix_surface_dl, so we scale q_prime by 1 / fix_surface_dl.
    axis = {"rho": 0, "poloidal": 1, "zeta": 2}
    column_id = axis[fix_label]
    mask = grid.nodes[:, column_id] == fix_surface[1]
    q_prime = (mask * jnp.atleast_1d(q).T / grid.spacing[:, column_id]).T
    (surface_label,) = axis.keys() - {line_label, fix_label}
    return surface_integrals(grid, q_prime, surface_label, expand_out, tol)


def surface_integrals(
    grid, q=jnp.array([1.0]), surface_label="rho", expand_out=True, tol=1e-14
):
    """Compute a surface integral for each surface in the grid.

    Notes
    -----
        It is assumed that the integration surface has area 4π² when the
        surface label is rho and area 2π when the surface label is theta or
        zeta. You may want to multiply the input by the surface area Jacobian.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to integrate.
        The first dimension of the array should have size ``grid.num_nodes``.
        When ``q`` is n-dimensional, the intention is to integrate,
        over the domain parameterized by rho, poloidal, and zeta,
        an n-dimensional function over the previously mentioned domain.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the integration over.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    integrals : ndarray
        Surface integral of the input over each surface in the grid.
        By default, the returned array has the same shape as the input.

    """
    return surface_integrals_map(grid, surface_label, expand_out, tol)(q)


def surface_integrals_map(grid, surface_label="rho", expand_out=True, tol=1e-14):
    """Returns a method to compute any surface integral for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the integration over.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    function : callable
        Method to compute any surface integral of the input ``q`` over each
        surface in the grid with code: ``function(q)``.

    """
    surface_label = grid.get_label(surface_label)
    warnif(
        surface_label == "poloidal" and isinstance(grid, ConcentricGrid),
        msg="Integrals over constant poloidal surfaces"
        " are poorly defined for ConcentricGrid.",
    )
    unique_size, inverse_idx, spacing, has_endpoint_dupe, has_idx = _get_grid_surface(
        grid, surface_label
    )
    spacing = jnp.prod(spacing, axis=1)

    if has_idx:
        # The ith row of masks is True only at the indices which correspond to the
        # ith surface. The integral over the ith surface is the dot product of the
        # ith row vector and the integrand defined over all the surfaces.
        mask = inverse_idx == jnp.arange(unique_size)[:, jnp.newaxis]
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
        # has_endpoint_dupe is true. If ``has_endpoint_dupe`` is true, this grid
        # has a duplicate surface at surface_label=0 and
        # surface_label=max surface value. Although the modulo of these values
        # are equal, their numeric values are not, so the integration
        # would treat them as different surfaces. We solve this issue by
        # combining the indices corresponding to the integrands of the duplicated
        # surface, so that the duplicate surface is treated as one, like in the
        # previous paragraph.
        mask = cond(
            has_endpoint_dupe,
            lambda _: put(mask, jnp.array([0, -1]), mask[0] | mask[-1]),
            lambda _: mask,
            None,
        )
    else:
        # If we don't have the idx attributes, we are forced to expand out.
        errorif(
            not has_idx and not expand_out,
            msg=f"Grid lacks attributes 'num_{surface_label}' and "
            f"'inverse_{surface_label}_idx', so this method "
            f"can't satisfy the request expand_out={expand_out}.",
        )
        # don't try to expand if already expanded
        expand_out = expand_out and has_idx
        axis = {"rho": 0, "poloidal": 1, "zeta": 2}[surface_label]
        # Converting nodes from numpy.ndarray to jaxlib.xla_extension.ArrayImpl
        # reduces memory usage by > 400% for the forward computation and Jacobian.
        nodes = jnp.asarray(grid.nodes[:, axis])
        # This branch will execute for custom grids, which don't have a use
        # case for having duplicate nodes, so we don't bother to modulo nodes
        # by 2pi or 2pi/NFP.
        mask = jnp.abs(nodes - nodes[:, jnp.newaxis]) <= tol
        # The above implementation was benchmarked to be more efficient than
        # alternatives with explicit loops in GitHub pull request #934.

    def integrate(q=jnp.array([1.0])):
        """Compute a surface integral for each surface in the grid.

        Notes
        -----
            It is assumed that the integration surface has area 4π² when the
            surface label is rho and area 2π when the surface label is theta or
            zeta. You may want to multiply the input by the surface area Jacobian.

        Parameters
        ----------
        q : ndarray
            Quantity to integrate.
            The first dimension of the array should have size ``grid.num_nodes``.
            When ``q`` is n-dimensional, the intention is to integrate,
            over the domain parameterized by rho, poloidal, and zeta,
            an n-dimensional function over the previously mentioned domain.

        Returns
        -------
        integrals : ndarray
            Surface integral of the input over each surface in the grid.

        """
        integrands = (spacing * jnp.nan_to_num(q).T).T
        integrals = jnp.tensordot(mask, integrands, axes=1)
        return grid.expand(integrals, surface_label) if expand_out else integrals

    return integrate


def surface_averages(
    grid,
    q,
    sqrt_g=jnp.array([1.0]),
    surface_label="rho",
    denominator=None,
    expand_out=True,
    tol=1e-14,
):
    """Compute a surface average for each surface in the grid.

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
        The first dimension of the array should have size ``grid.num_nodes``.
        When ``q`` is n-dimensional, the intention is to average,
        over the domain parameterized by rho, poloidal, and zeta,
        an n-dimensional function over the previously mentioned domain.
    sqrt_g : ndarray
        Coordinate system Jacobian determinant; see ``data_index["sqrt(g)"]``.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the average over.
    denominator : ndarray
        By default, the denominator is computed as the surface integral of
        ``sqrt_g``. This parameter can optionally be supplied to avoid
        redundant computations or to use a different denominator to compute
        the average. This array should broadcast with arrays of size
        ``grid.num_nodes`` (``grid.num_surface_label``) if ``expand_out``
        is true (false).
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    averages : ndarray
        Surface average of the input over each surface in the grid.
        By default, the returned array has the same shape as the input.

    """
    return surface_averages_map(grid, surface_label, expand_out, tol)(
        q, sqrt_g, denominator
    )


def surface_averages_map(grid, surface_label="rho", expand_out=True, tol=1e-14):
    """Returns a method to compute any surface average for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the average over.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    function : callable
        Method to compute any surface average of the input ``q`` and optionally
        the volume Jacobian ``sqrt_g`` over each surface in the grid with code:
        ``function(q, sqrt_g)``.

    """
    surface_label = grid.get_label(surface_label)
    has_idx = hasattr(grid, f"num_{surface_label}") and hasattr(
        grid, f"_inverse_{surface_label}_idx"
    )
    # If we don't have the idx attributes, we are forced to expand out.
    errorif(
        not has_idx and not expand_out,
        msg=f"Grid lacks attributes 'num_{surface_label}' and "
        f"'inverse_{surface_label}_idx', so this method "
        f"can't satisfy the request expand_out={expand_out}.",
    )
    integrate = surface_integrals_map(
        grid, surface_label, expand_out=not has_idx, tol=tol
    )
    # don't try to expand if already expanded
    expand_out = expand_out and has_idx

    def _surface_averages(q, sqrt_g=jnp.array([1.0]), denominator=None):
        """Compute a surface average for each surface in the grid.

        Notes
        -----
            Implements the flux-surface average formula given by equation 4.9.11 in
            W.D. D'haeseleer et al. (1991) doi:10.1007/978-3-642-75595-8.

        Parameters
        ----------
        q : ndarray
            Quantity to average.
            The first dimension of the array should have size ``grid.num_nodes``.
            When ``q`` is n-dimensional, the intention is to average,
            over the domain parameterized by rho, poloidal, and zeta,
            an n-dimensional function over the previously mentioned domain.
        sqrt_g : ndarray
            Coordinate system Jacobian determinant; see ``data_index["sqrt(g)"]``.
        denominator : ndarray
            By default, the denominator is computed as the surface integral of
            ``sqrt_g``. This parameter can optionally be supplied to avoid
            redundant computations or to use a different denominator to compute
            the average. This array should broadcast with arrays of size
            ``grid.num_nodes`` (``grid.num_surface_label``) if ``expand_out``
            is true (false).

        Returns
        -------
        averages : ndarray
            Surface average of the input over each surface in the grid.

        """
        q, sqrt_g = jnp.atleast_1d(q, sqrt_g)
        numerator = integrate((sqrt_g * q.T).T)
        # memory optimization to call expand() at most once
        if denominator is None:
            # skip integration if constant
            denominator = (
                (4 * jnp.pi**2 if surface_label == "rho" else 2 * jnp.pi) * sqrt_g
                if sqrt_g.size == 1
                else integrate(sqrt_g)
            )
            averages = (numerator.T / denominator).T
            if expand_out:
                averages = grid.expand(averages, surface_label)
        else:
            if expand_out:
                # implies denominator given with size grid.num_nodes
                numerator = grid.expand(numerator, surface_label)
            averages = (numerator.T / denominator).T
        return averages

    return _surface_averages


def surface_integrals_transform(grid, surface_label="rho"):
    """Returns a method to compute any integral transform over each surface in grid.

    The returned method takes an array input ``q`` and returns an array output.

    Given a set of kernel functions in ``q``, each parameterized by at most
    five variables, the returned method computes an integral transform,
    reducing ``q`` to a set of functions of at most three variables.

    Define the domain D = u₁ × u₂ × u₃ and the codomain C = u₄ × u₅ × u₆.
    For every surface of constant u₁ in the domain, the returned method
    evaluates the transform Tᵤ₁ : u₂ × u₃ × C → C, where Tᵤ₁ projects
    away the parameters u₂ and u₃ via an integration of the given kernel
    function Kᵤ₁ over the corresponding surface of constant u₁.

    Notes
    -----
        It is assumed that the integration surface has area 4π² when the
        surface label is rho and area 2π when the surface label is theta or
        zeta. You may want to multiply the input ``q`` by the surface area
        Jacobian.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the integration over.
        These correspond to the domain parameters discussed in this method's
        description. In particular, ``surface_label`` names u₁.

    Returns
    -------
    function : callable
        Method to compute any surface integral transform of the input ``q`` over
        each surface in the grid with code: ``function(q)``.

        The first dimension of ``q`` should always discretize some function, g,
        over the domain, and therefore, have size ``grid.num_nodes``.
        The second dimension may discretize some function, f, over the
        codomain, and therefore, have size that matches the desired number of
        points at which the output is evaluated.

        This method can also be used to compute the output one point at a time,
        in which case ``q`` can have shape (``grid.num_nodes``, ).

        Input
        -----
        If ``q`` has one dimension, then it should have shape
        (``grid.num_nodes``, ).
        If ``q`` has multiple dimensions, then it should have shape
        (``grid.num_nodes``, *f.shape).

        Output
        ------
        Each element along the first dimension of the returned array, stores
        Tᵤ₁ for a particular surface of constant u₁ in the given grid.
        The order is sorted in increasing order of the values which specify u₁.

        If ``q`` has one dimension, the returned array has shape
        (grid.num_surface_label, ).
        If ``q`` has multiple dimensions, the returned array has shape
        (grid.num_surface_label, *f.shape).

    """
    # Expansion should not occur here. The typical use case of this method is to
    # transform into the computational domain, so the second dimension that
    # discretizes f over the codomain will typically have size grid.num_nodes
    # to broadcast with quantities in data_index.
    surface_label = grid.get_label(surface_label)
    has_idx = hasattr(grid, f"num_{surface_label}") and hasattr(
        grid, f"_inverse_{surface_label}_idx"
    )
    errorif(
        not has_idx,
        msg=f"Grid lacks attributes 'num_{surface_label}' and "
        f"'inverse_{surface_label}_idx', which are required for this function.",
    )
    return surface_integrals_map(grid, surface_label, expand_out=False)


def surface_variance(
    grid,
    q,
    weights=jnp.array([1.0]),
    bias=False,
    surface_label="rho",
    expand_out=True,
    tol=1e-14,
):
    """Compute the weighted sample variance of ``q`` on each surface of the grid.

    Computes nₑ / (nₑ − b) * (∑ᵢ₌₁ⁿ (qᵢ − q̅)² wᵢ) / (∑ᵢ₌₁ⁿ wᵢ).
    wᵢ is the weight assigned to qᵢ given by the product of ``weights[i]`` and
       the differential surface area element (not already weighted by the area
       Jacobian) at the node where qᵢ is evaluated,
    q̅ is the weighted mean of q,
    b is 0 if the biased sample variance is to be returned and 1 otherwise,
    n is the number of samples on a surface, and
    nₑ ≝ (∑ᵢ₌₁ⁿ wᵢ)² / ∑ᵢ₌₁ⁿ wᵢ² is the effective number of samples.

    As the weights wᵢ approach each other, nₑ approaches n, and the output
    converges to ∑ᵢ₌₁ⁿ (qᵢ − q̅)² / (n − b).

    Notes
    -----
        There are three different methods to unbias the variance of a weighted
        sample so that the computed variance better estimates the true variance.
        Whether the method is correct for a particular use case depends on what
        the weights assigned to each sample represent.

        This function implements the first case, where the weights are not random
        and are intended to assign more weight to some samples for reasons
        unrelated to differences in uncertainty between samples. See
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights.

        The second case is when the weights are intended to assign more weight
        to samples with less uncertainty. See
        https://en.wikipedia.org/wiki/Inverse-variance_weighting.
        The unbiased sample variance for this case is obtained by replacing the
        effective number of samples in the formula this function implements,
        nₑ, with the actual number of samples n.

        The third case is when the weights denote the integer frequency of each
        sample. See
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights.
        This is indeed a distinct case from the above two because here the
        weights encode additional information about the distribution.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    q : ndarray
        Quantity to compute the sample variance.
    weights : ndarray
        Weight assigned to each sample of ``q``.
        A good candidate for this parameter is the surface area Jacobian.
    bias : bool
        If this condition is true, then the biased estimator of the sample
        variance is returned. This is desirable if you are only concerned with
        computing the variance of the given set of numbers and not the
        distribution the numbers are (potentially) sampled from.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute the variance over.
    expand_out : bool
        Whether to expand the output array so that the output has the same
        shape as the input. Defaults to true so that the output may be
        broadcast in the same way as the input. Setting to false will save
        memory.
    tol : float
        Tolerance for considering nodes the same.
        Only relevant if the grid object doesn't already have this information.

    Returns
    -------
    variance : ndarray
        Variance of the given weighted sample over each surface in the grid.
        By default, the returned array has the same shape as the input.

    """
    surface_label = grid.get_label(surface_label)
    _, _, spacing, _, has_idx = _get_grid_surface(grid, surface_label)
    # If we don't have the idx attributes, we are forced to expand out.
    errorif(
        not has_idx and not expand_out,
        msg=f"Grid lacks attributes 'num_{surface_label}' and "
        f"'inverse_{surface_label}_idx', so this method "
        f"can't satisfy the request expand_out={expand_out}.",
    )
    integrate = surface_integrals_map(
        grid, surface_label, expand_out=not has_idx, tol=tol
    )

    v1 = integrate(weights)
    v2 = integrate(weights**2 * jnp.prod(spacing, axis=-1))
    # effective number of samples per surface
    n_e = v1**2 / v2
    # analogous to Bessel's bias correction
    correction = n_e / (n_e - (not bias))

    q = jnp.atleast_1d(q)
    # compute variance in two passes to avoid catastrophic round off error
    mean = (integrate((weights * q.T).T).T / v1).T
    if has_idx:  # guard so that we don't try to expand when already expanded
        mean = grid.expand(mean, surface_label)
    variance = (correction * integrate((weights * ((q - mean) ** 2).T).T).T / v1).T
    if expand_out and has_idx:
        return grid.expand(variance, surface_label)
    else:
        return variance


def surface_max(grid, x, surface_label="rho"):
    """Get the max of x for each surface in the grid.

    Parameters
    ----------
    grid : Grid
        Collocation grid containing the nodes to evaluate at.
    x : ndarray
        Quantity to find max.
        The array should have size grid.num_nodes.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute max over.

    Returns
    -------
    maxs : ndarray
        Maximum of x over each surface in grid.
        The returned array has the same shape as the input.

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
        The array should have size grid.num_nodes.
    surface_label : str
        The surface label of rho, poloidal, or zeta to compute min over.

    Returns
    -------
    mins : ndarray
        Minimum of x over each surface in grid.
        The returned array has the same shape as the input.

    """
    surface_label = grid.get_label(surface_label)
    unique_size, inverse_idx, _, _, has_idx = _get_grid_surface(grid, surface_label)
    errorif(
        not has_idx,
        NotImplementedError,
        msg=f"Grid lacks attributes 'num_{surface_label}' and "
        f"'inverse_{surface_label}_idx', which are required for this function.",
    )
    inverse_idx = jnp.asarray(inverse_idx)
    x = jnp.asarray(x)
    mins = jnp.full(unique_size, jnp.inf)

    def body(i, mins):
        mins = put(mins, inverse_idx[i], jnp.minimum(x[i], mins[inverse_idx[i]]))
        return mins

    mins = fori_loop(0, inverse_idx.size, body, mins)
    # The above implementation was benchmarked to be more efficient than
    # alternatives without explicit loops in GitHub pull request #501.
    return grid.expand(mins, surface_label)
