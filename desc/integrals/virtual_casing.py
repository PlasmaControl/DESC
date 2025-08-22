"""Virtual casing surface integrals for points strictly outside the surface."""

from desc.backend import fori_loop, jnp
from desc.batching import batch_map
from desc.utils import errorif, xyz2rpz_vec


def integrate_surface(coords, source_data, source_grid, kernel, chunk_size=None):
    """
    Integrate kernel over a surface at a point strictly outside that surface.

    For integration on the surface itself, see desc.singularities.singular_integral.

    Parameters
    ----------
    coords : array-like, shape (n,3)
        Evaluation points, in cylindrical coordinates.
    source_data : dict
        Dictionary of data at source points (corresponding to source_grid). Keys
        should be those required by kernel as kernel.keys. Vector data should be in
        rpz basis.
    source_grid : _Grid
        Grid in flux coordinates over which the kernel should be integrated.
    kernel : callable
        Kernel function to evaluate and integrate over surface described by source_grid.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    """
    assert (
        source_grid.num_rho == 1
    ), f"""source_grid must be on a flux surface.
            Got source_grid.num_rho = {source_grid.num_rho}"""

    # make sure source dict has zeta and phi to avoid
    # adding keys to dict during iteration
    source_zeta = source_data.setdefault("zeta", source_grid.nodes[:, 2])
    source_phi = source_data["phi"]

    # Convert coords to the format expected by the kernel
    eval_data = {"R": coords[:, 0], "phi": coords[:, 1], "Z": coords[:, 2]}

    # Calculate weights for 2D surface integral
    ht = 2 * jnp.pi / source_grid.num_theta
    hz = 2 * jnp.pi / source_grid.num_zeta / source_grid.NFP
    w = source_data["|e_theta x e_zeta|"][jnp.newaxis] * ht * hz

    def nfp_loop(j, f_data):
        """Calculate effects from source points on a single field period.

        The surface integral is computed on the full domain because the kernels of
        interest have toroidal variation and are not NFP periodic. To that end, the
        integral is computed on every field period and summed. The ``source_grid`` is
        the first field period because DESC truncates the computational domain to
        ζ ∈ [0, 2π/grid.NFP) and changes variables to the spectrally condensed
        ζ* = basis.NFP ζ. Therefore, we shift the domain to the next field period by
        incrementing the toroidal coordinate of the grid by 2π/NFP. For an axisymmetric
        configuration, it is most efficient for ``source_grid`` to be a single toroidal
        cross-section. To capture toroidal effects of the kernels on those grids for
        axisymmetric configurations, we set a dummy value for NFP to an integer larger
        than 1 so that the toroidal increment can move to a new spot.
        """
        f, source_data = f_data
        source_data["zeta"] = (source_zeta + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )
        source_data["phi"] = (source_phi + j * 2 * jnp.pi / source_grid.NFP) % (
            2 * jnp.pi
        )

        # nest this def to avoid having to pass the modified source_data around the loop
        # easier to just close over it and let JAX figure it out
        def eval_pt(eval_data_i):
            k = kernel(eval_data_i, source_data).reshape(
                -1, source_grid.num_nodes, kernel.ndim
            )
            return jnp.sum(k * w[..., jnp.newaxis], axis=1)

        f += batch_map(eval_pt, eval_data, chunk_size).reshape(
            coords.shape[0], kernel.ndim
        )
        return f, source_data

    # This error should be raised earlier since this is not the only place
    # we need the higher dummy NFP value, but the error message is more
    # helpful with the nfp loop docstring.
    errorif(
        source_grid.num_zeta == 1 and source_grid.NFP == 1,
        msg="Source grid cannot compute toroidal effects.\n"
        "Increase NFP of source grid to e.g. 64.\n"
        "This is required to " + nfp_loop.__doc__,
    )
    f = jnp.zeros((coords.shape[0], kernel.ndim))
    f, _ = fori_loop(0, source_grid.NFP, nfp_loop, (f, source_data))

    # undo rotation of source_zeta
    source_data["zeta"] = source_zeta
    source_data["phi"] = source_phi
    # we sum vectors at different points, so they need to be in xyz for that to work
    # but then need to convert vectors back to rpz
    if kernel.ndim == 3:
        f = xyz2rpz_vec(f, phi=eval_data["phi"])

    return f
