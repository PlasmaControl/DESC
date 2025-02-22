"""High order accurate Laplace solver.

This is used to solve vacuum equilibria without assuming nested flux surfaces.
Once pushed into an optimization loop, we can just do constrained optimization
under linear constraint B₀⋅n = -∇ϕ⋅n so that we avoid inverting the large system
for the Fourier harmonics of ϕ. The methods here perform the inversion as this
is necessary to write correctness tests.
"""

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.batching import vmap_chunked
from desc.grid import LinearGrid
from desc.integrals.quad_utils import _zero
from desc.integrals.singularities import (
    _get_interpolator,
    _kernel_biot_savart_coulomb,
    _kernel_Bn_over_r,
    _kernel_Phi_dGp_dn,
    _nonsingular_part,
    singular_integral,
)
from desc.transform import Transform


def get_laplace_dict(
    eq,
    B0,
    evl_grid,
    phi_grid=None,
    src_grid=None,
    chunk_size=None,
    evl_names=None,
    B0n=None,
):
    """Compute quantities needed for Laplace solver.

    Parameters
    ----------
    eq : Equilibrium
        Configuration with surface geometry defining ∂D.
    B0 : _MagneticField
        Magnetic field such that ∇ × 𝐁₀ = μ₀ 𝐉
        where 𝐉 is the current in amperes everywhere.
    evl_grid : Grid
        Evaluation points on D for the magnetic field.
    phi_grid : Grid
        Interpolation points on ∂D.
        Resolution determines accuracy of Φ interpolation.
    src_grid : Grid
        Source points on ∂D for quadrature of kernels.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    evl_names : list[str]
        Additional names of quantities to compute on ``evl_grid``.
    B0n : jnp.ndarray
        Optional, B₀⋅n on ∂D.

    Returns
    -------
    laplace : dict
        Dictionary with needed stuff for Laplace solver.

    """
    position = ["R", "phi", "Z"]
    if evl_names is not None:
        evl_names = position + evl_names
    phi_names = position
    src_names = position + ["n_rho", "|e_theta x e_zeta|", "e_theta", "e_zeta"]

    phi_grid = phi_grid or LinearGrid(
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP if eq.N > 0 else 64,
    )
    src_grid = src_grid or phi_grid
    # TODO: add basis symmetry
    basis = DoubleFourierSeries(M=phi_grid.M, N=phi_grid.N, NFP=eq.NFP)

    # Compute source grid data.
    src_transform = Transform(src_grid, basis, derivs=1)
    src_data = eq.compute(src_names, grid=src_grid)
    # Compute Phi grid data.
    src_grid_is_phi_grid = src_grid.equiv(phi_grid)
    if src_grid_is_phi_grid:
        phi_transform = src_transform
        phi_data = src_data
    else:
        phi_transform = Transform(phi_grid, basis)
        phi_data = eq.compute(phi_names, grid=phi_grid)
    # Compute eval grid data.
    if evl_grid.equiv(phi_grid):
        evl_data = phi_data
    elif not src_grid_is_phi_grid and evl_grid.equiv(src_grid):
        evl_data = src_data
    else:
        evl_data = eq.compute(evl_names, grid=evl_grid)

    if B0n is not None:
        src_data["Bn"] = B0n
    elif B0 is not None:
        src_data["Bn"], _ = B0.compute_Bnormal(
            eq.surface,
            eval_grid=src_grid,
            source_grid=src_grid,
            vc_source_grid=src_grid,
            chunk_size=chunk_size,
        )

    return {
        "evl_grid": evl_grid,
        "phi_transform": phi_transform,
        "src_transform": src_transform,
        "evl_data": evl_data,
        "phi_data": phi_data,
        "src_data": src_data,
    }


def _compute_Phi_mn(laplace, chunk_size=None, **kwargs):
    """Compute Fourier coefficients of vacuum potential Φ on ∂D.

    Parameters
    ----------
    laplace : dict
        Dictionary with needed stuff for Laplace solver.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    laplace : dict
        ``laplace["phi_data"]["Phi_mn"]`` stores Fourier coefficients of Φ on ∂D.

    """
    same_transform = laplace["phi_transform"] == laplace["src_transform"]
    num_modes = laplace["phi_transform"].num_modes
    interpolator = _get_interpolator(
        laplace["phi_transform"].grid,
        laplace["src_transform"].grid,
        laplace["src_data"],
        **kwargs,
    )

    def func(Phi_mn):
        """Compute left hand side of Green's function solution.

        After Fourier transform, ``func`` is linear in the spectral coefficients Φₘₙ.
        We approximate this as finite-dimensional, which enables writing the left
        hand side as A @ Φₘₙ. Then Φₘₙ is found by solving func(Φₘₙ) = A @ Φₘₙ = b.
        """
        src_data = laplace["src_data"].copy()
        src_data["Phi"] = laplace["src_transform"].transform(Phi_mn)
        Phi = (
            src_data["Phi"]
            if same_transform
            else laplace["phi_transform"].transform(Phi_mn)
        )
        return Phi + singular_integral(
            laplace["phi_data"],
            src_data,
            _kernel_Phi_dGp_dn,
            interpolator,
            chunk_size,
        ).squeeze() / (2 * jnp.pi)

    b = -singular_integral(
        laplace["phi_data"],
        laplace["src_data"],
        _kernel_Bn_over_r,
        interpolator,
        chunk_size,
    ).squeeze() / (2 * jnp.pi)

    # func is expensive, so it is better to construct full Jacobian once
    # rather than iterative solves like jax.scipy.sparse.linalg.cg.
    A = vmap_chunked(func, chunk_size=1)(jnp.eye(num_modes)).T
    laplace["phi_data"]["Phi_mn"] = jnp.linalg.solve(A, b)
    return laplace


def _compute_shielding_current(laplace):
    """K_vc = -n × ∇Φ."""
    Phi_mn = laplace["phi_data"]["Phi_mn"]
    src_data = laplace["src_data"]
    src_data["Phi_t"] = laplace["src_transform"].transform(Phi_mn, dt=1)
    src_data["Phi_z"] = laplace["src_transform"].transform(Phi_mn, dz=1)
    src_data["K^theta"] = src_data["Phi_z"] / src_data["|e_theta x e_zeta|"]
    src_data["K^zeta"] = -src_data["Phi_t"] / src_data["|e_theta x e_zeta|"]
    src_data["K_vc"] = (
        src_data["K^theta"][:, jnp.newaxis] * src_data["e_theta"]
        + src_data["K^zeta"][:, jnp.newaxis] * src_data["e_zeta"]
    )
    return laplace


def _compute_grad_Phi(laplace, chunk_size=None, interior=True, **kwargs):
    """Computes vacuum field ∇Φ.

    Parameters
    ----------
    laplace : dict
        Dictionary with needed stuff for Laplace solver.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    interior : bool
        If true, it is assumed the evaluation grid is subset of D.
        If false, it is assumed the evaluation grid is subset of ∂D.

    Returns
    -------
    laplace : dict
        Vacuum field ∇Φ stored in ``laplace["evl_data"]["grad(Phi)"]``.
        Shape (``eval_grid.grid.num_nodes``, 3).

    """
    laplace["evl_data"]["grad(Phi)"] = (
        _nonsingular_part(
            laplace["evl_data"],
            laplace["evl_grid"],
            laplace["src_data"],
            laplace["src_transform"].grid,
            st=jnp.nan,
            sz=jnp.nan,
            kernel=_kernel_biot_savart_coulomb,
            chunk_size=chunk_size,
            _eta=_zero,
        )
        if interior
        else 2
        * singular_integral(
            laplace["evl_data"],
            laplace["src_data"],
            kernel=_kernel_biot_savart_coulomb,
            interpolator=_get_interpolator(
                laplace["evl_grid"],
                laplace["src_transform"].grid,
                laplace["src_data"],
                **kwargs,
            ),
            chunk_size=chunk_size,
        )
    )
    return laplace


def compute_B_from_B0(B0, laplace, chunk_size=None, interior=True, **kwargs):
    """Compute vacuum field that satisfies LCFS boundary condition in plasma interior.

    Let D, D^∁ denote the interior, exterior of a toroidal region with
    boundary ∂D. Computes the magnetic field 𝐁 in units of Tesla such that

    - 𝐁 = 𝐁₀ + ∇Φ     on D
    - ∇ × 𝐁₀ = μ₀ 𝐉   on D ∪ D^∁
    - ∇ ⋅ 𝐁₀ = 0      on D ∪ D^∁
    - 𝐁 ⋅ ∇ρ = 0      on ∂D
    - ∇²Φ = 0         on D

    Examples
    --------
    In a vacuum, the magnetic field may be written 𝐁 = ∇𝛷.
    The solution to ∇²𝛷 = 0, under a homogenous boundary
    condition 𝐁 ⋅ ∇ρ = 0, is 𝛷 = 0. To obtain a non-trivial solution,
    the boundary condition may be modified.
    Let 𝐁 = 𝐁₀ + ∇Φ.
    If 𝐁₀ ≠ 0 and satisfies ∇ × 𝐁₀ = 0, then ∇²Φ = 0 solved
    under an inhomogeneous boundary condition yields a non-trivial solution.
    If 𝐁₀ ≠ -∇Φ, then 𝐁 ≠ 0.

    Parameters
    ----------
    B0 : _MagneticField
        Magnetic field such that ∇ × 𝐁₀ = μ₀ 𝐉
        where 𝐉 is the current in amperes everywhere.
    laplace : dict
        Dictionary with needed stuff for Laplace solver.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    interior : bool
        If true, it is assumed the evaluation grid is subset of D.
        If false, it is assumed the evaluation grid is subset of ∂D.

    Returns
    -------
    B : jnp.ndarray
        Magnetic field evaluated on ``eval_grid``.

    """
    laplace = _compute_Phi_mn(laplace, chunk_size, **kwargs)
    laplace = _compute_shielding_current(laplace)
    laplace = _compute_grad_Phi(laplace, chunk_size, interior, **kwargs)
    B0 = B0.compute_magnetic_field(
        coords=jnp.column_stack(
            [
                laplace["evl_data"]["R"],
                laplace["evl_data"]["phi"],
                laplace["evl_data"]["Z"],
            ]
        ),
        source_grid=laplace["src_transform"].grid,
        chunk_size=chunk_size,
    )
    return B0 + laplace["evl_data"]["grad(Phi)"]
