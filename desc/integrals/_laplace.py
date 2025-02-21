"""High order accurate Laplace solver.

This is used to solve vacuum equilibria without assuming nested flux surfaces.
Once pushed into an optimization loop, we can just do constrained optimization
under linear constraint B₀⋅n = -∇ϕ⋅n so that we avoid inverting the large system
for the Fourier harmonics of ϕ. The methods here perform the inversion as this
is necessary to write correctness tests.
"""

from scipy.constants import mu_0

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.batching import vmap_chunked
from desc.compute.geom_utils import rpz2xyz, xyz2rpz_vec
from desc.grid import LinearGrid
from desc.integrals.singularities import (
    _get_interpolator,
    _kernel_biot_savart,
    _kernel_Bn_over_r,
    _kernel_Phi_dG_dn,
    singular_integral,
)
from desc.magnetic_fields import FourierCurrentPotentialField
from desc.magnetic_fields._core import coulomb_general
from desc.transform import Transform
from desc.utils import dot

# TODO(KAYA) Finish fix this on Feb 21 morning.


def compute_laplace(eq, B0, evl_grid, phi_grid=None, src_grid=None, bs_chunk_size=None):
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
    bs_chunk_size : int
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    Returns
    -------
    laplace : dict
        Dictionary with needed stuff for Laplace solver.

    """
    phi_names = ["R", "phi", "Z"]
    evl_names = phi_names + ["n_rho"]
    src_names = phi_names + ["n_rho", "|e_theta x e_zeta|", "e_theta", "e_zeta"]

    phi_grid = phi_grid or LinearGrid(
        M=eq.M_grid,
        N=eq.N_grid,
        NFP=eq.NFP if eq.N > 0 else 64,
    )
    src_grid = src_grid or phi_grid
    basis = DoubleFourierSeries(M=phi_grid.M, N=phi_grid.N, NFP=eq.NFP)

    # TODO: Can't pass [0, 1, 1]?
    src_transform = Transform(src_grid, basis, derivs=1)
    src_data = eq.compute(src_names, grid=src_grid)

    evl_data = (
        src_data if src_grid.equiv(evl_grid) else eq.compute(evl_names, grid=evl_grid)
    )
    if src_grid.equiv(phi_grid):
        phi_transform = src_transform
        phi_data = src_data
    else:
        phi_transform = Transform(phi_grid, basis)
        phi_data = (
            evl_data
            if evl_grid.equiv(phi_grid)
            else eq.compute(phi_names, grid=phi_grid)
        )

    if hasattr(B0, "compute_Bnormal"):
        src_data["Bn"], _ = B0.compute_Bnormal(
            eq.surface,
            eval_grid=src_grid,
            source_grid=src_grid,
            vc_source_grid=src_grid,
            chunk_size=bs_chunk_size,
        )
    else:
        # Then assume Bn was given.
        src_data["Bn"] = B0

    laplace = {
        "evl_grid": evl_grid,
        "phi_transform": phi_transform,
        "src_transform": src_transform,
        "evl_data": evl_data,
        "phi_data": phi_data,
        "src_data": src_data,
    }
    return laplace


def _compute_coulomb_field(evl_data, src_data, bs_chunk_size=None):
    coulomb = coulomb_general(
        re=rpz2xyz(jnp.column_stack([evl_data["R"], evl_data["phi"], evl_data["Z"]])),
        rs=rpz2xyz(jnp.column_stack([src_data["R"], src_data["phi"], src_data["Z"]])),
        Bn=src_data["Bn"],
        dS=src_data["|e_theta x e_zeta|"],
        chunk_size=bs_chunk_size,
    )
    coulomb = xyz2rpz_vec(coulomb, phi=evl_data["phi"])
    return coulomb


def _compute_Phi_mn(laplace, chunk_size=None, **kwargs):
    """Compute Fourier coefficients of vacuum potential Φ on ∂D.

    Let D, D^∁ denote the interior, exterior of a toroidal region with
    boundary ∂D. Computes the magnetic field 𝐁 in units of Tesla such that

    - 𝐁 = 𝐁₀ + ∇Φ     on D
    - ∇ × 𝐁₀ = μ₀ 𝐉   on D ∪ D^∁
    - ∇ ⋅ 𝐁₀ = 0      on D ∪ D^∁
    - 𝐁 ⋅ ∇ρ = 0      on ∂D
    - ∇²Φ = 0         on D

    Parameters
    ----------
    laplace : dict
        Dictionary with needed stuff for Laplace solver.
    chunk_size : int or None
        Size to split singular integration computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    Phi_mn : jnp.ndarray
        Fourier coefficients of Φ on ∂D.

    """
    phi_is_src = laplace["phi_transform"] == laplace["src_transform"]
    num_modes = laplace["phi_transform"].num_modes
    interpolator = _get_interpolator(
        laplace["phi_transform"].grid,
        laplace["src_transform"].grid,
        laplace["src_data"],
        **kwargs,
    )

    def LHS(Phi_mn):
        """Compute left hand side of Green's function solution.

        After Fourier transform, the LHS is linear in the spectral coefficients Φₘₙ.
        We approximate this as finite-dimensional, which enables writing the left
        hand side as A @ Φₘₙ. Then Φₘₙ is found by solving LHS(Φₘₙ) = A @ Φₘₙ = RHS.
        """
        src_data = laplace["src_data"].copy()
        src_data["Phi"] = laplace["src_transform"].transform(Phi_mn)

        bs = singular_integral(
            laplace["phi_data"],
            src_data,
            _kernel_Phi_dG_dn,
            interpolator,
            chunk_size,
        ).squeeze() / (2 * jnp.pi)

        Phi = (
            src_data["Phi"]
            if phi_is_src
            else laplace["phi_transform"].transform(Phi_mn)
        )
        return Phi + bs

    RHS = -singular_integral(
        laplace["phi_data"],
        laplace["src_data"],
        _kernel_Bn_over_r,
        interpolator,
        chunk_size,
    ).squeeze() / (2 * jnp.pi)

    # LHS is expensive, so it is better to construct full Jacobian once
    # rather than iterative solves like jax.scipy.sparse.linalg.cg.
    A = vmap_chunked(LHS, chunk_size=1)(jnp.eye(num_modes)).T
    return jnp.linalg.solve(A, RHS)


def _compute_dPhi_dn(laplace, Phi_mn, chunk_size=None, bs_chunk_size=None, **kwargs):
    """Computes vacuum field ∇Φ ⋅ n on ∂D.

    Let D, D^∁ denote the interior, exterior of a toroidal region with
    boundary ∂D. Computes the magnetic field 𝐁 in units of Tesla such that

    - 𝐁 = 𝐁₀ + ∇Φ     on D
    - ∇ × 𝐁₀ = μ₀ 𝐉   on D ∪ D^∁
    - ∇ ⋅ 𝐁₀ = 0      on D ∪ D^∁
    - 𝐁 ⋅ ∇ρ = 0      on ∂D
    - ∇²Φ = 0         on D

    Parameters
    ----------
    laplace : dict
        Dictionary with needed stuff for Laplace solver.
    Phi_mn : jnp.ndarray
        Fourier coefficients of Φ on the boundary.
    chunk_size : int or None
        Size to split singular integration computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    Returns
    -------
    dPhi_dn : jnp.ndarray
        Shape (``eval_grid.grid.num_nodes``, ).
        Vacuum field ∇Φ ⋅ n on ∂D.

    """
    interpolator = _get_interpolator(
        laplace["evl_grid"],
        laplace["src_transform"].grid,
        laplace["src_data"],
        **kwargs,
    )

    evl_data = laplace["evl_data"]
    src_data = laplace["src_data"].copy()
    src_data["Phi_t"] = laplace["src_transform"].transform(Phi_mn, dt=1)
    src_data["Phi_z"] = laplace["src_transform"].transform(Phi_mn, dz=1)
    # K_vc = -n × ∇Φ
    src_data["K^theta"] = src_data["Phi_z"] / src_data["|e_theta x e_zeta|"]
    src_data["K^zeta"] = -src_data["Phi_t"] / src_data["|e_theta x e_zeta|"]
    src_data["K_vc"] = (
        src_data["K^theta"][:, jnp.newaxis] * src_data["e_theta"]
        + src_data["K^zeta"][:, jnp.newaxis] * src_data["e_zeta"]
    )

    # ∇Φ = ∂Φ/∂ρ ∇ρ + ∂Φ/∂θ ∇θ + ∂Φ/∂ζ ∇ζ
    # but we can not obtain ∂Φ/∂ρ from Φₘₙ. Biot-Savart gives
    # K_vc = -n × ∇Φ where Φ has units Tesla-meters
    # ∇Φ(x ∈ ∂D) = 1/2π ∫_∂D df' K_vc × ∇G(x,x') - coulomb
    # Biot-Savart kernel assumes Φ in amperes, so we account for that.
    fcp = (2 / mu_0) * singular_integral(
        evl_data,
        src_data,
        _kernel_biot_savart,
        interpolator,
        chunk_size,
    )
    coulomb = _compute_coulomb_field(evl_data, src_data, bs_chunk_size)
    return dot(fcp - coulomb, evl_data["n_rho"])


def compute_B_from_B0(
    surface, B0, laplace, chunk_size=None, bs_chunk_size=None, **kwargs
):
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
    surface : Surface
        Surface geometry defining ∂D.
    B0 : _MagneticField
        Magnetic field such that ∇ × 𝐁₀ = μ₀ 𝐉
        where 𝐉 is the current in amperes everywhere.
    laplace : dict
        Dictionary with needed stuff for Laplace solver.
    chunk_size : int or None
        Size to split singular integration computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    Returns
    -------
    B : jnp.ndarray
        Magnetic field evaluated on ``eval_grid``.

    """
    re = jnp.column_stack(
        [laplace["evl_data"]["R"], laplace["evl_data"]["phi"], laplace["evl_data"]["Z"]]
    )
    Phi_mn = _compute_Phi_mn(laplace, chunk_size, **kwargs)
    fcp = FourierCurrentPotentialField.from_surface(
        surface,
        -Phi_mn / mu_0,
        laplace["phi_transform"].basis.modes[:, 1:],
    )
    fcp = fcp.compute_magnetic_field(
        coords=re,
        source_grid=laplace["src_transform"].grid,
        chunk_size=bs_chunk_size,
    )
    grad_Phi = fcp - _compute_coulomb_field(
        laplace["evl_data"],
        laplace["src_data"],
        bs_chunk_size,
    )
    B0 = B0.compute_magnetic_field(
        coords=re,
        source_grid=laplace["src_transform"].grid,
        chunk_size=bs_chunk_size,
    )
    return B0 + grad_Phi
