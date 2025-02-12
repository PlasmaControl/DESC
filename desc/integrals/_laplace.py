"""High order accurate Laplace solver."""

import numpy as np
from jax import jacfwd
from scipy.constants import mu_0

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.compute.geom_utils import rpz2xyz, rpz2xyz_vec
from desc.grid import LinearGrid
from desc.integrals.singularities import (
    DFTInterpolator,
    FFTInterpolator,
    _kernel_biot_savart,
    _kernel_Bn_over_r,
    _kernel_Phi_dG_dn,
    best_ratio,
    heuristic_support_params,
    singular_integral,
)
from desc.magnetic_fields import FourierCurrentPotentialField
from desc.transform import Transform
from desc.utils import dot, safediv, safenorm, warnif


def compute_B_laplace(
    eq,
    B0,
    eval_grid,
    source_grid=None,
    Phi_grid=None,
    chunk_size=None,
    check=False,
):
    """Compute magnetic field in interior of plasma due to vacuum potential.

    Let D, D^∁ denote the interior, exterior of a toroidal region with
    boundary ∂D. Computes the magnetic field 𝐁 in units of Tesla such that

    - 𝐁 = 𝐁₀ + ∇Φ     on D
    - ∇ × 𝐁 = ∇ × 𝐁₀  on D ∪ D^∁ (i.e. ∇Φ is single-valued or periodic)
    - ∇ × 𝐁₀ = μ₀ 𝐉    on D ∪ D^∁
    - 𝐁 * ∇ρ = 0      on ∂D
    - ∇²Φ = 0         on D

    Examples
    --------
    In a vacuum, the magnetic field may be written 𝐁 = ∇𝛷.
    The solution to ∇²𝛷 = 0, under a homogenous boundary
    condition 𝐁 * ∇ρ = 0, is 𝛷 = 0. To obtain a non-trivial solution,
    the boundary condition may be modified.
    Let 𝐁 = 𝐁₀ + ∇Φ.
    If 𝐁₀ ≠ 0 and satisfies ∇ × 𝐁₀ = 0, then ∇²Φ = 0 solved
    under an inhomogeneous boundary condition yields a non-trivial solution.
    If 𝐁₀ ≠ -∇Φ, then 𝐁 ≠ 0.

    Parameters
    ----------
    eq : Equilibrium
        Configuration with surface geometry defining ∂D.
    B0 : _MagneticField
        Magnetic field such that ∇ × 𝐁₀ = μ₀ 𝐉
        where 𝐉 is the current in amperes everywhere.
    eval_grid : Grid
        Evaluation points on D for the magnetic field.
    source_grid : Grid
        Source points on ∂D for quadrature of kernels.
        Resolution determines the accuracy of the boundary condition,
        and evaluation of the magnetic field.
    Phi_grid : Grid
        Source points on ∂D.
        Resolution determines accuracy of the spectral coefficients.
        Default is no symmetry.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    B : jnp.ndarray
        Magnetic field evaluated on ``eval_grid``.

    """
    if source_grid is None:
        source_grid = LinearGrid(
            rho=jnp.array([1.0]),
            M=eq.M_grid,
            N=eq.N_grid,
            NFP=eq.NFP if eq.N > 0 else 64,
        )
    B0n, _ = B0.compute_Bnormal(
        eq.surface,
        eval_grid=source_grid,
        source_grid=source_grid,
        vc_source_grid=source_grid,
        chunk_size=chunk_size,
    )
    Phi_mn, Phi_transform = compute_Phi_mn(
        eq, B0n, Phi_grid, source_grid=source_grid, chunk_size=chunk_size, check=check
    )
    # 𝐁 - 𝐁₀ = ∇Φ = 𝐁_vacuum in the interior.
    # Merkel eq. 1.4 is the Green's function solution to ∇²Φ = 0 in the interior.
    # Note that 𝐁₀′ in eq. 3.5 has the wrong sign.
    grad_Phi = FourierCurrentPotentialField.from_surface(
        eq.surface, Phi_mn / mu_0, Phi_transform.basis.modes[:, 1:]
    )
    data = eq.compute(["R", "phi", "Z"], grid=eval_grid)
    coords = jnp.column_stack([data["R"], data["phi"], data["Z"]])
    B = (B0 + grad_Phi).compute_magnetic_field(
        coords, source_grid=source_grid, chunk_size=chunk_size
    )
    return B


def compute_Phi_mn(
    eq,
    B0n,
    eval_grid=None,
    source_grid=None,
    chunk_size=None,
    check=False,
):
    """Compute Fourier coefficients of vacuum potential Φ on ∂D.

    Let D, D^∁ denote the interior, exterior of a toroidal region with
    boundary ∂D. Computes the magnetic field 𝐁 in units of Tesla such that

    - 𝐁 = 𝐁₀ + ∇Φ     on D
    - ∇ × 𝐁 = ∇ × 𝐁₀  on D ∪ D^∁ (i.e. ∇Φ is single-valued or periodic)
    - ∇ × 𝐁₀ = μ₀ 𝐉    on D ∪ D^∁
    - 𝐁 * ∇ρ = 0      on ∂D
    - ∇²Φ = 0         on D

    Parameters
    ----------
    eq : Equilibrium
        Configuration with surface geometry defining ∂D.
    B0n : MagneticField
        𝐁₀ * ∇ρ / |∇ρ| evaluated on ``source_grid`` of magnetic field
        such that ∇ × 𝐁₀ = μ₀ 𝐉 where 𝐉 is the current in amperes everywhere.
    eval_grid : Grid
        Evaluation points on ∂D.
        Resolution determines accuracy of the spectral coefficients of Φ.
    source_grid : Grid
        Source points on ∂D for quadrature of kernels.
        Resolution determines the accuracy of the boundary condition.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    Phi_mn, Phi_transform : jnp.ndarray, Transform
        Fourier coefficients of Φ on ∂D.

    """
    if eval_grid is None:
        eval_grid = LinearGrid(
            rho=jnp.array([1.0]),
            M=2 * eq.M,
            N=2 * eq.N,
            NFP=eq.NFP if eq.N > 0 else 64,
        )
    if source_grid is None:
        source_grid = eval_grid
    basis = DoubleFourierSeries(M=eval_grid.M, N=eval_grid.N, NFP=eq.NFP)

    names = ["R", "phi", "Z"]
    Phi_data = eq.compute(names, grid=eval_grid)
    src_data = eq.compute(
        names + ["n_rho", "|e_theta x e_zeta|", "e_theta", "e_zeta"], grid=source_grid
    )
    src_data["Bn"] = B0n
    src_transform = Transform(source_grid, basis)
    Phi_transform = Transform(eval_grid, basis)

    st, sz, q = heuristic_support_params(source_grid, best_ratio(src_data)[0])
    try:
        interpolator = FFTInterpolator(eval_grid, source_grid, st, sz, q)
    except AssertionError as e:
        warnif(
            True,
            msg="Could not build fft interpolator, switching to dft which is slow."
            "\nReason: " + str(e),
        )
        interpolator = DFTInterpolator(eval_grid, source_grid, st, sz, q)

    def LHS(Phi_mn):
        # After Fourier transform, the LHS is linear in the spectral coefficients Φₘₙ.
        # We approximate this as finite-dimensional, which enables writing the left
        # hand side as A @ Φₘₙ. Then Φₘₙ is found by solving LHS(Φₘₙ) = A @ Φₘₙ = RHS.
        src_data_2 = src_data.copy()
        src_data_2["Phi"] = src_transform.transform(Phi_mn)
        I = singular_integral(
            Phi_data,
            src_data_2,
            _kernel_Phi_dG_dn,
            interpolator,
            chunk_size,
        ).squeeze()
        Phi = Phi_transform.transform(Phi_mn)
        return Phi + I / (2 * jnp.pi)

    # LHS is expensive, so it is better to construct full Jacobian once
    # rather than iterative solves like jax.scipy.sparse.linalg.cg.
    A = jacfwd(LHS)(jnp.ones(basis.num_modes))
    RHS = -singular_integral(
        Phi_data,
        src_data,
        _kernel_Bn_over_r,
        interpolator,
        chunk_size,
    ).squeeze() / (2 * jnp.pi)
    # Fourier coefficients of Φ on boundary
    Phi_mn, _, _, _ = jnp.linalg.lstsq(A, RHS)
    if check:
        np.testing.assert_allclose(LHS(Phi_mn), A @ Phi_mn, atol=1e-6)
    return Phi_mn, Phi_transform


def compute_dPhi_dn(eq, eval_grid, source_grid, Phi_mn, basis, chunk_size=None):
    """Computes vacuum field ∇Φ ⋅ n on ∂D.

    Let D, D^∁ denote the interior, exterior of a toroidal region with
    boundary ∂D. Computes the magnetic field 𝐁 in units of Tesla such that

    - 𝐁 = 𝐁₀ + ∇Φ     on D
    - ∇ × 𝐁 = ∇ × 𝐁₀  on D ∪ D^∁ (i.e. ∇Φ is single-valued or periodic)
    - ∇ × 𝐁₀ = μ₀ 𝐉    on D ∪ D^∁
    - 𝐁 * ∇ρ = 0      on ∂D
    - ∇²Φ = 0         on D

    Parameters
    ----------
    eq : Equilibrium
        Configuration with surface geometry defining ∂D.
    eval_grid : Grid
        Evaluation points on ∂D.
    source_grid : Grid
        Source points on ∂D for quadrature of kernels.
    Phi_mn : jnp.ndarray
        Fourier coefficients of Φ on the boundary.
    basis : DoubleFourierSeries
        Basis for Φₘₙ.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    dPhi_dn : jnp.ndarray
        Shape (``eval_grid.grid.num_nodes``, 3).
        Vacuum field ∇Φ ⋅ n on ∂D.

    """
    names = ["R", "phi", "Z"]
    evl_data = eq.compute(names + ["n_rho"], grid=eval_grid)
    transform = Transform(source_grid, basis, derivs=1)
    src_data = {
        "Phi_t": transform.transform(Phi_mn, dt=1),
        "Phi_z": transform.transform(Phi_mn, dz=1),
    }
    src_data = eq.compute(
        names + ["|e_theta x e_zeta|", "e_theta", "e_zeta"],
        grid=source_grid,
        data=src_data,
    )
    src_data["K^theta"] = -src_data["Phi_z"] / src_data["|e_theta x e_zeta|"]
    src_data["K^zeta"] = src_data["Phi_t"] / src_data["|e_theta x e_zeta|"]
    src_data["K_vc"] = (
        src_data["K^theta"][:, jnp.newaxis] * src_data["e_theta"]
        + src_data["K^zeta"][:, jnp.newaxis] * src_data["e_zeta"]
    )

    st, sz, q = heuristic_support_params(source_grid, best_ratio(src_data)[0])
    try:
        interpolator = FFTInterpolator(eval_grid, source_grid, st, sz, q)
    except AssertionError as e:
        warnif(
            True,
            msg="Could not build fft interpolator, switching to dft which is slow."
            "\nReason: " + str(e),
        )
        interpolator = DFTInterpolator(eval_grid, source_grid, st, sz, q)

    # ∇Φ = ∂Φ/∂ρ ∇ρ + ∂Φ/∂θ ∇θ + ∂Φ/∂ζ ∇ζ
    # but we can not obtain ∂Φ/∂ρ from Φₘₙ. Biot-Savart gives
    # K_vc = n × ∇Φ where Φ has units Tesla-meters
    # ∇Φ(x ∈ ∂D) dot n = [1/2π ∫_∂D df' K_vc × ∇G(x,x')] dot n
    # (Same instructions but divide by 2 for x ∈ D).
    # Biot-Savart kernel assumes Φ in amperes, so we account for that.
    dPhi_dn = (2 / mu_0) * dot(
        singular_integral(
            evl_data, src_data, _kernel_biot_savart, interpolator, chunk_size
        ),
        evl_data["n_rho"],
    )
    return dPhi_dn


# alternative methods just to try checking for agreement


# TODO: surface integral correctness validation: should match output of compute_dPhi_dn.
def _dPhi_dn_triple_layer(
    eq,
    B0n,
    eval_grid,
    source_grid,
    Phi_mn,
    basis,
    chunk_size=None,
):
    """Compute ∇Φ ⋅ n on ∂D.

    Parameters
    ----------
    eq : Equilibrium
        Configuration with surface geometry defining ∂D.
    B0n : MagneticField
        𝐁₀ * ∇ρ / |∇ρ| evaluated on ``source_grid`` of magnetic field
        such that ∇ × 𝐁₀ = μ₀ 𝐉 where 𝐉 is the current in amperes everywhere.
    eval_grid : Grid
        Evaluation points on ∂D.
    source_grid : Grid
        Source points on ∂D for quadrature of kernels.
    Phi_mn : jnp.ndarray
        Fourier coefficients of Φ on the boundary.
    basis : DoubleFourierSeries
        Basis for Φₘₙ.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    dPhi_dn : jnp.ndarray
        Shape (``Phi_trans.grid.num_nodes``, ).
        ∇Φ ⋅ n on ∂D.

    """
    names = ["R", "phi", "Z", "n_rho"]
    evl_data = eq.compute(names, grid=eval_grid)
    src_data = eq.compute(
        names + ["|e_theta x e_zeta|", "e_theta", "e_zeta"], grid=source_grid
    )
    src_data["Bn"] = B0n

    st, sz, q = heuristic_support_params(source_grid, best_ratio(src_data)[0])
    try:
        interpolator = FFTInterpolator(eval_grid, source_grid, st, sz, q)
    except AssertionError as e:
        warnif(
            True,
            msg="Could not build fft interpolator, switching to dft which is slow."
            "\nReason: " + str(e),
        )
        interpolator = DFTInterpolator(eval_grid, source_grid, st, sz, q)

    I2 = -singular_integral(
        evl_data,
        src_data,
        _kernel_Bn_grad_G_dot_n,
        interpolator,
        chunk_size,
    )
    src_data["Phi"] = Transform(source_grid, basis).transform(Phi_mn)
    I1 = singular_integral(
        evl_data,
        src_data,
        # triple layer kernel may need more resolution
        _kernel_Phi_grad_dG_dn_dot_m,
        interpolator,
        chunk_size,
    )
    dPhi_dn = -I1 + I2
    return dPhi_dn


def _kernel_Phi_grad_dG_dn_dot_m(eval_data, source_data, diag=False):
    #   Phi(x') * grad dG(x,x')/dn' dot n
    # = Phi' * n' dot (grad(dx) / |dx|^3 - 3 dx transpose(dx) / |dx|^5) dot n
    # = Phi' * n' dot (n / |dx|^3 - 3 dx (dx dot n) / |dx|^5)
    # = Phi' * n' dot [n |dx|^2 - 3 dx (dx dot n)] / |dx|^5
    # where Phi has units tesla-meters.
    source_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([source_data["R"], source_data["phi"], source_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - source_x
    else:
        dx = eval_x[:, None] - source_x[None]
    # this is n'
    nn = rpz2xyz_vec(source_data["n_rho"], phi=source_data["phi"])
    # this is n
    n = rpz2xyz_vec(eval_data["n_rho"], phi=eval_data["phi"])
    dx_norm = safenorm(dx, axis=-1)
    return safediv(
        source_data["Phi"] * (dot(nn, n) * dx_norm**2 - 3 * dot(nn, dx) * dot(dx, n)),
        dx_norm**5,
    )


def _kernel_Bn_grad_G_dot_n(eval_data, source_data, diag=False):
    # Bn(x') * dG(x,x')/dn = - Bn * n dot dx / |dx|^3
    source_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([source_data["R"], source_data["phi"], source_data["Z"]]).T)
    )
    eval_x = jnp.atleast_2d(
        rpz2xyz(jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T)
    )
    if diag:
        dx = eval_x - source_x
    else:
        dx = eval_x[:, None] - source_x[None]
    n = rpz2xyz_vec(eval_data["n_rho"], phi=eval_data["phi"])
    return safediv(
        -source_data["Bn"] * dot(n, dx),
        safenorm(dx, axis=-1) ** 3,
    )


def compute_K_mn(eq, G, grid=None, check=False):
    """Compute Fourier coefficients of surface current on ∂D.

    Parameters
    ----------
    eq : Equilibrium
        Configuration with surface geometry defining ∂D.
    G : float
        Secular term of poloidal current in amperes.
        Should be ``2*np.pi/mu_0*data["G"]``.
    grid : Grid
        Points on ∂D.

    Returns
    -------
    K_mn, K_sec, K_transform : jnp.ndarray, Transform
        Fourier coefficients of surface current on ∂D.

    """
    if grid is None:
        grid = LinearGrid(
            rho=jnp.array([1.0]),
            # 3x higher than typical since we need to fit a vector
            M=6 * eq.M,
            N=6 * eq.N,
            NFP=eq.NFP if eq.N > 0 else 64,
            sym=False,
        )
    basis = DoubleFourierSeries(M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP)
    transform = Transform(grid, basis)
    K_sec = FourierCurrentPotentialField.from_surface(eq.surface, G=G)
    K_secular = K_sec.compute("K", grid=grid)["K"]
    n = eq.compute("n_rho", grid=grid)["n_rho"]

    def LHS(K_mn):
        # After Fourier transform, the LHS is linear in the spectral coefficients Kₘₙ.
        # We approximate this as finite-dimensional, which enables writing the left
        # hand side as A @ Kₘₙ. Then Kₘₙ is found by solving LHS(Kₘₙ) = A @ Kₘₙ = RHS.
        num_coef = K_mn.size // 3
        K_R = transform.transform(K_mn[:num_coef])
        K_phi = transform.transform(K_mn[num_coef : 2 * num_coef])
        K_Z = transform.transform(K_mn[2 * num_coef :])
        K_fourier = jnp.column_stack([K_R, K_phi, K_Z])
        return dot(K_fourier + K_secular, n)

    A = jacfwd(LHS)(jnp.ones(basis.num_modes * 3))
    K_mn, _, _, _ = jnp.linalg.lstsq(A, jnp.zeros(grid.num_nodes))
    if check:
        np.testing.assert_allclose(LHS(K_mn), A @ K_mn, atol=1e-6)
    return K_mn, K_sec, transform


def compute_B_dot_n_from_K(
    eq, eval_grid, source_grid, K_mn, K_sec, basis, chunk_size=None
):
    """Computes B ⋅ n on ∂D from surface current.

    Parameters
    ----------
    eq : Equilibrium
        Configuration with surface geometry defining ∂D.
    eval_grid : Grid
        Evaluation points on ∂D.
    source_grid : Grid
        Source points on ∂D for quadrature of kernels.
    K_mn : jnp.ndarray
        Fourier coefficients of surface current on ∂D.
    K_sec : FourierCurrentPotentialField
        Secular part of Fourier current potential field.
    basis : DoubleFourierSeries
        Basis for Kₘₙ.
    chunk_size : int or None
        Size to split computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.

    Returns
    -------
    B_dot_n : jnp.ndarray
        Shape (``eval_grid.grid.num_nodes``, 3).

    """
    names = ["R", "phi", "Z"]
    evl_data = eq.compute(names + ["n_rho"], grid=eval_grid)
    transform = Transform(source_grid, basis)
    src_data = eq.compute(
        names + ["|e_theta x e_zeta|", "e_theta", "e_zeta"], grid=source_grid
    )
    num_coef = K_mn.size // 3
    K_R = transform.transform(K_mn[:num_coef])
    K_phi = transform.transform(K_mn[num_coef : 2 * num_coef])
    K_Z = transform.transform(K_mn[2 * num_coef :])
    K_fourier = jnp.column_stack([K_R, K_phi, K_Z])
    K_secular = K_sec.compute("K", grid=source_grid)["K"]
    src_data["K_vc"] = K_fourier + K_secular

    st, sz, q = heuristic_support_params(source_grid, best_ratio(src_data)[0])
    try:
        interpolator = FFTInterpolator(eval_grid, source_grid, st, sz, q)
    except AssertionError as e:
        warnif(
            True,
            msg="Could not build fft interpolator, switching to dft which is slow."
            "\nReason: " + str(e),
        )
        interpolator = DFTInterpolator(eval_grid, source_grid, st, sz, q)

    # Biot-Savart gives K_vc = n × B
    # B(x ∈ ∂D) dot n = [μ₀/2π ∫_∂D df' K_vc × ∇G(x,x')] dot n
    # (Same instructions but divide by 2 for x ∈ D).
    Bn = 2 * dot(
        singular_integral(
            evl_data, src_data, _kernel_biot_savart, interpolator, chunk_size
        ),
        evl_data["n_rho"],
    )
    return Bn
