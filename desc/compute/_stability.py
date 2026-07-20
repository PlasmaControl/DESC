"""Compute functions for stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""
from functools import partial
import os
import time

import numpy as np
from jax.scipy.sparse.linalg import bicgstab, cg, gmres
try:
    from matfree import decomp, eig
except ModuleNotFoundError:
    decomp = None
    eig = None
from scipy.constants import mu_0
from scipy.sparse.linalg import LinearOperator, eigsh

from desc.backend import eigh_tridiagonal, jax, jit, jnp, scan

from ..integrals.surface_integral import surface_integrals_map
from ..utils import dot, safediv
from .data_index import register_compute_fun


def _agni_mem_trace_enabled(kwargs):
    flag = os.environ.get("AGNI_MEM_TRACE", "0").strip().lower()
    return bool(kwargs.get("debug_matfree", False)) or flag not in {"", "0", "false", "no", "off"}


def _agni_mem_trace(kwargs, *parts):
    if _agni_mem_trace_enabled(kwargs):
        print(*parts)


def _require_matfree_backend():
    if decomp is None or eig is None:
        raise ModuleNotFoundError(
            "matfree is required only for matfree_solver='shiftinvert_cg' or "
            "'shiftinvert_pcg'. "
            "Install matfree or use matfree_solver='eigsh_no_shiftinvert'."
        )


def _get_zernike_penalty(transforms, rt_size):
    """Return ``(alpha, Q_rt, rank)`` for a DiffMat-owned penalty."""
    diffmat = transforms.get("diffmat", None)
    if diffmat is None:
        return 0.0, None, None
    alpha = float(getattr(diffmat, "zernike_penalty_alpha", 0.0) or 0.0)
    if alpha <= 0.0:
        return alpha, None, None
    Q_rt = getattr(diffmat, "zernike_penalty_projector", None)
    if Q_rt is None:
        raise ValueError(
            "DiffMat has zernike_penalty_alpha > 0 but no "
            "zernike_penalty_projector. Rebuild DiffMat with coupled "
            "D_rho/D_theta matrices or pass a precomputed projector."
        )
    if tuple(Q_rt.shape) != (rt_size, rt_size):
        raise ValueError(
            "DiffMat zernike_penalty_projector shape does not match the "
            f"coupled_rt grid: got {Q_rt.shape}, expected {(rt_size, rt_size)}."
        )
    return alpha, Q_rt, getattr(diffmat, "zernike_penalty_rank", None)


def _assemble_diagblocks_comp_major(blocks, rho_idx, theta_idx, zeta_idx, sym=False):
    """Assemble a (3N, 3N) component-major matrix from (N, 3, 3) diagonal blocks.

    blocks: (N,3,3). Works for L (lower-tri) or B_blocks (symmetric).

    *_idx:  python slices for component-major ranges.

    NOTE that it currently only works for assembling lower diagonal
    matrices such as the ones formed by cholesky. Generalize logic later.

    This is used only by the Linv_full vs compact-Linv equivalence checks
    (the active compute paths reconstruct xi directly from the compact Linv);
    it is kept at module scope so the equivalence test in tests/ can import it.
    """
    N = blocks.shape[0]
    big = jnp.zeros((3 * N, 3 * N))

    # Diagonal sub-blocks
    big = big.at[rho_idx, rho_idx].set(jnp.diag(blocks[:, 0, 0]))
    big = big.at[theta_idx, theta_idx].set(jnp.diag(blocks[:, 1, 1]))
    big = big.at[zeta_idx, zeta_idx].set(jnp.diag(blocks[:, 2, 2]))

    # Off-diagonal (lower) subblocks — upper are zero for a Cholesky L anyway
    big = big.at[theta_idx, rho_idx].set(jnp.diag(blocks[:, 1, 0]))
    big = big.at[zeta_idx, rho_idx].set(jnp.diag(blocks[:, 2, 0]))
    big = big.at[zeta_idx, theta_idx].set(jnp.diag(blocks[:, 2, 1]))

    if sym:
        big = big.at[rho_idx, theta_idx].set(jnp.diag(blocks[:, 0, 1]))
        big = big.at[rho_idx, zeta_idx].set(jnp.diag(blocks[:, 0, 2]))
        big = big.at[theta_idx, zeta_idx].set(jnp.diag(blocks[:, 1, 2]))

    return big


@register_compute_fun(
    name="D_shear",
    label="D_{\\mathrm{shear}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion magnetic shear term",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["iota_psi"],
)
def _D_shear(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.16 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_shear"] = data["iota_psi"] ** 2 / (16 * jnp.pi**2)
    return data


@register_compute_fun(
    name="D_current",
    label="D_{\\mathrm{current}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion toroidal current term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "psi_r",
        "iota_psi",
        "B",
        "J",
        "G",
        "I_r",
        "|grad(psi)|",
        "|e_theta x e_zeta|",
    ],
    resolution_requirement="tz",
)
def _D_current(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.17 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    Xi = mu_0 * data["J"] - (data["I_r"] / data["psi_r"] * data["B"].T).T
    integrate = surface_integrals_map(transforms["grid"])
    data["D_current"] = (
        -jnp.sign(data["G"])
        / (2 * jnp.pi) ** 4
        * data["iota_psi"]
        * transforms["grid"].replace_at_axis(
            integrate(
                data["|e_theta x e_zeta|"]
                / data["|grad(psi)|"] ** 3
                * dot(Xi, data["B"])
            ),
            # TODO(#671): implement equivalent of equation 4.3 in desc coordinates
            jnp.nan,
        )
    )
    return data


@register_compute_fun(
    name="D_well",
    label="D_{\\mathrm{well}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion magnetic well term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "p_r",
        "psi",
        "psi_r",
        "psi_rr",
        "V_rr(r)",
        "V_r(r)",
        "|B|^2",
        "|grad(psi)|",
        "|e_theta x e_zeta|",
    ],
    resolution_requirement="tz",
)
def _D_well(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.18 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    integrate = surface_integrals_map(transforms["grid"])
    dp_dpsi = mu_0 * data["p_r"] / data["psi_r"]
    d2V_dpsi2 = (
        data["V_rr(r)"] * data["psi_r"] - data["V_r(r)"] * data["psi_rr"]
    ) / data["psi_r"] ** 3
    data["D_well"] = (
        dp_dpsi
        * (
            jnp.sign(data["psi"]) * d2V_dpsi2
            - dp_dpsi
            * integrate(
                data["|e_theta x e_zeta|"] / (data["|B|^2"] * data["|grad(psi)|"])
            )
        )
        * integrate(
            data["|e_theta x e_zeta|"] * data["|B|^2"] / data["|grad(psi)|"] ** 3
        )
        / (2 * jnp.pi) ** 6
    )
    # Axis limit does not exist as ∂ᵨ ψ and ‖∇ ψ‖ terms dominate so that D_well
    # is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="D_geodesic",
    label="D_{\\mathrm{geodesic}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion geodesic curvature term",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|grad(psi)|", "J*B", "|B|^2", "|e_theta x e_zeta|"],
    resolution_requirement="tz",
)
def _D_geodesic(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.19 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    integrate = surface_integrals_map(transforms["grid"])
    data["D_geodesic"] = transforms["grid"].replace_at_axis(
        (
            integrate(
                data["|e_theta x e_zeta|"]
                * mu_0
                * data["J*B"]
                / data["|grad(psi)|"] ** 3
            )
            ** 2
            - integrate(
                data["|e_theta x e_zeta|"] * data["|B|^2"] / data["|grad(psi)|"] ** 3
            )
            * integrate(
                data["|e_theta x e_zeta|"]
                * mu_0**2
                * data["J*B"] ** 2
                / (data["|B|^2"] * data["|grad(psi)|"] ** 3),
            )
        )
        / (2 * jnp.pi) ** 6,
        jnp.nan,  # enforce manually because our integration replaces nan with 0
    )
    # Axis limit does not exist as ‖∇ ψ‖ terms dominate so that D_geodesic
    # is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="D_Mercier",
    label="D_{\\mathrm{Mercier}}",
    units="Wb^{-2}",
    units_long="Inverse Webers squared",
    description="Mercier stability criterion (positive/negative value "
    + "denotes stability/instability)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["D_shear", "D_current", "D_well", "D_geodesic"],
)
def _D_Mercier(params, transforms, profiles, data, **kwargs):
    # Implements equation 4.20 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    data["D_Mercier"] = (
        data["D_shear"] + data["D_current"] + data["D_well"] + data["D_geodesic"]
    )
    # The axis limit does not exist as D_Mercier is of the order ρ⁻² near axis.
    return data


@register_compute_fun(
    name="magnetic well",
    label="\\mathrm{Magnetic~Well}",
    units="~",
    units_long="None",
    description="Magnetic well proxy for MHD stability (positive/negative value "
    + "denotes stability/instability)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["V(r)", "V_r(r)", "p_r", "<|B|^2>", "<|B|^2>_r"],
)
def _magnetic_well(params, transforms, profiles, data, **kwargs):
    # Implements equation 3.2 in M. Landreman & R. Jorge (2020)
    # doi:10.1017/S002237782000121X.
    # pressure = thermal + magnetic = 2 mu_0 p + |B|^2
    # The surface average operation is an additive homomorphism.
    # Thermal pressure is constant over a rho surface.
    # surface average(pressure) = thermal + surface average(magnetic)
    # The sign of sqrt(g) is enforced to be non-negative.
    data["magnetic well"] = transforms["grid"].replace_at_axis(
        safediv(
            data["V(r)"] * (2 * mu_0 * data["p_r"] + data["<|B|^2>_r"]),
            (data["V_r(r)"] * data["<|B|^2>"]),
        ),
        0,  # coefficient of limit is V_r / V_rr, rest is finite
    )
    return data


@register_compute_fun(
    name="gds2",
    # |∇(α + ι ζ₀ sign ι)|² ρ²
    label="\\vert \\nabla (\\alpha + "
    "\\iota \\zeta_0 \\mathrm{sign} \\iota) \\vert^2 \\rho^2",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Parameter in ideal ballooning equation",
    dim=2,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["rho", "g^aa", "g^rr", "g^ra", "shear", "iota"],
    zeta0="array: points of vanishing integrated local shear to scan over. "
    "Default 15 points linearly spaced in [-π/2,π/2]. "
    "The values ``zeta0`` correspond to values of ι ζ₀ and not ζ₀.",
    public=False,
)
def _gds2(params, transforms, profiles, data, **kwargs):
    zeta0 = kwargs.get("zeta0", jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, 15))
    zeta0 = zeta0.reshape(-1, 1)

    data["gds2"] = (
        data["g^aa"] * data["rho"] ** 2
        - 2
        * data["g^ra"]
        * data["rho"]
        * jnp.sign(data["iota"])
        * data["shear"]
        * zeta0
        + data["g^rr"] * data["shear"] ** 2 * zeta0**2
    )
    return data


@register_compute_fun(
    name="c ballooning",
    # c = 2 a³ Bₙ μ₀ sign(ψ) dp/dψ / (|B|² b⋅∇ζ) (b × 𝛋) ⋅ ∇(α + ι ζ₀) ρ²
    label="2 a^3 B_n \\mu_0 \\mathrm{sign}(\\psi) (\\partial_{\\psi} p) / "
    "(\\vert B \\vert^2 b \\cdot \\nabla ζ) (b \\times \\kappa) \\cdot "
    "\\nabla (\\alpha + \\iota \\zeta_0) \\rho^2",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=2,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "p_r", "psi", "psi_r", "B^zeta", "rho", "cvdrift", "cvdrift0", "shear"],
    zeta0="array: points of vanishing integrated local shear to scan over. "
    "Default 15 points linearly spaced in [-π/2,π/2]. "
    "The values ``zeta0`` correspond to values of ι ζ₀ and not ζ₀.",
)
def _c_balloon(params, transforms, profiles, data, **kwargs):
    """Dimensionless c sign(ψ) ρ².

    Where c mentioned immediately prior is defined in
    eq. 25b of arxiv.org/abs/2410.04576. Also α = α_{DESC} + ι ζ₀ here,
    consistent with above link.
    """
    zeta0 = kwargs.get("zeta0", jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, 15))
    zeta0 = zeta0.reshape(-1, 1)

    psi_boundary = params["Psi"] / (2 * jnp.pi)
    data["c ballooning"] = (
        (2 * psi_boundary * data["a"] * mu_0)  # a³ Bₙ μ₀
        * jnp.sign(data["psi"])
        * data["p_r"]
        / data["psi_r"]
        / data["B^zeta"]
        * (
            2 * data["rho"] ** 2 * data["cvdrift"]
            - data["cvdrift0"] * data["shear"] * zeta0
        )
    )
    return data


@register_compute_fun(
    name="f ballooning",
    # f = a Bₙ³ |B|⁻² / (B⋅∇ζ) |∇(α + ι ζ₀ sign ι)|² ρ²
    label="a B_n^3 \\vert B \\vert^{-2} / (B \\cdot \\nabla ζ) "
    "\\vert \\nabla (\\alpha + \\iota \\zeta_0 \\mathrm{sign} \\iota) \\vert^2 \\rho^2",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=2,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "|B|^2", "B^zeta", "gds2"],
)
def _f_balloon(params, transforms, profiles, data, **kwargs):
    """Dimensionless f ρ² where f is defined in eq. 25c of arxiv.org/abs/2410.04576.

    Also α = α_{DESC} + ι ζ₀ sign ι here whereas above link has α = α_{DESC} + ι ζ₀.
    """
    psi_boundary = params["Psi"] / (2 * jnp.pi)
    B_n = 2 * psi_boundary / data["a"] ** 2
    data["f ballooning"] = (
        data["a"] * B_n**3 / data["|B|^2"] / data["B^zeta"]
    ) * data["gds2"]
    return data


@register_compute_fun(
    name="g ballooning",
    # g = a³ Bₙ |B|⁻² (B⋅∇ζ) |∇(α + ι ζ₀ sign ι)|² ρ²
    label="a^3 B_n \\vert B \\vert^{-2} (B \\cdot \\nabla ζ) "
    "\\vert \\nabla (\\alpha + \\iota \\zeta_0 \\mathrm{sign} \\iota) \\vert^2 \\rho^2",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=2,
    params=["Psi"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["a", "|B|^2", "B^zeta", "gds2"],
)
def _g_balloon(params, transforms, profiles, data, **kwargs):
    """Dimensionless ρ² g where g is defined in eq. 25a of arxiv.org/abs/2410.04576.

    Also α = α_{DESC} + ι ζ₀ sign ι here whereas above link has α = α_{DESC} + ι ζ₀.
    """
    psi_boundary = params["Psi"] / (2 * jnp.pi)
    B_n = 2 * psi_boundary / data["a"] ** 2
    data["g ballooning"] = (
        data["a"] ** 3 * B_n * data["B^zeta"] / data["|B|^2"]
    ) * data["gds2"]
    return data


@register_compute_fun(
    name="ideal ballooning lambda",
    label="\\lambda_{\\mathrm{ballooning}}=\\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared ideal ballooning growth rate",
    dim=4,
    params=[],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=["c ballooning", "f ballooning", "g ballooning"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    Neigvals="int: number of largest eigenvalues to return, default value is 1.`"
    "If `Neigvals=2` eigenvalues are `[-1, 0, 1]` we get `[1, 0]`",
)
@partial(jit, static_argnames=["Neigvals"])
def _ideal_ballooning_lambda(params, transforms, profiles, data, **kwargs):
    """Eigenvalues of ideal-ballooning equation.

    A finite-difference method is used to calculate the maximum
    growth rate against the infinite-n ideal ballooning mode.
    The equation being solved is

    d/dζ(g dX/dζ) + c X = λ f X, g, f > 0

    where

      λ = a² / v_A² * γ²
    v_A = Bₙ / sqrt(μ₀ n₀ M) is the Alfven speed

    Returns
    -------
    Ideal-ballooning lambda eigenvalues
        Shape (num_rho, num alpha, num zeta0, num eigvals).

    """
    Neigvals = kwargs.get("Neigvals", 1)
    grid = transforms["grid"].source_grid

    num_zeta = grid.num_zeta
    num_zeta0 = data["c ballooning"].shape[0]

    def reshape(f):
        assert f.shape == (num_zeta0, grid.num_nodes)
        f = jnp.swapaxes(grid.meshgrid_reshape(f.T, "raz"), -1, -2)
        assert f.shape == (grid.num_rho, grid.num_alpha, num_zeta0, grid.num_zeta)
        return f

    c = reshape(data["c ballooning"])
    f = reshape(data["f ballooning"])
    g = reshape(data["g ballooning"])

    diffmat = transforms.get("diffmat", None)
    if diffmat is not None and diffmat.D_zeta is not None:

        # Check that the gradients of D_zeta are not calculated
        D_zeta = diffmat.D_zeta
        W_zeta = diffmat.W_zeta

        # W_zeta is purely diagonal for all the quadratures used
        # This will give wrong answers for a non-diagonal W_zeta
        w = jnp.diag(W_zeta)

        wg = -1 * w * g
        A = D_zeta.T @ (wg[..., :, None] * D_zeta)

        idx = jnp.arange(num_zeta)
        A = A.at[..., idx, idx].add(w * c)

        b_inv = jnp.sqrt(jnp.reciprocal(w * f))

        A = (b_inv[..., :, None] * A) * b_inv[..., None, :]

        # apply dirichlet BC to X
        w, v = jnp.linalg.eigh(A[..., 1:-1, 1:-1])

    else:
        # toroidal step size between points along field lines is assumed uniform
        dz = grid.nodes[grid.unique_zeta_idx[:2], 2]
        dz = dz[1] - dz[0]

        # Approximate derivative along field line with second order finite differencing.
        # Use g on the half grid for numerical stability.
        g_half = (g[..., 1:] + g[..., :-1]) / (2 * dz**2)
        b_inv = jnp.reciprocal(f[..., 1:-1])
        diag_inner = (c[..., 1:-1] - g_half[..., 1:] - g_half[..., :-1]) * b_inv
        diag_outer = g_half[..., 1:-1] * jnp.sqrt(b_inv[..., :-1] * b_inv[..., 1:])

        # TODO: Issue #1750
        w, v = eigh_tridiagonal(diag_inner, diag_outer)

    w, top_idx = jax.lax.top_k(w, k=Neigvals)
    assert w.shape == (grid.num_rho, grid.num_alpha, num_zeta0, Neigvals)
    data["ideal ballooning lambda"] = w

    # v becomes less than the machine precision at some points which gives NaNs.
    # stop_gradient prevents that.
    v = jax.lax.stop_gradient(v)
    v = jnp.take_along_axis(v, top_idx[..., jnp.newaxis, :], axis=-1)

    assert v.shape == (
        grid.num_rho,
        grid.num_alpha,
        num_zeta0,
        grid.num_zeta - 2,
        Neigvals,
    )
    data["ideal ballooning eigenfunction"] = v

    return data


@register_compute_fun(
    name="ideal ballooning eigenfunction",
    label="X_{\\mathrm{ballooning}}",
    units="~",
    units_long="None",
    description="Ideal ballooning eigenfunction",
    dim=5,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["ideal ballooning lambda"],
)
def _ideal_ballooning_eigenfunction(params, transforms, profiles, data, **kwargs):
    """Eigenfunctions of ideal-ballooning equation.

    Returns
    -------
    Ideal-ballooning lambda eigenfunctions
        Shape (num rho, num alpha, num zeta0, num zeta - 2, num eigvals).

    """
    return data  # noqa: unused dependency


@register_compute_fun(
    name="Newcomb ballooning metric",
    label="\\mathrm{Newcomb-ballooning-metric}",
    units="~",
    units_long="None",
    description="A measure of Newcomb's distance from marginal ballooning stability",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["c ballooning", "g ballooning"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
@jit
def _Newcomb_ball_metric(params, transforms, profiles, data, **kwargs):
    """Ideal-ballooning growth rate proxy.

    A finite-difference method is used to integrate the
    marginal stability ideal-ballooning equation

    d/dζ(g dX/dζ) + c X = 0, g > 0

    where

      λ = a² / v_A² * γ²
    v_A = Bₙ / sqrt(μ₀ n₀ M) is the Alfven speed

    The Newcomb's stability criterion is used.
    We define the Newcomb metric as follows:
    If zero crossing is at -inf (root finder failed), use the Y coordinate as a
    metric of stability. Otherwise use the zero-crossing point on the X-axis.
    This idea behind Newcomb's method is explained further in Appendix D of
    [Gaur _et al._](https://doi.org/10.1017/S0022377823000107).

    """
    grid = transforms["grid"].source_grid
    # toroidal step size between points along field lines is assumed uniform
    zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
    dz = zeta[1] - zeta[0]
    num_zeta0 = data["c ballooning"].shape[0]

    def reshape(f):
        assert f.shape == (num_zeta0, grid.num_nodes)
        f = jnp.moveaxis(grid.meshgrid_reshape(f.T, "raz"), -2, 0)
        assert f.shape == (grid.num_zeta, grid.num_rho, grid.num_alpha, num_zeta0)
        return f

    c = reshape(data["c ballooning"])[:-1]
    g = reshape(data["g ballooning"])

    def integrator(carry, x):
        """Update ``y`` and its derivative using leapfrog-like method.

        Assumed that y starts nonnegative with positive dy.

        Returns
        -------
        Cumulative integration of ``y`` and markers for the sign change.
        """
        y, dy = carry
        c, g = x
        y_new = y + dz * dy / g
        dy_new = dy - c * y_new * dz
        return (y_new, dy_new), (y_new, y_new < 0)

    dy_dz_initial = 5e-3
    _, (y, is_root) = scan(
        integrator,
        init=(jnp.zeros(c.shape[1:]), jnp.full(c.shape[1:], dy_dz_initial)),
        # Use g on the half grid for numerical stability.
        xs=(c, (g[1:] + g[:-1]) / 2),
    )

    idx_right_root = jnp.argmax(is_root.at[-1].set(True), axis=0, keepdims=True)
    y_left_root = jnp.take_along_axis(y, idx_right_root - 1, axis=0)
    # derivative of linear approximation of ζ ↦ y(ζ) near root
    dy_dz = (jnp.take_along_axis(y, idx_right_root, axis=0) - y_left_root) / dz

    # crossing from stable to unstable regime
    x = zeta[idx_right_root] - jnp.where(
        idx_right_root < (is_root.shape[0] - 1), y_left_root / dy_dz * dz, 0
    )
    # We take the signed distance X - ζ max < 0 as the distance to stability.
    # If there was no crossing we take y[ζ = ζ max] > 0.
    # This metric is only C0. Maybe think of something better?
    # RG: Peak of the metric does not match mean peak of the growth rate in ρ.
    data["Newcomb ballooning metric"] = (
        jnp.where(
            idx_right_root < (is_root.shape[0] - 1),
            (x - zeta[-1]) / (zeta[-1] - zeta[0]),
            y[-1],
        )
        .min((-1, -2))
        .squeeze(0)
    )
    return data




def _agni3_assemble(params, transforms, profiles, data, **kwargs):
    """Assemble the reduced, whitened dense finite-n lambda3 matrix ``A``.

    The kron assembly lifted verbatim out of ``_AGNI3``, stopping right after the
    keep-mask reduction and BEFORE any eigensolve, so the matrix can be had without
    committing to ARPACK. ``finite-n lambda3 rayleigh`` needs exactly that: it must
    eigensolve ``A(p)`` at the CURRENT p (on the host, since ARPACK cannot take
    tracers) and contract the resulting v against the same traced ``A(p)``.

    Both ``_AGNI3`` and that path call this, so the assembled operator has one
    definition and the two cannot drift apart.
    """
    """
    AGNI: Analysis of Global Normal-modes in Ideal MHD.

    Based on the original source here:
    https://github.com/rahulgaur104/AGNI/tree/master

    A finite-n stability eigenvalue solver.
    Currenly only finds fixed boundary unstable modes at
    low to medium resolution.

    This version of the code keeps the derivatives of the form
    partial_rho (iota psi' xi^rho) more compact which leads to
    fewer terms and even order derivatives. For this version
    the PSD version of A is actually very close to PSD ~ 1e-12.
    B is perfectly PSD
    
    The difference between this version and finite-n lambda is
    a variable transformation that matrix assembly significantly
    efficient. Moreover, we minimize the number of full matrices
    that are materialized. 

    A test compares the eigenvalue and eigenfunction of this version
    with finite-n lambda.
    """
    a_N = data["a"]
    B_N = abs(params["Psi"] / (jnp.pi * a_N**2))

    iota = data["iota"][:, None]
    iotainv = (1 / data["iota"])[:, None]

    psi_r = data["psi_r"][:, None] / (a_N**2 * B_N)
    psi_r0 = 1.0
    #psi_r0 = a_N**2 * B_N
    psi_r02 = psi_r0 ** 2

    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r2 = iota * psi_r2

    # Add a tiny shift because sometimes the pressure can be
    # slightly negative in the edge
    p0 = 1.0 * mu_0 * data["p"][:, None] /B_N**2 + 1e-12
    p_r = 1.0 * mu_0 * data["p_r"][:, None] /B_N**2

    axisym = kwargs.get("axisym", False)

    # Large gamma is an alternate way to impose incompressibility
    gamma = kwargs.get("gamma", 10.0)

    # For axisymmetric equilibria n_mode_axisym will decide the toroidal
    # mode number to analyze.
    n_mode_axisym = kwargs.get("n_mode_axisym", 1)
    incompressible = kwargs.get("incompressible", False)

    def _cT(x):
        return jnp.conjugate(jnp.transpose(x))

    if axisym:
        if n_mode_axisym == 0 and incompressible:
            return NotImplementedError
        else:
            # Each componenet of xi can be written as the Fourier sum of
            # two modes in the toroidal direction
            D_zeta0 = 1j * n_mode_axisym * jnp.array([[1]])
    else:
        D_zeta0 = transforms["diffmat"].D_zeta

    # Get differentiation matrices
    D_rho0 = transforms["diffmat"].D_rho
    D_theta0 = transforms["diffmat"].D_theta

    W_rho = transforms["diffmat"].W_rho
    W_theta = transforms["diffmat"].W_theta
    W_zeta = transforms["diffmat"].W_zeta

    # Square matrix.
    # When `coupled_rt` is set, D_rho0/D_theta0 are the full, non-separable 2D
    # (rho, theta) Zernike-Fourier operators of shape (n_rho*n_theta,)*2, so the
    # per-direction node counts must be supplied explicitly via kwargs.
    coupled_rt = kwargs.get("coupled_rt", False)
    if coupled_rt:
        n_rho_max = kwargs["n_rho_coupled"]
        n_theta_max = kwargs["n_theta_coupled"]
    else:
        n_rho_max = D_rho0.shape[0]
        n_theta_max = D_theta0.shape[0]
    n_zeta_max = D_zeta0.shape[0]

    def _reshape(u):
        return u.reshape(n_rho_max, n_theta_max, n_zeta_max)

    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    if coupled_rt:
        # D_rho0/D_theta0 already couple (rho, theta); only tensor with zeta.
        I_rt0 = jax.lax.stop_gradient(jnp.eye(n_rho_max * n_theta_max))
        D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, I_zeta0))
        D_theta = jax.lax.stop_gradient(jnp.kron(D_theta0, I_zeta0))
        D_zeta = jax.lax.stop_gradient(jnp.kron(I_rt0, D_zeta0))
        D_thetaT = jax.lax.stop_gradient(jnp.kron(_cT(D_theta0), I_zeta0))
        D_zetaT = jax.lax.stop_gradient(jnp.kron(I_rt0, _cT(D_zeta0)))
    else:
        I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
        I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
        D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
        D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
        D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))
        D_thetaT = jax.lax.stop_gradient(
            jnp.kron(I_rho0, jnp.kron(_cT(D_theta0), I_zeta0))
        )
        D_zetaT = jax.lax.stop_gradient(
            jnp.kron(I_rho0, jnp.kron(I_theta0, _cT(D_zeta0)))
        )

    # Quadrature weights still factorize (tensor-product) in both modes.
    W = jnp.kron(W_rho, jnp.kron(W_theta, W_zeta))[:, None]
    n_total = n_rho_max * n_theta_max * n_zeta_max

    # Arbitrary choice. Mostly used to decide the range of eigenvalues of
    # the mass matrix. Pre-conditioning should remove this factor
    n0 = jnp.asarray(kwargs.get("density", jnp.ones(n_total))).reshape(n_total, 1)

    # Define block indices
    rho_idx = slice(0, n_total)
    ups_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)

    ## Create the full matrix
    if axisym:
        A = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.complex128)
        B = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.complex128)
    else:
        A = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.float64)
        B = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.float64)


    sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3

    sqrtg_r = data["(sqrt(g)_PEST_r)|PEST"][:, None] * 1 / a_N**3
    sqrtg_v = data["(sqrt(g)_PEST_v)|PEST"][:, None] * 1 / a_N**3
    sqrtg_p = data["(sqrt(g)_PEST_p)|PEST"][:, None] * 1 / a_N**3

    partial_z_log_sqrtg = (sqrtg_p / sqrtg).flatten()
    partial_r_log_sqrtg = (sqrtg_r / sqrtg).flatten()
    partial_v_log_sqrtg = (sqrtg_v / sqrtg).flatten()

    psi_r_over_sqrtg = psi_r / sqrtg

    g_rr = data["g_rr|PEST"][:, None] * 1 / a_N**2
    g_vv = data["g_vv|PEST"][:, None] * 1 / a_N**2
    g_pp = data["g_pp|PEST"][:, None] * 1 / a_N**2  # finite on-axis

    g_rv = data["g_rv|PEST"][:, None] * 1 / a_N**2
    g_rp = data["g_rp|PEST"][:, None] * 1 / a_N**2
    g_vp = data["g_vp|PEST"][:, None] * 1 / a_N**2

    g_sup_rr = data["g^rr"][:, None] * a_N**2

    # Uses the identity g¹² = (g₁₃g₂₃ − g₁₂g₃₃)/(√g)² for the 3x3 metric tensor
    g_sup_rv_term = psi_r_over_sqrtg * (g_rp * g_vp - g_rv * g_pp)
    g_sup_rp_term = psi_r_over_sqrtg * (g_rv * g_vp - g_rp * g_vv)

    J2 = ((mu_0 * data["|J|"]) ** 2)[:, None] * (a_N / B_N) ** 2
    j_sup_zeta = mu_0 * data["J^zeta"][:, None] * a_N**2 / B_N
    # Can be obtained from ideal-MHD force balance
    j_sup_theta = iota * j_sup_zeta + p_r / psi_r

    # instability drive term. f_scale (default 1) temporarily amplifies the drive
    # so callers can isolate the physical unstable mode at f_scale>1, then continue
    # back to f_scale=1 using that eigenvalue/eigenfunction as sigma/v_guess.
    f_scale = kwargs.get("f_scale", 1.0)
    F = -1.0 * f_scale * mu_0 * data["finite-n instability drive"][:, None] * (1 / B_N) ** 2

    C_zeta = jnp.diag(partial_z_log_sqrtg) + D_zeta
    C_rho = jnp.diag(partial_r_log_sqrtg) + D_rho  # (n_total, n_total)
    C_theta = jnp.diag(partial_v_log_sqrtg) + D_theta

    ####################
    ####----Q²_ρρ----###
    ####################
    A = A.at[rho_idx, rho_idx].add(
        D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * psi_r02 * W * g_rr) * D_theta)
        + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * psi_r02 * g_rr) * D_zeta)
        + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * psi_r02 * W * g_rr) * D_zeta)
        + _cT((psi_r_over_sqrtg * iota * psi_r3 * psi_r02 * W * g_rr) * D_zeta) @ D_theta
    )

    ####################
    ####----Q²_ϑϑ ---###
    ####################
    # enforcing symmetry exactly
    A = A.at[ups_idx, ups_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + _cT((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta) @ D_zeta
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        +_cT(D_rho * iota_psi_r2.T)
        @ ((psi_r_over_sqrtg * psi_r02 * W * g_vv / psi_r) * (D_rho * iota_psi_r2.T))
    )

    A = A.at[rho_idx, ups_idx].add(
        -1 * _cT(D_rho * iota_psi_r2.T) @ ((psi_r_over_sqrtg * psi_r0 * W * g_vv) * D_zeta)
    )

    ####################
    ####----Q²_ζζ---####
    ####################
    A = A.at[ups_idx, ups_idx].add(
        0.5
        * (
            _cT(D_theta) @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + _cT((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta) @ D_theta
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        +_cT(D_rho * psi_r2.T)
        @ ((psi_r_over_sqrtg * psi_r02 * W * g_pp / psi_r) * (D_rho * psi_r2.T))
    )

    A = A.at[rho_idx, ups_idx].add(
        1 * _cT(D_rho * psi_r2.T) @ ((psi_r_over_sqrtg * psi_r0 * W * g_pp) * D_theta)
    )

    ####################
    ####----Q²_ρϑ----###
    ####################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            _cT(D_theta)
            @ ((iota * psi_r * psi_r02 * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            + _cT(D_zeta)
            @ ((psi_r * psi_r02 * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            _cT((iota * psi_r * psi_r02 * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            @ D_theta
            + _cT((psi_r * psi_r02 * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            @ D_zeta
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        _cT(D_theta) @ ((iota * psi_r2 * psi_r0 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        + _cT(D_zeta) @ ((psi_r2 * psi_r0 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
    )

    ######################
    ####-----Q²_ρζ-----###
    ######################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            _cT(D_theta)
            @ ((iota * psi_r * psi_r02 * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
            + _cT(D_zeta) @ ((psi_r * psi_r02 * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            _cT((iota * psi_r * psi_r02 * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
            @ D_theta
            + _cT((psi_r * psi_r02 * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T)) @ D_zeta
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        -1
        * (
            _cT(D_theta) @ ((iota * psi_r2 * psi_r0 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + _cT(D_zeta) @ ((psi_r2 * psi_r0 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )

    ##########################
    ######-----Q²_ϑζ-----#####
    ##########################
    A = A.at[ups_idx, ups_idx].add(
        -1
        * (
            _cT(D_zeta) @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + _cT((psi_r_over_sqrtg* W * psi_r * g_vp) * D_theta) @ D_zeta
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        -1
        * (
            _cT(D_rho * psi_r2.T) @ ((psi_r_over_sqrtg * psi_r0 * W * g_vp) * D_zeta)
            - _cT(D_rho * iota_psi_r2.T) @ ((psi_r_over_sqrtg * psi_r0 * W * g_vp) * D_theta)
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            _cT(D_rho * iota_psi_r2.T)
            @ ((psi_r_over_sqrtg * psi_r02 * W * g_vp / psi_r) * (D_rho * psi_r2.T))
        )
    )
    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            _cT((psi_r_over_sqrtg * psi_r02 * W * g_vp / psi_r) * (D_rho * psi_r2.T))
            @ (D_rho * iota_psi_r2.T)
        )
    )

    # Mixed Q-J term ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐
    # \xi^{\rho} (\mathbf{J} \times \nabla\rho)/|\nabla \rho|^2 \cdot \mathbf{Q}
    # Some algebra is performed to replace g_sup_rv and g_sup_rp
    A = A.at[rho_idx, rho_idx].add(
        -1.
        * (
            (
                W
                * psi_r2 * psi_r02
                * (j_sup_theta * g_sup_rp_term + j_sup_zeta * g_sup_rv_term)
                / g_sup_rr
            )
            * (iota * D_theta + D_zeta)
            + (W * sqrtg * psi_r * psi_r02 * j_sup_zeta) * (D_rho * iota_psi_r2.T)
            + (W * sqrtg * psi_r * psi_r02 * j_sup_theta) * (D_rho * psi_r2.T)
        )
    )

    # ρ-ρ block transposed for symmetry
    A = A.at[rho_idx, rho_idx].add(
        -1.
        * (
            _cT(
                (
                    W
                    * psi_r2
                    * psi_r02
                    * (j_sup_theta * g_sup_rp_term + j_sup_zeta * g_sup_rv_term)
                    / g_sup_rr
                )
                * (iota * D_theta + D_zeta)
            )
            + _cT((W * sqrtg * psi_r * psi_r02 * j_sup_zeta) * (D_rho * iota_psi_r2.T))
            + _cT((W * sqrtg * psi_r * psi_r02 * j_sup_theta) * (D_rho * psi_r2.T))
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        -(1.* W * psi_r2 * psi_r0 * sqrtg * j_sup_theta) * D_theta
        + (1. * W * psi_r2 * psi_r0 * sqrtg * j_sup_zeta) * D_zeta
    )

    ## diagonal |J|² term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((psi_r2 * psi_r02 * W * sqrtg * J2).flatten()))

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(jnp.diag((n0 * W * psi_r2 * psi_r02 * sqrtg * g_rr).flatten()))
    B = B.at[ups_idx, ups_idx].add(jnp.diag((n0 * W * sqrtg * g_vv).flatten()))

    B = B.at[rho_idx, ups_idx].add(
        jnp.diag((n0 * W * psi_r * psi_r0 * sqrtg * g_rv).flatten())
    )

    # typical in magnetic mirrors. `ismirror` is a TRACED bool (depends on iota), so a
    # Python `if ismirror` raises TracerBoolConversionError under jit (the assembly
    # runs concrete on the dense/callback paths but traced on the jax_lanczos path).
    # jnp.where selects the same values without a host-side branch. For a non-mirror
    # equilibrium (iota != 0, our case) iotainv is finite so both arms are finite and
    # `where` picks the non-mirror arm exactly as the old `else` did.
    ismirror = jnp.all(jnp.abs(iota) < 1e-12)

    zz = jnp.where(
        ismirror,
        n0 * W * sqrtg * g_pp,
        n0 * W * sqrtg * (g_vv + 2 * iotainv * g_vp + iotainv**2 * g_pp),
    )
    rz = jnp.where(
        ismirror,
        n0 * W * psi_r * psi_r0 * sqrtg * g_rp,
        n0 * W * psi_r * psi_r0 * sqrtg * (g_rv + iotainv * g_rp),
    )
    uz = jnp.where(
        ismirror,
        n0 * W * psi_r * sqrtg * g_vp,
        n0 * W * sqrtg * (g_vv + iotainv * g_vp),
    )
    B = B.at[zeta_idx, zeta_idx].add(jnp.diag(zz.flatten()))
    B = B.at[rho_idx, zeta_idx].add(jnp.diag(rz.flatten()))
    B = B.at[ups_idx, zeta_idx].add(jnp.diag(uz.flatten()))

    ##A = np.where(np.abs(A) >= 1e-11, 1.0, 0.0)
    #from matplotlib import pyplot as plt
    #plt.spy(A, precision=1e-11)
    #plt.savefig("test.png", dpi=400)

    # purely stabilizing and doesn't change the marginal stability
    A = A.at[rho_idx, rho_idx].add(
        _cT(C_rho * psi_r.T) @ ((gamma * sqrtg * W * psi_r02 * p0) * (C_rho * psi_r.T))
    )
    A = A.at[ups_idx, ups_idx].add(
        _cT(C_theta) @ ((gamma * sqrtg * W * p0) * C_theta)
    )
    A = A.at[rho_idx, ups_idx].add(
        _cT(C_rho * psi_r.T) @ ((gamma * sqrtg * W * psi_r0 * p0) * C_theta)
    )

    A = A.at[zeta_idx, zeta_idx].add(
        _cT(C_theta + C_zeta * iotainv.T)
        @ ((gamma * sqrtg * W * p0) * (C_theta + C_zeta * iotainv.T))
    )
    A = A.at[rho_idx, zeta_idx].add(
        _cT(C_rho * psi_r.T) @ ((gamma * sqrtg * W * psi_r0 * p0) * (C_theta + C_zeta * iotainv.T))
    )
    A = A.at[ups_idx, zeta_idx].add(
        _cT(C_theta) @ ((gamma * sqrtg * W * p0) * (C_theta + C_zeta * iotainv.T))
    )


    #### Instability drive term
    #Au = jnp.zeros((3 * n_total, 3 * n_total))
    #Au = Au.at[rho_idx, rho_idx].add(jnp.diag((W * psi_r2 * sqrtg * F).flatten()))
    au_diag = (W * psi_r2 * psi_r02 * sqrtg * F).flatten()

    rt_size = n_rho_max * n_theta_max
    zernike_penalty_alpha, Q_rt, penalty_rank = _get_zernike_penalty(
        transforms, rt_size
    )
    if coupled_rt and zernike_penalty_alpha > 0.0:
        Q = Q_rt if n_zeta_max == 1 else np.kron(Q_rt, np.eye(n_zeta_max))
        penalty = jnp.asarray(zernike_penalty_alpha * Q, dtype=A.dtype)
        A = A.at[rho_idx, rho_idx].add(penalty)
        A = A.at[ups_idx, ups_idx].add(penalty)
        A = A.at[zeta_idx, zeta_idx].add(penalty)
        rank_msg = "unknown" if penalty_rank is None else str(penalty_rank)
        #penalized_msg = (
        #    "unknown" if penalty_rank is None else str(rt_size - penalty_rank)
        #)
        #print(
        #    "[finite-n lambda3:coupled penalty]",
        #    f"alpha={zernike_penalty_alpha:.3e}",
        #    f"rank={rank_msg}/{rt_size}",
        #    f"penalized_rt={penalized_msg}",
        #    flush=True,
        #)

    A = A.at[ups_idx, rho_idx].set(_cT(A[rho_idx, ups_idx]))
    A = A.at[zeta_idx, rho_idx].set(_cT(A[rho_idx, zeta_idx]))
    A = A.at[zeta_idx, ups_idx].set(_cT(A[ups_idx, zeta_idx]))

    B = B.at[ups_idx, rho_idx].set(_cT(B[rho_idx, ups_idx]))
    B = B.at[zeta_idx, rho_idx].set(_cT(B[rho_idx, zeta_idx]))
    B = B.at[zeta_idx, ups_idx].set(_cT(B[ups_idx, zeta_idx]))


    d = 1 / jnp.sqrt(jnp.diag(B))  # 1D array

    # MEMORY (Step 1): the A whitening is DEFERRED to after B_blocks is extracted and B
    # is released -- see the optimization_barrier below. Doing it here holds
    # A_old + A_new + a broadcast transient WHILE B_old + B_new are also live: ~5 full
    # (3*n_total)^2 copies. Measured peak with it here 28.9 GB @24 / 51.5 GB @32;
    # deferred, 19.97 GB @24 / 34.51 GB @32 (jobs 56091564/65).
    au_diag = d[rho_idx] ** 2 * au_diag
    B = d[:, None] * B * d[None, :]

    # TODO: B_blocks will always be real for axisym=True, complex data type
    # is used to avoid trivial dtype-related errors. Fix later!
    if axisym:
        B_blocks = jnp.zeros((n_total, 3, 3), dtype=jnp.complex128)
        I3 = jnp.tile(jnp.eye(3, dtype=jnp.complex128), (n_total, 1, 1))
    else:
        B_blocks = jnp.zeros((n_total, 3, 3))
        I3 = jnp.tile(jnp.eye(3), (n_total, 1, 1))

    B_blocks = B_blocks.at[:, 0, 0].set(jnp.diag(B[rho_idx, rho_idx]))
    B_blocks = B_blocks.at[:, 1, 1].set(jnp.diag(B[ups_idx, ups_idx]))
    B_blocks = B_blocks.at[:, 2, 2].set(jnp.diag(B[zeta_idx, zeta_idx]))

    B_blocks = B_blocks.at[:, 0, 1].set(jnp.diag(B[rho_idx, ups_idx]))
    B_blocks = B_blocks.at[:, 1, 0].set(jnp.diag(B[ups_idx, rho_idx]))

    B_blocks = B_blocks.at[:, 2, 0].set(jnp.diag(B[rho_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 0, 2].set(jnp.diag(B[zeta_idx, rho_idx]))

    B_blocks = B_blocks.at[:, 1, 2].set(jnp.diag(B[ups_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 2, 1].set(jnp.diag(B[zeta_idx, ups_idx]))

    # B is DEAD here -- only the small (N,3,3) B_blocks is used downstream. Whiten A
    # NOW, after B can be released, so the two whitenings never overlap.
    # The barrier is LOAD-BEARING: under jit XLA schedules by DATAFLOW, not source
    # order, so merely moving these lines does not stop it interleaving the A whitening
    # with the B work and holding both.
    A, B_blocks, d = jax.lax.optimization_barrier((A, B_blocks, d))
    B = None
    A = d[:, None] * A * d[None, :]

    # Enforce physical ξ^ρ BC in the transformed per-node blocks
    # BEFORE taking the Schur complement
    n_per_shell = n_theta_max * n_zeta_max
    # NumPy (concrete), not jnp: n_total/n_per_shell/n_rho_max are all static ints, so
    # `boundary` is a compile-time-known mask. A concrete boolean index is jit-safe;
    # a traced one (jnp.arange) raises NonConcreteBooleanIndexError under the
    # jax_lanczos path where the assembly is traced.
    node_ids = np.arange(n_total)
    rho_shell = node_ids // n_per_shell
    boundary = np.asarray((rho_shell == 0) | (rho_shell == (n_rho_max - 1)))
    #boundary = (rho_shell == (n_rho_max - 1))

    # In lambda3 the local basis is (rho, upsilon, zeta)
    # so remove rho-upsilon and rho-zeta couplings on the boundary
    B_blocks = B_blocks.at[boundary, 0, 1].set(0)
    B_blocks = B_blocks.at[boundary, 1, 0].set(0)
    B_blocks = B_blocks.at[boundary, 0, 2].set(0)
    B_blocks = B_blocks.at[boundary, 2, 0].set(0)

    L = jnp.linalg.cholesky(B_blocks)  # (N,3,3)

    # Diagnostic: conditioning of the per-node 3x3 mass blocks. Near-singular
    # blocks make Linv ~ 1/sqrt(sigma_min) huge, which inflates lambda_max of
    # the whitened A and hence the eigh roundoff floor (eps*||A||). Cheap: N
    # 3x3 eigendecompositions.
    # Diagnostic only. np.asarray(B_blocks) + float()/int() concretize a TRACED array,
    # which raises under jit (jax_lanczos path). Skip when B_blocks is a tracer; the
    # dense/callback/eager paths still print it. Silent when AGNI_DIAG=0.
    # try/except is LOAD-BEARING, not defensive habit: this diagnostic CRASHED a
    # real optimization (job 56161838, UNFIX_K=4). np.linalg.eigvalsh raised
    # "Eigenvalues did not converge" on a B_blocks the optimizer had wandered into,
    # killing a run whose PHYSICS was fine -- the eigensolve never got to run. A
    # diagnostic must never be able to terminate the job it is reporting on. If the
    # condition number cannot be computed, say so and continue: an unprintable
    # B_blocks is itself the useful signal (it means the boundary went somewhere bad).
    if os.environ.get("AGNI_DIAG", "1") != "0" and not isinstance(
        B_blocks, jax.core.Tracer
    ):
        try:
            _bb = np.asarray(B_blocks)
            _n_bad = int(np.count_nonzero(~np.isfinite(_bb)))
            if _n_bad:
                raise ValueError(f"{_n_bad} non-finite entries")
            _bb_eigs = np.linalg.eigvalsh(_bb)
            _bb_min = _bb_eigs.min(axis=1)
            _bb_max = _bb_eigs.max(axis=1)
            _imin = int(np.argmin(_bb_min))
            _shell = _imin // (n_theta_max * n_zeta_max)
            print(
                f"[B_blocks cond] min sigma={float(_bb_min.min()):.3e} at node {_imin} "
                f"(rho_shell {_shell}/{n_rho_max-1}); "
                f"max 3x3 cond={float((_bb_max / np.clip(_bb_min, 1e-300, None)).max()):.3e}; "
                f"n(sigma<1e-3)={int(np.count_nonzero(_bb_min < 1e-3))}",
                flush=True,
            )
        except Exception as _exc:  # diagnostic only -- NEVER kill the run
            print(
                f"[B_blocks cond] UNAVAILABLE ({type(_exc).__name__}: {_exc}). "
                "Continuing -- this usually means the boundary moved somewhere "
                "degenerate; check the shape constraints.",
                flush=True,
            )

    Linv = jax.lax.linalg.triangular_solve(L, I3, left_side=True, lower=True)  # (N,3,3)

    def component_to_node_permutn(N: int) -> jnp.ndarray:
        """
        Build the permutation that converts component-major ordering to node-major.

        Component-major vector layout (length 3N):
            [ rho_1..N | theta_1..N | zeta_1..N ]

        Node-major vector layout (length 3N):
            [ rho_1, theta_1, zeta_1 | ... | rho_N, theta_N, zeta_N ]

        The returned permutation `p` satisfies:
            x_node = x_comp[p]
            M_node = M_comp[p][:, p]

        Parameters
        ----------
        N : int
            Number of spatial nodes per component.

        Returns
        -------
        jnp.ndarray, shape (3*N,)
            Permutation indices from component-major to node-major.
        """
        k = jnp.arange(N, dtype=jnp.int64)

        perm = jnp.empty(3 * N, dtype=jnp.int64)
        perm = perm.at[3 * k + 0].set(k)
        perm = perm.at[3 * k + 1].set(N + k)
        perm = perm.at[3 * k + 2].set(2 * N + k)

        return perm

    # components to node permutations
    p = component_to_node_permutn(n_total)
    A = A[p][:, p]

    # L^-1 A L^-T
    A = A.reshape(n_total, 3, n_total, 3)
    A = jnp.einsum("ikl,iljq,jbq->ikjb", Linv, A, Linv)

    node_idx = jnp.arange(n_total)

    # Add a constant shift to the diagonal of A (in the whitened L^-1 A L^-T
    # basis) for positive-definiteness, BEFORE adding the instability drive.
    # This uniformly shifts every eigenvalue by the constant.
    A = A.at[node_idx, :, node_idx, :].add(1e-14 * jnp.eye(3))

    # Add transformed instability-drive contribution without materializing Au.
    L0 = Linv[:, :, 0]
    au_node = au_diag[:, None, None] * L0[:, :, None] * L0[:, None, :]
    A = A.at[node_idx, :, node_idx, :].add(au_node)

    A = A.reshape(3 * n_total, 3 * n_total)

    # node to component permutation
    pinv = jnp.empty_like(p)
    pinv = pinv.at[p].set(jnp.arange(3 * n_total))

    A = A[pinv][:, pinv]

    # store indices needed to apply dirichlet BC to ξ^ρ
    n_shell = n_theta_max * n_zeta_max
    rho_start = n_shell
    rho_end = n_total - n_shell
    keep_1 = jnp.arange(rho_start, rho_end)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    A = A[jnp.ix_(keep, keep)]

    return {
        "A": A,
        "D_rho0": D_rho0,
        "D_theta0": D_theta0,
        "D_zeta0": D_zeta0,
        "Linv": Linv,
        "_reshape": _reshape,
        "coupled_rt": coupled_rt,
        "d": d,
        "g_pp": g_pp,
        "g_rp": g_rp,
        "g_rr": g_rr,
        "g_rv": g_rv,
        "g_vp": g_vp,
        "g_vv": g_vv,
        "iota": iota,
        "keep": keep,
        "n_rho_max": n_rho_max,
        "n_theta_max": n_theta_max,
        "n_total": n_total,
        "n_zeta_max": n_zeta_max,
        "psi_r": psi_r,
        "psi_r_over_sqrtg": psi_r_over_sqrtg,
        "rho_idx": rho_idx,
        "ups_idx": ups_idx,
        "zeta_idx": zeta_idx,
    }


@register_compute_fun(
    name="finite-n lambda3",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate"
    + "using the most compact representation of diffmatrices",
    dim=1,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "g^rr",
        "J^zeta",
        "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota",
        "psi_r",
        "psi_rr",
        "p",
        "p_r",
        "a",
    ],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    density="ndarray: the radial density profile",
    stable_only="bool: for testing only, materialize "
    + "and eigendecompose the stable part of the matrix",
    v_guess="ndarray: eigenfunction guess to initialize the "
    + "iterative eigenvalue solver",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier (rho, theta) "
    "operators instead of separable 1D matrices",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
    sigma="float: shift for the shift-invert eigensolver (default -0.1)",
    f_scale="float: multiplier on the instability drive F (default 1.0); use "
    ">1 to isolate the physical unstable mode, then continue back to 1",
    full_spectrum="bool: if True, dense-eigendecompose the full reduced matrix "
    "with jnp.linalg.eigh and store every eigenvalue under "
    "'finite-n lambda3 spectrum'; the returned dominant eigenmode is unchanged. "
    "Default False (iterative eigsh for the single dominant mode).",
)
def _AGNI3(params, transforms, profiles, data, **kwargs):
    """AGNI dense finite-n lambda3: assemble the matrix, then ARPACK eigsh."""
    _as = _agni3_assemble(params, transforms, profiles, data, **kwargs)
    A = _as["A"]
    D_rho0 = _as["D_rho0"]
    D_theta0 = _as["D_theta0"]
    D_zeta0 = _as["D_zeta0"]
    Linv = _as["Linv"]
    _reshape = _as["_reshape"]
    coupled_rt = _as["coupled_rt"]
    d = _as["d"]
    g_pp = _as["g_pp"]
    g_rp = _as["g_rp"]
    g_rr = _as["g_rr"]
    g_rv = _as["g_rv"]
    g_vp = _as["g_vp"]
    g_vv = _as["g_vv"]
    iota = _as["iota"]
    keep = _as["keep"]
    n_rho_max = _as["n_rho_max"]
    n_theta_max = _as["n_theta_max"]
    n_total = _as["n_total"]
    n_zeta_max = _as["n_zeta_max"]
    psi_r = _as["psi_r"]
    psi_r_over_sqrtg = _as["psi_r_over_sqrtg"]
    rho_idx = _as["rho_idx"]
    ups_idx = _as["ups_idx"]
    zeta_idx = _as["zeta_idx"]


    v0 = kwargs.get("v_guess", None)
    if v0 is not None:
        v0 = np.asarray(v0).reshape(-1)
        # `finite-n eigenfunction3` is stored full-length (3*n_total) with zeros in
        # the dropped Dirichlet slots, so feeding a previous solve's eigenfunction
        # straight back in arrives with the full length, not A's reduced length.
        # Restrict it to the kept DOF rather than silently discarding the warm start.
        if v0.size == 3 * n_total:
            v0 = v0[np.asarray(keep)]
        elif v0.size != A.shape[0]:
            print(
                f"finite-n lambda ignoring v_guess: got size={v0.size}, expected={A.shape[0]}"
            )
            v0 = None
    _agni_mem_trace(
        kwargs,
        "[finite-n lambda:dense] preparing scipy.eigsh",
        f"n_keep={A.shape[0]}",
        "converting A2 to NumPy",
    )

    
    sigma = kwargs.get("sigma", -1e-1)
    full_spectrum = kwargs.get("full_spectrum", False)
    if full_spectrum:
        # Dense symmetric/Hermitian eigendecomposition of the full reduced
        # matrix. eigh returns eigenvalues in ascending order, so the most
        # negative (most unstable) eigenpair is index 0. We keep the entire
        # spectrum for inspection but still hand only the dominant eigenmode
        # downstream, matching the default eigsh path.
        #
        # Use scipy.linalg.eigh (LAPACK syevr, O(N) workspace) rather than
        # jnp.linalg.eigh (syevd, ~2N^2 workspace): jax sizes the syevd buffer
        # with int32, so its eigh overflows for N > ~32768 on both CPU and GPU.
        # A itself has N^2 < 2^31 elements, so it converts to numpy fine.
        from scipy.linalg import eigh as _dense_eigh

        A_np = np.asarray(A)
        w_all, v_all = _dense_eigh(A_np, overwrite_a=True)
        data["finite-n lambda3 spectrum"] = w_all
        w = w_all[:1]
        v = v_all[:, :1]
    else:
        if v0 is None:
            w, v = eigsh(
                np.asarray(A),
                k=1,
                sigma=sigma,
                which="LM",
                tol=1e-8,
                return_eigenvectors=True,
            )
        else:
            w, v = eigsh(
                np.asarray(A),
                k=1,
                sigma=sigma,
                v0=v0,
                which="LM",
                tol=1e-8,
                return_eigenvectors=True,
            )
        data["finite-n lambda3 spectrum"] = np.asarray(w)

    idxs = jnp.where(jnp.abs(v) > 5e-5)[0]
    y = A @ v
    if os.environ.get("AGNI_DIAG","1")!="0": print(f"eigval res={jnp.linalg.norm(y[idxs]/v[idxs]-w)}")
    if os.environ.get("AGNI_DIAG","1")!="0": print(f"eigenvalue: {w}")


    # Reduced eigenvector -> full component-major vector [rho,theta,zeta].
    v_mode = v[:, 0] if jnp.ndim(v) == 2 else v
    v_full = jnp.zeros(3 * n_total, dtype=v_mode.dtype).at[keep].set(v_mode)

    if coupled_rt:
        # In coupled mode D is the full (n_rho*n_theta) operator: flatten the
        # (rho, theta) axes, apply, reshape back. d_dz stays separable in zeta.
        def d_dr(D, u):
            """Radial derivative via the coupled (rho, theta) operator."""
            U = u.reshape(n_rho_max * n_theta_max, n_zeta_max)
            return (D @ U).reshape(n_rho_max, n_theta_max, n_zeta_max)

        def d_dv(D, u):
            """Poloidal derivative via the coupled (rho, theta) operator."""
            U = u.reshape(n_rho_max * n_theta_max, n_zeta_max)
            return (D @ U).reshape(n_rho_max, n_theta_max, n_zeta_max)

        def d_dz(D, u):
            return jnp.einsum("ij,klj->kli", D, u)
    else:
        def d_dr(D, u):
            """Calculate the radial derivative of u."""
            return jnp.einsum("ij,jkl->ikl", D, u)  # (Nr, Nθ, Nζ)

        def d_dv(D, u):
            return jnp.einsum("ij,kjl->kil", D, u)

        def d_dz(D, u):
            return jnp.einsum("ij,klj->kli", D, u)

    psi_r_over_sqrtg = _reshape(psi_r_over_sqrtg)
    psi_r = _reshape(psi_r)
    iota = _reshape(iota)
    g_rr = _reshape(g_rr)
    g_vv = _reshape(g_vv)
    g_pp = _reshape(g_pp)
    g_rv = _reshape(g_rv)
    g_rp = _reshape(g_rp)
    g_vp = _reshape(g_vp)

    # Reconstruct the physical displacement directly from the compact Linv blocks
    # (equivalent to d * (LinvT_full @ v_full); the equivalence is checked in tests/).
    vr, vv, vz = v_full[rho_idx], v_full[ups_idx], v_full[zeta_idx]
    xi_full = jnp.concatenate([
        d[rho_idx]  * (Linv[:, 0, 0] * vr + Linv[:, 1, 0] * vv + Linv[:, 2, 0] * vz),
        d[ups_idx]  * (                     Linv[:, 1, 1] * vv + Linv[:, 2, 1] * vz),
        d[zeta_idx] * (                                          Linv[:, 2, 2] * vz),
    ])

    # Phase rotation doesn't change the physics. Here, we use it to make the eigenmode up-down symmetric.
    # phase_offset (default 0) is an optional tunable rotation applied on top of the mean-based alignment.
    phase_offset = kwargs.get("phase_offset", 0.0)
    xi_ref = xi_full[rho_idx]
    phase_angle = jnp.arctan2(jnp.mean(xi_ref.real), jnp.mean(xi_ref.imag))
    per_elem_angles = jnp.arctan2(xi_ref.real, xi_ref.imag)
    angle_diff = per_elem_angles - phase_angle
    mags = jnp.abs(xi_ref)
    threshold = 0.01 * jnp.max(mags)
    mask = mags > threshold
    angle_diff_valid = jnp.where(mask, jnp.abs(angle_diff), jnp.nan)
    if os.environ.get("AGNI_DIAG","1")!="0": print(f"phase_angle (mean-based): {phase_angle:.4f} rad  |  per-elem deviation (all): max={jnp.max(jnp.abs(angle_diff)):.4f}, mean={jnp.mean(jnp.abs(angle_diff)):.4f}, std={jnp.std(angle_diff):.4f} rad")
    if os.environ.get("AGNI_DIAG","1")!="0": print(f"  deviation (|xi|>1% max, n={int(jnp.sum(mask))}/{xi_ref.size}): max={float(jnp.nanmax(angle_diff_valid)):.4f}, mean={float(jnp.nanmean(angle_diff_valid)):.4f} rad")
    xr = (xi_full[rho_idx].reshape(n_rho_max, n_theta_max, n_zeta_max)*jnp.exp(1j * (phase_angle + phase_offset))).imag
    xv = (xi_full[ups_idx].reshape(n_rho_max, n_theta_max, n_zeta_max)*jnp.exp(1j * (phase_angle + phase_offset))).imag
    xz = (xi_full[zeta_idx].reshape(n_rho_max, n_theta_max, n_zeta_max)*jnp.exp(1j * (phase_angle + phase_offset))).imag

    # precomputed forward derivatives (re-used below)
    xr_v = d_dv(D_theta0, xr)
    xr_z = d_dz(D_zeta0, xr)

    xv_v = d_dv(D_theta0, xv+xz)
    xv_z = d_dz(D_zeta0, xv+xz)

    xz_v = d_dv(D_theta0, xz/iota)
    xz_z = d_dz(D_zeta0, xz/iota)

    test_v = d_dv(D_theta0, xv)
    test_z = d_dz(D_zeta0, xv)

    # combos used many times
    xr_r = d_dr(D_rho0,  xr)  # dρ(ι ψ′² xr)
    psi_rr = d_dr(D_rho0,  psi_r)  # dρ(ι ψ′² xr)
    iota_r = d_dr(D_rho0,  iota)  # dρ(ι ψ′² xr)

    if os.environ.get("AGNI_DIAG","1")!="0": print(f"xr_v shape: {xr_v.shape}, xv_z shape: {xv_z.shape}, xz_z shape: {xz_z.shape}, xr_r shape: {xr_r.shape}, psi_r shape: {psi_r.shape}, psi_rr shape: {psi_rr.shape}, iota_r shape: {iota_r.shape}")

    deltaB_r = psi_r_over_sqrtg * psi_r * (iota * xr_v + xr_z)
    deltaB_v = psi_r_over_sqrtg * (1.* (test_z) - 1.*(xr_r * iota *psi_r + (2 * iota * psi_rr + iota_r * psi_r)* xr))
    deltaB_z = -psi_r_over_sqrtg * (1.* (test_v) + 1.*(xr_r * psi_r + 2 * psi_rr * xr))

    deltaV_r = psi_r * xr
    deltaV_v = xv + xz 
    deltaV_z = xz * 1/iota

    if os.environ.get("AGNI_DIAG","1")!="0": print(f"deltaB_r shape: {deltaB_r.shape}, deltaB_v shape: {deltaB_v.shape}, deltaB_z shape: {deltaB_z.shape}")

    deltaB2 = g_rr * deltaB_r ** 2 + 1.*g_vv * deltaB_v ** 2  + g_pp * deltaB_z ** 2 + 2. * (g_rv * deltaB_r * deltaB_v + g_rp * deltaB_r * deltaB_z +  g_vp * deltaB_v * deltaB_z)
    deltaV2 = g_rr * deltaV_r ** 2 + 1.*g_vv * deltaV_v ** 2  + g_pp * deltaV_z ** 2 + 2. * (g_rv * deltaV_r * deltaV_v + g_rp * deltaV_r * deltaV_z +  g_vp * deltaV_v * deltaV_z)

    data["finite-n lambda3"] = w
    data["finite-n eigenfunction3"] = v_full
    data["finite-n xi"] = xi_full
    data["finite-n deltaB"] = np.sqrt(deltaB2)
    data["finite-n deltaV"] = np.sqrt(deltaV2)
    data["finite-n deltaB_r"] = deltaB_r
    data["finite-n deltaB_v"] = deltaB_v
    data["finite-n deltaB_z"] = deltaB_z
    data["finite-n deltaV_r"] = deltaV_r
    data["finite-n deltaV_v"] = deltaV_v
    data["finite-n deltaV_z"] = deltaV_z

    return data


@register_compute_fun(
    name="finite-n eigenfunction3",
    label="\\xi",
    units="~",
    units_long="None",
    description="Finite-n eigenfunction",
    dim=5,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=["finite-n lambda3"],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    stable_only="bool: for testing only, materialize "
    + "and eigendecompose the stable part of the matrix",
    v_guess="ndarray: eigenfunction guess to initialize the "
    + "iterative eigenvalue solver",
    density="ndarray: the radial density profile",
)
def _AGNI_eigenfunction3(params, transforms, profiles, data, **kwargs):
    """Eigenfunctions of finite-n stability solver.

    Returns
    -------
    Finite-n lambda eigenfunctions
        Shape (num_eigenvalues, num rho, num theta, num zeta, 3).

    """
    return data  # noqa: unused dependency



def _agni3_matfree_operator(params, transforms, profiles, data, **kwargs):
    """Build the reduced, whitened finite-n lambda3 operator ``Ax``.

    This is the operator construction lifted verbatim out of
    ``_AGNI3_matfree`` so that it has exactly one definition. Both the
    matrix-free eigensolver (``finite-n lambda3 matfree``) and the
    fixed-vector Rayleigh quotient (``finite-n lambda3 rayleigh``) call
    this, so the two can never drift apart.

    Returns a dict of the operator plus the intermediates the callers need.
    """
    a_N = data["a"]
    B_N = abs(params["Psi"] / (jnp.pi * a_N**2))

    axisym = kwargs.get("axisym", False)
    gamma = kwargs.get("gamma", 10.0)
    n_mode_axisym = kwargs.get("n_mode_axisym", 1)
    incompressible = kwargs.get("incompressible", False)

    def _cT(x):
        return jnp.conjugate(jnp.transpose(x))

    if axisym:
        if n_mode_axisym == 0 and incompressible:
            return NotImplementedError
        D_zeta0 = 1j * n_mode_axisym * jnp.array([[1]])
    else:
        D_zeta0 = transforms["diffmat"].D_zeta

    D_rho0 = transforms["diffmat"].D_rho
    D_theta0 = transforms["diffmat"].D_theta
    w_rho = transforms["diffmat"].W_rho
    w_theta = transforms["diffmat"].W_theta
    w_zeta = transforms["diffmat"].W_zeta

    # When `coupled_rt` is set, D_rho0/D_theta0 are the full, non-separable 2D
    # (rho, theta) Zernike-Fourier operators of shape (n_rho*n_theta,)*2, so the
    # per-direction node counts must be supplied explicitly via kwargs.
    coupled_rt = kwargs.get("coupled_rt", False)
    if coupled_rt:
        n_rho = kwargs["n_rho_coupled"]
        n_theta = kwargs["n_theta_coupled"]
    else:
        n_rho = D_rho0.shape[0]
        n_theta = D_theta0.shape[0]
    n_zeta = D_zeta0.shape[0]
    n_total = n_rho * n_theta * n_zeta


    def _reshape(u):
        return u.reshape(n_rho, n_theta, n_zeta)

    n0 = _reshape(kwargs.get("density", np.ones((n_rho, n_theta, n_zeta))))
    W = _reshape(jnp.kron(w_rho, jnp.kron(w_theta, w_zeta)))

    iota = _reshape(data["iota"])
    iotainv = _reshape(1.0 / data["iota"])
    psi_r = _reshape(data["psi_r"]) / (a_N**2 * B_N)
    psi_r2 = psi_r**2
    psi_r3 = psi_r**3
    iota_psi_r2 = iota * psi_r2

    p0 = _reshape(mu_0 * data["p"][:, None] / B_N**2 + 1e-12)
    p_r = _reshape(mu_0 * data["p_r"][:, None] / B_N**2)

    sqrtg = _reshape(data["sqrt(g)_PEST"]) / a_N**3
    sqrtg_r = _reshape(data["(sqrt(g)_PEST_r)|PEST"]) / a_N**3
    sqrtg_v = _reshape(data["(sqrt(g)_PEST_v)|PEST"]) / a_N**3
    sqrtg_p = _reshape(data["(sqrt(g)_PEST_p)|PEST"]) / a_N**3

    partial_r_log_sqrtg = sqrtg_r / sqrtg
    partial_v_log_sqrtg = sqrtg_v / sqrtg
    partial_p_log_sqrtg = sqrtg_p / sqrtg

    psi_r_over_sqrtg = psi_r / sqrtg

    g_rr = _reshape(data["g_rr|PEST"]) / a_N**2
    g_vv = _reshape(data["g_vv|PEST"]) / a_N**2
    g_pp = _reshape(data["g_pp|PEST"]) / a_N**2
    g_rv = _reshape(data["g_rv|PEST"]) / a_N**2
    g_rp = _reshape(data["g_rp|PEST"]) / a_N**2
    g_vp = _reshape(data["g_vp|PEST"]) / a_N**2

    g_sup_rr = _reshape(data["g^rr"]) * a_N**2
    g_sup_rv = _reshape(data["g^rv"]) * a_N**2
    g_sup_rp = _reshape(data["g^rz"]) * a_N**2

    # Match _agni3_assemble's route to g^rv/g^rz exactly: build them from the PEST
    # lower metric via g¹² = (g₁₃g₂₃ - g₁₂g₃₃)/(√g)², rather than reading data["g^rv"].
    # These absorb a psi_r*sqrtg: g_sup_rv_term == psi_r * sqrtg * g^rv.
    jq_lower_metric = kwargs.get("jq_lower_metric", True)
    g_sup_rv_term = psi_r_over_sqrtg * (g_rp * g_vp - g_rv * g_pp)
    g_sup_rp_term = psi_r_over_sqrtg * (g_rv * g_vp - g_rp * g_vv)

    J2 = _reshape((mu_0 * data["|J|"]) ** 2) * (a_N / B_N) ** 2
    j_sup_zeta = mu_0 * _reshape(data["J^zeta"]) * a_N**2 / B_N
    j_sup_theta = iota * j_sup_zeta + p_r / psi_r

    F = -mu_0 * _reshape(data["finite-n instability drive"]) / B_N**2

    if coupled_rt:
        # D is the full (n_rho*n_theta) coupled (rho, theta) operator: flatten the
        # (rho, theta) axes, apply, reshape back. d_dz stays separable in zeta.
        def d_dr(D, u):
            """Radial derivative via the coupled (rho, theta) operator."""
            U = u.reshape(n_rho * n_theta, n_zeta)
            return (D @ U).reshape(n_rho, n_theta, n_zeta)

        def d_dv(D, u):
            """Poloidal derivative via the coupled (rho, theta) operator."""
            U = u.reshape(n_rho * n_theta, n_zeta)
            return (D @ U).reshape(n_rho, n_theta, n_zeta)

        def d_dz(D, u):
            return jnp.einsum("ij,klj->kli", D, u)
    else:
        def d_dr(D, u):
            return jnp.einsum("ij,jkl->ikl", D, u)

        def d_dv(D, u):
            return jnp.einsum("ij,kjl->kil", D, u)

        def d_dz(D, u):
            return jnp.einsum("ij,klj->kli", D, u)

    if axisym:
        B_blocks = jnp.zeros((n_total, 3, 3), dtype=jnp.complex128)
    else:
        B_blocks = jnp.zeros((n_total, 3, 3))

    B_blocks = B_blocks.at[:, 0, 0].set((n0 * W * psi_r2 * sqrtg * g_rr).flatten())
    B_blocks = B_blocks.at[:, 1, 1].set((n0 * W * sqrtg * g_vv).flatten())
    B_blocks = B_blocks.at[:, 0, 1].set((n0 * W * psi_r * sqrtg * g_rv).flatten())
    B_blocks = B_blocks.at[:, 1, 0].set((n0 * W * psi_r * sqrtg * g_rv).flatten())

    ismirror = bool(kwargs.get("mirror", False))
    if ismirror:
        B_blocks = B_blocks.at[:, 2, 2].set((n0 * W * sqrtg * g_pp).flatten())
        B_blocks = B_blocks.at[:, 0, 2].set((n0 * W * psi_r * sqrtg * g_rp).flatten())
        B_blocks = B_blocks.at[:, 2, 0].set((n0 * W * psi_r * sqrtg * g_rp).flatten())
        B_blocks = B_blocks.at[:, 1, 2].set((n0 * W * sqrtg * g_vp).flatten())
        B_blocks = B_blocks.at[:, 2, 1].set((n0 * W * sqrtg * g_vp).flatten())
    else:
        B_blocks = B_blocks.at[:, 2, 2].set(
            (n0 * W * sqrtg * (g_vv + 2.0 * iotainv * g_vp + iotainv**2 * g_pp)).flatten()
        )
        B_blocks = B_blocks.at[:, 0, 2].set(
            (n0 * W * psi_r * sqrtg * (g_rv + iotainv * g_rp)).flatten()
        )
        B_blocks = B_blocks.at[:, 2, 0].set(
            (n0 * W * psi_r * sqrtg * (g_rv + iotainv * g_rp)).flatten()
        )
        B_blocks = B_blocks.at[:, 1, 2].set(
            (n0 * W * sqrtg * (g_vv + iotainv * g_vp)).flatten()
        )
        B_blocks = B_blocks.at[:, 2, 1].set(
            (n0 * W * sqrtg * (g_vv + iotainv * g_vp)).flatten()
        )

    diagBsqinv = 1.0 / jnp.sqrt(
        jnp.stack((B_blocks[:, 0, 0], B_blocks[:, 1, 1], B_blocks[:, 2, 2]), axis=1)
    )
    B_scaled = jnp.einsum("...ij,...i,...j->...ij", B_blocks, diagBsqinv, diagBsqinv)

    n_per_shell = n_theta * n_zeta
    boundary_idx = jnp.concatenate(
        [jnp.arange(n_per_shell), jnp.arange(n_total - n_per_shell, n_total)]
    )
    keep_rho = jnp.arange(n_per_shell, n_total - n_per_shell)
    keep_tangent = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_rho, keep_tangent])
    n_keep = keep.size

    B_scaled = B_scaled.at[boundary_idx, 0, 1].set(0)
    B_scaled = B_scaled.at[boundary_idx, 1, 0].set(0)
    B_scaled = B_scaled.at[boundary_idx, 0, 2].set(0)
    B_scaled = B_scaled.at[boundary_idx, 2, 0].set(0)

    L_D = jnp.linalg.cholesky(B_scaled)
    I3 = jnp.broadcast_to(jnp.eye(3, dtype=L_D.dtype), L_D.shape)
    Linv_D = jax.lax.linalg.triangular_solve(L_D, I3, left_side=True, lower=True)
    Linv_DT = jnp.swapaxes(Linv_D, -1, -2)

    # Coupled_rt de-aliasing penalty, matrix-free mirror of the dense
    # ``finite-n lambda3`` path. The expensive SVD projector is owned by
    # DiffMat and is only applied here.
    rt_size = n_rho * n_theta
    if coupled_rt:
        zernike_penalty_alpha, Q_rt, penalty_rank = _get_zernike_penalty(
            transforms, rt_size
        )
        apply_penalty = zernike_penalty_alpha > 0.0
    else:
        zernike_penalty_alpha, Q_rt, penalty_rank = 0.0, None, None
        apply_penalty = False
    if apply_penalty:
        # Fold alpha into the projector so a single matmul applies the penalty.
        alphaQ_rt = jnp.asarray(zernike_penalty_alpha * Q_rt)
        if os.environ.get("AGNI_DIAG", "1") != "0":
            rank_msg = "unknown" if penalty_rank is None else str(penalty_rank)
            penalized_msg = (
                "unknown" if penalty_rank is None else str(rt_size - penalty_rank)
            )
            print(
                "[finite-n lambda3 matfree:coupled penalty]",
                f"alpha={zernike_penalty_alpha:.3e}",
                f"rank={rank_msg}/{rt_size}",
                f"penalized_rt={penalized_msg}",
                flush=True,
            )

        def _apply_penalty(u):
            # Q = kron(Q_rt, I_zeta) acting on rho-major (rho, theta, zeta) data.
            U = u.reshape(rt_size, n_zeta)
            return (alphaQ_rt @ U).reshape(n_rho, n_theta, n_zeta)

    def Ax_full(x_flat):
        x = jnp.transpose(x_flat.reshape(3, n_total), axes=(1, 0))
        x = diagBsqinv * jnp.einsum("lij,lj->li", Linv_DT, x)
        x = x.reshape((n_rho, n_theta, n_zeta, 3))

        xr = x[..., 0]
        xu = x[..., 1]
        xz = x[..., 2]

        xr_v = d_dv(D_theta0, xr)
        xr_z = d_dz(D_zeta0, xr)
        xu_v = d_dv(D_theta0, xu)
        xu_z = d_dz(D_zeta0, xu)
        xz_v = d_dv(D_theta0, xz)
        xz_z = d_dz(D_zeta0, xz)

        xr1_r = d_dr(D_rho0, iota_psi_r2 * xr)
        xr2_r = d_dr(D_rho0, psi_r2 * xr)
        xr3_r = d_dr(D_rho0, psi_r * xr)

        Ar = jnp.zeros((n_rho, n_theta, n_zeta), dtype=xr.dtype)
        Au = jnp.zeros_like(Ar)
        Az = jnp.zeros_like(Ar)
        Aur = jnp.zeros_like(Ar)

        # Q^2_rr
        Ar += (
            d_dv(_cT(D_theta0), (psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * xr_v)
            + d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * psi_r3 * W * g_rr) * xr_z)
            + d_dv(_cT(D_theta0), (psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * xr_z)
            + d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * xr_v)
        )

        # Q^2_vv -> upsilon block
        Au += d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * psi_r * W * g_vv) * xu_z)
        Ar += iota_psi_r2 * d_dr(
            _cT(D_rho0), (psi_r_over_sqrtg * W * g_vv / psi_r) * xr1_r
        )
        Ar += -iota_psi_r2 * d_dr(
            _cT(D_rho0), (psi_r_over_sqrtg * W * g_vv) * xu_z
        )
        Au += -d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * W * g_vv) * xr1_r)

        # Q^2_zz -> upsilon block after variable change
        Au += d_dv(_cT(D_theta0), (psi_r_over_sqrtg * psi_r * W * g_pp) * xu_v)
        Ar += psi_r2 * d_dr(
            _cT(D_rho0), (psi_r_over_sqrtg * W * g_pp / psi_r) * xr2_r
        )
        Ar += psi_r2 * d_dr(
            _cT(D_rho0), (psi_r_over_sqrtg * W * g_pp) * xu_v
        )
        Au += d_dv(_cT(D_theta0), (psi_r_over_sqrtg * W * g_pp) * xr2_r)

        # Q^2_rv and transpose
        Ar += -(
            d_dv(_cT(D_theta0), (iota * psi_r * psi_r_over_sqrtg * W * g_rv) * xr1_r)
            + d_dz(_cT(D_zeta0), (psi_r * psi_r_over_sqrtg * W * g_rv) * xr1_r)
        )
        Ar += -(
            iota_psi_r2 * d_dr(_cT(D_rho0), (iota * psi_r * psi_r_over_sqrtg * W * g_rv) * xr_v)
            + iota_psi_r2 * d_dr(_cT(D_rho0), (psi_r * psi_r_over_sqrtg * W * g_rv) * xr_z)
        )
        Ar += (
            d_dv(_cT(D_theta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * xu_z)
            + d_dz(_cT(D_zeta0), (psi_r2 * psi_r_over_sqrtg * W * g_rv) * xu_z)
        )
        Au += (
            d_dz(_cT(D_zeta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * xr_v)
            + d_dz(_cT(D_zeta0), (psi_r2 * psi_r_over_sqrtg * W * g_rv) * xr_z)
        )

        # Q^2_rz and transpose
        Ar += -(
            d_dv(_cT(D_theta0), (iota * psi_r * psi_r_over_sqrtg * W * g_rp) * xr2_r)
            + d_dz(_cT(D_zeta0), (psi_r * psi_r_over_sqrtg * W * g_rp) * xr2_r)
        )
        Ar += -(
            psi_r2 * d_dr(_cT(D_rho0), (iota * psi_r * psi_r_over_sqrtg * W * g_rp) * xr_v)
            + psi_r2 * d_dr(_cT(D_rho0), (psi_r * psi_r_over_sqrtg * W * g_rp) * xr_z)
        )
        Ar += -(
            d_dv(_cT(D_theta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * xu_v)
            + d_dz(_cT(D_zeta0), (psi_r2 * psi_r_over_sqrtg * W * g_rp) * xu_v)
        )
        Au += -(
            d_dv(_cT(D_theta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * xr_v)
            + d_dv(_cT(D_theta0), (psi_r2 * psi_r_over_sqrtg * W * g_rp) * xr_z)
        )

        # Q^2_vz and transpose
        Au += -(
            d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * psi_r * W * g_vp) * xu_v)
            + d_dv(_cT(D_theta0), (psi_r_over_sqrtg * psi_r * W * g_vp) * xu_z)
        )
        Ar += -psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_vp) * xu_z)
        Ar += iota_psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_vp) * xu_v)
        Au += -d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * W * g_vp) * xr2_r)
        Au += d_dv(_cT(D_theta0), (psi_r_over_sqrtg * W * g_vp) * xr1_r)
        Ar += iota_psi_r2 * d_dr(
            _cT(D_rho0), (psi_r_over_sqrtg * W * g_vp / psi_r) * xr2_r
        )
        Ar += psi_r2 * d_dr(
            _cT(D_rho0), (psi_r_over_sqrtg * W * g_vp / psi_r) * xr1_r
        )

        # J cross Q terms. The two branches are algebraically identical; they differ
        # only in how g^rv/g^rz are obtained. jq_lower_metric=True is the dense
        # _agni3_assemble route (the _term factors already carry psi_r*sqrtg, hence
        # psi_r2 here where the other branch has psi_r3*sqrtg).
        if jq_lower_metric:
            jq = (
                psi_r2 * W
                * (j_sup_theta * g_sup_rp_term + j_sup_zeta * g_sup_rv_term)
                / g_sup_rr
            )
        else:
            jq = psi_r3 * sqrtg * W * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv) / g_sup_rr
        Ar += -(jq * (iota * xr_v + xr_z))
        Ar += -(psi_r * sqrtg * W * j_sup_zeta) * xr1_r
        Ar += -(psi_r * sqrtg * W * j_sup_theta) * xr2_r
        Ar += -(iota * d_dv(_cT(D_theta0), jq * xr) + d_dz(_cT(D_zeta0), jq * xr))
        Ar += -iota_psi_r2 * d_dr(_cT(D_rho0), psi_r * sqrtg * W * j_sup_zeta * xr)
        Ar += -psi_r2 * d_dr(_cT(D_rho0), psi_r * sqrtg * W * j_sup_theta * xr)
        Ar += -(W * psi_r2 * sqrtg * j_sup_theta) * xu_v + (W * psi_r2 * sqrtg * j_sup_zeta) * xu_z
        Au += -d_dv(_cT(D_theta0), W * psi_r2 * sqrtg * j_sup_theta * xr)
        Au += d_dz(_cT(D_zeta0), W * psi_r2 * sqrtg * j_sup_zeta * xr)

        # |J|^2 and instability drive
        Ar += (psi_r2 * W * sqrtg * J2) * xr
        Aur = (W * psi_r2 * sqrtg * F) * xr

        # Compressibility terms
        gp = gamma * sqrtg * W * p0
        cr = psi_r * partial_r_log_sqrtg * xr + xr3_r
        cu = partial_v_log_sqrtg * xu + xu_v
        cz = partial_v_log_sqrtg * xz + xz_v + iotainv * (partial_p_log_sqrtg * xz + xz_z)

        Ar += psi_r * (
            partial_r_log_sqrtg * gp * cr + d_dr(_cT(D_rho0), gp * cr)
        )
        Au += partial_v_log_sqrtg * gp * cu + d_dv(_cT(D_theta0), gp * cu)
        Az += (
            partial_v_log_sqrtg * gp * cz
            + d_dv(_cT(D_theta0), gp * cz)
            + iotainv * (partial_p_log_sqrtg * gp * cz + d_dz(_cT(D_zeta0), gp * cz))
        )

        Ar += psi_r * (
            partial_r_log_sqrtg * gp * cu + d_dr(_cT(D_rho0), gp * cu)
        )
        Au += partial_v_log_sqrtg * gp * cr + d_dv(_cT(D_theta0), gp * cr)

        Ar += psi_r * (
            partial_r_log_sqrtg * gp * cz + d_dr(_cT(D_rho0), gp * cz)
        )
        Az += (
            partial_v_log_sqrtg * gp * cr
            + d_dv(_cT(D_theta0), gp * cr)
            + iotainv * (partial_p_log_sqrtg * gp * cr + d_dz(_cT(D_zeta0), gp * cr))
        )

        Au += partial_v_log_sqrtg * gp * cz + d_dv(_cT(D_theta0), gp * cz)
        Az += (
            partial_v_log_sqrtg * gp * cu
            + d_dv(_cT(D_theta0), gp * cu)
            + iotainv * (partial_p_log_sqrtg * gp * cu + d_dz(_cT(D_zeta0), gp * cu))
        )

        # Coupled_rt de-aliasing penalty on the physical diagonal blocks,
        # added before whitening to match the dense finite-n lambda3 path.
        if apply_penalty:
            Ar = Ar + _apply_penalty(xr)
            Au = Au + _apply_penalty(xu)
            Az = Az + _apply_penalty(xz)

        As = jnp.stack([Ar, Au, Az], axis=-1).reshape((n_total, 3))
        Aus = jnp.stack(
            [Aur, jnp.zeros_like(Aur), jnp.zeros_like(Aur)], axis=-1
        ).reshape((n_total, 3))
        ys = jnp.einsum("lij,lj->li", Linv_D, diagBsqinv * As) #stable
        yus = jnp.einsum("lij,lj->li", Linv_D, diagBsqinv * Aus) #unstable
        y = ys + yus
        return y.T.reshape(-1) + 1e-16 * x_flat

    def Ax(x_reduced):
        x_full = jnp.zeros(3 * n_total, dtype=x_reduced.dtype)
        # unique_indices=True: keep is a concatenation of disjoint aranges, so
        # the indices are unique. Declaring it lets JAX form the scatter's
        # transpose, which jax.scipy cg needs for its (symmetric) transpose-solve
        # in the matfree-library shiftinvert_cg path.
        x_full = x_full.at[keep].set(x_reduced, unique_indices=True)
        y_full = Ax_full(x_full)
        return y_full[keep]

    return {
        "Ax": Ax,
        "D_rho0": D_rho0,
        "D_theta0": D_theta0,
        "D_zeta0": D_zeta0,
        "Linv_DT": Linv_DT,
        "diagBsqinv": diagBsqinv,
        "g_pp": g_pp,
        "g_rp": g_rp,
        "g_rr": g_rr,
        "g_rv": g_rv,
        "g_vp": g_vp,
        "g_vv": g_vv,
        "iota": iota,
        "keep": keep,
        "n_keep": n_keep,
        "n_rho": n_rho,
        "n_theta": n_theta,
        "n_total": n_total,
        "n_zeta": n_zeta,
        "psi_r": psi_r,
        "psi_r_over_sqrtg": psi_r_over_sqrtg,
        "d_dr": d_dr,
        "d_dv": d_dv,
        "d_dz": d_dz,
    }


@register_compute_fun(
    name="finite-n lambda3 matfree",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate using lambda3 algebra in matrix-free form",
    dim=1,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "g^rr",
        "g^rv",
        "g^rz",
        "J^theta_PEST",
        "J^zeta",
        "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota",
        "psi_r",
        "psi_rr",
        "p",
        "p_r",
        "a",
    ],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    stable_only="bool: for testing only, materialize and eigendecompose the stable part of the matrix",
    v_guess="ndarray: eigenfunction guess to initialize the iterative eigenvalue solver",
    matfree_solver="str: matrix-free eigensolver backend ('eigsh_shiftinvert' (default), 'eigsh_shiftinvert_pcg', 'shiftinvert_cg', or experimental 'shiftinvert_pcg')",
    eigsh_tol="float: tolerance for ARPACK eigsh in matrix-free mode",
    eigsh_maxiter="int: max iterations for ARPACK eigsh in matrix-free mode",
    eigsh_ncv="int: number of Lanczos vectors used by ARPACK eigsh",
    debug_matfree="bool: print matrix-free solver diagnostics",
    sigma="float: shift used by shiftinvert_cg solver",
    num_matvecs="int: Lanczos matvec count for shiftinvert_cg solver",
    cg_tol="float: CG tolerance inside shiftinvert_cg solver",
    cg_maxiter="int: CG max iterations inside shiftinvert_cg solver",
    pcg_preconditioner="str: preconditioner for matfree_solver='eigsh_shiftinvert_pcg' ('fourier_mode_blocks' or experimental dense 'fourier_band_spd') or matfree_solver='shiftinvert_pcg' ('fourier_mode_blocks')",
    pcg_bandwidth="int: theta-mode cyclic bandwidth for pcg_preconditioner='fourier_band_spd'",
    pcg_floor_rel="float: relative eigenvalue floor for pcg_preconditioner='fourier_band_spd'",
    pcg_build_batch_size="int: number of Fourier-basis columns to batch while building the preconditioner",
    check_v_guess_only="bool: apply matrix-free Ax to v_guess without eigensolve",
    lambda_guess="float: eigenvalue used by check_v_guess_only",
    build_matrix="bool: materialize and return the reduced operator A (column j = Ax(e_j))",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier (rho, theta) "
    "operators instead of separable 1D matrices",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
)
def _AGNI3_matfree(params, transforms, profiles, data, **kwargs):
    """Matrix-free version of finite-n lambda3 in the (rho, upsilon, zeta) basis.

    When ``coupled_rt`` is set, ``D_rho0``/``D_theta0`` are the full,
    non-separable 2D (rho, theta) Zernike-Fourier operators of shape
    ``(n_rho*n_theta,)*2`` and the per-direction node counts come from
    ``n_rho_coupled``/``n_theta_coupled``. This mirrors the ``coupled_rt``
    treatment in ``finite-n lambda3`` (``_AGNI3``): the quadrature weights still
    factorize (``W_rho (x) W_theta (x) W_zeta``), node ordering stays rho-major
    (rho outer, theta inner, zeta innermost), and only the ``d_dr``/``d_dv``
    derivative helpers change to flatten the (rho, theta) axes and apply the
    full coupled operator.
    """
    _op = _agni3_matfree_operator(params, transforms, profiles, data, **kwargs)
    Ax = _op["Ax"]
    D_rho0 = _op["D_rho0"]
    D_theta0 = _op["D_theta0"]
    D_zeta0 = _op["D_zeta0"]
    Linv_DT = _op["Linv_DT"]
    diagBsqinv = _op["diagBsqinv"]
    g_pp = _op["g_pp"]
    g_rp = _op["g_rp"]
    g_rr = _op["g_rr"]
    g_rv = _op["g_rv"]
    g_vp = _op["g_vp"]
    g_vv = _op["g_vv"]
    iota = _op["iota"]
    keep = _op["keep"]
    n_keep = _op["n_keep"]
    n_rho = _op["n_rho"]
    n_theta = _op["n_theta"]
    n_total = _op["n_total"]
    n_zeta = _op["n_zeta"]
    psi_r = _op["psi_r"]
    psi_r_over_sqrtg = _op["psi_r_over_sqrtg"]
    d_dr = _op["d_dr"]
    d_dv = _op["d_dv"]
    d_dz = _op["d_dz"]


    n_rho_reduced = (n_rho - 2) * n_theta * n_zeta

    def _theta_mode_groups():
        """Coefficient-space indices grouped by theta Fourier mode."""
        groups = []

        def _component_indices(offset, nr_comp, k):
            rr = np.arange(nr_comp, dtype=np.int64)[:, None]
            zz = np.arange(n_zeta, dtype=np.int64)[None, :]
            return (offset + rr * n_theta * n_zeta + k * n_zeta + zz).reshape(-1)

        for k in range(n_theta):
            parts = [
                _component_indices(0, n_rho - 2, k),
                _component_indices(n_rho_reduced, n_rho, k),
                _component_indices(n_rho_reduced + n_total, n_rho, k),
            ]
            groups.append(np.concatenate(parts))
        return groups

    def _theta_mode_labels():
        labels = np.empty(n_keep, dtype=np.int64)
        for k, idxs in enumerate(_theta_mode_groups()):
            labels[idxs] = k
        return labels

    def _red_to_theta_coeff_batch(x):
        """Reduced nodal row-vectors -> normalized theta-Fourier coefficients."""
        x = np.asarray(x)
        one_dim = x.ndim == 1
        if one_dim:
            x = x[None, :]
        scale = np.sqrt(n_theta)
        rho_part = x[:, :n_rho_reduced].reshape(-1, n_rho - 2, n_theta, n_zeta)
        ups_part = x[:, n_rho_reduced : n_rho_reduced + n_total].reshape(
            -1, n_rho, n_theta, n_zeta
        )
        zet_part = x[:, n_rho_reduced + n_total :].reshape(
            -1, n_rho, n_theta, n_zeta
        )
        out = np.concatenate(
            [
                (np.fft.fft(rho_part, axis=2) / scale).reshape(x.shape[0], -1),
                (np.fft.fft(ups_part, axis=2) / scale).reshape(x.shape[0], -1),
                (np.fft.fft(zet_part, axis=2) / scale).reshape(x.shape[0], -1),
            ],
            axis=1,
        )
        return out[0] if one_dim else out

    def _theta_coeff_to_red_batch(xh):
        """Normalized theta-Fourier coefficient row-vectors -> reduced nodal vectors."""
        xh = np.asarray(xh)
        one_dim = xh.ndim == 1
        if one_dim:
            xh = xh[None, :]
        scale = np.sqrt(n_theta)
        rho_part = xh[:, :n_rho_reduced].reshape(-1, n_rho - 2, n_theta, n_zeta)
        ups_part = xh[:, n_rho_reduced : n_rho_reduced + n_total].reshape(
            -1, n_rho, n_theta, n_zeta
        )
        zet_part = xh[:, n_rho_reduced + n_total :].reshape(
            -1, n_rho, n_theta, n_zeta
        )
        out = np.concatenate(
            [
                (np.fft.ifft(rho_part, axis=2) * scale).reshape(xh.shape[0], -1),
                (np.fft.ifft(ups_part, axis=2) * scale).reshape(xh.shape[0], -1),
                (np.fft.ifft(zet_part, axis=2) * scale).reshape(xh.shape[0], -1),
            ],
            axis=1,
        )
        return out[0] if one_dim else out

    def _red_to_theta_coeff_jax(x):
        """Reduced nodal vector -> normalized theta-Fourier coefficients."""
        scale = jnp.sqrt(jnp.asarray(n_theta, dtype=jnp.real(x).dtype))
        rho_part = x[:n_rho_reduced].reshape(n_rho - 2, n_theta, n_zeta)
        ups_part = x[n_rho_reduced : n_rho_reduced + n_total].reshape(
            n_rho, n_theta, n_zeta
        )
        zet_part = x[n_rho_reduced + n_total :].reshape(
            n_rho, n_theta, n_zeta
        )
        return jnp.concatenate(
            [
                (jnp.fft.fft(rho_part, axis=1) / scale).reshape(-1),
                (jnp.fft.fft(ups_part, axis=1) / scale).reshape(-1),
                (jnp.fft.fft(zet_part, axis=1) / scale).reshape(-1),
            ]
        )

    def _theta_coeff_to_red_jax(xh):
        """Normalized theta-Fourier coefficients -> reduced nodal vector."""
        scale = jnp.sqrt(jnp.asarray(n_theta, dtype=jnp.real(xh).dtype))
        rho_part = xh[:n_rho_reduced].reshape(n_rho - 2, n_theta, n_zeta)
        ups_part = xh[n_rho_reduced : n_rho_reduced + n_total].reshape(
            n_rho, n_theta, n_zeta
        )
        zet_part = xh[n_rho_reduced + n_total :].reshape(
            n_rho, n_theta, n_zeta
        )
        return jnp.concatenate(
            [
                (jnp.fft.ifft(rho_part, axis=1) * scale).reshape(-1),
                (jnp.fft.ifft(ups_part, axis=1) * scale).reshape(-1),
                (jnp.fft.ifft(zet_part, axis=1) * scale).reshape(-1),
            ]
        )

    def _apply_A_batch(x_batch):
        return np.asarray(jax.vmap(Ax)(jnp.asarray(x_batch)))

    def _coefficient_columns(idxs, dtype):
        cols = np.zeros((idxs.size, n_keep), dtype=dtype)
        cols[np.arange(idxs.size), idxs] = 1.0
        return cols

    def _build_theta_mode_block_preconditioner(op_dtype):
        """Build an SPD block inverse in theta-Fourier coefficient space.

        This costs one application of ``Ax`` per reduced unknown, but stores only
        same-mode blocks. It is experimental: useful when many inner CG solves
        are expected for one fixed equilibrium, expensive if rebuilt repeatedly.
        """
        from scipy.linalg import cho_factor, cho_solve

        build_t0 = time.time()
        groups = _theta_mode_groups()
        factors = []
        min_eigs = []
        batch_size = int(kwargs.get("pcg_build_batch_size", 128))
        for idxs in groups:
            block_cols = []
            for i0 in range(0, idxs.size, batch_size):
                ib = idxs[i0 : i0 + batch_size]
                coeff_basis = _coefficient_columns(ib, op_dtype)
                red_basis = _theta_coeff_to_red_batch(coeff_basis)
                y = _apply_A_batch(red_basis) - sigma * red_basis
                yhat = _red_to_theta_coeff_batch(y)
                block_cols.append(yhat[:, idxs].T)
            block = np.concatenate(block_cols, axis=1)
            block = 0.5 * (block + block.conj().T)
            min_eigs.append(float(np.linalg.eigvalsh(block)[0]))
            factors.append(cho_factor(block, lower=True, check_finite=False))

        if debug_matfree:
            print(
                "[finite-n lambda3 matfree pcg]",
                "preconditioner=fourier_mode_blocks",
                f"build_time={time.time() - build_t0:.2f}s",
                f"min_block_eig={min(min_eigs):.3e}",
                flush=True,
            )

        def _mv(x):
            xh = _red_to_theta_coeff_batch(x)
            yh = np.zeros_like(xh)
            for idxs, fac in zip(groups, factors):
                yh[idxs] = cho_solve(fac, xh[idxs], check_finite=False)
            return _theta_coeff_to_red_batch(yh)

        return LinearOperator(shape=(n_keep, n_keep), matvec=_mv, dtype=op_dtype)

    def _build_theta_mode_block_preconditioner_jax(op_dtype):
        """Build a JAX-callable HPD mode-block preconditioner for CG.

        The operator passed to CG remains ``A - sigma I``. This function is
        supplied as the CG ``M`` callback, so CG uses the standard HPD
        preconditioned recurrence instead of applying a nonsymmetric ``M A``.
        """

        build_t0 = time.time()
        groups_np = _theta_mode_groups()
        blocks = []
        min_eigs = []
        batch_size = int(kwargs.get("pcg_build_batch_size", 128))
        np_dtype = np.dtype(op_dtype)
        for idxs in groups_np:
            block_cols = []
            for i0 in range(0, idxs.size, batch_size):
                ib = idxs[i0 : i0 + batch_size]
                coeff_basis = _coefficient_columns(ib, np_dtype)
                red_basis = _theta_coeff_to_red_batch(coeff_basis)
                y = _apply_A_batch(red_basis) - sigma * red_basis
                yhat = _red_to_theta_coeff_batch(y)
                block_cols.append(yhat[:, idxs].T)
            block = np.concatenate(block_cols, axis=1)
            block = 0.5 * (block + block.conj().T)
            evals = np.linalg.eigvalsh(block)
            min_eigs.append(float(evals[0]))
            if evals[0] <= 0.0:
                raise np.linalg.LinAlgError(
                    "theta Fourier mode block preconditioner is not HPD: "
                    f"min_eig={evals[0]:.3e}"
                )
            blocks.append(jnp.asarray(block, dtype=op_dtype))

        if debug_matfree:
            print(
                "[finite-n lambda3 matfree pcg]",
                "preconditioner=fourier_mode_blocks_jax",
                f"build_time={time.time() - build_t0:.2f}s",
                f"min_block_eig={min(min_eigs):.3e}",
                flush=True,
            )

        def _mv(x):
            xh = _red_to_theta_coeff_jax(x)
            yh = jnp.zeros_like(xh)
            for idxs, block in zip(groups_np, blocks):
                yh = yh.at[idxs].set(jnp.linalg.solve(block, xh[idxs]))
            return _theta_coeff_to_red_jax(yh)

        return _mv

    def _build_theta_band_spd_preconditioner(op_dtype):
        """Build a dense SPD-projected theta-Fourier band preconditioner.

        This is a diagnostic-quality preconditioner: it materializes the
        Fourier-space operator, drops mode couplings outside a cyclic bandwidth,
        and floors the eigenvalues before applying the dense inverse. It is not
        intended as the final large-scale implementation.
        """
        from scipy.linalg import eigh

        build_t0 = time.time()
        bandwidth = int(kwargs.get("pcg_bandwidth", 2))
        floor_rel = float(kwargs.get("pcg_floor_rel", 1e-6))
        batch_size = int(kwargs.get("pcg_build_batch_size", 128))
        groups = _theta_mode_groups()
        labels = _theta_mode_labels()
        Ahat = np.zeros((n_keep, n_keep), dtype=op_dtype)
        all_idxs = np.arange(n_keep)
        for i0 in range(0, n_keep, batch_size):
            ib = all_idxs[i0 : i0 + batch_size]
            coeff_basis = _coefficient_columns(ib, op_dtype)
            red_basis = _theta_coeff_to_red_batch(coeff_basis)
            y = _apply_A_batch(red_basis) - sigma * red_basis
            yhat = _red_to_theta_coeff_batch(y)
            Ahat[:, ib] = yhat.T

        dm = np.abs(labels[:, None] - labels[None, :])
        cyclic_dm = np.minimum(dm, n_theta - dm)
        Ahat = np.where(cyclic_dm <= bandwidth, Ahat, 0.0)
        Ahat = 0.5 * (Ahat + Ahat.conj().T)
        vals, vecs = eigh(Ahat, check_finite=False)
        floor = floor_rel * max(float(vals[-1]), 1.0)
        floored = np.maximum(vals, floor)
        inv_vals = 1.0 / floored

        if debug_matfree:
            print(
                "[finite-n lambda3 matfree pcg]",
                "preconditioner=fourier_band_spd",
                f"bandwidth={bandwidth}",
                f"floor_rel={floor_rel:.3e}",
                f"build_time={time.time() - build_t0:.2f}s",
                f"raw_min={vals[0]:.3e}",
                f"floor={floor:.3e}",
                f"n_floored={int(np.count_nonzero(floored > vals))}",
                flush=True,
            )

        def _mv(x):
            xh = _red_to_theta_coeff_batch(x)
            yh = vecs @ (inv_vals * (vecs.conj().T @ xh))
            return _theta_coeff_to_red_batch(yh)

        return LinearOperator(shape=(n_keep, n_keep), matvec=_mv, dtype=op_dtype)

    def _build_pcg_preconditioner(op_dtype):
        preconditioner = str(
            kwargs.get("pcg_preconditioner", "fourier_mode_blocks")
        ).lower()
        if preconditioner in {"none", "false", "0", "off"}:
            return None
        if n_theta <= 1:
            if debug_matfree:
                print(
                    "[finite-n lambda3 matfree pcg]",
                    "theta preconditioner disabled for n_theta <= 1",
                    flush=True,
                )
            return None
        if preconditioner in {"fourier_mode_blocks", "mode_blocks", "blocks"}:
            return _build_theta_mode_block_preconditioner(op_dtype)
        if preconditioner in {"fourier_band_spd", "band_spd", "band"}:
            return _build_theta_band_spd_preconditioner(op_dtype)
        raise ValueError(
            f"Unknown pcg_preconditioner={preconditioner!r}; expected "
            "'fourier_mode_blocks', 'fourier_band_spd', or 'none'."
        )

    # Optionally materialize the reduced operator A (column j = Ax(e_j)).
    # BATCHED + host-accumulated: never forms the full n x n identity (only small
    # (b, n_keep) sub-identity blocks), vmaps Ax over one batch at a time, and moves
    # each batch to host. So the DEVICE only ever holds one batch's intermediates
    # (not the width-n vmap), which lets moderate/high n build on a GPU; the full
    # A_mat lives on host (numpy) for the CPU eigh. Batch size via AGNI_BUILD_BATCH.
    if bool(kwargs.get("build_matrix", False)):
        import os as _os
        _dt = Ax(jnp.ones(n_keep)).dtype
        bs = int(kwargs.get("build_matrix_batch", _os.environ.get("AGNI_BUILD_BATCH", 512)))
        bs = max(1, min(bs, n_keep))
        rows = []
        for j0 in range(0, n_keep, bs):
            j1 = min(j0 + bs, n_keep)
            blk = jnp.zeros((j1 - j0, n_keep), dtype=_dt).at[
                jnp.arange(j1 - j0), jnp.arange(j0, j1)].set(1.0)  # e_{j0..j1-1}
            rows.append(np.asarray(jax.vmap(Ax)(blk)))            # row i = A e_{j0+i}
        A_mat = np.concatenate(rows, axis=0).T                    # columns = A e_j
        data["finite-n lambda3 matfree operator"] = A_mat
        data["finite-n lambda3 matfree keep"] = keep
        return data

    v0 = kwargs.get("v_guess", jnp.ones(n_keep))
    v0 = jnp.asarray(v0).reshape(-1)
    if v0.size == 3 * n_total:
        v0 = v0[keep]
    elif v0.size != n_keep:
        print(
            "finite-n lambda3 matfree ignoring invalid v_guess size:",
            f"got={v0.size}, expected={n_keep} or {3 * n_total}",
        )
        v0 = jnp.ones(n_keep)

    matfree_solver = str(kwargs.get("matfree_solver", "eigsh_shiftinvert")).lower()
    debug_matfree = bool(kwargs.get("debug_matfree", False))
    check_v_guess_only = bool(kwargs.get("check_v_guess_only", False))
    pcg_stats = None
    shiftinvert_stats = None

    if check_v_guess_only:
        ax_v0 = Ax(v0)
        lambda_guess = kwargs.get("lambda_guess", None)
        if lambda_guess is None:
            lambda_guess = jnp.real(jnp.vdot(v0, ax_v0) / jnp.vdot(v0, v0))
        w = jnp.atleast_1d(jnp.asarray(lambda_guess, dtype=ax_v0.dtype))
        v = v0
        # Ratio residual: per-component (Av - lam v)/v evaluated ONLY where the
        # eigenVECTOR is significant (|v| > 1e-5) -- i.e. on the mode's own support,
        # where Av/v must equal lam. Masking on |v| (not |Av|) avoids keying on the
        # high-mode-contaminated components, so this is far less sensitive to A's
        # ~1e6 high modes than the relative_residual ||Av-lam v||/||Av|| below.
        ratio_mask = jnp.abs(v0) > 1e-5
        n_ratio = jnp.sum(ratio_mask)
        safe_v = jnp.where(ratio_mask, v0, 1.0)
        per_comp = jnp.where(ratio_mask, (ax_v0 - w[0] * v0) / safe_v, 0.0)
        ratio_residual = jnp.linalg.norm(per_comp)
        # dimensionless RMS: mean per-node |Av/v - lam| relative to |lam|
        ratio_residual_rel = ratio_residual / (
            jnp.abs(w[0]) * jnp.sqrt(jnp.maximum(n_ratio, 1))
        )
        relative_residual = jnp.linalg.norm(ax_v0 - w[0] * v0) / (
            jnp.linalg.norm(ax_v0) + 1e-300
        )
        data["finite-n lambda3 matfree check ratio_residual"] = ratio_residual
        data["finite-n lambda3 matfree check ratio_residual_rel"] = ratio_residual_rel
        data["finite-n lambda3 matfree check relative_residual"] = relative_residual
        data["finite-n lambda3 matfree check n_checked"] = n_ratio
        # Norm-wise Rayleigh residual r = ||A v - lam_R v|| / (|lam_R| ||v||) with the
        # Rayleigh quotient lam_R = v^H A v / v^H v. Unlike relative_residual (which is
        # divided by ||Av|| and therefore deflated by A's ~1e6 high modes), this
        # normalization exposes eigenvector contamination faithfully -- it is the metric
        # the matfree-zernike convergence test compares v_mf against v_ref on.
        lam_rayleigh = jnp.real(jnp.vdot(v0, ax_v0) / jnp.vdot(v0, v0))
        rayleigh_residual = jnp.linalg.norm(ax_v0 - lam_rayleigh * v0) / (
            jnp.abs(lam_rayleigh) * jnp.linalg.norm(v0) + 1e-300
        )
        data["finite-n lambda3 matfree check lambda_rayleigh"] = lam_rayleigh
        data["finite-n lambda3 matfree check rayleigh_residual"] = rayleigh_residual
    elif matfree_solver in {"eigsh_shiftinvert", "eigsh_shiftinvert_pcg"}:
        # Matrix-free analogue of the dense path: build the shift-inverted
        # operator (A - sigma I)^{-1} and ask ARPACK for its largest-magnitude
        # eigenvalues (which="LM"), which map back to the eigenvalues of A
        # nearest sigma -- i.e. the most-unstable mode. (A - sigma I) is SPD
        # when sigma sits below the most-negative eigenvalue, so CG converges
        # and supplies OPinv without ever materializing A. The ``*_pcg`` variant
        # builds one experimental preconditioner for that shifted inner solve.
        from scipy.sparse.linalg import cg as _scipy_cg

        sigma = float(kwargs.get("sigma", -1e-1))
        tol = float(kwargs.get("eigsh_tol", 1e-8))
        maxiter = kwargs.get("eigsh_maxiter", None)
        maxiter = None if maxiter is None else int(maxiter)
        ncv = kwargs.get("eigsh_ncv", None)
        ncv = None if ncv is None else int(ncv)
        cg_tol = float(kwargs.get("cg_tol", 1e-8))
        cg_maxiter = int(kwargs.get("cg_maxiter", 5 * n_keep))

        v0_np = np.asarray(v0)
        ax0 = np.asarray(Ax(jnp.asarray(v0_np)))
        op_dtype = np.result_type(ax0.dtype, v0_np.dtype)

        def _ax_np(x):
            return np.asarray(Ax(jnp.asarray(x)), dtype=op_dtype)

        Aop = LinearOperator(shape=(n_keep, n_keep), matvec=_ax_np, dtype=op_dtype)
        Ashift = LinearOperator(
            shape=(n_keep, n_keep),
            matvec=lambda x: _ax_np(x) - sigma * x,
            dtype=op_dtype,
        )

        Mop = None
        shiftinvert_stats = {"solves": 0, "total_iters": 0, "infos": []}
        if matfree_solver == "eigsh_shiftinvert_pcg":
            pcg_stats = shiftinvert_stats
            Mop = _build_pcg_preconditioner(op_dtype)

        def _opinv(b):
            iters = {"n": 0}
            y, _info = _scipy_cg(
                Ashift,
                b,
                rtol=cg_tol,
                atol=0.0,
                maxiter=cg_maxiter,
                M=Mop,
                callback=lambda _x: iters.__setitem__("n", iters["n"] + 1),
            )
            shiftinvert_stats["solves"] += 1
            shiftinvert_stats["total_iters"] += iters["n"]
            shiftinvert_stats["infos"].append(int(_info))
            return y

        OPinv = LinearOperator(shape=(n_keep, n_keep), matvec=_opinv, dtype=op_dtype)
        w_np, v_np = eigsh(
            Aop,
            k=1,
            sigma=sigma,
            which="LM",
            OPinv=OPinv,
            v0=v0_np.astype(op_dtype, copy=False),
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
            return_eigenvectors=True,
        )
        w = jnp.asarray(w_np)
        v = jnp.asarray(v_np[:, 0])
    elif matfree_solver in {"shiftinvert_cg", "shiftinvert_pcg",
                            "shiftinvert_bicgstab", "shiftinvert_gmres"}:
        _require_matfree_backend()
        # Default sigma must sit BELOW the most-unstable (most-negative)
        # eigenvalue so (A - sigma I) is SPD and CG is valid; matches the
        # scipy 'eigsh_shiftinvert' default. The ``*_pcg`` variant keeps this
        # Hermitian shifted operator as the CG matrix and supplies an HPD
        # approximate inverse through CG's M callback.
        sigma = kwargs.get("sigma", -1e-1)
        num_matvecs = int(kwargs.get("num_matvecs", 20))
        cg_tol = float(kwargs.get("cg_tol", 1e-5))
        cg_maxiter = int(kwargs.get("cg_maxiter", n_keep))

        # The operator is complex (e.g. axisym D_zeta = 1j n), so the jax CG
        # while_loop carry must be complex from the start; a real v0 makes the
        # carry input (real) and output (complex) dtypes disagree and crashes.
        op_dtype = jnp.result_type(Ax(v0).dtype, v0.dtype)
        v0 = v0.astype(op_dtype)
        M_apply = None
        if matfree_solver == "shiftinvert_pcg":
            preconditioner = str(
                kwargs.get("pcg_preconditioner", "fourier_mode_blocks")
            ).lower()
            if preconditioner in {"none", "false", "0", "off"} or n_theta <= 1:
                if debug_matfree:
                    print(
                        "[finite-n lambda3 matfree pcg]",
                        "preconditioner disabled in shiftinvert_pcg",
                        flush=True,
                    )
            elif preconditioner in {"fourier_mode_blocks", "mode_blocks", "blocks"}:
                M_apply = _build_theta_mode_block_preconditioner_jax(op_dtype)
            else:
                raise ValueError(
                    "matfree_solver='shiftinvert_pcg' currently supports only "
                    "pcg_preconditioner='fourier_mode_blocks' or 'none'."
                )

        # Inner Krylov solver for (A - sigma I) x = b. cg assumes the shifted
        # operator is Hermitian SPD; bicgstab/gmres are robust to a non-Hermitian
        # Ax (e.g. if the raw operator is not exactly Hermitian), which can clean
        # up the shift-invert Lanczos vectors.
        _inner = {
            "shiftinvert_cg": cg, "shiftinvert_pcg": cg,
            "shiftinvert_bicgstab": bicgstab, "shiftinvert_gmres": gmres,
        }[matfree_solver]

        def OPinv(b):
            def Ashift(x):
                return Ax(x) - sigma * x

            y, _ = _inner(
                Ashift,
                b.astype(op_dtype),
                tol=cg_tol,
                maxiter=cg_maxiter,
                M=M_apply,
            )
            return y

        tridiag = decomp.tridiag_sym(
            num_matvecs, reortho="full", materialize=True
        )
        alg = eig.eigh_partial(tridiag)
        mu, vecs = alg(lambda x: OPinv(x), v0)
        sort_idxs = jnp.argsort(mu, descending=True)
        w = sigma + 1.0 / mu[sort_idxs]
        v = vecs[sort_idxs][0, :]
    else:
        raise ValueError(
            f"Unknown matfree_solver={matfree_solver!r}; expected "
            "'eigsh_shiftinvert', 'eigsh_shiftinvert_pcg', 'shiftinvert_cg', "
            "'shiftinvert_pcg', 'shiftinvert_bicgstab', or 'shiftinvert_gmres'."
        )

    v_full = jnp.zeros(3 * n_total, dtype=v.dtype)
    v_full = v_full.at[keep].set(v)

    x = jnp.transpose(v_full.reshape(3, n_total), axes=(1, 0))
    x = diagBsqinv * jnp.einsum("lij,lj->li", Linv_DT, x)
    x = x.reshape((n_rho, n_theta, n_zeta, 3))
    xr = x[..., 0]
    xv = x[..., 1]
    xz = x[..., 2]

    # Phase rotation doesn't change the physics. Here, we use it to make the eigenmode up-down symmetric.
    # phase_offset (default 0) is an optional tunable rotation applied on top of the mean-based alignment.
    # Same post-processing as the dense solvers; for real (non-axisym) modes this reduces to the identity.
    phase_offset = kwargs.get("phase_offset", 0.0)
    xi_ref = xr
    phase_angle = jnp.arctan2(jnp.mean(xi_ref.real), jnp.mean(xi_ref.imag))
    xr = (xr * jnp.exp(1j * (phase_angle + phase_offset))).imag
    xv = (xv * jnp.exp(1j * (phase_angle + phase_offset))).imag
    xz = (xz * jnp.exp(1j * (phase_angle + phase_offset))).imag

    # Forward derivatives for the homogenized deltaB (matches finite-n lambda7).
    xr_v = d_dv(D_theta0, xr)
    xr_z = d_dz(D_zeta0, xr)
    test_v = d_dv(D_theta0, xv)
    test_z = d_dz(D_zeta0, xv)
    xr_r = d_dr(D_rho0, xr)
    psi_rr = d_dr(D_rho0, psi_r)
    iota_r = d_dr(D_rho0, iota)

    deltaB_r = psi_r_over_sqrtg * psi_r * (iota * xr_v + xr_z)
    deltaB_v = psi_r_over_sqrtg * (1.* (test_z) - 1.*(xr_r * iota *psi_r + (2 * iota * psi_rr + iota_r * psi_r)* xr))
    deltaB_z = -psi_r_over_sqrtg * (1.* (test_v) + 1.*(xr_r * psi_r + 2 * psi_rr * xr))

    deltaV_r = psi_r * xr
    deltaV_v = xv + xz
    deltaV_z = xz * 1/iota

    deltaB2 = g_rr * deltaB_r ** 2 + 1.*g_vv * deltaB_v ** 2  + g_pp * deltaB_z ** 2 + 2. * (g_rv * deltaB_r * deltaB_v + g_rp * deltaB_r * deltaB_z +  g_vp * deltaB_v * deltaB_z)
    deltaV2 = g_rr * deltaV_r ** 2 + 1.*g_vv * deltaV_v ** 2  + g_pp * deltaV_z ** 2 + 2. * (g_rv * deltaV_r * deltaV_v + g_rp * deltaV_r * deltaV_z +  g_vp * deltaV_v * deltaV_z)

    data["finite-n lambda3 matfree"] = w
    data["finite-n eigenfunction3 matfree"] = v_full
    data["finite-n xi3 matfree"] = x.reshape((n_rho, n_theta, n_zeta, 3))
    data["finite-n deltaB"] = jnp.sqrt(deltaB2)
    data["finite-n deltaB_r"] = deltaB_r
    data["finite-n deltaB_v"] = deltaB_v
    data["finite-n deltaB_z"] = deltaB_z
    data["finite-n deltaV"] = jnp.sqrt(deltaV2)
    data["finite-n deltaV_r"] = deltaV_r
    data["finite-n deltaV_v"] = deltaV_v
    data["finite-n deltaV_z"] = deltaV_z
    if shiftinvert_stats is not None:
        data["finite-n lambda3 matfree shiftinvert solves"] = jnp.asarray(
            shiftinvert_stats["solves"]
        )
        data["finite-n lambda3 matfree shiftinvert total_iters"] = jnp.asarray(
            shiftinvert_stats["total_iters"]
        )
        data["finite-n lambda3 matfree shiftinvert infos"] = jnp.asarray(
            shiftinvert_stats["infos"]
        )
        if debug_matfree:
            print(
                "[finite-n lambda3 matfree shiftinvert]",
                f"solves={shiftinvert_stats['solves']}",
                f"total_cg_iters={shiftinvert_stats['total_iters']}",
                f"infos={shiftinvert_stats['infos']}",
                flush=True,
            )
    if pcg_stats is not None:
        data["finite-n lambda3 matfree pcg solves"] = jnp.asarray(pcg_stats["solves"])
        data["finite-n lambda3 matfree pcg total_iters"] = jnp.asarray(
            pcg_stats["total_iters"]
        )
        data["finite-n lambda3 matfree pcg infos"] = jnp.asarray(pcg_stats["infos"])
        if debug_matfree:
            print(
                "[finite-n lambda3 matfree pcg]",
                f"solves={pcg_stats['solves']}",
                f"total_cg_iters={pcg_stats['total_iters']}",
                f"infos={pcg_stats['infos']}",
                flush=True,
            )

    if debug_matfree:
        idxs = jnp.where(jnp.abs(v) > 1e-4)[0]
        res = Ax(v)
        print(
            jnp.max(res[idxs] / v[idxs]),
            jnp.min(res[idxs] / v[idxs]),
            jnp.mean(res[idxs] / v[idxs]),
        )

    return data


@register_compute_fun(
    name="finite-n lambda3 rayleigh",
    label="\\lambda_R = v^T A v / v^T v",
    units="~",
    units_long="None",
    description="Fixed-vector Rayleigh quotient of the finite-n lambda3 operator. "
    "The eigenvector v is supplied and held fixed, so this is a plain "
    "differentiable scalar function of the equilibrium parameters: AD through it "
    "supplies the Hellmann-Feynman contraction v^T (dA/dp) v / v^T v. No "
    "eigensolver runs here.",
    dim=1,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "g^rr",
        "g^rv",
        "g^rz",
        "J^theta_PEST",
        "J^zeta",
        "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota",
        "psi_r",
        "psi_rr",
        "p",
        "p_r",
        "a",
    ],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    density="ndarray: the radial density profile",
    f_scale="float: multiplier on the instability drive F (default 1.0)",
    sigma="float: shift for the ARPACK eigsh that supplies the fresh eigenvector",
    eigsh_tol="float: tolerance for the ARPACK eigsh",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier (rho, theta) "
    "operators instead of separable 1D matrices",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
)
def _AGNI3_rayleigh(params, transforms, profiles, data, **kwargs):
    """Rayleigh quotient of finite-n lambda3 with a FRESH eigenvector.

        lambda_R(p) = v^T A(p) v / (v^T v),   v = eigsh(A(p))  computed AT THIS p

    Why v is recomputed rather than cached
    --------------------------------------
    The (rho, theta_PEST, zeta) grid is fixed but the MESH is not: theta_PEST is
    defined through lambda (`L_lmn`), so `map_coordinates` solves
    `theta + lambda = theta_PEST` and any change in L_lmn puts the same PEST label at
    a different physical point. ProximalProjection re-solves the equilibrium before
    every objective evaluation, and L_lmn moves even when R_lmn/Z_lmn barely do.
    Measured at 24x32x12 (job 55995328): L_lmn moved 3.6e-3, theta moved 7.0e-5, and
    a CACHED v then gave lambda_R = +6.85e-03 with residual r = 4815 where the truth
    is -1.03e-04 with r = 0.39. A 7e-5 mesh shift destroys the eigenvector, so no
    trust radius and no pre-solve can rescue a cache.
    See AGNI_var/dense-eigsh-optimization/WHY_V_CANNOT_BE_CACHED.md.

    This mirrors BallooningStability, which calls `jnp.linalg.eigh` inside its own
    compute and so always holds the primal point's eigenvector. Our operator is
    26880^2 and cannot go through `eigh` under AD, so ARPACK supplies the primal via
    a host callback instead.

    Why no custom_jvp is needed
    ---------------------------
    A `jax.pure_callback` output carries ZERO tangent, so AD differentiates only
    `A(p)` and the quotient's derivative collapses to

        dlambda/dp = v^T (dA/dp) v / (v^T v)

    which is exactly Hellmann-Feynman -- the same rule `jnp.linalg.eigh` applies
    internally. `stop_gradient` is applied to v as well so the assumption is stated
    in the code rather than resting on a JAX implementation detail.
    """
    # The matfree OPERATOR (not the matfree EIGENSOLVER -- `shiftinvert_cg` sign-flips
    # at 40/48 and is never called). Verified against the dense A to 2e-11 on random
    # vectors (job 56012614); they are the same operator. This is the ONLY thing that
    # is traced: Ax's intermediates are (nr,nt,nz) arrays -- D_rho0/D_theta0/D_zeta0
    # are small per-dimension operators applied by einsum -- so reverse mode tapes
    # only VECTORS and one backward pass yields dlambda/dp for ALL parameters.
    _op = _agni3_matfree_operator(params, transforms, profiles, data, **kwargs)
    n_keep = _op["n_keep"]
    # A's dtype mirrors `A = einsum(Linv, A, Linv)` in the assembly: complex128 when
    # axisym (B_blocks is built complex there), float64 otherwise.
    _dtype = _op["Linv_DT"].dtype

    sigma = kwargs.get("sigma", -1e-1)
    eigsh_tol = kwargs.get("eigsh_tol", 1e-8)

    # ---------------------------------------------------------------------------
    # THE PRIMAL, ENTIRELY OUTSIDE THE TRACED GRAPH.
    #
    # `stop_gradient(A)` is NOT enough and was measured to fail (job 56013034, OOM
    # at 6115295232 bytes == 27648^2 * 8, an assembly temporary -- NOT outer(v,v)).
    # It stops the cotangent, but the assembly still runs inside the graph that
    # `jax.grad` traces, and its long `A.at[...].add(...)` chain stops being updated
    # in place once buffer donation is constrained under vjp. Each update then wants
    # a fresh 6.1 GB.
    #
    # So the whole primal -- assemble AND ARPACK -- goes inside the callback, where
    # params/data arrive CONCRETE. The assembly runs eagerly (on GPU), ARPACK runs on
    # the host, the 6.1 GB is freed before the traced part resumes, and no 27648^2
    # array ever enters the graph in either direction.
    #
    # `data` must be passed THROUGH the callback rather than closed over: it is traced
    # (it depends on params via _flux_data and the mapped grid). `transforms` and
    # `profiles` are concrete and are closed over.
    # ---------------------------------------------------------------------------
    _array_data, _other_data = {}, {}
    for _k, _val in data.items():
        if isinstance(_val, (jnp.ndarray, np.ndarray, float, int)):
            _array_data[_k] = jnp.asarray(_val)
        else:
            _other_data[_k] = _val

    # GPU shift-invert: factorize (A - sigma I) with cuSolver getrf on the GPU
    # instead of letting scipy's ARPACK do a dense LU on the CPU.
    #
    # DEFAULT OFF. In ISOLATION it is a big win -- n=26880 (job 56043107): GPU getrf
    # 1.37 s + eigsh 1 s vs 41 s CPU (~17x), lambda identical to 3.4e-08 (SAME mode,
    # not a null-cluster mode; getrf's O(n) workspace dodges the int32 overflow that
    # kills syevd/eigh above ~17k, [[project_agni_eigenspectrum]]). But INTEGRATED
    # into the full objective it is a NET LOSS: ~170 s/call vs ~118 s CPU (job
    # 56044503, steady across 3 calls, so per-call not compile). The LU's transient
    # 5.7 GB allocations (identity, A-sigma I, factors) contend with the resident Ax
    # operator + assembled A under MEM_FRACTION=.93, and that ~50 s of allocator
    # pressure is paid every call. Re-enable with AGNI_GPU_LU=1 only after the
    # transient allocations are cut (drop the full jnp.eye, free A, tune the fraction).
    import jax.scipy.linalg as _jsla

    _use_gpu_lu = os.environ.get("AGNI_GPU_LU", "0").lower() not in {
        "0", "false", "no", "off"
    }

    def _assemble_and_solve_host(params_host, data_host):
        """Assemble + ARPACK on concrete values. Never sees a tracer."""
        p_h = {k: jnp.asarray(val) for k, val in params_host.items()}
        d_h = dict(_other_data)
        d_h.update({k: jnp.asarray(val) for k, val in data_host.items()})
        A_h = _agni3_assemble(p_h, transforms, profiles, d_h, **kwargs)["A"]  # GPU
        nA = A_h.shape[0]

        if _use_gpu_lu:
            # Factor (A - sigma I) once on the GPU; ARPACK's shift-invert then only
            # needs OPinv @ b = (A - sigma I)^-1 @ b, done as GPU triangular solves.
            #
            # Memory-lean to fight the allocator pressure that made the naive version
            # a net loss (job 56044503). Three cuts:
            #   - no full jnp.eye: subtract sigma off the diagonal in place-ish.
            #   - FREE A's 5.7 GB device buffer the instant the shifted matrix M is
            #     formed -- A is not needed again (shift-invert never applies A, only
            #     OPinv), so peak during getrf is M + LU, not A + M + LU.
            #   - pass A to eigsh as a LinearOperator whose matvec is never called
            #     (ARPACK mode 3 uses OPinv only), so no 5.7 GB device->host copy.
            diag_idx = jnp.diag_indices(nA)
            M = A_h.at[diag_idx].add(-sigma)
            jax.block_until_ready(M)
            A_h.delete()  # free A (5.7 GB) before the factorization allocates the LU
            lu_piv = _jsla.lu_factor(M)
            jax.block_until_ready(lu_piv)
            M.delete()  # LU factors hold everything now; drop the shifted matrix

            def _opinv(b):
                x = _jsla.lu_solve(lu_piv, jnp.asarray(b, dtype=lu_piv[0].dtype))
                return np.asarray(x)

            def _never(_x):  # ARPACK mode 3 (shift-invert) never applies A
                raise RuntimeError("A.matvec should not be called in shift-invert")

            A_op = LinearOperator((nA, nA), matvec=_never, dtype=np.float64)
            OPinv = LinearOperator((nA, nA), matvec=_opinv, dtype=np.float64)
            # ARPACK already computes the eigenvalue; it used to be discarded.
            # Return it so both eigensolver paths have the SAME (v, lam) signature
            # and the cross-check monitor works on either.
            w_h, v_h = eigsh(
                A_op, k=1, sigma=sigma, OPinv=OPinv, which="LM",
                tol=eigsh_tol, return_eigenvectors=True,
            )
            return (
                np.asarray(v_h[:, 0], dtype=np.float64),
                np.asarray(w_h[0], dtype=np.float64),
            )

        A_np = np.asarray(A_h)
        w_h, v_h = eigsh(
            A_np, k=1, sigma=sigma, which="LM", tol=eigsh_tol,
            return_eigenvectors=True,
        )
        return (
            np.asarray(v_h[:, 0], dtype=A_np.dtype),
            np.asarray(w_h[0], dtype=A_np.dtype),
        )

    # -----------------------------------------------------------------------
    # PURE-JAX eigensolve (no callback): assemble + GPU getrf + matfree Lanczos.
    #
    # The callback exists ONLY to keep the dense assembly out of the reverse-mode
    # (vjp) trace, where XLA buffer donation is disabled and the assembly's scatter-
    # adds + getrf each allocate a fresh 5.7 GB (OOM). custom_vjp is an alternative
    # isolation: its fwd is not differentiated through (the bwd is the zero-cotangent
    # HF rule), so a jitted fwd MAY keep donation on and run in-place -- no callback,
    # all on GPU. AGNI_EIGENSOLVER=jax_lanczos selects this; default keeps the
    # validated callback path. If it OOMs, the assembly donation is not preserved and
    # the callback stays.
    #
    # The eigensolver is matfree's Lanczos (decomp.tridiag_sym reortho="full" +
    # eig.eigh_partial) -- the SAME machinery `finite-n lambda3 matfree` already uses,
    # but with an EXACT LU OPinv (lu_solve of getrf(A-sigma I)) in place of the CG
    # OPinv that failed (kappa~2e11, no preconditioner). LU is exact, so no CG
    # convergence problem; matfree is pure JAX, so no scipy and no re-entrant callback.
    _eigensolver = os.environ.get("AGNI_EIGENSOLVER", "eigsh_callback").lower()
    _num_matvecs = int(os.environ.get("AGNI_NUM_MATVECS", str(kwargs.get("num_matvecs", 50))))
    # --- TWO knobs, not seven. Consolidated 2026-07-19. -----------------------
    #
    # AGNI_SIGMA_MODE : fixed (default) | track | adapt      -- how sigma is chosen
    #   fixed  sigma = sigma_factor * lambda_guess, constant. Historical behaviour.
    #   track  re-based once per OUTER step on the trusted lambda. Superseded --
    #          too coarse, lambda moves ~18x WITHIN a step (BENCHMARKING.md 10.9).
    #   adapt  re-shifted INSIDE every solve onto that solve's own lam_mu. Makes
    #          the starting sigma nearly irrelevant (10.10).
    # AGNI_SIGMA_FACTOR : the multiplier for track/adapt (default 2.5, the measured
    #   sweet spot -- closer blows up LU conditioning, further loses separation).
    #   ONE factor for both modes; there used to be two that meant the same thing.
    # AGNI_DIAG : 0 = silent | 1 = cross-check (DEFAULT) | 2 = + r_mu/separation
    #   Level 1 is on by default because it is a safety monitor, not a diagnostic:
    #   it is the only thing standing between a contaminated eigenvector and a
    #   silently wrong descent direction. Level 2 adds one lu_solve per eigensolve.
    #
    # Replaced: AGNI_RMU, AGNI_XCHECK, AGNI_XCHECK_TOL, AGNI_SIGMA_ADAPT,
    # AGNI_SIGMA_ADAPT_FACTOR, AGNI_SIGMA_TRACK, AGNI_SIGMA_FACTOR.
    # fixed | track | adapt | track+adapt   (the last two are ORTHOGONAL:
    #   track  sets the INCOMING sigma once per OUTER step (objectives/_stability.py)
    #   adapt  re-shifts sigma INSIDE each solve from that solve's own lam_mu
    # so "track+adapt" = a good starting shift each step, corrected within the step.
    _sigma_mode = os.environ.get("AGNI_SIGMA_MODE", "fixed").lower()
    _valid = {"fixed", "track", "adapt", "track+adapt", "adapt+track"}
    if _sigma_mode not in _valid:
        raise ValueError(
            "AGNI_SIGMA_MODE must be one of %s, got %r" % (sorted(_valid), _sigma_mode)
        )
    _adapt = "adapt" in _sigma_mode
    _adapt_factor = float(os.environ.get("AGNI_SIGMA_FACTOR", "2.5"))
    _diag = int(os.environ.get("AGNI_DIAG", "1"))
    _xcheck = _diag >= 1
    _rmu_diag = _diag >= 2
    _xcheck_tol = 1e-2  # a constant, not a knob: nothing ever needed to tune it

    def _free(x):
        # Return the ~5.7 GB device buffer to the pool NOW rather than at GC. Each
        # eigensolve allocates A + M + LU (~17 GB); without prompt frees the leftover
        # chunks fragment the preallocated pool and a later call's 5.78 GB assembly
        # gather (jnp.ix_(keep,keep)) OOMs (job 56049551, ~4th call). Eager only;
        # a tracer has no buffer to delete, so guard.
        try:
            x.delete()
        except Exception:
            pass

    def _emit_rmu_diagnostic(sig, mu, idx, v_out, opinv):
        """AGNI_DIAG=2: print convergence + cluster separation. Costs one lu_solve.

        Q1 -- did Lanczos converge? The usual residual
        r = ||Ax(v) - lam v|| / (|lam| ||v||) CANNOT answer this: it is RETRACTED.
        |lam| ~ 1e-4 sits ~11 orders below ||A|| ~ 2.8e7, so it reports round-off
        multiplied by 2.7e11 (r = 0.39 is its FLOOR, not an error). In shift-invert
        space that inversion is gone -- shift-invert maps the wanted eigenvalue to
        the DOMINANT one, so |mu| ~ ||OP|| and r_mu is honestly scaled.
            r_mu small -> converged; the drift is NOT the eigensolver.
            r_mu large -> not converged. Treat r_mu > 1e-5 as danger.

        Q2 -- is the spectrum near sigma clustered? The Ritz values ARE approximate
        eigenvalues (lam = sigma + 1/mu), so their spread is free. If the top few
        |mu| are within a few percent the modes are unresolvable and MORE MATVECS
        CANNOT HELP (measured: identical to 7 digits at m=50/100/200) -- only
        moving sigma changes separation.
        """
        if not _rmu_diag:
            return
        mu_i = mu[idx]
        res = opinv(v_out) - mu_i * v_out  # the one extra triangular solve
        r_mu = jnp.linalg.norm(res) / (jnp.abs(mu_i) * jnp.linalg.norm(v_out))
        mu_s = mu[jnp.argsort(-jnp.abs(mu))][:6]
        jax.debug.print(
            "[rmu] sigma={s:.6e}  num_matvecs={k}\n"
            "[rmu] r_mu={r:.6e}   (>1e-5 => DANGER, not converged)\n"
            "[rmu] mu_sel={m:.8e}  lam_from_mu={l:.8e}\n"
            "[rmu] separation |mu_1|/|mu_2| = {sep:.4f}"
            "   (~1 => clustered, more matvecs cannot help)\n"
            "[rmu] top |mu|: {mus}\n"
            "[rmu] implied lam: {lams}",
            s=sig, k=_num_matvecs, r=r_mu, m=mu_i, l=sig + 1.0 / mu_i,
            sep=jnp.abs(mu_s[0]) / jnp.abs(mu_s[1]),
            mus=mu_s, lams=sig + 1.0 / mu_s,
        )

    def _eigensolve_jax(params_d, data_d):
        d_h = dict(_other_data)
        d_h.update(data_d)
        A = _agni3_assemble(params_d, transforms, profiles, d_h, **kwargs)["A"]
        nA = A.shape[0]
        # (A - sigma I): subtract on the diagonal (no full identity materialized).
        # The manual frees below are EAGER-ONLY: block_until_ready + .delete() force
        # concrete evaluation and cannot run under a jit trace (the optimizer jits
        # _proximal_jvp_blocked_pure). Under trace, skip them -- jit's own buffer
        # management (donation) handles memory, which is the end goal anyway.
        _eager = not isinstance(A, jax.core.Tracer)

        def _solve_at(sig, keep_A):
            """One shift-invert Lanczos solve at shift `sig`.

            Returns (v, lam_mu, sep):
              v      eigenvector (the primal output)
              lam_mu sigma + 1/mu -- the ROBUST eigenvalue estimate
              sep    |mu_1|/|mu_2| -- separation from the near-zero cluster.
                     ~1 means the modes are unresolvable and v is a MIXTURE.

            `keep_A` must be True if another pass will need A (adaptive re-shift);
            otherwise A's buffer is reclaimed as soon as M exists.
            """
            M = A.at[jnp.diag_indices(nA)].add(-sig)
            if _eager:
                jax.block_until_ready(M)  # M built before A's buffer is reclaimed
                if not keep_A:
                    _free(A)
            lu = _jsla.lu_factor(M)
            if _eager:
                jax.block_until_ready(lu)  # LU built before M is reclaimed
                _free(M)

            def _OPinv(b):  # exact shift-invert, GPU triangular solves
                return _jsla.lu_solve(lu, b)

            _tri = decomp.tridiag_sym(_num_matvecs, reortho="full", materialize=True)
            _alg = eig.eigh_partial(_tri)
            _v0 = jnp.asarray(
                np.random.default_rng(0).standard_normal(nA), dtype=M.dtype
            )
            _v0 = _v0 / jnp.linalg.norm(_v0)
            mu, vecs = _alg(_OPinv, _v0)
            # mu are OP eigenvalues 1/(lambda - sigma); largest |mu| == the
            # eigenvalue NEAREST sigma.
            idx = jnp.argmax(jnp.abs(mu))
            v_out = vecs[idx]

            mu_i = mu[idx]
            lam_out = sig + 1.0 / jnp.where(mu_i == 0, jnp.inf, mu_i)
            ordered = jnp.abs(mu[jnp.argsort(-jnp.abs(mu))])
            sep_out = ordered[0] / jnp.where(ordered[1] == 0, jnp.inf, ordered[1])

            _emit_rmu_diagnostic(sig, mu, idx, v_out, _OPinv)

            if _eager:
                jax.block_until_ready(v_out)  # v built before the LU is reclaimed
                _free(lu[0])
            return v_out, lam_out, sep_out

        # ---- PASS 1, at the incoming shift ----
        v, lam_mu, sep_pass1 = _solve_at(sigma, keep_A=_adapt)
        sep = sep_pass1

        # ---- PASS 2 (AGNI_SIGMA_ADAPT=1): re-shift onto THIS solve's own answer --
        #
        # AGNI_SIGMA_MODE=adapt. Rationale/costs: BENCHMARKING.md 10.10. In brief:
        #   * legitimate where carrying sigma between evaluations is not, because NO
        #     STATE CROSSES AN EVALUATION BOUNDARY -- lam_mu was measured at THIS p,
        #     on THIS mesh, so the staleness objection does not apply and the
        #     objective stays a pure function of params.
        #   * unconditional, NOT a `lax.cond` on separation: a data-dependent branch
        #     would make the objective piecewise and confuse the line search.
        #   * costs +11% wall (the assembly is not repeated) and keeps A resident
        #     across pass 1 (+10.1 GB at 32x32x12).
        if _adapt:
            sigma2 = _adapt_factor * jax.lax.stop_gradient(lam_mu)
            # Guard: if lam_mu is not finite or is non-negative, the mode we want
            # does not exist below zero at this p -- keep pass 1 rather than shift
            # onto a meaningless target.
            sigma2 = jnp.where(
                jnp.isfinite(sigma2) & (sigma2 < 0), sigma2, sigma
            )
            v, lam_mu, sep = _solve_at(sigma2, keep_A=False)
            if _rmu_diag:  # AGNI_DIAG=2 only; the re-shift itself always runs
                jax.debug.print(
                    "[adapt] sigma1={s1:.6e} sep1={p1:.4f} -> sigma2={s2:.6e} "
                    "sep2={p2:.4f}  lam_mu={l:.8e}",
                    s1=sigma, p1=sep_pass1, s2=sigma2, p2=sep, l=lam_mu,
                )

        # lam_mu (= sigma + 1/mu) is returned to the CALLER rather than compared
        # here, so the cross-check can run outside this function.
        #
        # TRACER-LEAK RULE (learned the hard way, job 56110541): inside this
        # function -- the custom_vjp fwd -- use ONLY `params_d`/`data_d` and values
        # derived from them. `_op` is built in the OUTER trace (~line 2922); calling
        # `_op["Ax"](v)` here closes over outer tracers from a separate
        # transformation and raises UnexpectedTracerError under `jac`
        # (float64[12288,3,3] = B_blocks escaping a LinearizeTracer). It passes the
        # value tests, which never linearize, and only explodes in the optimizer.
        return v, lam_mu

    def _eigensolve(params_d, data_d):
        if _eigensolver == "jax_lanczos":
            return _eigensolve_jax(params_d, data_d)
        # Both paths return (eigenvector, eigenvalue). The scalar is the Ritz/ARPACK
        # eigenvalue, used only by the cross-check monitor at the call site.
        return jax.pure_callback(
            _assemble_and_solve_host,
            (
                jax.ShapeDtypeStruct((n_keep,), _dtype),
                jax.ShapeDtypeStruct((), _dtype),
            ),
            params_d,
            data_d,
        )

    # A custom differentiation rule is REQUIRED, not stylistic: the callback's inputs
    # are traced, so JAX demands one and `pure_callback` has none ("Pure callbacks do
    # not support JVP", job 56013473). stop_gradient on the OUTPUT cannot help -- the
    # rule is invoked on the way IN.
    #
    # custom_VJP, not custom_jvp: under a custom_jvp the JVP rule recomputes the
    # primal (`_v_primal(*primals)`) DURING linearization, so every reverse-mode
    # jacobian ran the assemble+ARPACK callback TWICE -- once for the value, once
    # inside the rule (measured: 10 dense assembles/outer step where the optimizer
    # asked for 2 fn + 2 jac evals). custom_vjp's fwd runs the callback ONCE and
    # hands the result to bwd, halving the eigsh count on every jacobian.
    @jax.custom_vjp
    def _v_primal(params_d, data_d):
        return _eigensolve(params_d, data_d)

    def _v_primal_fwd(params_d, data_d):
        v_out = _eigensolve(params_d, data_d)
        # residuals: the input pytrees, only so bwd can shape its zero cotangents.
        return v_out, (params_d, data_d)

    def _v_primal_bwd(res, g):
        # dv/dp carries ZERO cotangent to (params, data). This is not a convenience
        # -- it is Hellmann-Feynman. Differentiating the quotient exactly:
        #
        #   dlam/dp = v^T(dA/dp)v/(v^Tv) + 2 (dv/dp)^T (A - lam) v / (v^Tv)
        #                                   \____ identically 0, since Av = lam v
        #
        # so the eigenvector's own derivative cannot contribute; the whole gradient
        # flows through the Ax(v) contraction below, never through v. Same rule
        # `jnp.linalg.eigh` applies internally (why BallooningStability's gradient
        # works). v is FRESH at this p -- never cached, never stale.
        params_d, data_d = res
        return (
            jax.tree_util.tree_map(jnp.zeros_like, params_d),
            jax.tree_util.tree_map(jnp.zeros_like, data_d),
        )

    _v_primal.defvjp(_v_primal_fwd, _v_primal_bwd)

    v, lam_mu = _v_primal(params, _array_data)

    Av = _op["Ax"](v)
    vv = jnp.vdot(v, v)
    lam_R = jnp.real(jnp.vdot(v, Av) / vv)

    # ---- CROSS-CHECK MONITOR (AGNI_DIAG>=1, ON by default) --------------------
    #
    # lam_R (the objective) is FRAGILE -- second order in eigenvector contamination
    # against ||A|| ~ 2.8e7. lam_mu (the Ritz value) is ROBUST. They agree when the
    # eigenvector is clean and diverge when it is not, so the gap flags an
    # untrustworthy evaluation. Full argument + measurements: BENCHMARKING.md 10.4a.
    #
    # TWO INVARIANTS, both learned by breaking them:
    #   * lam_mu is a MONITOR, NEVER the objective. It comes from the custom_vjp
    #     whose bwd returns zero cotangents, so it carries NO GRADIENT -- swapping
    #     it in gives a zero descent direction that looks like clean convergence.
    #   * This must live HERE, not inside _eigensolve_jax: `_op` belongs to THIS
    #     trace, and touching it inside the custom_vjp fwd leaks a tracer under
    #     `jac` (job 56110541).
    if _xcheck:
        _den = jnp.maximum(jnp.abs(lam_mu), 1e-300)
        _gap = jnp.abs(lam_R - lam_mu) / _den
        # SUSPECT on sign disagreement too -- that IS the drift signature
        # (objective says stable, Ritz value says unstable).
        _sign_ok = jnp.sign(lam_R) == jnp.sign(lam_mu)
        _ok = (_gap < _xcheck_tol) & _sign_ok
        # No jnp.where on strings (it takes arrays). Grep for "trusted=False".
        jax.debug.print(
            "[xcheck] lam_R={a:+.8e}  lam_mu={b:+.8e}  rel_gap={g:.3e}"
            "  sign_ok={s}  trusted={t}",
            a=lam_R, b=lam_mu, g=_gap, s=_sign_ok, t=_ok,
        )
    # Norm-wise residual r = ||Ax(v) - lam_R v|| / (|lam_R| ||v||).
    #
    # WARNING: r IS NOT A CORRECTNESS METRIC HERE and must not be thresholded.
    # |lam| ~ 1e-4 sits ~11 orders below ||A|| ~ 2.8e7, so r reports round-off
    # amplified by ~2.7e11. r = 0.3945 is the FLOOR -- a perfectly fresh eigenvector
    # gives exactly that. It is retained only for continuity with the logs.
    # See AGNI_var/dense-eigsh-optimization/WHY_V_CANNOT_BE_CACHED.md (CORRECTION).
    resid = jnp.linalg.norm(Av - lam_R * v) / (
        jnp.abs(lam_R) * jnp.sqrt(jnp.real(vv)) + 1e-300
    )

    data["finite-n lambda3 rayleigh"] = jnp.atleast_1d(lam_R)
    data["finite-n lambda3 rayleigh residual"] = jnp.atleast_1d(resid)
    return data



@register_compute_fun(
    name="finite-n lambda3 matfree pcg",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Experimental preconditioned matrix-free finite-n lambda3 solver",
    dim=1,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "g^rr",
        "g^rv",
        "g^rz",
        "J^theta_PEST",
        "J^zeta",
        "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota",
        "psi_r",
        "psi_rr",
        "p",
        "p_r",
        "a",
    ],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    stable_only="bool: for testing only, materialize and eigendecompose the stable part of the matrix",
    v_guess="ndarray: eigenfunction guess to initialize the iterative eigenvalue solver",
    eigsh_tol="float: tolerance for ARPACK eigsh in matrix-free mode",
    eigsh_maxiter="int: max iterations for ARPACK eigsh in matrix-free mode",
    eigsh_ncv="int: number of Lanczos vectors used by ARPACK eigsh",
    debug_matfree="bool: print matrix-free solver diagnostics",
    sigma="float: shift used by shift-invert solver",
    cg_tol="float: CG tolerance inside the preconditioned shift-invert solve",
    cg_maxiter="int: CG max iterations inside the preconditioned shift-invert solve",
    pcg_preconditioner="str: 'fourier_mode_blocks' (default) or experimental dense 'fourier_band_spd'",
    pcg_bandwidth="int: theta-mode cyclic bandwidth for pcg_preconditioner='fourier_band_spd'",
    pcg_floor_rel="float: relative eigenvalue floor for pcg_preconditioner='fourier_band_spd'",
    pcg_build_batch_size="int: number of Fourier-basis columns to batch while building the preconditioner",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier (rho, theta) "
    "operators instead of separable 1D matrices",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
)
def _AGNI3_matfree_pcg(params, transforms, profiles, data, **kwargs):
    """Experimental PCG wrapper for :func:`_AGNI3_matfree`.

    The preconditioner is built once per ``eq.compute`` call from the current
    equilibrium data. During optimization, the equilibrium-dependent operator
    changes, so this is not automatically amortized across objective calls.
    """
    kwargs = dict(kwargs)
    kwargs.setdefault("matfree_solver", "eigsh_shiftinvert_pcg")
    kwargs.setdefault("pcg_preconditioner", "fourier_mode_blocks")
    out = _AGNI3_matfree(params, transforms, profiles, data, **kwargs)
    if "finite-n lambda3 matfree" in out:
        out["finite-n lambda3 matfree pcg"] = out["finite-n lambda3 matfree"]
    if "finite-n eigenfunction3 matfree" in out:
        out["finite-n eigenfunction3 matfree pcg"] = out[
            "finite-n eigenfunction3 matfree"
        ]
    return out




@register_compute_fun(
    name="finite-n lambda32",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Finite-n lambda3 block dump in AGNI6 format; no reduced-matrix assembly",
    dim=1,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "g^rr",
        "g^rv",
        "g^rz",
        "J^theta_PEST",
        "J^zeta",
        "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota",
        "psi_r",
        "psi_rr",
        "p",
        "p_r",
        "a",
        "R",
        "Z",
    ],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    bc_rho_inner="bool: impose xi^rho Dirichlet BC at rho=0",
    bc_rho_outer="bool: impose xi^rho Dirichlet BC at rho=1",
    gamma="float: adiabatic constant",
    stable_only="bool: ignored here",
    v_guess="ndarray: ignored here",
    gpu_assembly="bool: when True, assemble streamed source blocks on GPU",
    gpu_chunk_size="int: streamed GPU chunk size used for both row panels and k-scan chunks",
    matrix_dump_only="bool: stop after writing scaled source blocks to disk",
    memmap_blocks="bool: use memmap-backed block files for scaled source blocks",
    keep_source_blocks="bool: keep scaled source block files on disk",
)
def _AGNI32(params, transforms, profiles, data, **kwargs):
    """Dump six diagonally-scaled lambda3 source blocks plus B metadata.

    This routine keeps the lambda3 algebra in the (rho, upsilon, zeta) basis,
    but stops before forming the full transformed/reduced matrix. The expensive
    bilinear terms are assembled one block at a time, using a single outer CPU
    loop over destination row panels and one jitted ``lax.scan`` over the shared
    contraction index. The saved outputs are the six diagonally-scaled A blocks,
    the six diagonally-scaled B blocks, and the metadata required for the later
    CPU-side Linv/LinvT/keep reduction.

    This version uses GPU accelerator to calculate terms of the stability matrix
    and assembles them into multiple usints that can be be saved on a CPU.
    The units are later assembles into a giant matrix A_full and solved as an 
    eigenvalue problem. This is done because of current A100 GPU VRAM limitations
    and the size of the stiffness matrix.
    """
    a_N = np.asarray(data["a"]).item()
    axisym = kwargs.get("axisym", False)
    np_dtype = np.complex128 if axisym else np.float64
    B_N = abs(params["Psi"] / (np.pi * a_N**2))
    gamma = kwargs.get("gamma", 10.0)
    n_mode_axisym = kwargs.get("n_mode_axisym", 1)
    bc_rho_inner = kwargs.get("bc_rho_inner", True)
    bc_rho_outer = kwargs.get("bc_rho_outer", True)
    gpu_assembly = kwargs.get("gpu_assembly", True)
    gpu_chunk = int(kwargs.get("gpu_chunk_size", _env5("AGNI_GPU_CHUNK_SIZE", 2048)))
    dump_only = kwargs.get("matrix_dump_only", True)
    memmap_blocks = kwargs.get("memmap_blocks", False)
    keep_source_blocks = kwargs.get("keep_source_blocks", True)

    dump_dir = _env5("AGNI_LAMBDA32_DUMP_DIR", os.getcwd(), "AGNI_LAMBDA32_DUMP_DIR")
    dump_basename = _env5("AGNI_LAMBDA32_DUMP_BASENAME", "finite_n_lambda32", "AGNI_LAMBDA32_DUMP_BASENAME")
    progress_enabled = _env5("AGNI_LAMBDA32_PROGRESS", "0", "AGNI_LAMBDA32_PROGRESS").strip().lower() not in {
        "", "0", "false", "no", "off",
    }
    progress_chunk_every = max(1, int(_env5("AGNI_LAMBDA32_PROGRESS_CHUNK_EVERY", "1", "AGNI_LAMBDA32_PROGRESS_CHUNK_EVERY")))
    progress_t0 = time.time()
    run_id = int(time.time())
    shape_tag = f"{int(np.asarray(transforms['diffmat'].D_rho).shape[0])}x{int(np.asarray(transforms['diffmat'].D_theta).shape[0])}x{int(np.asarray(np.asarray(transforms['diffmat'].D_zeta)).shape[0])}"
    file_root = os.path.join(dump_dir, f"{dump_basename}_{run_id}_{shape_tag}")
    os.makedirs(dump_dir, exist_ok=True)

    def _progress(msg):
        """Print a timestamped lambda6 progress line.

        Parameters
        ----------
        msg : str
            Human-readable description of the current assembly stage. The
            message is emitted only when ``AGNI_LAMBDA32_PROGRESS`` is enabled,
            so callers can use this helper freely without cluttering the main
            algebra with environment checks.
        """
        if progress_enabled:
            print("[finite-n lambda32 progress]", f"t={time.time() - progress_t0:.1f}s", msg, flush=True)

    iota = np.asarray(data["iota"], dtype=np_dtype).reshape(-1, 1)
    iotainv = 1.0 / iota
    psi_r = np.asarray(data["psi_r"], dtype=np_dtype).reshape(-1, 1) / (a_N**2 * B_N)
    psi_r2 = psi_r**2
    psi_r3 = psi_r**3
    iota_psi_r2 = iota * psi_r2
    p0 = mu_0 * np.asarray(data["p"], dtype=np_dtype).reshape(-1, 1) / B_N**2 + 1e-12
    p_r = mu_0 * np.asarray(data["p_r"], dtype=np_dtype).reshape(-1, 1) / B_N**2
    n0 = 1e0

    if axisym:
        D_zeta0 = 1j * n_mode_axisym * np.array([[1]], dtype=np.complex128)
    else:
        D_zeta0 = np.asarray(transforms["diffmat"].D_zeta, dtype=np_dtype)

    D_rho0 = np.asarray(transforms["diffmat"].D_rho, dtype=np_dtype)
    D_theta0 = np.asarray(transforms["diffmat"].D_theta, dtype=np_dtype)
    W_rho = np.asarray(transforms["diffmat"].W_rho, dtype=np_dtype)
    W_theta = np.asarray(transforms["diffmat"].W_theta, dtype=np_dtype)
    W_zeta = np.asarray(transforms["diffmat"].W_zeta, dtype=np_dtype)

    n_rho_max = D_rho0.shape[0]
    n_theta_max = D_theta0.shape[0]
    n_zeta_max = D_zeta0.shape[0]
    n_total = n_rho_max * n_theta_max * n_zeta_max
    n_shell = n_theta_max * n_zeta_max
    n_kchunks = (n_total + gpu_chunk - 1) // gpu_chunk

    W = np.kron(W_rho, np.kron(W_theta, W_zeta)).astype(np_dtype, copy=False).reshape(-1, 1)

    sqrtg = np.asarray(data["sqrt(g)_PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**3
    sqrtg_r = np.asarray(data["(sqrt(g)_PEST_r)|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**3
    sqrtg_v = np.asarray(data["(sqrt(g)_PEST_v)|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**3
    sqrtg_p = np.asarray(data["(sqrt(g)_PEST_p)|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**3

    R = np.asarray(data["R"], dtype=np_dtype).reshape(-1, 1)
    Z = np.asarray(data["Z"], dtype=np_dtype).reshape(-1, 1)

    partial_z_log_sqrtg = (sqrtg_p / sqrtg).reshape(-1)
    partial_r_log_sqrtg = (sqrtg_r / sqrtg).reshape(-1)
    partial_v_log_sqrtg = (sqrtg_v / sqrtg).reshape(-1)
    psi_r_over_sqrtg = psi_r / sqrtg

    g_rr = np.asarray(data["g_rr|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**2
    g_vv = np.asarray(data["g_vv|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**2
    g_pp = np.asarray(data["g_pp|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**2
    g_rv = np.asarray(data["g_rv|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**2
    g_rp = np.asarray(data["g_rp|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**2
    g_vp = np.asarray(data["g_vp|PEST"], dtype=np_dtype).reshape(-1, 1) / a_N**2

    g_sup_rr = np.asarray(data["g^rr"], dtype=np_dtype).reshape(-1, 1) * a_N**2
    g_sup_rv = np.asarray(data["g^rv"], dtype=np_dtype).reshape(-1, 1) * a_N**2
    g_sup_rp = np.asarray(data["g^rz"], dtype=np_dtype).reshape(-1, 1) * a_N**2

    print(f"{file_root}")
    #np.savez(f"{dump_dir}/{file_root}_eq_data.npz", psi_r_over_sqrtg=psi_r_over_sqrtg, psi_r=psi_r, iota=iota, allow_pickle=True)
    #np.savez(f"{dump_dir}/{file_root}_eq_data.npz", psi_r_over_sqrtg=psi_r_over_sqrtg, psi_r=psi_r, iota=iota, g_rr=g_rr, g_vv=g_vv, g_pp=g_pp, g_rv=g_rv, g_rp=g_rp, g_vp=g_vp, R=R, Z=Z, D_rho0=D_rho0, D_theta0=D_theta0, D_zeta0=D_zeta0, sqrtg=sqrtg)
    np.savez(f"{file_root}_eq_data.npz", psi_r_over_sqrtg=psi_r_over_sqrtg, psi_r=psi_r, iota=iota, g_rr=g_rr, g_vv=g_vv, g_pp=g_pp, g_rv=g_rv, g_rp=g_rp, g_vp=g_vp, R=R, Z=Z, D_rho0=D_rho0, D_theta0=D_theta0, D_zeta0=D_zeta0, sqrtg=sqrtg)


    J2 = ((mu_0 * np.asarray(data["|J|"], dtype=np_dtype).reshape(-1, 1)) ** 2) * (a_N / B_N) ** 2
    j_sup_zeta = mu_0 * np.asarray(data["J^zeta"], dtype=np_dtype).reshape(-1, 1) * a_N**2 / B_N
    j_sup_theta = iota * j_sup_zeta + p_r / psi_r
    F = -mu_0 * np.asarray(data["finite-n instability drive"], dtype=np_dtype).reshape(-1, 1) * (1 / B_N) ** 2

    assembly_device = None
    if gpu_assembly:
        assembly_device = next(filter(lambda dev: dev.platform == "gpu", jax.devices()), None)

    def _dev(x):
        """Move host data into the active JAX assembly backend.

        Parameters
        ----------
        x : array-like
            Host scalar or array to stage for the jitted assembly kernels.

        Returns
        -------
        jax.Array
            Device-resident array on the first visible GPU when GPU assembly is
            enabled, otherwise a normal JAX array on the default backend.
        """
        arr = jnp.asarray(x)
        return arr if assembly_device is None else jax.device_put(arr, assembly_device)

    def _host(x):
        """Bring a JAX array back to host memory as a writable NumPy array.

        Parameters
        ----------
        x : jax.Array
            Device-side result produced by one of the panel kernels.

        Returns
        -------
        numpy.ndarray
            Host copy with ``copy=True`` so subsequent in-place updates of the
            destination block never hit NumPy read-only view errors.
        """
        return np.array(jax.device_get(x), copy=True)

    all_idx = np.arange(n_total, dtype=np.int32)
    rho_of = (all_idx // n_shell).astype(np.int32)
    theta_of = ((all_idx // n_zeta_max) % n_theta_max).astype(np.int32)
    zeta_of = (all_idx % n_zeta_max).astype(np.int32)

    all_idx_d = _dev(all_idx)
    all_mask_d = _dev(np.ones(n_total, dtype=np_dtype))
    rho_of_d = _dev(rho_of)
    theta_of_d = _dev(theta_of)
    zeta_of_d = _dev(zeta_of)
    D_rho0_d = _dev(D_rho0)
    D_theta0_d = _dev(D_theta0)
    D_zeta0_d = _dev(D_zeta0)
    partial_r_d = _dev(partial_r_log_sqrtg.astype(np_dtype))
    partial_v_d = _dev(partial_v_log_sqrtg.astype(np_dtype))
    partial_z_d = _dev(partial_z_log_sqrtg.astype(np_dtype))

    k_base = np.arange(n_kchunks, dtype=np.int32)[:, None] * gpu_chunk
    k_off = np.arange(gpu_chunk, dtype=np.int32)[None, :]
    k_idx_mat = k_base + k_off
    k_mask_mat = (k_idx_mat < n_total).astype(np_dtype)
    k_idx_mat = np.minimum(k_idx_mat, max(n_total - 1, 0)).astype(np.int32)
    k_idx_mat_d = _dev(k_idx_mat)
    k_mask_mat_d = _dev(k_mask_mat)

    @partial(jit, static_argnames=("op_name",))
    def _fetch(op_name, row_idx, col_idx, row_mask, col_mask):
        """Build one explicit operator tile directly from the small 1D factors.

        Parameters
        ----------
        op_name : {"D_rho", "D_theta", "D_zeta", "C_rho", "C_theta", "C_zeta"}
            Name of the Kronecker-structured derivative/operator to evaluate.
        row_idx, col_idx : jax.Array
            Flat global indices defining the requested tile.
        row_mask, col_mask : jax.Array
            Padding masks with entries 0 or 1. These zero out padded rows and
            columns in the last row panel or k-chunk without needing shape-
            changing control flow inside the jitted kernel.

        Returns
        -------
        jax.Array
            Dense tile of shape ``(len(row_idx), len(col_idx))`` suitable for
            immediate use in the batched contraction kernels.
        """
        rr = rho_of_d[row_idx][:, None]
        rc = rho_of_d[col_idx][None, :]
        tr = theta_of_d[row_idx][:, None]
        tc = theta_of_d[col_idx][None, :]
        zr = zeta_of_d[row_idx][:, None]
        zc = zeta_of_d[col_idx][None, :]
        base = row_mask[:, None] * col_mask[None, :]
        if op_name == "D_rho":
            tile = D_rho0_d[rr, rc] * (tr == tc) * (zr == zc)
        elif op_name == "D_theta":
            tile = (rr == rc) * D_theta0_d[tr, tc] * (zr == zc)
        elif op_name == "D_zeta":
            tile = (rr == rc) * (tr == tc) * D_zeta0_d[zr, zc]
        elif op_name == "C_rho":
            diagmask = row_idx[:, None] == col_idx[None, :]
            tile = D_rho0_d[rr, rc] * (tr == tc) * (zr == zc) + partial_r_d[row_idx][:, None] * diagmask
        elif op_name == "C_theta":
            diagmask = row_idx[:, None] == col_idx[None, :]
            tile = (rr == rc) * D_theta0_d[tr, tc] * (zr == zc) + partial_v_d[row_idx][:, None] * diagmask
        elif op_name == "C_zeta":
            diagmask = row_idx[:, None] == col_idx[None, :]
            tile = (rr == rc) * (tr == tc) * D_zeta0_d[zr, zc] + partial_z_d[row_idx][:, None] * diagmask
        else:
            raise ValueError(op_name)
        return base * tile

    @partial(jit, static_argnames=("left_name", "right_name"))
    def _panel_db_scan(left_name, right_name, row_idx, row_mask, left_row, left_col, right_row, right_col, alpha):
        """Assemble one destination row panel for a bilinear term using one scan.

        This kernel computes the row panel of

            alpha * L^H * diag(m) * R

        where the middle diagonal factor is represented by the pointwise product
        of ``left_row`` and ``right_row`` and the optional output-side diagonal
        factors are represented by ``left_col`` and ``right_col``. The only
        batched dimension is the shared contraction index ``k``, which is split
        into fixed-size chunks and reduced with ``jax.lax.scan``.
        """
        left_row = jnp.ones(n_total, dtype=jnp.asarray(alpha).dtype) if left_row is None else left_row
        right_row = jnp.ones(n_total, dtype=jnp.asarray(alpha).dtype) if right_row is None else right_row
        left_col_panel = None if left_col is None else jnp.conjugate(left_col[row_idx])
        right_col_all = None if right_col is None else right_col
        out0 = jnp.zeros((gpu_chunk, n_total), dtype=jnp.result_type(left_row, right_row, alpha))

        def body(carry, xs):
            k_idx, k_mask = xs
            left_panel = _fetch(left_name, k_idx, row_idx, k_mask, row_mask)
            right_panel = _fetch(right_name, k_idx, all_idx_d, k_mask, all_mask_d)
            left_panel = left_panel * left_row[k_idx][:, None]
            right_panel = right_panel * right_row[k_idx][:, None]
            part = jnp.matmul(jnp.conjugate(left_panel).T, right_panel)
            if left_col_panel is not None:
                part = left_col_panel[:, None] * part
            if right_col_all is not None:
                part = part * right_col_all[None, :]
            return carry + alpha * (row_mask[:, None] * part), None

        out, _ = jax.lax.scan(body, out0, (k_idx_mat_d, k_mask_mat_d))
        return out

    @partial(jit, static_argnames=("op_name", "transpose"))
    def _panel_scaled(op_name, transpose, row_idx, row_mask, left_row, right_row, alpha):
        """Build one destination row panel for a non-contracted scaled derivative.

        Parameters
        ----------
        op_name : str
            Operator name understood by ``_fetch``.
        transpose : bool
            When ``False`` the panel corresponds to ``diag(left_row) M diag(right_row)``.
            When ``True`` it corresponds to the Hermitian-transposed contribution.
        row_idx, row_mask : jax.Array
            Fixed-size padded row indices and validity mask for the destination
            panel currently being assembled.
        left_row, right_row : jax.Array or None
            Optional diagonal factors multiplying the operator on its left or
            right. ``right_row`` is typically used when the term contains a
            factor like ``D_rho * psi_r^2``.
        alpha : scalar
            Overall scalar coefficient applied to the returned panel.
        """
        if not transpose:
            panel = _fetch(op_name, row_idx, all_idx_d, row_mask, all_mask_d)
            if left_row is not None:
                panel = left_row[row_idx][:, None] * panel
            if right_row is not None:
                panel = panel * right_row[None, :]
            return alpha * (row_mask[:, None] * panel)
        panel = _fetch(op_name, all_idx_d, row_idx, all_mask_d, row_mask)
        if left_row is not None:
            panel = left_row[:, None] * panel
        if right_row is not None:
            panel = panel * right_row[row_idx][None, :]
        return alpha * (row_mask[:, None] * jnp.conjugate(panel.T))

    def _accumulate(dst, term):
        """Accumulate one compact algebraic term into a full CPU-resident block.

        Parameters
        ----------
        dst : numpy.ndarray or numpy.memmap
            Destination source block being assembled on the CPU.
        term : tuple
            Compact descriptor of one lambda3 contribution. The first entry is
            the mode (``diag``, ``db``, ``sym``, ``scaled``, or ``scaled_h``),
            followed by operator names, optional diagonal vectors, and a scalar
            coefficient. The expensive ``db``/``sym`` terms use the jitted
            one-scan kernel above; the direct scaled terms fetch one explicit
            row panel at a time.
        """
        mode = term[0]
        if mode == "diag":
            dst.flat[:: dst.shape[1] + 1] += term[1]
            return
        _, left, right, left_row, left_col, right_row, right_col, alpha = term
        left_row_d = None if left_row is None else _dev(left_row)
        left_col_d = None if left_col is None else _dev(left_col)
        right_row_d = None if right_row is None else _dev(right_row)
        right_col_d = None if right_col is None else _dev(right_col)
        chunk_id = 0
        i0 = 0
        while i0 < n_total:
            i1 = min(i0 + gpu_chunk, n_total)
            if chunk_id % progress_chunk_every == 0:
                _progress(f"{mode} {left}->{right} row_chunk={i0}:{i1}/{n_total}")
            row_idx = np.zeros(gpu_chunk, dtype=np.int32)
            row_mask = np.zeros(gpu_chunk, dtype=np_dtype)
            row_idx[: i1 - i0] = np.arange(i0, i1, dtype=np.int32)
            row_mask[: i1 - i0] = 1
            row_idx_d = _dev(row_idx)
            row_mask_d = _dev(row_mask)
            if mode == "db":
                panel = _host(_panel_db_scan(left, right, row_idx_d, row_mask_d, left_row_d, left_col_d, right_row_d, right_col_d, alpha))
                dst[i0:i1, :] += panel[: i1 - i0, :]
            elif mode == "sym":
                panel = _host(_panel_db_scan(left, right, row_idx_d, row_mask_d, left_row_d, left_col_d, right_row_d, right_col_d, alpha))
                panel = panel[: i1 - i0, :]
                dst[i0:i1, :] += panel
                dst[:, i0:i1] += np.conjugate(panel.T)
            elif mode == "scaled":
                panel = _host(_panel_scaled(left, False, row_idx_d, row_mask_d, left_row_d, right_col_d, alpha))
                dst[i0:i1, :] += panel[: i1 - i0, :]
            elif mode == "scaled_h":
                panel = _host(_panel_scaled(left, True, row_idx_d, row_mask_d, left_row_d, right_col_d, alpha))
                dst[i0:i1, :] += panel[: i1 - i0, :]
            else:
                raise ValueError(mode)
            i0 += gpu_chunk
            chunk_id += 1
        if isinstance(dst, np.memmap):
            dst.flush()

    def _scale_save(label, terms, left_scale, right_scale):
        """Build one full source block, apply final diagonal scaling, and save it.

        Parameters
        ----------
        label : str
            Output suffix such as ``A_rr`` or ``A_uz``.
        terms : list[tuple]
            Compact descriptors consumed by ``_accumulate``.
        left_scale, right_scale : array-like
            Final blockwise ``B^{-1/2}`` diagonal factors applied row-wise and
            column-wise before the block is written to disk.
        """
        _progress(f"building {label}")
        path = f"{file_root}_{label}.npy"
        if memmap_blocks:
            dst = np.lib.format.open_memmap(path, mode="w+", dtype=np_dtype, shape=(n_total, n_total))
            dst[:] = 0
        else:
            dst = np.zeros((n_total, n_total), dtype=np_dtype)
        for term in terms:
            _accumulate(dst, term)
        _progress(f"scaling {label}")
        i0 = 0
        while i0 < n_total:
            i1 = min(i0 + gpu_chunk, n_total)
            dst[i0:i1, :] *= left_scale[i0:i1, None]
            dst[i0:i1, :] *= right_scale[None, :]
            i0 += gpu_chunk
        if isinstance(dst, np.memmap):
            dst.flush()
        else:
            np.save(path, dst)
        del dst
        _progress(f"saved {label} -> {path}")

    ismirror = bool(np.all(np.abs(iota) < 1e-12))
    b_rr = (n0 * (W * psi_r2 * sqrtg * g_rr)).reshape(-1)
    b_uu = (n0 * (W * sqrtg * g_vv)).reshape(-1)
    b_ru = (n0 * (W * psi_r * sqrtg * g_rv)).reshape(-1)
    if ismirror:
        b_zz = (n0 * (W * sqrtg * g_pp)).reshape(-1)
        b_rz = (n0 * (W * psi_r * sqrtg * g_rp)).reshape(-1)
        b_uz = (n0 * (W * sqrtg * g_vp)).reshape(-1)
    else:
        b_zz = (n0 * (W * sqrtg * (g_vv + 2 * iotainv * g_vp + iotainv**2 * g_pp))).reshape(-1)
        b_rz = (n0 * (W * psi_r * sqrtg * (g_rv + iotainv * g_rp))).reshape(-1)
        b_uz = (n0 * (W * sqrtg * (g_vv + iotainv * g_vp))).reshape(-1)

    d = 1.0 / np.sqrt(np.concatenate([b_rr, b_uu, b_zz], axis=0))
    dr = d[:n_total]
    du = d[n_total : 2 * n_total]
    dz = d[2 * n_total :]

    b_rr = b_rr * dr * dr
    b_uu = b_uu * du * du
    b_zz = b_zz * dz * dz
    b_ru = b_ru * dr * du
    b_rz = b_rz * dr * dz
    b_uz = b_uz * du * dz

    B_blocks = np.zeros((n_total, 3, 3), dtype=np_dtype)
    B_blocks[:, 0, 0] = b_rr
    B_blocks[:, 1, 1] = b_uu
    B_blocks[:, 2, 2] = b_zz
    B_blocks[:, 0, 1] = b_ru
    B_blocks[:, 1, 0] = np.conjugate(b_ru)
    B_blocks[:, 0, 2] = b_rz
    B_blocks[:, 2, 0] = np.conjugate(b_rz)
    B_blocks[:, 1, 2] = b_uz
    B_blocks[:, 2, 1] = np.conjugate(b_uz)

    node_ids = np.arange(n_total)
    rho_shell = node_ids // n_shell
    boundary = np.zeros(n_total, dtype=bool)
    if bc_rho_inner:
        boundary |= rho_shell == 0
    if bc_rho_outer:
        boundary |= rho_shell == (n_rho_max - 1)
    B_blocks[boundary, 0, 1] = 0
    B_blocks[boundary, 1, 0] = 0
    B_blocks[boundary, 0, 2] = 0
    B_blocks[boundary, 2, 0] = 0

    _progress("cholesky(B_blocks)")
    L = np.linalg.cholesky(B_blocks)
    Linv = np.linalg.inv(L)
    LinvT = np.conjugate(np.swapaxes(Linv, 1, 2))
    del L

    rho_start = n_shell if bc_rho_inner else 0
    rho_end = n_total - n_shell if bc_rho_outer else n_total
    keep = np.concatenate([np.arange(rho_start, rho_end), np.arange(n_total, 3 * n_total)])

    np.save(f"{file_root}_d.npy", d)
    np.save(f"{file_root}_dr.npy", dr)
    np.save(f"{file_root}_du.npy", du)
    np.save(f"{file_root}_dz.npy", dz)
    np.save(f"{file_root}_b_rr.npy", b_rr)
    np.save(f"{file_root}_b_ru.npy", b_ru)
    np.save(f"{file_root}_b_rz.npy", b_rz)
    np.save(f"{file_root}_b_uu.npy", b_uu)
    np.save(f"{file_root}_b_uz.npy", b_uz)
    np.save(f"{file_root}_b_zz.npy", b_zz)
    np.save(f"{file_root}_Linv.npy", Linv)
    np.save(f"{file_root}_LinvT.npy", LinvT)
    np.save(f"{file_root}_keep.npy", keep)

    facp = (gamma * sqrtg * W * p0).reshape(-1)
    terms_rr = [
        ("db", "D_theta", "D_theta", None, None, (psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr).reshape(-1), None, 1.0),
        ("db", "D_zeta", "D_zeta", None, None, (psi_r_over_sqrtg * psi_r3 * W * g_rr).reshape(-1), None, 1.0),
        ("sym", "D_theta", "D_zeta", None, None, (psi_r_over_sqrtg * iota * psi_r3 * W * g_rr).reshape(-1), None, 1.0),
        ("db", "D_rho", "D_rho", None, iota_psi_r2.reshape(-1), (psi_r_over_sqrtg * W * g_vv / psi_r).reshape(-1), iota_psi_r2.reshape(-1), 1.0),
        ("db", "D_rho", "D_rho", None, psi_r2.reshape(-1), (psi_r_over_sqrtg * W * g_pp / psi_r).reshape(-1), psi_r2.reshape(-1), 1.0),
        ("sym", "D_theta", "D_rho", None, None, (iota * psi_r * psi_r_over_sqrtg * W * g_rv).reshape(-1), iota_psi_r2.reshape(-1), -1.0),
        ("sym", "D_zeta", "D_rho", None, None, (psi_r * psi_r_over_sqrtg * W * g_rv).reshape(-1), iota_psi_r2.reshape(-1), -1.0),
        ("sym", "D_theta", "D_rho", None, None, (iota * psi_r * psi_r_over_sqrtg * W * g_rp).reshape(-1), psi_r2.reshape(-1), -1.0),
        ("sym", "D_zeta", "D_rho", None, None, (psi_r * psi_r_over_sqrtg * W * g_rp).reshape(-1), psi_r2.reshape(-1), -1.0),
        ("sym", "D_rho", "D_rho", None, iota_psi_r2.reshape(-1), (psi_r_over_sqrtg * W * g_vp / psi_r).reshape(-1), psi_r2.reshape(-1), 1.0),
        ("scaled", "D_theta", None, (W * psi_r3 * sqrtg * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv) / g_sup_rr).reshape(-1) * iota.reshape(-1), None, None, None, -1.0),
        ("scaled", "D_zeta", None, (W * psi_r3 * sqrtg * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv) / g_sup_rr).reshape(-1), None, None, None, -1.0),
        ("scaled", "D_rho", None, (W * sqrtg * psi_r * j_sup_zeta).reshape(-1), None, None, iota_psi_r2.reshape(-1), -1.0),
        ("scaled", "D_rho", None, (W * sqrtg * psi_r * j_sup_theta).reshape(-1), None, None, psi_r2.reshape(-1), -1.0),
        ("scaled_h", "D_theta", None, (W * psi_r3 * sqrtg * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv) / g_sup_rr).reshape(-1) * iota.reshape(-1), None, None, None, -1.0),
        ("scaled_h", "D_zeta", None, (W * psi_r3 * sqrtg * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv) / g_sup_rr).reshape(-1), None, None, None, -1.0),
        ("scaled_h", "D_rho", None, (W * sqrtg * psi_r * j_sup_zeta).reshape(-1), None, None, iota_psi_r2.reshape(-1), -1.0),
        ("scaled_h", "D_rho", None, (W * sqrtg * psi_r * j_sup_theta).reshape(-1), None, None, psi_r2.reshape(-1), -1.0),
        ("db", "C_rho", "C_rho", None, psi_r.reshape(-1), facp, psi_r.reshape(-1), 1.0),
        ("diag", (psi_r2 * W * sqrtg * J2).reshape(-1) + (W * psi_r2 * sqrtg * F).reshape(-1)),
    ]
    terms_ru = [
        ("db", "D_rho", "D_zeta", None, iota_psi_r2.reshape(-1), (psi_r_over_sqrtg * W * g_vv).reshape(-1), None, -1.0),
        ("db", "D_rho", "D_theta", None, psi_r2.reshape(-1), (psi_r_over_sqrtg * W * g_pp).reshape(-1), None, 1.0),
        ("db", "D_theta", "D_zeta", None, None, (iota * psi_r2 * psi_r_over_sqrtg * W * g_rv).reshape(-1), None, 1.0),
        ("db", "D_zeta", "D_zeta", None, None, (psi_r2 * psi_r_over_sqrtg * W * g_rv).reshape(-1), None, 1.0),
        ("db", "D_theta", "D_theta", None, None, (iota * psi_r2 * psi_r_over_sqrtg * W * g_rp).reshape(-1), None, -1.0),
        ("db", "D_zeta", "D_theta", None, None, (psi_r2 * psi_r_over_sqrtg * W * g_rp).reshape(-1), None, -1.0),
        ("db", "D_rho", "D_zeta", None, psi_r2.reshape(-1), (psi_r_over_sqrtg * W * g_vp).reshape(-1), None, -1.0),
        ("db", "D_rho", "D_theta", None, iota_psi_r2.reshape(-1), (psi_r_over_sqrtg * W * g_vp).reshape(-1), None, 1.0),
        ("scaled", "D_theta", None, (-(W * psi_r2 * sqrtg * j_sup_theta)).reshape(-1), None, None, None, 1.0),
        ("scaled", "D_zeta", None, (W * psi_r2 * sqrtg * j_sup_zeta).reshape(-1), None, None, None, 1.0),
        ("db", "C_rho", "C_theta", None, psi_r.reshape(-1), facp, None, 1.0),
    ]
    terms_rz = [
        ("db", "C_rho", "C_theta", None, psi_r.reshape(-1), facp, None, 1.0),
        ("db", "C_rho", "C_zeta", None, psi_r.reshape(-1), facp, iotainv.reshape(-1), 1.0),
    ]
    terms_uu = [
        ("sym", "D_zeta", "D_zeta", None, None, (psi_r_over_sqrtg * psi_r * W * g_vv).reshape(-1), None, 0.5),
        ("sym", "D_theta", "D_theta", None, None, (psi_r_over_sqrtg * psi_r * W * g_pp).reshape(-1), None, 0.5),
        ("sym", "D_zeta", "D_theta", None, None, (psi_r_over_sqrtg * W * psi_r * g_vp).reshape(-1), None, -1.0),
        ("db", "C_theta", "C_theta", None, None, facp, None, 1.0),
    ]
    terms_uz = [
        ("db", "C_theta", "C_theta", None, None, facp, None, 1.0),
        ("db", "C_theta", "C_zeta", None, None, facp, iotainv.reshape(-1), 1.0),
    ]
    terms_zz = [
        ("db", "C_theta", "C_theta", None, None, facp, None, 1.0),
        ("sym", "C_theta", "C_zeta", None, None, facp, iotainv.reshape(-1), 1.0),
        ("db", "C_zeta", "C_zeta", None, iotainv.reshape(-1), facp, iotainv.reshape(-1), 1.0),
    ]

    _progress(f"assembly config n_total={n_total} n_rho={n_rho_max} n_theta={n_theta_max} n_zeta={n_zeta_max} chunk={gpu_chunk} gpu_assembly={gpu_assembly}")
    _scale_save("A_rr", terms_rr, dr, dr)
    _scale_save("A_ru", terms_ru, dr, du)
    _scale_save("A_rz", terms_rz, dr, dz)
    _scale_save("A_uu", terms_uu, du, du)
    _scale_save("A_uz", terms_uz, du, dz)
    _scale_save("A_zz", terms_zz, dz, dz)

    if not keep_source_blocks:
        pass

    if not dump_only:
        raise NotImplementedError("finite-n lambda32 block-dump mode only; assemble/eigensolve in separate CPU script")

    data["finite-n lambda32"] = jnp.asarray(np.array([np.nan]))
    data["finite-n eigenfunction32"] = jnp.asarray(np.full((1,), np.nan))
    data["finite-n xi"] = jnp.asarray(np.full((1,), np.nan))
    data["finite-n deltaB"] = jnp.asarray(np.full((1,), np.nan))
    data["finite-n deltaB_r"] = jnp.asarray(np.full((1,), np.nan))
    data["finite-n deltaB_v"] = jnp.asarray(np.full((1,), np.nan))
    data["finite-n deltaB_z"] = jnp.asarray(np.full((1,), np.nan))
    return data




@register_compute_fun(
    name="finite-n eigenfunction32",
    label="\\xi",
    units="~",
    units_long="None",
    description="Finite-n eigenfunction",
    dim=5,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=["finite-n lambda32"],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    stable_only="bool: ignored here",
)
def _AGNI_eigenfunction32(params, transforms, profiles, data, **kwargs):
    return data



@register_compute_fun(
    name="finite-n lambda",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate"
    + "using the most compact representation of diffmatrices",
    dim=1,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "g_rr|PEST",
        "g_rv|PEST",
        "g_rp|PEST",
        "g_vv|PEST",
        "g_vp|PEST",
        "g_pp|PEST",
        "g^rr",
        "g^rv",
        "g^rz",
        "J^theta_PEST",
        "J^zeta",
        "|J|",
        "sqrt(g)_PEST",
        "(sqrt(g)_PEST_r)|PEST",
        "(sqrt(g)_PEST_v)|PEST",
        "(sqrt(g)_PEST_p)|PEST",
        "finite-n instability drive",
        "iota",
        "psi_r",
        "psi_rr",
        "p",
        "p_r",
        "a",
    ],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    stable_only="bool: for testing only, materialize "
    + "and eigendecompose the stable part of the matrix",
    v_guess="ndarray: eigenfunction guess to initialize the "
    + "iterative eigenvalue solver",
    coupled_rt="bool: use full 2D Zernike-Fourier (rho, theta) diffmatrices",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
    sigma="float: shift for the shift-invert eigensolver (default -1.0)",
    f_scale="float: multiplier on the instability drive F (default 1.0); use "
    ">1 to isolate the physical unstable mode, then continue back to 1",
    full_spectrum="bool: if True, dense-eigendecompose the full reduced matrix "
    "with scipy.linalg.eigh and store every eigenvalue under "
    "'finite-n lambda spectrum'; the returned dominant eigenmode is unchanged. "
    "Default False (iterative eigsh for the single dominant mode).",
    eigsh_tol="float: ARPACK eigsh convergence tolerance; coupled penalty path "
    "defaults to 1e-5",
    eigsh_maxiter="int: optional ARPACK eigsh iteration cap",
)
def _AGNI(params, transforms, profiles, data, **kwargs):
    """
    AGNI: Analysis of Global Normal-modes in Ideal MHD.

    Based on the original source here:
    https://github.com/rahulgaur104/AGNI/tree/master

    A finite-n stability eigenvalue solver.
    Currenly only finds fixed boundary unstable modes at
    low to medium resolution.

    This version of the code keeps the derivatives of the form
    partial_rho (iota psi' xi^rho) more compact which leads to
    fewer terms and even order derivatives. For this version
    the PSD version of A is actually very close to PSD ~ 1e-12.
    B is perfectly PSD
    """
    a_N = data["a"]
    B_N = abs(params["Psi"] / (jnp.pi * a_N**2))

    iota = data["iota"][:, None]
    iotainv = (1 / data["iota"])[:, None]

    psi_r = data["psi_r"][:, None] / (a_N**2 * B_N)

    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r2 = iota * psi_r2

    # Add a tiny shift because sometimes the pressure can be
    # slightly negative in the edge
    p0 = mu_0 * data["p"][:, None] /B_N**2 + 1e-12
    p_r = mu_0 * data["p_r"][:, None] /B_N**2 

    # Arbitrary choice. Mostly used to decide the range of eigenvalues of
    # the mass matrix. Pre-conditioning should remove this factor
    n0 = 1e0

    axisym = kwargs.get("axisym", False)

    # Large gamma is an alternate way to impose incompressibility
    gamma = kwargs.get("gamma", 10.0)

    # For axisymmetric equilibria n_mode_axisym will decide the toroidal
    # mode number to analyze.
    n_mode_axisym = kwargs.get("n_mode_axisym", 1)
    incompressible = kwargs.get("incompressible", False)
    bc_rho_inner = kwargs.get("bc_rho_inner", True)
    bc_rho_outer = kwargs.get("bc_rho_outer", True)

    def _cT(x):
        return jnp.conjugate(jnp.transpose(x))

    if axisym:
        if n_mode_axisym == 0 and incompressible:
            return NotImplementedError
        else:
            # Each componenet of xi can be written as the Fourier sum of
            # two modes in the toroidal direction
            D_zeta0 = 1j * n_mode_axisym * jnp.array([[1]])
    else:
        D_zeta0 = transforms["diffmat"].D_zeta

    # Get differentiation matrices
    D_rho0 = transforms["diffmat"].D_rho
    D_theta0 = transforms["diffmat"].D_theta

    W_rho = transforms["diffmat"].W_rho
    W_theta = transforms["diffmat"].W_theta
    W_zeta = transforms["diffmat"].W_zeta

    # Square matrix.
    # When `coupled_rt` is set, D_rho0/D_theta0 are the full, non-separable 2D
    # (rho, theta) Zernike-Fourier operators of shape (n_rho*n_theta,)*2, so the
    # per-direction node counts must be supplied explicitly via kwargs.
    coupled_rt = kwargs.get("coupled_rt", False)
    if coupled_rt:
        n_rho_max = kwargs["n_rho_coupled"]
        n_theta_max = kwargs["n_theta_coupled"]
    else:
        n_rho_max = D_rho0.shape[0]
        n_theta_max = D_theta0.shape[0]
    n_zeta_max = D_zeta0.shape[0]
    n_total = n_rho_max * n_theta_max * n_zeta_max

    # Dense solver memory estimate (rough):
    # 3*(3N x 3N) matrices + 2*reduced matrices + 5*(N x N) derivative operators.
    n_shell = n_theta_max * n_zeta_max
    n_keep_est = 3 * n_total - 2 * n_shell
    full_gb = ((3 * n_total) ** 2) * 8 / (1024**3)
    keep_gb = (n_keep_est**2) * 8 / (1024**3)
    d_gb = (n_total**2) * 8 / (1024**3)

    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    if coupled_rt:
        # D_rho0/D_theta0 already couple (rho, theta); only tensor with zeta.
        I_rt0 = jax.lax.stop_gradient(jnp.eye(n_rho_max * n_theta_max))
        D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, I_zeta0))
        D_theta = jax.lax.stop_gradient(jnp.kron(D_theta0, I_zeta0))
        D_zeta = jax.lax.stop_gradient(jnp.kron(I_rt0, D_zeta0))
        D_thetaT = jax.lax.stop_gradient(jnp.kron(_cT(D_theta0), I_zeta0))
        D_zetaT = jax.lax.stop_gradient(jnp.kron(I_rt0, _cT(D_zeta0)))
    else:
        I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
        I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
        D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
        D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
        D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))
        D_thetaT = jax.lax.stop_gradient(
            jnp.kron(I_rho0, jnp.kron(_cT(D_theta0), I_zeta0))
        )
        D_zetaT = jax.lax.stop_gradient(
            jnp.kron(I_rho0, jnp.kron(I_theta0, _cT(D_zeta0)))
        )

    # Quadrature weights still factorize (tensor-product) in both modes.
    W = jnp.kron(W_rho, jnp.kron(W_theta, W_zeta))[:, None]

    # Define block indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)

    ## Create the full matrix
    _agni_mem_trace(
        kwargs,
        "[finite-n lambda:dense] materializing dense A/B",
        f"A_shape={(3 * n_total, 3 * n_total)}",
    )
    if axisym:
        A = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.complex128)
        B = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.complex128)
    else:
        A = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.float64)
        B = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.float64)

    sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3

    sqrtg_r = data["(sqrt(g)_PEST_r)|PEST"][:, None] * 1 / a_N**3
    sqrtg_v = data["(sqrt(g)_PEST_v)|PEST"][:, None] * 1 / a_N**3
    sqrtg_p = data["(sqrt(g)_PEST_p)|PEST"][:, None] * 1 / a_N**3

    partial_z_log_sqrtg = (sqrtg_p / sqrtg).flatten()
    partial_r_log_sqrtg = (sqrtg_r / sqrtg).flatten()
    partial_v_log_sqrtg = (sqrtg_v / sqrtg).flatten()

    psi_r_over_sqrtg = psi_r / sqrtg

    g_rr = data["g_rr|PEST"][:, None] * 1 / a_N**2
    g_vv = data["g_vv|PEST"][:, None] * 1 / a_N**2
    g_pp = data["g_pp|PEST"][:, None] * 1 / a_N**2  # finite on-axis

    g_rv = data["g_rv|PEST"][:, None] * 1 / a_N**2
    g_rp = data["g_rp|PEST"][:, None] * 1 / a_N**2
    g_vp = data["g_vp|PEST"][:, None] * 1 / a_N**2

    g_sup_rr = data["g^rr"][:, None] * a_N**2
    g_sup_rv = data["g^rv"][:, None] * a_N**2
    g_sup_rp = data["g^rz"][:, None] * a_N**2

    J2 = ((mu_0 * data["|J|"]) ** 2)[:, None] * (a_N / B_N) ** 2
    j_sup_zeta = mu_0 * data["J^zeta"][:, None] * a_N**2 / B_N
    j_sup_theta = iota * j_sup_zeta + p_r / psi_r
    #j_sup_theta = mu_0 * data["J^theta_PEST"][:, None] * a_N**2 / B_N

    # instability drive term. f_scale (default 1) temporarily amplifies the drive
    # so callers can isolate the physical unstable mode at f_scale>1, then continue
    # back to f_scale=1 using that eigenvalue/eigenfunction as sigma/v_guess.
    f_scale = kwargs.get("f_scale", 1.0)
    F = -1 * f_scale * mu_0 * data["finite-n instability drive"][:, None] * (1 / B_N) ** 2

    C_zeta = jnp.diag(partial_z_log_sqrtg) + D_zeta
    C_rho = jnp.diag(partial_r_log_sqrtg) + D_rho  # (n_total, n_total)
    C_theta = jnp.diag(partial_v_log_sqrtg) + D_theta

    ####################
    ####----Q²_ρρ----###
    ####################
    A = A.at[rho_idx, rho_idx].add(
        D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * D_theta)
        + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * g_rr) * D_zeta)
        + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta)
        + _cT((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta) @ D_theta
    )

    ####################
    ####----Q²_ϑϑ ---###
    ####################
    # enforcing symmetry exactly
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + _cT((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta) @ D_zeta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + _cT((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta) @ D_zeta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1.0 * (D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta))
    )

    A = A.at[rho_idx, rho_idx].add(
        +_cT(D_rho * iota_psi_r2.T)
        @ ((psi_r_over_sqrtg * W * g_vv / psi_r) * (D_rho * iota_psi_r2.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        -1 * _cT(D_rho * iota_psi_r2.T) @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        1 * _cT(D_rho * iota_psi_r2.T) @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    ####################
    ####----Q²_ζζ---####
    ####################
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            _cT(D_theta) @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + _cT((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta) @ D_theta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            _cT(D_theta) @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + _cT((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta) @ D_theta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1 * _cT(D_theta) @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, rho_idx].add(
        +_cT(D_rho * psi_r2.T)
        @ ((psi_r_over_sqrtg * W * g_pp / psi_r) * (D_rho * psi_r2.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        1 * _cT(D_rho * psi_r2.T) @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        -1 * _cT(D_rho * psi_r2.T) @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    ####################
    ####----Q²_ρϑ----###
    ####################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            _cT(D_theta)
            @ ((iota * psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            + _cT(D_zeta)
            @ ((psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            _cT((iota * psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            @ D_theta
            + _cT((psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        _cT(D_theta) @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        + _cT(D_zeta) @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            _cT(D_theta) @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
            + _cT(D_zeta) @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    ######################
    ####-----Q²_ρζ-----###
    ######################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            _cT(D_theta)
            @ ((iota * psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
            + _cT(D_zeta) @ ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            _cT((iota * psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
            @ D_theta
            + _cT((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T)) @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            _cT(D_theta) @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + _cT(D_zeta) @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            _cT(D_theta) @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + _cT(D_zeta) @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )

    ##########################
    ######-----Q²_ϑζ-----#####
    ##########################
    A = A.at[theta_idx, theta_idx].add(
        -1
        * (
            _cT(D_zeta) @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + _cT((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta) @ D_zeta
        )
    )

    A = A.at[zeta_idx, zeta_idx].add(
        -1
        * (
            _cT(D_zeta) @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + _cT((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta) @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            _cT(D_rho * psi_r2.T) @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
            - _cT(D_rho * iota_psi_r2.T) @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            _cT(D_rho * psi_r2.T) @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
            - _cT(D_rho * iota_psi_r2.T) @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        _cT(D_zeta) @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
        + _cT((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta) @ D_zeta
    )

    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            _cT(D_rho * iota_psi_r2.T)
            @ ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * psi_r2.T))
        )
    )
    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            _cT((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * psi_r2.T))
            @ (D_rho * iota_psi_r2.T)
        )
    )

    # Mixed Q-J term ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐
    # \xi^{\rho} (\mathbf{J} \times \nabla\rho)/|\nabla \rho|^2 \cdot \mathbf{Q}
    A = A.at[rho_idx, rho_idx].add(
        -1.
        * (
            (
                W
                * psi_r3
                * sqrtg
                * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                / g_sup_rr
            )
            * (iota * D_theta + D_zeta)
            + (W * sqrtg * psi_r * j_sup_zeta) * (D_rho * iota_psi_r2.T)
            + (W * sqrtg * psi_r * j_sup_theta) * (D_rho * psi_r2.T)
        )
    )

    # ρ-ρ block transposed for symmetry
    A = A.at[rho_idx, rho_idx].add(
        -1.
        * (
            _cT(
                (
                    W
                    * psi_r3
                    * sqrtg
                    * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                    / g_sup_rr
                )
                * (iota * D_theta + D_zeta)
            )
            + _cT((W * sqrtg * psi_r * j_sup_zeta) * (D_rho * iota_psi_r2.T))
            + _cT((W * sqrtg * psi_r * j_sup_theta) * (D_rho * psi_r2.T))
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -(1. * W * psi_r2 * sqrtg * j_sup_theta) * D_theta
        + (1. * W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
    )
    A = A.at[rho_idx, zeta_idx].add(
        +(1. * W * psi_r2 * sqrtg * j_sup_theta) * D_theta
        - (1. * W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
    )

    ## diagonal |J|² term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((psi_r2 * W * sqrtg * J2).flatten()))

    #A = A.at[theta_idx, rho_idx].set(_cT(A[rho_idx, theta_idx]))
    #A = A.at[zeta_idx, rho_idx].set(_cT(A[rho_idx, zeta_idx]))
    #A = A.at[zeta_idx, theta_idx].set(_cT(A[theta_idx, zeta_idx]))

    #w, _ = jnp.linalg.eig(A) 

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(jnp.diag(n0 * (W * psi_r2 * sqrtg * g_rr).flatten()))
    B = B.at[theta_idx, theta_idx].add(jnp.diag(n0 * (W * sqrtg * g_vv).flatten()))

    B = B.at[rho_idx, theta_idx].add(
        jnp.diag(n0 * (W * psi_r * sqrtg * g_rv).flatten())
    )

    # typical in magnetic mirrors
    ismirror = jnp.all(jnp.abs(iota) < 1e-12)

    if ismirror:
        B = B.at[zeta_idx, zeta_idx].add(jnp.diag(n0 * (W * sqrtg * g_pp).flatten()))
        B = B.at[rho_idx, zeta_idx].add(
            jnp.diag(n0 * (W * psi_r * sqrtg * g_rp).flatten())
        )
        B = B.at[theta_idx, zeta_idx].add(jnp.diag(n0 * (W * sqrtg * g_vp).flatten()))
    else:
        B = B.at[zeta_idx, zeta_idx].add(
            jnp.diag(n0 * (W * iotainv**2 * sqrtg * g_pp).flatten())
        )
        B = B.at[rho_idx, zeta_idx].add(
            jnp.diag(n0 * (W * psi_r * iotainv * sqrtg * g_rp).flatten())
        )
        B = B.at[theta_idx, zeta_idx].add(
            jnp.diag(n0 * (W * iotainv * sqrtg * g_vp).flatten())
        )

    if incompressible is False:
        # purely stabilizing and doesn't change the marginal stability
        # To improve performance set exact to False
        A = A.at[rho_idx, rho_idx].add(
            _cT(C_rho * psi_r.T) @ ((gamma * sqrtg * W * p0) * (C_rho * psi_r.T))
        )
        A = A.at[theta_idx, theta_idx].add(
            _cT(C_theta) @ ((gamma * sqrtg * W * p0) * C_theta)
        )
        A = A.at[rho_idx, theta_idx].add(
            _cT(C_rho * psi_r.T) @ ((gamma * sqrtg * W * p0) * C_theta)
        )

        if ismirror:
            A = A.at[zeta_idx, zeta_idx].add(
                _cT(C_zeta) @ ((gamma * sqrtg * W * p0) * (C_zeta))
            )
            A = A.at[rho_idx, zeta_idx].add(
                _cT(C_rho * psi_r.T) @ ((gamma * sqrtg * W * p0) * (C_zeta))
            )
            A = A.at[theta_idx, zeta_idx].add(
                _cT(C_theta) @ ((gamma * sqrtg * W * p0) * (C_zeta))
            )
        else:
            A = A.at[zeta_idx, zeta_idx].add(
                _cT(C_zeta * iotainv.T)
                @ ((gamma * sqrtg * W * p0) * (C_zeta * iotainv.T))
            )
            A = A.at[rho_idx, zeta_idx].add(
                _cT(C_rho * psi_r.T) @ ((gamma * sqrtg * W * p0) * (C_zeta * iotainv.T))
            )
            A = A.at[theta_idx, zeta_idx].add(
                _cT(C_theta) @ ((gamma * sqrtg * W * p0) * (C_zeta * iotainv.T))
            )

    ### Instability drive term
    Au = jnp.zeros((3 * n_total, 3 * n_total))
    Au = Au.at[rho_idx, rho_idx].add(jnp.diag((W * psi_r2 * sqrtg * F).flatten()))

    rt_size = n_rho_max * n_theta_max
    zernike_penalty_alpha, Q_rt, penalty_rank = _get_zernike_penalty(
        transforms, rt_size
    )
    if coupled_rt and zernike_penalty_alpha > 0.0:
        Q = Q_rt if n_zeta_max == 1 else np.kron(Q_rt, np.eye(n_zeta_max))
        penalty = jnp.asarray(zernike_penalty_alpha * Q, dtype=A.dtype)
        A = A.at[rho_idx, rho_idx].add(penalty)
        A = A.at[theta_idx, theta_idx].add(penalty)
        A = A.at[zeta_idx, zeta_idx].add(penalty)
        rank_msg = "unknown" if penalty_rank is None else str(penalty_rank)
        penalized_msg = (
            "unknown" if penalty_rank is None else str(rt_size - penalty_rank)
        )
        print(
            "[finite-n lambda:coupled penalty]",
            f"alpha={zernike_penalty_alpha:.3e}",
            f"rank={rank_msg}/{rt_size}",
            f"penalized_rt={penalized_msg}",
            flush=True,
        )

    A = A.at[theta_idx, rho_idx].set(_cT(A[rho_idx, theta_idx]))
    A = A.at[zeta_idx, rho_idx].set(_cT(A[rho_idx, zeta_idx]))
    A = A.at[zeta_idx, theta_idx].set(_cT(A[theta_idx, zeta_idx]))

    B = B.at[theta_idx, rho_idx].set(_cT(B[rho_idx, theta_idx]))
    B = B.at[zeta_idx, rho_idx].set(_cT(B[rho_idx, zeta_idx]))
    B = B.at[zeta_idx, theta_idx].set(_cT(B[theta_idx, zeta_idx]))

    #D = jnp.diag(1 / jnp.sqrt(jnp.diag(B)))

    ## Preconditioning improves B, does not affect A
    #A = D @ (A @ D.T)
    #Au = D @ (Au @ D.T)
    #B = D @ (B @ D.T)

    d = 1 / jnp.sqrt(jnp.diag(B))  # 1D array

    # D @ X @ D.T for diagonal D is just d[:, None] * X * d[None, :]
    A = d[:, None] * A * d[None, :]
    Au = d[:, None] * Au * d[None, :]
    B = d[:, None] * B * d[None, :]

    #w, _ = jnp.linalg.eigh(B)
    #print(w[:50])

    au_diag = jnp.diagonal(Au)[:n_total]

    def component_to_node_permutn(N: int) -> jnp.ndarray:
        """
        Build the permutation that converts component-major ordering to node-major.

        Component-major vector layout (length 3N):
            [ rho_1..N | theta_1..N | zeta_1..N ]

        Node-major vector layout (length 3N):
            [ rho_1, theta_1, zeta_1 | ... | rho_N, theta_N, zeta_N ]

        The returned permutation `p` satisfies:
            x_node = x_comp[p]
            M_node = M_comp[p][:, p]

        Parameters
        ----------
        N : int
            Number of spatial nodes per component.

        Returns
        -------
        jnp.ndarray, shape (3*N,)
            Permutation indices from component-major to node-major.
        """
        k = jnp.arange(N, dtype=jnp.int64)

        perm = jnp.empty(3 * N, dtype=jnp.int64)
        perm = perm.at[3 * k + 0].set(k)
        perm = perm.at[3 * k + 1].set(N + k)
        perm = perm.at[3 * k + 2].set(2 * N + k)

        return perm

    # TODO: B_blocks will always be real for axisym=True, complex data type
    # is used to avoid trivial dtype-related errors. Fix later!
    if axisym:
        B_blocks = jnp.zeros((n_total, 3, 3), dtype=jnp.complex128)
        I3 = jnp.tile(jnp.eye(3, dtype=jnp.complex128), (n_total, 1, 1))
    else:
        B_blocks = jnp.zeros((n_total, 3, 3))
        I3 = jnp.tile(jnp.eye(3), (n_total, 1, 1))

    B_blocks = B_blocks.at[:, 0, 0].set(jnp.diag(B[rho_idx, rho_idx]))
    B_blocks = B_blocks.at[:, 1, 1].set(jnp.diag(B[theta_idx, theta_idx]))
    B_blocks = B_blocks.at[:, 2, 2].set(jnp.diag(B[zeta_idx, zeta_idx]))

    B_blocks = B_blocks.at[:, 0, 1].set(jnp.diag(B[rho_idx, theta_idx]))
    B_blocks = B_blocks.at[:, 1, 0].set(jnp.diag(B[theta_idx, rho_idx]))

    B_blocks = B_blocks.at[:, 2, 0].set(jnp.diag(B[rho_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 0, 2].set(jnp.diag(B[zeta_idx, rho_idx]))

    B_blocks = B_blocks.at[:, 1, 2].set(jnp.diag(B[theta_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 2, 1].set(jnp.diag(B[zeta_idx, theta_idx]))

    # Enforce physical ξ^ρ BC in the per-node blocks
    # rho index == 0 and n_rho_max - 1
    n_per_shell = n_theta_max * n_zeta_max
    node_ids    = jnp.arange(n_total)
    rho_shell   = node_ids // n_per_shell
    boundary = jnp.zeros(n_total, dtype=bool)
    #if bc_rho_inner:
    #    boundary = boundary | (rho_shell == 0)
    #if bc_rho_outer:
    boundary = (rho_shell == 0) | (rho_shell == (n_rho_max - 1))

    # Eliminate ρ–θ and ρ–ζ couplings corresponding to the physical BC 
    # on ξ^ρ = 0 at the inner and outer radial boundaries.
    B_blocks = B_blocks.at[boundary, 0, 1].set(0)
    B_blocks = B_blocks.at[boundary, 1, 0].set(0)
    B_blocks = B_blocks.at[boundary, 0, 2].set(0)
    B_blocks = B_blocks.at[boundary, 2, 0].set(0)

    L = jnp.linalg.cholesky(B_blocks)  # (N,3,3)

    Linv = jax.lax.linalg.triangular_solve(L, I3, left_side=True, lower=True)  # (N,3,3)

    # components to node permutations
    p = component_to_node_permutn(n_total)
    A = A[p][:, p]
    Au = Au[p][:, p]

    # L^-1 A L^-T
    A = A.reshape(n_total, 3, n_total, 3)
    A = jnp.einsum("ikl,iljq,jbq->ikjb", Linv, A, Linv)

    # Inject transformed instability-drive contribution while A is still
    # node-major 4D to avoid materializing the full Au matrix.
    L0 = Linv[:, :, 0]  # (N, 3)

    au_node = au_diag[:, None, None] * L0[:, :, None] * L0[:, None, :]  # (N, 3, 3)
    i = jnp.arange(n_total)
    A = A.at[i, :, i, :].add(au_node)

    A = A.reshape(3 * n_total, 3 * n_total)

    Au = Au.reshape(n_total, 3, n_total, 3)
    Au = jnp.einsum("ikl,iljq,jbq->ikjb", Linv, Au, Linv)
    Au = Au.reshape(3 * n_total, 3 * n_total)

    # node to component permutation
    pinv = jnp.empty_like(p)
    pinv = pinv.at[p].set(jnp.arange(3 * n_total))

    A = A[pinv][:, pinv]
    Au = Au[pinv][:, pinv]

    # store indices needed to apply dirichlet BC to ξ^ρ
    n_shell = n_theta_max * n_zeta_max
    rho_start = n_shell if bc_rho_inner else 0
    rho_end = n_total - n_shell if bc_rho_outer else n_total
    keep_1 = jnp.arange(rho_start, rho_end)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    ## store indices needed to apply dirichlet BC to ξ^ρ
    #keep_1 = jnp.arange(0., n_total - n_theta_max * n_zeta_max)
    #keep_2 = jnp.arange(n_total, 3 * n_total)
    #keep = jnp.concatenate([keep_1, keep_2])

    if incompressible:  # Only enforce incompressibility here
        # ∇⋅𝛏 = C_ρ ξ^ρ + C_θ ξ^θ + C_ζ ξ^ζ

        ## Assemble L_full from blocks of L (only for comparison)
        # --no-verify Linv_full = _assemble_diagblocks_comp_major(Linv, rho_idx)
        # --no-verify L_test = jnp.linalg.cholesky(B)
        ##max|Linv_full - L_test⁻¹| ≈ 3.55e-15

        # C.shape (N, 3N)
        d_r = d[rho_idx]
        d_v = d[theta_idx]
        d_z = d[zeta_idx]

        C_zeta = (C_zeta * d_z[None, :]) * iotainv.T
        C_rho = (C_rho * d_r[None, :]) * psi_r.T
        C_theta = C_theta * d_v[None, :]

        # C.shape (N, 3N)
        C_scaled = jnp.concatenate([C_rho, C_theta, C_zeta], axis=1)

        # Apply L2⁻ᵀ per node using the existing L2inv
        # Ĉ = C D L⁻ᵀ
        Linv_T = jnp.swapaxes(Linv, 1, 2)  # (N, 3, 3)
        C_node = C_scaled[:, p].reshape(n_total, n_total, 3)
        Chat_node = jnp.einsum("mil, ilk -> mik", C_node, Linv_T)
        Chat = Chat_node.reshape(n_total, 3 * n_total)[:, pinv]

        Chat = Chat[keep_1][:, keep]
        row_norm = jnp.clip(
            jnp.linalg.norm(Chat, axis=1, keepdims=True), 1e-300, jnp.inf
        )
        Chat = Chat / row_norm

        # Orthogonal projector P = I - Ĉᵀ (L_G L_Gᵀ)⁻¹ Ĉ
        G = Chat @ _cT(Chat)
        G = (G + _cT(G)) / 2 + 1e-14 * jnp.eye(keep_1.size)  # Gram matrix w ridge

        #print(jnp.linalg.cond(G))
        # The will become one of the most expensive parts
        L_G = jnp.linalg.cholesky(G)

        Y = jax.lax.linalg.triangular_solve(L_G, Chat, left_side=True, lower=True)
        S = jax.lax.linalg.triangular_solve(_cT(L_G), Y, left_side=True, lower=False)
        CTS = _cT(Chat) @ S  # = C^T (L_G L_G^T)⁻¹ Ĉ

        ## applying the boundary condition first
        ## BCs before projection ≠ projection before BCs
        A_bc = A[jnp.ix_(keep, keep)]

        ## Projected operator A_proj = P A P without forming P
        A_proj = A_bc - A_bc @ CTS - CTS @ A_bc + CTS @ A_bc @ CTS
        A_proj = (A_proj + _cT(A_proj)) / 2

        ##A_proj = A_proj.at[jnp.diag_indices_from(A2_proj)].add(1e-12)

        #Au_proj = Au_bc - Au_bc @ CTS - CTS @ Au_bc + CTS @ Au_bc @ CTS
        #Au_proj = (Au_proj + _cT(Au_proj)) / 2

        #A_proj = Au_proj + A_proj

        # A_proj is already reduced to (keep, keep); do NOT re-index with `keep`
        # again (those indices run up to 3*n_total and JAX silently clamps the
        # out-of-range ones, corrupting the operator).
        A = A_proj
        #w, v = jnp.linalg.eigh(A)

        #print(w)
        ## Small for modes far from marginality
        #print(jnp.max(jnp.abs(Chat1 @ v[:, 0])))

        # --no-verify P = jnp.eye(CTS.shape[0], CTS.dtype) - CTS
        # --no-verify print("sym=", float(jnp.linalg.norm(P - P.T)),
        # --no-verify       "idem=", float(jnp.linalg.norm(P@P - P)),
        # --no-verify        "CP=", float(jnp.linalg.norm(Chat @ P)))

    else:
        ## Shift the diagonal of A to ensure positive definiteness
        ## The estimate must be accurate. If A is diagonally dominant
        ## use Gerhsgorin theorem to estimate the lowest eigenvalue
        A = A.at[jnp.diag_indices_from(A)].add(1e-13)

        A = A[jnp.ix_(keep, keep)] 
        #A = A[jnp.ix_(keep, keep)] + Au[jnp.ix_(keep, keep)]
        A = (A + _cT(A)) / 2


    v0 = kwargs.get("v_guess", None)
    if v0 is not None:
        v0 = np.asarray(v0).reshape(-1)
        if v0.size != A.shape[0]:
            print(
                f"finite-n lambda ignoring v_guess: got size={v0.size}, expected={A.shape[0]}"
            )
            v0 = None
    _agni_mem_trace(
        kwargs,
        "[finite-n lambda:dense] preparing scipy.eigsh",
        f"n_keep={A.shape[0]}",
        "converting A2 to NumPy",
    )

    default_sigma = -1e-3 if coupled_rt and zernike_penalty_alpha > 0.0 else -1e-0
    sigma = kwargs.get("sigma", default_sigma)
    eigsh_tol = kwargs.get(
        "eigsh_tol", 1e-5 if coupled_rt and zernike_penalty_alpha > 0.0 else 0.0
    )
    eigsh_maxiter = kwargs.get("eigsh_maxiter", None)
    full_spectrum = kwargs.get("full_spectrum", False)
    if full_spectrum:
        # Dense symmetric eigendecomposition of the full reduced matrix (every
        # eigenvalue, ascending) via scipy.linalg.eigh (LAPACK syevr, O(N)
        # workspace); jnp.linalg.eigh's syevd overflows int32 for N>~32768. The
        # dominant (most-negative) eigenpair is still handed downstream.
        from scipy.linalg import eigh as _dense_eigh

        w_all, v_all = _dense_eigh(np.asarray(A), overwrite_a=True)
        data["finite-n lambda spectrum"] = w_all
        w = w_all[:1]
        v = v_all[:, :1]
    elif v0 is None:
        w, v = eigsh(
            np.asarray(A),
            k=1,
            sigma=sigma,
            which="LM",
            tol=eigsh_tol,
            maxiter=eigsh_maxiter,
            return_eigenvectors=True,
        )
        data["finite-n lambda spectrum"] = np.asarray(w)
    else:
        w, v = eigsh(
            np.asarray(A),
            k=1,
            sigma=sigma,
            v0=v0,
            which="LM",
            tol=eigsh_tol,
            maxiter=eigsh_maxiter,
            return_eigenvectors=True,
        )
        data["finite-n lambda spectrum"] = np.asarray(w)

    idxs = jnp.where(jnp.abs(v) > 5e-5)[0]
    y = A @ v
    if os.environ.get("AGNI_DIAG","1")!="0": print(f"eigval res={jnp.linalg.norm(y[idxs]/v[idxs]-w)}")
    if os.environ.get("AGNI_DIAG","1")!="0": print(f"eigenvalue: {w}")
    if incompressible:
        print(jnp.max(jnp.abs(Chat @ v[:, 0])))


    # Reduced eigenvector -> full component-major vector [rho,theta,zeta].
    v_mode = v[:, 0] if jnp.ndim(v) == 2 else v
    v_full = jnp.zeros(3 * n_total, dtype=v_mode.dtype).at[keep].set(v_mode)

    def _reshape(u):
        return u.reshape(n_rho_max, n_theta_max, n_zeta_max)

    if coupled_rt:
        # In coupled mode D is the full (n_rho*n_theta) operator: flatten the
        # (rho, theta) axes, apply, reshape back. d_dz stays separable in zeta.
        def d_dr(D, u):
            """Radial derivative via the coupled (rho, theta) operator."""
            U = u.reshape(n_rho_max * n_theta_max, n_zeta_max)
            return (D @ U).reshape(n_rho_max, n_theta_max, n_zeta_max)

        def d_dv(D, u):
            """Poloidal derivative via the coupled (rho, theta) operator."""
            U = u.reshape(n_rho_max * n_theta_max, n_zeta_max)
            return (D @ U).reshape(n_rho_max, n_theta_max, n_zeta_max)

        def d_dz(D, u):
            return jnp.einsum("ij,klj->kli", D, u)
    else:
        def d_dr(D, u):
            """Calculate the radial derivative of u."""
            return jnp.einsum("ij,jkl->ikl", D, u)  # (Nr, Nθ, Nζ)

        def d_dv(D, u):
            return jnp.einsum("ij,kjl->kil", D, u)

        def d_dz(D, u):
            return jnp.einsum("ij,klj->kli", D, u)

    psi_r_over_sqrtg = _reshape(psi_r_over_sqrtg)
    psi_r = _reshape(psi_r)
    iota = _reshape(iota)
    g_rr = _reshape(g_rr)
    g_vv = _reshape(g_vv)
    g_pp = _reshape(g_pp)
    g_rv = _reshape(g_rv)
    g_rp = _reshape(g_rp)
    g_vp = _reshape(g_vp)

    # Reconstruct the physical displacement directly from the compact Linv blocks
    # (equivalent to d * (LinvT_full @ v_full); the equivalence is checked in tests/).
    vr, vv, vz = v_full[rho_idx], v_full[theta_idx], v_full[zeta_idx]
    xi_full = jnp.concatenate([
        d[rho_idx]   * (Linv[:, 0, 0] * vr + Linv[:, 1, 0] * vv + Linv[:, 2, 0] * vz),
        d[theta_idx] * (                     Linv[:, 1, 1] * vv + Linv[:, 2, 1] * vz),
        d[zeta_idx]  * (                                          Linv[:, 2, 2] * vz),
    ])

    # Phase rotation doesn't change the physics. Here, we use it to make the eigenmode up-down symmetric.
    # phase_offset (default 0) is an optional tunable rotation applied on top of the mean-based alignment.
    phase_offset = kwargs.get("phase_offset", 0.0)
    xi_ref = xi_full[rho_idx]
    phase_angle = jnp.arctan2(jnp.mean(xi_ref.real), jnp.mean(xi_ref.imag))
    per_elem_angles = jnp.arctan2(xi_ref.real, xi_ref.imag)
    angle_diff = per_elem_angles - phase_angle
    mags = jnp.abs(xi_ref)
    threshold = 0.01 * jnp.max(mags)
    mask = mags > threshold
    angle_diff_valid = jnp.where(mask, jnp.abs(angle_diff), jnp.nan)
    if os.environ.get("AGNI_DIAG","1")!="0": print(f"phase_angle (mean-based): {phase_angle:.4f} rad  |  per-elem deviation (all): max={jnp.max(jnp.abs(angle_diff)):.4f}, mean={jnp.mean(jnp.abs(angle_diff)):.4f}, std={jnp.std(angle_diff):.4f} rad")
    if os.environ.get("AGNI_DIAG","1")!="0": print(f"  deviation (|xi|>1% max, n={int(jnp.sum(mask))}/{xi_ref.size}): max={float(jnp.nanmax(angle_diff_valid)):.4f}, mean={float(jnp.nanmean(angle_diff_valid)):.4f} rad")
    xr = (xi_full[rho_idx].reshape(n_rho_max, n_theta_max, n_zeta_max)*jnp.exp(1j * (phase_angle + phase_offset))).imag
    xv = (xi_full[theta_idx].reshape(n_rho_max, n_theta_max, n_zeta_max)*jnp.exp(1j * (phase_angle + phase_offset))).imag
    xz = (xi_full[zeta_idx].reshape(n_rho_max, n_theta_max, n_zeta_max)*jnp.exp(1j * (phase_angle + phase_offset))).imag

    # precomputed forward derivatives (re-used below)
    xr_v = d_dv(D_theta0, xr)
    xr_z = d_dz(D_zeta0, xr)

    xv_v = d_dv(D_theta0, xv+xz)
    xv_z = d_dz(D_zeta0, xv+xz)

    xz_v = d_dv(D_theta0, xz/iota)
    xz_z = d_dz(D_zeta0, xz/iota)

    test_v = d_dv(D_theta0, xv)
    test_z = d_dz(D_zeta0, xv)

    # combos used many times
    xr_r = d_dr(D_rho0,  xr)  # dρ(ι ψ′² xr)
    psi_rr = d_dr(D_rho0,  psi_r)  # dρ(ι ψ′² xr)
    iota_r = d_dr(D_rho0,  iota)  # dρ(ι ψ′² xr)


    if os.environ.get("AGNI_DIAG","1")!="0": print(f"xr_v shape: {xr_v.shape}, xv_z shape: {xv_z.shape}, xz_z shape: {xz_z.shape}, xr_r shape: {xr_r.shape}, psi_r shape: {psi_r.shape}, psi_rr shape: {psi_rr.shape}, iota_r shape: {iota_r.shape}")

    deltaB_r = psi_r_over_sqrtg * psi_r * (iota * xr_v + xr_z)
    deltaB_v = psi_r_over_sqrtg * (1.* (test_z) - 1.*(xr_r * iota *psi_r + (2 * iota * psi_rr + iota_r * psi_r)* xr))
    deltaB_z = -psi_r_over_sqrtg * (1.* (test_v) + 1.*(xr_r * psi_r + 2 * psi_rr * xr))

    deltaV_r = psi_r * xr
    deltaV_v = xv + xz
    deltaV_z = xz * 1/iota

    if os.environ.get("AGNI_DIAG","1")!="0": print(f"deltaB_r shape: {deltaB_r.shape}, deltaB_v shape: {deltaB_v.shape}, deltaB_z shape: {deltaB_z.shape}")

    deltaB2 = g_rr * deltaB_r ** 2 + 1.*g_vv * deltaB_v ** 2  + g_pp * deltaB_z ** 2 + 2. * (g_rv * deltaB_r * deltaB_v + g_rp * deltaB_r * deltaB_z +  g_vp * deltaB_v * deltaB_z)
    deltaV2 = g_rr * deltaV_r ** 2 + 1.*g_vv * deltaV_v ** 2  + g_pp * deltaV_z ** 2 + 2. * (g_rv * deltaV_r * deltaV_v + g_rp * deltaV_r * deltaV_z +  g_vp * deltaV_v * deltaV_z)

    data["finite-n lambda"] = w
    data["finite-n eigenfunction"] = v_full
    data["finite-n xi"] = xi_full
    data["finite-n deltaB"] = np.sqrt(deltaB2)
    data["finite-n deltaV"] = np.sqrt(deltaV2)
    data["finite-n deltaB_r"] = deltaB_r
    data["finite-n deltaB_v"] = deltaB_v
    data["finite-n deltaB_z"] = deltaB_z
    data["finite-n deltaV_r"] = deltaV_r
    data["finite-n deltaV_v"] = deltaV_v
    data["finite-n deltaV_z"] = deltaV_z

    return data

@register_compute_fun(
    name="finite-n eigenfunction",
    label="\\xi",
    units="~",
    units_long="None",
    description="Finite-n eigenfunction",
    dim=5,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=["finite-n lambda"],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    stable_only="bool: for testing only, materialize "
    + "and eigendecompose the stable part of the matrix",
)
def _AGNI_eigenfunction(params, transforms, profiles, data, **kwargs):
    """Eigenfunctions of finite-n stability solver.

    Returns
    -------
    Finite-n lambda eigenfunctions
        Shape (num_eigenvalues, num rho, num theta, num zeta, 3).

    """
    return data  # noqa: unused dependency
