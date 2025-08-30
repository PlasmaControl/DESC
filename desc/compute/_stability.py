"""Compute functions for stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

import pdb
import time
from functools import partial

import numpy as np
from jax.scipy.linalg import solve_triangular
from scipy.constants import mu_0
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from desc.backend import eigh_tridiagonal, jax, jit, jnp, scan

from ..diffmat_utils import (
    D1_FD_4,
    fourier_diffmat,
    legendre_D1,
    legendre_lobatto_nodes,
    legendre_lobatto_weights,
)
from ..integrals.surface_integral import surface_integrals_map
from ..utils import dot
from .data_index import register_compute_fun


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
        data["V(r)"]
        * (2 * mu_0 * data["p_r"] + data["<|B|^2>_r"])
        / (data["V_r(r)"] * data["<|B|^2>"]),
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
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["c ballooning", "f ballooning", "g ballooning"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    Neigvals="int: number of largest eigenvalues to return, default value is 1.`"
    "If `Neigvals=2` eigenvalues are `[-1, 0, 1]` we get `[1, 0]`",
    diffmat="str: option to use a differentiation matricex based solver"
    "Default is None, other options are 'Legendre', 'Lele'",
)
@partial(jit, static_argnames=["Neigvals", "diffmat"])
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
    diffmat = kwargs.get("diffmat", None)
    grid = transforms["grid"].source_grid

    num_zeta0 = data["c ballooning"].shape[0]

    def reshape(f):
        assert f.shape == (num_zeta0, grid.num_nodes)
        f = jnp.swapaxes(grid.meshgrid_reshape(f.T, "raz"), -1, -2)
        assert f.shape == (grid.num_rho, grid.num_alpha, num_zeta0, grid.num_zeta)
        return f

    c = reshape(data["c ballooning"])
    f = reshape(data["f ballooning"])
    g = reshape(data["g ballooning"])

    if diffmat == "Legendre":

        def _eval_1D(f, x, scale, shift):
            return jax.vmap(lambda x_val: f(x_val, scale, shift))(x)

        def _f(x, scale, shift):
            x0 = 0.0
            x1 = 0.4
            m1 = 1.0
            m2 = 1.0
            m3 = 20
            m4 = 20

            wL = 0.5 * (1.0 + x0)  # left-side weigh
            wR = 0.5 * (1.0 - x0)  # right-side weight

            lower = wL * (
                1.0 - jnp.exp(-m1 * (x + 1.0)) + 0.5 * (x + 1.0) * jnp.exp(-2.0 * m1)
            )
            upper = wR * (
                jnp.exp(m2 * (x - 1.0)) + 0.5 * (x - 1.0) * jnp.exp(-2.0 * m2)
            )

            # Rescale to span [0,1] and shift to [-1,1]
            g_cluster = 2.0 * (lower + upper) - 1.0

            # Left logistic function
            s_axis = 1.0 / (1.0 + jnp.exp(-m3 * (x + 1.0)))
            s_axis0 = 1.0 / (1.0 + jnp.exp(-m3 * 0.0))
            s_axis1 = 1.0 / (1.0 + jnp.exp(-m3 * 2.0))
            axis = wL * (s_axis - s_axis0) / (s_axis1 - s_axis0)

            # Right logistic fn, also increasing after the flip
            s_edge_raw = 1.0 / (1.0 + jnp.exp(m4 * (x - 1.0)))
            s_edge = 1.0 - s_edge_raw
            s_edge0 = 1.0 - 1.0 / (1.0 + jnp.exp(m4 * -2.0))
            s_edge1 = 1.0 - 1.0 / (1.0 + jnp.exp(0.0))
            edge = wR * (s_edge - s_edge0) / (s_edge1 - s_edge0)

            g_axisedge = 2.0 * (axis + edge) - 1.0

            # Identity map contributes (1-x1) · x
            return (1.0 - x1) * x + x1 * (g_cluster + g_axisedge - x)

        dx_f = jax.grad(_f, argnums=0)

        num_alpha = grid.num_alpha
        num_rho = grid.num_rho
        num_zeta = grid.num_zeta

        ## The points in the supplied grid must be consistent with how
        ## the kronecker product is created
        x0 = grid.nodes[:: num_alpha * num_rho, 2]

        # The factor of two because we are mapping from (-1, 1) -> (-ntor pi, ntor pi)
        scale = (x0[-1] - x0[0]) / 2
        shift = 1 - x0[0] / scale

        x = legendre_lobatto_nodes(num_zeta - 1)

        scale_vector1 = (_eval_1D(dx_f, x, scale, shift)) ** -1 * 1 / scale

        scale_x1 = scale_vector1[:, None]

        # Get differentiation matrices
        # RG: setting the gradient to 0 to save some memory?
        D_zeta = jax.lax.stop_gradient(legendre_D1(num_zeta - 1) * scale_x1)

        # 2D matrices stacked in rho, alpha and zeta_0 dimensions
        w = (1 / scale_vector1) * (legendre_lobatto_weights(num_zeta - 1))
        wg = -1 * w * g

        # Row scaling D_rho by wg
        A = D_zeta.T @ (wg[..., :, None] * D_zeta)

        # the scale due to the derivative
        wc = w * c
        idx = jnp.arange(num_zeta)

        A = A.at[..., idx, idx].add(wc)

        b_inv = jnp.sqrt(jnp.reciprocal(w * f))

        A = (b_inv[..., :, None] * A) * b_inv[..., None, :]

        # apply dirichlet BC to X
        w, v = jnp.linalg.eigh(A[..., 1:-1, 1:-1])

    elif diffmat == "Lele":

        def _eval_1D(f, x, scale, shift):
            return jax.vmap(lambda x_val: f(x_val, scale, shift))(x)

        def _f(x, scale, shift):
            y = x / scale
            y = y - shift
            return y

        dx_f = jax.grad(_f, argnums=0)

        num_alpha = grid.num_alpha
        num_rho = grid.num_rho
        num_zeta = grid.num_zeta

        ## The points in the supplied grid must be consistent with how
        ## the kronecker product is created
        x0 = grid.nodes[:: num_alpha * num_rho, 2]

        # The factor of two because we are mapping from (-1, 1) -> (-ntor pi, ntor pi)
        scale = (x0[-1] - x0[0]) / 2
        shift = 1 + x0[0] / scale

        x = _eval_1D(_f, x0, scale, shift)
        # x = x0

        scale_vector1 = (_eval_1D(dx_f, x, scale, shift)) ** -1

        scale_x1 = scale_vector1[:, None]

        h = x[1] - x[0]

        # Get differentiation matrices
        # RG: setting the gradient to 0 to save some memory?
        A, B = create_lele_D1_6_matrix(num_zeta, h)

        D1_lele = jnp.linalg.solve(A, B)

        D_zeta = jax.lax.stop_gradient(D1_lele * scale_x1)

        idx = jnp.arange(num_zeta)
        ## Simpson's gives oscillatory eigenfunctions
        # w0   = 2.0 + 2.0 * (idx & 1)
        # w0   = w0.at[jnp.array([0, num_zeta-1])].set(1.0)
        # w0   = (h / 3.0) * w0
        # Uniform weights
        w0 = h

        w = w0

        wg = -1 * w * g

        # Row scaling D_rho by wg
        A = D_zeta.T @ (wg[..., :, None] * D_zeta)

        # the scale due to the derivative
        wc = w * c

        A = A.at[..., idx, idx].add(wc)

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
        Shape (num_rho, num alpha, num zeta0, num zeta - 2, num eigvals).

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


@register_compute_fun(
    name="finite-n lambda",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
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
        "finite-n instability drive",
        "iota",
        "iota_r",
        "psi_r",
        "p",
        "a",
    ],
    n_rho_max="int: 2 x maximum radial mode number",
    n_theta_max="int: 2 x maximum poloidal mode number",
    n_zeta_max="int: 2 x maximum toroidal mode number",
    axisym="bool: if the equilibrium is axisymmetric",
)
def _AGNI(params, transforms, profiles, data, **kwargs):
    """
    AGNI: Analysis of Global Normal-modes in Ideal MHD.

    Based on the original source here:
    https://github.com/rahulgaur104/AGNI/tree/master

    A finite-n stability eigenvalue solver.
    Currenly only finds fixed boundary unstable modes at
    low to medium resolution.

    This version of the code expands all the derivatives of the form
    partial_rho (iota psi' xi^rho) which means there are terms
    that only have a single D_rho operator. Also, some of the terms
    where we enforce symmetry may be wrong.
    """
    a_N = data["a"]
    B_N = params["Psi"] / (jnp.pi * a_N**2)

    iota = data["iota"][:, None]
    iota_r = data["iota_r"][:, None]

    psi_r = data["psi_r"][:, None] / (0.5 * a_N**2 * B_N)
    psi_rr = data["psi_rr"][:, None] / (0.5 * a_N**2 * B_N)

    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r = iota * psi_r
    iota_psi_r2 = iota * psi_r2

    dpsi_r2_drho = 2 * psi_r * psi_rr
    diota_psi_r2_drho = iota_r * psi_r**2 + iota * dpsi_r2_drho

    p0 = data["p"] / B_N**2
    p0 = p0.at[p0 < 0].set(1e-8)

    n0 = p0 ** (1 / 3) + 1e-4 / (psi_r2.flatten())
    # n0 = 1e-4

    axisym = kwargs.get("axisym", False)

    n_rho_max = kwargs.get("n_rho_max", 8)
    n_theta_max = kwargs.get("n_theta_max", 8)

    if axisym:
        # Each componenet of xi can be written as the Fourier sum of two modes in
        # the toroidal direction
        n_mode = 1
        # --no-verify D_zeta0 = 1j * n_mode * jnp.array([[0, -1], [1, 0]])
        D_zeta0 = 1j * n_mode * jnp.array([[1]])
        n_zeta_max = 1
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]
    else:
        n_zeta_max = kwargs.get("n_zeta_max", 4)
        D_zeta0 = fourier_diffmat(n_zeta_max)
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]

    def _eval_1D(f, x):
        return jax.vmap(lambda x_val: f(x_val))(x)

    def _f(x):
        x_0 = 0.75
        m_1 = 3.0
        m_2 = 2.0
        lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
        upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
        eps = 1.0e-3
        return eps + (1 - eps) * (lower + upper)

    # def _f(x):
    #    eps = 1e-5
    #    return eps + (1 - eps) * (x+1)/2

    dx_f = jax.grad(_f)

    x = legendre_lobatto_nodes(len(x0) - 1)

    scale_vector1 = (_eval_1D(dx_f, x)) ** -1
    # scale_vector1 = jnp.ones_like(x0) * (2 * jnp.pi-1e-3)

    # scale_vector1 = jnp.ones_like(x0) * (1 - 1e-3)
    # h = (1-1e-3)/(n_rho_max-1)

    scale_x1 = scale_vector1[:, None]

    # Get differentiation matrices
    # RG: setting the gradient to 0 to save some memory?
    D_rho0 = legendre_D1(n_rho_max - 1) * scale_x1
    # D_rho0 = fourier_diffmat(n_rho_max) * scale_x1
    # D_rho0, W0 = D1_FD_4(n_rho_max, h)
    # D_rho0 = D_rho0 * scale_x1

    D_theta0 = fourier_diffmat(n_theta_max)

    w0 = jnp.diag(1 / scale_vector1 * legendre_lobatto_weights(n_rho_max - 1))
    w0 = w0.at[jnp.abs(w0) < 1e-12].set(0.0)
    # w0 = jnp.diag((2*jnp.pi-1e-3)/n_rho_max * jnp.ones_like(x0))

    # w0 = jnp.diag(1 / scale_vector1 * W0.flatten())
    # w1 = jnp.diag(legendre_lobatto_weights(n_rho_max - 1))

    I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
    I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
    D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
    D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))

    # D_rho1 = jax.lax.stop_gradient(jnp.kron(D_rho1, jnp.kron(I_theta0, I_zeta0)))

    D_rhoT = jax.lax.stop_gradient(
        jnp.kron(D_rho0.conj().T, jnp.kron(I_theta0, I_zeta0))
    )
    D_thetaT = jax.lax.stop_gradient(
        jnp.kron(I_rho0, jnp.kron(D_theta0.conj().T, I_zeta0))
    )
    D_zetaT = jax.lax.stop_gradient(
        jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0.conj().T))
    )

    n_total = n_rho_max * n_theta_max * n_zeta_max
    ## Create the full matrix

    if axisym:
        A = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.complex128)
        B = jnp.zeros((3 * n_total, 3 * n_total))
    else:
        A = jnp.zeros((3 * n_total, 3 * n_total))
        B = jnp.zeros((3 * n_total, 3 * n_total))

    # Define field component indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)
    all_idx = slice(0, 3 * n_total)

    ### assuming uniform spacing in and θ and ζ
    dtheta = 2 * jnp.pi / n_theta_max
    dzeta = 2 * jnp.pi / n_zeta_max

    W = jnp.diag(jnp.kron(w0 * dtheta * dzeta, jnp.kron(I_theta0, I_zeta0)))[:, None]
    # W1 = jnp.diag(jnp.kron(w1 * dtheta * dzeta, jnp.kron(I_theta0, I_zeta0)))[:, None]

    # pdb.set_trace()

    def nan_to_zero(x):
        return jnp.where(jnp.isnan(x), 0.0, x)

    sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3

    psi_r_over_sqrtg = nan_to_zero(psi_r / sqrtg)

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
    j_sup_theta = mu_0 * data["J^theta_PEST"][:, None] * a_N**2 / B_N
    j_sup_zeta = mu_0 * data["J^zeta"][:, None] * a_N**2 / B_N

    # manually set the instability drive to 0
    F = 1 * mu_0 * data["finite-n instability drive"][:, None] * (a_N / B_N) ** 2

    def _stab_fix(mat, safety=1e-12):
        """
        Using Gershgorin circle idea to ensure the spectrum is positive
        definite.

        This is basically equivalent to shifting the whole
        eigenspectrum.
        """
        diag = jnp.real(jnp.diag(mat))
        row_sum = jnp.sum(jnp.abs(mat), axis=1) - jnp.abs(diag)
        # eigenvalue >= diag - row_sum  (Gershgorin)
        gap = -(diag - row_sum) + safety * diag.max()
        pad = jnp.clip(gap, 0.0)  # only add if gap>0
        return mat + jnp.diag(pad)

    def force_psd(M, floor=-1e-15):
        w, V = jnp.linalg.eigh(M, UPLO="U")
        w = jnp.clip(w, floor * w.max(), None)
        proj = (V * w) @ V.conj().T
        # Becomes identity during backward pass
        return M + jax.lax.stop_gradient(proj - M)

    ####################
    ####----Q_11----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * D_theta)
        + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * g_rr) * D_zeta)
        + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta)
        + ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta).conj().T @ D_theta
    )

    ####################
    ####----Q_22----####
    ####################
    # enforcing symmetry exactly
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).conj().T @ D_zeta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).conj().T @ D_zeta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).conj().T @ D_zeta
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        jnp.diag((diota_psi_r2_drho**2 / psi_r * psi_r_over_sqrtg * W * g_vv).flatten())
        + D_rho.T @ (((iota_psi_r) ** 2 * psi_r_over_sqrtg * psi_r * W * g_vv) * D_rho)
    )

    A = A.at[rho_idx, rho_idx].add(
        D_rho.T
        * (W * diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv).flatten()
        + (W * diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv) * D_rho
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            (diota_psi_r2_drho * psi_r_over_sqrtg * W * g_vv) * D_zeta
            + 0.5
            * (
                D_rho.conj().T @ ((iota_psi_r2 * psi_r_over_sqrtg * W * g_vv) * D_zeta)
                + D_zeta.conj().T
                @ ((iota_psi_r2 * psi_r_over_sqrtg * W * g_vv) * D_rho)
            )
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        (diota_psi_r2_drho * psi_r_over_sqrtg * W * g_vv) * D_zeta
        + D_rho.T @ ((iota_psi_r2 * psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    # A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
    # A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
    # A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)
    # w, _ = jnp.linalg.eigh(A)
    # print(w)

    # w = jnp.linalg.eigvalsh(A)
    # delta = (-w.min()) * 1.001         # 0.1% above the most negative λ
    # A = A + delta * jnp.eye(A.shape[0])

    # Extra test terms
    ###A = A.at[rho_idx, rho_idx].add(-1*(
    ###        D_rho.T @ ((W * diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv) * D_rho)))

    ###A = A.at[rho_idx, rho_idx].add(
    ###   D_rho.T * (W * diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv).flatten()
    ###   + (W * diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv) * D_rho
    ###)

    ###w1, _ = jnp.linalg.eigh(A)
    ###print(w1[:100])

    # pdb.set_trace()

    ####################
    ####----Q_33----####
    ####################
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1 * D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, rho_idx].add(
        jnp.diag((dpsi_r2_drho**2 / psi_r * psi_r_over_sqrtg * W * g_pp).flatten())
        + 0.5
        * (
            D_rho.T
            @ (
                ((psi_r2) ** 2 * psi_r_over_sqrtg * psi_r * W * g_pp) * D_rho
            )  # enforcing symmetry exactly
            + (((psi_r2) ** 2 * psi_r_over_sqrtg * psi_r * W * g_pp) * D_rho).T @ D_rho
        )
        + D_rho.T * (dpsi_r2_drho * psi_r_over_sqrtg * iota_psi_r * W * g_pp).flatten()
        + (dpsi_r2_drho * psi_r_over_sqrtg * iota_psi_r * W * g_pp) * D_rho
    )
    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            (dpsi_r2_drho * psi_r_over_sqrtg * W * g_pp) * D_theta
            + D_rho.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_pp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        (dpsi_r2_drho * psi_r_over_sqrtg * W * g_pp) * D_theta
        + D_rho.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    # A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
    # A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
    # A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)
    # w, _ = jnp.linalg.eigh(A)
    # print(w[:100])
    # pdb.set_trace()

    # from matplotlib import pyplot as plt
    # plt.spy(A)
    # plt.show()

    B = A.copy()
    B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)
    B = B.at[zeta_idx, rho_idx].set(B[rho_idx, zeta_idx].T)
    B = B.at[zeta_idx, theta_idx].set(B[theta_idx, zeta_idx].T)
    w, _ = jnp.linalg.eigh(B)

    print(w)
    from matplotlib import pyplot as plt

    plt.yscale("symlog", linthresh=1e-20)
    plt.plot(w)
    pdb.set_trace()

    # A = A.at[all_idx, all_idx].set(force_psd(A))

    ####################
    ####----Q_12----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T
            * (iota * psi_r * psi_r_over_sqrtg * diota_psi_r2_drho * W * g_rv).flatten()
            + D_zeta.conj().T
            * (psi_r * psi_r_over_sqrtg * diota_psi_r2_drho * W * g_rv).flatten()
            + D_theta.T @ ((iota**2 * psi_r3 * psi_r_over_sqrtg * W * g_rv) * D_rho)
            + D_zeta.conj().T @ ((iota * psi_r3 * psi_r_over_sqrtg * W * g_rv) * D_rho)
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (iota * psi_r * psi_r_over_sqrtg * diota_psi_r2_drho * W * g_rv) * D_theta
            + (psi_r * psi_r_over_sqrtg * diota_psi_r2_drho * W * g_rv) * D_zeta
            + D_rho.T @ ((iota**2 * psi_r3 * psi_r_over_sqrtg * W * g_rv) * D_theta)
            + D_rho.T @ ((iota * psi_r3 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        + D_zeta.conj().T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
            + D_zeta.conj().T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    ######################
    ####-----Q_13-----####
    ######################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T
            * (iota * psi_r * psi_r_over_sqrtg * dpsi_r2_drho * W * g_rp).flatten()
            + D_zeta.conj().T
            * (psi_r * psi_r_over_sqrtg * dpsi_r2_drho * W * g_rp).flatten()
            + D_theta.T @ ((iota * psi_r3 * psi_r_over_sqrtg * W * g_rp) * D_rho)
            + D_zeta.conj().T @ ((psi_r3 * psi_r_over_sqrtg * W * g_rp) * D_rho)
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (iota * psi_r * psi_r_over_sqrtg * dpsi_r2_drho * W * g_rp) * D_theta
            + (psi_r * psi_r_over_sqrtg * dpsi_r2_drho * W * g_rp) * D_zeta
            + D_rho.T @ ((iota * psi_r3 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_rho.T @ ((psi_r3 * psi_r_over_sqrtg * W * g_rp) * D_zeta)
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        + D_zeta.conj().T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_zeta.conj().T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )

    # C = A.copy()
    # C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    # C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    # C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)
    # w, _ = jnp.linalg.eigh(C)
    # print(w)

    # A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
    # A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
    # A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)
    # w, _ = jnp.linalg.eigh(A)
    # print(w)

    ##########################
    #######-----Q_23-----#####
    ##########################
    A = A.at[theta_idx, theta_idx].add(
        -1 * (D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta))
        - 1 * (((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta)
    )

    A = A.at[zeta_idx, zeta_idx].add(
        -1 * (D_zeta.conj().T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta))
        - 1 * (((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta)
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            D_rho.T @ ((psi_r_over_sqrtg * W * psi_r2 * g_vp) * D_zeta)
            + (psi_r_over_sqrtg * W * dpsi_r2_drho * g_vp) * D_zeta
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            D_rho.T @ ((psi_r_over_sqrtg * W * psi_r2 * g_vp) * D_zeta)
            + (psi_r_over_sqrtg * W * dpsi_r2_drho * g_vp) * D_zeta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        D_zeta.conj().T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
    )

    A = A.at[rho_idx, theta_idx].add(
        1
        * (
            D_rho.T @ ((psi_r_over_sqrtg * W * iota * psi_r2 * g_vp) * D_theta)
            + (psi_r_over_sqrtg * W * diota_psi_r2_drho * g_vp) * D_theta
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_rho.T @ ((psi_r_over_sqrtg * W * iota * psi_r2 * g_vp) * D_theta)
            + (psi_r_over_sqrtg * W * diota_psi_r2_drho * g_vp) * D_theta
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        D_rho.T @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_vp) * D_rho)
        + jnp.diag(
            psi_r_over_sqrtg * diota_psi_r2_drho * (dpsi_r2_drho / psi_r) * W * g_vp
        )
        + D_rho.T
        * (
            psi_r_over_sqrtg * iota * psi_r2 * (dpsi_r2_drho / psi_r) * W * g_vp
        ).flatten()
        + (psi_r_over_sqrtg * diota_psi_r2_drho * psi_r * W * g_vp) * D_rho
    )

    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        ((psi_r_over_sqrtg * iota * psi_r3 * W * g_vp) * D_rho).T @ D_rho
        + jnp.diag(
            psi_r_over_sqrtg * diota_psi_r2_drho * (dpsi_r2_drho / psi_r) * W * g_vp
        )
        + (psi_r_over_sqrtg * iota * psi_r2 * (dpsi_r2_drho / psi_r) * W * g_vp) * D_rho
        + ((psi_r_over_sqrtg * diota_psi_r2_drho * psi_r * W * g_vp) * D_rho).T
    )

    ## diagonal |J|^2 term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((psi_r2 * W * sqrtg * J2).flatten()))

    # C = A.copy()
    # C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    # C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    # C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)
    # w, _ = jnp.linalg.eigh(C)
    # print(w)
    # from matplotlib import pyplot as plt

    # plt.yscale("symlog", linthresh=1e-20);
    # plt.plot(w);
    # pdb.set_trace()

    # Mixed Q-J term
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (
                W
                * psi_r3
                * sqrtg
                * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                / g_sup_rr
            )
            * (D_theta + iota * D_zeta)
            + jnp.diag(
                (
                    -W * psi_r * sqrtg * j_sup_theta * diota_psi_r2_drho
                    + W * psi_r * sqrtg * j_sup_zeta * dpsi_r2_drho
                ).flatten()
            )
            - (W * iota * sqrtg * psi_r3) * D_rho
            + (W * sqrtg * psi_r3) * D_rho
        )
    )

    # ρ-ρ block transposed for symmetry
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (
                (
                    W
                    * psi_r3
                    * sqrtg
                    * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                    / g_sup_rr
                )
                * (D_theta + iota * D_zeta)
            )
            .conj()
            .T
            + jnp.diag(
                (
                    -W * psi_r * sqrtg * j_sup_theta * diota_psi_r2_drho
                    + W * psi_r * sqrtg * j_sup_zeta * dpsi_r2_drho
                ).flatten()
            )
            - ((W * iota * sqrtg * psi_r3) * D_rho).T
            + ((W * sqrtg * psi_r3) * D_rho).T
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        (W * psi_r2 * sqrtg * j_sup_theta) * D_zeta
        - (W * psi_r2 * sqrtg * j_sup_zeta) * D_theta
    )
    A = A.at[rho_idx, zeta_idx].add(
        -(W * psi_r2 * sqrtg * j_sup_theta) * D_zeta
        + (W * psi_r2 * sqrtg * j_sup_zeta) * D_theta
    )

    C = A.copy()
    C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - (n_theta_max * n_zeta_max))
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    w, v = jnp.linalg.eigh(C[jnp.ix_(keep, keep)])
    # w, v = jnp.linalg.eigh(C[n_theta_max*n_zeta_max:, n_theta_max*n_zeta_max:])
    print(w)

    # from jax.experimental.sparse import linalg as linalg
    # linalg.lobpcg_standard(C, v[:, :100].T, m=100)

    from matplotlib import pyplot as plt

    plt.yscale("symlog", linthresh=1e-20)
    plt.plot(w)
    plt.show()

    pdb.set_trace()

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(jnp.diag(n0 * (W * psi_r2 * sqrtg * g_rr).flatten()))
    B = B.at[theta_idx, theta_idx].add(jnp.diag(n0 * (W * sqrtg * g_vv).flatten()))
    B = B.at[zeta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota**2 * sqrtg * g_pp).flatten())
    )
    B = B.at[rho_idx, theta_idx].add(
        jnp.diag(n0 * (W * psi_r * sqrtg * g_rv).flatten())
    )
    B = B.at[rho_idx, zeta_idx].add(
        jnp.diag(n0 * (W * psi_r * iota * sqrtg * g_rp).flatten())
    )
    B = B.at[theta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota * sqrtg * g_vp).flatten())
    )

    # symmetrizing the matrix
    if axisym:
        A = A.at[theta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, theta_idx]).T)
        A = A.at[zeta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, zeta_idx]).T)
        A = A.at[zeta_idx, theta_idx].set(jnp.conjugate(A[theta_idx, zeta_idx]).T)

        B = B.at[theta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, theta_idx]).T)
        B = B.at[zeta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, zeta_idx]).T)
        B = B.at[zeta_idx, theta_idx].set(jnp.conjugate(B[theta_idx, zeta_idx]).T)
    else:
        A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
        A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
        A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)

        B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)
        B = B.at[zeta_idx, rho_idx].set(B[rho_idx, zeta_idx].T)
        B = B.at[zeta_idx, theta_idx].set(B[theta_idx, zeta_idx].T)

    # Force all the
    A = A.at[all_idx, all_idx].set(force_psd(A))

    # Finally add the only instability drive term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((W * psi_r2 * sqrtg * F).flatten()))

    y = A - A.conj().T
    print(jnp.max(jnp.abs(y)))

    y = B - B.conj().T
    print(jnp.max(jnp.abs(y)))

    # pdb.set_trace()

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    # --no-verify w, _ = jnp.linalg.eigh(B[jnp.ix_(keep, keep)])
    # --no-verify w, v = jnp.linalg.eigh(A[jnp.ix_(keep, keep)])
    # --no-verify w, v = jax.scipy.linalg.eigh(A[jnp.ix_(keep, keep)], B[jnp.ix_(keep, keep)])
    # --no-verify w, v =  eigh(jax.numpy.asarray(A[jnp.ix_(keep, keep)]), jax.numpy.asarray(B[jnp.ix_(keep, keep)]), subset_by_index=[0, 2])
    w, v = eigh(
        jax.numpy.asarray(A[jnp.ix_(keep, keep)]),
        jax.numpy.asarray(B[jnp.ix_(keep, keep)]),
    )

    ## This will be the most expensive but easiest automatically differentiable way.
    ## TODO: Multiply B with a permutation matrix P so that it becomes block diagonal
    ## That will make cholesky ~ 20 x faster
    # L = jnp.linalg.cholesky(B[jnp.ix_(keep, keep)])
    # w, v = jnp.linalg.eigh(L.T @ A[jnp.ix_(keep, keep)] @ L)

    data["finite-n lambda"] = w
    data["finite-n eigenfunction"] = v

    return data


@register_compute_fun(
    name="finite-n lambda2",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
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
        "finite-n instability drive",
        "iota",
        "iota_r",
        "psi_r",
        "p",
        "a",
    ],
    n_rho_max="int: 2 x maximum radial mode number",
    n_theta_max="int: 2 x maximum poloidal mode number",
    n_zeta_max="int: 2 x maximum toroidal mode number",
    axisym="bool: if the equilibrium is axisymmetric",
)
def _AGNI2(params, transforms, profiles, data, **kwargs):
    """
    AGNI: Analysis of Global Normal-modes in Ideal MHD.

    Based on the original source here:
    https://github.com/rahulgaur104/AGNI/tree/master

    A finite-n stability eigenvalue solver.
    Currenly only finds fixed boundary unstable modes at
    low to medium resolution.

    This version of the code expands all the derivatives of the form
    partial_rho (iota psi' xi^rho) which means there are terms
    that only have a single D_rho operator. Also, some of the terms
    where we enforce symmetry may be wrong.
    While mostly similar to v1, we have added functions to check
    and forcibly ensure positive-definiteness.
    """
    a_N = data["a"]
    B_N = params["Psi"] / (jnp.pi * a_N**2)

    iota = data["iota"][:, None]
    iota_r = data["iota_r"][:, None]

    psi_r = data["psi_r"][:, None] / (0.5 * a_N**2 * B_N)
    psi_rr = data["psi_rr"][:, None] / (0.5 * a_N**2 * B_N)

    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r = iota * psi_r
    iota_psi_r2 = iota * psi_r2

    dpsi_r2_drho = 2 * psi_r * psi_rr
    diota_psi_r2_drho = iota_r * psi_r**2 + iota * dpsi_r2_drho

    p = data["p"] / B_N**2

    n0 = p ** (1 / 3) + 1 / (psi_r2.flatten())
    # --no-verify n0 = p ** (1 / 3) + 10

    axisym = kwargs.get("axisym", False)

    n_rho_max = kwargs.get("n_rho_max", 8)
    n_theta_max = kwargs.get("n_theta_max", 8)

    if axisym:
        # Each componenet of xi can be written as the Fourier sum of two modes in
        # the toroidal direction
        n_mode = 1
        # --no-verify D_zeta0 = 1j * n_mode * jnp.array([[0, -1], [1, 0]])
        D_zeta0 = 1j * n_mode * jnp.array([[1]])
        n_zeta_max = 1
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]
    else:
        n_zeta_max = kwargs.get("n_zeta_max", 4)
        D_zeta0 = fourier_diffmat(n_zeta_max)
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]

    def _eval_1D(f, x):
        return jax.vmap(lambda x_val: f(x_val))(x)

    def _f(x):
        x_0 = 0.75
        m_1 = 3.0
        m_2 = 2.0
        lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
        upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
        eps = 1.0e-3
        return eps + (1 - eps) * (lower + upper)

    # def _f(x):
    #    eps = 1e-5
    #    return eps + (1 - eps) * (x+1)/2

    # \int d\rho_s (\partialX/\partial\rho_{s}) = int d\rho f'(\rho) (\partial\rho/\partial\rho_{s}) (\partial X/\partial\rho)
    # ∫dρₛ (∂X/∂ρₛ) = ∫dρ f'(ρ) (∂ρ/∂ρₛ) (∂X/∂ρ)

    dx_f = jax.grad(_f)

    x = legendre_lobatto_nodes(len(x0) - 1)

    scale_vector1 = (_eval_1D(dx_f, x)) ** -1
    # scale_vector1 = jnp.ones_like(x0) * (2 * jnp.pi-1e-3)

    # scale_vector1 = jnp.ones_like(x0) * (1 - 1e-3)
    # h = (1-1e-3)/(n_rho_max-1)

    scale_x1 = scale_vector1[:, None]

    # Get differentiation matrices
    # RG: setting the gradient to 0 to save some memory?
    D_rho0 = legendre_D1(n_rho_max - 1) * scale_x1
    # D_rho0 = fourier_diffmat(n_rho_max) * scale_x1
    # D_rho0, W0 = D1_FD_4(n_rho_max, h)
    # D_rho0 = D_rho0 * scale_x1

    D_theta0 = fourier_diffmat(n_theta_max)

    w0 = jnp.diag(1 / scale_vector1 * legendre_lobatto_weights(n_rho_max - 1))
    w0 = w0.at[jnp.abs(w0) < 1e-12].set(0.0)
    # w0 = jnp.diag((2*jnp.pi-1e-3)/n_rho_max * jnp.ones_like(x0))

    # w0 = jnp.diag(1 / scale_vector1 * W0.flatten())
    # w1 = jnp.diag(legendre_lobatto_weights(n_rho_max - 1))

    I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
    I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
    D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
    D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))

    # D_rho1 = jax.lax.stop_gradient(jnp.kron(D_rho1, jnp.kron(I_theta0, I_zeta0)))

    D_rhoT = jax.lax.stop_gradient(
        jnp.kron(D_rho0.conj().T, jnp.kron(I_theta0, I_zeta0))
    )
    D_thetaT = jax.lax.stop_gradient(
        jnp.kron(I_rho0, jnp.kron(D_theta0.conj().T, I_zeta0))
    )
    D_zetaT = jax.lax.stop_gradient(
        jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0.conj().T))
    )

    n_total = n_rho_max * n_theta_max * n_zeta_max
    ## Create the full matrix

    if axisym:
        A = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.complex128)
        B = jnp.zeros((3 * n_total, 3 * n_total))
    else:
        A = jnp.zeros((3 * n_total, 3 * n_total))
        B = jnp.zeros((3 * n_total, 3 * n_total))

    # Define field component indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)
    all_idx = slice(0, 3 * n_total)

    ### assuming uniform spacing in and θ and ζ
    dtheta = 2 * jnp.pi / n_theta_max
    dzeta = 2 * jnp.pi / n_zeta_max

    W = jnp.diag(jnp.kron(w0 * dtheta * dzeta, jnp.kron(I_theta0, I_zeta0)))[:, None]
    # W1 = jnp.diag(jnp.kron(w1 * dtheta * dzeta, jnp.kron(I_theta0, I_zeta0)))[:, None]

    # pdb.set_trace()

    def nan_to_zero(x):
        return jnp.where(jnp.isnan(x), 0.0, x)

    sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3

    psi_r_over_sqrtg = nan_to_zero(psi_r / sqrtg)

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
    j_sup_theta = mu_0 * data["J^theta_PEST"][:, None] * a_N**2 / B_N
    j_sup_zeta = mu_0 * data["J^zeta"][:, None] * a_N**2 / B_N

    # manually set the instability drive to 0
    F = 1 * mu_0 * data["finite-n instability drive"][:, None] * (a_N / B_N) ** 2

    def _stab_fix(mat, safety=1e-17):
        """
        Using Gershgorin circle idea to ensure the spectrum is positive
        definite.
        """
        diag = jnp.real(jnp.diag(mat))
        row_sum = jnp.sum(jnp.abs(mat), axis=1) - jnp.abs(diag)
        # eigenvalue >= diag - row_sum  (Gershgorin)
        gap = -(diag - row_sum) + safety * diag.max()
        pad = jnp.clip(gap, 0.0)  # only add if gap>0
        return mat + jnp.diag(pad)

    def force_psd(M, floor=-1e-20):
        w, V = jnp.linalg.eigh(M, UPLO="U")
        w = jnp.clip(w, floor * w.max(), None)
        return (V * w) @ V.conj().T

    ####################
    ####----Q_11----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * D_theta)
        + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * g_rr) * D_zeta)
        + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta)
        + ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta).conj().T @ D_theta
    )

    ####################
    ####----Q_22----####
    ####################
    # enforcing symmetry exactly
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).conj().T @ D_zeta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).conj().T @ D_zeta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).conj().T @ D_zeta
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        jnp.diag((diota_psi_r2_drho**2 / psi_r * psi_r_over_sqrtg * W * g_vv).flatten())
        + D_rho.T @ (((iota_psi_r) ** 2 * psi_r_over_sqrtg * psi_r * W * g_vv) * D_rho)
    )

    A = A.at[rho_idx, rho_idx].add(
        -1
        * jnp.diag(
            (
                W * (D_rho @ (diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv))
            ).flatten()
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            (diota_psi_r2_drho * psi_r_over_sqrtg * W * g_vv) * D_zeta
            + 0.5
            * (
                D_rho.conj().T @ ((iota_psi_r2 * psi_r_over_sqrtg * W * g_vv) * D_zeta)
                + D_zeta.conj().T
                @ ((iota_psi_r2 * psi_r_over_sqrtg * W * g_vv) * D_rho)
            )
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        (diota_psi_r2_drho * psi_r_over_sqrtg * W * g_vv) * D_zeta
        + 0.5
        * (
            D_rho.conj().T @ ((iota_psi_r2 * psi_r_over_sqrtg * W * g_vv) * D_zeta)
            + D_zeta.conj().T @ ((iota_psi_r2 * psi_r_over_sqrtg * W * g_vv) * D_rho)
        )
    )

    # A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
    # A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
    # A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)
    # w, _ = jnp.linalg.eigh(A)
    # print(w)

    # w = jnp.linalg.eigvalsh(A)
    # delta = (-w.min()) * 1.001         # 0.1% above the most negative λ
    # A = A + delta * jnp.eye(A.shape[0])

    # Extra test terms
    ###A = A.at[rho_idx, rho_idx].add(-1*(
    ###        D_rho.T @ ((W * diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv) * D_rho)))

    ###A = A.at[rho_idx, rho_idx].add(
    ###   D_rho.T * (W * diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv).flatten()
    ###   + (W * diota_psi_r2_drho * psi_r_over_sqrtg * iota_psi_r * g_vv) * D_rho
    ###)

    ###w1, _ = jnp.linalg.eigh(A)
    ###print(w1[:100])

    # pdb.set_trace()

    ####################
    ####----Q_33----####
    ####################
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1 * D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, rho_idx].add(
        jnp.diag((dpsi_r2_drho**2 / psi_r * psi_r_over_sqrtg * W * g_pp).flatten())
        + 0.5
        * (
            D_rho.T
            @ (
                ((psi_r2) ** 2 * psi_r_over_sqrtg * psi_r * W * g_pp) * D_rho
            )  # enforcing symmetry exactly
            + (((psi_r2) ** 2 * psi_r_over_sqrtg * psi_r * W * g_pp) * D_rho).T @ D_rho
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        -1
        * jnp.diag(
            (W * (D_rho @ (dpsi_r2_drho * psi_r_over_sqrtg * psi_r * g_pp))).flatten()
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            (dpsi_r2_drho * psi_r_over_sqrtg * W * g_pp) * D_theta
            + D_rho.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_pp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        (dpsi_r2_drho * psi_r_over_sqrtg * W * g_pp) * D_theta
        + D_rho.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    ## A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
    ## A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
    ## A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)
    ## w, _ = jnp.linalg.eigh(A)
    ## print(w[:100])
    ## pdb.set_trace()

    ## from matplotlib import pyplot as plt
    ## plt.spy(A)
    ## plt.show()

    # C = A.copy()
    # C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    # C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    # C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)

    # keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - (n_theta_max * n_zeta_max))
    # keep_2 = jnp.arange(n_total, 3 * n_total)
    # keep = jnp.concatenate([keep_1, keep_2])

    # w, _ = jnp.linalg.eigh(C[jnp.ix_(keep, keep)])

    # print(w)
    # from matplotlib import pyplot as plt

    # plt.yscale("symlog", linthresh=1e-20);
    # plt.plot(w);
    # pdb.set_trace()

    ####################
    ####----Q_12----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            -1
            * jnp.diag(
                (
                    0.5
                    * W
                    * (
                        D_theta
                        @ (iota * psi_r * psi_r_over_sqrtg * diota_psi_r2_drho * g_rv)
                    )
                ).flatten()
            )
            - jnp.diag(
                (
                    0.5
                    * W
                    * (D_zeta @ (psi_r * psi_r_over_sqrtg * diota_psi_r2_drho * g_rv))
                ).flatten()
            )
            + D_theta.T @ ((iota**2 * psi_r3 * psi_r_over_sqrtg * W * g_rv) * D_rho)
            + D_zeta.conj().T @ ((iota * psi_r3 * psi_r_over_sqrtg * W * g_rv) * D_rho)
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            -1
            * jnp.diag(
                (
                    0.5
                    * W
                    * (
                        D_theta
                        @ (iota * psi_r * psi_r_over_sqrtg * diota_psi_r2_drho * g_rv)
                    )
                ).flatten()
            )
            - jnp.diag(
                (
                    0.5
                    * W
                    * (D_zeta @ (psi_r * psi_r_over_sqrtg * diota_psi_r2_drho * g_rv))
                ).flatten()
            )
            + D_rho.T @ ((iota**2 * psi_r3 * psi_r_over_sqrtg * W * g_rv) * D_theta)
            + D_rho.T @ ((iota * psi_r3 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        + D_zeta.conj().T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
            + D_zeta.conj().T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    ######################
    ####-----Q_13-----####
    ######################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            -1
            * jnp.diag(
                (
                    0.5
                    * W
                    * (
                        D_theta
                        @ (iota * psi_r * psi_r_over_sqrtg * dpsi_r2_drho * g_rp)
                    )
                ).flatten()
            )
            - 1
            * jnp.diag(
                (
                    0.5
                    * W
                    * (D_zeta @ (psi_r * psi_r_over_sqrtg * dpsi_r2_drho * g_rp))
                ).flatten()
            )
            + D_theta.T @ ((iota * psi_r3 * psi_r_over_sqrtg * W * g_rp) * D_rho)
            + D_zeta.conj().T @ ((psi_r3 * psi_r_over_sqrtg * W * g_rp) * D_rho)
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            -1
            * jnp.diag(
                (
                    0.5
                    * W
                    * (
                        D_theta
                        @ (iota * psi_r * psi_r_over_sqrtg * dpsi_r2_drho * g_rp)
                    )
                ).flatten()
            )
            - 1
            * jnp.diag(
                (
                    0.5
                    * W
                    * (D_zeta @ (psi_r * psi_r_over_sqrtg * dpsi_r2_drho * g_rp))
                ).flatten()
            )
            + D_rho.T @ ((iota * psi_r3 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_rho.T @ ((psi_r3 * psi_r_over_sqrtg * W * g_rp) * D_zeta)
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        + D_zeta.conj().T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_zeta.conj().T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )

    # C = A.copy()
    # C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    # C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    # C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)
    # w, _ = jnp.linalg.eigh(C)
    # print(w)

    # A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
    # A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
    # A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)
    # w, _ = jnp.linalg.eigh(A)
    # print(w)

    ##########################
    #######-----Q_23-----#####
    ##########################
    A = A.at[theta_idx, theta_idx].add(
        -1 * (D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta))
        - 1 * (((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta)
    )

    A = A.at[zeta_idx, zeta_idx].add(
        -1 * (D_zeta.conj().T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta))
        - 1 * (((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta)
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            D_rho.T @ ((psi_r_over_sqrtg * W * psi_r2 * g_vp) * D_zeta)
            + jnp.diag(
                (
                    -0.5 * W * (D_zeta @ (psi_r_over_sqrtg * dpsi_r2_drho * g_vp))
                ).flatten()
            )
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            D_rho.T @ ((psi_r_over_sqrtg * W * psi_r2 * g_vp) * D_zeta)
            + jnp.diag(
                (
                    -0.5 * W * (D_zeta @ (psi_r_over_sqrtg * dpsi_r2_drho * g_vp))
                ).flatten()
            )
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        D_zeta.conj().T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
    )

    A = A.at[rho_idx, theta_idx].add(
        1
        * (
            D_rho.T @ ((psi_r_over_sqrtg * W * iota * psi_r2 * g_vp) * D_theta)
            + jnp.diag(
                (
                    -0.5 * W * (D_theta @ (psi_r_over_sqrtg * diota_psi_r2_drho * g_vp))
                ).flatten()
            )
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_rho.T @ ((psi_r_over_sqrtg * W * iota * psi_r2 * g_vp) * D_theta)
            + jnp.diag(
                (
                    -0.5 * W * (D_theta @ (psi_r_over_sqrtg * diota_psi_r2_drho * g_vp))
                ).flatten()
            )
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        D_rho.T @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_vp) * D_rho)
        + jnp.diag(
            psi_r_over_sqrtg * diota_psi_r2_drho * (dpsi_r2_drho / psi_r) * W * g_vp
        )
        + D_rho.T
        * (
            psi_r_over_sqrtg * iota * psi_r2 * (dpsi_r2_drho / psi_r) * W * g_vp
        ).flatten()
        + (psi_r_over_sqrtg * diota_psi_r2_drho * psi_r * W * g_vp) * D_rho
        + jnp.diag(
            -W
            * (
                D_rho
                @ (psi_r_over_sqrtg * iota * psi_r2 * (dpsi_r2_drho / psi_r) * g_vp)
            ).flatten()
        )
    )

    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        ((psi_r_over_sqrtg * iota * psi_r3 * W * g_vp) * D_rho).T @ D_rho
        + jnp.diag(
            psi_r_over_sqrtg * diota_psi_r2_drho * (dpsi_r2_drho / psi_r) * W * g_vp
        )
        + (psi_r_over_sqrtg * iota * psi_r2 * (dpsi_r2_drho / psi_r) * W * g_vp) * D_rho
        + ((psi_r_over_sqrtg * diota_psi_r2_drho * psi_r * W * g_vp) * D_rho).T
    )

    ## diagonal |J|^2 term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((psi_r2 * W * sqrtg * J2).flatten()))

    C = A.copy()
    C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)
    w, _ = jnp.linalg.eigh(C)
    print(w)
    from matplotlib import pyplot as plt

    plt.yscale("symlog", linthresh=1e-20)
    plt.plot(w)
    pdb.set_trace()

    # Mixed Q-J term
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (
                W
                * psi_r3
                * sqrtg
                * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                / g_sup_rr
            )
            * (D_theta + iota * D_zeta)
            + jnp.diag(
                (
                    -W * psi_r * sqrtg * j_sup_theta * diota_psi_r2_drho
                    + W * psi_r * sqrtg * j_sup_zeta * dpsi_r2_drho
                ).flatten()
            )
            - (W * iota * sqrtg * psi_r3) * D_rho
            + (W * sqrtg * psi_r3) * D_rho
        )
    )

    # ρ-ρ block transposed for symmetry
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (
                (
                    W
                    * psi_r3
                    * sqrtg
                    * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                    / g_sup_rr
                )
                * (D_theta + iota * D_zeta)
            )
            .conj()
            .T
            + jnp.diag(
                (
                    -W * psi_r * sqrtg * j_sup_theta * diota_psi_r2_drho
                    + W * psi_r * sqrtg * j_sup_zeta * dpsi_r2_drho
                ).flatten()
            )
            - ((W * iota * sqrtg * psi_r3) * D_rho).T
            + ((W * sqrtg * psi_r3) * D_rho).T
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        (W * psi_r2 * sqrtg * j_sup_theta) * D_zeta
        - (W * psi_r2 * sqrtg * j_sup_zeta) * D_theta
    )
    A = A.at[rho_idx, zeta_idx].add(
        -(W * psi_r2 * sqrtg * j_sup_theta) * D_zeta
        + (W * psi_r2 * sqrtg * j_sup_zeta) * D_theta
    )

    C = A.copy()
    C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - (n_theta_max * n_zeta_max))
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    w, v = jnp.linalg.eigh(C[jnp.ix_(keep, keep)])
    # w, v = jnp.linalg.eigh(C[n_theta_max*n_zeta_max:, n_theta_max*n_zeta_max:])
    print(w)

    # from jax.experimental.sparse import linalg as linalg
    # linalg.lobpcg_standard(C, v[:, :100].T, m=100)

    from matplotlib import pyplot as plt

    plt.yscale("symlog", linthresh=1e-20)
    plt.plot(w)
    plt.show()

    pdb.set_trace()

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(jnp.diag(n0 * (W * psi_r2 * sqrtg * g_rr).flatten()))
    B = B.at[theta_idx, theta_idx].add(jnp.diag(n0 * (W * sqrtg * g_vv).flatten()))
    B = B.at[zeta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota**2 * sqrtg * g_pp).flatten())
    )
    B = B.at[rho_idx, theta_idx].add(
        jnp.diag(n0 * (W * psi_r * sqrtg * g_rv).flatten())
    )
    B = B.at[rho_idx, zeta_idx].add(
        jnp.diag(n0 * (W * psi_r * iota * sqrtg * g_rp).flatten())
    )
    B = B.at[theta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota * sqrtg * g_vp).flatten())
    )

    # symmetrizing the matrix
    if axisym:
        A = A.at[theta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, theta_idx]).T)
        A = A.at[zeta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, zeta_idx]).T)
        A = A.at[zeta_idx, theta_idx].set(jnp.conjugate(A[theta_idx, zeta_idx]).T)

        B = B.at[theta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, theta_idx]).T)
        B = B.at[zeta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, zeta_idx]).T)
        B = B.at[zeta_idx, theta_idx].set(jnp.conjugate(B[theta_idx, zeta_idx]).T)
    else:
        A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
        A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
        A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)

        B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)
        B = B.at[zeta_idx, rho_idx].set(B[rho_idx, zeta_idx].T)
        B = B.at[zeta_idx, theta_idx].set(B[theta_idx, zeta_idx].T)

    # Force all the
    A = A.at[all_idx, all_idx].set(force_psd(A))

    # Finally add the only instability drive term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((W * psi_r2 * sqrtg * F).flatten()))

    y = A - A.conj().T
    print(jnp.max(jnp.abs(y)))

    y = B - B.conj().T
    print(jnp.max(jnp.abs(y)))

    # pdb.set_trace()

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    # --no-verify w, _ = jnp.linalg.eigh(B[jnp.ix_(keep, keep)])
    # --no-verify w, v = jnp.linalg.eigh(A[jnp.ix_(keep, keep)])
    # --no-verify w, v = jax.scipy.linalg.eigh(A[jnp.ix_(keep, keep)], B[jnp.ix_(keep, keep)])
    # --no-verify w, v =  eigh(jax.numpy.asarray(A[jnp.ix_(keep, keep)]), jax.numpy.asarray(B[jnp.ix_(keep, keep)]), subset_by_index=[0, 2])
    w, v = eigh(
        jax.numpy.asarray(A[jnp.ix_(keep, keep)]),
        jax.numpy.asarray(B[jnp.ix_(keep, keep)]),
    )

    ## This will be the most expensive but easiest automatically differentiable way.
    ## TODO: Multiply B with a permutation matrix P so that it becomes block diagonal
    ## That will make cholesky ~ 20 x faster
    # L = jnp.linalg.cholesky(B[jnp.ix_(keep, keep)])
    # w, v = jnp.linalg.eigh(L.T @ A[jnp.ix_(keep, keep)] @ L)

    data["finite-n lambda2"] = w
    data["finite-n eigenfunction2"] = v

    return data


@register_compute_fun(
    name="finite-n lambda3",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate"
    + "using the most compact representation of diffmatrices",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
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
        "finite-n instability drive",
        "iota",
        "iota_r",
        "psi_r",
        "p",
        "a",
    ],
    n_rho_max="int: 2 x maximum radial mode number",
    n_theta_max="int: 2 x maximum poloidal mode number",
    n_zeta_max="int: 2 x maximum toroidal mode number",
    axisym="bool: if the equilibrium is axisymmetric",
)
def _AGNI3(params, transforms, profiles, data, **kwargs):
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
    B_N = params["Psi"] / (jnp.pi * a_N**2)

    iota = data["iota"][:, None]
    iota_r = data["iota_r"][:, None]

    psi_r = data["psi_r"][:, None] / (0.5 * a_N**2 * B_N)
    psi_rr = data["psi_rr"][:, None] / (0.5 * a_N**2 * B_N)

    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r = iota * psi_r
    iota_psi_r2 = iota * psi_r2

    dpsi_r2_drho = 2 * psi_r * psi_rr
    diota_psi_r2_drho = iota_r * psi_r**2 + iota * dpsi_r2_drho

    p0 = data["p"] / B_N**2
    p0 = p0.at[p0 < 0].set(1e-8)

    n0 = 1e2

    axisym = kwargs.get("axisym", False)

    # For axisymmetric equilibria n_mode will decide the toroidal
    # mode number to analyze. Should work for n_mode = 0 (vertical instability)
    # For stellarator equilibrium n_mode will decide the n_mode family
    n_mode = kwargs.get("n_mode", 1)

    # n_mode = jnp.mod(n_mode, NFP)

    n_rho_max = kwargs.get("n_rho_max", 8)
    n_theta_max = kwargs.get("n_theta_max", 8)

    if axisym:
        # Each componenet of xi can be written as the Fourier sum of two modes in
        # the toroidal direction
        D_zeta0 = 1j * n_mode * jnp.array([[1]])
        n_zeta_max = 1
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]
    else:
        n_zeta_max = kwargs.get("n_zeta_max", 4)
        D_zeta0 = fourier_diffmat(n_zeta_max)
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]

    def _eval_1D(f, x):
        return jax.vmap(lambda x_val: f(x_val))(x)

    def _f(x):
        x_0 = 0.4
        m_1 = 3.0
        m_2 = 3.0
        lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
        upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
        eps = 1.0e-3
        # eps1 = 1.5e-2
        # return eps + (1 - eps1) * (lower + upper)
        return eps + (1 - eps) * (lower + upper)

    # ∫dρₛ (∂X/∂ρₛ) = ∫dρ f'(ρ) (∂ρ/∂ρₛ) (∂X/∂ρ)

    dx_f = jax.grad(_f)

    x = legendre_lobatto_nodes(len(x0) - 1)

    scale_vector1 = (_eval_1D(dx_f, x)) ** -1
    # scale_vector1 = jnp.ones_like(x0) * (2 * jnp.pi-1e-3)

    # scale_vector1 = jnp.ones_like(x0) * (1 - 1e-3)
    # h = (1-1e-3)/(n_rho_max-1)

    scale_x1 = scale_vector1[:, None]

    # Get differentiation matrices
    # RG: setting the gradient to 0 to save some memory?
    D_rho0 = legendre_D1(n_rho_max - 1) * scale_x1
    # D_rho0 = fourier_diffmat(n_rho_max) * scale_x1
    # D_rho0, W0 = D1_FD_4(n_rho_max, h)
    # D_rho0 = D_rho0 * scale_x1

    D_theta0 = fourier_diffmat(n_theta_max)

    wrho = jnp.diag(1 / scale_vector1 * legendre_lobatto_weights(n_rho_max - 1))
    wrho = wrho.at[jnp.abs(wrho) < 1e-12].set(0.0)

    I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
    I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
    D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
    D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))

    D_rhoT = jax.lax.stop_gradient(jnp.kron(D_rho0.T, jnp.kron(I_theta0, I_zeta0)))
    D_thetaT = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0.T, I_zeta0)))
    D_zetaT = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0.T)))

    n_total = n_rho_max * n_theta_max * n_zeta_max

    ## Create the full matrix
    if axisym:
        A = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.complex128)
        B = jnp.zeros((3 * n_total, 3 * n_total))
    else:
        A = jnp.zeros((3 * n_total, 3 * n_total))
        B = jnp.zeros((3 * n_total, 3 * n_total))

    # Define block indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)
    all_idx = slice(0, 3 * n_total)

    ### assuming uniform spacing in and θ and ζ
    wtheta = 2 * jnp.pi / n_theta_max
    wzeta = 2 * jnp.pi / n_zeta_max

    W = jnp.diag(jnp.kron(wrho * wtheta * wzeta, jnp.kron(I_theta0, I_zeta0)))[:, None]

    sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3

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
    j_sup_theta = mu_0 * data["J^theta_PEST"][:, None] * a_N**2 / B_N
    j_sup_zeta = mu_0 * data["J^zeta"][:, None] * a_N**2 / B_N

    # instability drive term
    F = 1 * mu_0 * data["finite-n instability drive"][:, None] * (a_N / B_N) ** 2

    ####################
    ####----Q_ρρ----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * D_theta)
        + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * g_rr) * D_zeta)
        + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta)
        + ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta).T @ D_theta
    )

    ####################
    ####----Q_ϑϑ ----####
    ####################
    # enforcing symmetry exactly
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).T @ D_zeta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).T @ D_zeta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1.0 * (D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta))
    )

    A = A.at[rho_idx, rho_idx].add(
        +(D_rho * iota_psi_r2.T).T
        @ ((psi_r_over_sqrtg * W * g_vv / psi_r) * (D_rho * iota_psi_r2.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        -1 * (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        1 * (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    ####################
    ####----Q_ζζ----####
    ####################
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1 * D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, rho_idx].add(
        +(D_rho * psi_r2.T).T
        @ ((psi_r_over_sqrtg * W * g_pp / psi_r) * (D_rho * psi_r2.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        1 * (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        -1 * (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    ####################
    ####----Q_ρϑ----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T
            @ ((iota * psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            + D_zeta.T
            @ ((psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            ((iota * psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T)).T
            @ D_theta
            + ((psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T)).T
            @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    ######################
    ####-----Q_ρζ-----####
    ######################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T
            @ ((iota * psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
            + D_zeta.T @ ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            ((iota * psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T)).T
            @ D_theta
            + ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T)).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )

    ##########################
    #######-----Q_ϑζ-----#####
    ##########################
    A = A.at[theta_idx, theta_idx].add(
        -1
        * (
            D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
        )
    )

    A = A.at[zeta_idx, zeta_idx].add(
        -1
        * (
            D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
            - (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
            - (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
        + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
    )

    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            (D_rho * iota_psi_r2.T).T
            @ ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * psi_r2.T))
        )
    )
    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * psi_r2.T)).T
            @ (D_rho * iota_psi_r2.T)
        )
    )

    # Mixed Q-J term ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐
    # \xi^{\rho} (\mathbf{J} \times \nabla\rho)/|\nabla \rho|^2 \cdot \mathbf{Q}
    A = A.at[rho_idx, rho_idx].add(
        -1
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
        -1
        * (
            (
                (
                    W
                    * psi_r3
                    * sqrtg
                    * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                    / g_sup_rr
                )
                * (iota * D_theta + D_zeta)
            ).T
            + ((W * sqrtg * psi_r * j_sup_zeta) * (D_rho * iota_psi_r2.T)).T
            + ((W * sqrtg * psi_r * j_sup_theta) * (D_rho * psi_r2.T)).T
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -(W * psi_r2 * sqrtg * j_sup_theta) * D_theta
        + (W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
    )
    A = A.at[rho_idx, zeta_idx].add(
        +(W * psi_r2 * sqrtg * j_sup_theta) * D_theta
        - (W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
    )

    ## diagonal |J|^2 term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((psi_r2 * W * sqrtg * J2).flatten()))

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(jnp.diag(n0 * (W * psi_r2 * sqrtg * g_rr).flatten()))
    B = B.at[theta_idx, theta_idx].add(jnp.diag(n0 * (W * sqrtg * g_vv).flatten()))
    B = B.at[zeta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota**2 * sqrtg * g_pp).flatten())
    )

    B = B.at[rho_idx, theta_idx].add(
        jnp.diag(n0 * (W * psi_r * sqrtg * g_rv).flatten())
    )
    B = B.at[rho_idx, zeta_idx].add(
        jnp.diag(n0 * (W * psi_r * iota * sqrtg * g_rp).flatten())
    )
    B = B.at[theta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota * sqrtg * g_vp).flatten())
    )

    # The matrix must be Hermitian so we fill out the lower blocks
    if axisym:
        A = A.at[theta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, theta_idx]).T)
        A = A.at[zeta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, zeta_idx]).T)
        A = A.at[zeta_idx, theta_idx].set(jnp.conjugate(A[theta_idx, zeta_idx]).T)

        B = B.at[theta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, theta_idx]).T)
        B = B.at[zeta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, zeta_idx]).T)
        B = B.at[zeta_idx, theta_idx].set(jnp.conjugate(B[theta_idx, zeta_idx]).T)
    else:
        A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
        A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
        A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)

        B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)
        B = B.at[zeta_idx, rho_idx].set(B[rho_idx, zeta_idx].T)
        B = B.at[zeta_idx, theta_idx].set(B[theta_idx, zeta_idx].T)

    ## Shift the diagon of A to ensure positive definiteness
    ## The estimate must be accurate. If A is diagonally dominant
    ## use Gerhsgorin theorem to estimate the lowest eigenvalue
    # A = 0.5 * (A + A.T)
    # B = 0.5 * (B + B.T)

    A = A.at[jnp.diag_indices_from(A)].add(1e-11)
    # LAinv = jnp.linalg.inv(jnp.linalg.cholesky(A))

    ## Similarity transform to
    # A = LAinv @ A @ LAinv.T
    # B = LAinv @ B @ LAinv.T

    # B = B.at[jnp.diag_indices_from(B)].add(1e-11)

    ## Finally add the only instability drive term
    Au = jnp.zeros((3 * n_total, 3 * n_total))
    Au = Au.at[rho_idx, rho_idx].add(jnp.diag((W * psi_r2 * sqrtg * F).flatten()))

    ## D = 1.0 /jnp.tile((W * sqrtg).flatten(), 3)[:, None]
    D = jnp.diag(1 / jnp.sqrt(jnp.diag(B)))

    A2 = D @ (A @ D.T)
    Au2 = D @ (Au @ D.T)
    B2 = D @ (B @ D.T)

    ##w2, _ = jnp.linalg.eigh(A)
    ##print(w2)
    # w4, _ = jnp.linalg.eigh((B2 + B2.T)/2)
    # print(w4)

    A2 = A2.at[jnp.diag_indices_from(A2)].add(1e-11)

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    A3 = A2[jnp.ix_(keep, keep)] + Au2[jnp.ix_(keep, keep)]

    # w2, _ = jnp.linalg.eigh((A3 + A3.T) / 2)
    # print(w2)

    # tic = time.time()
    # w, v = jax.scipy.linalg.eigh(A3, B2[jnp.ix_(keep, keep)])
    # toc = time.time()
    # print(toc-tic)

    ### This will be the most expensive but easiest automatically differentiable way.
    ### TODO: Multiply B with a permutation matrix P so that it becomes block diagonal
    ### then Cholesky factorize each block. That will make cholesky ~ 3**3 x faster
    # L = jnp.linalg.cholesky(B2[jnp.ix_(keep, keep)])
    ## Linv = jnp.linalg.inv(L)
    ## Right-multiply by L^{-T}:  ALt = A L^{-T}
    # ALt = solve_triangular(L, A3.T, lower=True).T
    ## Left-multiply by L^{-1}:   C = L^{-1} (A L^{-T})
    # A3 = solve_triangular(L, ALt, lower=True)

    # tic = time.time()
    ### This will be the most expensive but easiest automatically differentiable way.
    ### TODO: Multiply B with a permutation matrix P so that it becomes block diagonal
    ### then Cholesky factorize each block. That will make cholesky ~ 3**3 x faster
    # L = jnp.linalg.cholesky(B2[jnp.ix_(keep, keep)])
    ## Linv = jnp.linalg.inv(L)
    ## Right-multiply by L^{-T}:  ALt = A L^{-T}
    # ALt = solve_triangular(L, A3.T, lower=True).T
    ## Left-multiply by L^{-1}:   C = L^{-1} (A L^{-T})
    # A4 = solve_triangular(L, ALt, lower=True)
    # print(A4)
    # toc = time.time()
    # print("time taken by LU =", toc-tic)

    ### w, v = jnp.linalg.eigh(Linv @ A[jnp.ix_(keep, keep)] @ Linv.T)
    # tic = time.time()
    # w, v = jnp.linalg.eigh(A3)

    scale1 = 1e3

    A3 = scale1 * np.asarray(A3[jnp.ix_(keep, keep)])
    B2 = np.asarray(B2[jnp.ix_(keep, keep)])
    print("arrays created!")
    tic = time.time()
    # w, v = eigsh(A3, k=10, which="SA", sigma=-1e-3, tol=1e-5, maxiter=100)
    w, v = eigsh(
        A3, M=B2, k=5, which="SA", sigma=-1.0, tol=1e-5, maxiter=200, mode="cayley"
    )

    print(w)
    toc = time.time()
    print(toc - tic)
    pdb.set_trace()

    data["finite-n lambda3"] = w / scale1
    data["finite-n eigenfunction3"] = v

    return data


@register_compute_fun(
    name="finite-n lambda4",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate"
    + "using the most compact representation of diffmatrices",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
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
        "finite-n instability drive",
        "iota",
        "iota_r",
        "psi_r",
        "p",
        "a",
    ],
    n_rho_max="int: 2 x maximum radial mode number",
    n_theta_max="int: 2 x maximum poloidal mode number",
    n_zeta_max="int: 2 x maximum toroidal mode number",
    axisym="bool: if the equilibrium is axisymmetric",
)
def _AGNI4(params, transforms, profiles, data, **kwargs):
    """
    AGNI4: Analysis of Global Normal-modes in Ideal MHD.

    Based on the original source here:
    https://github.com/rahulgaur104/AGNI/tree/master

    A finite-n stability eigenvalue solver.
    Currenly only finds fixed boundary unstable modes at
    low to medium resolution.

    This version of the code keeps is similar to the previous one
    except we don't scale xi^rho by psi_r or multiply the energey integral
    by an extra sqrtg so a bunch of factors are different
    """
    a_N = data["a"]
    B_N = params["Psi"] / (jnp.pi * a_N**2)

    iota = data["iota"][:, None]
    iota_r = data["iota_r"][:, None]

    psi_r = data["psi_r"][:, None] / (0.5 * a_N**2 * B_N)
    psi_rr = data["psi_rr"][:, None] / (0.5 * a_N**2 * B_N)

    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r = iota * psi_r
    iota_psi_r2 = iota * psi_r2

    dpsi_r2_drho = 2 * psi_r * psi_rr
    diota_psi_r2_drho = iota_r * psi_r**2 + iota * dpsi_r2_drho

    p0 = data["p"] / B_N**2
    p0 = p0.at[p0 < 0].set(1e-8)

    # n0 = (p0 ** (1 / 3) + 1e-4 / (psi_r2.flatten()))
    n0 = 1e2

    axisym = kwargs.get("axisym", False)

    # For axisymmetric equilibria n_mode will decide the toroidal
    # mode number to analyze. Should work for n_mode = 0 (vertical instability)
    # For stellarator equilibrium n_mode will decide the n_mode family
    n_mode = kwargs.get("n_mode", 1)

    # n_mode = jnp.mod(n_mode, NFP)

    n_rho_max = kwargs.get("n_rho_max", 8)
    n_theta_max = kwargs.get("n_theta_max", 8)

    if axisym:
        # Each componenet of xi can be written as the Fourier sum of two modes in
        # the toroidal direction
        D_zeta0 = 1j * n_mode * jnp.array([[1]])
        n_zeta_max = 1
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]
    else:
        n_zeta_max = kwargs.get("n_zeta_max", 4)
        D_zeta0 = fourier_diffmat(n_zeta_max)
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]

    def _eval_1D(f, x):
        return jax.vmap(lambda x_val: f(x_val))(x)

    def _f(x):
        x_0 = 0.4
        m_1 = 3.0
        m_2 = 3.0
        lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
        upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
        eps = 5.0e-3
        return eps + (1 - eps) * (lower + upper)

    # ∫dρₛ (∂X/∂ρₛ) = ∫dρ f'(ρ) (∂ρ/∂ρₛ) (∂X/∂ρ)

    dx_f = jax.grad(_f)

    x = legendre_lobatto_nodes(len(x0) - 1)

    scale_vector1 = (_eval_1D(dx_f, x)) ** -1
    # scale_vector1 = jnp.ones_like(x0) * (2 * jnp.pi-1e-3)

    # scale_vector1 = jnp.ones_like(x0) * (1 - 1e-3)
    # h = (1-1e-3)/(n_rho_max-1)

    scale_x1 = scale_vector1[:, None]

    # Get differentiation matrices
    # RG: setting the gradient to 0 to save some memory?
    D_rho0 = legendre_D1(n_rho_max - 1) * scale_x1
    # D_rho0 = fourier_diffmat(n_rho_max) * scale_x1
    # D_rho0, W0 = D1_FD_4(n_rho_max, h)
    # D_rho0 = D_rho0 * scale_x1

    D_theta0 = fourier_diffmat(n_theta_max)

    wrho = jnp.diag(1 / scale_vector1 * legendre_lobatto_weights(n_rho_max - 1))
    wrho = wrho.at[jnp.abs(wrho) < 1e-12].set(0.0)
    # w0 = jnp.diag((2*jnp.pi-1e-3)/n_rho_max * jnp.ones_like(x0))

    I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
    I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
    D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
    D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))

    D_rhoT = jax.lax.stop_gradient(jnp.kron(D_rho0.T, jnp.kron(I_theta0, I_zeta0)))
    D_thetaT = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0.T, I_zeta0)))
    D_zetaT = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0.T)))

    n_total = n_rho_max * n_theta_max * n_zeta_max

    ## Create the full matrix
    if axisym:
        A = jnp.zeros((3 * n_total, 3 * n_total), dtype=jnp.complex128)
        B = jnp.zeros((3 * n_total, 3 * n_total))
    else:
        A = jnp.zeros((3 * n_total, 3 * n_total))
        B = jnp.zeros((3 * n_total, 3 * n_total))

    # Define block indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)
    all_idx = slice(0, 3 * n_total)

    ### assuming uniform spacing in and θ and ζ
    wtheta = 2 * jnp.pi / n_theta_max
    wzeta = 2 * jnp.pi / n_zeta_max

    W = jnp.diag(jnp.kron(wrho * wtheta * wzeta, jnp.kron(I_theta0, I_zeta0)))[:, None]

    sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3

    sqrtg2 = sqrtg**2
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
    j_sup_theta = mu_0 * data["J^theta_PEST"][:, None] * a_N**2 / B_N
    j_sup_zeta = mu_0 * data["J^zeta"][:, None] * a_N**2 / B_N

    # instability drive term
    F = 1 * mu_0 * data["finite-n instability drive"][:, None] * (a_N / B_N) ** 2

    ####################
    ####----Q_11----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        D_thetaT @ ((iota**2 * psi_r2 * W * g_rr) * D_theta)
        + D_zetaT @ ((W * psi_r2 * g_rr) * D_zeta)
        + D_thetaT @ ((iota * psi_r2 * W * g_rr) * D_zeta)
        + ((iota * psi_r2 * W * g_rr) * D_zeta).T @ D_theta
    )

    ####################
    ####----Q_22----####
    ####################
    # enforcing symmetry exactly
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r2 * W * g_vv) * D_zeta)
            + ((psi_r2 * W * g_vv) * D_zeta).T @ D_zeta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r2 * W * g_vv) * D_zeta)
            + ((psi_r2 * W * g_vv) * D_zeta).T @ D_zeta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(-1.0 * (D_zetaT @ ((psi_r2 * W * g_vv) * D_zeta)))

    # W g_ϑϑ (∂ᵨ (ι ψ' ξ^ρ))²
    A = A.at[rho_idx, rho_idx].add(
        +(D_rho * iota_psi_r.T).T @ ((W * g_vv) * (D_rho * iota_psi_r.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        -1 * (D_rho * iota_psi_r.T).T @ ((psi_r * W * g_vv) * D_zeta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        1 * (D_rho * iota_psi_r.T).T @ ((psi_r * W * g_vv) * D_zeta)
    )

    ####################
    ####----Q_33----####
    ####################
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r2 * W * g_pp) * D_theta)
            + ((psi_r2 * W * g_pp) * D_theta).T @ D_theta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r2 * W * g_pp) * D_theta)
            + ((psi_r2 * W * g_pp) * D_theta).T @ D_theta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(-1 * D_theta.T @ ((psi_r2 * W * g_pp) * D_theta))

    A = A.at[rho_idx, rho_idx].add(
        +(D_rho * psi_r.T).T @ ((W * g_pp) * (D_rho * psi_r.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        1 * (D_rho * psi_r.T).T @ ((psi_r * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        -1 * (D_rho * psi_r.T).T @ ((psi_r * W * g_pp) * D_theta)
    )

    # C = A.copy()
    # C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    # C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    # C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)

    ## apply dirichlet BC to ξ^ρ
    # keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - (n_theta_max * n_zeta_max))
    # keep_2 = jnp.arange(n_total, 3 * n_total)
    # keep = jnp.concatenate([keep_1, keep_2])

    # w, v = jnp.linalg.eigh(C[jnp.ix_(keep, keep)])
    ## w, v = jnp.linalg.eigh(C[n_theta_max*n_zeta_max:, n_theta_max*n_zeta_max:])
    # print(w)
    # pdb.set_trace()

    ####################
    ####----Q_12----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T @ ((iota_psi_r * W * g_rv) * (D_rho * iota_psi_r.T))
            + D_zeta.T @ ((psi_r * W * g_rv) * (D_rho * iota_psi_r.T))
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            ((iota_psi_r * W * g_rv) * (D_rho * iota_psi_r.T)).T @ D_theta
            + ((psi_r * W * g_rv) * (D_rho * iota_psi_r.T)).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        D_theta.T @ ((iota * psi_r2 * W * g_rv) * D_zeta)
        + D_zeta.T @ ((psi_r2 * W * g_rv) * D_zeta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_theta.T @ ((iota_psi_r2 * W * g_rv) * D_zeta)
            + D_zeta.T @ ((psi_r2 * W * g_rv) * D_zeta)
        )
    )

    ######################
    ####-----Q_13-----####
    ######################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r * W * g_rp) * (D_rho * psi_r.T))
            + D_zeta.T @ ((psi_r * W * g_rp) * (D_rho * psi_r.T))
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            ((iota * psi_r * W * g_rp) * (D_rho * psi_r.T)).T @ D_theta
            + ((psi_r * W * g_rp) * (D_rho * psi_r.T)).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * W * g_rp) * D_theta)
            + D_zeta.T @ ((psi_r2 * W * g_rp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            D_theta.T @ ((iota * psi_r2 * W * g_rp) * D_theta)
            + D_zeta.T @ ((psi_r2 * W * g_rp) * D_theta)
        )
    )

    ##########################
    #######-----Q_23-----#####
    ##########################
    A = A.at[theta_idx, theta_idx].add(
        -1
        * (
            D_zeta.T @ ((psi_r * W * psi_r * g_vp) * D_theta)
            + ((psi_r * W * psi_r * g_vp) * D_theta).T @ D_zeta
        )
    )

    A = A.at[zeta_idx, zeta_idx].add(
        -1
        * (
            D_zeta.T @ ((psi_r * W * psi_r * g_vp) * D_theta)
            + ((psi_r * W * psi_r * g_vp) * D_theta).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            (D_rho * psi_r.T).T @ ((psi_r * W * g_vp) * D_zeta)
            - (D_rho * iota_psi_r.T).T @ ((psi_r * W * g_vp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            (D_rho * psi_r.T).T @ ((psi_r * W * g_vp) * D_zeta)
            - (D_rho * iota_psi_r.T).T @ ((psi_r * W * g_vp) * D_theta)
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        D_zeta.T @ ((psi_r * W * psi_r * g_vp) * D_theta)
        + ((psi_r * W * psi_r * g_vp) * D_theta).T @ D_zeta
    )

    A = A.at[rho_idx, rho_idx].add(
        1 * ((D_rho * iota_psi_r.T).T @ ((W * g_vp) * (D_rho * psi_r.T)))
    )
    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        1 * (((W * g_vp) * (D_rho * psi_r.T)).T @ (D_rho * iota_psi_r.T))
    )

    # C = A.copy()
    # C = C.at[theta_idx, rho_idx].set(C[rho_idx, theta_idx].T)
    # C = C.at[zeta_idx, rho_idx].set(C[rho_idx, zeta_idx].T)
    # C = C.at[zeta_idx, theta_idx].set(C[theta_idx, zeta_idx].T)

    ## apply dirichlet BC to ξ^ρ
    # keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - (n_theta_max * n_zeta_max))
    # keep_2 = jnp.arange(n_total, 3 * n_total)
    # keep = jnp.concatenate([keep_1, keep_2])

    # w, v = jnp.linalg.eigh(C[jnp.ix_(keep, keep)])
    ## w, v = jnp.linalg.eigh(C[n_theta_max*n_zeta_max:, n_theta_max*n_zeta_max:])
    # print(w)
    # pdb.set_trace()

    # Mixed Q-J term
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (
                W
                * psi_r
                * sqrtg2
                * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                / g_sup_rr
            )
            * (iota * D_theta + D_zeta)
            + (W * sqrtg2 * j_sup_zeta) * (D_rho * iota_psi_r.T)
            + (W * sqrtg2 * j_sup_theta) * (D_rho * psi_r.T)
        )
    )

    # ρ-ρ block transposed for symmetry
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            (
                (
                    W
                    * psi_r
                    * sqrtg2
                    * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                    / g_sup_rr
                )
                * (iota * D_theta + D_zeta)
            ).T
            + ((W * sqrtg2 * j_sup_zeta) * (D_rho * iota_psi_r.T)).T
            + ((W * sqrtg2 * j_sup_theta) * (D_rho * psi_r.T)).T
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -(W * psi_r * sqrtg2 * j_sup_theta) * D_theta
        + (W * psi_r * sqrtg2 * j_sup_zeta) * D_zeta
    )
    A = A.at[rho_idx, zeta_idx].add(
        +(W * psi_r * sqrtg2 * j_sup_theta) * D_theta
        - (W * psi_r * sqrtg2 * j_sup_zeta) * D_zeta
    )

    ## diagonal |J|^2 term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((W * sqrtg2 * J2).flatten()))

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(jnp.diag(n0 * (W * sqrtg2 * g_rr).flatten()))
    B = B.at[theta_idx, theta_idx].add(jnp.diag(n0 * (W * sqrtg2 * g_vv).flatten()))
    B = B.at[zeta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota**2 * sqrtg2 * g_pp).flatten())
    )

    B = B.at[rho_idx, theta_idx].add(jnp.diag(n0 * (W * sqrtg2 * g_rv).flatten()))
    B = B.at[rho_idx, zeta_idx].add(jnp.diag(n0 * (W * iota * sqrtg2 * g_rp).flatten()))
    B = B.at[theta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota * sqrtg2 * g_vp).flatten())
    )

    # The matrix must be Hermitian so we fill out the lower blocks
    if axisym:
        A = A.at[theta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, theta_idx]).T)
        A = A.at[zeta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, zeta_idx]).T)
        A = A.at[zeta_idx, theta_idx].set(jnp.conjugate(A[theta_idx, zeta_idx]).T)

        B = B.at[theta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, theta_idx]).T)
        B = B.at[zeta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, zeta_idx]).T)
        B = B.at[zeta_idx, theta_idx].set(jnp.conjugate(B[theta_idx, zeta_idx]).T)
    else:
        A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
        A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
        A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)

        B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)
        B = B.at[zeta_idx, rho_idx].set(B[rho_idx, zeta_idx].T)
        B = B.at[zeta_idx, theta_idx].set(B[theta_idx, zeta_idx].T)

    # A = A.at[jnp.diag_indices_from(A)].add(1e-11)

    # Finally add the only instability drive term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((W * sqrtg2 * F).flatten()))

    ## D = 1.0 /jnp.tile((W * sqrtg).flatten(), 3)[:, None]
    D = jnp.diag(1 / jnp.sqrt(jnp.diag(B)))
    # D = jnp.diag(1 / jnp.diag(jnp.ones_like(B)))

    A2 = D @ (A @ D.T)
    B2 = D @ (B @ D.T)

    # w4, _ = jnp.linalg.eigh((B2 + B2.T) / 2)
    # print(w4)
    # w2, _ = jnp.linalg.eigh((A2 + A2.T)/2)
    # print(w2)

    A2 = A2.at[jnp.diag_indices_from(A2)].add(1e-11)

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    A3 = A2[jnp.ix_(keep, keep)]

    pdb.set_trace()

    ## This will be the most expensive but easiest automatically differentiable way.
    ## TODO: Multiply B with a permutation matrix P so that it becomes block diagonal
    ## then Cholesky factorize each block. That will make cholesky ~ 3**3 x faster
    L = jnp.linalg.cholesky(B2[jnp.ix_(keep, keep)])
    # Linv = jnp.linalg.inv(L)
    # Right-multiply by L^{-T}:  ALt = A L^{-T}
    ALt = solve_triangular(L, A3.T, lower=True).T
    # Left-multiply by L^{-1}:   C = L^{-1} (A L^{-T})
    A3 = solve_triangular(L, ALt, lower=True)

    # w, v = jnp.linalg.eigh(Linv @ A[jnp.ix_(keep, keep)] @ Linv.T)
    w, v = jnp.linalg.eigh(A3)

    data["finite-n lambda4"] = w
    data["finite-n eigenfunction4"] = v

    return data


@register_compute_fun(
    name="finite-n lambda5",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate"
    + "using the most compact representation of diffmatrices",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
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
        "iota_r",
        "psi_r",
        "p",
        "a",
    ],
    n_rho_max="int: 2 x maximum radial mode number",
    n_theta_max="int: 2 x maximum poloidal mode number",
    n_zeta_max="int: 2 x maximum toroidal mode number",
    axisym="bool if the equilibrium is axisymmetric",
    n_zeta_axisym="bool: max axisym mode number to analyze",
    compressible="bool: if the perturbation is compressible",
)
def _AGNI5(params, transforms, profiles, data, **kwargs):
    """
    AGNI: Analysis of Global Normal-modes in Ideal MHD.

    Based on the original source here:
    https://github.com/rahulgaur104/AGNI/tree/master

    A finite-n stability eigenvalue solver.
    Currenly only finds fixed boundary unstable modes at
    low to medium resolution.

    In this version, we add the incompressibility constraint,
    reducing the size of the matrix.
    """
    a_N = data["a"]
    B_N = params["Psi"] / (jnp.pi * a_N**2)

    iota = data["iota"][:, None]
    iota_r = data["iota_r"][:, None]

    psi_r = data["psi_r"][:, None] / (0.5 * a_N**2 * B_N)
    psi_rr = data["psi_rr"][:, None] / (0.5 * a_N**2 * B_N)

    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r = iota * psi_r
    iota_psi_r2 = iota * psi_r2

    dpsi_r2_drho = 2 * psi_r * psi_rr
    diota_psi_r2_drho = iota_r * psi_r**2 + iota * dpsi_r2_drho

    p0 = data["p"] / B_N**2
    p0 = p0.at[p0 < 0].set(1e-8)

    n0 = 1e2

    axisym = kwargs.get("axisym", False)
    compressible = kwargs.get("compressible", False)
    n_zeta_axisym = kwargs.get("n_zeta_axisym", 1)

    # For axisymmetric equilibria n_mode will decide the toroidal
    # mode number to analyze. Should work for n_mode = 0 (vertical instability)
    # For stellarator equilibrium n_mode will decide the n_mode family
    n_mode = kwargs.get("n_mode", 1)

    # n_mode = jnp.mod(n_mode, NFP)

    n_rho_max = kwargs.get("n_rho_max", 8)
    n_theta_max = kwargs.get("n_theta_max", 8)

    if axisym:
        # Each componenet of xi can be written as the Fourier sum of two modes in
        # the toroidal direction
        D_zeta0 = n_zeta_axisym * jnp.array([[0, -1], [1, 0]])
        n_zeta_max = 1
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]
    else:
        n_zeta_max = kwargs.get("n_zeta_max", 4)
        D_zeta0 = fourier_diffmat(n_zeta_max)
        x0 = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]

    def _eval_1D(f, x):
        return jax.vmap(lambda x_val: f(x_val))(x)

    def _f(x):
        x_0 = 0.4
        m_1 = 3.0
        m_2 = 3.0
        lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
        upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
        eps = 1.0e-3
        # eps1 = 1.5e-2
        # return eps + (1 - eps1) * (lower + upper)
        return eps + (1 - eps) * (lower + upper)

    # ∫dρₛ (∂X/∂ρₛ) = ∫dρ f'(ρ) (∂ρ/∂ρₛ) (∂X/∂ρ)

    dx_f = jax.grad(_f)

    x = legendre_lobatto_nodes(len(x0) - 1)

    scale_vector1 = (_eval_1D(dx_f, x)) ** -1
    # scale_vector1 = jnp.ones_like(x0) * (2 * jnp.pi-1e-3)

    # scale_vector1 = jnp.ones_like(x0) * (1 - 1e-3)
    # h = (1-1e-3)/(n_rho_max-1)

    scale_x1 = scale_vector1[:, None]

    # Get differentiation matrices
    # RG: setting the gradient to 0 to save some memory?
    D_rho0 = legendre_D1(n_rho_max - 1) * scale_x1
    # D_rho0 = fourier_diffmat(n_rho_max) * scale_x1
    # D_rho0, W0 = D1_FD_4(n_rho_max, h)
    # D_rho0 = D_rho0 * scale_x1

    D_theta0 = fourier_diffmat(n_theta_max)

    wrho = jnp.diag(1 / scale_vector1 * legendre_lobatto_weights(n_rho_max - 1))
    wrho = wrho.at[jnp.abs(wrho) < 1e-12].set(0.0)

    I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
    I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0)))
    D_theta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0)))
    D_zeta = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0)))

    D_rhoT = jax.lax.stop_gradient(jnp.kron(D_rho0.T, jnp.kron(I_theta0, I_zeta0)))
    D_thetaT = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(D_theta0.T, I_zeta0)))
    D_zetaT = jax.lax.stop_gradient(jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0.T)))

    n_total = n_rho_max * n_theta_max * n_zeta_max

    # Define block indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)
    all_idx = slice(0, 3 * n_total)

    ### assuming uniform spacing in and θ and ζ
    wtheta = 2 * jnp.pi / n_theta_max
    wzeta = 2 * jnp.pi / n_zeta_max

    W = jnp.diag(jnp.kron(wrho * wtheta * wzeta, jnp.kron(I_theta0, I_zeta0)))[:, None]

    sqrtg = data["sqrt(g)_PEST"][:, None] * 1 / a_N**3

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
    j_sup_theta = mu_0 * data["J^theta_PEST"][:, None] * a_N**2 / B_N
    j_sup_zeta = mu_0 * data["J^zeta"][:, None] * a_N**2 / B_N

    # instability drive term
    F = 1 * mu_0 * data["finite-n instability drive"][:, None] * (a_N / B_N) ** 2

    A = jnp.zeros((3 * n_total, 3 * n_total))
    B = jnp.zeros((3 * n_total, 3 * n_total))

    ####################
    ####----Q_ρρ----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * D_theta)
        + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * g_rr) * D_zeta)
        + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta)
        + ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta).T @ D_theta
    )

    ####################
    ####----Q_ϑϑ ----####
    ####################
    # enforcing symmetry exactly
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).T @ D_zeta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
            + ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta).T @ D_zeta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1.0 * (D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta))
    )

    A = A.at[rho_idx, rho_idx].add(
        +(D_rho * iota_psi_r2.T).T
        @ ((psi_r_over_sqrtg * W * g_vv / psi_r) * (D_rho * iota_psi_r2.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        -1 * (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        1 * (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
    )

    ####################
    ####----Q_ζζ----####
    ####################
    A = A.at[theta_idx, theta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )
    A = A.at[zeta_idx, zeta_idx].add(
        0.5
        * (
            D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
            + ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta).T @ D_theta
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        -1 * D_theta.T @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, rho_idx].add(
        +(D_rho * psi_r2.T).T
        @ ((psi_r_over_sqrtg * W * g_pp / psi_r) * (D_rho * psi_r2.T))
    )

    A = A.at[rho_idx, theta_idx].add(
        1 * (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    A = A.at[rho_idx, zeta_idx].add(
        -1 * (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
    )

    ####################
    ####----Q_ρϑ----####
    ####################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T
            @ ((iota * psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
            + D_zeta.T
            @ ((psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T))
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            ((iota * psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T)).T
            @ D_theta
            + ((psi_r * psi_r_over_sqrtg * W * g_rv) * (D_rho * iota_psi_r2.T)).T
            @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
    )
    A = A.at[rho_idx, zeta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    ######################
    ####-----Q_ρζ-----####
    ######################
    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            D_theta.T
            @ ((iota * psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
            + D_zeta.T @ ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T))
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        -1
        * (
            ((iota * psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T)).T
            @ D_theta
            + ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * psi_r2.T)).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            D_theta.T @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            + D_zeta.T @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
        )
    )

    ##########################
    #######-----Q_ϑζ-----#####
    ##########################
    A = A.at[theta_idx, theta_idx].add(
        -1
        * (
            D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
        )
    )

    A = A.at[zeta_idx, zeta_idx].add(
        -1
        * (
            D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
            + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -1
        * (
            (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
            - (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        1
        * (
            (D_rho * psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
            - (D_rho * iota_psi_r2.T).T @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
        )
    )

    A = A.at[theta_idx, zeta_idx].add(
        D_zeta.T @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
        + ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta).T @ D_zeta
    )

    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            (D_rho * iota_psi_r2.T).T
            @ ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * psi_r2.T))
        )
    )
    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        1
        * (
            ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * psi_r2.T)).T
            @ (D_rho * iota_psi_r2.T)
        )
    )

    # Mixed Q-J term ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐
    # \xi^{\rho} (\mathbf{J} \times \nabla\rho)/|\nabla \rho|^2 \cdot \mathbf{Q}
    A = A.at[rho_idx, rho_idx].add(
        -1
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
        -1
        * (
            (
                (
                    W
                    * psi_r3
                    * sqrtg
                    * (j_sup_theta * g_sup_rp + j_sup_zeta * g_sup_rv)
                    / g_sup_rr
                )
                * (iota * D_theta + D_zeta)
            ).T
            + ((W * sqrtg * psi_r * j_sup_zeta) * (D_rho * iota_psi_r2.T)).T
            + ((W * sqrtg * psi_r * j_sup_theta) * (D_rho * psi_r2.T)).T
        )
    )

    A = A.at[rho_idx, theta_idx].add(
        -(W * psi_r2 * sqrtg * j_sup_theta) * D_theta
        + (W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
    )
    A = A.at[rho_idx, zeta_idx].add(
        +(W * psi_r2 * sqrtg * j_sup_theta) * D_theta
        - (W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
    )

    ## diagonal |J|^2 term
    A = A.at[rho_idx, rho_idx].add(jnp.diag((psi_r2 * W * sqrtg * J2).flatten()))

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(jnp.diag(n0 * (W * psi_r2 * sqrtg * g_rr).flatten()))
    B = B.at[theta_idx, theta_idx].add(jnp.diag(n0 * (W * sqrtg * g_vv).flatten()))
    B = B.at[zeta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota**2 * sqrtg * g_pp).flatten())
    )

    B = B.at[rho_idx, theta_idx].add(
        jnp.diag(n0 * (W * psi_r * sqrtg * g_rv).flatten())
    )
    B = B.at[rho_idx, zeta_idx].add(
        jnp.diag(n0 * (W * psi_r * iota * sqrtg * g_rp).flatten())
    )
    B = B.at[theta_idx, zeta_idx].add(
        jnp.diag(n0 * (W * iota * sqrtg * g_vp).flatten())
    )

    # The matrix must be Hermitian so we fill out the lower blocks
    if axisym:
        A = A.at[theta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, theta_idx]).T)
        A = A.at[zeta_idx, rho_idx].set(jnp.conjugate(A[rho_idx, zeta_idx]).T)
        A = A.at[zeta_idx, theta_idx].set(jnp.conjugate(A[theta_idx, zeta_idx]).T)

        B = B.at[theta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, theta_idx]).T)
        B = B.at[zeta_idx, rho_idx].set(jnp.conjugate(B[rho_idx, zeta_idx]).T)
        B = B.at[zeta_idx, theta_idx].set(jnp.conjugate(B[theta_idx, zeta_idx]).T)
    else:
        A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)
        A = A.at[zeta_idx, rho_idx].set(A[rho_idx, zeta_idx].T)
        A = A.at[zeta_idx, theta_idx].set(A[theta_idx, zeta_idx].T)

        B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)
        B = B.at[zeta_idx, rho_idx].set(B[rho_idx, zeta_idx].T)
        B = B.at[zeta_idx, theta_idx].set(B[theta_idx, zeta_idx].T)

    ## Shift the diagonal of A to ensure positive definiteness
    ## The estimate must be accurate. If A is diagonally dominant
    ## use Gerhsgorin theorem to estimate the lowest eigenvalue

    # A = A.at[jnp.diag_indices_from(A)].add(1e-11)
    # B = B.at[jnp.diag_indices_from(B)].add(1e-11)

    if compressible:
        ## Finally add the only instability drive term
        Au = jnp.zeros((3 * n_total, 3 * n_total))
        Au = Au.at[rho_idx, rho_idx].add(jnp.diag((W * psi_r2 * sqrtg * F).flatten()))

        A = A.at[all_idx, all_idx].add(Au)
    elif (compressible is False) and n_zeta_axisym != 0:
        # keeping compressibility but not solving for n = 0 axisym mode
        sqrtg_r = data["(sqrt(g)_PEST_r)|PEST"][:, None] * 1 / a_N**3
        sqrtg_v = data["(sqrt(g)_PEST_v)|PEST"][:, None] * 1 / a_N**3
        sqrtg_p = data["(sqrt(g)_PEST_p)|PEST"][:, None] * 1 / a_N**3

        C_zeta_inv = jnp.zeros((n_total, n_total))
        partial_z_log_sqrtg = jnp.reshape(
            (sqrtg_p / sqrtg).flatten(), (n_rho_max * n_theta_max, n_zeta_max)
        )
        partial_z_log_sqrtg = partial_z_log_sqrtg[..., None] * jnp.eye(n_zeta_max)

        C_zeta = partial_z_log_sqrtg + D_zeta0[None, ...]

        pdb.set_trace()
        partial_r_log_sqrtg = (sqrtg_r / sqrtg).flatten()
        C_rho = jnp.diag(partial_r_log_sqrtg) + D_rho  # (n_total, n_total)

        partial_v_log_sqrtg = (sqrtg_v / sqrtg).flatten()
        C_theta = jnp.diag(partial_v_log_sqrtg) + D_theta

        # batched inversion without actually forming C_zeta_inv
        C_rho_reshaped = C_rho.reshape(n_rho_max * n_theta_max, n_zeta_max, n_total)
        C_zeta_inv_C_rho_reshaped = jnp.linalg.solve(C_zeta, C_rho_reshaped)
        C_zeta_inv_C_rho = C_zeta_inv_C_rho_reshaped.reshape(n_total, n_total)

        C_theta_reshaped = C_theta.reshape(n_rho_max * n_theta_max, n_zeta_max, n_total)
        C_zeta_inv_C_theta_reshaped = jnp.linalg.solve(C_zeta, C_rho_reshaped)
        C_zeta_inv_C_theta = C_zeta_inv_C_theta_reshaped.reshape(n_total, n_total)

        ## Incompressibility gives us
        ## \xi^\zeta = -(C_\zeta^{-1} C\rho \xi^{\rho} + C_\zeta^{-1} C_\theta \xi^{\theta})
        ## ξ^ζ = −(C_ζ⁻¹ C_ρ ξ^ρ + C_ζ⁻¹ C_θ ξ^θ)

        # Impose incompressibility
        A = A.at[rho_idx, rho_idx].add(
            -1
            * (
                A[rho_idx, zeta_idx] @ C_zeta_inv_C_rho
                + (A[rho_idx, zeta_idx] @ C_zeta_inv_C_rho).T
            )
            + C_zeta_inv_C_rho.T @ A[zeta_idx, zeta_idx] @ C_zeta_inv_C_rho
        )

        A = A.at[rho_idx, theta_idx].add(
            -1
            * (
                A[rho_idx, zeta_idx] @ C_zeta_inv_C_theta
                + C_zeta_inv_C_rho @ A[rho_idx, theta_idx]
            )
            + C_zeta_inv_C_rho.T @ A[zeta_idx, zeta_idx] @ C_zeta_inv_C_theta
        )

        A = A.at[theta_idx, theta_idx].add(
            -1
            * (
                A[theta_idx, zeta_idx] @ C_zeta_inv_C_theta
                + (A[theta_idx, zeta_idx] @ C_zeta_inv_C_theta).T
            )
            + C_zeta_inv_C_theta.T @ A[zeta_idx, zeta_idx] @ C_zeta_inv_C_theta
        )

        # Fill out the lower part using symmetry
        A = A.at[theta_idx, rho_idx].set(A[rho_idx, theta_idx].T)

        B = B.at[rho_idx, rho_idx].add(
            -1
            * (
                B[rho_idx, zeta_idx] @ C_zeta_inv_C_rho
                + (B[rho_idx, zeta_idx] @ C_zeta_inv_C_rho).T
            )
            + C_zeta_inv_C_rho.T @ B[zeta_idx, zeta_idx] @ C_zeta_inv_C_rho
        )

        B = B.at[rho_idx, theta_idx].add(
            -1
            * (
                B[rho_idx, zeta_idx] @ C_zeta_inv_C_theta
                + C_zeta_inv_C_rho @ B[rho_idx, theta_idx]
            )
            + C_zeta_inv_C_rho.T @ B[zeta_idx, zeta_idx] @ C_zeta_inv_C_theta
        )

        B = B.at[theta_idx, theta_idx].add(
            -1
            * (
                B[theta_idx, zeta_idx] @ C_zeta_inv_C_theta
                + (B[theta_idx, zeta_idx] @ C_zeta_inv_C_theta).T
            )
            + C_zeta_inv_C_theta.T @ B[zeta_idx, zeta_idx] @ C_zeta_inv_C_theta
        )

        # Fill out the lower part using symmetry
        B = B.at[theta_idx, rho_idx].set(B[rho_idx, theta_idx].T)

    else:  # n_zeta_axisym is 0

        # keeping compressibility but not solving for n = 0 axisym mode
        sqrtg_r = data["(sqrt(g)_PEST_r)|PEST"][:, None] * 1 / a_N**3
        sqrtg_v = data["(sqrt(g)_PEST_v)|PEST"][:, None] * 1 / a_N**3

        partial_r_log_sqrtg = (sqrtg_r / sqrtg).flatten()
        C_rho = jnp.diag(partial_r_log_sqrtg) + D_rho  # (n_total, n_total)

        partial_v_log_sqrtg = (sqrtg_v / sqrtg).flatten()
        C_theta = jnp.diag(partial_v_log_sqrtg) + D_theta

        # batched inversion without actually forming C_theta_inv
        C_rho_reshaped = C_rho.reshape(n_rho_max * n_theta_max, n_zeta_max, n_total)
        C_theta_inv_C_rho = jnp.linalg.solve(C_theta, C_rho_reshaped)

        # Impose incompressibility
        A = A.at[rho_idx, rho_idx].add(
            -1
            * (
                A[rho_idx, theta_idx] @ C_theta_inv_C_rho
                + (A[rho_idx, theta_idx] @ C_theta_inv_C_rho).T
            )
            + C_theta_inv_C_rho.T @ A[theta_idx, theta_idx] @ C_theta_inv_C_rho
        )

        A = B.at[rho_idx, rho_idx].add(
            -1
            * (
                B[rho_idx, theta_idx] @ C_theta_inv_C_rho
                + (B[rho_idx, theta_idx] @ C_theta_inv_C_rho).T
            )
            + C_theta_inv_C_rho.T @ B[theta_idx, theta_idx] @ C_theta_inv_C_rho
        )

        # apply dirichlet BC to ξ^ρ
        keep = jnp.arange(n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max)

        w2, _ = jnp.linalg.eigh((A + A.T) / 2)
        w3, _ = eigh(A[jnp.ix_(keep, keep)])

        # L = jnp.linalg.cholesky(B2[jnp.ix_(keep, keep)])

        ### Finally add the only instability drive term
        # Au = jnp.zeros((3 * n_total, 3 * n_total))
        # Au = Au.at[rho_idx, rho_idx].add(jnp.diag((W * psi_r2 * sqrtg * F).flatten()))

        # A = A.at[rho_idx, rho_idx_idx].add(Au)

    # Improve the condition number of the mass matrix
    ## D = 1.0 /jnp.tile((W * sqrtg).flatten(), 3)[:, None]
    D = jnp.diag(1 / jnp.sqrt(jnp.diag(B)))

    A2 = D @ (A @ D.T)
    B2 = D @ (B @ D.T)

    ##w2, _ = jnp.linalg.eigh(A)
    ##print(w2)
    # w4, _ = jnp.linalg.eigh((B2 + B2.T)/2)
    # print(w4)

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    w2, _ = jnp.linalg.eigh((A2 + A2.T) / 2)
    print(w2)

    # tic = time.time()
    # w, v = jax.scipy.linalg.eigh(A3, B2[jnp.ix_(keep, keep)])
    # toc = time.time()
    # print(toc-tic)

    ## This will be the most expensive but easiest automatically differentiable way.
    ## TODO: Multiply B with a permutation matrix P so that it becomes block diagonal
    ## then Cholesky factorize each block. That will make cholesky ~ 3**3 x faster
    L = jnp.linalg.cholesky(B2[jnp.ix_(keep, keep)])
    # Linv = jnp.linalg.inv(L)
    # Right-multiply by L^{-T}:  ALt = A L^{-T}
    ALt = solve_triangular(L, A2.T, lower=True).T
    # Left-multiply by L^{-1}:   C = L^{-1} (A L^{-T})
    A3 = solve_triangular(L, ALt, lower=True)

    ## w, v = jnp.linalg.eigh(Linv @ A[jnp.ix_(keep, keep)] @ Linv.T)
    tic = time.time()
    w, v = jnp.linalg.eigh(A3)

    ##A3 = np.asarray(A3)
    ##tic = time.time()
    ##w, v = eigsh(A3, k=10, which="SA")
    toc = time.time()
    print(toc - tic)

    data["finite-n lambda5"] = w
    data["finite-n eigenfunction5"] = v

    return data
