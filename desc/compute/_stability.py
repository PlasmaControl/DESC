"""Compute functions for stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

import os
import time
from functools import partial

import numpy as np

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


def _solver_opt(kwargs, name, env, default, cast=None):
    """Resolve a solver option: KWARG FIRST, then environment, then default.

    The keyword argument wins. This inverts what several of these reads used to
    do -- ``os.environ.get("AGNI_NUM_MATVECS", str(kwargs.get("num_matvecs", 50)))``
    used the kwarg only as the environment's *default*, so an exported variable
    silently discarded an explicit argument. A caller that passes a value must
    get that value.

    The environment is kept as a fallback so existing job scripts keep working,
    but it can no longer override an argument. Anything resolved here is a
    NUMERICAL choice that changes the answer, so it belongs in the call, not in
    the shell.
    """
    val = kwargs.get(name, None)
    if val is None:
        val = os.environ.get(env, default)
    return cast(val) if cast is not None else val


def _solver_flag(kwargs, name, env, default="0"):
    """Boolean solver option, kwarg first. Accepts bools or the usual strings."""
    val = kwargs.get(name, None)
    if val is None:
        val = os.environ.get(env, default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() not in ("0", "false", "no", "off", "")


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
        # `w_zeta` is the 1D weight vector regardless of whether the caller
        # passed a diagonal matrix (every `*_diffmat` builder) or a vector
        # (`zernike_nodes_weights`). Any off-diagonal content is dropped, which
        # is exact for every quadrature used here.
        w = diffmat.w_zeta

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
    psi_r2 = psi_r**2
    psi_r3 = psi_r**3

    iota_psi_r2 = iota * psi_r2

    # Add a tiny shift because sometimes the pressure can be
    # slightly negative in the edge
    p0 = 1.0 * mu_0 * data["p"][:, None] / B_N**2 + 1e-12
    p_r = 1.0 * mu_0 * data["p_r"][:, None] / B_N**2

    axisym = kwargs.get("axisym", False)

    # Large gamma is an alternate way to impose incompressibility
    gamma = kwargs.get("gamma", 10.0)

    # For axisymmetric equilibria n_mode_axisym will decide the toroidal
    # mode number to analyze.
    n_mode_axisym = kwargs.get("n_mode_axisym", 1)
    incompressible = kwargs.get("incompressible", False)

    def _cT(x):
        return jnp.conjugate(jnp.transpose(x))

    # (A kron I_m)/(I_na kron B) sliced by column/row, without materializing
    # the full Kronecker product. Verified against dense kron+slice, 0 error.
    def _kron_I_right_slice_cols(A, m, idx):
        a_idx, i_idx = jnp.divmod(idx, m)
        cols = A[:, a_idx]
        na, k = A.shape[0], idx.size
        out = jnp.zeros((na, m, k), dtype=cols.dtype)
        out = out.at[
            jnp.arange(na)[:, None], i_idx[None, :], jnp.arange(k)[None, :]
        ].set(cols)
        return out.reshape(na * m, k)

    def _kron_I_left_slice_cols(B, na, idx):
        a_idx, j_idx = jnp.divmod(idx, B.shape[1])
        cols = B[:, j_idx]
        nb, k = B.shape[0], idx.size
        out = jnp.zeros((na, nb, k), dtype=cols.dtype)
        out = out.at[
            a_idx[None, :], jnp.arange(nb)[:, None], jnp.arange(k)[None, :]
        ].set(cols)
        return out.reshape(na * nb, k)

    def _kron_I_right_slice_rows(A, m, idx):  # (A kron I_m)^T = A^T kron I_m
        return _kron_I_right_slice_cols(A.T, m, idx).T

    def _kron_I_left_slice_rows(B, na, idx):  # (I_na kron B)^T = I_na kron B^T
        return _kron_I_left_slice_cols(B.T, na, idx).T

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

    W_rho = transforms["diffmat"].w_rho
    W_theta = transforms["diffmat"].w_theta
    W_zeta = transforms["diffmat"].w_zeta

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

    # Avoid materializing the full (n_total, n_total) kron just to slice it
    # down to one ring a few lines later (was OOMing at 3D scale). Every ring
    # fixes one zeta index, so this is exact and reduces to the untouched
    # original path whenever n_zeta_max==1 (axisym).
    _ring_precoupled = kwargs.get("ring_nodes", None)
    if coupled_rt and _ring_precoupled is not None and n_zeta_max > 1:
        _ridx = jnp.asarray(_ring_precoupled)
        _n_rt = n_rho_max * n_theta_max
        D_rho = jax.lax.stop_gradient(
            _kron_I_right_slice_cols(D_rho0, n_zeta_max, _ridx)
        )
        D_theta = jax.lax.stop_gradient(
            _kron_I_right_slice_cols(D_theta0, n_zeta_max, _ridx)
        )
        D_zeta = jax.lax.stop_gradient(_kron_I_left_slice_cols(D_zeta0, _n_rt, _ridx))
        D_thetaT = jax.lax.stop_gradient(
            _kron_I_right_slice_rows(_cT(D_theta0), n_zeta_max, _ridx)
        )
        D_zetaT = jax.lax.stop_gradient(
            _kron_I_left_slice_rows(_cT(D_zeta0), _n_rt, _ridx)
        )
        _coupled_rt_presliced = True
    elif coupled_rt:
        # D_rho0/D_theta0 already couple (rho, theta); only tensor with zeta.
        I_rt0 = jax.lax.stop_gradient(jnp.eye(n_rho_max * n_theta_max))
        D_rho = jax.lax.stop_gradient(jnp.kron(D_rho0, I_zeta0))
        D_theta = jax.lax.stop_gradient(jnp.kron(D_theta0, I_zeta0))
        D_zeta = jax.lax.stop_gradient(jnp.kron(I_rt0, D_zeta0))
        D_thetaT = jax.lax.stop_gradient(jnp.kron(_cT(D_theta0), I_zeta0))
        D_zetaT = jax.lax.stop_gradient(jnp.kron(I_rt0, _cT(D_zeta0)))
        _coupled_rt_presliced = False
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
        _coupled_rt_presliced = False

    # Quadrature weights still factorize (tensor-product) in both modes.
    W = jnp.kron(W_rho, jnp.kron(W_theta, W_zeta))[:, None]
    n_total = n_rho_max * n_theta_max * n_zeta_max

    # ------------------------------------------------------------------
    # RING RESTRICTION. With `ring_nodes=None` every helper below is the
    # IDENTITY, so the full-matrix path is bit-for-bit what it was. With a
    # ring supplied, the same code assembles only that ring's block: the
    # derivative matrices are sliced to the ring's columns (rows for the
    # transposes), the accumulators shrink to 3*|R|, and node-diagonal
    # quantities are restricted to the ring's nodes.
    #
    _Rnode = kwargs.get("ring_nodes", None)
    if _Rnode is None:
        _nR = n_total

        def _selc(M):
            return M

        def _selr(M):
            return M

        def _nodesel(v):
            return v

        def _diag_r(v):
            return jnp.diag(v)

        def _diag_col(v):
            return jnp.diag(jnp.asarray(v).reshape(-1))

        def _fit(M):
            return M

    else:
        _Rnode = jnp.asarray(_Rnode)
        _nR = int(_Rnode.size)
        _ar = jnp.arange(_nR)

        def _selc(M):
            return M[:, _Rnode]

        def _selr(M):
            return M[_Rnode, :]

        def _nodesel(v):
            v = jnp.asarray(v)
            return v[_Rnode] if v.shape[0] == n_total else v

        def _diag_r(v):
            # 2-D in -> extraction from an already ring-local matrix.
            v = jnp.asarray(v)
            if v.ndim == 2:
                return jnp.diag(v)
            v = v.reshape(-1)
            return jnp.diag(v[_Rnode]) if v.size == n_total else jnp.diag(v)

        def _diag_col(v):
            # A node-diagonal OPERATOR that gets added to a D matrix must be
            # column-sliced, not reduced to a ring block: diag(v)[:, R].
            v = jnp.asarray(v).reshape(-1)
            return (
                jnp.zeros((n_total, _nR), dtype=v.dtype).at[_Rnode, _ar].set(v[_Rnode])
            )

        def _fit(M):
            # A term with a derivative on only one side keeps the full node
            # dimension on the underived side, because the implicit identity
            # there was never sliced. Restrict any surviving full-length axis.
            M = jnp.asarray(M)
            if M.ndim == 2 and M.shape[0] == n_total:
                M = M[_Rnode, :]
            if M.ndim == 2 and M.shape[1] == n_total:
                M = M[:, _Rnode]
            return M

    if not _coupled_rt_presliced:
        D_rho = _selc(D_rho)
        D_theta = _selc(D_theta)
        D_zeta = _selc(D_zeta)
        D_thetaT = _selr(D_thetaT)
        D_zetaT = _selr(D_zetaT)
    # else: already built directly in sliced form above -- _selc/_selr would
    # double-slice with the wrong shape.

    # Arbitrary choice. Mostly used to decide the range of eigenvalues of
    # the mass matrix. Pre-conditioning should remove this factor
    n0 = jnp.asarray(kwargs.get("density", jnp.ones(n_total))).reshape(n_total, 1)

    # Define block indices
    rho_idx = slice(0, _nR)
    ups_idx = slice(_nR, 2 * _nR)
    zeta_idx = slice(2 * _nR, 3 * _nR)

    ## Create the full matrix
    if axisym:
        A = jnp.zeros((3 * _nR, 3 * _nR), dtype=jnp.complex128)
        B = jnp.zeros((3 * _nR, 3 * _nR), dtype=jnp.complex128)
    else:
        A = jnp.zeros((3 * _nR, 3 * _nR), dtype=jnp.float64)
        B = jnp.zeros((3 * _nR, 3 * _nR), dtype=jnp.float64)

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

    F = -mu_0 * data["finite-n instability drive"][:, None] * (1 / B_N) ** 2

    C_zeta = _diag_col(partial_z_log_sqrtg) + D_zeta
    C_rho = _diag_col(partial_r_log_sqrtg) + D_rho  # (n_total, n_total)
    C_theta = _diag_col(partial_v_log_sqrtg) + D_theta

    ####################
    ####----Q²_ρρ----###
    ####################
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            D_thetaT @ ((psi_r_over_sqrtg * iota**2 * psi_r3 * W * g_rr) * D_theta)
            + D_zetaT @ ((psi_r_over_sqrtg * W * psi_r3 * g_rr) * D_zeta)
            + D_thetaT @ ((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta)
            + _cT((psi_r_over_sqrtg * iota * psi_r3 * W * g_rr) * D_zeta) @ D_theta
        )
    )

    ####################
    ####----Q²_ϑϑ ---###
    ####################
    # enforcing symmetry exactly
    A = A.at[ups_idx, ups_idx].add(
        _fit(
            0.5
            * (
                D_zetaT @ ((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta)
                + _cT((psi_r_over_sqrtg * psi_r * W * g_vv) * D_zeta) @ D_zeta
            )
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        _fit(
            +_cT(D_rho * _nodesel(iota_psi_r2).T)
            @ (
                (psi_r_over_sqrtg * W * g_vv / psi_r)
                * (D_rho * _nodesel(iota_psi_r2).T)
            )
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        _fit(
            -1
            * _cT(D_rho * _nodesel(iota_psi_r2).T)
            @ ((psi_r_over_sqrtg * W * g_vv) * D_zeta)
        )
    )

    ####################
    ####----Q²_ζζ---####
    ####################
    A = A.at[ups_idx, ups_idx].add(
        _fit(
            0.5
            * (
                _cT(D_theta) @ ((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta)
                + _cT((psi_r_over_sqrtg * psi_r * W * g_pp) * D_theta) @ D_theta
            )
        )
    )
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            +_cT(D_rho * _nodesel(psi_r2).T)
            @ ((psi_r_over_sqrtg * W * g_pp / psi_r) * (D_rho * _nodesel(psi_r2).T))
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        _fit(
            1
            * _cT(D_rho * _nodesel(psi_r2).T)
            @ ((psi_r_over_sqrtg * W * g_pp) * D_theta)
        )
    )

    ####################
    ####----Q²_ρϑ----###
    ####################
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            -1
            * (
                _cT(D_theta)
                @ (
                    (iota * psi_r * psi_r_over_sqrtg * W * g_rv)
                    * (D_rho * _nodesel(iota_psi_r2).T)
                )
                + _cT(D_zeta)
                @ (
                    (psi_r * psi_r_over_sqrtg * W * g_rv)
                    * (D_rho * _nodesel(iota_psi_r2).T)
                )
            )
        )
    )

    ## transposed part of the mixed term along the ρ-ρ block diagonal
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            -1
            * (
                _cT(
                    (iota * psi_r * psi_r_over_sqrtg * W * g_rv)
                    * (D_rho * _nodesel(iota_psi_r2).T)
                )
                @ D_theta
                + _cT(
                    (psi_r * psi_r_over_sqrtg * W * g_rv)
                    * (D_rho * _nodesel(iota_psi_r2).T)
                )
                @ D_zeta
            )
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        _fit(
            _cT(D_theta) @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
            + _cT(D_zeta) @ ((psi_r2 * psi_r_over_sqrtg * W * g_rv) * D_zeta)
        )
    )

    ######################
    ####-----Q²_ρζ-----###
    ######################
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            -1
            * (
                _cT(D_theta)
                @ (
                    (iota * psi_r * psi_r_over_sqrtg * W * g_rp)
                    * (D_rho * _nodesel(psi_r2).T)
                )
                + _cT(D_zeta)
                @ ((psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * _nodesel(psi_r2).T))
            )
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        _fit(
            -1
            * (
                _cT(
                    (iota * psi_r * psi_r_over_sqrtg * W * g_rp)
                    * (D_rho * _nodesel(psi_r2).T)
                )
                @ D_theta
                + _cT(
                    (psi_r * psi_r_over_sqrtg * W * g_rp) * (D_rho * _nodesel(psi_r2).T)
                )
                @ D_zeta
            )
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        _fit(
            -1
            * (
                _cT(D_theta) @ ((iota * psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
                + _cT(D_zeta) @ ((psi_r2 * psi_r_over_sqrtg * W * g_rp) * D_theta)
            )
        )
    )

    ##########################
    ######-----Q²_ϑζ-----#####
    ##########################
    A = A.at[ups_idx, ups_idx].add(
        _fit(
            -1
            * (
                _cT(D_zeta) @ ((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta)
                + _cT((psi_r_over_sqrtg * W * psi_r * g_vp) * D_theta) @ D_zeta
            )
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        _fit(
            -1
            * (
                _cT(D_rho * _nodesel(psi_r2).T)
                @ ((psi_r_over_sqrtg * W * g_vp) * D_zeta)
                - _cT(D_rho * _nodesel(iota_psi_r2).T)
                @ ((psi_r_over_sqrtg * W * g_vp) * D_theta)
            )
        )
    )

    A = A.at[rho_idx, rho_idx].add(
        _fit(
            1
            * (
                _cT(D_rho * _nodesel(iota_psi_r2).T)
                @ ((psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * _nodesel(psi_r2).T))
            )
        )
    )
    # ρ-ρ symmetrizing term
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            1
            * (
                _cT(
                    (psi_r_over_sqrtg * W * g_vp / psi_r) * (D_rho * _nodesel(psi_r2).T)
                )
                @ (D_rho * _nodesel(iota_psi_r2).T)
            )
        )
    )

    # Mixed Q-J term ξ^ρ (𝐉 × ∇ρ)/|∇ ρ|² ⋅ 𝐐
    # \xi^{\rho} (\mathbf{J} \times \nabla\rho)/|\nabla \rho|^2 \cdot \mathbf{Q}
    # Some algebra is performed to replace g_sup_rv and g_sup_rp
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            1.0
            * (
                (
                    W
                    * psi_r2
                    * (j_sup_theta * g_sup_rp_term - j_sup_zeta * g_sup_rv_term)
                    / g_sup_rr
                )
                * (iota * D_theta + D_zeta)
                - (W * sqrtg * psi_r * j_sup_zeta) * (D_rho * _nodesel(iota_psi_r2).T)
                + (W * sqrtg * psi_r * j_sup_theta) * (D_rho * _nodesel(psi_r2).T)
            )
        )
    )

    # ρ-ρ block transposed for symmetry
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            1.0
            * (
                _cT(
                    (
                        W
                        * psi_r2
                        * (j_sup_theta * g_sup_rp_term - j_sup_zeta * g_sup_rv_term)
                        / g_sup_rr
                    )
                    * (iota * D_theta + D_zeta)
                )
                - _cT(
                    (W * sqrtg * psi_r * j_sup_zeta) * (D_rho * _nodesel(iota_psi_r2).T)
                )
                + _cT((W * sqrtg * psi_r * j_sup_theta) * (D_rho * _nodesel(psi_r2).T))
            )
        )
    )

    A = A.at[rho_idx, ups_idx].add(
        _fit(
            (W * psi_r2 * sqrtg * j_sup_theta) * D_theta
            + (W * psi_r2 * sqrtg * j_sup_zeta) * D_zeta
        )
    )

    ## diagonal |J|² term
    A = A.at[rho_idx, rho_idx].add(
        _fit(_diag_r((psi_r2 * W * sqrtg * J2 / g_sup_rr).flatten()))
    )

    # Mass matrix (must be symmetric positive definite)
    B = B.at[rho_idx, rho_idx].add(
        _fit(_diag_r((n0 * W * psi_r2 * sqrtg * g_rr).flatten()))
    )
    B = B.at[ups_idx, ups_idx].add(_fit(_diag_r((n0 * W * sqrtg * g_vv).flatten())))

    B = B.at[rho_idx, ups_idx].add(
        _fit(_diag_r((n0 * W * psi_r * sqrtg * g_rv).flatten()))
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
        n0 * W * psi_r * sqrtg * g_rp,
        n0 * W * psi_r * sqrtg * (g_rv + iotainv * g_rp),
    )
    uz = jnp.where(
        ismirror,
        n0 * W * psi_r * sqrtg * g_vp,
        n0 * W * sqrtg * (g_vv + iotainv * g_vp),
    )
    B = B.at[zeta_idx, zeta_idx].add(_fit(_diag_r(zz.flatten())))
    B = B.at[rho_idx, zeta_idx].add(_fit(_diag_r(rz.flatten())))
    B = B.at[ups_idx, zeta_idx].add(_fit(_diag_r(uz.flatten())))

    ##A = np.where(np.abs(A) >= 1e-11, 1.0, 0.0)
    # from matplotlib import pyplot as plt
    # plt.spy(A, precision=1e-11)
    # plt.savefig("test.png", dpi=400)

    # purely stabilizing and doesn't change the marginal stability
    A = A.at[rho_idx, rho_idx].add(
        _fit(
            _cT(C_rho * _nodesel(psi_r).T)
            @ ((gamma * sqrtg * W * p0) * (C_rho * _nodesel(psi_r).T))
        )
    )
    A = A.at[ups_idx, ups_idx].add(
        _fit(_cT(C_theta) @ ((gamma * sqrtg * W * p0) * C_theta))
    )
    A = A.at[rho_idx, ups_idx].add(
        _fit(_cT(C_rho * _nodesel(psi_r).T) @ ((gamma * sqrtg * W * p0) * C_theta))
    )

    A = A.at[zeta_idx, zeta_idx].add(
        _fit(
            _cT(C_theta + C_zeta * _nodesel(iotainv).T)
            @ ((gamma * sqrtg * W * p0) * (C_theta + C_zeta * _nodesel(iotainv).T))
        )
    )
    A = A.at[rho_idx, zeta_idx].add(
        _fit(
            _cT(C_rho * _nodesel(psi_r).T)
            @ ((gamma * sqrtg * W * p0) * (C_theta + C_zeta * _nodesel(iotainv).T))
        )
    )
    A = A.at[ups_idx, zeta_idx].add(
        _fit(
            _cT(C_theta)
            @ ((gamma * sqrtg * W * p0) * (C_theta + C_zeta * _nodesel(iotainv).T))
        )
    )

    #### Instability drive term
    # Au = jnp.zeros((3 * n_total, 3 * n_total))
    # Au = Au.at[rho_idx, rho_idx].add(_diag_r((W * psi_r2 * sqrtg * F).flatten()))
    au_diag = _nodesel((W * psi_r2 * sqrtg * F).flatten())

    rt_size = n_rho_max * n_theta_max
    zernike_penalty_alpha, Q_rt, penalty_rank = _get_zernike_penalty(
        transforms, rt_size
    )
    if coupled_rt and zernike_penalty_alpha > 0.0:
        # Q_rt is the DiffMat's penalty projector, a real data leaf of the
        # pytree and therefore a tracer under jit -- `np.kron` on it raises
        # .
        #
        # But `jnp.kron` UNCONDITIONALLY was a memory regression: it moves the
        # (rt_size*n_zeta)^2 intermediate onto the DEVICE, where np.kron kept it
        # on the host for free. At 32x32x12 that is 1.21 GB for Q and another
        # 1.21 GB for `penalty`, and this assembly already peaks at 69.18 GB of
        # a 73.70 GB limit -- 4.5 GB of headroom. Job 56839366,
        # a case that had COMPLETED as, then OOMed on the 10.42 GB
        # gather below.
        #
        # So: device only when the input is actually traced.
        if _coupled_rt_presliced:
            # Ring fixes one zeta index for both row and column, so the
            # double-sliced kron(Q_rt, I_zeta) reduces to a direct submatrix
            # of Q_rt -- no kron, no (n_total, n_total) intermediate.
            _a_idx = _ridx // n_zeta_max
            Q = Q_rt[_a_idx][:, _a_idx]
        elif isinstance(Q_rt, jax.core.Tracer) or isinstance(A, jax.core.Tracer):
            Q = Q_rt if n_zeta_max == 1 else jnp.kron(Q_rt, jnp.eye(n_zeta_max))
        else:
            Q = (
                np.asarray(Q_rt)
                if n_zeta_max == 1
                else np.kron(np.asarray(Q_rt), np.eye(n_zeta_max))
            )
        penalty = jnp.asarray(zernike_penalty_alpha * Q, dtype=A.dtype)
        A = A.at[rho_idx, rho_idx].add(_fit(penalty))
        A = A.at[ups_idx, ups_idx].add(_fit(penalty))
        A = A.at[zeta_idx, zeta_idx].add(_fit(penalty))
        rank_msg = "unknown" if penalty_rank is None else str(penalty_rank)
        # penalized_msg = (
        #    "unknown" if penalty_rank is None else str(rt_size - penalty_rank)
        # )
        # print(
        #    "[finite-n lambda3:coupled penalty]",
        #    f"alpha={zernike_penalty_alpha:.3e}",
        #    f"rank={rank_msg}/{rt_size}",
        #    f"penalized_rt={penalized_msg}",
        #    flush=True,
        # )

    A = A.at[ups_idx, rho_idx].set(_cT(A[rho_idx, ups_idx]))
    A = A.at[zeta_idx, rho_idx].set(_cT(A[rho_idx, zeta_idx]))
    A = A.at[zeta_idx, ups_idx].set(_cT(A[ups_idx, zeta_idx]))

    B = B.at[ups_idx, rho_idx].set(_cT(B[rho_idx, ups_idx]))
    B = B.at[zeta_idx, rho_idx].set(_cT(B[rho_idx, zeta_idx]))
    B = B.at[zeta_idx, ups_idx].set(_cT(B[ups_idx, zeta_idx]))

    d = 1 / jnp.sqrt(_diag_r(B))  # 1D array

    # MEMORY (Step 1): the A whitening is DEFERRED to after B_blocks is extracted and B
    # is released -- see the optimization_barrier below. Doing it here holds
    # A_old + A_new + a broadcast transient WHILE B_old + B_new are also live: ~5 full
    # (3*n_total)^2 copies. Measured peak with it here 28.9 GB @24 / 51.5 GB @32;
    # deferred, 19.97 GB @24 / 34.51 GB @32.
    au_diag = d[rho_idx] ** 2 * au_diag
    B = d[:, None] * B * d[None, :]

    # TODO: B_blocks will always be real for axisym=True, complex data type
    # is used to avoid trivial dtype-related errors. Fix later!
    if axisym:
        B_blocks = jnp.zeros((_nR, 3, 3), dtype=jnp.complex128)
        I3 = jnp.tile(jnp.eye(3, dtype=jnp.complex128), (_nR, 1, 1))
    else:
        B_blocks = jnp.zeros((_nR, 3, 3))
        I3 = jnp.tile(jnp.eye(3), (_nR, 1, 1))

    B_blocks = B_blocks.at[:, 0, 0].set(_diag_r(B[rho_idx, rho_idx]))
    B_blocks = B_blocks.at[:, 1, 1].set(_diag_r(B[ups_idx, ups_idx]))
    B_blocks = B_blocks.at[:, 2, 2].set(_diag_r(B[zeta_idx, zeta_idx]))

    B_blocks = B_blocks.at[:, 0, 1].set(_diag_r(B[rho_idx, ups_idx]))
    B_blocks = B_blocks.at[:, 1, 0].set(_diag_r(B[ups_idx, rho_idx]))

    B_blocks = B_blocks.at[:, 2, 0].set(_diag_r(B[rho_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 0, 2].set(_diag_r(B[zeta_idx, rho_idx]))

    B_blocks = B_blocks.at[:, 1, 2].set(_diag_r(B[ups_idx, zeta_idx]))
    B_blocks = B_blocks.at[:, 2, 1].set(_diag_r(B[zeta_idx, ups_idx]))

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
    # jnp, not np: on the ring path `_nodesel` indexes this with `ring_nodes`,
    # which is a TRACER under the vmap that builds all rings at once. numpy
    # cannot be indexed by a tracer.
    boundary = _nodesel(jnp.asarray((rho_shell == 0) | (rho_shell == (n_rho_max - 1))))

    # In lambda3 the local basis is (rho, upsilon, zeta)
    # so remove rho-upsilon and rho-zeta couplings on the boundary.
    #
    # Written as `jnp.where` over the full leading axis rather than
    # `.at[boundary, i, j]`: a BOOLEAN-MASK index needs a concrete mask, and
    # under the ring vmap `boundary` is traced. The two forms are identical
    # elementwise.
    for _i, _j in ((0, 1), (1, 0), (0, 2), (2, 0)):
        B_blocks = B_blocks.at[:, _i, _j].set(
            jnp.where(boundary, 0.0, B_blocks[:, _i, _j])
        )

    L = jnp.linalg.cholesky(B_blocks)  # (N,3,3)

    # Diagnostic: conditioning of the per-node 3x3 mass blocks. Near-singular
    # blocks make Linv ~ 1/sqrt(sigma_min) huge, which inflates lambda_max of
    # the whitened A and hence the eigh roundoff floor (eps*||A||). Cheap: N
    # 3x3 eigendecompositions.
    # Diagnostic only. np.asarray(B_blocks) + float()/int() concretize a TRACED array,
    # which raises under jit (jax_lanczos path). Skip when B_blocks is a tracer; the
    # dense/callback/eager paths still print it. Silent when AGNI_DIAG=0.
    # try/except is LOAD-BEARING, not defensive habit: this diagnostic CRASHED a
    # real optimization. np.linalg.eigvalsh raised
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

    if _Rnode is not None:
        # RING PATH ends here. Everything past this point -- the component/node
        # permutation, the whitening, the keep-mask reduction -- is global to the
        # full matrix and meaningless for a single ring block. The caller
        # (`ring_block`) finishes the block from these three pieces.
        return {"A": A, "Linv": Linv, "au_diag": au_diag}

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
    A = A.at[node_idx, :, node_idx, :].add(_fit(1e-14 * jnp.eye(3)))

    # Add transformed instability-drive contribution without materializing Au.
    L0 = Linv[:, :, 0]
    au_node = au_diag[:, None, None] * L0[:, :, None] * L0[:, None, :]
    A = A.at[node_idx, :, node_idx, :].add(_fit(au_node))

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
    v_guess="ndarray: eigenfunction guess to initialize the "
    + "iterative eigenvalue solver",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier (rho, theta) "
    "operators instead of separable 1D matrices",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
    sigma="float: shift for the shift-invert eigensolver (default -0.1)",
    full_spectrum="bool: if True, dense-eigendecompose the full reduced matrix "
    "with jnp.linalg.eigh and store every eigenvalue under "
    "'finite-n lambda3 spectrum'; the returned dominant eigenmode is unchanged. "
    "Default False (iterative eigsh for the single dominant mode).",
)
def _AGNI3(params, transforms, profiles, data, **kwargs):
    """AGNI dense finite-n lambda3: assemble the matrix, then ARPACK eigsh."""
    # noqa: unused dependency
    _ = params["Psi"]
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
    xi_full = jnp.concatenate(
        [
            d[rho_idx] * (Linv[:, 0, 0] * vr + Linv[:, 1, 0] * vv + Linv[:, 2, 0] * vz),
            d[ups_idx] * (Linv[:, 1, 1] * vv + Linv[:, 2, 1] * vz),
            d[zeta_idx] * (Linv[:, 2, 2] * vz),
        ]
    )

    # A is real unless `axisym=True` sets D_zeta0 = 1j * n_mode_axisym. A real
    # eigenvector has no phase to fix: the rotation collapsed to a global sign
    # set by sign(mean(xi_rho)), and that mean is ~1e-12 for an oscillatory
    # mode, so the sign was noise. 3D takes the components as they are.
    _shape = (n_rho_max, n_theta_max, n_zeta_max)
    if jnp.iscomplexobj(xi_full):
        # Align the mean phase to make the mode up-down symmetric. Cosmetic:
        # the physics is invariant under it.
        xi_ref = xi_full[rho_idx]
        rot = jnp.exp(1j * jnp.arctan2(jnp.mean(xi_ref.real), jnp.mean(xi_ref.imag)))
        xr = (xi_full[rho_idx].reshape(_shape) * rot).imag
        xv = (xi_full[ups_idx].reshape(_shape) * rot).imag
        xz = (xi_full[zeta_idx].reshape(_shape) * rot).imag
    else:
        xr = xi_full[rho_idx].reshape(_shape)
        xv = xi_full[ups_idx].reshape(_shape)
        xz = xi_full[zeta_idx].reshape(_shape)

    # precomputed forward derivatives (re-used below)
    xr_v = d_dv(D_theta0, xr)
    xr_z = d_dz(D_zeta0, xr)

    xv_v = d_dv(D_theta0, xv + xz)
    xv_z = d_dz(D_zeta0, xv + xz)

    xz_v = d_dv(D_theta0, xz / iota)
    xz_z = d_dz(D_zeta0, xz / iota)

    test_v = d_dv(D_theta0, xv)
    test_z = d_dz(D_zeta0, xv)

    # combos used many times
    xr_r = d_dr(D_rho0, xr)  # dρ(ι ψ′² xr)
    psi_rr = d_dr(D_rho0, psi_r)  # dρ(ι ψ′² xr)
    iota_r = d_dr(D_rho0, iota)  # dρ(ι ψ′² xr)

    deltaB_r = psi_r_over_sqrtg * psi_r * (iota * xr_v + xr_z)
    deltaB_v = psi_r_over_sqrtg * (
        1.0 * (test_z)
        - 1.0 * (xr_r * iota * psi_r + (2 * iota * psi_rr + iota_r * psi_r) * xr)
    )
    deltaB_z = -psi_r_over_sqrtg * (
        1.0 * (test_v) + 1.0 * (xr_r * psi_r + 2 * psi_rr * xr)
    )

    deltaV_r = psi_r * xr
    deltaV_v = xv + xz
    deltaV_z = xz * 1 / iota

    deltaB2 = (
        g_rr * deltaB_r**2
        + 1.0 * g_vv * deltaB_v**2
        + g_pp * deltaB_z**2
        + 2.0
        * (
            g_rv * deltaB_r * deltaB_v
            + g_rp * deltaB_r * deltaB_z
            + g_vp * deltaB_v * deltaB_z
        )
    )
    deltaV2 = (
        g_rr * deltaV_r**2
        + 1.0 * g_vv * deltaV_v**2
        + g_pp * deltaV_z**2
        + 2.0
        * (
            g_rv * deltaV_r * deltaV_v
            + g_rp * deltaV_r * deltaV_z
            + g_vp * deltaV_v * deltaV_z
        )
    )

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
    _ = params["Psi"]
    return data  # noqa: unused dependency


def _agni3_matfree_operator(params, transforms, profiles, data, **kwargs):
    """Build the reduced, whitened finite-n lambda3 operator ``Ax``.

    Single definition of the operator. ``finite-n lambda3 rayleigh`` builds its
    ``jax_lanczos`` and ``pcg_deflated`` operators from this, and the ring
    preconditioner's blocks are sub-blocks of the same matrix, so everything
    matrix-free in this module agrees by construction rather than by
    maintenance.

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
    w_rho = transforms["diffmat"].w_rho
    w_theta = transforms["diffmat"].w_theta
    w_zeta = transforms["diffmat"].w_zeta

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

    # Match _agni3_assemble's route to g^rv/g^rz exactly: build them from the PEST
    # lower metric via g¹² = (g₁₃g₂₃ - g₁₂g₃₃)/(√g)², rather than reading data["g^rv"].
    # These absorb a psi_r*sqrtg: g_sup_rv_term == psi_r * sqrtg * g^rv.
    g_sup_rv_term = psi_r_over_sqrtg * (g_rp * g_vp - g_rv * g_pp)
    g_sup_rp_term = psi_r_over_sqrtg * (g_rv * g_vp - g_rp * g_vv)

    J2 = _reshape((mu_0 * data["|J|"]) ** 2) * (a_N / B_N) ** 2
    j_sup_zeta = mu_0 * _reshape(data["J^zeta"]) * a_N**2 / B_N
    j_sup_theta = iota * j_sup_zeta + p_r / psi_r

    F = -mu_0 * _reshape(data["finite-n instability drive"]) / B_N**2

    if coupled_rt:
        # D is the full (n_rho*n_theta) coupled (rho, theta) operator: flatten the
        # (rho, theta) axes, apply, reshape back. d_dz stays separable in zeta.
        def _rt_apply(D, u):
            """Real (rho, theta) matrix D applied to u, rho-major, per zeta plane.

            One zeta plane uses a matrix-vector product, and a complex u is split into
            real and imaginary parts so D is never converted to complex. Inside
            Jacobi-Davidson's compiled loops, ``D @ U`` with U of shape (n, 1) and
            complex was lowered to convert/broadcast/multiply with an n x n complex
            temporary per product: one 77.55 GB allocation at DSHAPE 96x128.
            """
            if n_zeta == 1:
                U = u.reshape(-1)
            else:
                U = u.reshape(n_rho * n_theta, n_zeta)
            if jnp.iscomplexobj(U) and not jnp.iscomplexobj(D):
                DU = D @ U.real + 1j * (D @ U.imag)
            else:
                DU = D @ U
            return DU.reshape(n_rho, n_theta, n_zeta)

        def d_dr(D, u):
            """Radial derivative via the coupled (rho, theta) operator."""
            return _rt_apply(D, u)

        def d_dv(D, u):
            """Poloidal derivative via the coupled (rho, theta) operator."""
            return _rt_apply(D, u)

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

    # RG: need to make this consistent with the dense case. `_agni3_assemble`
    # auto-detects the mirror from the equilibrium (`ismirror = jnp.all(jnp.abs(iota)
    # < 1e-12)`, traced, jit-safe), while this path takes a manual `mirror` kwarg
    # TODO: test mirror geometry
    ismirror = bool(kwargs.get("mirror", False))
    if ismirror:
        B_blocks = B_blocks.at[:, 2, 2].set((n0 * W * sqrtg * g_pp).flatten())
        B_blocks = B_blocks.at[:, 0, 2].set((n0 * W * psi_r * sqrtg * g_rp).flatten())
        B_blocks = B_blocks.at[:, 2, 0].set((n0 * W * psi_r * sqrtg * g_rp).flatten())
        B_blocks = B_blocks.at[:, 1, 2].set((n0 * W * sqrtg * g_vp).flatten())
        B_blocks = B_blocks.at[:, 2, 1].set((n0 * W * sqrtg * g_vp).flatten())
    else:
        B_blocks = B_blocks.at[:, 2, 2].set(
            (
                n0 * W * sqrtg * (g_vv + 2.0 * iotainv * g_vp + iotainv**2 * g_pp)
            ).flatten()
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

        def _apply_penalty(u):
            # Q = kron(Q_rt, I_zeta) acting on rho-major (rho, theta, zeta) data.
            return _rt_apply(alphaQ_rt, u)

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
        Ar += -iota_psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_vv) * xu_z)
        Au += -d_dz(_cT(D_zeta0), (psi_r_over_sqrtg * W * g_vv) * xr1_r)

        # Q^2_zz -> upsilon block after variable change
        Au += d_dv(_cT(D_theta0), (psi_r_over_sqrtg * psi_r * W * g_pp) * xu_v)
        Ar += psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_pp / psi_r) * xr2_r)
        Ar += psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_pp) * xu_v)
        Au += d_dv(_cT(D_theta0), (psi_r_over_sqrtg * W * g_pp) * xr2_r)

        # Q^2_rv and transpose
        Ar += -(
            d_dv(_cT(D_theta0), (iota * psi_r * psi_r_over_sqrtg * W * g_rv) * xr1_r)
            + d_dz(_cT(D_zeta0), (psi_r * psi_r_over_sqrtg * W * g_rv) * xr1_r)
        )
        Ar += -(
            iota_psi_r2
            * d_dr(_cT(D_rho0), (iota * psi_r * psi_r_over_sqrtg * W * g_rv) * xr_v)
            + iota_psi_r2
            * d_dr(_cT(D_rho0), (psi_r * psi_r_over_sqrtg * W * g_rv) * xr_z)
        )
        Ar += d_dv(
            _cT(D_theta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * xu_z
        ) + d_dz(_cT(D_zeta0), (psi_r2 * psi_r_over_sqrtg * W * g_rv) * xu_z)
        Au += d_dz(
            _cT(D_zeta0), (iota * psi_r2 * psi_r_over_sqrtg * W * g_rv) * xr_v
        ) + d_dz(_cT(D_zeta0), (psi_r2 * psi_r_over_sqrtg * W * g_rv) * xr_z)

        # Q^2_rz and transpose
        Ar += -(
            d_dv(_cT(D_theta0), (iota * psi_r * psi_r_over_sqrtg * W * g_rp) * xr2_r)
            + d_dz(_cT(D_zeta0), (psi_r * psi_r_over_sqrtg * W * g_rp) * xr2_r)
        )
        Ar += -(
            psi_r2
            * d_dr(_cT(D_rho0), (iota * psi_r * psi_r_over_sqrtg * W * g_rp) * xr_v)
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
        Ar += psi_r2 * d_dr(_cT(D_rho0), (psi_r_over_sqrtg * W * g_vp / psi_r) * xr1_r)

        # RG: J cross Q terms, by the same route as dense `_agni3_assemble`: g^rv and
        # g^rz come from the PEST LOWER metric via g12 = (g13 g23 - g12 g33)/(sqrt(g))^2
        # rather than from data["g^rv"]/data["g^rz"]. The `_term` factors already
        # carry psi_r*sqrtg, hence psi_r2 here.
        jq = (
            psi_r2
            * W
            * (j_sup_theta * g_sup_rp_term - j_sup_zeta * g_sup_rv_term)
            / g_sup_rr
        )
        Ar += +(jq * (iota * xr_v + xr_z))
        Ar += -(psi_r * sqrtg * W * j_sup_zeta) * xr1_r
        Ar += +(psi_r * sqrtg * W * j_sup_theta) * xr2_r
        # RG: `iota` must sit INSIDE the transposed theta derivative.
        # With separable D_theta the term and its Hermitian agree because iota = iota(rho)
        # commutes with a pure-theta operator. But with coupled_rt D_theta is the
        # full (rho, theta) Zernike-Fourier operator and they do not, which made
        # this operator non-Hermitian and disagree with `_agni3_assemble`.
        Ar += d_dv(_cT(D_theta0), iota * jq * xr) + d_dz(_cT(D_zeta0), jq * xr)
        Ar += -iota_psi_r2 * d_dr(_cT(D_rho0), psi_r * sqrtg * W * j_sup_zeta * xr)
        Ar += psi_r2 * d_dr(_cT(D_rho0), psi_r * sqrtg * W * j_sup_theta * xr)

        # Off-diagonal term and it's transpose
        Ar += (
            +(W * psi_r2 * sqrtg * j_sup_theta) * xu_v
            + (W * psi_r2 * sqrtg * j_sup_zeta) * xu_z
        )
        Au += +d_dv(_cT(D_theta0), W * psi_r2 * sqrtg * j_sup_theta * xr)
        Au += d_dz(_cT(D_zeta0), W * psi_r2 * sqrtg * j_sup_zeta * xr)

        # |J|^2 and instability drive
        Ar += (psi_r2 * W * sqrtg * J2) / g_sup_rr * xr
        Aur = (W * psi_r2 * sqrtg * F) * xr

        # Compressibility terms
        gp = gamma * sqrtg * W * p0
        cr = psi_r * partial_r_log_sqrtg * xr + xr3_r
        cu = partial_v_log_sqrtg * xu + xu_v
        cz = (
            partial_v_log_sqrtg * xz
            + xz_v
            + iotainv * (partial_p_log_sqrtg * xz + xz_z)
        )

        Ar += psi_r * (partial_r_log_sqrtg * gp * cr + d_dr(_cT(D_rho0), gp * cr))
        Au += partial_v_log_sqrtg * gp * cu + d_dv(_cT(D_theta0), gp * cu)
        Az += (
            partial_v_log_sqrtg * gp * cz
            + d_dv(_cT(D_theta0), gp * cz)
            + iotainv * (partial_p_log_sqrtg * gp * cz + d_dz(_cT(D_zeta0), gp * cz))
        )

        Ar += psi_r * (partial_r_log_sqrtg * gp * cu + d_dr(_cT(D_rho0), gp * cu))
        Au += partial_v_log_sqrtg * gp * cr + d_dv(_cT(D_theta0), gp * cr)

        Ar += psi_r * (partial_r_log_sqrtg * gp * cz + d_dr(_cT(D_rho0), gp * cz))
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
        ys = jnp.einsum("lij,lj->li", Linv_D, diagBsqinv * As)  # stable
        yus = jnp.einsum("lij,lj->li", Linv_D, diagBsqinv * Aus)  # unstable
        y = ys + yus
        # Constant diagonal shift in the whitened L^-1 A L^-T basis, matching the
        # 1e-14 that dense ``_agni3_assemble`` adds before the instability drive.
        # Must stay in sync with it or the two paths differ by a uniform shift.
        return y.T.reshape(-1) + 1e-14 * x_flat

    def Ax(x_reduced):
        x_full = jnp.zeros(3 * n_total, dtype=x_reduced.dtype)
        # unique_indices=True: keep is a concatenation of disjoint aranges, so
        # the indices are unique. Declaring it lets JAX form the scatter's
        # transpose, which reverse-mode AD through the scatter needs.
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


def _agni3_store_rayleigh_mode_data(data, v, op):
    """Store full eigenfunction, xi, deltaB, and deltaV from a Rayleigh vector."""
    n_total = op["n_total"]
    n_rho = op["n_rho"]
    n_theta = op["n_theta"]
    n_zeta = op["n_zeta"]
    shape = (n_rho, n_theta, n_zeta)

    keep = op["keep"]
    v = jnp.asarray(v).reshape(-1)
    v_full = jnp.zeros(3 * n_total, dtype=v.dtype)
    v_full = v_full.at[keep].set(v, unique_indices=True)

    x = jnp.transpose(v_full.reshape(3, n_total), axes=(1, 0))
    x = op["diagBsqinv"] * jnp.einsum("lij,lj->li", op["Linv_DT"], x)
    xi_full = jnp.transpose(x, axes=(1, 0)).reshape(-1)

    xr = x[:, 0].reshape(shape)
    xv = x[:, 1].reshape(shape)
    xz = x[:, 2].reshape(shape)

    if jnp.iscomplexobj(xi_full):
        xi_ref = xi_full[:n_total]
        rot = jnp.exp(1j * jnp.arctan2(jnp.mean(xi_ref.real), jnp.mean(xi_ref.imag)))
        xr = (xr * rot).imag
        xv = (xv * rot).imag
        xz = (xz * rot).imag

    D_rho0 = op["D_rho0"]
    D_theta0 = op["D_theta0"]
    D_zeta0 = op["D_zeta0"]
    d_dr = op["d_dr"]
    d_dv = op["d_dv"]
    d_dz = op["d_dz"]

    psi_r_over_sqrtg = op["psi_r_over_sqrtg"]
    psi_r = op["psi_r"]
    iota = op["iota"]
    g_rr = op["g_rr"]
    g_vv = op["g_vv"]
    g_pp = op["g_pp"]
    g_rv = op["g_rv"]
    g_rp = op["g_rp"]
    g_vp = op["g_vp"]

    xr_v = d_dv(D_theta0, xr)
    xr_z = d_dz(D_zeta0, xr)

    test_v = d_dv(D_theta0, xv)
    test_z = d_dz(D_zeta0, xv)

    xr_r = d_dr(D_rho0, xr)
    psi_rr = d_dr(D_rho0, psi_r)
    iota_r = d_dr(D_rho0, iota)

    deltaB_r = psi_r_over_sqrtg * psi_r * (iota * xr_v + xr_z)
    deltaB_v = psi_r_over_sqrtg * (
        test_z - (xr_r * iota * psi_r + (2 * iota * psi_rr + iota_r * psi_r) * xr)
    )
    deltaB_z = -psi_r_over_sqrtg * (test_v + xr_r * psi_r + 2 * psi_rr * xr)

    deltaV_r = psi_r * xr
    deltaV_v = xv + xz
    deltaV_z = xz / iota

    deltaB2 = (
        g_rr * deltaB_r**2
        + g_vv * deltaB_v**2
        + g_pp * deltaB_z**2
        + 2.0
        * (
            g_rv * deltaB_r * deltaB_v
            + g_rp * deltaB_r * deltaB_z
            + g_vp * deltaB_v * deltaB_z
        )
    )
    deltaV2 = (
        g_rr * deltaV_r**2
        + g_vv * deltaV_v**2
        + g_pp * deltaV_z**2
        + 2.0
        * (
            g_rv * deltaV_r * deltaV_v
            + g_rp * deltaV_r * deltaV_z
            + g_vp * deltaV_v * deltaV_z
        )
    )

    data["finite-n eigenfunction3 rayleigh"] = v_full
    data["finite-n xi rayleigh"] = xi_full
    data["finite-n deltaB rayleigh"] = jnp.sqrt(deltaB2)
    data["finite-n deltaV rayleigh"] = jnp.sqrt(deltaV2)
    data["finite-n deltaB_r rayleigh"] = deltaB_r
    data["finite-n deltaB_v rayleigh"] = deltaB_v
    data["finite-n deltaB_z rayleigh"] = deltaB_z
    data["finite-n deltaV_r rayleigh"] = deltaV_r
    data["finite-n deltaV_v rayleigh"] = deltaV_v
    data["finite-n deltaV_z rayleigh"] = deltaV_z
    return data


@register_compute_fun(
    name="finite-n lambda3 rayleigh",
    label="\\lambda_R = v^T A v / v^T v",
    units="~",
    units_long="None",
    description="Rayleigh quotient of the finite-n lambda3 operator. A fresh "
    "eigenvector is computed at the primal point, then held fixed for AD so the "
    "gradient is the Hellmann-Feynman contraction v^T (dA/dp) v / v^T v.",
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
    sigma="float: shift for the ARPACK eigsh that supplies the fresh eigenvector",
    v_guess="ndarray: pcg_deflated only. Full-length start vector; used instead of "
    "the coarse seed when nonzero",
    eigensolver="str: 'eigsh_callback' (default), 'jax_lanczos' or 'pcg_deflated'. "
    "Env AGNI_EIGENSOLVER is a fallback only -- the kwarg wins",
    coarse_num_matvecs="int: Lanczos matvec count for the COARSE generalized "
    "solve that builds the deflation basis (default 100). Env "
    "AGNI_COARSE_NUM_MATVECS is a fallback",
    sigma_mode="str: jax_lanczos only. 'fixed' (default) or 'adapt' (second pass "
    "at sigma_factor * lambda). Env AGNI_SIGMA_MODE is a fallback",
    sigma_factor="float: jax_lanczos only. Shift multiplier for sigma_mode='adapt' "
    "(default 2.5)",
    factor="str: dense factorization for the shift-invert, 'lu' (default) or "
    "'cholesky'. Env AGNI_FACTOR is a fallback",
    gpu_lu="bool: keep the dense LU on device (default False)",
    k_defl="int: deflation rank (default 50). Env AGNI_K_DEFL is a fallback",
    ring_batch="int: pcg_deflated. Rings assembled at once for the ring "
    "preconditioner (default 64); lower it when the ring build runs out of memory",
    jd_outer="int: pcg_deflated. Maximum Jacobi-Davidson iterations (default 200)",
    jd_inner="int: PCG iterations per Jacobi-Davidson correction solve (default 100)",
    jd_maxdim="int: Jacobi-Davidson search-space size before restart (default 60)",
    jd_keep="int: Ritz vectors kept at a Jacobi-Davidson restart (default 10)",
    jd_tol="float: stop Jacobi-Davidson when ||A u - theta u||/|theta| <= jd_tol "
    "(default 0 = off)",
    jd_theta_tol="float: stop Jacobi-Davidson when the relative change of theta "
    "between outer iterations <= jd_theta_tol (default 1e-8; 0 = off)",
    jd_print="int: print theta and residual every N Jacobi-Davidson outer "
    "iterations (default 0 = off)",
    eigsh_tol="float: tolerance for the ARPACK eigsh",
    num_matvecs="int: jax_lanczos only. Lanczos matvec count (default 50). "
    "Env AGNI_NUM_MATVECS is a fallback only -- the kwarg wins",
    coarse_grid="Grid: optional COARSE level (mapped to DESC coords at these "
    "params) whose generalized modes of (H_c, M_ring,c) supply the deflation "
    "space and the Jacobi-Davidson start vector. Used by eigensolver='pcg_deflated' "
    "whenever it is given (the FinitenStability objective builds it only with "
    "AGNI_COARSE_DEFL=1).",
    coarse_diffmat="DiffMat: differentiation operators for coarse_grid",
    coarse_data="dict: prefilled flux/0-D data on coarse_grid, from the coarse "
    "level's own LinearGrid (its rho set differs from the fine one)",
    coarse_params="dict: parameters for the coarse assembly, already "
    "stop_gradient'd -- the coarse solve is a solver aid and carries no derivative",
    coarse_density="ndarray: normalized density on coarse_grid",
    coarse_res="tuple: (n_rho, n_theta, n_zeta) of coarse_grid -- the coupled "
    "Zernike operator reshapes by these, so the fine values must not be inherited",
    coarse_rho="ndarray: the coarse level's 1D rho nodes, from the PEST grid. "
    "Concrete: the MAPPED grid's nodes are built from params and are traced",
    fine_rho="ndarray: the fine level's 1D rho nodes, from the PEST grid, for "
    "the same reason as coarse_rho",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier (rho, theta) "
    "operators instead of separable 1D matrices",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
    v_fixed="array, optional: skip the eigensolve and use this as v. Only valid "
    "when v came from a call at this exact same x. Default None.",
)
def _AGNI3_rayleigh(params, transforms, profiles, data, **kwargs):
    """Fresh-eigenvector Rayleigh objective for finite-n lambda3.

    Called by
    ---------
    ``FinitenStability.compute_data`` through DESC's normal ``eq.compute`` path,
    with the compute key ``"finite-n lambda3 rayleigh"``, during objective and
    Jacobian evaluations.

    Inputs
    ------
    ``params``, ``transforms``, ``profiles``, and ``data`` are the standard DESC
    compute-function arguments. This key expects the metric/current/pressure
    entries listed in the decorator. ``kwargs`` carries the finite-n settings:
    ``sigma``, ``eigsh_tol``, ``axisym``, ``n_mode_axisym``, ``coupled_rt``, and
    grid-size metadata.

    Runtime knobs
    -------------
    ``AGNI_EIGENSOLVER=eigsh_callback`` uses a host scipy ARPACK callback.
    ``AGNI_EIGENSOLVER=jax_lanczos`` assembles/factorizes in JAX and uses matfree
    Lanczos; ``AGNI_FACTOR=lu`` (default) or ``cholesky`` picks the dense
    factorization behind its shift-invert ``OPinv``, and ``AGNI_SIGMA_MODE=adapt``
    gives it a second pass at ``sigma_factor * lambda``.
    ``AGNI_EIGENSOLVER=pcg_deflated`` is matrix-free Jacobi-Davidson with the
    ring + coarse-deflation preconditioner (``jd_*`` options).
    ``AGNI_DIAG=1`` prints xcheck; ``AGNI_DIAG=2`` also prints r_mu and
    separation diagnostics.

    Outputs
    -------
    Adds ``"finite-n lambda3 rayleigh"`` and
    ``"finite-n lambda3 rayleigh residual"`` to ``data``. The objective is always
    ``lam_R = real(vdot(v, Ax(v)) / vdot(v, v))``. The eigensolve returns a fresh
    ``v`` at the current primal point, but the custom VJP gives ``v`` zero
    cotangent, so gradients flow only through ``Ax(v)``.

    Examples
    --------
    ``tests/test_AGNI.py``: ``test_jax_lanczos_matches_dense`` (dense path),
    ``test_jd_two_level_matches_dense`` and ``test_jd_gradient_matches_dense_eigenvector``
    (matrix-free JD), ``test_jd_optimization_runs`` (optimizer route through
    ``FinitenStability``).
    """
    # noqa: unused dependency
    _ = params["Psi"]
    _op = _agni3_matfree_operator(params, transforms, profiles, data, **kwargs)
    n_keep = _op["n_keep"]
    _dtype = _op["Linv_DT"].dtype

    sigma = kwargs.get("sigma", -1e-1)
    eigsh_tol = kwargs.get("eigsh_tol", 1e-8)

    _array_data, _other_data = {}, {}
    for _k, _val in data.items():
        if isinstance(_val, (jnp.ndarray, np.ndarray, float, int)):
            _array_data[_k] = jnp.asarray(_val)
        else:
            _other_data[_k] = _val

    import jax.scipy.linalg as _jsla

    _use_gpu_lu = _solver_flag(kwargs, "gpu_lu", "AGNI_GPU_LU")

    def _assemble_and_solve_host(params_host, data_host):
        """Assemble ``A`` and run scipy ``eigsh`` for the rayleigh primal pass.

        Called by
        ---------
        ``_eigensolve`` through ``jax.pure_callback`` when
        ``AGNI_EIGENSOLVER=eigsh_callback`` or when the variable is unset. This
        is the default path used by ``finite-n lambda3 rayleigh`` during DESC
        objective and Jacobian evaluations.

        Inputs
        ------
        ``params_host`` and ``data_host`` are the array-valued leaves that JAX
        is allowed to pass through a callback. Non-array entries from DESC's
        compute ``data`` are restored from the enclosing ``_other_data`` dict.
        ``transforms``, ``profiles``, ``kwargs``, ``sigma``, and ``eigsh_tol``
        are closed over from ``_AGNI3_rayleigh``.

        Outputs
        -------
        Returns ``(v, lam_mu)`` as numpy arrays. ``v`` has shape ``(n_keep,)`` and
        is the fresh shifted-eigsh eigenvector at the current equilibrium.
        ``lam_mu`` is the eigenvalue reported by scipy. The objective does not
        differentiate through either output; they only choose the vector used in
        the differentiable Rayleigh quotient below.

        Test examples
        -------------
        ``tests/test_AGNI.py::test_finiten_objective_gradient_is_hellmann_feynman``.
        """
        p_h = {k: jnp.asarray(val) for k, val in params_host.items()}
        d_h = dict(_other_data)
        d_h.update({k: jnp.asarray(val) for k, val in data_host.items()})
        A_h = _agni3_assemble(p_h, transforms, profiles, d_h, **kwargs)["A"]
        nA = A_h.shape[0]

        if _use_gpu_lu:
            M = A_h.at[jnp.diag_indices(nA)].add(-sigma)
            jax.block_until_ready(M)
            A_h.delete()
            lu_piv = _jsla.lu_factor(M)
            jax.block_until_ready(lu_piv)
            M.delete()

            def _opinv(b):
                """Apply the GPU LU inverse used as scipy ``OPinv``.

                Called by scipy ARPACK inside ``eigsh`` only when
                ``AGNI_GPU_LU=1``. Input ``b`` is one Krylov vector from scipy.
                Output is ``(A - sigma I)^{-1} b`` copied back to numpy so scipy
                can continue the shift-invert iteration. This helper is covered
                by the same regression/optimizer jobs when ``AGNI_GPU_LU=1`` is
                exported.
                """
                x = _jsla.lu_solve(lu_piv, jnp.asarray(b, dtype=lu_piv[0].dtype))
                return np.asarray(x)

            def _never(_x):
                """Fail if scipy tries the wrong operator in shift-invert mode.

                ``eigsh`` should use ``OPinv`` for the shifted solve. This dummy
                matvec is installed only to satisfy scipy's ``LinearOperator``
                constructor. If it is called, the GPU-LU path is not doing the
                intended inverse iteration, so raising is better than silently
                computing the wrong thing.
                """
                raise RuntimeError("A.matvec should not be called in shift-invert")

            A_op = LinearOperator((nA, nA), matvec=_never, dtype=np.float64)
            OPinv = LinearOperator((nA, nA), matvec=_opinv, dtype=np.float64)
            w_h, v_h = eigsh(
                A_op,
                k=1,
                sigma=sigma,
                OPinv=OPinv,
                which="LM",
                tol=eigsh_tol,
                return_eigenvectors=True,
            )
            return (
                np.asarray(v_h[:, 0], dtype=np.float64),
                np.asarray(w_h[0], dtype=np.float64),
            )

        A_np = np.asarray(A_h)
        w_h, v_h = eigsh(
            A_np,
            k=1,
            sigma=sigma,
            which="LM",
            tol=eigsh_tol,
            return_eigenvectors=True,
        )
        return (
            np.asarray(v_h[:, 0], dtype=A_np.dtype),
            np.asarray(w_h[0], dtype=A_np.dtype),
        )

    _eigensolver = str(
        _solver_opt(kwargs, "eigensolver", "AGNI_EIGENSOLVER", "eigsh_callback")
    ).lower()
    _num_matvecs = int(_solver_opt(kwargs, "num_matvecs", "AGNI_NUM_MATVECS", 50))
    _sigma_mode = str(
        _solver_opt(kwargs, "sigma_mode", "AGNI_SIGMA_MODE", "fixed")
    ).lower()
    _valid = {"fixed", "adapt"}
    if _sigma_mode not in _valid:
        raise ValueError(
            "AGNI_SIGMA_MODE must be one of {}, got {!r}".format(
                sorted(_valid), _sigma_mode
            )
        )
    _adapt = "adapt" in _sigma_mode
    _adapt_factor = _solver_opt(kwargs, "sigma_factor", "AGNI_SIGMA_FACTOR", 2.5, float)
    # Dense factorization used as the shift-invert OPinv in `_eigensolve_jax`.
    # `H = A - sigma I` is SPD whenever sigma sits below the whole spectrum, which
    # is the operating regime, so `potrf` (n^3/3) may replace `getrf` (2n^3/3).
    # Default stays `lu` so this is opt-in.
    _factor = str(_solver_opt(kwargs, "factor", "AGNI_FACTOR", "lu")).lower()
    if _factor not in {"lu", "cholesky"}:
        raise ValueError(
            "AGNI_FACTOR must be 'lu' or 'cholesky', got {!r}".format(_factor)
        )
    # 0 off, 1 time the factorization alone, 2 also its Frobenius backward error
    # (needs `M` kept alive plus one n x n product -- small resolutions only).
    # Eager path only; see the timing block in `_solve_at`.
    _diag = int(os.environ.get("AGNI_DIAG", "1"))
    _xcheck = _diag >= 1
    _xcheck_tol = 1e-2  # a constant, not a knob: nothing ever needed to tune it

    def _eigensolve_jax(params_d, data_d):
        """All-JAX shifted eigensolve for the rayleigh primal vector.

        Called by
        ---------
        ``_eigensolve`` when ``AGNI_EIGENSOLVER=jax_lanczos``. This is the
        accelerator-resident alternative to the host scipy callback. It is still
        a primal eigensolve only; the derivative rule is supplied by
        ``_v_primal_bwd``.

        Inputs
        ------
        ``params_d`` and ``data_d`` are the JAX versions of DESC's ``params`` and
        array-valued compute ``data``. Non-array data are restored from
        ``_other_data``. The routine closes over ``sigma``, ``_sigma_mode``,
        ``_adapt_factor``, ``_num_matvecs``, and diagnostics configured through
        ``AGNI_SIGMA_MODE``, ``AGNI_SIGMA_FACTOR``, ``AGNI_NUM_MATVECS``, and
        ``AGNI_DIAG``.

        Outputs
        -------
        Returns ``(v, lam_mu)``. ``v`` is the selected Lanczos vector in the
        reduced ``n_keep`` space. ``lam_mu`` is reconstructed from the
        shift-invert eigenvalue ``mu`` as ``sigma + 1 / mu``. The value passed to
        DESC is not ``lam_mu``; it is recomputed below as the fixed-vector
        Rayleigh quotient ``real(vdot(v, Ax(v)) / vdot(v, v))``.

        Test examples
        -------------
        ``tests/test_AGNI.py::test_jax_lanczos_matches_dense``.
        """
        d_h = dict(_other_data)
        d_h.update(data_d)
        A = _agni3_assemble(params_d, transforms, profiles, d_h, **kwargs)["A"]
        nA = A.shape[0]
        _eager = not isinstance(A, jax.core.Tracer)

        def _solve_at(sig, keep_A):
            """Run one exact-factorization shift-invert Lanczos solve at a shift.

            Called by
            ---------
            ``_eigensolve_jax`` once for fixed mode and twice for adapt
            modes. The first call uses the configured ``sigma``. The second call
            uses ``AGNI_SIGMA_FACTOR * lam_mu`` when that value is finite and
            negative.

            Inputs
            ------
            ``sig`` is the scalar shift used to form ``A - sig I``. ``keep_A``
            controls eager memory cleanup: adapt mode keeps ``A`` after the
            first factorization so the second shifted matrix can be formed.

            Outputs
            -------
            Returns ``(v_out, lam_out, sep_out, ok_out)``. ``v_out`` is the
            selected Lanczos vector, ``lam_out`` is ``sig + 1 / mu`` for the
            selected inverse eigenvalue, ``sep_out`` is the leading separation
            ratio used only for diagnostics, and ``ok_out`` is the boolean
            factorization guard (always True for LU; for Cholesky it is False
            when ``A - sig I`` was not positive definite, because ``potrf``
            returns NaN rather than raising).
            """
            M = A.at[jnp.diag_indices(nA)].add(-sig)
            mat_dtype = M.dtype
            if _eager:
                jax.block_until_ready(M)
                if not keep_A:
                    try:
                        A.delete()
                    except Exception:
                        pass
            _t_fac0 = time.perf_counter()
            if _factor == "cholesky":
                # `cho_factor` is `potrf`: half the flops of `getrf` and no pivot
                # vector. JAX passes `symmetrize_input=False`, so only ONE
                # triangle of `M` is read and an asymmetric `M` would be used
                # asymmetrically -- `_agni3_assemble` returns a symmetric
                # whitened operator, which is what makes this legal.
                # `lower=True` is not cosmetic: JAX's `lower=False` path is
                # `conj(cholesky(conj(a.mT)).mT)`, two full n x n transposes, and
                # at n_keep = 84800 each one is another 57 GB temporary.
                fac = _jsla.cho_factor(M, lower=True)
                # `potrf` returns NaN instead of raising on an indefinite input,
                # so the guard is mandatory, not defensive. It is a single
                # reduction over the factor.
                ok = jnp.isfinite(fac[0]).all()
            else:
                fac = _jsla.lu_factor(M)
                ok = jnp.array(True)
            if _eager:
                jax.block_until_ready(fac)
                # Isolating the factorization is the whole point of the LU/Cholesky
                # comparison: the 2x is on `getrf`/`potrf` alone, and the
                # `_num_matvecs` triangular solves plus the Lanczos
                # reorthogonalization cost the same either way, so a whole-solve
                # timing dilutes the effect. Eager only -- under jit there is no
                # host-side boundary to measure across.
                _t_fac = time.perf_counter() - _t_fac0
                try:
                    M.delete()
                except Exception:
                    pass

            def _OPinv(b):
                """Apply ``(A - sig I)^{-1}`` to one Lanczos vector.

                ``decomp.tridiag_sym`` calls this repeatedly while building the
                Krylov basis. Input and output have shape ``(n_keep,)`` and stay
                in JAX arrays. This is deliberately an exact-factorization
                shift-invert, not the old PCG approximation. The triangular
                solves cost ``2 n^2`` per right-hand side either way; only the
                factorization above differs between ``AGNI_FACTOR`` settings.
                """
                if _factor == "cholesky":
                    return _jsla.cho_solve(fac, b)
                return _jsla.lu_solve(fac, b)

            # `decomp.tridiag_sym` is a real symmetric Lanczos (v^T w, not
            # v^dagger w), so on the complex Hermitian A that `axisym=True`
            # builds it returns wrong Ritz vectors; the values survive, so only
            # the Rayleigh residual shows it. Run it on the real symmetric 2n
            # embedding [[Re A, -Im A], [Im A, Re A]] and unstack afterward,
            # keeping the cheaper complex n x n factorization.
            _is_complex = bool(np.issubdtype(np.dtype(mat_dtype), np.complexfloating))

            if _is_complex:

                def _OPinv_real(b):
                    """`_OPinv` lifted to the real 2n embedding."""
                    z = _OPinv(b[:nA] + 1j * b[nA:])
                    return jnp.concatenate([jnp.real(z), jnp.imag(z)])

                _op, _kdim, _v0_dtype = _OPinv_real, 2 * nA, jnp.float64
            else:
                _op, _kdim, _v0_dtype = _OPinv, nA, mat_dtype

            _tri = decomp.tridiag_sym(_num_matvecs, reortho="full", materialize=True)
            _alg = eig.eigh_partial(_tri)
            _v0 = jnp.asarray(
                np.random.default_rng(0).standard_normal(_kdim), dtype=_v0_dtype
            )
            _v0 = _v0 / jnp.linalg.norm(_v0)
            mu, vecs = _alg(_op, _v0)

            idx = jnp.argmax(jnp.abs(mu))
            v_out = vecs[idx]
            if _is_complex:
                # Unstack the embedding: [Re v; Im v] -> v.
                v_out = v_out[:nA] + 1j * v_out[nA:]

            mu_i = mu[idx]
            lam_out = sig + 1.0 / jnp.where(mu_i == 0, jnp.inf, mu_i)
            ordered = jnp.abs(mu[jnp.argsort(-jnp.abs(mu))])
            sep_out = ordered[0] / jnp.where(ordered[1] == 0, jnp.inf, ordered[1])

            # Catch the NaN wherever it surfaced: a bad `potrf` poisons the
            # factor, but it also poisons everything downstream of it, so test
            # the outputs too rather than trusting the factor check alone.
            ok_out = ok & jnp.isfinite(lam_out) & jnp.isfinite(v_out).all()

            if _eager:
                jax.block_until_ready(v_out)
                try:
                    fac[0].delete()
                except Exception:
                    pass
            return v_out, lam_out, sep_out, ok_out

        v, lam_mu, sep_pass1, ok1 = _solve_at(sigma, keep_A=_adapt)
        sep = sep_pass1
        if _factor == "cholesky" and _xcheck:
            jax.debug.print(
                "[chol] pass1 sigma={s:+.6e} positive_definite={ok}"
                "   (False => A - sigma I is indefinite; potrf returned NaN)",
                s=sigma,
                ok=ok1,
            )

        if _adapt:
            sigma2 = _adapt_factor * jax.lax.stop_gradient(lam_mu)
            sigma2 = jnp.where(jnp.isfinite(sigma2) & (sigma2 < 0), sigma2, sigma)
            v2, lam2, sep2, ok2 = _solve_at(sigma2, keep_A=False)
            # The adapt shift comes from an ESTIMATE, so it can land above
            # lambda_min and make `A - sigma2 I` indefinite. Cholesky reports
            # that as NaN; select back to the pass-1 result rather than
            # propagating it. `jnp.where` is a select, so NaN inputs on the
            # discarded branch are harmless, and this stays fixed-shape under
            # jit where a try/except could not.
            v = jnp.where(ok2, v2, v)
            lam_mu = jnp.where(ok2, lam2, lam_mu)
            sep = jnp.where(ok2, sep2, sep_pass1)
            if _factor == "cholesky" and _xcheck:
                jax.debug.print(
                    "[chol] pass2 sigma={s:+.6e} positive_definite={ok}"
                    "   (False => fell back to the pass-1 shift result)",
                    s=sigma2,
                    ok=ok2,
                )

        return v, lam_mu

    def _eigensolve_pcg(params_d, data_d, _Zc_in=None, _v0c_in=None, sigma=sigma):
        """Matrix-free Jacobi-Davidson eigensolve (``eigensolver='pcg_deflated'``).

        Never forms the fine matrix: ``A`` is applied through
        ``_agni3_matfree_operator``. Finds the lowest eigenpair of ``A`` with
        Jacobi-Davidson (JD):

        * Search space ``V``; each iteration does Rayleigh-Ritz with the exact
          ``A`` over ``V`` -> lowest Ritz pair ``(theta, u)``, residual
          ``r = A u - theta u``.
        * Correction equation ``(I - u u^H)(A - sigma I)(I - u u^H) t = -r``,
          solved with ``jd_inner`` projected PCG iterations, preconditioned by
          ``M^-1 = M_ring^-1 + Y Y^H``: ring block-Jacobi Cholesky of
          ``A - sigma I`` plus the deflation built from the coarse modes ``Z``.
          ``t`` is orthogonalized and appended to ``V``.
        * At ``jd_maxdim`` vectors, restart from the ``jd_keep`` lowest Ritz
          vectors. Stop at ``jd_outer`` iterations, or earlier on ``jd_tol``
          (eig-res) or ``jd_theta_tol`` (relative change of theta).

        ``sigma`` does not pick the eigenvalue (Rayleigh-Ritz does); it only
        shifts the preconditioner and the correction operator, and must sit below
        the spectrum so the ring blocks are SPD. Start vector: the prolonged
        softest coarse mode ``_v0c_in``, else a fixed random vector.

        ``_Zc_in`` / ``_v0c_in`` come from ``_coarse_space``, built outside the
        custom_vjp. Everything here is fixed-shape and traceable.
        """
        import numpy as _np

        from ._stability_solvers import build_ring_blocks as _build_rings
        from ._stability_solvers import deflation_Y as _defl_Y
        from ._stability_solvers import factor_ring_blocks_traced as _fbt
        from ._stability_solvers import make_block_precond as _mkprec
        from ._stability_solvers import ring_index_maps as _ring_maps

        _jd_outer = _solver_opt(kwargs, "jd_outer", "AGNI_JD_OUTER", 200, int)
        _jd_inner = _solver_opt(kwargs, "jd_inner", "AGNI_JD_INNER", 100, int)
        _jd_maxdim = _solver_opt(kwargs, "jd_maxdim", "AGNI_JD_MAXDIM", 60, int)
        _jd_keep = _solver_opt(kwargs, "jd_keep", "AGNI_JD_KEEP", 10, int)
        # DEBUG: print theta and |r| every N iterations (0 = off)
        _jd_print = _solver_opt(kwargs, "jd_print", "AGNI_JD_PRINT", 0, int)
        # Early stop (0 = off; jd_outer stays the maximum):
        #   jd_tol       stop when eig-res ||A u - theta u|| / |theta| <= jd_tol
        #   jd_theta_tol stop when |theta - theta_prev| / |theta| <= jd_theta_tol
        # Measured (C7): Zernike 3D eig-res plateaued at ~3e-3 while theta was
        # already fixed to 1e-9, so jd_tol alone may never trigger there.
        _jd_tol = _solver_opt(kwargs, "jd_tol", "AGNI_JD_TOL", 0.0, float)
        # Default 1e-8: measured at 48x48x16 it kept lambda within 1e-7 of the
        # converged value at 4-9x less work than a 200,000-multiplication budget.
        # The eigenvector is NOT converged at that point (residual ~1).
        _jd_theta_tol = _solver_opt(
            kwargs, "jd_theta_tol", "AGNI_JD_THETA_TOL", 1e-8, float
        )

        d_h = dict(_other_data)
        d_h.update(data_d)
        _opm = _agni3_matfree_operator(params_d, transforms, profiles, d_h, **kwargs)
        _Ax = _opm["Ax"]
        nA = int(_opm["n_keep"])
        n_rho = int(_opm["n_rho"])
        n_theta = int(_opm["n_theta"])
        n_zeta = int(_opm["n_zeta"])
        # HOST copy of `keep`, derived from the resolution rather than read back
        # from the operator dict: it feeds array SHAPES in the ring maps, so it
        # cannot be a tracer. It is grid structure,
        #     arange(n_shell, n_total - n_shell) U arange(n_total, 3*n_total),
        # with n_shell = n_total // n_rho.
        _ntot_op = int(_opm["n_total"])
        _nshell = _ntot_op // n_rho
        _keep = _np.concatenate(
            [
                _np.arange(_nshell, _ntot_op - _nshell),
                _np.arange(_ntot_op, 3 * _ntot_op),
            ]
        )
        # Check the re-derivation whenever the real one is concrete.
        if not isinstance(_opm["keep"], jax.core.Tracer):
            _keep_ref = _np.asarray(jax.device_get(_opm["keep"]))
            if not _np.array_equal(_keep, _keep_ref):
                raise RuntimeError(
                    "pcg_deflated: host-derived `keep` disagrees with the "
                    f"operator's (sizes {_keep.size} vs {_keep_ref.size}). The "
                    "construction in _agni3_matfree_operator has changed."
                )

        # ---- preconditioner at sigma: ring blocks + coarse deflation ----------
        # One vmapped ring assembly (sub-blocks of the dense matrix to 5e-16,
        # test_ring_blocks_vmapped_match_dense), shifted by -sigma inside.
        ring_sel, ring_pad, _G = _ring_maps(_keep, (n_rho, n_theta, n_zeta))
        _Gn = _np.asarray(_G)
        _bs = _build_rings(
            _agni3_assemble,
            params_d,
            transforms,
            profiles,
            d_h,
            kwargs,
            (n_rho, n_theta, n_zeta),
            ring_sel,
            ring_pad,
            sigma,
            _solver_opt(kwargs, "ring_batch", "AGNI_RING_BATCH", 64, int),
        )
        _L, _ok, _ = _fbt(_bs)
        if isinstance(_ok, jax.core.Tracer) or isinstance(sigma, jax.core.Tracer):
            jax.debug.print(
                "[jd] ring blocks SPD={o} at sigma={s:.6e} "
                "(False: sigma is above lambda_min and the solve is invalid)",
                o=_ok,
                s=sigma,
            )
        elif not bool(_ok):
            raise RuntimeError(
                f"pcg_deflated: ring blocks not SPD at sigma={sigma}. The shift is "
                "above lambda_min, so A - sigma I is indefinite."
            )
        _Mr = _mkprec(_L, _Gn, nA)
        _Mop, _rk = _Mr, 0
        if _Zc_in is not None and _Zc_in.shape[0] == nA:
            _Zj = jnp.asarray(_Zc_in)
            # vmap traces the operator once regardless of k_defl.
            _HZ = jax.vmap(lambda x: _Ax(x) - sigma * x, in_axes=1, out_axes=1)(_Zj)
            _Y, _rk = _defl_Y(_Zj, _HZ)

            # Hermitian outer product: Y is complex for axisym=True.
            def _Mdefl(r, _Y=_Y):
                return _Mr(r) + _Y @ (jnp.conj(jnp.swapaxes(_Y, 0, 1)) @ r)

            _Mop = _Mdefl

        def _H(X):
            return jnp.conj(jnp.swapaxes(X, -1, -2))

        # ---- Jacobi-Davidson ---------------------------------------------------
        _m = _jd_maxdim
        if _v0c_in is not None:
            _u0 = jnp.asarray(_v0c_in).astype(_dtype)
        else:
            _u0 = jnp.asarray(
                np.random.default_rng(0).standard_normal(nA), dtype=_dtype
            )
        _u0 = _u0 / jnp.linalg.norm(_u0)
        _V = jnp.zeros((nA, _m), dtype=_dtype).at[:, 0].set(_u0)
        _AV = jnp.zeros((nA, _m), dtype=_dtype).at[:, 0].set(_Ax(_u0))
        _live = jnp.arange(_m)

        def _ritz(V, AV, j):
            used = _live < j
            both = used[:, None] & used[None, :]
            S = _H(V) @ AV
            S = 0.5 * (S + _H(S))
            S = jnp.where(both, S, 0.0)
            # unused slots get a diagonal above every Ritz value, so the lowest
            # eigenpairs live on the used block only
            big = jnp.linalg.norm(S) + 1.0
            S = S + jnp.diag(jnp.where(used, 0.0, big)).astype(S.dtype)
            return jnp.linalg.eigh(S)

        def _proj_pcg(u, b):
            def P(x):
                return x - u * jnp.vdot(u, x)

            def Op(x):
                Px = P(x)
                return P(_Ax(Px) - sigma * Px)

            def Pre(x):
                return P(_Mop(P(x)))

            def body(st):
                x, r, p, rz, k, ok = st
                Ap = Op(p)
                curv = jnp.real(jnp.vdot(p, Ap))
                good = curv > 0
                a = jnp.where(good, rz / jnp.where(good, curv, 1.0), 0.0)
                x = x + a * p
                r = r - a * Ap
                z = Pre(r)
                rzn = jnp.real(jnp.vdot(r, z))
                p = z + (rzn / rz) * p
                return (x, r, p, rzn, k + jnp.where(good, 1, 0), good)

            r0 = P(b)
            z0 = Pre(r0)
            st = (
                jnp.zeros_like(b),
                r0,
                z0,
                jnp.real(jnp.vdot(r0, z0)),
                0,
                jnp.array(True),
            )
            x, _, _, _, k, _ = jax.lax.while_loop(
                lambda s: (s[4] < _jd_inner) & s[5], body, st
            )
            return P(x), k

        def _outer(state):
            V, AV, j, it, kt, th_prev, _, _ = state
            w, Y = _ritz(V, AV, j)
            y = Y[:, 0]
            u = V @ y
            Au = AV @ y
            nu = jnp.linalg.norm(u)
            u, Au = u / nu, Au / nu
            r = Au - w[0] * u
            rn = jnp.linalg.norm(r)
            # stopping quantities of the CURRENT space (checked by `_go`)
            eres = rn / jnp.maximum(jnp.abs(w[0]), 1e-300)
            dth = jnp.abs(w[0] - th_prev) / jnp.maximum(jnp.abs(w[0]), 1e-300)
            t, k = _proj_pcg(u, -r)

            def _restart(a):
                V_, AV_, _, Y_ = a
                Yk = Y_[:, :_jd_keep]
                Vn = jnp.zeros_like(V_).at[:, :_jd_keep].set(V_ @ Yk)
                AVn = jnp.zeros_like(AV_).at[:, :_jd_keep].set(AV_ @ Yk)
                return Vn, AVn, jnp.asarray(_jd_keep)

            V, AV, j = jax.lax.cond(
                j >= _m, _restart, lambda a: (a[0], a[1], a[2]), (V, AV, j, Y)
            )
            for _ in range(2):
                t = t - V @ (_H(V) @ t)
            tn = jnp.linalg.norm(t)
            good = jnp.isfinite(tn) & (tn > 1e-300)
            t = t / jnp.where(good, tn, 1.0)
            V = jnp.where(good, V.at[:, j].set(t), V)
            AV = jnp.where(good, AV.at[:, j].set(_Ax(t)), AV)
            j = j + jnp.where(good, 1, 0)
            if _jd_print:
                jax.lax.cond(
                    (it + 1) % _jd_print == 0,
                    lambda: jax.debug.print(
                        "[jd it] iter={o} theta={t:.10e} |r|={rn:.3e} "
                        "eig-res={e:.3e} dtheta={d:.3e} pcg={k}",
                        o=it + 1,
                        t=w[0],
                        rn=rn,
                        e=eres,
                        d=dth,
                        k=k,
                        ordered=False,
                    ),
                    lambda: None,
                )
            return (V, AV, j, it + 1, kt + k, w[0], eres, dth)

        def _go(state):
            it, eres, dth = state[3], state[6], state[7]
            done = jnp.array(False)
            if _jd_tol > 0:
                done = done | (eres <= _jd_tol)
            if _jd_theta_tol > 0:
                done = done | (dth <= _jd_theta_tol)
            return (it < _jd_outer) & ~done

        _inf = jnp.asarray(jnp.inf)
        _V, _AV, _j, _it, _kt, _, _eres, _dth = jax.lax.while_loop(
            _go,
            _outer,
            (_V, _AV, jnp.asarray(1), jnp.asarray(0), jnp.asarray(0), _inf, _inf, _inf),
        )
        _w, _Y = _ritz(_V, _AV, _j)
        _vv = _V @ _Y[:, 0]
        _vv = _vv / jnp.linalg.norm(_vv)
        if _xcheck:
            _rfin = jnp.linalg.norm(_AV @ _Y[:, 0] - _w[0] * (_V @ _Y[:, 0]))
            jax.debug.print(
                "[jd] n={n} rings={g} defl_rank={r} sigma={s:.6e} iters={ou} "
                "(max {o}) inner={i} maxdim={m} keep={kp} tol={tl:.1e} "
                "theta_tol={ttl:.1e} pcg_total={k} theta={t:.8e} eig-res={e:.3e} "
                "(last check {el:.3e}, dtheta {d:.3e})",
                n=nA,
                g=int(_Gn.shape[0]),
                r=_rk,
                s=sigma,
                ou=_it,
                o=_jd_outer,
                i=_jd_inner,
                m=_m,
                kp=_jd_keep,
                tl=_jd_tol,
                ttl=_jd_theta_tol,
                k=_kt,
                t=_w[0],
                e=_rfin / jnp.maximum(jnp.abs(_w[0]), 1e-300),
                el=_eres,
                d=_dth,
            )
        return _vv, _w[0]

    def _coarse_space(params_d, data_d):
        """Build the coarse deflation basis and seed OUTSIDE the custom_vjp.

        Returns ``(Z, v0)`` or ``(None, None)`` when no coarse level is given.
        The coarse level is a SOLVER AID -- already stop_gradient'd by the
        objective and carrying no derivative -- so computing it here and
        passing the two arrays in as explicit inputs is equivalent
        mathematically and legal for `jax.custom_vjp`, which cannot hold a
        freshly built traced subgraph among its jaxpr constants.
        """
        import numpy as _np  # aliased inside _eigensolve_pcg; needed here too

        _Z = None
        _seed_v0 = None
        _kdefl = _solver_opt(kwargs, "k_defl", "AGNI_K_DEFL", 50, int)
        d_h = dict(_other_data)
        d_h.update(data_d)
        _opm = _agni3_matfree_operator(params_d, transforms, profiles, d_h, **kwargs)
        nA = int(_opm["n_keep"])
        n_rho = int(_opm["n_rho"])
        n_theta = int(_opm["n_theta"])
        n_zeta = int(_opm["n_zeta"])
        # ---- COARSE-SPACE DEFLATION (active when coarse_grid is given) -------
        # The objective passes a SECOND level at this evaluation's parameters
        # (already stop_gradient'd there). Its generalized modes of the pencil
        # (H_c, M_ring,c) are the deflation space theory asks for -- eigenvectors
        # of the PRECONDITIONED operator, not of H alone -- and its softest mode,
        # prolonged, is the Jacobi-Davidson start vector.
        _cg = kwargs.get("coarse_grid", None)
        if _cg is not None:
            # Prolongation, coarse generalized eigensolve and the deflation basis.
            from ._stability_solvers import barycentric_matrix as _bary
            from ._stability_solvers import build_ring_blocks as _build_rings
            from ._stability_solvers import coarse_seed_and_deflation as _cseed
            from ._stability_solvers import fourier_interp_matrix as _fint
            from ._stability_solvers import level_meta as _oparr
            from ._stability_solvers import ring_index_maps as _ring_maps

            _cdm = kwargs["coarse_diffmat"]
            _cdata = kwargs["coarse_data"]
            _cpar = kwargs["coarse_params"]
            # the physics kwargs, minus this block's own plumbing
            _ckw = {
                _k: _v
                for _k, _v in kwargs.items()
                if not _k.startswith("coarse_")
                and _k not in ("sigma", "eigsh_tol", "v_guess")
            }
            if "coarse_density" in kwargs:
                _ckw["density"] = kwargs["coarse_density"]
            _cres = tuple(int(_x) for _x in kwargs["coarse_res"])
            if _ckw.get("coupled_rt", False):
                _ckw["n_rho_coupled"] = _cres[0]
                _ckw["n_theta_coupled"] = _cres[1]
            _ctr = {"grid": _cg, "diffmat": _cdm}
            _cop = _agni3_matfree_operator(_cpar, _ctr, profiles, _cdata, **_ckw)
            _cmeta = _oparr(_cop)
            _nc = int(_cop["n_keep"])
            # Shift the diagonal only: a dense eye(_nc) and an explicit
            # 0.5 * (A + A^H) would each hold another n x n copy. The Hermitian
            # part is taken inside `coarse_gen_modes` after the congruence,
            # which gives the same matrix (see the comment there).
            _cdiag = jnp.arange(_nc)
            _cHc = _agni3_assemble(_cpar, _ctr, profiles, _cdata, **_ckw)["A"]
            _cHc = _cHc.at[_cdiag, _cdiag].add(-sigma)
            # Ring blocks of H_c, with the -sigma shift applied inside (traced
            # build, safe under jit).
            # Host-derived for the same reason as the fine level above: this
            # feeds array shapes, so it cannot be a tracer.
            _cntot = int(_cop["n_total"])
            _cnshell = _cntot // int(_cop["n_rho"])
            _ckeep = _np.concatenate(
                [
                    _np.arange(_cnshell, _cntot - _cnshell),
                    _np.arange(_cntot, 3 * _cntot),
                ]
            )
            if not isinstance(_cop["keep"], jax.core.Tracer):
                _ckeep_ref = _np.asarray(jax.device_get(_cop["keep"]))
                if not _np.array_equal(_ckeep, _ckeep_ref):
                    raise RuntimeError(
                        "coarse deflation: host-derived `keep` disagrees with "
                        f"the operator's ({_ckeep.size} vs {_ckeep_ref.size})."
                    )
            _csel, _cpad, _cG = _ring_maps(_ckeep, _cres)
            _cblk = _build_rings(
                _agni3_assemble,
                _cpar,
                _ctr,
                profiles,
                _cdata,
                _ckw,
                _cres,
                _csel,
                _cpad,
                sigma,
                _solver_opt(kwargs, "ring_batch", "AGNI_RING_BATCH", 64, int),
            )
            _cGn = _np.asarray(_cG)
            # Interpolation matrices: radial node positions only, no equilibrium.
            #
            # These come in as kwargs from the PEST grids rather than being read
            # off `_cg.nodes` / `transforms["grid"].nodes`. Those grids are built
            # by `eq.map_coordinates(params=...)`, so their nodes are TRACED and
            # `_np.asarray` on them raises. rho is invariant under
            # the PEST->DESC map, so the values are the same either way.
            _rho_c = _np.asarray(kwargs["coarse_rho"], dtype=float).reshape(-1)
            _rho_f = _np.asarray(kwargs["fine_rho"], dtype=float).reshape(-1)
            if _rho_c.size != _cres[0] or _rho_f.size != n_rho:
                raise ValueError(
                    "coarse deflation: rho node counts disagree with the "
                    f"resolutions -- coarse {_rho_c.size} vs {_cres[0]}, fine "
                    f"{_rho_f.size} vs {n_rho}."
                )
            # Check the invariance claim instead of trusting it, whenever the
            # mapped nodes are concrete (every eager run).
            if not isinstance(_cg.nodes, jax.core.Tracer):
                _rc_ref = _np.asarray(jax.device_get(_cg.nodes[:, 0])).reshape(
                    _cres[0], -1
                )[:, 0]
                if not _np.allclose(_rho_c, _rc_ref, rtol=0, atol=1e-12):
                    raise RuntimeError(
                        "coarse deflation: PEST rho differs from the mapped "
                        "grid's rho by "
                        f"{_np.abs(_rho_c - _rc_ref).max():.3e}. rho is NOT "
                        "invariant under the PEST->DESC map here, so the "
                        "prolongation operator would be built on wrong nodes."
                    )
            _pr = _bary(_rho_c, _rho_f)
            _pt = _fint(_cres[1], n_theta, 2.0 * _np.pi)
            _pz = _fint(_cres[2], n_zeta, 2.0 * _np.pi / _cop.get("NFP", 1))
            _kc = min(_kdefl, _nc - 1)
            # `_cGn` carries -1 padding; the in-package routine derives its own
            # mask from it, so no separate mask argument is passed.
            _v0c, _Zc, _lamc = _cseed(
                _cHc,
                _cblk,
                _cGn,
                _cmeta,
                _oparr(_opm),
                _pr,
                _pt,
                _pz,
                _kc,
                _solver_opt(
                    kwargs, "coarse_num_matvecs", "AGNI_COARSE_NUM_MATVECS", 100, int
                ),
            )
            # Stays on device. `_Z` is only ever passed to `_solve_at`, which
            # does `jnp.asarray(_Zin)` and reads `.shape` -- both fine for a
            # device array. The round-trip here moved k*nA doubles (50 x 81792 =
            # 33 MB at 48x48x12) to the host for nothing, and broke trace.
            _Z = _Zc
            _seed_v0 = _v0c
            if _xcheck:
                jax.debug.print(
                    "[coarse_defl] n_c={nc} k={k} lam_c0={l:.6e} Z{z}",
                    nc=_nc,
                    k=_kc,
                    l=_lamc[0],
                    z=_Zc.shape[1],
                )

        return _Z, _seed_v0

    def _eigensolve(params_d, data_d, _Zc_in=None, _v0c_in=None, _sig_in=sigma):
        """Dispatch the primal eigensolve used by ``_v_primal``.

        Inputs are array-only DESC ``params`` and ``data``. Output is always
        ``(v, lam_mu)`` with ``v.shape == (n_keep,)`` and scalar ``lam_mu``.
        ``AGNI_EIGENSOLVER=jax_lanczos`` selects ``_eigensolve_jax``; otherwise
        this calls ``_assemble_and_solve_host`` through ``jax.pure_callback``.
        The regression and optimizer scripts exercise this dispatch whenever
        they compute ``"finite-n lambda3 rayleigh"``.
        """
        if _eigensolver == "jax_lanczos":
            return _eigensolve_jax(params_d, data_d)
        if _eigensolver == "pcg_deflated":
            return _eigensolve_pcg(params_d, data_d, _Zc_in, _v0c_in, _sig_in)
        return jax.pure_callback(
            _assemble_and_solve_host,
            (
                jax.ShapeDtypeStruct((n_keep,), _dtype),
                jax.ShapeDtypeStruct((), _dtype),
            ),
            params_d,
            data_d,
        )

    # Built OUTSIDE the custom_vjp, then passed in. See `_coarse_space`.
    _Zc_ext, _v0c_ext = _coarse_space(params, _array_data)
    # pcg_deflated warm start: a nonzero full-length `v_guess` (the previous
    # solve's `finite-n eigenfunction3 rayleigh`) replaces the coarse seed.
    _vg = kwargs.get("v_guess", None)
    if _eigensolver == "pcg_deflated" and _vg is not None:
        _vg = jnp.asarray(_vg).reshape(-1)
        if _vg.shape[0] == 3 * int(_op["n_total"]):
            _vg = _vg[_op["keep"]]
            if _v0c_ext is None:
                _v0c_ext = np.random.default_rng(0).standard_normal(_vg.shape[0])
            _v0c_ext = jnp.where(jnp.linalg.norm(_vg) > 0, _vg, _v0c_ext)

    @jax.custom_vjp
    def _v_primal(params_d, data_d, _Zc_in, _v0c_in, _sig_in):
        """Return the fresh primal eigenpair with a custom derivative rule.

        Called directly below before building ``Av``. Inputs are the array-only
        ``params`` and ``data`` dictionaries. Output is ``(v, lam_mu)`` from
        ``_eigensolve``. The important contract is that ``v`` is recomputed at
        the current primal point, but the backward pass returns zero cotangents
        for all inputs so the optimizer sees the fixed-vector Rayleigh gradient.
        """
        return _eigensolve(params_d, data_d, _Zc_in, _v0c_in, _sig_in)

    def _v_primal_fwd(params_d, data_d, _Zc_in, _v0c_in, _sig_in):
        """Forward rule for ``_v_primal``.

        Inputs are the same array-only ``params`` and ``data`` dictionaries. It
        calls ``_eigensolve`` once and returns ``(v, lam_mu)`` to the primal
        calculation. The residual saves only the inputs because the backward rule
        must manufacture matching zero cotangents for the same pytrees.
        """
        v_out = _eigensolve(params_d, data_d, _Zc_in, _v0c_in, _sig_in)
        return v_out, (params_d, data_d, _Zc_in, _v0c_in, _sig_in)

    def _v_primal_bwd(res, g):
        """Backward rule for ``_v_primal``.

        ``res`` contains the ``params`` and ``data`` pytrees saved by
        ``_v_primal_fwd``. ``g`` is the incoming cotangent for ``(v, lam_mu)`` and
        is intentionally ignored. Outputs are zero pytrees matching ``params``
        and ``data``. This is what enforces the desired math path:

        1. fresh ``v`` from the eigensolve at the primal point,
        2. no gradient through the eigensolve or eigenvector selection,
        3. gradient only through ``Av = Ax(v)`` in the Rayleigh quotient.
        """
        params_d, data_d, _Zc_in, _v0c_in, _sig_in = res
        # Zero for the coarse inputs too: the coarse level is a solver aid the
        # objective already stop_gradient'd, so it carries no derivative. `None`
        # is an empty pytree and tree_map returns None, which is the correct
        # cotangent for an absent input.
        return (
            jax.tree_util.tree_map(jnp.zeros_like, params_d),
            jax.tree_util.tree_map(jnp.zeros_like, data_d),
            jax.tree_util.tree_map(jnp.zeros_like, _Zc_in),
            jax.tree_util.tree_map(jnp.zeros_like, _v0c_in),
            jnp.zeros_like(_sig_in),
        )

    _v_primal.defvjp(_v_primal_fwd, _v_primal_bwd)

    # Opt-in: when the caller already has v from a call at this EXACT SAME x,
    # skip the eigensolve and treat v as a constant, so jax.grad differentiates
    # only through Ax(v). Reusing v at a DIFFERENT x is a measured catastrophic
    # failure -- a 7e-5 relative mesh shift flipped lam_R's sign and moved it
    # 66x -- so this must not be reachable by an optimizer or line search.
    _v_fixed = kwargs.get("v_fixed", None)
    if _v_fixed is not None:
        v = jnp.asarray(_v_fixed)
        lam_mu = jnp.asarray(jnp.nan)  # no eigensolve ran
    else:
        # sigma is an explicit input: under adapt it is traced, and a tracer
        # closed over inside the custom_vjp cannot be lowered.
        v, lam_mu = _v_primal(
            params, _array_data, _Zc_ext, _v0c_ext, jnp.asarray(sigma, dtype=float)
        )

    Av = _op["Ax"](v)
    vv = jnp.vdot(v, v)
    lam_R = jnp.real(jnp.vdot(v, Av) / vv)

    if _xcheck and _v_fixed is None:
        _den = jnp.maximum(jnp.abs(lam_mu), 1e-300)
        _gap = jnp.abs(lam_R - lam_mu) / _den
        _sign_ok = jnp.sign(lam_R) == jnp.sign(lam_mu)
        _ok = (_gap < _xcheck_tol) & _sign_ok
        jax.debug.print(
            "[xcheck] lam_R={a:+.8e}  lam_mu={b:+.8e}  rel_gap={g:.3e}"
            "  sign_ok={s}  trusted={t}",
            a=lam_R,
            b=lam_mu,
            g=_gap,
            s=_sign_ok,
            t=_ok,
        )

    resid = jnp.linalg.norm(Av - lam_R * v) / (
        jnp.abs(lam_R) * jnp.sqrt(jnp.real(vv)) + 1e-300
    )

    data["finite-n lambda3 rayleigh"] = jnp.atleast_1d(lam_R)
    data["finite-n lambda3 rayleigh residual"] = jnp.atleast_1d(resid)
    # So a caller can take v from a value call and pass it back as `v_fixed`.
    data["finite-n lambda3 rayleigh v"] = jnp.atleast_1d(v)
    data = _agni3_store_rayleigh_mode_data(data, v, _op)
    return data


@register_compute_fun(
    name="finite-n eigenfunction3 rayleigh",
    label="\\xi_R",
    units="~",
    units_long="None",
    description="Finite-n Rayleigh eigenfunction, full component-major vector",
    dim=5,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=["finite-n lambda3 rayleigh"],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    density="ndarray: the radial density profile",
    sigma="float: shift for the eigensolver",
    eigsh_tol="float: tolerance for the ARPACK eigsh",
    eigensolver="str: 'eigsh_callback', 'jax_lanczos' or 'pcg_deflated'",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier operators",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
)
def _AGNI_rayleigh_eigenfunction3(params, transforms, profiles, data, **kwargs):
    """Eigenfunction from the finite-n Rayleigh solve."""
    _ = params["Psi"]
    return data  # noqa: unused dependency


@register_compute_fun(
    name="finite-n xi rayleigh",
    label="\\xi_R",
    units="~",
    units_long="None",
    description="Physical displacement from the finite-n Rayleigh solve",
    dim=5,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=["finite-n lambda3 rayleigh"],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    density="ndarray: the radial density profile",
    sigma="float: shift for the eigensolver",
    eigsh_tol="float: tolerance for the ARPACK eigsh",
    eigensolver="str: 'eigsh_callback', 'jax_lanczos' or 'pcg_deflated'",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier operators",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
)
def _AGNI_rayleigh_xi(params, transforms, profiles, data, **kwargs):
    """Physical displacement from the finite-n Rayleigh solve."""
    _ = params["Psi"]
    return data  # noqa: unused dependency


@register_compute_fun(
    name="finite-n deltaB rayleigh",
    label="|\\delta B_R|",
    units="~",
    units_long="None",
    description="Magnetic perturbation magnitude from the finite-n Rayleigh solve",
    dim=3,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=["finite-n lambda3 rayleigh"],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    density="ndarray: the radial density profile",
    sigma="float: shift for the eigensolver",
    eigsh_tol="float: tolerance for the ARPACK eigsh",
    eigensolver="str: 'eigsh_callback', 'jax_lanczos' or 'pcg_deflated'",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier operators",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
)
def _AGNI_rayleigh_deltaB(params, transforms, profiles, data, **kwargs):
    """Magnetic perturbation magnitude from the finite-n Rayleigh solve."""
    _ = params["Psi"]
    return data  # noqa: unused dependency


@register_compute_fun(
    name="finite-n deltaV rayleigh",
    label="|\\delta V_R|",
    units="~",
    units_long="None",
    description="Volume displacement magnitude from the finite-n Rayleigh solve",
    dim=3,
    params=["Psi"],
    transforms={"grid": [], "diffmat": []},
    profiles=[],
    coordinates="rtz",
    data=["finite-n lambda3 rayleigh"],
    axisym="bool: if the equilibrium is axisymmetric",
    n_mode_axisym="int: toroidal mode number to study",
    incompressible="bool: imposes incompressibility",
    gamma="float: adiabatic constant",
    density="ndarray: the radial density profile",
    sigma="float: shift for the eigensolver",
    eigsh_tol="float: tolerance for the ARPACK eigsh",
    eigensolver="str: 'eigsh_callback', 'jax_lanczos' or 'pcg_deflated'",
    coupled_rt="bool: D_rho/D_theta are full 2D Zernike-Fourier operators",
    n_rho_coupled="int: number of rho nodes when coupled_rt is set",
    n_theta_coupled="int: number of theta nodes when coupled_rt is set",
)
def _AGNI_rayleigh_deltaV(params, transforms, profiles, data, **kwargs):
    """Volume displacement magnitude from the finite-n Rayleigh solve."""
    _ = params["Psi"]
    return data  # noqa: unused dependency
