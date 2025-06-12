"""Compute functions for stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

import numpy as np
from scipy.constants import mu_0

from desc.backend import jax, jit, jnp, scan, vmap

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
    name="c balloon",
    # c = a³ Bₙ  / b⋅∇ζ (dψ_N/dρ)² dp/dψ (b × 𝛋) ⋅ |∇α|/B²
    label="a^3 B_n / (b \\cdot \\nabla ζ) (\\partial_{\\rho} \\psi_N)^2 "
    "(\\partial_{\\psi} \\rho) (b \\times \\kappa) \\cdot "
    "\\vert \\nabla \\alpha \\vert^2 / B",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "a",
        "g^aa",
        "g^ra",
        "g^rr",
        "cvdrift",
        "cvdrift0",
        "|B|^2",
        "B^zeta",
        "p_r",
        "iota",
        "shear",
        "psi",
        "psi_r",
        "rho",
    ],
    zeta0="array: points of vanishing integrated local shear to scan over. "
    "Default 15 points linearly spaced in [-π/2,π/2]",
)
def _c_balloon(params, transforms, profiles, data, **kwargs):
    zeta0 = kwargs.get("zeta0", jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, 15))
    zeta0 = zeta0.reshape(-1, 1)
    psi_boundary = params["Psi"] / (2 * jnp.pi)
    B_n = 2 * psi_boundary / data["a"] ** 2
    constant1 = data["a"] * B_n**3
    constant2 = data["a"] ** 3 * B_n

    data["c balloon"] = (
        constant2
        * mu_0
        * data["p_r"]
        / data["psi_r"]
        * jnp.sign(data["psi"])
        / data["B^zeta"]
        * (
            2 * data["rho"] ** 2 * data["cvdrift"]
            - data["shear"] * data["cvdrift0"] * zeta0
        )
    )
    gds2 = (
        data["rho"] ** 2 * data["g^aa"]
        - 2
        * data["rho"]
        * jnp.sign(data["iota"])
        * data["shear"]
        * data["g^ra"]
        * zeta0
        + data["shear"] ** 2 * data["g^rr"] * zeta0**2
    )
    data["f balloon"] = (constant1 / data["|B|^2"] / data["B^zeta"]) * gds2
    data["g balloon"] = (constant2 / data["|B|^2"] * data["B^zeta"]) * gds2
    return data


@register_compute_fun(
    name="f balloon",
    # f = a  Bₙ³ / b⋅∇ζ (dψ_N/dρ)² |∇α|² / B³
    label="a B_n^3 / (b \\cdot \\nabla ζ) (\\partial_{\\rho} \\psi_N)^2 "
    "\\vert \\nabla \\alpha \\vert^2 / B^3",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["c_balloon"],
)
def _f_balloon(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    return data


@register_compute_fun(
    name="g balloon",
    # g = a³ Bₙ * b⋅∇ζ (dψ_N/dρ)² |∇α|² / B
    label="a^3 B_n b \\cdot \\nabla ζ (\\partial_{\\rho} \\psi_N)^2 "
    "\\vert \\nabla \\alpha \\vert^2 / B",
    units="~",
    units_long="None",
    description="Parameter in ideal ballooning equation",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["c_balloon"],
)
def _g_balloon(params, transforms, profiles, data, **kwargs):
    # noqa: unused dependency
    return data


@register_compute_fun(
    name="ideal ballooning lambda",
    label="\\lambda_{\\mathrm{ballooning}}=\\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared ideal ballooning growth rate",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["c balloon", "f balloon", "g balloon"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    Neigvals="int: number of largest eigenvalues to return, default value is 1.`"
    "If `Neigvals=2` eigenvalues are `[-1, 0, 1]` we get `[1, 0]`",
    eigfuns="bool: Whether to return eigenfunctions. Default is true.",
)
def _ideal_ballooning_lambda(params, transforms, profiles, data, **kwargs):
    """Eigenvalues of ideal-ballooning equation.

    A finite-difference method is used to calculate the maximum
    growth rate against the infinite-n ideal ballooning mode.
    The equation being solved is

    d/dζ(g dX/dζ) + c X = λ f X, g, f > 0

    where

    λ = a² / v_A² * γ²
    v_A = Bₙ / sqrt(μ₀ n₀ M) is the Alfven speed
    ψ_N = ψ/ψ_b     is the normalized toroidal flux
    ψ_b = a² Bₙ / 2 is the total enclosed toroidal flux

    Returns
    -------
    Ideal-ballooning lambda eigenvalues
        Shape (num_rho, num alpha, num zeta0, num eigvals).

    """
    Neigvals = kwargs.get("Neigvals", 1)
    eigfuns = kwargs.get("eigfuns", True)
    grid = transforms["grid"].source_grid
    # toroidal step size between points along field lines is assumed uniform
    dz = grid.nodes[grid.unique_zeta_idx[:2], 2]
    dz = dz[1] - dz[0]
    num_zeta0 = data["c balloon"].shape[0]

    def reshape(f):
        assert f.shape == (num_zeta0, grid.num_nodes)
        f = jnp.swapaxes(grid.meshgrid_reshape(f.T, "raz"), -1, -2)
        assert f.shape == (grid.num_rho, grid.num_alpha, num_zeta0, grid.num_zeta)
        return f

    c, f, g = map(reshape, (data["c balloon"], data["f balloon"], data["g balloon"]))
    g_half = (g[..., 1:] + g[..., :-1]) / 2
    diag_inner = c[..., 1:-1] - (g_half[..., 1:] + g_half[..., :-1]) / dz**2
    diag_outer = g_half[..., 1:-1] / dz**2

    j = np.arange(grid.num_zeta - 2)
    A = (
        jnp.zeros(
            (
                grid.num_rho,
                grid.num_alpha,
                num_zeta0,
                grid.num_zeta - 2,
                grid.num_zeta - 2,
            )
        )
        .at[..., j, j]
        .set(diag_inner, indices_are_sorted=True, unique_indices=True)
        .at[..., j[:-1], j[1:]]
        .set(diag_outer, indices_are_sorted=True, unique_indices=True)
        .at[..., j[1:], j[:-1]]
        .set(diag_outer, indices_are_sorted=True, unique_indices=True)
    )
    B_inv = jnp.reciprocal(jnp.sqrt(f[..., 1:-1]))
    A = B_inv[..., jnp.newaxis] * A * B_inv[..., jnp.newaxis, :]

    # TODO: Issue #1750
    # Try jax.scipy.eigh_tridiagonal or a better solver for improved performance
    if eigfuns:
        w, v = jnp.linalg.eigh(A)
    else:
        w = jnp.linalg.eigvalsh(A)

    w, top_idx = jax.lax.top_k(w, k=Neigvals)
    assert w.shape == (grid.num_rho, grid.num_alpha, num_zeta0, Neigvals)
    data["ideal ballooning lambda"] = w

    if eigfuns:
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
    dim=4,
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
    assert kwargs.get("eigfuns", True)
    return data  # noqa: unused dependency


@register_compute_fun(
    name="Newcomb ballooning metric",
    label="Newcomb ballooning metric",
    units="~",
    units_long="None",
    description="A measure of Newcomb's distance from marginal ballooning stability",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["c balloon", "f balloon", "g balloon"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _Newcomb_ball_metric(params, transforms, profiles, data, **kwargs):
    """Ideal-ballooning growth rate proxy.

    A finite-difference method is used to integrate the
    marginal stability ideal-ballooning equation

    d/dζ(g dX/dζ) + c X = 0, g > 0

    where

    λ = a² / v_A² * γ²
    v_A = Bₙ / sqrt(μ₀ n₀ M) is the Alfven speed
    ψ_N = ψ/ψ_b     is the normalized toroidal flux
    ψ_b = a² Bₙ / 2 is the total enclosed toroidal flux

    The Newcomb's stability criterion is used.
    We define the Newcomb metric as follows:
    If zero crossing is at -inf (root finder failed), use the Y coordinate as a
    metric of stability. Otherwise use the zero-crossing point on the X-axis.
    This idea behind Newcomb's method is explained further in Appendix D of
    [Gaur _et al._](https://doi.org/10.1017/S0022377823000107)

    """
    grid = transforms["grid"].source_grid
    # toroidal step size between points along field lines is assumed uniform
    zeta = grid.compress(grid.nodes[:, 2], surface_label="zeta")
    dz = zeta[1] - zeta[0]
    num_zeta0 = data["c balloon"].shape[0]

    def reshape(f):
        assert f.shape == (num_zeta0, grid.num_nodes)
        f = jnp.swapaxes(grid.meshgrid_reshape(f.T, "raz"), -1, -2)
        assert f.shape == (grid.num_rho, grid.num_alpha, num_zeta0, grid.num_zeta)
        return f

    c, f, g = map(reshape, (data["c balloon"], data["f balloon"], data["g balloon"]))
    c = c[..., :-1]
    g_half = (g[..., 1:] + g[..., :-1]) / 2

    j = np.arange(grid.num_zeta)
    X = (
        jnp.zeros((grid.num_rho, grid.num_alpha, num_zeta0, grid.num_zeta - 1))
        .at[..., j[:-1]]
        .set(zeta[:-1], unique_indices=True, indices_are_sorted=True)
    )

    Y = jnp.zeros(X.shape[:-1])
    eps = 5e-3  # slope of the test function
    Yp = eps * jnp.ones(Y.shape)

    @jit
    def integrator(carry, x):
        y, dy = carry
        g_element, c_element = x
        # Update the array (Y) and its derivative on scattered grids and
        # integrate using leapfrog-like method.
        y_new = y + dz * dy / g_element
        dy_new = dy - c_element * y_new * dz
        # y starts at 0 with positive slope. If y goes negative it's unstable,
        # so we look for a sign change.
        sign_change = y_new < 0.0
        return (y_new, dy_new), (y_new, sign_change)

    @jit
    def cumulative_update_jit(y, dy, g_half, c_full):
        _, scan_output = scan(integrator, (y, dy), (g_half, c_full))
        Y, sign_change = scan_output
        sign_change = sign_change.at[-1].set(1)
        first_negative_index = jnp.argmax(sign_change)
        # slope of Y where it crosses 0
        slope = (Y[first_negative_index] - Y[first_negative_index - 1]) / dz
        # This factor will give us the exact X point of intersection
        lin_interp_factor = jnp.where(
            first_negative_index != -1,
            -Y[first_negative_index - 1] / slope,
            0,
        )

        return Y, first_negative_index, lin_interp_factor

    # Vectorize over the first two dimensions
    vectorized_cumulative_update = jit(vmap(vmap(vmap(cumulative_update_jit))))
    Y, first_negative_indices, lin_interp_factors = vectorized_cumulative_update(
        Y, Yp, g_half, c
    )

    # x at crossing pts, or last value of x if there were no crossings
    X0 = jnp.zeros(Y.shape)
    X0 = X0.at[..., j, j].set(
        X[..., j, j, first_negative_indices[..., j, j]]
        + lin_interp_factors[..., j, j] * dz
    )
    # where X0 < phimax, it means there was a zero crossing so its unstable. We take
    # the distance from X0 to phimax as the distance to stability. If there was no
    # crossing we take Y[phi=phimax]. This gives a continuous metric, though
    # the first derivative will be discontinuous. Could maybe think of something better?
    # RG: Peak of the metric doesn't match mean peak of the growth rate in rho
    metric = jnp.where(
        first_negative_indices != -1,
        # if it crossed, then X0 < phimax, so this < 0
        (X0 - zeta[-1]) / (zeta[-1] - zeta[0]),
        # if it reached the end without crossing, this is >=0
        Y[..., -1],
    )

    data["Newcomb ballooning metric"] = jnp.min(metric)
    return data
