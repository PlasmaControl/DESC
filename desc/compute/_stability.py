"""Compute functions for stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from scipy.constants import mu_0

from desc.backend import jit, jnp, scan, vmap

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
            # Todo: implement equivalent of equation 4.3 in desc coordinates
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
    name="ideal ball lambda",
    label="\\lambda_{\\mathrm{ballooning}}=\\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared ideal ballooning growth rate, "
    "requires data along a field line",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "a",
        "g^aa",
        "g^ra",
        "g^rr",
        "cvdrift",
        "cvdrift0",
        "|B|",
        "B^zeta",
        "p_r",
        "iota",
        "shear",
        "psi",
        "psi_r",
        "rho",
    ],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    zeta0="array: points of vanishing integrated local shear to scan over. "
    "Default 15 points linearly spaced in [-π/2,π/2]",
)
def _ideal_ballooning_gamma2(params, transforms, profiles, data, **kwargs):
    """
    Ideal-ballooning growth rate finder.

    This function uses a finite-difference method
    to calculate the maximum growth rate against the
    infinite-n ideal ballooning mode. The equation being solved is

    d/dζ(g dX/dζ) + c X = λ f X, g, f > 0

    where

    𝛋 = b ⋅∇ b
    g = a_N^3 * B_N * (b ⋅∇ζ) * (dψ_N/dρ)² * |∇α|², / B,
    c = a_N^3 * B_N * (1/ b ⋅∇ζ) * (dψ_N/dρ)² * dp/dψ * (b × 𝛋) ⋅|∇α|/ B**2,
    f = a_N * B_N^3 * (dψ_N/dρ)² * |∇α|² / B^3 * (1/ b ⋅∇ζ) ,

    are needed along a field line to solve the ballooning equation once and
    find

    λ = a_N^2 / v_A^2 * γ²,

    where

    v_A = B_N /sqrt(mu_0 * n0 * M) is the Alfven speed, and
    ψ_N = ψ/ψ_b is the normalized toroidal flux, and
    ψ_b = 0.5*(B_N * a_N**2) is the total enclosed toroidal flux.

    To obtain the parameters g, c, and f, we need a set of parameters
    provided in the list ``data`` above. Here's a description of
    these parameters:

    - a: minor radius of the device
    - g^aa: |grad alpha|^2, field line bending term
    - g^ra: (grad alpha dot grad rho) integrated local shear
    - g^rr: |grad rho|^2 flux expansion term
    - cvdrift: geometric factor of the curvature drift
    - cvdrift0: geometric factor of curvature drift 2
    - |B|: magnitude of the magnetic field
    - B^zeta:  B dot grad zeta
    - p_r: dp/drho, pressure gradient
    - phi: coordinate describing the position in the toroidal angle
    along a field line
    """
    source_grid = transforms["grid"].source_grid
    # Vectorize in rho later
    rho = source_grid.meshgrid_reshape(data["rho"], "arz")

    psi_b = params["Psi"] / (2 * jnp.pi)
    a_N = data["a"]
    B_N = 2 * psi_b / a_N**2

    zeta0 = kwargs.get("zeta0", jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, 15))
    N_zeta0 = len(zeta0)

    # This would fail with rho vectorization
    iota = jnp.mean(data["iota"])
    shear = jnp.mean(data["shear"])
    psi = jnp.mean(data["psi"])
    sign_psi = jnp.sign(psi)
    sign_iota = jnp.sign(iota)

    N_rho = int(source_grid.num_rho)
    N_alpha = int(source_grid.num_alpha)

    # phi is the same for each alpha
    phi = source_grid.nodes[:: N_rho * N_alpha, 2]
    N_zeta = len(phi)

    B = source_grid.meshgrid_reshape(data["|B|"], "arz")
    B_sup_zeta = source_grid.meshgrid_reshape(data["B^zeta"], "arz")
    gradpar = B_sup_zeta / B

    # This would fail with rho vectorization
    dpdpsi = jnp.mean(mu_0 * data["p_r"] / data["psi_r"])

    g_sup_aa = source_grid.meshgrid_reshape(data["g^aa"], "arz")[None, ...]
    g_sup_ra = source_grid.meshgrid_reshape(data["g^ra"], "arz")[None, ...]
    g_sup_rr = source_grid.meshgrid_reshape(data["g^rr"], "arz")[None, ...]

    gds2 = jnp.reshape(
        rho**2
        * (
            g_sup_aa
            - 2 * sign_iota * shear / rho * zeta0[:, None, None, None] * g_sup_ra
            + zeta0[:, None, None, None] ** 2 * (shear / rho) ** 2 * g_sup_rr
        ),
        (N_alpha, N_zeta0, N_zeta),
    )

    f = a_N * B_N**3 * gds2 / B**3 * 1 / gradpar
    g = a_N**3 * B_N * gds2 / B * gradpar
    g_half = (g[:, :, 1:] + g[:, :, :-1]) / 2

    cvdrift = source_grid.meshgrid_reshape(data["cvdrift"], "arz")[None, ...]
    cvdrift0 = source_grid.meshgrid_reshape(data["cvdrift0"], "arz")[None, ...]

    c = (
        a_N**3
        * B_N
        * jnp.reshape(
            2
            / B_sup_zeta[None, ...]
            * sign_psi
            * rho**2
            * dpdpsi
            * (cvdrift - shear / (2 * rho**2) * zeta0[:, None, None, None] * cvdrift0),
            (N_alpha, N_zeta0, N_zeta),
        )
    )

    h = phi[1] - phi[0]

    i = jnp.arange(N_alpha)[:, None, None, None]
    l = jnp.arange(N_zeta0)[None, :, None, None]
    j = jnp.arange(N_zeta - 2)[None, None, :, None]
    k = jnp.arange(N_zeta - 2)[None, None, None, :]

    A = jnp.zeros((N_alpha, N_zeta0, N_zeta - 2, N_zeta - 2))
    B_inv = jnp.zeros((N_alpha, N_zeta0, N_zeta - 2, N_zeta - 2))

    A = A.at[i, l, j, k].set(
        g_half[i, l, k] / h**2 * (j - k == -1)
        + (-(g_half[i, l, j + 1] + g_half[i, l, j]) / h**2 + c[i, l, j + 1])
        * (j - k == 0)
        + g_half[i, l, j] / h**2 * (j - k == 1)
    )

    B_inv = B_inv.at[i, l, j, k].set(1 / jnp.sqrt(f[i, l, j + 1]) * (j - k == 0))

    A_redo = B_inv @ A @ jnp.transpose(B_inv, axes=(0, 1, 3, 2))

    w, _ = jnp.linalg.eigh(A_redo)
    # max over "zeta" axis, still a function of rho, alpha, zeta0
    gamma = jnp.real(jnp.max(w, axis=(2,)))

    data["ideal ball lambda"] = gamma.flatten()

    return data


@register_compute_fun(
    name="Newcomb ball metric",
    label="\\mathrm{Newcomb-ballooning-metric}",
    units="~",
    units_long="None",
    description="A measure of Newcomb's distance from marginal ballooning stability",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "a",
        "g^aa",
        "g^ra",
        "g^rr",
        "cvdrift",
        "cvdrift0",
        "|B|",
        "B^zeta",
        "p_r",
        "iota",
        "shear",
        "psi",
        "psi_r",
        "rho",
    ],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
    zeta0="array: points of vanishing integrated local shear to scan over. "
    "Default 15 points linearly spaced in [-π/2,π/2]",
)
def _Newcomb_ball_metric(params, transforms, profiles, data, **kwargs):
    """
    Ideal-ballooning growth rate proxy.

    This function uses a finite-difference method to integrate the
    marginal stability ideal-ballooning equation

    d/dζ(g dX/dζ) + c X = 0, g > 0

    using the Newcomb's stability criterion. The geometric factors

    𝛋 = b ⋅∇ b
    g = a_N^3 * B_N * (b ⋅∇ζ) * (dψ_N/dρ)² * |∇α|², / B,
    c = a_N^3 * B_N * (1/ b ⋅∇ζ) * (dψ_N/dρ)² * dp/dψ * (b × 𝛋) ⋅|∇α|/ B**2,

    are needed along a field line to solve the ballooning equation and
    ψ_N = ψ/ψ_b is the normalized toroidal flux, and
    ψ_b = 0.5*(B_N * a_N**2) is the enclosed toroidal flux by the boundary.

    To obtain the parameters g, c, and f, we need a set of parameters
    provided in the list ``data`` above. Here's a description of
    these parameters:

    - a: minor radius of the device
    - g^aa: |grad alpha|^2, field line bending term
    - g^ra: (grad alpha dot grad rho) integrated local shear
    - g^rr: |grad rho|^2 flux expansion term
    - cvdrift: geometric factor of the curvature drift
    - cvdrift0: geometric factor of curvature drift 2
    - |B|: magnitude of the magnetic field
    - B^zeta: B dot grad zeta
    - p_r: dp/drho, pressure gradient
    - psi_r: radial gradient of the toroidal flux
    - phi: coordinate describing the position in the toroidal angle
    along a field line

    Here's how we define the Newcomb metric:
    If zero crossing is at -inf (root finder failed), use the Y coordinate
    as a metric of stability else use the zero-crossing point on the X-axis
    as the metric
    This idea behind Newcomb's method is explained further in Appendix D of
    [Gaur _et al._](https://doi.org/10.1017/S0022377823000107)
    """
    source_grid = transforms["grid"].source_grid
    # Vectorize in rho later
    rho = source_grid.meshgrid_reshape(data["rho"], "arz")

    psi_b = params["Psi"] / (2 * jnp.pi)
    a_N = data["a"]
    B_N = 2 * psi_b / a_N**2

    zeta0 = kwargs.get("zeta0", jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, 15))
    N_zeta0 = len(zeta0)

    # This would fail with rho vectorization
    iota = jnp.mean(data["iota"])
    shear = jnp.mean(data["shear"])
    psi = jnp.mean(data["psi"])
    sign_psi = jnp.sign(psi)
    sign_iota = jnp.sign(iota)

    N_rho = int(source_grid.num_rho)
    N_alpha = int(source_grid.num_alpha)

    # phi is the same for each alpha
    phi = source_grid.nodes[:: N_rho * N_alpha, 2]
    N_zeta = len(phi)

    B = source_grid.meshgrid_reshape(data["|B|"], "arz")
    B_sup_zeta = source_grid.meshgrid_reshape(data["B^zeta"], "arz")
    gradpar = B_sup_zeta / B

    dpdpsi = source_grid.meshgrid_reshape(mu_0 * data["p_r"] / data["psi_r"], "arz")

    g_sup_aa = source_grid.meshgrid_reshape(data["g^aa"], "arz")[None, :]
    g_sup_ra = source_grid.meshgrid_reshape(data["g^ra"], "arz")[None, :]
    g_sup_rr = source_grid.meshgrid_reshape(data["g^rr"], "arz")[None, :]

    gds2 = jnp.reshape(
        rho**2
        * (
            g_sup_aa
            - 2 * sign_iota * shear / rho * zeta0[:, None] * g_sup_ra
            + zeta0[:, None] ** 2 * (shear / rho) ** 2 * g_sup_rr
        ),
        (N_alpha, N_zeta0, N_zeta),
    )

    g = a_N**3 * B_N * gds2 / B * gradpar
    g_half = (g[:, :, 1:] + g[:, :, :-1]) / 2

    cvdrift = source_grid.meshgrid_reshape(data["cvdrift"], "arz")[None, :]
    cvdrift0 = source_grid.meshgrid_reshape(data["cvdrift0"], "arz")[None, :]

    c = (
        a_N**3
        * B_N
        * jnp.reshape(
            2
            / B_sup_zeta[None, :]
            * sign_psi
            * rho**2
            * dpdpsi
            * (cvdrift - shear / (2 * rho**2) * zeta0[:, None] * cvdrift0),
            (N_alpha, N_zeta0, N_zeta),
        )
    )

    h = phi[1] - phi[0]

    # g_half on half grid points, c_full on full grid points
    g_half = (g[:, :, 1:] + g[:, :, :-1]) / 2
    c_full = c[:, :, :-1]

    i = jnp.arange(N_alpha)[:, None, None]
    j = jnp.arange(N_zeta0)[None, :, None]
    k = jnp.arange(N_zeta - 1)[None, None, :]

    X = jnp.zeros((N_alpha, N_zeta0, N_zeta - 1))
    X = X.at[i, j, k].set(phi[k])

    Y = jnp.zeros((N_alpha, N_zeta0))
    eps = 5e-3  # slope of the test functio
    Yp = eps * jnp.ones((N_alpha, N_zeta0))

    @jit
    def integrator(carry, x):
        y, dy = carry
        g_element, c_element = x
        # Update the array (Y) and its derivative on scattered grids and
        # integrate using leapfrog-like method.
        y_new = y + h * dy / g_element
        dy_new = dy - c_element * y_new * h
        # y starts at 0 with positive slope. If y goes negative it's unstable,
        # so we look for a sign change.
        sign_change = y_new < 0.0
        return (y_new, dy_new), (y_new, sign_change)

    @jit
    def cumulative_update_jit(y, dy, g_half, c_full):
        _, scan_output = scan(integrator, (y, dy), (g_half, c_full))
        Y, sign_change = scan_output
        # argmax of boolean array returns index if first True, where y goes negative
        first_negative_index = jnp.argmax(sign_change)
        # return last index if there are no sign crossings
        first_negative_index = jnp.where(
            ~jnp.any(sign_change),
            -1,
            first_negative_index,
        )
        # slope of Y where it crosses 0
        slope = (Y[first_negative_index] - Y[first_negative_index - 1]) / h
        # This factor will give us the exact X point of intersection
        lin_interp_factor = jnp.where(
            first_negative_index != -1,
            -Y[first_negative_index - 1] / slope,
            0,
        )

        return Y, first_negative_index, lin_interp_factor

    # Vectorize over the first two dimensions
    vectorized_cumulative_update = jit(vmap(vmap(cumulative_update_jit)))
    Y, first_negative_indices, lin_interp_factors = vectorized_cumulative_update(
        Y, Yp, g_half, c_full
    )

    # x at crossing pts, or last value of x if there were no crossings
    X0 = jnp.zeros((N_alpha, N_zeta0))
    i0 = jnp.arange(N_alpha)[:, None]
    j0 = jnp.arange(N_zeta0)[None, :]
    X0 = X0.at[i0, j0].set(
        X[i0, j0, first_negative_indices[i0, j0]] + lin_interp_factors[i0, j0] * h
    )
    # where X0 < phimax, it means there was a zero crossing so its unstable. We take
    # the distance from X0 to phimax as the distance to stability. If there was no
    # crossing we take Y[phi=phimax]. This gives a continuous metric, though
    # the first derivative will be discontinuous. Could maybe think of something better?
    # RG: Peak of the metric doesn't match mean peak of the growth rate in rho
    metric = jnp.where(
        first_negative_indices != -1,
        # if it crossed, then X0 < phimax, so this < 0
        (X0 - jnp.max(phi)) / jnp.ptp(phi),
        # if it reached the end without crossing, this is >=0
        Y[:, :, -1],
    )

    data["Newcomb ball metric"] = jnp.min(metric)

    return data
