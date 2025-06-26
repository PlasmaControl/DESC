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

from desc.backend import jax, jit, jnp, scan, vmap

from ..diffmat_utils import cheb_D1, cheb_D2, fourier_diffmat
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
    name="ideal ballooning lambda",
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
    Neigvals="int: number of largest eigenvalues to return, default value is 1.`"
    "If `Neigvals=2` eigenvalues are `[-1, 0, 1]` we get `[1, 0]`",
)
def _ideal_ballooning_lambda(params, transforms, profiles, data, **kwargs):
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
    Neigvals = kwargs.get("Neigvals", 1)
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
        jnp.transpose(
            rho**2
            * (
                g_sup_aa
                - 2 * sign_iota * shear / rho * zeta0[:, None, None, None] * g_sup_ra
                + zeta0[:, None, None, None] ** 2 * (shear / rho) ** 2 * g_sup_rr
            ),
            axes=(1, 0, 2, 3),
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
            jnp.transpose(
                2
                / B_sup_zeta[None, ...]
                * sign_psi
                * rho**2
                * dpdpsi
                * (
                    cvdrift
                    - shear / (2 * rho**2) * zeta0[:, None, None, None] * cvdrift0
                ),
                axes=(1, 0, 2, 3),
            ),
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

    # TODO: Issue #1750
    # Try jax.scipy.eigh_tridiagonal or a better solver for improved performance
    w, v = jnp.linalg.eigh(A_redo)

    # Find the top_k eigenvalues.
    top_eigvals, top_theta_idxs = jax.lax.top_k(w, k=Neigvals)

    # v becomes less than the machine precision at some theta points which gives NaNs
    # stop_gradient prevents that. Not sure how it will affect an objective that
    # requires both the eigenvalue and eigenfunction
    top_eigfuns = jnp.take_along_axis(
        jax.lax.stop_gradient(v), top_theta_idxs[..., None, :], axis=-1
    )

    data["ideal ballooning lambda"] = top_eigvals
    data["ideal ballooning eigenfunction"] = top_eigfuns

    return data


@register_compute_fun(
    name="ideal ballooning eigenfunction",
    label="X_{\\mathrm{ballooning}}",
    units="~",
    units_long="None",
    description="Ideal ballooning eigenfunction",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["ideal ballooning lambda"],
    parameterization=["desc.equilibrium.equilibrium.Equilibrium"],
    source_grid_requirement={"coordinates": "raz", "is_meshgrid": True},
)
def _ideal_ballooning_eigenfunction(params, transforms, profiles, data, **kwargs):
    return data  # noqa: unused dependency


@register_compute_fun(
    name="Newcomb ballooning metric",
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
        jnp.transpose(
            rho**2
            * (
                g_sup_aa
                - 2 * sign_iota * shear / rho * zeta0[:, None, None, None] * g_sup_ra
                + zeta0[:, None, None, None] ** 2 * (shear / rho) ** 2 * g_sup_rr
            ),
            axes=(1, 0, 2, 3),
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
            jnp.transpose(
                2
                / B_sup_zeta[None, ...]
                * sign_psi
                * rho**2
                * dpdpsi
                * (
                    cvdrift
                    - shear / (2 * rho**2) * zeta0[:, None, None, None] * cvdrift0
                ),
                axes=(1, 0, 2, 3),
            ),
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

    data["Newcomb ballooning metric"] = jnp.min(metric)
    return data


@register_compute_fun(
    name="finite-n lambda",
    label="low-\\n \\lambda = \\gamma^2",
    units="~",
    units_long="None",
    description="Normalized squared growth rate",
    dim=1,
    params=["Psi", "NFP"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "a",
        "g_rr|PEST",
        "g_rt|PEST",
        "g_rz|PEST",
        "g_tt|PEST",
        "g_tz|PEST",
        "g_zz|PEST",
        "J^theta|PEST",
        "J^zeta|PEST",
        "sqrt(g)_PEST",
        "sqrt(g)_r|PEST",
        "sqrt(g)_t|PEST",
        "sqrt(g)_z|PEST",
        "iota",
        "iota_r",
        "iota_rr",
        "p",
        "p_r",
        "p_rr",
        "psi_r",
        "psi_rr",
        "(psi_r/sqrt(g)_PEST)",
        "(chi_r/sqrt(g)_PEST)",
    ],
    n_rho_max="int: 2 x maximum radial mode number",
    n_theta_max="int: 2 x maximum radial mode number",
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
    """
    a_N = data["a"]
    B_N = params["Psi"] / (jnp.pi * a_N**2)

    # --no-verify NFP = params["NFP"]
    NFP = 1

    iota = data["iota"][:, None]
    iota_r = data["iota_r"][:, None]
    iota_rr = data["iota_rr"][:, None]

    p = mu_0 * data["p"][:, None] / B_N**2
    p_r = mu_0 * data["p_r"][:, None] / B_N**2
    p_rr = mu_0 * data["p_rr"][:, None] / B_N**2
    psi_r = data["psi_r"][:, None] / (a_N**2 * B_N)
    psi_rr = data["psi_rr"][:, None] / (a_N**2 * B_N)

    psi_r_sqrt_g = data["(psi_r/sqrt(g)_PEST)"][:, None] * (a_N / B_N)
    chi_r_sqrt_g = data["(chi_r/sqrt(g)_PEST)"][:, None] * (a_N / B_N)

    axisym = kwargs.get("axisym", False)

    n_rho_max = kwargs.get("n_rho_max", 8)
    n_theta_max = kwargs.get("n_theta_max", 8)
    n_zeta_max = kwargs.get("n_zeta_max", 4)

    if axisym:
        D_zeta0 = n_zeta_max * jnp.array([[0, -1], [1, 0]])
        n_zeta_max = 2
    else:
        D_zeta0 = fourier_diffmat(n_zeta_max)

    def _eval_1D(f, x):
        return vmap(lambda x_val: f(x_val))(x)

    def _f(x):
        x_0 = 0.5
        m_1 = 2.1
        m_2 = 2.1
        lower = x_0 * (1 - jnp.exp(-m_1 * (x + 1)) + 0.5 * (x + 1) * jnp.exp(-2 * m_1))
        upper = (1 - x_0) * (jnp.exp(m_2 * (x - 1)) + 0.5 * (x - 1) * jnp.exp(-2 * m_2))
        eps = 1.0e-3
        return eps + (1 - eps) * (lower + upper)

    dx_f = jax.grad(_f)
    dxx_f = jax.grad(dx_f)

    # The points in the supplied grid must be consistent with how
    # the kronecker product is created
    x = transforms["grid"].nodes[:: n_theta_max * n_zeta_max, 0]

    scale_vector1 = (_eval_1D(dx_f, x)) ** -1
    scale_vector2 = (_eval_1D(dxx_f, x)) * scale_vector1

    scale_x1 = scale_vector1[:, None]
    scale_x2 = scale_vector2[:, None]

    # Get differentiation matrices
    # RG: setting the gradient to 0 to save some memory?
    D_rho0 = jax.lax.stop_gradient(cheb_D1(n_rho_max) * scale_x1)
    D_rho0 = jax.lax.stop_gradient(D_rho0)

    D_theta0 = jax.lax.stop_gradient(fourier_diffmat(n_theta_max))

    D_zeta0 = jax.lax.stop_gradient(NFP * D_zeta0)

    I_rho0 = jax.lax.stop_gradient(jnp.eye(n_rho_max))
    I_theta0 = jax.lax.stop_gradient(jnp.eye(n_theta_max))
    I_zeta0 = jax.lax.stop_gradient(jnp.eye(n_zeta_max))

    D_rho = jnp.kron(D_rho0, jnp.kron(I_theta0, I_zeta0))
    D_theta = jnp.kron(I_rho0, jnp.kron(D_theta0, I_zeta0))
    D_zeta = jnp.kron(I_rho0, jnp.kron(I_theta0, D_zeta0))

    D_rho_D_rho = jax.lax.stop_gradient(
        jnp.kron(
            (cheb_D2(n_rho_max) - cheb_D1(n_rho_max) * scale_x2) * scale_x1**2,
            jnp.kron(I_theta0, I_zeta0),
        )
    )

    D_theta_D_theta = D_theta @ D_theta
    D_zeta_D_zeta = D_zeta @ D_zeta

    D_rho_D_theta = D_rho @ D_theta
    D_rho_D_zeta = D_rho @ D_zeta
    D_theta_D_zeta = D_theta @ D_zeta

    n_total = n_rho_max * n_theta_max * n_zeta_max
    # Create the full matrix
    A = jnp.zeros((3 * n_total, 3 * n_total))

    sqrt_g_Q_sup_rho = jnp.zeros((n_total, 3 * n_total))
    sqrt_g_Q_sup_theta = jnp.zeros((n_total, 3 * n_total))
    sqrt_g_Q_sup_zeta = jnp.zeros((n_total, 3 * n_total))

    partial_zeta_Q_rho = jnp.zeros((n_total, 3 * n_total))
    partial_rho_Q_zeta = jnp.zeros((n_total, 3 * n_total))

    partial_theta_Q_rho = jnp.zeros((n_total, 3 * n_total))
    partial_rho_Q_theta = jnp.zeros((n_total, 3 * n_total))

    partial_theta_Q_zeta = jnp.zeros((n_total, 3 * n_total))
    partial_zeta_Q_theta = jnp.zeros((n_total, 3 * n_total))

    # Define field component indices
    rho_idx = slice(0, n_total)
    theta_idx = slice(n_total, 2 * n_total)
    zeta_idx = slice(2 * n_total, 3 * n_total)

    all_idx = slice(0, 3 * n_total)

    g_rr = data["g_rr|PEST"][:, None] / a_N**2
    g_tt = data["g_tt|PEST"][:, None] / a_N**2
    g_zz = data["g_zz|PEST"][:, None] / a_N**2

    g_rt = data["g_rt|PEST"][:, None] / a_N**2
    g_tr = g_rt

    g_rz = data["g_rz|PEST"][:, None] / a_N**2
    # --no-verify g_zr = g_rz

    g_tz = data["g_tz|PEST"][:, None] / a_N**2
    g_zt = g_tz

    j_sup_theta = data["J^theta|PEST"][:, None] * a_N**2 / B_N
    j_sup_zeta = data["J^zeta|PEST"][:, None] * a_N**2 / B_N

    sqrt_g = data["sqrt(g)_PEST"][:, None] / a_N**3
    sqrt_g_r = data["sqrt(g)_r|PEST"][:, None] / a_N**3
    sqrt_g_t = data["sqrt(g)_t|PEST"][:, None] / a_N**3
    sqrt_g_z = data["sqrt(g)_z|PEST"][:, None] / a_N**3

    C_zeta = (sqrt_g_z / sqrt_g) + D_zeta
    C_theta = (sqrt_g_t / sqrt_g) + D_theta
    C_rho = (sqrt_g_r / sqrt_g) + D_rho

    # √g Q^ρ
    sqrt_g_Q_sup_rho = sqrt_g_Q_sup_rho.at[rho_idx, rho_idx].add(
        iota * D_theta + D_zeta
    )

    # √g Q^θ
    sqrt_g_Q_sup_theta = sqrt_g_Q_sup_theta.at[rho_idx, rho_idx].add(
        -iota * D_rho - iota_r
    )
    sqrt_g_Q_sup_theta = sqrt_g_Q_sup_theta.at[rho_idx, theta_idx].add(D_zeta)
    sqrt_g_Q_sup_theta = sqrt_g_Q_sup_theta.at[rho_idx, zeta_idx].add(-D_zeta)

    # √g Q^ζ
    sqrt_g_Q_sup_zeta = sqrt_g_Q_sup_zeta.at[rho_idx, rho_idx].add(-1 * D_rho)
    sqrt_g_Q_sup_zeta = sqrt_g_Q_sup_zeta.at[rho_idx, theta_idx].add(-1 * D_theta)
    sqrt_g_Q_sup_zeta = sqrt_g_Q_sup_zeta.at[rho_idx, zeta_idx].add(1 * D_theta)

    # ∂ᵨ (√g Q_ϑ) rho block
    partial_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, rho_idx].add(
        1
        / sqrt_g
        * (
            g_tr * (iota * D_rho_D_theta + iota_r * D_theta + D_rho_D_zeta)
            - g_tt * (D_rho_D_rho + 2 * iota_r * D_rho + iota_rr)
            - g_tz * D_rho_D_rho
            - g_tr * sqrt_g_r / sqrt_g * (iota * D_theta + D_zeta)
            + g_tt * sqrt_g_r / sqrt_g * (iota_r + iota * D_rho)
            + g_tz * sqrt_g_r / sqrt_g * D_rho
        )
    )

    # ∂ᵨ (√g Q_ϑ) theta block
    partial_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_tt * D_rho_D_zeta
            - g_tz * D_rho_D_theta
            - g_tt * sqrt_g_r / sqrt_g * D_zeta
            + g_tz * sqrt_g_r / sqrt_g * D_theta
        )
    )

    # ∂ᵨ (√g Q_ϑ) zeta block
    partial_rho_Q_theta = partial_rho_Q_theta.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_tt * D_rho_D_zeta
            - g_tz * D_rho_D_theta
            - g_tt * sqrt_g_r / sqrt_g * D_zeta
            + g_tz * sqrt_g_r / sqrt_g * D_theta
        )
    )

    # ∂_θ (√g Qᵨ) rho bloc
    partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, rho_idx].add(
        1
        / sqrt_g
        * (
            g_rr * (iota * D_theta_D_theta + 0 * iota_r * D_theta + D_theta_D_zeta)
            - g_rt * (D_rho_D_theta + 1.0 * iota_r * D_theta)
            - g_rz * D_rho_D_theta
            - g_rr * sqrt_g_t / sqrt_g * (iota * D_theta + D_zeta)
            + g_rt * sqrt_g_t / sqrt_g * (iota_r + iota * D_rho)
            + g_rz * sqrt_g_t / sqrt_g * D_rho
        )
    )

    # ∂_θ (√g Qᵨ) theta block
    partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_rt * D_theta_D_zeta
            - g_rz * D_theta_D_theta
            - g_rt * sqrt_g_t / sqrt_g * D_zeta
            + g_rz * sqrt_g_t / sqrt_g * D_theta
        )
    )

    # ∂_θ (√g Qᵨ) zeta block
    partial_theta_Q_rho = partial_theta_Q_rho.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_rt * D_theta_D_zeta
            - g_rz * D_theta_D_theta
            - g_rt * sqrt_g_t / sqrt_g * D_zeta
            + g_rz * sqrt_g_t / sqrt_g * D_theta
        )
    )

    # rho block
    partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, rho_idx].add(
        1
        / sqrt_g
        * (
            g_rt * (iota * D_theta_D_zeta + 0 * iota_r * D_theta + D_zeta_D_zeta)
            - g_tt * (D_rho_D_zeta + 1 * iota_r * D_zeta)
            - g_tz * D_rho_D_zeta
            - g_rt * sqrt_g_z / sqrt_g * (iota * D_theta + D_zeta)
            + g_tt * sqrt_g_z / sqrt_g * (iota_r + iota * D_rho)
            + g_tz * sqrt_g_z / sqrt_g * D_rho
        )
    )

    # theta block
    partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_tt * D_zeta_D_zeta
            - g_tz * D_theta_D_zeta
            - g_tt * sqrt_g_z / sqrt_g * D_zeta
            + g_tz * sqrt_g_z / sqrt_g * D_theta
        )
    )

    # zeta block
    partial_zeta_Q_theta = partial_zeta_Q_theta.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_tt * D_zeta_D_zeta
            - g_tz * D_theta_D_zeta
            - g_tt * sqrt_g_z / sqrt_g * D_zeta
            + g_tz * sqrt_g_z / sqrt_g * D_theta
        )
    )

    # rhobloc
    partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, rho_idx].add(
        1
        / sqrt_g
        * (
            g_rz * (iota * D_theta_D_theta + 0 * iota_r * D_theta + D_theta_D_zeta)
            - g_tz * (D_rho_D_theta + 1 * iota_r * D_theta)
            - g_zz * D_rho_D_theta
            - g_rz * sqrt_g_t / sqrt_g * (iota * D_theta + D_zeta)
            + g_tz * sqrt_g_t / sqrt_g * (iota_r + iota * D_rho)
            + g_zz * sqrt_g_t / sqrt_g * D_rho
        )
    )

    # theta block
    partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_zt * D_theta_D_zeta
            - g_zz * D_theta_D_theta
            - g_zt * sqrt_g_t / sqrt_g * D_zeta
            + g_zz * sqrt_g_t / sqrt_g * D_theta
        )
    )

    # zeta block
    partial_theta_Q_zeta = partial_theta_Q_zeta.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_zt * D_theta_D_zeta
            - g_zz * D_theta_D_theta
            - g_zt * sqrt_g_t / sqrt_g * D_zeta
            + g_zz * sqrt_g_t / sqrt_g * D_theta
        )
    )

    # rho block
    partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, rho_idx].add(
        g_rr * (iota * D_theta_D_zeta + 0 * iota_r * D_theta + D_zeta_D_zeta)
        - g_rt * (D_rho_D_zeta + 1 * iota_r * D_zeta + 0.0 * iota_rr)
        - g_rz * D_rho_D_zeta
        - g_rr * sqrt_g_z / sqrt_g * (iota * D_theta + D_zeta)
        + g_rt * sqrt_g_z / sqrt_g * (iota_r + iota * D_rho)
        + g_rz * sqrt_g_z / sqrt_g * D_rho
    )

    # theta block
    partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_rt * D_zeta_D_zeta
            - g_rz * D_theta_D_zeta
            - g_rt * sqrt_g_z / sqrt_g * D_zeta
            + g_rz * sqrt_g_z / sqrt_g * D_theta
        )
    )

    # zeta block
    partial_zeta_Q_rho = partial_zeta_Q_rho.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_rt * D_zeta_D_zeta
            - g_rz * D_theta_D_zeta
            - g_rt * sqrt_g_z / sqrt_g * D_zeta
            + g_rz * sqrt_g_z / sqrt_g * D_theta
        )
    )

    # rho block
    partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, rho_idx].add(
        g_rz * (iota * D_rho_D_theta + iota_r * D_theta + D_rho_D_zeta)
        - g_tz * (D_rho_D_rho + 2 * iota_r * D_rho + iota_rr)
        - g_zz * D_rho_D_rho
        - g_rz * sqrt_g_r / sqrt_g * (iota * D_theta + D_zeta)
        + g_tt * sqrt_g_r / sqrt_g * (iota_r + iota * D_rho)
        + g_zz * sqrt_g_r / sqrt_g * D_rho
    )

    # theta block
    partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, theta_idx].add(
        1
        / sqrt_g
        * (
            g_tz * D_rho_D_zeta
            - g_zz * D_rho_D_theta
            - g_tz * sqrt_g_r / sqrt_g * D_zeta
            + g_zz * sqrt_g_r / sqrt_g * D_theta
        )
    )

    # zeta block
    partial_rho_Q_zeta = partial_rho_Q_zeta.at[rho_idx, zeta_idx].add(
        -1
        / sqrt_g
        * (
            g_tz * D_rho_D_zeta
            - g_zz * D_rho_D_theta
            - g_tz * sqrt_g_r / sqrt_g * D_zeta
            + g_zz * sqrt_g_r / sqrt_g * D_theta
        )
    )

    A = A.at[rho_idx, all_idx].add(
        +1.0 * psi_r_sqrt_g * (partial_zeta_Q_rho - partial_rho_Q_zeta)
        - 1.0 * chi_r_sqrt_g * (partial_rho_Q_theta - partial_theta_Q_rho)
        + 1.0 * (j_sup_theta * sqrt_g_Q_sup_zeta - j_sup_zeta * sqrt_g_Q_sup_theta)
    )
    A = A.at[theta_idx, all_idx].add(
        -1.0 * psi_r_sqrt_g * (partial_theta_Q_zeta - partial_zeta_Q_theta)
        + j_sup_zeta * sqrt_g_Q_sup_rho
    )
    A = A.at[zeta_idx, all_idx].add(
        1.0 * chi_r_sqrt_g * (partial_theta_Q_zeta - partial_zeta_Q_theta)
        - j_sup_theta * sqrt_g_Q_sup_rho
    )

    # pressure-driven instability term
    A = A.at[rho_idx, rho_idx].add(
        (p_rr / psi_r - p_r * psi_rr / psi_r**2) + p_r / psi_r * D_rho
    )
    A = A.at[theta_idx, rho_idx].add(p_r / psi_r * D_theta)
    A = A.at[zeta_idx, rho_idx].add(p_r / psi_r * D_zeta)

    gamma0 = 5 / 3

    compress_rho = jnp.zeros((n_total, 3 * n_total))
    compress_theta = jnp.zeros((n_total, 3 * n_total))
    compress_zeta = jnp.zeros((n_total, 3 * n_total))

    # compressional term
    D_rho_C_rho = D_rho @ C_rho
    D_rho_C_theta = D_rho @ C_theta
    D_rho_C_zeta = D_rho @ C_zeta

    compress_rho = compress_rho.at[rho_idx, rho_idx].set(p_r * C_rho + p * D_rho_C_rho)
    compress_rho = compress_rho.at[rho_idx, theta_idx].set(
        p_r * C_theta + p * D_rho_C_theta
    )
    compress_rho = compress_rho.at[rho_idx, zeta_idx].set(
        p_r * C_zeta + p * D_rho_C_zeta
    )

    compress_theta = compress_theta.at[rho_idx, rho_idx].set(p * D_theta @ C_rho)
    compress_theta = compress_theta.at[rho_idx, theta_idx].set(p * D_theta @ C_theta)
    compress_theta = compress_theta.at[rho_idx, zeta_idx].set(p * D_theta @ C_zeta)

    compress_zeta = compress_zeta.at[rho_idx, rho_idx].set(p * D_zeta @ C_rho)
    compress_zeta = compress_zeta.at[rho_idx, theta_idx].set(p * D_zeta @ C_theta)
    compress_zeta = compress_zeta.at[rho_idx, zeta_idx].set(p * D_zeta @ C_zeta)

    A = A.at[rho_idx, all_idx].add(gamma0 * compress_rho)
    A = A.at[theta_idx, all_idx].add(gamma0 * compress_theta)
    A = A.at[zeta_idx, all_idx].add(gamma0 * compress_zeta)

    # apply dirichlet BC to ξ^ρ
    keep_1 = jnp.arange(n_theta_max * n_zeta_max, n_total - n_theta_max * n_zeta_max)
    keep_2 = jnp.arange(n_total, 3 * n_total)
    keep = jnp.concatenate([keep_1, keep_2])

    w, v = jnp.linalg.eig(A[jnp.ix_(keep, keep)])

    data["finite-n lambda"] = w

    return data
