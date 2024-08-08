"""Compute functions for Mercier stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from scipy.constants import mu_0

from desc.backend import eigvals, jit, jnp, scan, vmap

from .data_index import register_compute_fun
from .utils import dot, surface_integrals_map


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
    name="ideal_ball_gamma1",
    label="\\gamma^2",
    units=" ",
    units_long=" ",
    description="ideal ballooning growth rate" + "requires data along a field line",
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
        "phi",
        "iota",
        "iota_r",
        "psi",
        "psi_r",
        "rho",
    ],
)
def _ideal_ballooning_gamma1(params, transforms, profiles, data, *kwargs):
    """
    Ideal-ballooning growth rate finder.

    This function uses a finite-difference method
    to calculate the maximum growth rate against the
    infinite-n ideal ballooning mode. The equation being solved is

    d/dζ(g dX/dζ) + c X = lam f X, g, f > 0,

    where

    𝛋 = 𝐛 ⋅∇ 𝐛
    g = a_N^3 * B_N * (b ⋅∇ζ) * |∇α|², / B,
    c = a_N/B_N * (1/ b ⋅∇ζ) * dψ/dρ * dp/dψ * (b × 𝛋) ⋅|∇α|/ B**2,
    f = a_N * B_N^3 *|∇α|² / B^3 * (1/ b ⋅∇ζ) ,

    are needed along a field line to solve the ballooning equation once.

    To obtain the parameters g, c, and f, we need a set of parameters
    provided in the list ``data`` above. Here's a description of
    these parameters:

    - a: minor radius of the device
    - g^aa: |grad alpha|^2, field line bending term
    - g^ra: (grad alpha dot grad rho) integrated local shear
    - g^rr: |grad rho|^2 flux expansion term
    - cvdrift: geometric factor of the curvature drift
    - cvdrift0: geoetric factor of curvature drift 2
    - |B|: magnitude of the magnetic field
    - B^zeta: inverse of the jacobian
    - p_r: dp/drho, pressure gradient
    - phi: coordinate describing the position in the toroidal angle
    along a field line
    """
    rho = data["rho"]

    psi_b = params["Psi"] / (2 * jnp.pi)
    a_N = data["a"]
    B_N = 2 * psi_b / a_N**2

    N_zeta0 = int(11)
    # up-down symmetric equilibria only
    zeta0 = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, N_zeta0)

    iota = data["iota"]
    shat = -rho / iota * data["iota_r"]
    psi = data["psi"]
    sign_psi = jnp.sign(psi[-1])
    sign_iota = jnp.sign(iota[-1])

    phi = data["phi"]

    N_alpha = int(8)
    N_zeta = int(len(phi) / N_alpha)

    B = jnp.reshape(data["|B|"], (N_alpha, 1, N_zeta))
    gradpar = jnp.reshape(data["B^zeta"] / data["|B|"], (N_alpha, 1, N_zeta))
    dpdpsi = jnp.mean(mu_0 * data["p_r"] / data["psi_r"])

    gds2 = jnp.reshape(
        rho**2
        * (
            data["g^aa"][None, :]
            - 2 * sign_iota * shat / rho * zeta0[:, None] * data["g^ra"][None, :]
            + zeta0[:, None] ** 2 * (shat / rho) ** 2 * data["g^rr"][None, :]
        ),
        (N_alpha, N_zeta0, N_zeta),
    )

    f = a_N**3 * B_N * gds2 / B**3 * 1 / gradpar
    g = a_N**3 * B_N * gds2 / B * gradpar
    g_half = (g[:, :, 1:] + g[:, :, :-1]) / 2
    c = (
        1
        * a_N**3
        * B_N
        * jnp.reshape(
            2
            / data["B^zeta"][None, :]
            * sign_psi
            * rho**2
            * dpdpsi
            * (
                data["cvdrift"][None, :]
                - shat / rho * zeta0[:, None] * data["cvdrift0"][None, :]
            ),
            (N_alpha, N_zeta0, N_zeta),
        )
    )

    h = phi[1] - phi[0]

    A = jnp.zeros((N_alpha, N_zeta0, N_zeta - 2, N_zeta - 2))

    i = jnp.arange(N_alpha)[:, None, None, None]
    l = jnp.arange(N_zeta0)[None, :, None, None]
    j = jnp.arange(N_zeta - 2)[None, None, :, None]
    k = jnp.arange(N_zeta - 2)[None, None, None, :]

    A = A.at[i, l, j, k].set(
        g_half[i, l, k] / f[i, l, k] * 1 / h**2 * (j - k == -1)
        + (
            -(g_half[i, l, j + 1] + g_half[i, l, j]) / f[i, l, j + 1] * 1 / h**2
            + c[i, l, j + 1] / f[i, l, j + 1]
        )
        * (j - k == 0)
        + g_half[i, l, j] / f[i, l, j + 1] * 1 / h**2 * (j - k == 1)
    )

    w = eigvals(jnp.where(jnp.isfinite(A), A, 0))

    lam = jnp.real(jnp.max(w, axis=(2,)))

    lam = jnp.max(lam)

    data["ideal_ball_gamma1"] = lam

    return data


@register_compute_fun(
    name="ideal_ball_gamma2",
    label="\\gamma^2",
    units=" ",
    units_long=" ",
    description="ideal ballooning growth rate" + "requires data along a field line",
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
        "phi",
        "iota",
        "iota_r",
        "psi",
        "psi_r",
        "rho",
    ],
)
def _ideal_ballooning_gamma2(params, transforms, profiles, data, *kwargs):
    """
    Ideal-ballooning growth rate finder.

    This function uses a finite-difference method
    to calculate the maximum growth rate against the
    infinite-n ideal ballooning mode. The equation being solved is

    d/dζ(g dX/dζ) + c X = lam f X, g, f > 0

    where

    𝛋 = 𝐛 ⋅∇ 𝐛
    g = a_N^3 * B_N * (b ⋅∇ζ) * |∇α|², / B,
    c = a_N/B_N * (1/ b ⋅∇ζ) * dψ/dρ * dp/dψ * (b × 𝛋) ⋅|∇α|/ B**2,
    f = a_N * B_N^3 *|∇α|² / B^3 * (1/ b ⋅∇ζ) ,
    are needed along a field line to solve the ballooning equation once.

    To obtain the parameters g, c, and f, we need a set of parameters
    provided in the list ``data`` above. Here's a description of
    these parameters:

    - a: minor radius of the device
    - g^aa: |grad alpha|^2, field line bending term
    - g^ra: (grad alpha dot grad rho) integrated local shear
    - g^rr: |grad rho|^2 flux expansion term
    - cvdrift: geometric factor of the curvature drift
    - cvdrift0: geoetric factor of curvature drift 2
    - |B|: magnitude of the magnetic field
    - B^zeta: inverse of the jacobian
    - p_r: dp/drho, pressure gradient
    - phi: coordinate describing the position in the toroidal angle
    along a field line
    """
    rho = data["rho"]

    psi_b = params["Psi"] / (2 * jnp.pi)
    a_N = data["a"]
    B_N = 2 * psi_b / a_N**2

    N_zeta0 = int(15)
    # up-down symmetric equilibria only
    zeta0 = jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, N_zeta0)

    iota = data["iota"]
    shat = -rho / iota * data["iota_r"]
    psi = data["psi"]
    sign_psi = jnp.sign(psi[-1])
    sign_iota = jnp.sign(iota[-1])

    phi = data["phi"]

    N_alpha = int(8)
    N_zeta = int(len(phi) / N_alpha)

    B = jnp.reshape(data["|B|"], (N_alpha, 1, N_zeta))
    gradpar = jnp.reshape(data["B^zeta"] / data["|B|"], (N_alpha, 1, N_zeta))
    dpdpsi = jnp.mean(mu_0 * data["p_r"] / data["psi_r"])

    gds2 = jnp.reshape(
        rho**2
        * (
            data["g^aa"][None, :]
            - 2 * sign_iota * shat / rho * zeta0[:, None] * data["g^ra"][None, :]
            + zeta0[:, None] ** 2 * (shat / rho) ** 2 * data["g^rr"][None, :]
        ),
        (N_alpha, N_zeta0, N_zeta),
    )

    f = a_N**3 * B_N * gds2 / B**3 * 1 / gradpar
    g = a_N**3 * B_N * gds2 / B * gradpar
    g_half = (g[:, :, 1:] + g[:, :, :-1]) / 2
    c = (
        1
        * a_N**3
        * B_N
        * jnp.reshape(
            2
            / data["B^zeta"][None, :]
            * sign_psi
            * rho**2
            * dpdpsi
            * (
                data["cvdrift"][None, :]
                - shat / rho * zeta0[:, None] * data["cvdrift0"][None, :]
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
    B = jnp.zeros((N_alpha, N_zeta0, N_zeta - 2, N_zeta - 2))
    B_inv = jnp.zeros((N_alpha, N_zeta0, N_zeta - 2, N_zeta - 2))

    A = A.at[i, l, j, k].set(
        g_half[i, l, k] * 1 / h**2 * (j - k == -1)
        + (-(g_half[i, l, j + 1] + g_half[i, l, j]) * 1 / h**2 + c[i, l, j + 1])
        * (j - k == 0)
        + g_half[i, l, j] * 1 / h**2 * (j - k == 1)
    )

    B = B.at[i, l, j, k].set(jnp.sqrt(f[i, l, j + 1]) * (j - k == 0))
    B_inv = B_inv.at[i, l, j, k].set(1 / jnp.sqrt(f[i, l, j + 1]) * (j - k == 0))

    A_redo = B_inv @ A @ jnp.transpose(B_inv, axes=(0, 1, 3, 2))

    w, _ = jnp.linalg.eigh(A_redo)

    lam = jnp.real(jnp.max(w, axis=(2,)))

    lam = jnp.max(lam)

    data["ideal_ball_gamma2"] = lam

    return data


@register_compute_fun(
    name="Newcomb_metric",
    label="\\mathrm{Nwecomb-metric}",
    units=" ",
    units_long=" ",
    description="A measure of Newcomb's distance from marginality",
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
        "phi",
        "iota",
        "iota_r",
        "psi",
        "psi_r",
        "rho",
    ],
)
def _Newcomb_metric(params, transforms, profiles, data, *kwargs):
    """
    Ideal-ballooning growth rate proxy.

    This function uses a finite-difference method to integrate the
    marginal stability ideal-ballooning equation

    d/dζ(g dX/dζ) + c X = 0, g > 0

    using the Newcomb's stability criterion

    where

    kappa = b dot grad b
    g = a_N^3 * B_N * (b dot grad zeta) * |grad alpha|^2 / B,
    c = a_N/B_N * (1/ b dot grad zeta) * dpsi/drho * dp/dpsi
        * (b cross kappa) dot grad alpha/ B**2,
    f = a_N * B_N^3 *|grad alpha|^2 / bmag^3 * 1/(b dot grad zeta),
    are needed along a field line to solve the ballooning equation once.

    To obtain the parameters g, c, and f, we need a set of parameters
    provided in the list ``data`` above. Here's a description of
    these parameters:

    - a: minor radius of the device
    - g^aa: |grad alpha|^2, field line bending term
    - g^ra: (grad alpha dot grad rho) integrated local shear
    - g^rr: |grad rho|^2 flux expansion term
    - cvdrift: geometric factor of the curvature drift
    - cvdrift0: geoetric factor of curvature drift 2
    - |B|: magnitude of the magnetic field
    - B^zeta: inverse of the jacobian
    - p_r: dp/drho, pressure gradient
    - psi_r: radial gradient of the toroidal flux
    - phi: coordinate describing the position in the toroidal angle
    along a field line

    Here's how we define the Newcomb metric:
    If zero crossing is at -inf (root finder failed), use the Y coordinate
    as a metric of stability else use the zero-crossing point on the X-axis
    as the metric
    """
    rho = data["rho"]

    psi_b = params["Psi"] / (2 * jnp.pi)
    a_N = data["a"]
    # a_N is 0.4
    B_N = 2 * psi_b / a_N**2

    N_zeta0 = int(11)
    # up-down symmetric equilibria only
    zeta0 = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, N_zeta0)

    iota = data["iota"]
    shat = -rho / iota * data["iota_r"]
    psi = data["psi"]
    sign_psi = jnp.sign(psi[-1])
    sign_iota = jnp.sign(iota[-1])

    phi = data["phi"]

    # Count the number of 0s in phi (= number of field lines)
    N_alpha = int(8)
    N_zeta = int(len(phi) / N_alpha)

    B = jnp.reshape(data["|B|"], (N_alpha, 1, N_zeta))
    gradpar = jnp.reshape(data["B^zeta"] / data["|B|"], (N_alpha, 1, N_zeta))
    dpdpsi = jnp.mean(mu_0 * data["p_r"] / data["psi_r"])

    gds2 = jnp.reshape(
        rho**2
        * (
            data["g^aa"][None, :]
            - 2 * sign_iota * shat / rho * zeta0[:, None] * data["g^ra"][None, :]
            + zeta0[:, None] ** 2 * (shat / rho) ** 2 * data["g^rr"][None, :]
        ),
        (N_alpha, N_zeta0, N_zeta),
    )

    g = a_N**3 * B_N * gds2 / B * gradpar
    g_half = (g[:, :, 1:] + g[:, :, :-1]) / 2
    c = (
        1
        * a_N**3
        * B_N
        * jnp.reshape(
            2
            / data["B^zeta"][None, :]
            * sign_psi
            * rho**2
            * dpdpsi
            * (
                data["cvdrift"][None, :]
                - shat / rho * zeta0[:, None] * data["cvdrift0"][None, :]
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
    eps = 5e-3  # slope of the test function
    Yp = eps * jnp.ones((N_alpha, N_zeta0))

    @jit
    def integrator(carry, x):
        arr, arr_d = carry
        g_element, c_element = x
        # Update the array (Y) and its derivative
        arr_updated = (g_element * arr + arr_d * h) / g_element
        arr_d_updated = arr_d - c_element * arr_updated * h
        # Calculate the sign of the product of arr and arr_updated
        sign_product = jnp.sign(arr * arr_updated)

        return (arr_updated, arr_d_updated), (arr_updated, sign_product)

    @jit
    def cumulative_update_jit(arr, arr_d, g_half, c_full):
        _, scan_output = scan(integrator, (arr, arr_d), (g_half, c_full))
        Y, sign_product = scan_output
        # Create a mask where sign_product is negative
        negative_mask = sign_product < 0
        # Find the first occurrence of a negative value
        first_negative_index = jnp.argmax(negative_mask)
        # Use where to return 0 if there are no negative values
        first_negative_index = jnp.where(
            jnp.any(negative_mask), first_negative_index, -1
        )
        # This factor will give us the exact X point of intersection
        lin_interp_factor = jnp.where(
            first_negative_index != -1,
            1 / (1 - Y[first_negative_index + 1] / Y[first_negative_index]),
            0,
        )

        return Y, first_negative_index, lin_interp_factor

    # Vectorize over the first two dimensions
    vectorized_cumulative_update = jit(vmap(vmap(cumulative_update_jit)))
    Y, first_negative_indices, lin_interp_factors = vectorized_cumulative_update(
        Y, Yp, g_half, c_full
    )

    X0 = jnp.zeros((N_alpha, N_zeta0))
    Y0 = jnp.zeros((N_alpha, N_zeta0))
    i0 = jnp.arange(N_alpha)[:, None]
    j0 = jnp.arange(N_zeta0)[None, :]
    X0 = X0.at[i0, j0].set(
        X[i0, j0, first_negative_indices[i0, j0]] + lin_interp_factors[i0, j0] * h
    )

    # value of Y at the zero-crossing. If there is a zero-crossing,
    # Y0 is very small, if not it's the Y intercept because first
    # negative index will be -1
    Y0 = Y0.at[i0, j0].set(Y[i0, j0, first_negative_indices[i0, j0]])
    Y0 = jnp.where(first_negative_indices != -1, 0, Y0)

    data2 = jnp.min(1 + jnp.tanh(jnp.cbrt(Y0[:, :])))

    data3 = 2 - data2

    data["Newcomb_metric"] = data3

    return data
