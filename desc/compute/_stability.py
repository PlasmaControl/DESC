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

from desc.backend import eigvals, jax, jit, jnp, vmap

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

    d / d z (g d X / d z) + c * X - lam * f * X = 0, g, f > 0

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

    N_alpha = int(len(jnp.where(jnp.abs(phi) <= 1e-10)[0]))
    N = int(len(phi) / N_alpha)

    B = jnp.reshape(data["|B|"], (N_alpha, 1, N))
    gradpar = jnp.reshape(data["B^zeta"] / data["|B|"], (N_alpha, 1, N))
    dpdpsi = jnp.mean(mu_0 * data["p_r"] / data["psi_r"])

    gds2 = jnp.reshape(
        rho**2
        * (
            data["g^aa"][None, :]
            - 2 * sign_iota * shat / rho * zeta0[:, None] * data["g^ra"][None, :]
            + zeta0[:, None] ** 2 * (shat / rho) ** 2 * data["g^rr"][None, :]
        ),
        (N_alpha, N_zeta0, N),
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
            (N_alpha, N_zeta0, N),
        )
    )

    h = phi[1] - phi[0]

    A = jnp.zeros((N_alpha, N_zeta0, N - 2, N - 2))

    i = jnp.arange(N_alpha)[:, None, None, None]
    l = jnp.arange(N_zeta0)[None, :, None, None]
    j = jnp.arange(N - 2)[None, None, :, None]
    k = jnp.arange(N - 2)[None, None, None, :]

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

    d / d z (g d X / d z) + c * X - lam * f * X = 0, g, f > 0

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
    N = int(len(phi) / N_alpha)

    B = jnp.reshape(data["|B|"], (N_alpha, 1, N))
    gradpar = jnp.reshape(data["B^zeta"] / data["|B|"], (N_alpha, 1, N))
    dpdpsi = jnp.mean(mu_0 * data["p_r"] / data["psi_r"])

    gds2 = jnp.reshape(
        rho**2
        * (
            data["g^aa"][None, :]
            - 2 * sign_iota * shat / rho * zeta0[:, None] * data["g^ra"][None, :]
            + zeta0[:, None] ** 2 * (shat / rho) ** 2 * data["g^rr"][None, :]
        ),
        (N_alpha, N_zeta0, N),
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
            (N_alpha, N_zeta0, N),
        )
    )

    h = phi[1] - phi[0]

    i = jnp.arange(N_alpha)[:, None, None, None]
    l = jnp.arange(N_zeta0)[None, :, None, None]
    j = jnp.arange(N - 2)[None, None, :, None]
    k = jnp.arange(N - 2)[None, None, None, :]

    A = jnp.zeros((N_alpha, N_zeta0, N - 2, N - 2))
    B = jnp.zeros((N_alpha, N_zeta0, N - 2, N - 2))
    B_inv = jnp.zeros((N_alpha, N_zeta0, N - 2, N - 2))

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

    d / d z (g d X / d z) + c * X = 0, g, f > 0

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
    N = int(len(phi) / N_alpha)

    B = jnp.reshape(data["|B|"], (N_alpha, 1, N))
    gradpar = jnp.reshape(data["B^zeta"] / data["|B|"], (N_alpha, 1, N))
    dpdpsi = jnp.mean(mu_0 * data["p_r"] / data["psi_r"])

    gds2 = jnp.reshape(
        rho**2
        * (
            data["g^aa"][None, :]
            - 2 * sign_iota * shat / rho * zeta0[:, None] * data["g^ra"][None, :]
            + zeta0[:, None] ** 2 * (shat / rho) ** 2 * data["g^rr"][None, :]
        ),
        (N_alpha, N_zeta0, N),
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
            (N_alpha, N_zeta0, N),
        )
    )

    h = phi[1] - phi[0]

    # g_half on half grid points, c_full on full grid points
    g_half = (g[:, :, 1:] + g[:, :, :-1]) / 2
    c_full = c[:, :, :-1]

    i = jnp.arange(N_alpha)[:, None, None]
    j = jnp.arange(N_zeta0)[None, :, None]
    k = jnp.arange(N - 1)[None, None, :]

    X = jnp.zeros((N_alpha, N_zeta0, N - 1))
    X = X.at[i, j, k].set(phi[k])

    Y = jnp.zeros((N_alpha, N_zeta0))
    eps = 5e-3  # slope of the test function
    Yp = eps * jnp.ones((N_alpha, N_zeta0))

    @jax.jit
    def integrator(carry, x):
        arr, arr_d = carry
        g_element, c_element = x
        # Update the array (Y) and its derivative
        arr_updated = (g_element * arr + arr_d * h) / g_element
        arr_d_updated = arr_d - c_element * arr_updated * h
        # Calculate the sign of the product of arr and arr_updated
        sign_product = jnp.sign(arr * arr_updated)

        return (arr_updated, arr_d_updated), (arr_updated, sign_product)

    @jax.jit
    def cumulative_update_jit(arr, arr_d, g_half, c_full):
        _, scan_output = jax.lax.scan(integrator, (arr, arr_d), (g_half, c_full))
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


@register_compute_fun(
    name="effective ballooning potential",
    label="V_{ball}",
    units=" ",
    units_long=" ",
    description="ideal ballooning effect potential"
    + "requires data along a field line",
    dim=1,
    params=["Psi"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "a",
        "rho",
        "p_r",
        "psi_r",
        "phi",
        "iota",
        "psi",
        "g^aa",
        "g^ra",
        "g^rr",
        "g^aa_z",
        "g^aa_t",
        "g^aa_zz",
        "g^aa_tt",
        "g^aa_tz",
        "g^ra_z",
        "g^ra_t",
        "g^ra_zz",
        "g^ra_tt",
        "g^ra_tz",
        "g^rr_z",
        "g^rr_t",
        "g^rr_zz",
        "g^rr_tt",
        "g^rr_tz",
        "cvdrift",
        "cvdrift0",
        "lambda_t",
        "lambda_z",
        "lambda_tt",
        "lambda_zz",
        "lambda_tz",
        "|B|",
        "|B|_z",
        "|B|_t",
        "|B|_zz",
        "|B|_tt",
        "|B|_tz",
        "B^zeta",
        "B^zeta_t",
        "B^zeta_z",
        "B^zeta_tt",
        "B^zeta_zz",
        "B^zeta_tz",
    ],
)
def _effective_ballooning_potential(params, transforms, profiles, data, *kwargs):
    """
    Effective ballooning potential.

    We can rewrite the ideal ballooning equation
    to a Schrodinger-like equation with an effective potential
    V

    d^2 X / d z^2 + V * X - lam * b * X = 0, b > 0.

    bmag = B/B_N,
    kappa = b dot grad b
    g = a_N^3 * B_N * (b dot grad zeta) * |grad alpha|^2 / B,
    c = a_N/B_N * (1/ b dot grad zeta) * dpsi/drho * dp/dpsi
        * (b cross kappa) dot grad alpha/ B**2,
    V = d^2 g/d z^2 - d c/d z

    Parameters
    ----------
    eq :  Input equilibrium object
    N : resolution
    """
    rho = data["rho"]

    psi_b = params["Psi"] / (2 * jnp.pi)
    a_N = data["a"]
    B_N = 2 * psi_b / a_N**2

    N_zeta0 = int(11)
    zeta0 = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, N_zeta0)

    iota = data["iota"]
    psi = data["psi"]
    sign_psi = jnp.sign(psi[-1])
    sign_iota = jnp.sign(iota[-1])

    temp_fac1 = 1 / (1 + data["lambda_t"])
    temp_fac2 = (iota - data["lambda_z"]) * temp_fac1

    g_sup_rr_z0 = data["g^rr_z"] + data["g^rr_t"] * temp_fac2
    g_sup_rr_zz0 = (
        data["g^rr_zz"]
        + 2 * data["g^rr_tz"] * temp_fac2
        - data["g^rr_t"]
        * (
            data["lambda_zz"] * temp_fac1
            + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
            + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
        )
        + data["g^rr_tt"] * temp_fac2**2
    )

    g_sup_ra_z0 = data["g^ra_z"] + data["g^ra_t"] * temp_fac2
    g_sup_ra_zz0 = (
        data["g^ra_zz"]
        + 2 * data["g^ra_tz"] * temp_fac2
        - data["g^ra_t"]
        * (
            data["lambda_zz"] * temp_fac1
            + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
            + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
        )
        + data["g^ra_tt"] * temp_fac2**2
    )

    g_sup_aa_z0 = data["g^aa_z"] + data["g^aa_t"] * temp_fac2
    g_sup_aa_zz0 = (
        data["g^aa_zz"]
        + 2 * data["g^aa_tz"] * temp_fac2
        - data["g^aa_t"]
        * (
            data["lambda_zz"] * temp_fac1
            + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
            + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
        )
        + data["g^aa_tt"] * temp_fac2**2
    )

    modB = data["|B|"]
    modB_z0 = data["|B|_z"] + data["|B|_t"] * temp_fac2
    modB_zz0 = (
        data["|B|_zz"]
        + 2 * data["|B|_tz"] * temp_fac2
        - data["|B|_t"]
        * (
            data["lambda_zz"] * temp_fac1
            + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
            + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
        )
        + data["|B|_tt"] * temp_fac2**2
    )

    B_sup_zeta = data["B^zeta"]
    B_sup_zeta_z0 = data["B^zeta_z"] + temp_fac2 * data["B^zeta_t"]
    B_sup_zeta_zz0 = (
        data["B^zeta_zz"]
        + 2 * data["B^zeta_tz"] * temp_fac2
        - data["B^zeta_t"]
        * (
            data["lambda_zz"] * temp_fac1
            + 2 * temp_fac2 * data["lambda_tz"] * temp_fac1
            + temp_fac2**2 * temp_fac1 * data["lambda_tt"]
        )
        + data["B^zeta_tt"] * temp_fac2**2
    )

    gradpar = B_sup_zeta / modB
    gradpar_z = B_sup_zeta_z0 / modB - modB_z0 * B_sup_zeta / modB**2
    gradpar_zz = (
        B_sup_zeta_zz0 / modB
        - 2 * modB_z0 * B_sup_zeta_z0 / modB**2
        - modB_z0 * B_sup_zeta / modB**2
        + modB_zz0 * B_sup_zeta / modB**2
    )

    gds2 = data["g^aa"] + 2 * zeta0 * data["g^ra"] + zeta0**2 * data["g^rr"]
    gds2_z = rho**2 * (
        g_sup_aa_z0 - 2 * sign_iota * zeta0 * g_sup_ra_z0 + zeta0**2 * g_sup_rr_z0
    )
    gds2_zz = rho**2 * (
        g_sup_aa_zz0 - 2 * sign_iota * zeta0 * g_sup_ra_zz0 + zeta0**2 * g_sup_rr_zz0
    )

    dpdpsi = mu_0 * data["p_r"] / data["psi_r"]

    g = a_N**3 * B_N * gds2 / modB * gradpar
    g_z = (
        a_N**3
        * B_N
        * (
            (gds2_z * gradpar + gds2 * gradpar_z) / modB
            - gds2 * gradpar * modB_z0 / modB**2
        )
    )
    g_zz = (
        a_N**3
        * B_N
        * (
            (gds2_zz * gradpar + 2 * gds2_z * gradpar_z + gds2 * gradpar_zz) / modB
            - (gds2_z * gradpar + gds2 * gradpar_z) * modB_z0 / modB**2
            - (
                gds2_z * gradpar * modB_z0
                + gds2 * gradpar_z * modB_z0
                + gds2 * gradpar * modB_zz0
            )
            / modB**2
            + 2 * (gds2_z * gradpar * modB_z0**2) / modB**3
        )
    )

    c = (
        a_N
        * 2
        / data["B^zeta"]
        * rho
        * sign_psi
        * dpdpsi
        * (data["cvdrift"] + zeta0 * data["cvdrift0"])
    )

    V = c / g + 1 / 4 * g_z**2 / g**2 - 1 / 2 * g_zz / g

    data["effective ballooning potential"] = V
    return data
