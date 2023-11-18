"""Compute functions for Mercier stability objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

import numpy as np
import scipy
from scipy.constants import mu_0
from scipy.integrate import simpson as simps

from desc.backend import eigvals, jnp
from desc.grid import Grid

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
    name="ideal_ball_gamma",
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
        "psi_r",
        "phi",
        "iota",
        "psi",
        "rho",
    ],
)
def _ideal_ballooning_gamma(params, transforms, profiles, data, *kwargs):
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
    - psi_r: radial gradient of the toroidal flux
    - phi: coordinate describing the position in the toroidal angle
    along a field line
    """
    rho = data["rho"]

    psi_b = params["Psi"] / (2 * np.pi)
    a_N = data["a"]
    B_N = 2 * psi_b / a_N**2

    N_zeta0 = int(10)
    # up-down symmetric equilibria only
    zeta0 = jnp.linspace(0, np.pi, N_zeta0)

    iota = data["iota"]
    psi = data["psi"]
    sign_psi = jnp.sign(psi[-1])
    sign_iota = jnp.sign(iota[-1])

    phi = data["phi"]
    N = len(phi)

    gradpar = data["B^zeta"] / data["|B|"]
    dpdpsi = mu_0 * data["p_r"] / data["psi_r"]

    gds2 = rho**2 * (
        data["g^aa"][None, :]
        - 2 * sign_iota * zeta0[:, None] * data["g^ra"][None, :]
        + zeta0[:, None] ** 2 * data["g^rr"][None, :]
    )
    dpdpsi = mu_0 * data["p_r"] / data["psi_r"]

    f = a_N * B_N**3 * gds2 / data["|B|"][None, :] ** 3 * 1 / gradpar[None, :]
    g = a_N**3 * B_N * gds2 / data["|B|"][None, :] * gradpar[None, :]
    g_half = (g[:, 1:] + g[:, :-1]) / 2
    c = (
        a_N
        * 2
        / data["B^zeta"][None, :]
        * rho
        * sign_psi
        * dpdpsi
        * (data["cvdrift"][None, :] + zeta0[:, None] * data["cvdrift0"][None, :])
    )
    h = (phi[-1] - phi[0]) / (N - 1)

    i = jnp.arange(N_zeta0)[:, None, None]
    j = jnp.arange(N - 2)[None, :, None]
    k = jnp.arange(N - 2)[None, None, :]

    A = jnp.zeros((N_zeta0, N - 2, N - 2))

    A = A.at[i, j, k].set(
        g_half[i, k] / f[i, k] * 1 / h**2 * (j - k == -1)
        + (
            -(g_half[i, j + 1] + g_half[i, j]) / f[i, j + 1] * 1 / h**2
            + c[i, j + 1] / f[i, j + 1]
        )
        * (j - k == 0)
        + g_half[i, j] / f[i, j + 1] * 1 / h**2 * (j - k == 1)
    )

    w = eigvals(jnp.where(jnp.isfinite(A), A, 0))

    lam = jnp.real(jnp.max(w))
    data["ideal_ball_gamma"] = lam * (lam >= -0.001)

    return data


def _gamma_ideal_ballooning_FD2(eq):
    """
    Ideal-ballooning growth rate finder.

    A finite-difference solver to calculate the maximum
    growth rate against the infinite-n ideal ballooning mode.
    The problem to solve is

    d^2 X / d z^2 + V * X - lam * b * X = 0, b > 0.

    bmag = B/B_N,
    kappa = b dot grad b
    g = a_N^3 * B_N * (b dot grad zeta) * |grad alpha|^2 / B,
    c = a_N/B_N * (1/ b dot grad zeta) * dpsi/drho * dp/dpsi
        * (b cross kappa) dot grad alpha/ B**2,
    f = a_N * B_N^3 *|grad alpha|^2 / bmag^3
        * 1/(b dot grad zeta)

    Parameters
    ----------
    eq :  Input equilibrium object
    N : resolution
    """
    ns = 1
    nalpha = 1
    s = np.linspace(0.9, 1.00, ns)
    rho = np.sqrt(s)

    alpha = np.linspace(0, np.pi, nalpha)
    zeta_0 = 0.0
    eq_keys = ["iota", "rho", "psi", "a"]

    data_eq = eq.compute(eq_keys)

    iota = np.interp(rho, data_eq["rho"], data_eq["iota"])
    psi = data_eq["psi"]
    sign_psi = np.sign(psi[-1])
    sign_iota = np.sign(iota[-1])

    a_N = data_eq["a"]
    B_N = 2 * sign_psi * psi[-1] / a_N**2

    nperiod = 3
    # Number of toroidal turns
    ntor = 2 * nperiod - 1
    N = 2 * (2 * eq.M_grid * eq.N_grid) * ntor + 1

    rho_full = np.ones(
        int(N * ns * nalpha),
    )
    theta_full = np.zeros(
        int(N * ns * nalpha),
    )
    zeta_full = np.zeros(
        int(N * ns * nalpha),
    )

    zeta = np.linspace(-ntor * np.pi, ntor * np.pi, N)

    for i in range(ns):
        for j in range(nalpha):
            rho_full[i * N : (i + 1) * N] = rho[i] * np.ones(
                N,
            )
            zeta_full[i * N : (i + 1) * N] = zeta
            theta_full[i * N : (i + 1) * N] = (
                alpha[j]
                * np.ones(
                    N,
                )
                + iota[i] * zeta
            )

    C0 = np.vstack([rho_full, theta_full, zeta_full]).T
    coords = eq.compute_theta_coords(C0, tol=1e-10, maxiter=50)
    grid = Grid(coords, sort=False)

    data_names = [
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
        "p_r",
        "psi_r",
    ]

    data = eq.compute(data_names, grid)

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

    gds2 = data["g^aa"] + 2 * zeta_0 * data["g^ra"] + zeta_0**2 * data["g^rr"]
    gds2_z = rho**2 * (
        g_sup_aa_z0 - 2 * sign_iota * zeta_0 * g_sup_ra_z0 + zeta_0**2 * g_sup_rr_z0
    )
    gds2_zz = rho**2 * (
        g_sup_aa_zz0
        - 2 * sign_iota * zeta_0 * g_sup_ra_zz0
        + zeta_0**2 * g_sup_rr_zz0
    )

    dpdpsi = mu_0 * data["p_r"] / data["psi_r"]

    f = a_N * B_N**3 * gds2 / modB**3 * 1 / gradpar
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
        * (data["cvdrift"] + zeta_0 * data["cvdrift0"])
    )

    V = c / g + 1 / 4 * g_z**2 / g**2 - 1 / 2 * g_zz / g

    b = f / g

    # grid spacing
    h = 2 * ntor * np.pi / (N - 1)

    sub_diag = (
        1
        / h**2
        * np.ones(
            N - 3,
        )
    )
    sup_diag = sub_diag
    diag = (
        -2
        / h**2
        * np.ones(
            N - 2,
        )
    )

    D2 = np.diag(sub_diag, -1) + np.diag(diag + V[1:-1], 0) + np.diag(sup_diag, 1)

    b = f / g
    M = np.diag(b[1:-1], 0)

    vguess = np.exp(-np.abs(zeta[1:-1]) / 2)
    w, v = scipy.sparse.linalg.eigs(D2, k=1, M=M, sigma=2.0, v0=vguess, OPpart="r")

    # variational refinement here
    X = np.zeros((N,))
    dX = np.zeros((N,))

    X[1:-1] = np.reshape(v[:, 0].real, (-1,)) / np.max(np.abs(v[:, 0].real))

    X[0] = 0.0
    X[-1] = 0.0

    dX[0] = (-1.5 * X[0] + 2 * X[1] - 0.5 * X[2]) / h
    dX[1] = (X[2] - X[0]) / (2 * h)

    dX[-2] = (X[-1] - X[-3]) / (2 * h)
    dX[-1] = (0.5 * X[-3] - 2 * X[-2] + 1.5 * 0.0) / (h)

    dX[2:-2] = 2 / (3 * h) * (X[3:-1] - X[1:-3]) - (X[4:] - X[0:-4]) / (12 * h)

    Y0 = -1 * dX**2 + V * X**2
    Y1 = b * X**2
    lam = simps(Y0) / simps(Y1)

    return lam


def _gamma_ideal_ballooning_Fourier(eq):
    """
    Ideal-ballooning growth rate finder.

    This function uses a pseudospectral Fourier technique to
    calculate the maximum growth rate against the infinite-n
    ideal ballooning mode

    d^2 X / d z^2 + V * X - lam * b * X = 0, b > 0

    where

    kappa = b dot grad b
    g = a_N^3 * B_N* (b dot grad zeta) * |grad alpha|^2 / B,
    c = a_N/B_N * (1/ b dot grad zeta) * dpsi/drho * dp/dpsi
        * (b cross kappa) dot grad alpha/ B**2,
    f = a_N * B_N^3 *|grad alpha|^2 / bmag^3 * 1/(b dot grad zeta)

    Parameters
    ----------
    eq :  Input equilibrium object
    D2 = jax.scipy.linalg.toeplitz(col1, row1) + np.diag(V, 0)
    x = jax.scipy.linalg.eikh(D2, b * np.eye(N))
    """
    ns = 1
    nalpha = 1
    s = np.linspace(0.9, 1.00, ns)
    rho = np.sqrt(s)
    alpha = np.linspace(0, np.pi, nalpha)
    zeta_0 = 0.0

    eq_keys = ["iota", "rho", "psi", "a"]

    data_eq = eq.compute(eq_keys)

    iota = np.interp(rho, data_eq["rho"], data_eq["iota"])
    psi = data_eq["psi"]
    sign_psi = np.sign(psi[-1])
    sign_iota = np.sign(iota[-1])

    a_N = data_eq["a"]
    B_N = 2 * sign_psi * psi[-1] / a_N**2

    nperiod = 3
    ntor = 2 * nperiod - 1  # Number of toroidal turns
    N = int((2 * eq.M_grid * eq.N_grid) * ntor)

    rho_full = np.ones(
        int(N * ns * nalpha),
    )
    theta_full = np.zeros(
        int(N * ns * nalpha),
    )
    zeta_full = np.zeros(
        int(N * ns * nalpha),
    )

    zeta = np.linspace(-ntor * np.pi, ntor * np.pi, N)

    for i in range(ns):
        for j in range(nalpha):
            rho_full[i * N : (i + 1) * N] = rho[i] * np.ones(
                N,
            )
            zeta_full[i * N : (i + 1) * N] = zeta
            theta_full[i * N : (i + 1) * N] = (
                alpha[j]
                * np.ones(
                    N,
                )
                + iota[i] * zeta
            )

    C0 = np.vstack([rho_full, theta_full, zeta_full]).T
    coords = eq.compute_theta_coords(C0, tol=1e-10, maxiter=50)
    grid = Grid(coords, sort=False)

    data_names = [
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
        "p_r",
        "psi_r",
    ]

    data = eq.compute(data_names, grid)

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

    gds2 = data["g^aa"] + 2 * zeta_0 * data["g^ra"] + zeta_0**2 * data["g^rr"]
    gds2_z = rho**2 * (
        g_sup_aa_z0 - 2 * sign_iota * zeta_0 * g_sup_ra_z0 + zeta_0**2 * g_sup_rr_z0
    )
    gds2_zz = rho**2 * (
        g_sup_aa_zz0
        - 2 * sign_iota * zeta_0 * g_sup_ra_zz0
        + zeta_0**2 * g_sup_rr_zz0
    )

    dpdpsi = mu_0 * data["p_r"] / data["psi_r"]

    f = a_N * B_N**3 * gds2 / modB**3 * 1 / gradpar
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
        * (data["cvdrift"] + zeta_0 * data["cvdrift0"])
    )

    V = c / g + 1 / 4 * g_z**2 / g**2 - 1 / 2 * g_zz / g

    b = f / g

    # Since N remains fixed during optimization, we don't need to recalculate
    # the matrix
    h = 2 * np.pi / N
    n1 = int(np.ceil((N - 1) / 2))
    n2 = int(np.floor((N - 1) / 2))
    kk1 = np.linspace(1, n1, n1)
    kk2 = np.linspace(n1 + 1, n1 + n2, n2)

    if np.mod(N, 2) == 0:  # of 2nd derivative matrix
        topc = 1 / (np.sin(np.linspace(1, n1, n1) * h / 2)) ** 2
        col1 = np.concatenate(
            (
                np.array([-(np.pi**2 / 3) / h**2 - 1 / 6]),
                -0.5 * ((-1) ** kk1) * topc,
                -0.5 * ((-1) ** kk2) * topc[0:n2][::-1],
            )
        )
    else:
        topc = 1 / (
            np.sin(np.linspace(h / 2, h / 2 * n1, n1))
            * np.tan(np.linspace(h / 2, h / 2 * n1, n1))
        )
        col1 = np.concatenate(
            (
                np.array([-(np.pi**2 / 3) / h**2 + 1 / 12]),
                0.5 * ((-1) ** kk1) * topc,
                -0.5 * ((-1) ** kk2) * topc[0:n2][::-1],
            )
        )

    row1 = col1  # first row

    D2 = scipy.linalg.toeplitz(col1, r=row1) * (1 / ntor) ** 2 + np.diag(V, 0)

    ## eigvals will be deprecated, replace by subset_by_idx in scipy >= 1.12
    w, v = scipy.linalg.eigh(D2, b=np.diag(b, 0), eigvals=[N - 1, N - 1])
    return w
