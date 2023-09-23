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

from desc.backend import jnp
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


def _gamma_ideal_ballooning_FD1(eq):
    """
    Ideal-ballooning growth rate finder.

    This function uses a pseudospectral Fourier technique to
    calculate the maximum growth rate against the infinite-n
    ideal ballooning mode

    Parameters
    ----------
    eq :  Input equilibrium object
    N : resolution
    """
    s = 0.81
    alpha = 0.0
    iota = 0.5
    nperiod = 5
    ntor = 2 * nperiod - 1
    N = (2 * eq.Mgrid * eq.Ngrid + 1) * ntor
    coords = np.ones((N, 3))
    coords = coords.at[:, 0].set(coords[:, 0] * jnp.sqrt(s))
    coords = coords.at[:, 2].set(np.linspace(-ntor * np.pi, ntor * np.pi, N))
    guess = coords.copy()

    coords = coords.at[:, 1].set(coords[:, 1] * alpha)  # set which field line we want

    guess = guess.at[:, 1].set((alpha - guess[:, 2]) / iota)

    coords1 = eq.map_coordinates(
        coords=coords,
        inbasis=["rho", "alpha", "zeta"],
        outbasis=["rho", "theta", "zeta"],
        period=[-np.inf, 2 * np.pi, np.inf],
        guess=guess,
    )

    coords1 = coords1.at[:, 2].set(coords[:, 2])

    zeta_0 = 0
    print("mapped coords")
    grid2 = Grid(coords1, sort=False)

    data_names = [
        "g^aa",
        "g^ra",
        "g^rr",
        "cvdrift",
        "cvdrift0",
        "|B|",
        "sqrt(g)_PEST",
        "p_r",
        "psi_r",
    ]

    data = eq.compute(data_names, grid2)

    # sqrt(g)_PEST = B dot grad zeta
    gradpar = data["sqrt(g)_PEST"] / data["|B|"]

    gds2 = data["g^aa"] + zeta_0 * data["g^ra"] + zeta_0**2 * data["g^rr"]

    dpdpsi = mu_0 * data["p_r"] / data["psi_r"]

    f = gds2 / data["|B|"] ** 2

    g = f * data["sqrt(g)_PEST"]

    g_half = (g[1:] + g[:-1]) / 2

    c = 1 / gradpar * dpdpsi * data["cvdrift"] + zeta_0 * data["cvdrift0"]

    # Equation: d/dz (g dX/ dz) + c * X - lam * f * X = 0, f > 0

    # grid spacing
    h = 2 * ntor * np.pi / N

    A = (
        np.diag(g_half[1:-1] / f[2:-1] * 1 / h**2, -1)
        + np.diag(
            -(g_half[1:] + g_half[:-1]) / f[1:-1] * 1 / h**2 + c[1:-1] / f[1:-1],
            0,
        )
        + np.diag(g_half[1:-1] / f[1:-2] * 1 / h**2, 1)
    )

    w, v = scipy.sparse.linalg.eigs(A, k=1)

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

    Y0 = -g * dX**2 + c * X**2
    Y1 = f * X**2
    gam = simps(Y0) / simps(Y1)

    return gam


def _gamma_ideal_ballooning_FD(eq):
    """
    Ideal-ballooning growth rate finder.

    This function uses a pseudospectral Fourier technique to
    calculate the maximum growth rate against the infinite-n
    ideal ballooning mode

    Parameters
    ----------
    eq :  Input equilibrium object
    N : resolution
    """
    s = 0.81
    alpha = 0.0
    iota = 0.5
    nperiod = 5
    ntor = 2 * nperiod - 1
    N = (2 * eq.Mpol * eq.Ntor + 1) * ntor
    coords = np.ones((N, 3))
    coords = coords.at[:, 0].set(coords[:, 0] * jnp.sqrt(s))
    coords = coords.at[:, 2].set(np.linspace(-ntor * np.pi, ntor * np.pi, N))
    guess = coords.copy()

    coords = coords.at[:, 1].set(coords[:, 1] * alpha)  # set which field line we want

    guess = guess.at[:, 1].set((alpha - guess[:, 2]) / iota)

    coords1 = eq.map_coordinates(
        coords=coords,
        inbasis=["rho", "alpha", "zeta"],
        outbasis=["rho", "theta", "zeta"],
        period=[-np.inf, 2 * np.pi, np.inf],
        guess=guess,
    )

    coords1 = coords1.at[:, 2].set(coords[:, 2])

    zeta_0 = 0
    print("mapped coords")
    grid2 = Grid(coords1, sort=False)

    data_names = [
        "g^aa",
        "g^ra",
        "g^rr",
        "cvdrift",
        "cvdrift0",
        "|B|",
        "sqrt(g)_PEST",
        "sqrt(g)_PEST_z",
        "sqrt(g)_PEST_zz",
        "p_r",
        "psi_r",
    ]

    data = eq.compute(data_names, grid2)

    # sqrt(g)_PEST = B dot grad zeta
    gradpar = data["sqrt(g)_PEST"] / data["|B|"]

    gds2 = data["g^aa"] + zeta_0 * data["g^ra"] + zeta_0**2 * data["g^rr"]
    gds2_z = data["g^aa_z"] + zeta_0 * data["g^ra_z"] + zeta_0**2 * data["g^rr_z"]
    gds2_zz = data["g^aa_zz"] + zeta_0 * data["g^ra_zz"] + zeta_0**2 * data["g^rr_zz"]

    dpdpsi = mu_0 * data["p_r"] / data["psi_r"]

    f = gds2 / data["|B|"] ** 2
    f_z = gds2_z / data["|B|"] ** 2 - 2 * data["B_z"] * gds2
    f_zz = (
        gds2_zz / data["|B|"] ** 2
        - 4 * data["|B|_z"] * gds2_z
        - 2 * data["|B|_zz"] * gds2
    )

    g = f * data["sqrt(g)_PEST"]
    g_z = f_z * data["sqrt(g)_PEST"] + f * data["sqrt(g)_PEST_z"]
    g_zz = (
        f_zz * data["sqrt(g)_PEST"]
        + 2 * f_z * data["sqrt(g)_PEST_z"]
        + f * data["sqrt(g)_PEST_zz"]
    )

    c = 1 / gradpar * dpdpsi * data["cvdrift"] + zeta_0 * data["cvdrift0"]

    V = (c + 1 / 4 * g_z**2 - 1 / 2 * g_zz) / g

    b = f / g

    # Equation: d^2 X / d z^2 + V * X - lam * b * X = 0, b > 0

    # grid spacing
    h = 2 * ntor * np.pi / N

    D2 = np.diag(1 / h**2, -1) + np.diag(-2 / h**2 + V, 0) + np.diag(1 / h**2, 1)

    b = f / g * np.eye(N)

    w, v = scipy.sparse.linalg.eigs(D2, b, k=1)

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

    Y0 = -g * dX**2 + c * X**2
    Y1 = f * X**2
    gam = simps(Y0) / simps(Y1)

    return gam


def _gamma_ideal_ballooning_spectral(eq):
    """
    Ideal-ballooning growth rate finder.

    This function uses a pseudospectral Fourier technique to
    calculate the maximum growth rate against the infinite-n
    ideal ballooning mode

    Parameters
    ----------
    eq :  Input equilibrium object
    x = jax.scipy.linalg.eikh(D2, b * np.eye(N))
    Equation: d^2 X / d z^2 + V * X - lam * b * X = 0, b > 0
    Psedospectral representation D^2 X + V * X - lam * b * X = 0
    grid spacing
    obtain g, c, and f using eq and other classes
    """
    s = 0.81
    alpha = 0.0
    iota = 0.5
    nperiod = 5
    ntor = 2 * nperiod - 1
    N = (2 * eq.Mpol * eq.Ntor + 1) * ntor
    coords = np.ones((N, 3))
    coords = coords.at[:, 0].set(coords[:, 0] * jnp.sqrt(s))
    coords = coords.at[:, 2].set(np.linspace(-ntor * np.pi, ntor * np.pi, N))
    guess = coords.copy()

    coords = coords.at[:, 1].set(coords[:, 1] * alpha)  # set which field line we want

    guess = guess.at[:, 1].set((alpha - guess[:, 2]) / iota)

    coords1 = eq.map_coordinates(
        coords=coords,
        inbasis=["rho", "alpha", "zeta"],
        outbasis=["rho", "theta", "zeta"],
        period=[-np.inf, 2 * np.pi, np.inf],
        guess=guess,
    )

    coords1 = coords1.at[:, 2].set(coords[:, 2])

    zeta_0 = 0
    print("mapped coords")
    grid2 = Grid(coords1, sort=False)

    data_names = [
        "g^aa",
        "g^ra",
        "g^rr",
        "cvdrift",
        "cvdrift0",
        "|B|",
        "sqrt(g)_PEST",
        "sqrt(g)_PEST_z",
        "sqrt(g)_PEST_zz",
        "p_r",
        "psi_r",
    ]

    data = eq.compute(data_names, grid2)

    # sqrt(g)_PEST = B dot grad zeta
    gradpar = data["sqrt(g)_PEST"] / data["|B|"]

    gds2 = data["g^aa"] + zeta_0 * data["g^ra"] + zeta_0**2 * data["g^rr"]
    gds2_z = data["g^aa_z"] + zeta_0 * data["g^ra_z"] + zeta_0**2 * data["g^rr_z"]
    gds2_zz = data["g^aa_zz"] + zeta_0 * data["g^ra_zz"] + zeta_0**2 * data["g^rr_zz"]

    dpdpsi = mu_0 * data["p_r"] / data["psi_r"]

    f = gds2 / data["|B|"] ** 2
    f_z = gds2_z / data["|B|"] ** 2 - 2 * data["B_z"] * gds2
    f_zz = (
        gds2_zz / data["|B|"] ** 2
        - 4 * data["|B|_z"] * gds2_z
        - 2 * data["|B|_zz"] * gds2
    )

    g = f * data["sqrt(g)_PEST"]
    g_z = f_z * data["sqrt(g)_PEST"] + f * data["sqrt(g)_PEST_z"]
    g_zz = (
        f_zz * data["sqrt(g)_PEST"]
        + 2 * f_z * data["sqrt(g)_PEST_z"]
        + f * data["sqrt(g)_PEST_zz"]
    )

    c = 1 / gradpar * dpdpsi * data["cvdrift"] + zeta_0 * data["cvdrift0"]

    V = (c + 1 / 4 * g_z**2 - 1 / 2 * g_zz) / g

    b = f / g

    h = 2 * ntor * np.pi / N
    kk = np.linspace(1, N - 1, N - 1)
    n1 = np.floor((N - 1) / 2)
    n2 = np.ceil((N - 1) / 2)

    if jnp.mod(N, 2) == 0:  # of 2nd derivative matrix
        topc = 1 / (np.sin(jnp.linspace(1, n2, n2) * h / 2)) ** 2
        col1 = [
            -(np.pi**2 / 3) / h**2 - (1 / 6),
            -0.5 * ((-1) ** kk) * [topc, np.flipud(topc[1:n1])],
        ]
    else:
        topc = (
            1
            / (np.sin(jnp.linspace(1, n2, n2) * h / 2))
            * 1
            / (np.tan(jnp.linspace(1, n2, n2) * h / 2))
        )
        col1 = [
            -np.pi**2 / 3 / h**2 + 1 / 12,
            -0.5 * ((-1) ** kk) * [topc, -np.flipud(topc[1:n1])],
        ]

    row1 = col1  # first row

    D2 = np.toeplitz(col1, row1) + V * np.eye(N)

    x = scipy.linalg.eik(D2, b * np.eye(N))

    return x
