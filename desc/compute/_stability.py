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

from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import cross, dot, surface_integrals_map


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
    name="gds2",
    label="\\mathrm{gds2}",
    units="~",
    units_long="None",
    description="Fieldline bending term1",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["grad(alpha)", "dpsidr"],
)
def _gds2(params, transforms, profiles, data, **kwargs):
    data["gds2"] = data["dpsidr"] ** 2 * dot(data["grad(alpha)"], data["grad(alpha)"])


@register_compute_fun(
    name="gds21",
    label="\\mathrm{gds21}",
    units="~",
    units_long="None",
    description="Fieldline bending term2",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["grad(alpha)", "grad(rho)", "dpsidr", "iota_r"],
)
def _gds21(params, transforms, profiles, data, **kwargs):
    data["gds21"] = (
        data["dpsidr"] ** 2
        * data["iota_r"]
        * dot(data["grad(alpha)"], data["grad(rho)"])
    )


@register_compute_fun(
    name="gds22",
    label="\\mathrm{gds22}",
    units="~",
    units_long="None",
    description="Fieldline bending term3",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "dpsidr", "iota_r"],
)
def _gds22(params, transforms, profiles, data, **kwargs):
    data["gds22"] = (
        data["dpsidr"] ** 2 * data["iota_r"] * dot(data["e^rho"], data["e^rho"])
    )


@register_compute_fun(
    name="gds22_z",
    label="\\mathrm{gds22}_{z}",
    units="~",
    units_long="None",
    description="Fieldline bending term3",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^rho_z", "dpsidr", "iota_r"],
)
def _gds22_z(params, transforms, profiles, data, **kwargs):
    data["gds22_z"] = (
        data["dpsidr"] ** 2 * data["iota_r"] * 2 * dot(data["e^rho_z"], data["e^rho"])
    )


@register_compute_fun(
    name="gds22_zz",
    label="\\mathrm{gds22}_{zz}",
    units="~",
    units_long="None",
    description="Fieldline bending term3",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["e^rho", "e^rho_z", "e^rho_zz", "dpsidr", "iota_r"],
)
def _gds22_zz(params, transforms, profiles, data, **kwargs):
    data["gds22_zz"] = (
        data["dpsidr"] ** 2
        * data["iota_r"]
        * 2
        * (dot(data["e^rho_zz"], data["e^rho"]) + dot(data["e^rho_z"], data["e^rho_z"]))
    )


@register_compute_fun(
    name="gbdrift",
    label="\\mathrm{gradB-drift}",
    units="~",
    units_long="None",
    description="Geometric part of the gradB drift"
    + "used for local stability analyses",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "b", "sqrt(g)", "grad(alpha)" "grad(|B|)", "dpsidr"],
)
def _gbdrift(params, transforms, profiles, data, **kwargs):
    data["gbdrift"] = (
        data["dpsidr"]
        * 2
        / data["|B|"] ** 2
        * dot(data["b"], cross(data["grad(|B|)"], data["grad(alpha)"]))
    )


@register_compute_fun(
    name="cvdrift",
    label="\\mathrm{curvature-drift}",
    units="~",
    units_long="None",
    description="Geometric part of the curvature drift"
    + "used for local stability analyses",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["dp_dpsi", "psi_r", "|B|", "gbdrift", "dpsidr"],
)
def _cvdrift(params, transforms, profiles, data, **kwargs):
    data["cvdrift"] = (
        data["dpsidr"] * 1 / data["B"] ** 2 * data["dp_dpsi"] + data["gbdrift"]
    )


@register_compute_fun(
    name="cvdrift0",
    label="\\mathrm{curvature-drift-1}",
    units="~",
    units_long="None",
    description="Geometric part of the curvature drift1"
    + "used for local stability analyses",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["|B|", "b", "iota", "iota_r", "sqrt(g)", "grad(alpha)" "grad(|B|)"],
)
def _cvdrift0(params, transforms, profiles, data, **kwargs):
    data["cvdrift0"] = (
        1
        / data["|B|"] ** 2
        * data["iota_r"]
        * (dot(data["b"], cross(data["grad(|B|)"], data["grad(rho)"])))
    )


def _gamma_ideal_ballooning_spectral(eq, N=1000):
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
    # gridpoints
    N = 100  # temporary N = angular resolution
    # grid spacing
    h = 2 * jnp.pi / N
    kk = jnp.linspace(1, N - 1, N - 1)
    n1 = jnp.floor((N - 1) / 2)
    n2 = jnp.ceil((N - 1) / 2)

    if jnp.mod(N, 2) == 0:  # of 2nd derivative matrix
        topc = 1 / (jnp.sin(jnp.linspace(1, n2, n2) * h / 2)) ** 2
        col1 = [
            -(jnp.pi**2 / 3) / h**2 - (1 / 6),
            -0.5 * ((-1) ** kk) * [topc, jnp.flipud(topc[1:n1])],
        ]
    else:
        topc = (
            1
            / (jnp.sin(jnp.linspace(1, n2, n2) * h / 2))
            * 1
            / (jnp.tan(jnp.linspace(1, n2, n2) * h / 2))
        )
        col1 = [
            -jnp.pi**2 / 3 / h**2 + 1 / 12,
            -0.5 * ((-1) ** kk) * [topc, -jnp.flipud(topc[1:n1])],
        ]

    row1 = col1  # first row

    D2 = jnp.toeplitz(col1, row1)

    ## obtain g, c, and f using eq and other classes

    x = jnp.linalg.eik(D2)

    return x
