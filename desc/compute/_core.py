"""Core compute functions, for profiles, geometry, and basis vectors/jacobians."""

from scipy.constants import mu_0

from desc.backend import jnp

from .data_index import data_index
from .utils import cross, dot, has_dependencies, surface_averages, surface_integrals


def compute_flux_coords(params, transforms, profiles, data=None, **kwargs):
    """Compute flux coordinates (rho,theta,zeta)."""
    if data is None:
        data = {}

    data["rho"] = transforms["grid"].nodes[:, 0]
    data["theta"] = transforms["grid"].nodes[:, 1]
    data["zeta"] = transforms["grid"].nodes[:, 2]

    return data


def compute_toroidal_flux(params, transforms, profiles, data=None, **kwargs):
    """Compute toroidal magnetic flux profile."""
    data = compute_flux_coords(params, transforms, profiles, data=data, **kwargs)

    # psi = params["Psi"] / 2*pi  # noqa: E800
    if has_dependencies("psi", params, transforms, profiles, data):
        data["psi"] = params["Psi"] * data["rho"] ** 2 / (2 * jnp.pi)
    if has_dependencies("psi_r", params, transforms, profiles, data):
        data["psi_r"] = 2 * params["Psi"] * data["rho"] / (2 * jnp.pi)
    if has_dependencies("psi_rr", params, transforms, profiles, data):
        data["psi_rr"] = 2 * params["Psi"] * jnp.ones_like(data["rho"]) / (2 * jnp.pi)

    return data


def compute_toroidal_flux_gradient(
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute grad(psi)."""
    data = compute_toroidal_flux(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_contravariant_metric_coefficients(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("grad(psi)", params, transforms, profiles, data):
        data["grad(psi)"] = (data["psi_r"] * data["e^rho"].T).T
    if has_dependencies("|grad(psi)|^2", params, transforms, profiles, data):
        data["|grad(psi)|^2"] = data["psi_r"] ** 2 * data["g^rr"]
    if has_dependencies("|grad(psi)|", params, transforms, profiles, data):
        data["|grad(psi)|"] = jnp.sqrt(data["|grad(psi)|^2"])

    return data


def compute_toroidal_coords(params, transforms, profiles, data=None, **kwargs):
    """Compute toroidal coordinates (R,phi,Z)."""
    if data is None:
        data = {}

    derivs = [
        "",
        "_r",
        "_t",
        "_z",
        "_rr",
        "_tt",
        "_zz",
        "_rt",
        "_rz",
        "_tz",
        "_rrr",
        "_ttt",
        "_zzz",
        "_rrt",
        "_rtt",
        "_rrz",
        "_rzz",
        "_ttz",
        "_tzz",
        "_rtz",
    ]

    for d in derivs:
        keyR = "R" + d
        keyZ = "Z" + d
        if has_dependencies(keyR, params, transforms, profiles, data):
            data[keyR] = transforms["R"].transform(
                params["R_lmn"], *data_index[keyR]["dependencies"]["transforms"]["R"][0]
            )
        if has_dependencies(keyZ, params, transforms, profiles, data):
            data[keyZ] = transforms["Z"].transform(
                params["Z_lmn"], *data_index[keyZ]["dependencies"]["transforms"]["Z"][0]
            )

    return data


def compute_cartesian_coords(params, transforms, profiles, data=None, **kwargs):
    """Compute Cartesian coordinates (X,Y,Z)."""
    data = compute_flux_coords(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_toroidal_coords(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("phi", params, transforms, profiles, data):
        data["phi"] = data["zeta"]
    if has_dependencies("X", params, transforms, profiles, data):
        data["X"] = data["R"] * jnp.cos(data["phi"])
    if has_dependencies("Y", params, transforms, profiles, data):
        data["Y"] = data["R"] * jnp.sin(data["phi"])

    return data


def compute_lambda(params, transforms, profiles, data=None, **kwargs):
    """Compute lambda such that theta* = theta + lambda is a sfl coordinate."""
    if data is None:
        data = {}

    keys = [
        "lambda",
        "lambda_r",
        "lambda_t",
        "lambda_z",
        "lambda_rr",
        "lambda_tt",
        "lambda_zz",
        "lambda_rt",
        "lambda_rz",
        "lambda_tz",
        "lambda_rrr",
        "lambda_ttt",
        "lambda_zzz",
        "lambda_rrt",
        "lambda_rtt",
        "lambda_rrz",
        "lambda_rzz",
        "lambda_ttz",
        "lambda_tzz",
        "lambda_rtz",
    ]

    for key in keys:
        if has_dependencies(key, params, transforms, profiles, data):
            data[key] = transforms["L"].transform(
                params["L_lmn"], *data_index[key]["dependencies"]["transforms"]["L"][0]
            )

    return data


def compute_pressure(params, transforms, profiles, data=None, **kwargs):
    """Compute pressure profile."""
    if data is None:
        data = {}

    if has_dependencies("p", params, transforms, profiles, data):
        data["p"] = profiles["pressure"].compute(params["p_l"], dr=0)
    if has_dependencies("p_r", params, transforms, profiles, data):
        data["p_r"] = profiles["pressure"].compute(params["p_l"], dr=1)

    return data


def compute_pressure_gradient(params, transforms, profiles, data=None, **kwargs):
    """Compute pressure gradient and volume average."""
    data = compute_pressure(
        params,
        transforms,
        profiles,
        data,
    )
    data = compute_contravariant_metric_coefficients(
        params,
        transforms,
        profiles,
        data,
    )
    data = compute_geometry(
        params,
        transforms,
        profiles,
        data,
    )
    if has_dependencies("grad(p)", params, transforms, profiles, data):
        data["grad(p)"] = (data["p_r"] * data["e^rho"].T).T
    if has_dependencies("|grad(p)|", params, transforms, profiles, data):
        data["|grad(p)|"] = jnp.sqrt(data["p_r"] ** 2) * data["|grad(rho)|"]
    if has_dependencies("<|grad(p)|>_vol", params, transforms, profiles, data):
        data["<|grad(p)|>_vol"] = (
            jnp.sum(
                data["|grad(p)|"]
                * jnp.abs(data["sqrt(g)"])
                * transforms["grid"].weights
            )
            / data["V"]
        )
    return data


def compute_rotational_transform(params, transforms, profiles, data=None, **kwargs):
    """
    Compute rotational transform profile from the iota or the toroidal current profile.

    Notes
    -----
        The rotational transform is computed from the toroidal current profile using
        equation 11 in S.P. Hishman & J.T. Hogan (1986)
        doi:10.1016/0021-9991(86)90197-X. Their "zero current algorithm" is supplemented
        with an additional term to account for finite net toroidal currents. Note that
        the flux surface average integrals in their formula should not be weighted by a
        coordinate Jacobian factor, meaning the sqrt(g) terms in the denominators of
        these averages will not be canceled out.

    """
    if data is None:
        data = {}

    if profiles["iota"] is not None:
        data["iota"] = profiles["iota"].compute(params["i_l"], dr=0)
        data["iota_r"] = profiles["iota"].compute(params["i_l"], dr=1)

    elif profiles["current"] is not None:
        data = compute_toroidal_flux(
            params,
            transforms,
            profiles,
            data=data,
            **kwargs,
        )
        data = compute_lambda(
            params,
            transforms,
            profiles,
            data=data,
            **kwargs,
        )
        data = compute_jacobian(
            params,
            transforms,
            profiles,
            data=data,
            **kwargs,
        )
        data = compute_covariant_metric_coefficients(
            params,
            transforms,
            profiles,
            data=data,
            **kwargs,
        )

        if has_dependencies("iota", params, transforms, profiles, data):
            # current_term = 2*pi * I / params["Psi"]_r = mu_0 / 2*pi * current / psi_r
            current_term = (
                mu_0
                / (2 * jnp.pi)
                * profiles["current"].compute(params["c_l"], dr=0)
                / data["psi_r"]
            )
            num = (
                data["lambda_z"] * data["g_tt"] - (1 + data["lambda_t"]) * data["g_tz"]
            ) / data["sqrt(g)"]
            den = data["g_tt"] / data["sqrt(g)"]
            num_avg = surface_averages(transforms["grid"], num)
            den_avg = surface_averages(transforms["grid"], den)
            data["iota"] = (current_term + num_avg) / den_avg

        if has_dependencies("iota_r", params, transforms, profiles, data):
            current_term_r = (
                mu_0
                / (2 * jnp.pi)
                * profiles["current"].compute(params["c_l"], dr=1)
                / data["psi_r"]
                - current_term * data["psi_rr"] / data["psi_r"]
            )
            num_r = (
                data["lambda_rz"] * data["g_tt"]
                + data["lambda_z"] * data["g_tt_r"]
                - data["lambda_rt"] * data["g_tz"]
                - (1 + data["lambda_t"]) * data["g_tz_r"]
            ) / data["sqrt(g)"] - num * data["sqrt(g)_r"] / data["sqrt(g)"]
            den_r = (
                data["g_tt_r"] / data["sqrt(g)"]
                - data["g_tt"] * data["sqrt(g)_r"] / data["sqrt(g)"] ** 2
            )
            num_avg_r = surface_averages(transforms["grid"], num_r)
            den_avg_r = surface_averages(transforms["grid"], den_r)
            data["iota_r"] = (
                current_term_r + num_avg_r - data["iota"] * den_avg_r
            ) / den_avg

    return data


def compute_covariant_basis(  # noqa: C901
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute covariant basis vectors."""
    data = compute_toroidal_coords(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data["0"] = jnp.zeros(transforms["grid"].num_nodes)

    # 0th order derivatives
    if has_dependencies("e_rho", params, transforms, profiles, data):
        data["e_rho"] = jnp.array([data["R_r"], data["0"], data["Z_r"]]).T
    if has_dependencies("e_theta", params, transforms, profiles, data):
        data["e_theta"] = jnp.array([data["R_t"], data["0"], data["Z_t"]]).T
    if has_dependencies("e_zeta", params, transforms, profiles, data):
        data["e_zeta"] = jnp.array([data["R_z"], data["R"], data["Z_z"]]).T

    # 1st order derivatives
    if has_dependencies("e_rho_r", params, transforms, profiles, data):
        data["e_rho_r"] = jnp.array([data["R_rr"], data["0"], data["Z_rr"]]).T
    if has_dependencies("e_rho_t", params, transforms, profiles, data):
        data["e_rho_t"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]]).T
    if has_dependencies("e_rho_z", params, transforms, profiles, data):
        data["e_rho_z"] = jnp.array([data["R_rz"], data["0"], data["Z_rz"]]).T
    if has_dependencies("e_theta_r", params, transforms, profiles, data):
        data["e_theta_r"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]]).T
    if has_dependencies("e_theta_t", params, transforms, profiles, data):
        data["e_theta_t"] = jnp.array([data["R_tt"], data["0"], data["Z_tt"]]).T
    if has_dependencies("e_theta_z", params, transforms, profiles, data):
        data["e_theta_z"] = jnp.array([data["R_tz"], data["0"], data["Z_tz"]]).T
    if has_dependencies("e_zeta_r", params, transforms, profiles, data):
        data["e_zeta_r"] = jnp.array([data["R_rz"], data["R_r"], data["Z_rz"]]).T
    if has_dependencies("e_zeta_t", params, transforms, profiles, data):
        data["e_zeta_t"] = jnp.array([data["R_tz"], data["R_t"], data["Z_tz"]]).T
    if has_dependencies("e_zeta_z", params, transforms, profiles, data):
        data["e_zeta_z"] = jnp.array([data["R_zz"], data["R_z"], data["Z_zz"]]).T

    # 2nd order derivatives
    if has_dependencies("e_rho_rr", params, transforms, profiles, data):
        data["e_rho_rr"] = jnp.array([data["R_rrr"], data["0"], data["Z_rrr"]]).T
    if has_dependencies("e_rho_tt", params, transforms, profiles, data):
        data["e_rho_tt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]]).T
    if has_dependencies("e_rho_zz", params, transforms, profiles, data):
        data["e_rho_zz"] = jnp.array([data["R_rzz"], data["0"], data["Z_rzz"]]).T
    if has_dependencies("e_rho_rt", params, transforms, profiles, data):
        data["e_rho_rt"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]]).T
    if has_dependencies("e_rho_rz", params, transforms, profiles, data):
        data["e_rho_rz"] = jnp.array([data["R_rrz"], data["0"], data["Z_rrz"]]).T
    if has_dependencies("e_rho_tz", params, transforms, profiles, data):
        data["e_rho_tz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]]).T

    if has_dependencies("e_theta_rr", params, transforms, profiles, data):
        data["e_theta_rr"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]]).T
    if has_dependencies("e_theta_tt", params, transforms, profiles, data):
        data["e_theta_tt"] = jnp.array([data["R_ttt"], data["0"], data["Z_ttt"]]).T
    if has_dependencies("e_theta_zz", params, transforms, profiles, data):
        data["e_theta_zz"] = jnp.array([data["R_tzz"], data["0"], data["Z_tzz"]]).T
    if has_dependencies("e_theta_rt", params, transforms, profiles, data):
        data["e_theta_rt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]]).T
    if has_dependencies("e_theta_rz", params, transforms, profiles, data):
        data["e_theta_rz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]]).T
    if has_dependencies("e_theta_tz", params, transforms, profiles, data):
        data["e_theta_tz"] = jnp.array([data["R_ttz"], data["0"], data["Z_ttz"]]).T

    if has_dependencies("e_zeta_rr", params, transforms, profiles, data):
        data["e_zeta_rr"] = jnp.array([data["R_rrz"], data["R_rr"], data["Z_rrz"]]).T
    if has_dependencies("e_zeta_tt", params, transforms, profiles, data):
        data["e_zeta_tt"] = jnp.array([data["R_ttz"], data["R_tt"], data["Z_ttz"]]).T
    if has_dependencies("e_zeta_zz", params, transforms, profiles, data):
        data["e_zeta_zz"] = jnp.array([data["R_zzz"], data["R_zz"], data["Z_zzz"]]).T
    if has_dependencies("e_zeta_rt", params, transforms, profiles, data):
        data["e_zeta_rt"] = jnp.array([data["R_rtz"], data["R_rt"], data["Z_rtz"]]).T
    if has_dependencies("e_zeta_rz", params, transforms, profiles, data):
        data["e_zeta_rz"] = jnp.array([data["R_rzz"], data["R_rz"], data["Z_rzz"]]).T
    if has_dependencies("e_zeta_tz", params, transforms, profiles, data):
        data["e_zeta_tz"] = jnp.array([data["R_tzz"], data["R_tz"], data["Z_tzz"]]).T

    return data


def compute_contravariant_basis(params, transforms, profiles, data=None, **kwargs):
    """Compute contravariant basis vectors."""
    if data is None or "sqrt(g)" not in data:
        data = compute_jacobian(
            params,
            transforms,
            profiles,
            data=data,
            **kwargs,
        )

    if has_dependencies("e^rho", params, transforms, profiles, data):
        data["e^rho"] = (cross(data["e_theta"], data["e_zeta"]).T / data["sqrt(g)"]).T
    if has_dependencies("e^theta", params, transforms, profiles, data):
        data["e^theta"] = (cross(data["e_zeta"], data["e_rho"]).T / data["sqrt(g)"]).T
    if has_dependencies("e^zeta", params, transforms, profiles, data):
        data["e^zeta"] = jnp.array([data["0"], 1 / data["R"], data["0"]]).T

    return data


def compute_jacobian(params, transforms, profiles, data=None, **kwargs):
    """Compute coordinate system Jacobian."""
    data = compute_covariant_basis(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("sqrt(g)", params, transforms, profiles, data):
        data["sqrt(g)"] = dot(data["e_rho"], cross(data["e_theta"], data["e_zeta"]))
    if has_dependencies("|e_theta x e_zeta|", params, transforms, profiles, data):
        data["|e_theta x e_zeta|"] = jnp.linalg.norm(
            cross(data["e_theta"], data["e_zeta"]), axis=1
        )
    if has_dependencies("|e_zeta x e_rho|", params, transforms, profiles, data):
        data["|e_zeta x e_rho|"] = jnp.linalg.norm(
            cross(data["e_zeta"], data["e_rho"]), axis=1
        )
    if has_dependencies("|e_rho x e_theta|", params, transforms, profiles, data):
        data["|e_rho x e_theta|"] = jnp.linalg.norm(
            cross(data["e_rho"], data["e_theta"]), axis=1
        )

    # 1st order derivatives
    if has_dependencies("sqrt(g)_r", params, transforms, profiles, data):
        data["sqrt(g)_r"] = (
            dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_r"]))
        )
    if has_dependencies("sqrt(g)_t", params, transforms, profiles, data):
        data["sqrt(g)_t"] = (
            dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_t"]))
        )
    if has_dependencies("sqrt(g)_z", params, transforms, profiles, data):
        data["sqrt(g)_z"] = (
            dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_z"]))
        )

    # 2nd order derivatives
    if has_dependencies("sqrt(g)_rr", params, transforms, profiles, data):
        data["sqrt(g)_rr"] = (
            dot(data["e_rho_rr"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_rr"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rr"]))
            + 2 * dot(data["e_rho_r"], cross(data["e_theta_r"], data["e_zeta"]))
            + 2 * dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_r"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_r"]))
        )
    if has_dependencies("sqrt(g)_tt", params, transforms, profiles, data):
        data["sqrt(g)_tt"] = (
            dot(data["e_rho_tt"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_tt"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tt"]))
            + 2 * dot(data["e_rho_t"], cross(data["e_theta_t"], data["e_zeta"]))
            + 2 * dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_t"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_t"]))
        )
    if has_dependencies("sqrt(g)_zz", params, transforms, profiles, data):
        data["sqrt(g)_zz"] = (
            dot(data["e_rho_zz"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_zz"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_zz"]))
            + 2 * dot(data["e_rho_z"], cross(data["e_theta_z"], data["e_zeta"]))
            + 2 * dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_z"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_z"]))
        )
    if has_dependencies("sqrt(g)_tz", params, transforms, profiles, data):
        data["sqrt(g)_tz"] = (
            dot(data["e_rho_tz"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho_z"], cross(data["e_theta_t"], data["e_zeta"]))
            + dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_t"]))
            + dot(data["e_rho_t"], cross(data["e_theta_z"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_tz"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_t"]))
            + dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_z"]))
            + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_z"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tz"]))
        )

    return data


def compute_covariant_metric_coefficients(
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute metric coefficients."""
    data = compute_covariant_basis(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("g_rr", params, transforms, profiles, data):
        data["g_rr"] = dot(data["e_rho"], data["e_rho"])
    if has_dependencies("g_tt", params, transforms, profiles, data):
        data["g_tt"] = dot(data["e_theta"], data["e_theta"])
    if has_dependencies("g_zz", params, transforms, profiles, data):
        data["g_zz"] = dot(data["e_zeta"], data["e_zeta"])
    if has_dependencies("g_rt", params, transforms, profiles, data):
        data["g_rt"] = dot(data["e_rho"], data["e_theta"])
    if has_dependencies("g_rz", params, transforms, profiles, data):
        data["g_rz"] = dot(data["e_rho"], data["e_zeta"])
    if has_dependencies("g_tz", params, transforms, profiles, data):
        data["g_tz"] = dot(data["e_theta"], data["e_zeta"])

    # todo: add other derivatives
    if has_dependencies("g_tt_r", params, transforms, profiles, data):
        data["g_tt_r"] = 2 * dot(data["e_theta"], data["e_theta_r"])
    if has_dependencies("g_tz_r", params, transforms, profiles, data):
        data["g_tz_r"] = dot(data["e_theta_r"], data["e_zeta"]) + dot(
            data["e_theta"], data["e_zeta_r"]
        )

    return data


def compute_contravariant_metric_coefficients(
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute reciprocal metric coefficients."""
    data = compute_contravariant_basis(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("g^rr", params, transforms, profiles, data):
        data["g^rr"] = dot(data["e^rho"], data["e^rho"])
    if has_dependencies("g^tt", params, transforms, profiles, data):
        data["g^tt"] = dot(data["e^theta"], data["e^theta"])
    if has_dependencies("g^zz", params, transforms, profiles, data):
        data["g^zz"] = dot(data["e^zeta"], data["e^zeta"])
    if has_dependencies("g^rt", params, transforms, profiles, data):
        data["g^rt"] = dot(data["e^rho"], data["e^theta"])
    if has_dependencies("g^rz", params, transforms, profiles, data):
        data["g^rz"] = dot(data["e^rho"], data["e^zeta"])
    if has_dependencies("g^tz", params, transforms, profiles, data):
        data["g^tz"] = dot(data["e^theta"], data["e^zeta"])

    if has_dependencies("|grad(rho)|", params, transforms, profiles, data):
        data["|grad(rho)|"] = jnp.sqrt(data["g^rr"])
    if has_dependencies("|grad(theta)|", params, transforms, profiles, data):
        data["|grad(theta)|"] = jnp.sqrt(data["g^tt"])
    if has_dependencies("|grad(zeta)|", params, transforms, profiles, data):
        data["|grad(zeta)|"] = jnp.sqrt(data["g^zz"])

    return data


def compute_geometry(params, transforms, profiles, data=None, **kwargs):
    """Compute geometric quantities such as plasma volume, aspect ratio, etc."""
    data = compute_jacobian(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if has_dependencies("V(r)", params, transforms, profiles, data):
        # divergence theorem: integral(dV div [0, 0, Z]) = integral(dS dot [0, 0, Z])
        data["V(r)"] = jnp.abs(
            surface_integrals(
                transforms["grid"],
                cross(data["e_theta"], data["e_zeta"])[:, 2] * data["Z"],
            )
        )
    if has_dependencies("V_r(r)", params, transforms, profiles, data):
        # eq. 4.9.10 in W.D. D'haeseleer et al. (1991) doi:10.1007/978-3-642-75595-8.
        data["V_r(r)"] = surface_integrals(transforms["grid"], jnp.abs(data["sqrt(g)"]))
    if has_dependencies("S(r)", params, transforms, profiles, data):
        data["S(r)"] = surface_integrals(transforms["grid"], data["|e_theta x e_zeta|"])
    if has_dependencies("V", params, transforms, profiles, data):
        data["V"] = jnp.sum(jnp.abs(data["sqrt(g)"]) * transforms["grid"].weights)
    if has_dependencies("A", params, transforms, profiles, data):
        data["A"] = jnp.mean(
            surface_integrals(
                transforms["grid"],
                jnp.abs(data["sqrt(g)"] / data["R"]),
                surface_label="zeta",
            )
        )
    if has_dependencies("R0", params, transforms, profiles, data):
        data["R0"] = data["V"] / (2 * jnp.pi * data["A"])
    if has_dependencies("a", params, transforms, profiles, data):
        data["a"] = jnp.sqrt(data["A"] / jnp.pi)
    if has_dependencies("R0/a", params, transforms, profiles, data):
        data["R0/a"] = data["R0"] / data["a"]
    if has_dependencies("V_rr(r)", params, transforms, profiles, data):
        data["V_rr(r)"] = surface_integrals(
            transforms["grid"], data["sqrt(g)_r"] * jnp.sign(data["sqrt(g)"])
        )

    return data
