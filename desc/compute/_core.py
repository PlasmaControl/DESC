"""Core compute functions, for profiles, geometry, and basis vectors/jacobians."""

from scipy.constants import mu_0

from desc.backend import jnp

from .data_index import data_index
from .utils import check_derivs, cross, dot, surface_averages, surface_integrals


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
    data["psi"] = params["Psi"] * data["rho"] ** 2 / (2 * jnp.pi)
    data["psi_r"] = 2 * params["Psi"] * data["rho"] / (2 * jnp.pi)
    data["psi_rr"] = 2 * params["Psi"] * jnp.ones_like(data["rho"]) / (2 * jnp.pi)

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
        if check_derivs(keyR, transforms["R"], transforms["Z"]):
            data[keyR] = transforms["R"].transform(
                params["R_lmn"], *data_index[keyR]["R_derivs"][0]
            )
            data[keyZ] = transforms["Z"].transform(
                params["Z_lmn"], *data_index[keyZ]["R_derivs"][0]
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

    data["phi"] = data["zeta"]
    data["X"] = data["R"] * jnp.cos(data["phi"])
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
        if check_derivs(key, L_transform=transforms["L"]):
            data[key] = transforms["L"].transform(
                params["L_lmn"], *data_index[key]["L_derivs"][0]
            )

    return data


def compute_pressure(params, transforms, profiles, data=None, **kwargs):
    """Compute pressure profile."""
    if data is None:
        data = {}

    data["p"] = profiles["pressure"].compute(params["p_l"], dr=0)
    data["p_r"] = profiles["pressure"].compute(params["p_l"], dr=1)

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

    grid = transforms["R"].grid
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

        if check_derivs("iota", transforms["R"], transforms["Z"], transforms["L"]):
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
            num_avg = surface_averages(grid, num)
            den_avg = surface_averages(grid, den)
            data["iota"] = (current_term + num_avg) / den_avg

        if check_derivs("iota_r", transforms["R"], transforms["Z"], transforms["L"]):
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
            num_avg_r = surface_averages(grid, num_r)
            den_avg_r = surface_averages(grid, den_r)
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
    data["0"] = jnp.zeros(transforms["R"].num_nodes)

    # 0th order derivatives
    if check_derivs("e_rho", transforms["R"], transforms["Z"]):
        data["e_rho"] = jnp.array([data["R_r"], data["0"], data["Z_r"]]).T
    if check_derivs("e_theta", transforms["R"], transforms["Z"]):
        data["e_theta"] = jnp.array([data["R_t"], data["0"], data["Z_t"]]).T
    if check_derivs("e_zeta", transforms["R"], transforms["Z"]):
        data["e_zeta"] = jnp.array([data["R_z"], data["R"], data["Z_z"]]).T

    # 1st order derivatives
    if check_derivs("e_rho_r", transforms["R"], transforms["Z"]):
        data["e_rho_r"] = jnp.array([data["R_rr"], data["0"], data["Z_rr"]]).T
    if check_derivs("e_rho_t", transforms["R"], transforms["Z"]):
        data["e_rho_t"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]]).T
    if check_derivs("e_rho_z", transforms["R"], transforms["Z"]):
        data["e_rho_z"] = jnp.array([data["R_rz"], data["0"], data["Z_rz"]]).T
    if check_derivs("e_theta_r", transforms["R"], transforms["Z"]):
        data["e_theta_r"] = jnp.array([data["R_rt"], data["0"], data["Z_rt"]]).T
    if check_derivs("e_theta_t", transforms["R"], transforms["Z"]):
        data["e_theta_t"] = jnp.array([data["R_tt"], data["0"], data["Z_tt"]]).T
    if check_derivs("e_theta_z", transforms["R"], transforms["Z"]):
        data["e_theta_z"] = jnp.array([data["R_tz"], data["0"], data["Z_tz"]]).T
    if check_derivs("e_zeta_r", transforms["R"], transforms["Z"]):
        data["e_zeta_r"] = jnp.array([data["R_rz"], data["R_r"], data["Z_rz"]]).T
    if check_derivs("e_zeta_t", transforms["R"], transforms["Z"]):
        data["e_zeta_t"] = jnp.array([data["R_tz"], data["R_t"], data["Z_tz"]]).T
    if check_derivs("e_zeta_z", transforms["R"], transforms["Z"]):
        data["e_zeta_z"] = jnp.array([data["R_zz"], data["R_z"], data["Z_zz"]]).T

    # 2nd order derivatives
    if check_derivs("e_rho_rr", transforms["R"], transforms["Z"]):
        data["e_rho_rr"] = jnp.array([data["R_rrr"], data["0"], data["Z_rrr"]]).T
    if check_derivs("e_rho_tt", transforms["R"], transforms["Z"]):
        data["e_rho_tt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]]).T
    if check_derivs("e_rho_zz", transforms["R"], transforms["Z"]):
        data["e_rho_zz"] = jnp.array([data["R_rzz"], data["0"], data["Z_rzz"]]).T
    if check_derivs("e_rho_rt", transforms["R"], transforms["Z"]):
        data["e_rho_rt"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]]).T
    if check_derivs("e_rho_rz", transforms["R"], transforms["Z"]):
        data["e_rho_rz"] = jnp.array([data["R_rrz"], data["0"], data["Z_rrz"]]).T
    if check_derivs("e_rho_tz", transforms["R"], transforms["Z"]):
        data["e_rho_tz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]]).T
    if check_derivs("e_theta_rr", transforms["R"], transforms["Z"]):
        data["e_theta_rr"] = jnp.array([data["R_rrt"], data["0"], data["Z_rrt"]]).T
    if check_derivs("e_theta_tt", transforms["R"], transforms["Z"]):
        data["e_theta_tt"] = jnp.array([data["R_ttt"], data["0"], data["Z_ttt"]]).T
    if check_derivs("e_theta_zz", transforms["R"], transforms["Z"]):
        data["e_theta_zz"] = jnp.array([data["R_tzz"], data["0"], data["Z_tzz"]]).T
    if check_derivs("e_theta_rt", transforms["R"], transforms["Z"]):
        data["e_theta_rt"] = jnp.array([data["R_rtt"], data["0"], data["Z_rtt"]]).T
    if check_derivs("e_theta_rz", transforms["R"], transforms["Z"]):
        data["e_theta_rz"] = jnp.array([data["R_rtz"], data["0"], data["Z_rtz"]]).T
    if check_derivs("e_theta_tz", transforms["R"], transforms["Z"]):
        data["e_theta_tz"] = jnp.array([data["R_ttz"], data["0"], data["Z_ttz"]]).T
    if check_derivs("e_zeta_rr", transforms["R"], transforms["Z"]):
        data["e_zeta_rr"] = jnp.array([data["R_rrz"], data["R_rr"], data["Z_rrz"]]).T
    if check_derivs("e_zeta_tt", transforms["R"], transforms["Z"]):
        data["e_zeta_tt"] = jnp.array([data["R_ttz"], data["R_tt"], data["Z_ttz"]]).T
    if check_derivs("e_zeta_zz", transforms["R"], transforms["Z"]):
        data["e_zeta_zz"] = jnp.array([data["R_zzz"], data["R_zz"], data["Z_zzz"]]).T
    if check_derivs("e_zeta_rt", transforms["R"], transforms["Z"]):
        data["e_zeta_rt"] = jnp.array([data["R_rtz"], data["R_rt"], data["Z_rtz"]]).T
    if check_derivs("e_zeta_rz", transforms["R"], transforms["Z"]):
        data["e_zeta_rz"] = jnp.array([data["R_rzz"], data["R_rz"], data["Z_rzz"]]).T
    if check_derivs("e_zeta_tz", transforms["R"], transforms["Z"]):
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

    if check_derivs("e^rho", transforms["R"], transforms["Z"]):
        data["e^rho"] = (cross(data["e_theta"], data["e_zeta"]).T / data["sqrt(g)"]).T
    if check_derivs("e^theta", transforms["R"], transforms["Z"]):
        data["e^theta"] = (cross(data["e_zeta"], data["e_rho"]).T / data["sqrt(g)"]).T
    if check_derivs("e^zeta", transforms["R"], transforms["Z"]):
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

    if check_derivs("sqrt(g)", transforms["R"], transforms["Z"]):
        data["sqrt(g)"] = dot(data["e_rho"], cross(data["e_theta"], data["e_zeta"]))
        data["|e_theta x e_zeta|"] = jnp.linalg.norm(
            cross(data["e_theta"], data["e_zeta"]), axis=1
        )
        data["|e_zeta x e_rho|"] = jnp.linalg.norm(
            cross(data["e_zeta"], data["e_rho"]), axis=1
        )
        data["|e_rho x e_theta|"] = jnp.linalg.norm(
            cross(data["e_rho"], data["e_theta"]), axis=1
        )

    # 1st order derivatives
    if check_derivs("sqrt(g)_r", transforms["R"], transforms["Z"]):
        data["sqrt(g)_r"] = (
            dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_r"]))
        )
    if check_derivs("sqrt(g)_t", transforms["R"], transforms["Z"]):
        data["sqrt(g)_t"] = (
            dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_t"]))
        )
    if check_derivs("sqrt(g)_z", transforms["R"], transforms["Z"]):
        data["sqrt(g)_z"] = (
            dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_z"]))
        )

    # 2nd order derivatives
    if check_derivs("sqrt(g)_rr", transforms["R"], transforms["Z"]):
        data["sqrt(g)_rr"] = (
            dot(data["e_rho_rr"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_rr"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_rr"]))
            + 2 * dot(data["e_rho_r"], cross(data["e_theta_r"], data["e_zeta"]))
            + 2 * dot(data["e_rho_r"], cross(data["e_theta"], data["e_zeta_r"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_r"], data["e_zeta_r"]))
        )
    if check_derivs("sqrt(g)_tt", transforms["R"], transforms["Z"]):
        data["sqrt(g)_tt"] = (
            dot(data["e_rho_tt"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_tt"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_tt"]))
            + 2 * dot(data["e_rho_t"], cross(data["e_theta_t"], data["e_zeta"]))
            + 2 * dot(data["e_rho_t"], cross(data["e_theta"], data["e_zeta_t"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_t"], data["e_zeta_t"]))
        )
    if check_derivs("sqrt(g)_zz", transforms["R"], transforms["Z"]):
        data["sqrt(g)_zz"] = (
            dot(data["e_rho_zz"], cross(data["e_theta"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta_zz"], data["e_zeta"]))
            + dot(data["e_rho"], cross(data["e_theta"], data["e_zeta_zz"]))
            + 2 * dot(data["e_rho_z"], cross(data["e_theta_z"], data["e_zeta"]))
            + 2 * dot(data["e_rho_z"], cross(data["e_theta"], data["e_zeta_z"]))
            + 2 * dot(data["e_rho"], cross(data["e_theta_z"], data["e_zeta_z"]))
        )
    if check_derivs("sqrt(g)_tz", transforms["R"], transforms["Z"]):
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

    if check_derivs("g_rr", transforms["R"], transforms["Z"]):
        data["g_rr"] = dot(data["e_rho"], data["e_rho"])
    if check_derivs("g_tt", transforms["R"], transforms["Z"]):
        data["g_tt"] = dot(data["e_theta"], data["e_theta"])
    if check_derivs("g_zz", transforms["R"], transforms["Z"]):
        data["g_zz"] = dot(data["e_zeta"], data["e_zeta"])
    if check_derivs("g_rt", transforms["R"], transforms["Z"]):
        data["g_rt"] = dot(data["e_rho"], data["e_theta"])
    if check_derivs("g_rz", transforms["R"], transforms["Z"]):
        data["g_rz"] = dot(data["e_rho"], data["e_zeta"])
    if check_derivs("g_tz", transforms["R"], transforms["Z"]):
        data["g_tz"] = dot(data["e_theta"], data["e_zeta"])

    if check_derivs("g_tt_r", transforms["R"], transforms["Z"]):
        data["g_tt_r"] = 2 * dot(data["e_theta"], data["e_theta_r"])
    if check_derivs("g_tz_r", transforms["R"], transforms["Z"]):
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

    if check_derivs("g^rr", transforms["R"], transforms["Z"]):
        data["g^rr"] = dot(data["e^rho"], data["e^rho"])
    if check_derivs("g^tt", transforms["R"], transforms["Z"]):
        data["g^tt"] = dot(data["e^theta"], data["e^theta"])
    if check_derivs("g^zz", transforms["R"], transforms["Z"]):
        data["g^zz"] = dot(data["e^zeta"], data["e^zeta"])
    if check_derivs("g^rt", transforms["R"], transforms["Z"]):
        data["g^rt"] = dot(data["e^rho"], data["e^theta"])
    if check_derivs("g^rz", transforms["R"], transforms["Z"]):
        data["g^rz"] = dot(data["e^rho"], data["e^zeta"])
    if check_derivs("g^tz", transforms["R"], transforms["Z"]):
        data["g^tz"] = dot(data["e^theta"], data["e^zeta"])

    if check_derivs("|grad(rho)|", transforms["R"], transforms["Z"]):
        data["|grad(rho)|"] = jnp.sqrt(data["g^rr"])
    if check_derivs("|grad(theta)|", transforms["R"], transforms["Z"]):
        data["|grad(theta)|"] = jnp.sqrt(data["g^tt"])
    if check_derivs("|grad(zeta)|", transforms["R"], transforms["Z"]):
        data["|grad(zeta)|"] = jnp.sqrt(data["g^zz"])

    return data


def compute_toroidal_flux_gradient(
    params,
    transforms,
    profiles,
    data=None,
    **kwargs,
):
    """Compute reciprocal metric coefficients."""
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

    if check_derivs("grad(psi)", transforms["R"], transforms["Z"]):
        data["grad(psi)"] = (data["psi_r"] * data["e^rho"].T).T
    if check_derivs("|grad(psi)|", transforms["R"], transforms["Z"]):
        data["|grad(psi)|^2"] = data["psi_r"] ** 2 * data["g^rr"]
        data["|grad(psi)|"] = jnp.sqrt(data["|grad(psi)|^2"])

    return data


def compute_geometry(params, transforms, profiles, data=None, **kwargs):
    """Compute geometric quantities such as plasma volume, aspect ratio, etc."""
    grid = transforms["R"].grid
    data = compute_jacobian(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    if check_derivs("V(r)", transforms["R"], transforms["Z"]):
        # divergence theorem: integral(dV div [0, 0, Z]) = integral(dS dot [0, 0, Z])
        data["V(r)"] = jnp.abs(
            surface_integrals(
                grid, cross(data["e_theta"], data["e_zeta"])[:, 2] * data["Z"]
            )
        )
    if check_derivs("V_r(r)", transforms["R"], transforms["Z"]):
        # eq. 4.9.10 in W.D. D'haeseleer et al. (1991) doi:10.1007/978-3-642-75595-8.
        data["V_r(r)"] = surface_integrals(grid, jnp.abs(data["sqrt(g)"]))
        data["S(r)"] = surface_integrals(grid, data["|e_theta x e_zeta|"])
        data["V"] = jnp.sum(jnp.abs(data["sqrt(g)"]) * grid.weights)
        data["A"] = jnp.mean(
            surface_integrals(
                grid, jnp.abs(data["sqrt(g)"] / data["R"]), surface_label="zeta"
            )
        )
        data["R0"] = data["V"] / (2 * jnp.pi * data["A"])
        data["a"] = jnp.sqrt(data["A"] / jnp.pi)
        data["R0/a"] = data["V"] / (2 * jnp.sqrt(jnp.pi * data["A"] ** 3))
    if check_derivs("V_rr(r)", transforms["R"], transforms["Z"]):
        data["V_rr(r)"] = surface_integrals(
            grid, data["sqrt(g)_r"] * jnp.sign(data["sqrt(g)"])
        )

    return data
