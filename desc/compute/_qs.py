"""Compute functions for quasisymmetry objectives."""

from desc.backend import jnp, put, sign

from ._field import (
    compute_B_dot_gradB,
    compute_boozer_magnetic_field,
    compute_covariant_magnetic_field,
    compute_magnetic_field_magnitude,
)
from .utils import has_dependencies


def compute_boozer_coordinates(params, transforms, profiles, data=None, **kwargs):
    """Compute Boozer coordinates.

    Assumes transform grids are uniform spacing on single flux surface without symmetry.
    """
    data = compute_magnetic_field_magnitude(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    # TODO: can remove this call if compute_|B| changed to use B_covariant
    data = compute_covariant_magnetic_field(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    # covariant Boozer components: I = B_theta, G = B_zeta (in Boozer coordinates)
    if "B" in transforms:
        idx0 = transforms["B"].basis.get_idx(M=0, N=0)
        B_theta_mn = transforms["B"].fit(data["B_theta"])
        B_zeta_mn = transforms["B"].fit(data["B_zeta"])
        data["I"] = B_theta_mn[idx0]
        data["G"] = B_zeta_mn[idx0]

    # w (RHS of eq 10 in Hirshman 1995 "Transformation from VMEC to Boozer Coordinates")
    if "w" in transforms:
        w_mn = jnp.zeros((transforms["w"].basis.num_modes,))
        NFP = transforms["w"].basis.NFP
        for k, (l, m, n) in enumerate(transforms["w"].basis.modes):
            if m != 0:
                idx = transforms["B"].basis.get_idx(M=-m, N=n)
                w_mn = put(w_mn, k, (sign(n) * B_theta_mn[idx] / jnp.abs(m))[0])
            elif n != 0:
                idx = transforms["B"].basis.get_idx(M=m, N=-n)
                w_mn = put(w_mn, k, (sign(m) * B_zeta_mn[idx] / jnp.abs(NFP * n))[0])

    # nu = zeta_Boozer - zeta   # noqa: E800
    GI = data["G"] + data["iota"] * data["I"]
    if has_dependencies("nu", params, transforms, profiles, data):
        w = transforms["w"].transform(w_mn)
        data["nu"] = (w - data["I"] * data["lambda"]) / GI
    if has_dependencies("nu_t", params, transforms, profiles, data):
        w_t = transforms["w"].transform(w_mn, dr=0, dt=1, dz=0)
        data["nu_t"] = (w_t - data["I"] * data["lambda_t"]) / GI
    if has_dependencies("nu_z", params, transforms, profiles, data):
        w_z = transforms["w"].transform(w_mn, dr=0, dt=0, dz=1)
        data["nu_z"] = (w_z - data["I"] * data["lambda_z"]) / GI

    # Boozer angles
    if has_dependencies("theta_B", params, transforms, profiles, data):
        data["theta_B"] = data["theta"] + data["lambda"] + data["iota"] * data["nu"]
    if has_dependencies("zeta_B", params, transforms, profiles, data):
        data["zeta_B"] = data["zeta"] + data["nu"]

    # Jacobian of Boozer coordinates wrt (theta,zeta) coordinates
    if has_dependencies("sqrt(g)_B", params, transforms, profiles, data):
        data["sqrt(g)_B"] = (1 + data["lambda_t"]) * (1 + data["nu_z"]) + (
            data["iota"] - data["lambda_z"]
        ) * data["nu_t"]

    # Riemann sum integration
    if has_dependencies("|B|_mn", params, transforms, profiles, data):
        nodes = jnp.array([data["rho"], data["theta_B"], data["zeta_B"]]).T
        norm = 2 ** (3 - jnp.sum((transforms["B"].basis.modes == 0), axis=1))
        data["|B|_mn"] = (
            norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            * jnp.matmul(
                transforms["B"].basis.evaluate(nodes).T, data["sqrt(g)_B"] * data["|B|"]
            )
            / transforms["B"].grid.num_nodes
        )
    if has_dependencies("B modes", params, transforms, profiles, data):
        data["B modes"] = transforms["B"].basis.modes

    return data


def compute_quasisymmetry_error(params, transforms, profiles, data=None, **kwargs):
    """Compute quasi-symmetry triple product and two-term errors.

    f_C computation assumes transform grids are a single flux surface.
    """
    data = compute_B_dot_gradB(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )
    data = compute_boozer_magnetic_field(
        params,
        transforms,
        profiles,
        data=data,
        **kwargs,
    )

    M = kwargs.get("helicity", (1, 0))[0]
    N = kwargs.get("helicity", (1, 0))[1]

    # QS two-term (T^3)
    if has_dependencies("f_C", params, transforms, profiles, data):
        data["f_C"] = (M * data["iota"] - N) * (data["psi_r"] / data["sqrt(g)"]) * (
            data["B_zeta"] * data["|B|_t"] - data["B_theta"] * data["|B|_z"]
        ) - (M * data["G"] + N * data["I"]) * data["B*grad(|B|)"]

    # QS triple product (T^4/m^2)
    if has_dependencies("f_T", params, transforms, profiles, data):
        data["f_T"] = (data["psi_r"] / data["sqrt(g)"]) * (
            data["|B|_t"] * data["(B*grad(|B|))_z"]
            - data["|B|_z"] * data["(B*grad(|B|))_t"]
        )

    return data
