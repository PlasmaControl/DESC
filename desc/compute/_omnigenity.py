"""Compute functions for omnigenity objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

import functools

from interpax import interp1d

from desc.backend import jnp, sign, vmap, switch, cond, gammaln
from jax.lax import dynamic_slice

from ..utils import cross, dot, safediv
from .data_index import register_compute_fun


@register_compute_fun(
    name="B_theta_mn",
    label="B_{\\theta, m, n}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Fourier coefficients for covariant poloidal component of "
    "magnetic field.",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    data=["B_theta"],
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _B_theta_mn(params, transforms, profiles, data, **kwargs):
    B_theta = transforms["grid"].meshgrid_reshape(data["B_theta"], "rtz")

    def fitfun(x):
        return transforms["B"].fit(x.flatten(order="F"))

    B_theta_mn = vmap(fitfun)(B_theta)
    # modes stored as shape(rho, mn) flattened
    data["B_theta_mn"] = B_theta_mn.flatten()
    return data


# TODO (#568): do math to change definition of nu so that we can just use B_zeta_mn here
@register_compute_fun(
    name="B_phi_mn",
    label="B_{\\phi, m, n}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Fourier coefficients for covariant toroidal component of "
    "magnetic field in (ρ,θ,ϕ) coordinates.",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["B_phi|r,t"],
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    aliases="B_zeta_mn",  # TODO(#568): remove when phi != zeta
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _B_phi_mn(params, transforms, profiles, data, **kwargs):
    B_phi = transforms["grid"].meshgrid_reshape(data["B_phi|r,t"], "rtz")

    def fitfun(x):
        return transforms["B"].fit(x.flatten(order="F"))

    B_zeta_mn = vmap(fitfun)(B_phi)
    # modes stored as shape(rho, mn) flattened
    data["B_phi_mn"] = B_zeta_mn.flatten()
    return data


@register_compute_fun(
    name="w_Boozer_mn",
    label="w_{\\mathrm{Boozer},m,n}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="RHS of eq 10 in Hirshman 1995 'Transformation from VMEC to "
    + "Boozer Coordinates'",
    dim=1,
    params=[],
    transforms={"w": [[0, 0, 0]], "B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    data=["B_theta_mn", "B_phi_mn"],
    grid_requirement={"is_meshgrid": True, "sym": False},
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _w_mn(params, transforms, profiles, data, **kwargs):
    w_mn = jnp.zeros((transforms["grid"].num_rho, transforms["w"].basis.num_modes))
    Bm = transforms["B"].basis.modes[:, 1]
    Bn = transforms["B"].basis.modes[:, 2]
    wm = transforms["w"].basis.modes[:, 1]
    wn = transforms["w"].basis.modes[:, 2]
    NFP = transforms["w"].basis.NFP
    mask_t = (Bm[:, None] == -wm) & (Bn[:, None] == wn) & (wm != 0)
    mask_z = (Bm[:, None] == wm) & (Bn[:, None] == -wn) & (wm == 0) & (wn != 0)

    num_t = (mask_t @ sign(wn)) * data["B_theta_mn"].reshape(
        (transforms["grid"].num_rho, -1)
    )
    den_t = mask_t @ jnp.abs(wm)
    num_z = (mask_z @ sign(wm)) * data["B_phi_mn"].reshape(
        (transforms["grid"].num_rho, -1)
    )
    den_z = mask_z @ jnp.abs(NFP * wn)

    w_mn = jnp.where(mask_t.any(axis=0), (mask_t.T @ safediv(num_t, den_t).T).T, w_mn)
    w_mn = jnp.where(mask_z.any(axis=0), (mask_z.T @ safediv(num_z, den_z).T).T, w_mn)

    data["w_Boozer_mn"] = w_mn.flatten()
    return data


@register_compute_fun(
    name="w_Boozer",
    label="w_{\\mathrm{Boozer}}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Inverse Fourier transform of RHS of eq 10 in Hirshman 1995 "
    + "'Transformation from VMEC to Boozer Coordinates'",
    dim=1,
    params=[],
    transforms={"w": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer_mn"],
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _w(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"]
    w_mn = data["w_Boozer_mn"].reshape((grid.num_rho, -1))
    w = vmap(transforms["w"].transform)(w_mn)  # shape(rho, theta*zeta)
    w = w.reshape((grid.num_rho, grid.num_theta, grid.num_zeta), order="F")
    w = jnp.moveaxis(w, 0, 1)
    data["w_Boozer"] = w.flatten(order="F")
    return data


@register_compute_fun(
    name="w_Boozer_t",
    label="\\partial_{\\theta} w_{\\mathrm{Boozer}}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Inverse Fourier transform of RHS of eq 10 in Hirshman 1995 "
    + "'Transformation from VMEC to Boozer Coordinates', poloidal derivative",
    dim=1,
    params=[],
    transforms={"w": [[0, 1, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer_mn"],
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _w_t(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"]
    w_mn = data["w_Boozer_mn"].reshape((grid.num_rho, -1))
    # need to close over dt which can't be vmapped
    fun = lambda x: transforms["w"].transform(x, dt=1)
    w_t = vmap(fun)(w_mn)  # shape(rho, theta*zeta)
    w_t = w_t.reshape((grid.num_rho, grid.num_theta, grid.num_zeta), order="F")
    w_t = jnp.moveaxis(w_t, 0, 1)
    data["w_Boozer_t"] = w_t.flatten(order="F")
    return data


@register_compute_fun(
    name="w_Boozer_z",
    label="\\partial_{\\zeta} w_{\\mathrm{Boozer}}",
    units="T \\cdot m",
    units_long="Tesla * meters",
    description="Inverse Fourier transform of RHS of eq 10 in Hirshman 1995 "
    + "'Transformation from VMEC to Boozer Coordinates', toroidal derivative",
    dim=1,
    params=[],
    transforms={"w": [[0, 0, 1]], "grid": []},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer_mn"],
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _w_z(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"]
    w_mn = data["w_Boozer_mn"].reshape((grid.num_rho, -1))
    # need to close over dz which can't be vmapped
    fun = lambda x: transforms["w"].transform(x, dz=1)
    w_z = vmap(fun)(w_mn)  # shape(rho, theta*zeta)
    w_z = w_z.reshape((grid.num_rho, grid.num_theta, grid.num_zeta), order="F")
    w_z = jnp.moveaxis(w_z, 0, 1)
    data["w_Boozer_z"] = w_z.flatten(order="F")
    return data


@register_compute_fun(
    name="nu",
    label="\\nu = \\zeta_{B} - \\zeta",
    units="rad",
    units_long="radians",
    description="Boozer toroidal stream function",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer", "G", "I", "iota", "lambda"],
)
def _nu(params, transforms, profiles, data, **kwargs):
    GI = data["G"] + data["iota"] * data["I"]
    data["nu"] = (data["w_Boozer"] - data["I"] * data["lambda"]) / GI
    return data


@register_compute_fun(
    name="nu_B_mn",
    label="\\nu_{mn} = (\\zeta_{B} - \\zeta)_{mn}",
    units="rad",
    units_long="radians",
    description="Boozer harmonics of Boozer toroidal stream function",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "sqrt(g)_Boozer_DESC",
        "nu",
        "rho",
        "theta_B",
        "zeta_B",
        "Boozer transform modes norm",
    ],
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _nu_B_mn(params, transforms, profiles, data, **kwargs):
    norm = data["Boozer transform modes norm"]
    grid = transforms["grid"]

    def fun(rho, theta_B, zeta_B, sqrtg_B_desc, quant):
        # this fits Boozer modes on a single surface
        nodes = jnp.array([rho, theta_B, zeta_B]).T
        quant_mn = (
            norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            * (transforms["B"].basis.evaluate(nodes).T @ (sqrtg_B_desc * quant))
            / transforms["B"].grid.num_nodes
        )
        return quant_mn

    def reshape(x):
        return grid.meshgrid_reshape(x, "rtz").reshape((grid.num_rho, -1))

    rho, theta_B, zeta_B, sqrtg_B_desc, nu = map(
        reshape,
        (
            data["rho"],
            data["theta_B"],
            data["zeta_B"],
            data["sqrt(g)_Boozer_DESC"],
            data["nu"],
        ),
    )
    nu_B_mn = vmap(fun)(rho, theta_B, zeta_B, sqrtg_B_desc, nu)
    data["nu_B_mn"] = nu_B_mn.flatten()
    return data


@register_compute_fun(
    name="nu_t",
    label="\\partial_{\\theta} \\nu",
    units="rad",
    units_long="radians",
    description="Boozer toroidal stream function, derivative wrt poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer_t", "G", "I", "iota", "lambda_t"],
)
def _nu_t(params, transforms, profiles, data, **kwargs):
    GI = data["G"] + data["iota"] * data["I"]
    data["nu_t"] = (data["w_Boozer_t"] - data["I"] * data["lambda_t"]) / GI
    return data


@register_compute_fun(
    name="nu_z",
    label="\\partial_{\\zeta} \\nu",
    units="rad",
    units_long="radians",
    description="Boozer toroidal stream function, derivative wrt toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer_z", "G", "I", "iota", "lambda_z"],
)
def _nu_z(params, transforms, profiles, data, **kwargs):
    GI = data["G"] + data["iota"] * data["I"]
    data["nu_z"] = (data["w_Boozer_z"] - data["I"] * data["lambda_z"]) / GI
    return data


@register_compute_fun(
    name="theta_B",
    label="\\theta_{B}",
    units="rad",
    units_long="radians",
    description="Boozer poloidal angular coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST", "iota", "nu"],
)
def _theta_B(params, transforms, profiles, data, **kwargs):
    data["theta_B"] = data["theta_PEST"] + data["iota"] * data["nu"]
    return data


@register_compute_fun(
    name="zeta_B",
    label="\\zeta_{B}",
    units="rad",
    units_long="radians",
    description="Boozer toroidal angular coordinate",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["phi", "nu"],
)
def _zeta_B(params, transforms, profiles, data, **kwargs):
    data["zeta_B"] = data["phi"] + data["nu"]
    return data


@register_compute_fun(
    name="sqrt(g)_Boozer_DESC",
    label="\\frac{\\partial(\\theta_B,\\zeta_B)}{\\theta_{DESC},\\zeta_{DESC}}",
    units="~",
    units_long="None",
    description="Jacobian determinant from Boozer coordinates (rho, theta_B, zeta_B)"
    " to DESC coordinates (rho,theta,zeta).",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST_t", "theta_PEST_z", "phi_t", "phi_z", "nu_t", "nu_z", "iota"],
    aliases=["sqrt(g)_B"],
)
def _sqrt_g_Boozer_DESC(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_Boozer_DESC"] = (
        data["theta_PEST_t"] * (data["phi_z"] + data["nu_z"])
        - data["theta_PEST_z"] * (data["phi_t"] + data["nu_t"])
        + data["iota"] * (data["nu_t"] * data["phi_z"] - data["nu_z"] * data["phi_t"])
    )
    return data


@register_compute_fun(
    name="sqrt(g)_Boozer",
    label="\\sqrt{g}_Boozer",
    units="m^{3}",
    units_long="cubic meters",
    description="Jacobian determinant from (rho, theta_B, zeta_B)"
    " Boozer coordinates to (R,phi,Z) lab frame.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)_Boozer_DESC", "sqrt(g)"],
)
def _sqrtg_B(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_Boozer"] = data["sqrt(g)"] / data["sqrt(g)_Boozer_DESC"]
    return data


@register_compute_fun(
    name="sqrt(g)_Boozer_mn",
    label="\\sqrt{g}_{B,mn}",
    units="m^{3}",
    units_long="cubic meters",
    description="Boozer harmonics of Jacobian determinant from (rho, theta_B, zeta_B)"
    " Boozer coordinates to (R,phi,Z) lab frame.",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    data=[
        "sqrt(g)_Boozer",
        "sqrt(g)_Boozer_DESC",
        "rho",
        "theta_B",
        "zeta_B",
        "Boozer transform modes norm",
    ],
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _sqrtg_Boozer_mn(params, transforms, profiles, data, **kwargs):
    norm = data["Boozer transform modes norm"]
    grid = transforms["grid"]

    def fun(rho, theta_B, zeta_B, sqrtg_B_desc, quant):
        # this fits Boozer modes on a single surface
        nodes = jnp.array([rho, theta_B, zeta_B]).T
        quant_mn = (
            norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            * (transforms["B"].basis.evaluate(nodes).T @ (sqrtg_B_desc * quant))
            / transforms["B"].grid.num_nodes
        )
        return quant_mn

    def reshape(x):
        return grid.meshgrid_reshape(x, "rtz").reshape((grid.num_rho, -1))

    rho, theta_B, zeta_B, sqrtg_B_desc, sqrtg_B = map(
        reshape,
        (
            data["rho"],
            data["theta_B"],
            data["zeta_B"],
            data["sqrt(g)_Boozer_DESC"],
            data["sqrt(g)_Boozer"],
        ),
    )
    sqrtg_B_mn = vmap(fun)(rho, theta_B, zeta_B, sqrtg_B_desc, sqrtg_B)
    data["sqrt(g)_Boozer_mn"] = sqrtg_B_mn.flatten()
    return data


@register_compute_fun(
    name="|B|_mn_B",
    label="B_{mn}^{\\mathrm{Boozer}}",
    units="T",
    units_long="Tesla",
    description="Boozer harmonics of magnetic field",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "sqrt(g)_Boozer_DESC",
        "|B|",
        "rho",
        "theta_B",
        "zeta_B",
        "Boozer transform modes norm",
    ],
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
    aliases=["|B|_mn"],
)
def _B_mn(params, transforms, profiles, data, **kwargs):
    norm = data["Boozer transform modes norm"]
    grid = transforms["grid"]

    def fun(rho, theta_B, zeta_B, sqrtg_B_desc, quant):
        # this fits Boozer modes on a single surface
        nodes = jnp.array([rho, theta_B, zeta_B]).T
        B_mn = (
            norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            * (transforms["B"].basis.evaluate(nodes).T @ (sqrtg_B_desc * quant))
            / transforms["B"].grid.num_nodes
        )
        return B_mn

    def reshape(x):
        return grid.meshgrid_reshape(x, "rtz").reshape((grid.num_rho, -1))

    rho, theta_B, zeta_B, sqrtg_B_desc, B = map(
        reshape,
        (
            data["rho"],
            data["theta_B"],
            data["zeta_B"],
            data["sqrt(g)_Boozer_DESC"],
            data["|B|"],
        ),
    )
    B_mn = vmap(fun)(rho, theta_B, zeta_B, sqrtg_B_desc, B)
    data["|B|_mn_B"] = B_mn.flatten()
    return data


@register_compute_fun(
    name="R_mn_B",
    label="R_{mn}^{\\mathrm{Boozer}}",
    units="m",
    units_long="meters",
    description="Boozer harmonics of radial toroidal coordinate of a flux surface",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    data=[
        "R",
        "sqrt(g)_Boozer_DESC",
        "rho",
        "theta_B",
        "zeta_B",
        "Boozer transform modes norm",
    ],
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _R_mn(params, transforms, profiles, data, **kwargs):
    norm = data["Boozer transform modes norm"]
    grid = transforms["grid"]

    def fun(rho, theta_B, zeta_B, sqrtg_B_desc, quant):
        # this fits Boozer modes on a single surface
        nodes = jnp.array([rho, theta_B, zeta_B]).T
        quant_mn = (
            norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            * (transforms["B"].basis.evaluate(nodes).T @ (sqrtg_B_desc * quant))
            / transforms["B"].grid.num_nodes
        )
        return quant_mn

    def reshape(x):
        return grid.meshgrid_reshape(x, "rtz").reshape((grid.num_rho, -1))

    rho, theta_B, zeta_B, sqrtg_B_desc, R = map(
        reshape,
        (
            data["rho"],
            data["theta_B"],
            data["zeta_B"],
            data["sqrt(g)_Boozer_DESC"],
            data["R"],
        ),
    )
    R_mn = vmap(fun)(rho, theta_B, zeta_B, sqrtg_B_desc, R)
    data["R_mn_B"] = R_mn.flatten()
    return data


@register_compute_fun(
    name="Z_mn_B",
    label="Z_{mn}^{\\mathrm{Boozer}}",
    units="m",
    units_long="meters",
    description="Boozer harmonics of vertical coordinate of a flux surface",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True, "sym": False},
    data=[
        "Z",
        "sqrt(g)_Boozer_DESC",
        "rho",
        "theta_B",
        "zeta_B",
        "Boozer transform modes norm",
    ],
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _Z_mn(params, transforms, profiles, data, **kwargs):
    norm = data["Boozer transform modes norm"]
    grid = transforms["grid"]

    def fun(rho, theta_B, zeta_B, sqrtg_B_desc, quant):
        # this fits Boozer modes on a single surface
        nodes = jnp.array([rho, theta_B, zeta_B]).T
        quant_mn = (
            norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            * (transforms["B"].basis.evaluate(nodes).T @ (sqrtg_B_desc * quant))
            / transforms["B"].grid.num_nodes
        )
        return quant_mn

    def reshape(x):
        return grid.meshgrid_reshape(x, "rtz").reshape((grid.num_rho, -1))

    rho, theta_B, zeta_B, sqrtg_B_desc, Z = map(
        reshape,
        (
            data["rho"],
            data["theta_B"],
            data["zeta_B"],
            data["sqrt(g)_Boozer_DESC"],
            data["Z"],
        ),
    )
    Z_mn = vmap(fun)(rho, theta_B, zeta_B, sqrtg_B_desc, Z)
    data["Z_mn_B"] = Z_mn.flatten()
    return data


@register_compute_fun(
    name="B modes",
    label="\\mathrm{Boozer~modes}",
    units="~",
    units_long="None",
    description="Boozer harmonics",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _B_modes(params, transforms, profiles, data, **kwargs):
    data["B modes"] = transforms["B"].basis.modes
    return data


@register_compute_fun(
    name="Boozer transform modes norm",
    label="",
    units="~",
    units_long="None",
    description="Inner product norm for boozer modes basis. This norm is used as a"
    "weight when performing the integral of the Boozer transform to get the "
    "correct Boozer Fourier amplitudes.",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _boozer_modes_norm(params, transforms, profiles, data, **kwargs):
    # norm is 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
    norm = 2 ** (3 - jnp.sum((transforms["B"].basis.modes == 0), axis=1))
    data["Boozer transform modes norm"] = norm
    return data


@register_compute_fun(
    name="f_C",
    label="[(M \\iota - N) (\\mathbf{B} \\times \\nabla \\psi)"
    + " - (M G + N I) \\mathbf{B}] \\cdot \\nabla B",
    units="T^{3}",
    units_long="Tesla cubed",
    description="Two-term quasisymmetry metric",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "iota",
        "psi_r/sqrt(g)",
        "B_theta",
        "B_zeta",
        "|B|_t",
        "|B|_z",
        "G",
        "I",
        "B*grad(|B|)",
    ],
    helicity="tuple: Type of quasisymmetry, (M,N). Default (1,0)",
)
def _f_C(params, transforms, profiles, data, **kwargs):
    M, N = kwargs.get("helicity", (1, 0))
    data["f_C"] = (M * data["iota"] - N) * data["psi_r/sqrt(g)"] * (
        data["B_zeta"] * data["|B|_t"] - data["B_theta"] * data["|B|_z"]
    ) - (M * data["G"] + N * data["I"]) * data["B*grad(|B|)"]
    return data


@register_compute_fun(
    name="f_T",
    label="\\nabla \\psi \\times \\nabla B \\cdot \\nabla "
    + "(\\mathbf{B} \\cdot \\nabla B)",
    units="T^{4} \\cdot m^{-2}",
    units_long="Tesla quarted / square meters",
    description="Triple product quasisymmetry metric",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["psi_r/sqrt(g)", "|B|_t", "|B|_z", "(B*grad(|B|))_t", "(B*grad(|B|))_z"],
)
def _f_T(params, transforms, profiles, data, **kwargs):
    data["f_T"] = data["psi_r/sqrt(g)"] * (
        data["|B|_t"] * data["(B*grad(|B|))_z"]
        - data["|B|_z"] * data["(B*grad(|B|))_t"]
    )
    return data


@register_compute_fun(
    name="eta",
    label="\\eta",
    units="rad",
    units_long="radians",
    description="Intermediate omnigenity coordinate along field lines",
    dim=1,
    params=[],
    transforms={"h": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.magnetic_fields._core.OmnigenousField",
)
def _eta(params, transforms, profiles, data, **kwargs):
    data["eta"] = transforms["h"].grid.nodes[:, 1]
    return data


@register_compute_fun(
    name="alpha",
    label="\\alpha",
    units="rad",
    units_long="radians",
    description="Field line label, defined on [0, 2pi)",
    dim=1,
    params=[],
    transforms={"h": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.magnetic_fields._core.OmnigenousField",
)
def _alpha(params, transforms, profiles, data, **kwargs):
    data["alpha"] = transforms["h"].grid.nodes[:, 2]
    return data


@register_compute_fun(
    name="h",
    label="h = \\theta + (N / M) \\zeta",
    units="rad",
    units_long="radians",
    description="Omnigenity symmetry angle",
    dim=1,
    params=["x_lmn"],
    transforms={"h": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["eta"],
    resolution_requirement="tz",
    parameterization="desc.magnetic_fields._core.OmnigenousField",
)
def _omni_angle(params, transforms, profiles, data, **kwargs):
    data["h"] = transforms["h"].transform(params["x_lmn"]) + 2 * data["eta"] + jnp.pi
    return data


@register_compute_fun(
    name="theta_B",
    label="\\theta_{B}",
    units="rad",
    units_long="radians",
    description="Boozer poloidal angle",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["alpha", "h"],
    parameterization="desc.magnetic_fields._core.OmnigenousField",
    helicity="tuple: Type of quasisymmetry, (M,N). Default (1,0)",
    iota="float: Value of rotational transform on the Omnigenous surface. Default 1.0",
)
def _omni_map_theta_B(params, transforms, profiles, data, **kwargs):
    M, N = kwargs.get("helicity", (1, 0))
    iota = kwargs.get("iota", jnp.ones(transforms["grid"].num_rho))

    theta_B, zeta_B = _omnigenity_mapping(
        M, N, iota, data["alpha"], data["h"], transforms["grid"]
    )
    data["theta_B"] = theta_B
    data["zeta_B"] = zeta_B
    return data


def _omnigenity_mapping(M, N, iota, alpha, h, grid):
    iota = jnp.atleast_1d(iota)
    assert (
        len(iota) == grid.num_rho
    ), f"got ({len(iota)}) iota values for grid with {grid.num_rho} surfaces"
    matrix = jnp.atleast_3d(_omnigenity_mapping_matrix(M, N, iota))
    # solve for (theta_B,zeta_B) corresponding to (eta,alpha)
    alpha = grid.meshgrid_reshape(alpha, "trz")
    h = grid.meshgrid_reshape(h, "trz")
    coords = jnp.stack((alpha, h))
    # matrix has shape (nr,2,2), coords is shape (2, nt, nr, nz)
    # we vectorize the matmul over rho
    booz = jnp.einsum("rij,jtrz->itrz", matrix, coords)
    theta_B = booz[0].flatten(order="F")
    zeta_B = booz[1].flatten(order="F")
    return theta_B, zeta_B


@functools.partial(jnp.vectorize, signature="(),(),()->(2,2)")
def _omnigenity_mapping_matrix(M, N, iota):
    # need a bunch of wheres to avoid division by zero causing NaN in backward pass
    # this is fine since the incorrect values get ignored later, except in OT or OH
    # where fieldlines are exactly parallel to |B| contours, but this is a degenerate
    # case of measure 0 so this kludge shouldn't affect things too much.
    mat_OP = jnp.array(
        [[N, iota / jnp.where(N == 0, 1, N)], [0, 1 / jnp.where(N == 0, 1, N)]]
    )
    mat_OT = jnp.array([[0, -1], [M, -1 / jnp.where(iota == 0, 1.0, iota)]])
    den = jnp.where((N - M * iota) == 0, 1.0, (N - M * iota))
    mat_OH = jnp.array([[N, M * iota / den], [M, M / den]])
    matrix = jnp.where(
        M == 0,
        mat_OP,
        jnp.where(
            N == 0,
            mat_OT,
            mat_OH,
        ),
    )
    return matrix


@register_compute_fun(
    name="zeta_B",
    label="\\zeta_{B}",
    units="rad",
    units_long="radians",
    description="Boozer toroidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_B"],
    parameterization="desc.magnetic_fields._core.OmnigenousField",
)
def _omni_map_zeta_B(params, transforms, profiles, data, **kwargs):
    return data  # noqa: unused dependency


@register_compute_fun(
    name="|B|",
    label="|\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of omnigenous magnetic field",
    dim=1,
    params=["B_lm"],
    transforms={"grid": [], "B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["eta"],
    parameterization="desc.magnetic_fields._core.OmnigenousField",
)
def _B_omni(params, transforms, profiles, data, **kwargs):
    # reshaped to size (L_B, M_B)
    B_lm = params["B_lm"].reshape((transforms["B"].basis.L + 1, -1))

    def _transform(x):
        y = transforms["B"].transform(x)
        return transforms["grid"].compress(y)

    B_input = vmap(_transform)(B_lm.T)
    # B_input has shape (num_knots, num_rho)
    B_input = jnp.sort(B_input, axis=0)  # sort to ensure monotonicity
    eta_input = jnp.linspace(0, jnp.pi / 2, num=B_input.shape[0])
    eta = transforms["grid"].meshgrid_reshape(data["eta"], "rtz")
    eta = eta.reshape((transforms["grid"].num_rho, -1))

    def _interp(x, B):
        return interp1d(x, eta_input, B, method="monotonic-0")

    # |B|_omnigeneous is an even function so B(-eta) = B(+eta) = B(|eta|)
    B = vmap(_interp)(jnp.abs(eta), B_input.T)  # shape (nr, nt*nz)
    B = B.reshape(
        (
            transforms["grid"].num_rho,
            transforms["grid"].num_poloidal,
            transforms["grid"].num_zeta,
        )
    )
    B = jnp.moveaxis(B, 0, 1)
    data["|B|"] = B.flatten(order="F")
    return data


@register_compute_fun(
    name="isodynamicity",
    label="1/|B|^2 (\\mathbf{b} \\times \\nabla B) \\cdot \\nabla \\psi",
    units="~",
    units_long="None",
    description="Measure of cross field drift at each point, "
    + "unweighted by particle energy",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["b", "grad(|B|)", "|B|^2", "grad(psi)"],
)
def _isodynamicity(params, transforms, profiles, data, **kwargs):
    data["isodynamicity"] = (
        dot(cross(data["b"], data["grad(|B|)"]), data["grad(psi)"]) / data["|B|^2"]
    )
    return data


@register_compute_fun(
    name="S_list",
    label="S_{n}",
    units="~",
    units_long="None",
    description="Omnigenity S coefficients, used in OOPS/LCForm",
    dim=1,
    params=["S_list"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._core.OmnigenousFieldOOPS",
        "desc.magnetic_fields._core.OmnigenousFieldLCForm",
    ],
)
def _S_list(params, transforms, profiles, data, **kwargs):
    """
    S_list is a list of coefficients for the omnigenity S shape.
    It is used in OOPS/LCForm to define the omnigenity symmetry angle.
    """
    if "S_list" not in params:
        raise ValueError("S_list parameter is required for OOPS/LCForm")
    data["S_list"] = jnp.array(params["S_list"])
    return data


@register_compute_fun(
    name="D_list",
    label="D_{n}",
    units="~",
    units_long="None",
    description="Omnigenity D coefficients, used in OOPS/LCForm",
    dim=1,
    params=["D_list"],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization=[
        "desc.magnetic_fields._core.OmnigenousFieldOOPS",
        "desc.magnetic_fields._core.OmnigenousFieldLCForm",
    ],
)
def _D_list(params, transforms, profiles, data, **kwargs):
    """
    D_list is a list of coefficients for the omnigenity D shape.
    It is used in OOPS/LCForm to define the omnigenity symmetry angle.
    """
    if "D_list" not in params:
        raise ValueError("D_list parameter is required for OOPS/LCForm")
    data["D_list"] = jnp.array(params["D_list"])
    return data


@register_compute_fun(
    name="alpha_OOPS",
    label="\\alpha_OOPS",
    units="rad",
    units_long="radians",
    description="Field line label, defined on [0, 2pi)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldOOPS",
)
def _alpha_OOPS(params, transforms, profiles, data, **kwargs):
    data["alpha_OOPS"] = transforms["grid"].nodes[:, 1]
    return data


@register_compute_fun(
    name="eta_OOPS",
    label="\\eta_OOPS",
    units="rad",
    units_long="radians",
    description="Intermediate omnigenity coordinate along field lines",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldOOPS",
)
def _eta_OOPS(params, transforms, profiles, data, **kwargs):
    # we need rescale to [-pi,pi], nodes is divided by NFP, so we multiply by NFP
    data["eta_OOPS"] = transforms["grid"].nodes[:, 2] * transforms["grid"].NFP - jnp.pi
    return data


@register_compute_fun(
    name="zeta_B_OOPS",
    label="\\zeta_{B}_OOPS",
    units="rad",
    units_long="radians",
    description="Boozer toroidal angle",
    dim=1,
    params=["S_list", "D_list"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldOOPS",
    helicity="tuple: Type of quasisymmetry, (M,N). Default (1,0)",
    iota="float: Value of rotational transform on the Omnigenous surface. Default 1.0",
)
def _omni_map_zeta_B_OOPS(params, transforms, profiles, data, **kwargs):
    M = kwargs.get("helicity", (1, 0))[0]
    N = kwargs.get("helicity", (1, 0))[1]
    iota = kwargs.get("iota", jnp.ones(transforms["grid"].num_rho))
    S_list = params["S_list"]
    D_list = params["D_list"]

    theta_B, zeta_B = _omnigenity_mapping_OOPS(
        M, N, iota, S_list, D_list, transforms["grid"]
    )

    data["theta_B_OOPS"] = theta_B
    data["zeta_B_OOPS"] = zeta_B
    return data


def _generate_S_shape(S_list, y):
    # needed by OOPS
    n = S_list.size
    i = jnp.arange(1, n + 1, dtype=y.dtype).reshape((n,) + (1,) * y.ndim)
    return jnp.einsum("i...,i->...", jnp.sin(i * y[None, ...]), S_list)


def _generate_D_shape(D_list, x):
    # needed by OOPS
    n = D_list.size
    i = jnp.arange(n, dtype=x.dtype)
    k = ((2 * i + 1) / 2).reshape((n,) + (1,) * x.ndim)
    return jnp.einsum("i...,i->...", jnp.cos(k * x[None, ...]), D_list)


def _map_toroidal_OOPS(eta2d, alp2d, iota, nfp, S_list, D_list):
    S = _generate_S_shape(S_list, (alp2d - eta2d / iota) * nfp)
    D = _generate_D_shape(D_list, eta2d) + jnp.pi - jnp.abs(eta2d)
    h_o = eta2d - S * D
    theta2d_trans_real = h_o
    zeta2d_trans_real = alp2d
    return theta2d_trans_real, zeta2d_trans_real


def _map_poloidal_OOPS(eta2d, alp2d, iota, nfp, S_list, D_list):
    S = _generate_S_shape(S_list, alp2d - iota / nfp * eta2d)
    D = _generate_D_shape(D_list, eta2d) + jnp.pi - jnp.abs(eta2d)
    h_o = eta2d - S * D
    theta2d_trans_real = alp2d
    zeta2d_trans_real = h_o / nfp + jnp.pi / nfp
    return theta2d_trans_real, zeta2d_trans_real


def _map_helical_OOPS(eta2d, alp2d, iota, nfp, S_list, D_list):
    S = _generate_S_shape(S_list, (alp2d + 1 / (1 + nfp / iota) * eta2d))
    D = _generate_D_shape(D_list, eta2d) + jnp.pi - jnp.abs(eta2d)
    h_o = eta2d - S * D
    theta2d_trans_real = alp2d
    zeta2d_trans_real = -(h_o + alp2d) / nfp  # + jnp.pi/nfp
    return theta2d_trans_real, zeta2d_trans_real


OOPS_branches = (_map_poloidal_OOPS, _map_toroidal_OOPS, _map_helical_OOPS)


def _omnigenity_mapping_OOPS(M, N, iota, S_list, D_list, grid):
    # iota is a vector of length grid.num_rho
    iota = jnp.atleast_1d(iota)
    assert (
        len(iota) == grid.num_rho
    ), f"got ({len(iota)}) iota values for grid with {grid.num_rho} surfaces"

    NFP = grid.NFP

    # we need rescale to [-pi,pi], nodes is divided by NFP, so we multiply by NFP
    eta2d = grid.nodes[:, 2].reshape(grid.num_theta, grid.num_zeta).T * NFP - jnp.pi
    alp2d = grid.nodes[:, 1].reshape(grid.num_theta, grid.num_zeta).T

    index = jnp.where(M == 0, 0, jnp.where(N == 0, 1, 2))

    operands = (eta2d, alp2d, iota[0], NFP, S_list, D_list)
    theta_B, zeta_B = switch(index, OOPS_branches, *operands)

    return theta_B, zeta_B


@register_compute_fun(
    name="theta_B_OOPS",
    label="\\theta_{B}_OOPS",
    units="rad",
    units_long="radians",
    description="Boozer poloidal angle",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    data=["zeta_B_OOPS"],
    coordinates="rtz",
    parameterization="desc.magnetic_fields._core.OmnigenousFieldOOPS",
)
def _omni_map_theta_B_OOPS(params, transforms, profiles, data, **kwargs):
    return data


@register_compute_fun(
    name="|B|_OOPS",
    label="|\\mathbf{B}|_OOPS",
    units="T",
    units_long="Tesla",
    description="Ideal Magnitude of omnigenous magnetic field",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["eta_OOPS"],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldOOPS",
)
def _B_omni_OOPS(params, transforms, profiles, data, **kwargs):
    def fake_B_target(eta):
        return 1 + 0.25 + 0.25 * jnp.cos(eta - jnp.pi)

    eta = transforms["grid"].meshgrid_reshape(data["eta_OOPS"], "rtz")
    B = fake_B_target(eta)
    # Here B is 2d
    B = jnp.moveaxis(B, 0, 1)
    data["|B|_OOPS"] = B.flatten(order="F")
    return data


# @register_compute_fun(
#     name="S_list_LCForm",
#     label="S_{LC,n}",
#     units="~",
#     units_long="None",
#     description="Omnigenity S coefficients, used in Landreman Form",
#     dim=1,
#     params=["S_list"],
#     transforms={},
#     profiles=[],
#     coordinates="rtz",
#     data=[],
#     parameterization="desc.magnetic_fields._core.OmnigenousFieldLCForm",
# )
# def _S_list(params, transforms, profiles, data, **kwargs):
#     """
#     S_list_LCForm is a list of coefficients for the omnigenity S shape.
#     It is used in OOPS to define the omnigenity symmetry angle.
#     """
#     if "S_list_LCForm" not in params:
#         raise ValueError("S_list_LCForm parameter is required for Landreman Form")
#     data["S_list_LCForm"] = jnp.array(params["S_list_LCForm"])
#     return data


# @register_compute_fun(
#     name="D_list_LCForm",
#     label="D_{LC,n}",
#     units="~",
#     units_long="None",
#     description="Omnigenity D coefficients, used in Landreman Form",
#     dim=1,
#     params=["D_list"],
#     transforms={},
#     profiles=[],
#     coordinates="rtz",
#     data=[],
#     parameterization="desc.magnetic_fields._core.OmnigenousFieldLCForm",
# )
# def _D_list(params, transforms, profiles, data, **kwargs):
#     """
#     D_list_LCForm is a list of coefficients for the omnigenity D shape.
#     It is used in Landreman Form to define the omnigenity symmetry angle.
#     """
#     if "D_list_LCForm" not in params:
#         raise ValueError("D_list_LCForm parameter is required for OOPS")
#     data["D_list_LCForm"] = jnp.array(params["D_list_LCForm"])
#     return data


@register_compute_fun(
    name="eta_LCForm",
    label="\\eta_LCForm",
    units="rad",
    units_long="radians",
    description="Intermediate omnigenity coordinate along field lines",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldLCForm",
)
def _eta_OOPS(params, transforms, profiles, data, **kwargs):
    # we need rescale to [-pi,pi], nodes is divided by NFP, so we multiply by NFP
    data["eta_LCForm"] = transforms["grid"].nodes[:, 2] * transforms["grid"].NFP
    return data


def _omnigenity_mapping_LandremanForm(M, N, iota, S_list, D_list, S_func, D_func, grid):
    """
    Landreman-like mapping zeta(eta, theta) with per-element branching at eta = pi,
    written in a JAX/JIT-safe, AD-friendly style.

    Returns
    -------
    theta2d, zeta2d : (num_zeta, num_theta) arrays
        Boozer angles on the evaluation surface (theta set to alpha here).
    """
    TWOPI = jnp.pi * 2.0
    iota = jnp.atleast_1d(iota)
    assert (
        len(iota) == grid.num_rho
    ), f"got ({len(iota)}) iota values for grid with {grid.num_rho} surfaces"
    iota0 = iota[-1]
    NFP = grid.NFP

    # Safe effective-iota per your formula, with guards to avoid NaNs in the inactive branch
    denom1 = jnp.where(iota0 == 0.0, 1.0, iota0)  # for 1/iota
    val1 = 1.0 / denom1
    denom2 = jnp.where(
        (N - iota0 * M) == 0.0, 1.0, (N - iota0 * M)
    )  # for iota/((N-iota*M)*NFP)
    val2 = iota0 / (denom2 * NFP)
    iota_eff = jnp.where(N == 0, val1, val2)

    # Build 2D coordinates (zeta, theta) = (num_zeta, num_theta)
    # Keep eta in [0, 2pi) to match your split at pi
    eta2d = grid.nodes[:, 2].reshape(grid.num_theta, grid.num_zeta).T * NFP
    theta2d = (
        grid.nodes[:, 1].reshape(grid.num_theta, grid.num_zeta).T
    )  # use alpha as "theta" input

    theta_1d = grid.nodes[grid.unique_theta_idx, 1]
    zeta_1d = grid.nodes[grid.unique_zeta_idx, 2]

    eta_1d = zeta_1d * NFP

    theta2d, eta2d = jnp.meshgrid(theta_1d, eta_1d, indexing="ij")

    # Evaluate D on eta and on its "mirror"
    D_eta = D_func(eta2d, D_list)  # shape (nz, nt)
    D_mirror = D_func(TWOPI - eta2d, D_list)

    # Evaluate S on the two branches
    # low-π branch:  zeta = π - S(η, θ + iota_eff * D(η)) - D(η)
    S_low = S_func(eta2d, theta2d + iota_eff * D_eta, S_list)
    zeta_low = jnp.pi - S_low - D_eta

    # up-π branch:   zeta = π + S(2π-η, -θ + iota_eff * D(2π-η)) + D(2π-η)
    S_up = S_func(TWOPI - eta2d, -theta2d + iota_eff * D_mirror, S_list)
    zeta_up = jnp.pi + S_up + D_mirror

    # Elementwise selection at η < π (no dynamic slicing)
    condition = eta2d < jnp.pi
    zeta2d = jnp.where(condition, zeta_low, zeta_up)

    # For this mapping, Boozer theta can be taken as alpha

    # Here in [0,2pi) range
    thetaB2d = theta2d
    zetaB2d = zeta2d

    # Here we trans it to real Boozer angles
    def _branch_N0(_):
        # N == 0: force M=1 convention; set nfp_eff=1 per your comment
        theta_real = zetaB2d
        zeta_real = thetaB2d  # / nfp_eff where nfp_eff = 1
        return zeta_real, theta_real

    def _branch_else(_):
        # N != 0:
        # zeta_real = zeta_calc/(nfp*N) + (M/N)*theta_real
        # theta_real = theta_calc
        N_change = jnp.where(N * M != 0, NFP, N)
        NFP_change = jnp.where(N * M != 0, N, NFP)

        theta_real = thetaB2d
        zeta_real = zetaB2d / (NFP_change * N_change) + (M / N_change) * theta_real
        return zeta_real, theta_real

    zeta2d, theta2d = cond(N == 0, _branch_N0, _branch_else, operand=None)

    return theta2d, zeta2d


@register_compute_fun(
    name="zeta_B_LCForm",
    label="\\zeta_{B}_LCForm",
    units="rad",
    units_long="radians",
    description="Boozer toroidal angle using Landreman-like mapping",
    dim=1,
    params=["S_list", "D_list"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldLCForm",
    helicity="tuple: Type of quasisymmetry, (M,N). Default (1,0)",
    iota="float: Value of rotational transform on the Omnigenous surface. Default 1.0",
    S_func="function: Function to compute S(eta,theta) given S_list",
    D_func="function: Function to compute D(eta) given D_list",
)
def _omni_map_zeta_B_LCForm(params, transforms, profiles, data, **kwargs):
    M = kwargs.get("helicity", (1, 0))[0]
    N = kwargs.get("helicity", (1, 0))[1]
    iota = kwargs.get("iota", jnp.ones(transforms["grid"].num_rho))
    S_list = params["S_list"]
    D_list = params["D_list"]
    S_func = kwargs.get("S_func", None)
    D_func = kwargs.get("D_func", None)
    if S_func is None or D_func is None:
        raise ValueError(
            "S_func and D_func must be provided in params for LandremanLCForm"
        )

    theta_B, zeta_B = _omnigenity_mapping_LandremanForm(
        M, N, iota, S_list, D_list, S_func, D_func, transforms["grid"]
    )

    data["theta_B_LCForm"] = theta_B
    data["zeta_B_LCForm"] = zeta_B
    return data


@register_compute_fun(
    name="theta_B_LCForm",
    label="\\theta_{B}_LCForm",
    units="rad",
    units_long="radians",
    description="Boozer poloidal angle using Landreman-like mapping",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    data=["zeta_B_LCForm"],
    coordinates="rtz",
    parameterization="desc.magnetic_fields._core.OmnigenousFieldLCForm",
)
def _omni_map_theta_B_LCForm(params, transforms, profiles, data, **kwargs):
    return data


@register_compute_fun(
    name="|B|_LCForm",
    label="|\\mathbf{B}|_LCForm",
    units="T",
    units_long="Tesla",
    description="Ideal Magnitude of omnigenous magnetic field using Landreman-like mapping",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["eta_LCForm"],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldLCForm",
)
def _B_omni_LCForm(params, transforms, profiles, data, **kwargs):
    def fake_B_target(eta):
        return 1 + 0.25 + 0.25 * jnp.cos(eta)

    eta = transforms["grid"].meshgrid_reshape(data["eta_LCForm"], "rtz")
    B = fake_B_target(eta)
    # Here B is 2d
    B = jnp.moveaxis(B, 0, 1)
    data["|B|_LCForm"] = B.flatten(order="F")
    return data


def _B_omni_nonsymmetric(
    B_eta_alpha, M_harmonics, N_harmonics, field_grid, is_imag=True
):
    # Normalized by B00
    n_theta, n_zeta = B_eta_alpha.shape

    B_mn = jnp.fft.fft2(B_eta_alpha / (field_grid.num_theta * field_grid.num_zeta))

    b00 = B_mn[0, 0]
    bnorm = jnp.where(b00 == 0, 1.0, b00)
    B_mn_normalized = B_mn / bnorm

    B_mn_shifted = jnp.fft.fftshift(B_mn_normalized)

    center_theta = n_theta // 2
    center_zeta = n_zeta // 2

    # m ∈ [1, M_harmonics], n ∈ [-N_harmonics, N_harmonics]
    r0 = center_theta + 1
    r1 = r0 + M_harmonics
    c0 = center_zeta - N_harmonics
    c1 = center_zeta + N_harmonics + 1

    nonsymmetric_block_complex = B_mn_shifted[r0:r1, c0:c1]

    error_coeffs_complex = nonsymmetric_block_complex.reshape(-1)

    if is_imag:
        # error_coeffs = jnp.concatenate(
        #     [error_coeffs_complex.real, error_coeffs_complex.imag], axis=0
        # )

        # imag_m0 = B_mn_shifted[center_theta, c0:c1].imag.reshape(-1)  # shape: (2N+1,)
        # error_coeffs = jnp.concatenate(
        #     [error_coeffs_complex.real, error_coeffs_complex.imag, imag_m0], axis=0
        # )

        real_part = jnp.ravel(nonsymmetric_block_complex.real)
        imag_part = jnp.ravel(nonsymmetric_block_complex.imag)

        # 2) 追加虚部 m=0, n∈[-N..N]
        imag_m0 = jnp.ravel(
            B_mn_shifted[center_theta, c0:c1].imag.reshape(-1)
        )  # (2N+1,)

        # 3) 追加不等式约束：B(m=0, n=+1) > 0  —— 以 hinge 罚实现
        #    注意：这里直接从全频谱中取 n=+1（fftshift 后是 center_zeta+1）
        b01_real = B_mn_shifted[center_theta, center_zeta + 1].real
        pen_b01 = 1 * jnp.maximum(0.0, -b01_real + 0.1)  # 满足>0时为0，<0时为正

        error_coeffs = jnp.concatenate(
            [real_part, imag_part, imag_m0, pen_b01[None]], axis=0
        )
    else:
        error_coeffs = error_coeffs_complex.real
    return error_coeffs


def _binom(n, k):
    k = jnp.asarray(k)
    valid = (k >= 0) & (k <= n)
    val = jnp.exp(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0))
    return jnp.where(valid, val, 0.0)


def _raised_cosine_shape_reg(
    B_eta_alpha,
    M_harmonics,
    N_harmonics,
    field_grid,
):
    """
    惩罚项 R：对 m=0 行（极向对称分量）的 |B_{0,n}| 做“升余弦幂”形状约束（只控形状，不控幅），
    并压制 |n|>q 的尾部。n=0 列默认不参与形状拟合。
    """

    q = 32
    w_tail = 1.0
    w_shape = 1.0
    eps = 1e-12

    # ===== 与前一致：取谱并 B00 归一化 =====
    n_theta, n_zeta = B_eta_alpha.shape
    B_mn = jnp.fft.fft2(B_eta_alpha / (field_grid.num_theta * field_grid.num_zeta))
    b00 = B_mn[0, 0]
    bnorm = jnp.where(b00 == 0, 1.0, b00)
    B_mn_norm = B_mn / bnorm
    B_shift = jnp.fft.fftshift(B_mn_norm)

    center_theta = n_theta // 2
    center_zeta = n_zeta // 2

    # ===== 只取 m=0 行的 n∈[-N..N]（这就是 x_n 分量）=====
    c0 = center_zeta - N_harmonics
    c1 = center_zeta + N_harmonics + 1
    row_m0 = B_shift[center_theta, c0:c1]  # shape: (2N+1,)
    mag = jnp.abs(row_m0)  # 幅值谱 |B_{0,n}|

    # ===== 目标模板：T_n ∝ C(2q, q-|n|)，对 |n|<=q；其它为 0 =====
    n_range = jnp.arange(-N_harmonics, N_harmonics + 1)  # [-N..N]
    n_abs = jnp.abs(n_range)

    mask_n0 = n_abs == 0  # n=0 不参与形状拟合
    mask_core = (n_abs >= 1) & (n_abs <= q)  # |n|∈[1..q]
    mask_tail = n_abs > q  # |n|>q

    T = _binom(2 * q, (q - n_abs).astype(float))  # C(2q, q-|n|)
    T = jnp.where(mask_core, T, 0.0)  # 只保留核心区
    T_norm = T / (jnp.linalg.norm(T) + eps)  # 归一化，纯形状比较

    # ===== m=0 行的最优缩放 A（只控形状，不控幅）=====
    T_fit = jnp.where(mask_core & (~mask_n0), T_norm, 0.0)
    A = jnp.sum(mag * T_fit)  # 因 ‖T_norm‖=1，A=mag·T
    resid_core = (mag - A * T_norm) * (mask_core & (~mask_n0))
    R_shape = w_shape * jnp.sum(resid_core**2)

    # ===== 尾部惩罚：希望 |n|>q 变小 =====
    R_tail = w_tail * jnp.sum((mag * mask_tail) ** 2)

    return R_shape + R_tail
