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

from desc.backend import jnp, sign, vmap
from desc.integrals import surface_averages

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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    grid_requirement={"is_meshgrid": True},
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
    import pdb

    pdb.set_trace()
    return data


@register_compute_fun(
    name="|B|_pwO",
    label="|\\mathbf{B}|",
    units="T",
    units_long="Tesla",
    description="Magnitude of omnigenous magnetic field",
    dim=1,
    params=["B_min", "B_max", "zeta_C", "theta_C", "t_1", "t_2", "w_2", "iota0"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],  # Potential error, we want eq |B|
    parameterization="desc.magnetic_fields._core.PiecewiseOmnigenousField",
)
def _B_piecewise_omni(params, transforms, profiles, data, **kwargs):
    theta_B = transforms["grid"].nodes[:, 1]
    zeta_B = transforms["grid"].nodes[:, 2]
    # NFP can't be a parameter. Must come from equilibrium
    NFP = transforms["grid"].NFP

    zeta_C = params["zeta_C"]
    theta_C = params["theta_C"]
    t_1 = params["t_1"]
    t_2 = params["t_2"]
    w_2 = params["w_2"]
    iota0 = params["iota0"]
    w_1 = jnp.pi / NFP * (1 - t_1 * t_2) / (1 + t_2 / iota0)
    B_min = params["B_min"]
    B_max = params["B_max"]
    p = int(10)
    exponent = -1 * (
        ((zeta_B - zeta_C + t_1 * (theta_B - theta_C)) / w_1) ** (2 * p)
        + ((theta_B - theta_C + t_2 * (zeta_B - zeta_C)) / w_2) ** (2 * p)
    )

    B_pwO = B_min + (B_max - B_min) * jnp.exp(exponent)

    # Flattened array. Reshaping may cause jit-related issues
    data["|B|_pwO"] = B_pwO

    return data


@register_compute_fun(
    name="Q_pwO",
    label="|\\mathbf{B}|",
    units="~",
    units_long="None",
    description="Self-overlap of the target field",
    dim=1,
    params=["t_1", "t_2", "w_2", "iota0"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],  # Potential error, we want eq |B|
    parameterization="desc.magnetic_fields._core.PiecewiseOmnigenousField",
)
def _Q_piecewise_omni(params, transforms, profiles, data, **kwargs):
    # NFP can't be a parameter. Must come from equilibrium
    NFP = transforms["grid"].NFP

    t_1 = params["t_1"]
    t_2 = params["t_2"]
    w_2 = params["w_2"]
    iota0 = params["iota0"]
    w_1 = jnp.pi / NFP * (1 - t_1 * t_2) / (1 + t_2 / iota0)

    zeta_pp = (w_1 - t_1 * w_2) / (1 - t_1 * t_2)
    zeta_pm = (w_1 + t_1 * w_2) / (1 - t_1 * t_2)
    theta_pp = (w_2 - t_2 * w_1) / (1 - t_1 * t_2)
    theta_pm = (-w_2 - t_2 * w_1) / (1 - t_1 * t_2)

    Q = (
        jnp.max(
            jnp.stack(
                [
                    NFP * zeta_pp - jnp.pi,
                    NFP * zeta_pm - jnp.pi,
                    theta_pp - jnp.pi,
                    -theta_pm - jnp.pi,
                ],
                axis=0,
            )
        )
        / jnp.pi
    )

    data["Q_pwO"] = Q

    return data


@register_compute_fun(
    name="Delta_BS",
    label="|\\mathbf{B}|",
    units="~",
    units_long="None",
    description="Delta proxi for zero pwO Bootstrap current",
    dim=1,
    params=["B_max", "t_1", "t_2", "w_2", "iota0"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["|B|_pwO"],  # Potential error, we want eq |B|
    parameterization="desc.magnetic_fields._core.PiecewiseOmnigenousField",
)
def _Delta_bs_piecewiseomni(params, transforms, profiles, data, **kwargs):
    # NFP can't be a parameter. Must come from equilibrium

    NFP = transforms["grid"].NFP

    t_1 = params["t_1"]
    t_2 = params["t_2"]
    w_2 = params["w_2"]
    iota0 = params["iota0"]
    B_max = params["B_max"]

    w_1 = ((jnp.pi / NFP) * (1 - t_1 * t_2)) / (1 + t_2 / iota0)

    A1 = jnp.abs((4 * w_2 * (w_1 - jnp.pi / NFP)) / (1 - t_1 * t_2))

    A2 = jnp.abs((4 * jnp.pi**2 / NFP) - (4 * w_2 * jnp.pi) / (NFP * (1 - t_1 * t_2)))

    B_pwO_squared_averaged = surface_averages(
        transforms["grid"],
        data["|B|_pwO"] ** 2,
    )[0]

    Delta = (B_pwO_squared_averaged / (4 * jnp.pi**2 * B_max**2)) * (
        (A1 / (iota0 + 1 / t_1)) + (A2 / (iota0 + t_2))
    )

    data["Delta_BS"] = Delta

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
