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

import jax
import numpy as np
from interpax import interp1d

from desc.backend import jnp, sign, vmap
from desc.batching import vmap_chunked

from ..utils import cross, dot, safediv
from .data_index import register_compute_fun

SOFTPLUS_SHARPNESS = 100.0
_trapz = getattr(jnp, "trapezoid", getattr(jnp, "trapz", None))


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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to ``grid.num_rho`` e.g. compute all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
)
def _B_theta_mn(params, transforms, profiles, data, **kwargs):
    B_theta = transforms["grid"].meshgrid_reshape(data["B_theta"], "rtz")

    def fitfun(x):
        return transforms["B"].fit(x.flatten(order="F"))

    B_theta_mn = vmap_chunked(fitfun, chunk_size=kwargs.get("surf_batch_size"))(B_theta)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to ``grid.num_rho`` e.g. compute all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
)
def _B_phi_mn(params, transforms, profiles, data, **kwargs):
    B_phi = transforms["grid"].meshgrid_reshape(data["B_phi|r,t"], "rtz")

    def fitfun(x):
        return transforms["B"].fit(x.flatten(order="F"))

    B_zeta_mn = vmap_chunked(fitfun, chunk_size=kwargs.get("surf_batch_size"))(B_phi)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to ``grid.num_rho`` e.g. compute all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
)
def _w(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"]
    w_mn = data["w_Boozer_mn"].reshape((grid.num_rho, -1))
    w = vmap_chunked(
        transforms["w"].transform, chunk_size=kwargs.get("surf_batch_size")
    )(
        w_mn
    )  # shape(rho, theta*zeta)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to ``grid.num_rho`` e.g. compute all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
)
def _w_t(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"]
    w_mn = data["w_Boozer_mn"].reshape((grid.num_rho, -1))
    # need to close over dt which can't be vmapped
    fun = lambda x: transforms["w"].transform(x, dt=1)
    w_t = vmap_chunked(fun, chunk_size=kwargs.get("surf_batch_size"))(
        w_mn
    )  # shape(rho, theta*zeta)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to ``grid.num_rho`` e.g. compute all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
)
def _w_z(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"]
    w_mn = data["w_Boozer_mn"].reshape((grid.num_rho, -1))
    # need to close over dz which can't be vmapped
    fun = lambda x: transforms["w"].transform(x, dz=1)
    w_z = vmap_chunked(fun, chunk_size=kwargs.get("surf_batch_size"))(
        w_mn
    )  # shape(rho, theta*zeta)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to computing all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
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
    nu_B_mn = vmap_chunked(
        fun, in_axes=(0, 0, 0, 0, 0), chunk_size=kwargs.get("surf_batch_size")
    )(rho, theta_B, zeta_B, sqrtg_B_desc, nu)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to computing all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
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
    sqrtg_B_mn = vmap_chunked(
        fun, in_axes=(0, 0, 0, 0, 0), chunk_size=kwargs.get("surf_batch_size")
    )(rho, theta_B, zeta_B, sqrtg_B_desc, sqrtg_B)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to computing all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
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
    B_mn = vmap_chunked(
        fun, in_axes=(0, 0, 0, 0, 0), chunk_size=kwargs.get("surf_batch_size")
    )(rho, theta_B, zeta_B, sqrtg_B_desc, B)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to computing all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
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
    R_mn = vmap_chunked(
        fun, in_axes=(0, 0, 0, 0, 0), chunk_size=kwargs.get("surf_batch_size")
    )(rho, theta_B, zeta_B, sqrtg_B_desc, R)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to computing all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
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
    Z_mn = vmap_chunked(
        fun, in_axes=(0, 0, 0, 0, 0), chunk_size=kwargs.get("surf_batch_size")
    )(rho, theta_B, zeta_B, sqrtg_B_desc, Z)
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
    surf_batch_size="int: Number of flux surfaces to compute simultaneously. Defaults"
    " to computing all flux surfaces simultaneously. Decrease "
    "to reduce memory required for computation.",
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
    B = vmap_chunked(_interp, in_axes=(0, 0), chunk_size=kwargs.get("surf_batch_size"))(
        jnp.abs(eta), B_input.T
    )  # shape (nr, nt*nz)
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


# ---------------------------------------------------------------------------
# Direct Second Adiabatic Invariant (J*) and Soft-Connectivity Kernels
# References: Chen et al., arXiv:2608.02418 (2026)
# ---------------------------------------------------------------------------


def _softplus_relu(x, beta=SOFTPLUS_SHARPNESS):
    """Sharp, smooth approximation of ``max(x, 0)``."""
    return jnp.logaddexp(beta * x, 0.0) / beta


def _softplus_relu_sigmoid(x, beta=SOFTPLUS_SHARPNESS):
    """Derivative of ``_softplus_relu``: ``sigmoid(beta * x)``."""
    return 1.0 / (1.0 + jnp.exp(-beta * x))


def _reshape_surface_coefficients(grid, values):
    """Reshape flattened per-surface Boozer coefficients."""
    return jnp.asarray(values).reshape((grid.num_rho, -1))


def _boozer_B_star_from_t(B_min, B_max, t):
    """Map normalized pitch samples ``t`` to surface-wise ``B_star`` values."""
    B_min = jnp.atleast_1d(jnp.asarray(B_min))
    B_max = jnp.atleast_1d(jnp.asarray(B_max))
    t = jnp.atleast_1d(jnp.asarray(t))
    return (1.0 - t[None, :]) * B_min[:, None] + t[None, :] * B_max[:, None]


def _smoothmax_logsumexp(x, axis, tau):
    """Differentiable upper envelope approximating max(x, axis=axis)."""
    tau = jnp.asarray(tau, dtype=x.dtype)
    tau = jnp.maximum(tau, jnp.finfo(x.dtype).eps)
    x_scaled = x / tau
    x_max = jnp.max(x_scaled, axis=axis, keepdims=True)
    lse = x_max + jnp.log(jnp.sum(jnp.exp(x_scaled - x_max), axis=axis, keepdims=True))
    return tau * lse


def _boozer_second_adiabatic_surface_alpha_deriv(
    basis,
    rho,
    coeff_B,
    iota,
    alpha,
    B_star,
    zeta_min,
    zeta_max,
    nzeta,
    softplus_sharpness,
):
    """Analytical dJ/dalpha on a single flux surface via chain-rule integration.

    Computes the derivative integrand directly:
        dJ/dalpha = integral [ df/dB * dB/dtheta_B ] dzeta
    where f = sqrt(cutoff) / B and theta_B = alpha + iota*(zeta - zeta_min).
    """
    zeta = jnp.linspace(zeta_min, zeta_max, nzeta)
    theta = alpha[:, None] + iota * (zeta[None, :] - zeta_min)
    rho2d = jnp.broadcast_to(rho, theta.shape)
    zeta2d = jnp.broadcast_to(zeta[None, :], theta.shape)
    nodes = jnp.stack((rho2d, theta, zeta2d), axis=-1).reshape((-1, 3))

    mat = basis.evaluate(nodes)
    mat_dt = basis.evaluate(nodes, derivatives=np.array([0, 1, 0]))

    B = (mat @ coeff_B).reshape((alpha.size, nzeta))
    dB_dt = (mat_dt @ coeff_B).reshape((alpha.size, nzeta))

    B_star = jnp.atleast_1d(B_star)
    arg = 1.0 - B[None] / B_star[:, None, None]
    cutoff = _softplus_relu(arg, beta=softplus_sharpness)
    sig = _softplus_relu_sigmoid(arg, beta=softplus_sharpness)
    sqrt_c = jnp.sqrt(jnp.maximum(cutoff, 1e-30))

    safe_sqrt_c = jnp.where(cutoff > 0, sqrt_c, 1e-30)
    # df/dB for f = sqrt(cutoff) / B:
    #   d/dB[sqrt(c)/B] = (dc/dB) * (1/(2*B*sqrt(c))) - sqrt(c)/B**2,
    #   with dc/dB = sigmoid(beta*arg) * (-1/B_star).
    df_dB = jnp.where(
        cutoff > 0,
        -sig / (2.0 * B_star[:, None, None] * safe_sqrt_c * B[None])
        - safe_sqrt_c / (B[None] ** 2),
        0.0,
    )

    integrand = df_dB * dB_dt[None]
    return _trapz(integrand, zeta, axis=-1).transpose(1, 0)


def boozer_second_adiabatic_invariant_alpha_derivative_analytical(
    basis,
    rho,
    iota,
    coeff_B,
    alpha,
    B_star,
    *,
    nzeta=1000,
    zeta_min=0.0,
    zeta_max=None,
    nfp=1,
    softplus_sharpness=SOFTPLUS_SHARPNESS,
):
    """Analytical dJ/dalpha via chain rule through the Boozer integral."""
    alpha = jnp.asarray(alpha)
    B_star = jnp.asarray(B_star)
    if B_star.ndim == 0:
        B_star = B_star[None, None]
    elif B_star.ndim == 1:
        B_star = B_star[None, :]
    B_star = jnp.broadcast_to(B_star, (rho.size, B_star.shape[-1]))
    zeta_max = 2 * jnp.pi / nfp if zeta_max is None else zeta_max

    return vmap(
        lambda r, it, cB, pi, a: _boozer_second_adiabatic_surface_alpha_deriv(
            basis,
            r,
            cB,
            it,
            a,
            pi,
            zeta_min,
            zeta_max,
            nzeta,
            softplus_sharpness,
        )
    )(rho, iota, coeff_B, B_star, alpha)


def boozer_second_adiabatic_invariant_alpha_derivative_from_data(
    grid,
    basis,
    data,
    t,
    *,
    num_alpha=48,
    nzeta=1000,
    zeta_min=0.0,
    zeta_max=None,
    softplus_sharpness=SOFTPLUS_SHARPNESS,
    soft_extrema_tau=0.1,
):
    """Convenience wrapper for ``dJ/dalpha`` using per-surface Boozer data."""
    rho = jnp.asarray(grid.compress(grid.nodes[:, 0]))
    iota = jnp.asarray(grid.compress(data["iota"]))
    coeff_B = _reshape_surface_coefficients(grid, data["|B|_mn_B"])

    nfp = grid.NFP
    alpha0 = jnp.pi - iota * (jnp.pi / nfp)
    alpha = jnp.linspace(-jnp.pi, 0.0, num_alpha, endpoint=False) + alpha0[:, None]

    A = basis.evaluate(grid.nodes[:: grid.num_rho])
    B_grid = coeff_B @ A.T
    num_eval = B_grid.shape[1]
    log_n = jnp.log(jnp.maximum(jnp.asarray(num_eval, dtype=B_grid.dtype), 2.0))
    B_range = jnp.max(B_grid, axis=1) - jnp.min(B_grid, axis=1)
    B_range = jnp.maximum(B_range, 1e-30)
    tau_eff = (soft_extrema_tau * B_range / log_n)[:, None]
    B_max = _smoothmax_logsumexp(B_grid, axis=1, tau=tau_eff).squeeze(-1)
    B_min = -_smoothmax_logsumexp(-B_grid, axis=1, tau=tau_eff).squeeze(-1)
    B_star = _boozer_B_star_from_t(B_min, B_max, t)[0]
    return boozer_second_adiabatic_invariant_alpha_derivative_analytical(
        basis,
        rho,
        iota,
        coeff_B,
        alpha,
        B_star,
        nzeta=nzeta,
        zeta_min=zeta_min,
        zeta_max=zeta_max,
        nfp=nfp,
        softplus_sharpness=softplus_sharpness,
    )


def _boozer_soft_connectivity_surface(
    basis,
    rho,
    coeff_B,
    iota,
    alpha,
    nfp,
    t,
    reduced_alpha_knots,
    zeta_min_knots,
    zeta_max_knots,
    sigmoid_sharpness,
    spline_symmetry,
):
    """Compute the structured soft-connectivity penalty on a single flux surface."""
    zeta_span = 2 * jnp.pi / nfp
    alpha_next = alpha + iota * zeta_span

    reduced_alpha_knots = jnp.asarray(reduced_alpha_knots)
    zeta_min_knots = jnp.clip(jnp.asarray(zeta_min_knots), 0.0, zeta_span)

    if spline_symmetry:
        alpha_min_knots = jnp.concatenate(
            [
                reduced_alpha_knots,
                jnp.mod(
                    2 * jnp.pi - reduced_alpha_knots - iota * zeta_span,
                    2 * jnp.pi,
                ),
            ]
        )
        zeta_min_knots = jnp.concatenate([zeta_min_knots, zeta_span - zeta_min_knots])
    else:
        alpha_min_knots = reduced_alpha_knots
    min_order = jnp.argsort(alpha_min_knots)
    alpha_min_knots = alpha_min_knots[min_order]
    zeta_min_knots = zeta_min_knots[min_order]

    zeta_min_alpha = interp1d(
        alpha, alpha_min_knots, zeta_min_knots, method="cubic", period=2 * jnp.pi
    )
    zeta_min_next = interp1d(
        alpha_next,
        alpha_min_knots,
        zeta_min_knots,
        method="cubic",
        period=2 * jnp.pi,
    )
    zeta_min_next = zeta_min_next + zeta_span

    if zeta_max_knots is not None:
        zeta_max_knots = (
            jnp.mod(jnp.asarray(zeta_max_knots) + 0.5 * zeta_span, zeta_span)
            - 0.5 * zeta_span
        )
        if spline_symmetry:
            alpha_max_knots = jnp.concatenate(
                [
                    reduced_alpha_knots,
                    jnp.mod(2 * jnp.pi - reduced_alpha_knots, 2 * jnp.pi),
                ]
            )
            zeta_max_knots = jnp.concatenate([zeta_max_knots, -zeta_max_knots])
        else:
            alpha_max_knots = reduced_alpha_knots
        max_order = jnp.argsort(alpha_max_knots)
        alpha_max_knots = alpha_max_knots[max_order]
        zeta_max_knots = zeta_max_knots[max_order]

        zeta_max_alpha = interp1d(
            alpha, alpha_max_knots, zeta_max_knots, method="cubic", period=2 * jnp.pi
        )
        zeta_max_next = interp1d(
            alpha_next,
            alpha_max_knots,
            zeta_max_knots,
            method="cubic",
            period=2 * jnp.pi,
        )
        zeta_max_next = zeta_max_next + zeta_span
    else:
        zeta_max_alpha = jnp.zeros(alpha.shape)
        zeta_max_next = jnp.full(alpha.shape, zeta_span)

    zeta = (1.0 - t[None, :]) * zeta_max_alpha[:, None] + (
        t[None, :] * zeta_max_next[:, None]
    )
    theta = alpha[:, None] + iota * zeta
    rho2d = jnp.broadcast_to(rho, theta.shape)
    nodes = jnp.stack((rho2d, theta, zeta), axis=-1).reshape((-1, 3))
    nt = t.size

    dB_dtheta = (
        basis.evaluate(nodes, derivatives=np.array([0, 1, 0])) @ coeff_B
    ).reshape((alpha.size, nt))
    dB_dzeta = (
        basis.evaluate(nodes, derivatives=np.array([0, 0, 1])) @ coeff_B
    ).reshape((alpha.size, nt))
    dB_dz_line = iota * dB_dtheta + dB_dzeta

    use_current_min = (zeta_min_alpha > zeta_max_alpha) & (
        zeta_min_alpha < zeta_max_next
    )
    zeta_min_shifted = jnp.where(use_current_min, zeta_min_alpha, zeta_min_next)
    delta = zeta - zeta_min_shifted[:, None]
    sig = jax.nn.sigmoid(sigmoid_sharpness * delta)
    penalty_left = _softplus_relu(dB_dz_line)
    penalty_right = _softplus_relu(-dB_dz_line)
    penalty = sig * penalty_right + (1.0 - sig) * penalty_left
    return penalty


def boozer_soft_connectivity_penalty(
    basis,
    rho,
    iota,
    coeff_B,
    alpha,
    nfp,
    t,
    *,
    reduced_alpha_knots,
    zeta_min_knots,
    zeta_max_knots=None,
    sigmoid_sharpness=50.0,
    spline_symmetry=True,
):
    """Compute the soft-connectivity penalty over multiple flux surfaces."""
    nfp_arr = jnp.broadcast_to(jnp.asarray(nfp), rho.shape)
    return vmap(
        lambda r, it, cB, a, nf: _boozer_soft_connectivity_surface(
            basis,
            r,
            cB,
            it,
            a,
            nf,
            t,
            reduced_alpha_knots,
            zeta_min_knots,
            zeta_max_knots,
            sigmoid_sharpness,
            spline_symmetry,
        )
    )(rho, iota, coeff_B, alpha, nfp_arr)


def boozer_soft_connectivity_penalty_from_data(
    grid,
    basis,
    data,
    t,
    *,
    reduced_alpha_knots,
    zeta_min_knots,
    zeta_max_knots=None,
    num_alpha=50,
    sigmoid_sharpness=50.0,
    spline_symmetry=True,
):
    """Convenience wrapper for the soft-connectivity penalty using per-surface Boozer data."""
    rho = jnp.asarray(grid.compress(grid.nodes[:, 0]))
    iota = jnp.asarray(grid.compress(data["iota"]))
    coeff_B = _reshape_surface_coefficients(grid, data["|B|_mn_B"])

    nfp = grid.NFP
    # alpha0 is the fixed point of the stellarator-symmetry mirror map
    # alpha -> 2π − alpha − iota*span. With symmetric splines the penalty
    # on the mirrored half is identical, so sampling one fundamental
    # domain (length π) suffices. With symmetry=False the knots span the
    # full [0, 2π) and every knot must be sampled: use the full period.
    alpha0 = jnp.pi - iota * (jnp.pi / nfp)
    alpha_span = jnp.pi if spline_symmetry else 2 * jnp.pi
    alpha = jnp.linspace(-alpha_span, 0.0, num_alpha, endpoint=False) + alpha0[:, None]

    return boozer_soft_connectivity_penalty(
        basis,
        rho,
        iota,
        coeff_B,
        alpha,
        nfp,
        t,
        reduced_alpha_knots=reduced_alpha_knots,
        zeta_min_knots=zeta_min_knots,
        zeta_max_knots=zeta_max_knots,
        sigmoid_sharpness=sigmoid_sharpness,
        spline_symmetry=spline_symmetry,
    )
