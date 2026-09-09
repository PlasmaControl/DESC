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

from desc.backend import jnp, scan, sign, vmap
from desc.batching import vmap_chunked
from desc.integrals._interp_utils import interp1d_vec

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


# SQuID-like construction of a scalar QI reference field. The shared-minimum
# single-well operation is independent of the equal-bounce-distance enhancement.
def _cumulative_minimum(x):
    """Cumulative minimum along the final axis, using the DESC backend."""

    def body(previous, current):
        result = jnp.minimum(previous, current)
        return result, result

    _, tail = scan(body, x[..., 0], jnp.moveaxis(x[..., 1:], -1, 0))
    return jnp.concatenate((x[..., :1], jnp.moveaxis(tail, 0, -1)), axis=-1)


def _construct_single_well(
    B,
    *,
    B_min=None,
    B_max=None,
    span_rtol=1e-12,
    field_tol=1e-12,
):
    """Squash physical samples and align minima, preserving independent endpoints.

    Parameters
    ----------
    B : ndarray, shape (nr, na, nz)
        Physical field strength in Tesla along sampled Boozer field lines.
    B_min, B_max : ndarray, shape (nr,), optional
        Reference extrema in Tesla. Defaults to extrema of the input samples.
    span_rtol, field_tol : float
        Relative surface-span threshold and dimensionless field tolerance.

    Returns
    -------
    well_data : dict of ndarray
        Physical ``B_single`` with common ``B_min``, independent ``endpoint_B``,
        ``minimum_indices`` (first and last tied minimum), normalization, and
        separate validity masks. The base layer retains the minimum interval.

    """
    B = jnp.asarray(B)
    valid_input = jnp.all(jnp.isfinite(B) & (B > 0), axis=(1, 2))
    safe_B = jnp.where(jnp.isfinite(B), B, 1.0)
    B_min = jnp.min(safe_B, axis=(1, 2)) if B_min is None else jnp.asarray(B_min)
    B_max = jnp.max(safe_B, axis=(1, 2)) if B_max is None else jnp.asarray(B_max)
    span = B_max - B_min
    valid_span = jnp.isfinite(span) & (
        span > span_rtol * jnp.max(jnp.abs(safe_B), axis=(1, 2))
    )
    scale = jnp.where(valid_span, span, 1.0)
    normalized = (safe_B - B_min[:, None, None]) / scale[:, None, None]
    minimum_index = jnp.argmin(safe_B, axis=-1)
    minimum = jnp.min(safe_B, axis=-1)
    near_minimum = (safe_B - minimum[..., None]) / scale[:, None, None] <= field_tol
    indices = jnp.arange(B.shape[-1])
    first = jnp.min(jnp.where(near_minimum, indices, B.shape[-1]), axis=-1)
    last = jnp.max(jnp.where(near_minimum, indices, -1), axis=-1)
    minimum_indices = jnp.stack((first, last), axis=-1)
    internal = (first > 0) & (last < B.shape[-1] - 1)
    minimum_interval = (indices >= first[..., None]) & (indices <= last[..., None])
    left = _cumulative_minimum(safe_B)
    right = _cumulative_minimum(safe_B[..., ::-1])[..., ::-1]
    squashed = jnp.where(indices < minimum_index[..., None], left, right)
    # Treat tolerance-level differences symmetrically. Squash makes the whole
    # interval between equal-depth minima flat; retain it in the base layer.
    squashed = jnp.where(minimum_interval, minimum[..., None], squashed)
    single = squashed - minimum[..., None] + B_min[:, None, None]
    # Keep each endpoint: a common maximum belongs to the QI enhancement only.
    endpoint = jnp.stack((single[..., 0], single[..., -1]), axis=-1)
    valid = internal & valid_input[:, None] & valid_span[:, None]
    return {
        "B_single": single,
        "B_min": B_min,
        "B_max": B_max,
        "B_normalized": normalized,
        "minimum_index": minimum_index,
        "minimum_indices": minimum_indices,
        "endpoint_B": endpoint,
        "valid_single_well": valid,
        "valid_input": valid_input,
        "valid_span": valid_span,
        "valid_minimum_interior": internal,
    }


def _project_centers(raw_centers, distances, zeta, knot_margin):
    """Project paired centers while preserving widths and the chosen bottom anchor.

    ``raw_centers`` has shape (nr, na, nB), ``distances`` shape (nr, nB).
    Return centers of the same shape and a feasibility mask of shape (nr, na).
    """
    period = zeta[-1] - zeta[0]
    middle = (zeta[-1] + zeta[0]) / 2
    h = jnp.diff(distances, axis=-1) / 2 - knot_margin * period
    remaining = jnp.concatenate(
        (
            jnp.cumsum(h[..., ::-1], axis=-1)[..., ::-1][..., 1:],
            jnp.zeros_like(h[..., :1]),
        ),
        axis=-1,
    )
    valid = jnp.all(h >= 0, axis=-1)[:, None] & (
        jnp.abs(raw_centers[..., 0] - middle) <= jnp.sum(h, axis=-1)[:, None]
    )

    def body(previous, current):
        raw, reach, future = current
        lower = jnp.maximum(previous - reach[:, None], middle - future[:, None])
        upper = jnp.minimum(previous + reach[:, None], middle + future[:, None])
        center = jnp.clip(raw, lower, upper)
        return center, center

    _, centers = scan(
        body,
        raw_centers[..., 0],
        (
            jnp.moveaxis(raw_centers[..., 1:], -1, 0),
            h.T,
            remaining.T,
        ),
    )
    centers = jnp.concatenate(
        (raw_centers[..., :1], jnp.moveaxis(centers, 0, -1)), axis=-1
    )
    # The final feasible interval is the fixed midpoint, including its derivative.
    centers = centers.at[..., -1].set(middle)
    return centers, valid


def _single_well_bounce_points(levels, knots, values):
    """Invert both monotone branches, selecting the edges of the strict sublevel set.

    Parameters
    ----------
    levels : ndarray, shape (..., num_level)
        Field levels. Leading dimensions broadcast with those of ``values``.
    knots : ndarray, shape (num_knot,)
        Finite, strictly increasing coordinates; at least three samples.
    values : ndarray, shape (..., num_knot)
        Continuous piecewise-linear single wells. Each well decreases to an
        interior minimum interval and then increases; flat segments are allowed.

    Returns
    -------
    left, right, valid : ndarray
        Shape ``broadcast_shape + (num_level,)``, without a well axis. Valid levels
        exceed the minimum and do not exceed either endpoint value. At a plateau,
        choose its edge adjacent to the strict sublevel set. Invalid pairs are
        both zero, with ``valid=False``; there is no sentinel convention.

    Notes
    -----
    DESC's general ``integrals._bounce_utils._bounce_points`` discovers and pairs
    wells from local-power cubic spline coefficients. QI squash/stretch already
    provides monotone piecewise-linear branches, including plateaus whose edges
    must bound the strict sublevel set. That convention is not part of the
    shared cubic-spline contract, so it is handled here without extending the
    bounce kernel used by other objectives.

    This is a local inverse of known single wells, not a general root finder.
    The field comparisons select a nonflat interval on each branch; DESC's linear
    interpolator then inverts its two endpoints, including their derivatives.
    Selecting the segment before inversion avoids clipping roots at shared knots,
    which would alter their derivatives. No separate custom JVP is needed.
    Exact shared knots with equal adjacent slopes have their classical derivative.
    Branch switches with unequal slopes and plateau levels need not be smooth.

    The equal-distance construction calls this on its stretched intermediate
    field. Queries on the final constructed field interpolate its stored centers
    and widths instead; this helper does not find wells in the original field.
    """
    dtype = jnp.result_type(levels, knots, values, 1.0)
    levels, knots, values = (
        jnp.asarray(x, dtype=dtype) for x in (levels, knots, values)
    )
    if knots.ndim != 1 or knots.size < 3 or values.shape[-1] != knots.size:
        raise ValueError("Single wells require at least three matching 1D knots.")
    levels = jnp.atleast_1d(levels)
    finite = jnp.all(jnp.isfinite(values), axis=-1)
    values = jnp.where(jnp.isfinite(values), values, 0.0)
    minimum_index = jnp.argmin(values, axis=-1)
    minimum = jnp.min(values, axis=-1)
    last_minimum = jnp.max(
        jnp.where(values == minimum[..., None], jnp.arange(knots.size), -1), axis=-1
    )
    differences = jnp.diff(values, axis=-1)
    monotone = jnp.all(
        jnp.where(
            jnp.arange(knots.size - 1) < minimum_index[..., None],
            differences <= 0,
            differences >= 0,
        ),
        axis=-1,
    )
    well_valid = (
        finite & monotone & (minimum_index > 0) & (last_minimum < knots.size - 1)
    )
    well_valid &= jnp.all(jnp.isfinite(knots)) & jnp.all(jnp.diff(knots) > 0)
    valid = (
        well_valid[..., None]
        & jnp.isfinite(levels)
        & (levels > minimum[..., None])
        & (levels <= jnp.minimum(values[..., 0], values[..., -1])[..., None])
    )
    low, high = values[..., None, :-1], values[..., None, 1:]
    descending = (low >= levels[..., None]) & (high < levels[..., None])
    ascending = (low < levels[..., None]) & (high >= levels[..., None])
    valid &= (jnp.sum(descending, axis=-1) == 1) & (jnp.sum(ascending, axis=-1) == 1)

    def invert(mask, reverse):
        index = jnp.argmax(mask, axis=-1)
        samples = jnp.broadcast_to(values[..., None, :], (*mask.shape[:-1], knots.size))
        field_pair = jnp.take_along_axis(
            samples, index[..., None] + jnp.arange(2), axis=-1
        )
        coordinate_pair = knots[index[..., None] + jnp.arange(2)]
        if reverse:
            field_pair, coordinate_pair = (
                field_pair[..., ::-1],
                coordinate_pair[..., ::-1],
            )
        # Finite private placeholders keep invalid inputs out of interpolator AD.
        field_pair = jnp.where(valid[..., None], field_pair, jnp.array([0.0, 1.0]))
        coordinate_pair = jnp.where(
            valid[..., None], coordinate_pair, jnp.array([0.0, 1.0])
        )
        query = jnp.where(valid, levels, 0.5)[..., None]
        root = interp1d_vec(query, field_pair, coordinate_pair, method="linear")[..., 0]
        return jnp.where(valid, root, 0.0)

    left, right = invert(descending, True), invert(ascending, False)
    valid &= jnp.isfinite(left) & jnp.isfinite(right) & (left < right)
    return jnp.where(valid, left, 0.0), jnp.where(valid, right, 0.0), valid


def _enforce_equal_bounce_distance(
    well_data,
    zeta,
    levels,
    *,
    field_tol=1e-12,
    weight_regularization=1e-12,
    knot_margin=1e-12,
    fieldline_batch_size=None,
    surf_batch_size=1,
):
    """Apply common tops, single-well inverse branches, and paired QI shuffle.

    Inputs are the array dictionary from ``_construct_single_well``, Boozer
    toroidal samples ``zeta`` (nz,), and normalized field levels ``levels`` (nB,).
    Returns the extended dictionary with normalized ``B_target``, shared bounce
    distances (nr, nB), centers (nr, na, nB), and detailed validity masks.

    Notes
    -----
    For input minima tied within ``field_tol``, the constructed field attains the
    common surface minimum ``B_min`` at the coordinate midpoint of the leftmost
    and rightmost tied samples. The midpoint is not rounded to a sample node;
    a unique minimum retains its position. The input field may exceed ``B_min``
    at this location. The zero bottom bounce distance replaces the minimum
    plateau retained by the base construction with a single bottom point.
    """
    zeta, levels = jnp.asarray(zeta), jnp.asarray(levels)
    B_min, B_max = well_data["B_min"], well_data["B_max"]
    span = jnp.where(well_data["valid_span"], B_max - B_min, 1.0)
    den = well_data["endpoint_B"] - B_min[:, None, None]
    valid_den = den / span[:, None, None] > field_tol
    den_safe = jnp.where(valid_den, den, 1.0)
    numerator = well_data["B_single"] - B_min[:, None, None]
    minimum_indices = well_data["minimum_indices"]
    # Place the constructed field's common B_min at the coordinate midpoint of
    # the leftmost and rightmost tied minima. Keep the midpoint between samples
    # when necessary to preserve reflection symmetry.
    minimum_zeta = (zeta[minimum_indices[..., 0]] + zeta[minimum_indices[..., 1]]) / 2
    # With d(0)=0, the inverse representation regularizes a minimum plateau into
    # a single bottom point. It does not retain every original minimum location.
    # TODO: support finite bottom widths, continuous transitions into/out of ties,
    # and validate AD and lower-level resolution convergence near these changes.
    stretched = jnp.where(
        zeta < minimum_zeta[..., None],
        numerator / den_safe[..., :1],
        numerator / den_safe[..., 1:],
    )
    # Invalid inactive inputs use a finite triangular well for branch inversion.
    valid_stretch = jnp.all(valid_den, axis=-1)
    active = well_data["valid_single_well"] & valid_stretch
    xi = (zeta - zeta[0]) / (zeta[-1] - zeta[0])
    root_B = jnp.where(active[..., None], stretched, jnp.abs(2 * xi - 1))

    def surface_roots(B):
        def line_roots(B_line):
            return _single_well_bounce_points(levels[1:-1], xi, B_line)

        return vmap_chunked(line_roots, chunk_size=fieldline_batch_size)(B)

    left, right, root_mask = vmap_chunked(surface_roots, chunk_size=surf_batch_size)(
        root_B
    )
    values_left = interp1d_vec(left, xi, root_B, method="linear")
    values_right = interp1d_vec(right, xi, root_B, method="linear")
    minimum_xi = (minimum_zeta - zeta[0]) / (zeta[-1] - zeta[0])
    valid_roots = (
        root_mask
        & jnp.isfinite(left)
        & jnp.isfinite(right)
        & (left >= 0)
        & (right <= 1)
        & (left < minimum_xi[..., None])
        & (right > minimum_xi[..., None])
        & (right > left)
        & (jnp.abs(values_left - levels[1:-1]) <= field_tol)
        & (jnp.abs(values_right - levels[1:-1]) <= field_tol)
    )
    period = zeta[-1] - zeta[0]
    bottom = minimum_zeta[..., None]
    left = jnp.concatenate(
        (bottom, zeta[0] + period * left, jnp.full_like(bottom, zeta[0])),
        axis=-1,
    )
    right = jnp.concatenate(
        (bottom, zeta[0] + period * right, jnp.full_like(bottom, zeta[-1])),
        axis=-1,
    )
    mismatch = jnp.mean((well_data["B_normalized"] - stretched) ** 2, axis=-1)
    weights = 1 / (mismatch + weight_regularization)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    distances = jnp.sum(weights[..., None] * (right - left), axis=1)
    distances = distances.at[:, 0].set(0).at[:, -1].set(period)
    centers, valid_projection = _project_centers(
        (left + right) / 2, distances, zeta, knot_margin
    )
    knots = _constructed_field_knots(centers, distances[:, None, :], zeta)
    gaps = jnp.diff(knots, axis=-1)
    roundoff = (
        32 * jnp.finfo(knots.dtype).eps * jnp.maximum(period, jnp.max(jnp.abs(zeta)))
    )
    valid_knots = jnp.all(
        (gaps > 0) & (gaps >= knot_margin * period - roundoff), axis=-1
    )
    valid_knots &= jnp.all(jnp.isfinite(knots), axis=-1)
    valid_knots &= (jnp.abs(knots[..., 0] - zeta[0]) <= roundoff) & (
        jnp.abs(knots[..., -1] - zeta[-1]) <= roundoff
    )
    valid_surface = jnp.all(
        active & jnp.all(valid_roots, axis=-1) & valid_projection & valid_knots, axis=-1
    )
    target = _evaluate_field_native(
        dict(
            zeta=zeta,
            B_levels=levels,
            bounce_centers=centers,
            bounce_distances=distances,
            valid_knots=valid_knots,
        )
    )
    valid_target = jnp.all(jnp.isfinite(target), axis=(1, 2))
    valid_surface &= valid_target
    return dict(
        well_data,
        B_stretched=stretched,
        B_target=target,
        B_levels=levels,
        minimum_zeta=minimum_zeta,
        zeta=zeta,
        bounce_centers=centers,
        bounce_distances=distances,
        weights=weights,
        valid_stretch=valid_stretch,
        valid_roots=valid_roots,
        valid_projection=valid_projection,
        valid_knots=valid_knots,
        valid_target=valid_target,
        valid_surface=valid_surface,
    )


def _sample_validity(
    B, rho, fieldline_labels, zeta, iota, *, NFP, sym=False, field_tol=1e-12
):
    """Return dynamic coordinate, iota, and reflection validity per surface."""
    rho, fieldline_labels, zeta, iota = map(
        jnp.asarray, (rho, fieldline_labels, zeta, iota)
    )
    period = 2 * jnp.pi / NFP
    coordinate_tol = (
        32
        * jnp.finfo(jnp.result_type(zeta, 1.0)).eps
        * jnp.maximum(1.0, jnp.max(jnp.abs(zeta)))
    )
    label_tol = 32 * jnp.finfo(jnp.result_type(fieldline_labels, 1.0)).eps * 2 * jnp.pi
    dz = jnp.diff(zeta)
    label_gaps = jnp.diff(
        jnp.concatenate((fieldline_labels, fieldline_labels[:1] + 2 * jnp.pi))
    )
    valid = (
        jnp.all(jnp.isfinite(rho))
        & jnp.all((rho > 0) & (rho <= 1))
        & jnp.all(jnp.diff(rho) > 0)
    )
    valid &= jnp.all(jnp.isfinite(zeta)) & jnp.all(dz > 0)
    valid &= jnp.abs(zeta[-1] - zeta[0] - period) <= coordinate_tol
    valid &= jnp.all(jnp.abs(dz - period / (zeta.size - 1)) <= coordinate_tol)
    valid &= jnp.all(jnp.isfinite(fieldline_labels)) & jnp.all(
        (fieldline_labels >= 0) & (fieldline_labels < 2 * jnp.pi)
    )
    valid &= jnp.all(label_gaps > 0) & jnp.all(
        jnp.abs(label_gaps - 2 * jnp.pi / fieldline_labels.size) <= label_tol
    )
    valid_symmetry = jnp.ones(rho.shape, dtype=bool)
    if sym:
        reflection_error = jnp.abs(
            jnp.mod(
                fieldline_labels[:, None] + fieldline_labels[None, :] + jnp.pi,
                2 * jnp.pi,
            )
            - jnp.pi
        )
        reflection_indices = jnp.argmin(reflection_error, axis=-1)
        closed = jnp.all(jnp.min(reflection_error, axis=-1) <= label_tol)
        phase = 2 * zeta[0] / period
        closed &= jnp.abs(phase - jnp.round(phase)) <= coordinate_tol
        span = jnp.max(B, axis=(1, 2)) - jnp.min(B, axis=(1, 2))
        error = jnp.max(jnp.abs(B - B[:, reflection_indices, ::-1]), axis=(1, 2))
        valid_symmetry = closed & (error <= field_tol * span)
    return {
        "valid_coordinates": jnp.broadcast_to(valid, rho.shape),
        "valid_iota": jnp.isfinite(iota),
        "valid_symmetry": valid_symmetry,
    }


def _construct_field(
    B,
    zeta,
    levels,
    *,
    span_rtol=1e-12,
    field_tol=1e-12,
    weight_regularization=1e-12,
    knot_margin=1e-12,
    fieldline_batch_size=None,
    surf_batch_size=1,
    rho=None,
    fieldline_labels=None,
    iota=None,
    NFP=None,
    sym=False,
):
    """Construct a QI field with the SQuID-like method, returning arrays and masks.

    ``B`` has shape (nr, na, nz), ``zeta`` shape (nz,), and ``levels`` shape (nB,).
    Optional coordinate metadata is supplied together by the public factories;
    it adds full coordinate and stellarator-reflection validation.
    """
    well = _construct_single_well(
        B,
        span_rtol=span_rtol,
        field_tol=field_tol,
    )
    result = _enforce_equal_bounce_distance(
        well,
        zeta,
        levels,
        field_tol=field_tol,
        weight_regularization=weight_regularization,
        knot_margin=knot_margin,
        fieldline_batch_size=fieldline_batch_size,
        surf_batch_size=surf_batch_size,
    )
    if rho is not None:
        masks = _sample_validity(
            B, rho, fieldline_labels, zeta, iota, NFP=NFP, sym=sym, field_tol=field_tol
        )
        if sym:
            output_masks = _sample_validity(
                result["B_target"],
                rho,
                fieldline_labels,
                zeta,
                iota,
                NFP=NFP,
                sym=True,
                field_tol=field_tol,
            )
            masks["valid_symmetry"] &= output_masks["valid_symmetry"]
        result.update(masks)
        for mask in masks.values():
            result["valid_surface"] &= mask
    return result


def _constructed_field_knots(centers, distances, zeta):
    """Join inverse branches with exact period endpoints, broadcasting widths."""
    left = centers - distances / 2
    right = centers + distances / 2
    knots = jnp.concatenate((left[..., ::-1], right[..., 1:]), axis=-1)
    # Reconstructing endpoints as midpoint +/- half-period can move them inward
    # by one rounding unit. The extreme levels use the exact period endpoints.
    return knots.at[..., 0].set(zeta[0]).at[..., -1].set(zeta[-1])


def _evaluate_field_native(data):
    """Evaluate normalized B on the stored construction labels and zeta samples.

    Return shape (nr, na, nz) directly from the inverse branches, without label
    interpolation. Invalid knots use finite private placeholders; callers must
    apply the surface validity mask before exposing values.
    """
    zeta, levels = data["zeta"], data["B_levels"]
    knots = _constructed_field_knots(
        data["bounce_centers"], data["bounce_distances"][:, None, :], zeta
    )
    safe_knots = jnp.where(
        data["valid_knots"][..., None],
        knots,
        jnp.linspace(zeta[0], zeta[-1], knots.shape[-1]),
    )
    values = jnp.concatenate((levels[::-1], levels[1:]))
    return interp1d_vec(zeta, safe_knots, values, method="linear")


def _constructed_surface_index(rho, query_rho):
    """Match literal query surfaces to stored surfaces, returning -1 if absent."""
    match = query_rho[:, None] == rho[None, :]
    return jnp.where(jnp.any(match, axis=1), jnp.argmax(match, axis=1), -1)


@register_compute_fun(
    name="Bc normalized",
    label="b_C",
    units="~",
    units_long="None",
    description="Constructed magnetic field strength normalized by its stored "
    "surface minimum and maximum",
    dim=1,
    params=[
        "rho",
        "zeta",
        "B_levels",
        "bounce_centers",
        "bounce_distances",
        "diagnostics",
        "fieldline_labels",
        "iota",
        "valid_surface",
    ],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldConstructed",
    native_grid="bool: Query the original construction nodes in C order "
    "(rho, midpoint field-line label, zeta_B). Default False. The field interface "
    "sets this when no explicit query grid is supplied.",
)
def _Bc_normalized(params, transforms, profiles, data, **kwargs):
    representation = {
        "zeta": params["zeta"],
        "B_levels": params["B_levels"],
        "bounce_centers": params["bounce_centers"],
        "bounce_distances": params["bounce_distances"],
    }
    representation["valid_knots"] = params["diagnostics"]["valid_knots"]
    if kwargs.get("native_grid", False):
        normalized = _evaluate_field_native(representation)
        valid = params["valid_surface"][:, None, None]
    else:
        grid = transforms["grid"]
        index = _constructed_surface_index(params["rho"], grid.nodes[:, 0])
        safe_index = jnp.maximum(index, 0)
        valid = (index >= 0) & params["valid_surface"][safe_index]
        zeta = params["zeta"]
        zeta_bar = zeta[0] + jnp.mod(grid.nodes[:, 2] - zeta[0], 2 * jnp.pi / grid.NFP)
        iota = jnp.where(valid, params["iota"][safe_index], 0.0)
        chi = jnp.mod(
            grid.nodes[:, 1] - iota * (zeta_bar - (zeta[0] + zeta[-1]) / 2),
            2 * jnp.pi,
        )
        normalized = _evaluate_field(
            representation, params["fieldline_labels"], chi, zeta_bar, safe_index
        )
    data["Bc normalized"] = jnp.where(valid, normalized, jnp.nan).ravel()
    return data


@register_compute_fun(
    name="|B| constructed",
    label="|\\mathbf{B}_c|",
    units="T",
    units_long="Tesla",
    description="Magnitude of the constructed magnetic field at literal Boozer "
    "query coordinates",
    dim=1,
    params=["rho", "zeta", "B_min", "B_max", "valid_surface"],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["Bc normalized"],
    parameterization="desc.magnetic_fields._core.OmnigenousFieldConstructed",
    native_grid="bool: Query the original construction nodes in C order "
    "(rho, midpoint field-line label, zeta_B). Default False. The field interface "
    "sets this when no explicit query grid is supplied.",
)
def _B_constructed(params, transforms, profiles, data, **kwargs):
    if kwargs.get("native_grid", False):
        # Retain the evaluator's layout so XLA can fuse both requested strengths.
        normalized = data["Bc normalized"].reshape(
            (params["rho"].size, -1, params["zeta"].size)
        )
        valid = params["valid_surface"][:, None, None]
        minimum, maximum = (
            params["B_min"][:, None, None],
            params["B_max"][:, None, None],
        )
    else:
        index = _constructed_surface_index(
            params["rho"], transforms["grid"].nodes[:, 0]
        )
        safe_index = jnp.maximum(index, 0)
        normalized = data["Bc normalized"]
        valid = (index >= 0) & params["valid_surface"][safe_index]
        minimum, maximum = params["B_min"][safe_index], params["B_max"][safe_index]
    # Sanitize before multiplication so masked NaNs cannot poison reverse AD.
    minimum = jnp.where(valid, minimum, 0.0)
    maximum = jnp.where(valid, maximum, 1.0)
    normalized = jnp.where(valid, normalized, 0.0)
    field = minimum + (maximum - minimum) * normalized
    data["|B| constructed"] = jnp.where(valid, field, jnp.nan).ravel()
    return data


def _evaluate_bounce(data, beta):
    """Interpolate inverse branches at beta (nr, nq); return roots (nr, na, nq)."""
    beta = jnp.asarray(beta)
    centers = interp1d_vec(
        beta[:, None, :], data["B_levels"], data["bounce_centers"], method="linear"
    )
    distances = interp1d_vec(
        beta, data["B_levels"], data["bounce_distances"], method="linear"
    )
    left, right = (
        centers - distances[:, None, :] / 2,
        centers + distances[:, None, :] / 2,
    )
    if "zeta" in data:
        left = jnp.where(beta[:, None, :] == 1, data["zeta"][0], left)
        right = jnp.where(beta[:, None, :] == 1, data["zeta"][-1], right)
    return left, right


def _evaluate_field(data, fieldline_labels, chi, zeta_query, surface_index):
    """Evaluate normalized field by periodic center interpolation on selected surfaces.

    Query arrays broadcast to a common shape. The caller wraps toroidal coordinates
    and forms the corresponding midpoint labels; this helper does not change zeta.
    """
    chi, zeta_query, surface_index = jnp.broadcast_arrays(
        chi, zeta_query, surface_index
    )
    labels = jnp.concatenate((fieldline_labels, fieldline_labels[:1] + 2 * jnp.pi))
    values = jnp.concatenate((data["B_levels"][::-1], data["B_levels"][1:]))

    def one(label, zeta, surface):
        centers = data["bounce_centers"][surface]
        centers = jnp.concatenate((centers, centers[:1]), axis=0).T
        label = jnp.mod(label - labels[0], 2 * jnp.pi) + labels[0]
        centers = interp1d_vec(jnp.atleast_1d(label), labels, centers, method="linear")[
            :, 0
        ]
        distances = data["bounce_distances"][surface]
        left, right = centers - distances / 2, centers + distances / 2
        knots = jnp.concatenate((left[::-1], right[1:]))
        if "zeta" in data:
            knots = knots.at[0].set(data["zeta"][0]).at[-1].set(data["zeta"][-1])
        return interp1d_vec(jnp.atleast_1d(zeta), knots, values, method="linear")[0]

    result = vmap_chunked(one, in_axes=(0, 0, 0))(
        chi.ravel(), zeta_query.ravel(), surface_index.ravel()
    )
    return result.reshape(chi.shape)


def _sample_boozer_data(
    transforms,
    data,
    fieldline_labels,
    zeta,
    *,
    fieldline_batch_size=None,
    surf_batch_size=1,
):
    """Evaluate current Boozer harmonics at fixed midpoint labels and current iota."""
    grid, basis = transforms["grid"], transforms["B"].basis
    coefficients = data["|B|_mn_B"].reshape((grid.num_rho, -1))
    iota = data["iota"][grid.unique_rho_idx]
    midpoint = (zeta[0] + zeta[-1]) / 2

    def surface(B_mn, iota):
        def line(chi):
            theta = chi + iota * (zeta - midpoint)
            nodes = jnp.column_stack((jnp.zeros_like(zeta), theta, zeta))
            return basis.evaluate(nodes) @ B_mn

        return vmap_chunked(line, chunk_size=fieldline_batch_size)(fieldline_labels)

    B = vmap_chunked(surface, in_axes=(0, 0), chunk_size=surf_batch_size)(
        coefficients, iota
    )
    return B, iota
