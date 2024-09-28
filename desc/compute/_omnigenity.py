"""Compute functions for omnigenity objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from interpax import interp1d

from desc.backend import jnp, sign, vmap

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


# TODO: do math to change definition of nu so that we can just use B_zeta_mn here
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
    aliases="B_zeta_mn",  # TODO: remove when phi != zeta
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
    name="sqrt(g)_B",
    label="\\sqrt{g}_{B}",
    units="~",
    units_long="None",
    description="Jacobian determinant from Boozer to DESC coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["theta_PEST_t", "theta_PEST_z", "phi_t", "phi_z", "nu_t", "nu_z", "iota"],
)
def _sqrtg_B(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_B"] = (
        data["theta_PEST_t"] * (data["phi_z"] + data["nu_z"])
        - data["theta_PEST_z"] * (data["phi_t"] + data["nu_t"])
        + data["iota"] * (data["nu_t"] * data["phi_z"] - data["nu_z"] * data["phi_t"])
    )
    return data


@register_compute_fun(
    name="|B|_mn",
    label="B_{mn}^{\\mathrm{Boozer}}",
    units="T",
    units_long="Tesla",
    description="Boozer harmonics of magnetic field",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)_B", "|B|", "rho", "theta_B", "zeta_B"],
    resolution_requirement="tz",
    grid_requirement={"is_meshgrid": True},
    M_booz="int: Maximum poloidal mode number for Boozer harmonics. Default 2*eq.M",
    N_booz="int: Maximum toroidal mode number for Boozer harmonics. Default 2*eq.N",
)
def _B_mn(params, transforms, profiles, data, **kwargs):
    norm = 2 ** (3 - jnp.sum((transforms["B"].basis.modes == 0), axis=1))
    grid = transforms["grid"]

    def fun(rho, theta_B, zeta_B, sqrtg_B, B):
        # this fits Boozer modes on a single surface
        nodes = jnp.array([rho, theta_B, zeta_B]).T
        B_mn = (
            norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            * (transforms["B"].basis.evaluate(nodes).T @ (sqrtg_B * B))
            / transforms["B"].grid.num_nodes
        )
        return B_mn

    def reshape(x):
        return grid.meshgrid_reshape(x, "rtz").reshape((grid.num_rho, -1))

    rho, theta_B, zeta_B, sqrtg_B, B = map(
        reshape,
        (data["rho"], data["theta_B"], data["zeta_B"], data["sqrt(g)_B"], data["|B|"]),
    )
    B_mn = vmap(fun)(rho, theta_B, zeta_B, sqrtg_B, B)
    data["|B|_mn"] = B_mn.flatten()
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
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["alpha", "h"],
    parameterization="desc.magnetic_fields._core.OmnigenousField",
    helicity="tuple: Type of quasisymmetry, (M,N). Default (1,0)",
    iota="float: Value of rotational transform on the Omnigenous surface. Default 1.0",
)
def _omni_map_theta_B(params, transforms, profiles, data, **kwargs):
    M, N = kwargs.get("helicity", (1, 0))
    iota = kwargs.get("iota", 1)

    # coordinate mapping matrix from (alpha,h) to (theta_B,zeta_B)
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

    # solve for (theta_B,zeta_B) corresponding to (eta,alpha)
    booz = matrix @ jnp.vstack((data["alpha"], data["h"]))
    data["theta_B"] = booz[0, :]
    data["zeta_B"] = booz[1, :]
    return data


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
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["eta"],
    parameterization="desc.magnetic_fields._core.OmnigenousField",
)
def _B_omni(params, transforms, profiles, data, **kwargs):
    # reshaped to size (L_B, M_B)
    B_lm = params["B_lm"].reshape((transforms["B"].basis.L + 1, -1))
    # assuming single flux surface, so only take first row (single node)
    B_input = vmap(lambda x: transforms["B"].transform(x))(B_lm.T)[:, 0]
    B_input = jnp.sort(B_input)  # sort to ensure monotonicity
    eta_input = jnp.linspace(0, jnp.pi / 2, num=B_input.size)

    # |B|_omnigeneous is an even function so B(-eta) = B(+eta) = B(|eta|)
    data["|B|"] = interp1d(
        jnp.abs(data["eta"]), eta_input, B_input, method="monotonic-0"
    )
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
