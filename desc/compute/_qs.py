"""Compute functions for quasisymmetry objectives.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

import numpy as np

from desc.backend import jnp, put, sign

from .data_index import register_compute_fun
from .utils import cross, dot


@register_compute_fun(
    name="B_theta_mn",
    label="B_{\\theta, m, n}",
    units="T \\cdot m}",
    units_long="Tesla * meters",
    description="Fourier coefficients for covariant poloidal component of "
    + "magnetic field",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["B_theta"],
)
def _B_theta_mn(params, transforms, profiles, data, **kwargs):
    data["B_theta_mn"] = transforms["B"].fit(data["B_theta"])
    return data


@register_compute_fun(
    name="B_zeta_mn",
    label="B_{\\zeta, m, n}",
    units="T \\cdot m}",
    units_long="Tesla * meters",
    description="Fourier coefficients for covariant toroidal component of "
    + "magnetic field",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["B_zeta"],
)
def _B_zeta_mn(params, transforms, profiles, data, **kwargs):
    data["B_zeta_mn"] = transforms["B"].fit(data["B_zeta"])
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
    transforms={"w": [[0, 0, 0]], "B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["B_theta_mn", "B_zeta_mn"],
)
def _w_mn(params, transforms, profiles, data, **kwargs):
    w_mn = jnp.zeros((transforms["w"].basis.num_modes,))
    Bm = transforms["B"].basis.modes[:, 1]
    Bn = transforms["B"].basis.modes[:, 2]
    wm = transforms["w"].basis.modes[:, 1]
    wn = transforms["w"].basis.modes[:, 2]
    NFP = transforms["w"].basis.NFP
    # indices of matching modes in w and B bases
    # need to use np instead of jnp here as jnp.where doesn't work under jit
    # even if the args are static
    ib, iw = np.where((Bm[:, None] == -wm) & (Bn[:, None] == wn) & (wm != 0))
    jb, jw = np.where(
        (Bm[:, None] == wm) & (Bn[:, None] == -wn) & (wm == 0) & (wn != 0)
    )
    w_mn = put(w_mn, iw, sign(wn[iw]) * data["B_theta_mn"][ib] / jnp.abs(wm[iw]))
    w_mn = put(w_mn, jw, sign(wm[jw]) * data["B_zeta_mn"][jb] / jnp.abs(NFP * wn[jw]))
    data["w_Boozer_mn"] = w_mn
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
    transforms={"w": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer_mn"],
)
def _w(params, transforms, profiles, data, **kwargs):
    data["w_Boozer"] = transforms["w"].transform(data["w_Boozer_mn"])
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
    transforms={"w": [[0, 1, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer_mn"],
)
def _w_t(params, transforms, profiles, data, **kwargs):
    data["w_Boozer_t"] = transforms["w"].transform(data["w_Boozer_mn"], dt=1)
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
    transforms={"w": [[0, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=["w_Boozer_mn"],
)
def _w_z(params, transforms, profiles, data, **kwargs):
    data["w_Boozer_z"] = transforms["w"].transform(data["w_Boozer_mn"], dz=1)
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
    name="nu_mn",
    label="\\nu_{mn} = (\\zeta_{B} - \\zeta)_{mn}",
    units="rad",
    units_long="radians",
    description="Boozer harmonics of Boozer toroidal stream function",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "nu",
        "rho",
        "theta_B",
        "zeta_B",
        "jac_Boozer_DESC",
        "Boozer transform prefactor",
    ],
)
def _nu_mn(params, transforms, profiles, data, **kwargs):
    nodes = jnp.array([data["rho"], data["theta_B"], data["zeta_B"]]).T
    norm = data["Boozer transform prefactor"]
    data["nu_mn"] = (
        norm
        * (  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            transforms["B"].basis.evaluate(nodes).T
            @ (data["jac_Boozer_DESC"] * data["nu"])
        )
        / transforms["B"].grid.num_nodes
    )
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
    data=["theta", "lambda", "iota", "nu"],
)
def _theta_B(params, transforms, profiles, data, **kwargs):
    data["theta_B"] = data["theta"] + data["lambda"] + data["iota"] * data["nu"]
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
    data=["zeta", "nu"],
)
def _zeta_B(params, transforms, profiles, data, **kwargs):
    data["zeta_B"] = data["zeta"] + data["nu"]
    return data


@register_compute_fun(
    name="jac_Boozer_DESC",
    label="\\frac{\\partial(\\theta_B,\\zeta_B)}{\\theta_{DESC},\\zeta_{DESC}}",
    units="~",
    units_long="None",
    description="Jacobian determinant from Boozer coordinates to DESC coordinates.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_t", "lambda_z", "nu_t", "nu_z", "iota"],
)
def _jac_Boozer_DESC(params, transforms, profiles, data, **kwargs):
    data["jac_Boozer_DESC"] = (1 + data["lambda_t"]) * (1 + data["nu_z"]) + (
        data["iota"] - data["lambda_z"]
    ) * data["nu_t"]
    return data


@register_compute_fun(
    name="sqrt(g)_B",
    label="\\sqrt{g}_B",
    units="~",
    units_long="None",
    description="Jacobian determinant from (rho, theta_B, zeta_B)"
    " Boozer coordinates to (R,phi,Z) lab frame.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["jac_Boozer_DESC", "sqrt(g)"],
)
def _sqrtg_B(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_B"] = data["sqrt(g)"] / data["jac_Boozer_DESC"]
    return data


@register_compute_fun(
    name="sqrt(g)_B_mn",
    label="\\sqrt{g}_{B,mn}",
    units="~",
    units_long="None",
    description="Boozer harmonics of Jacobian determinant from (rho, theta_B, zeta_B)"
    " Boozer coordinates to (R,phi,Z) lab frame.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=[
        "rho",
        "sqrt(g)_B",
        "jac_Boozer_DESC",
        "theta_B",
        "zeta_B",
        "Boozer transform prefactor",
    ],
)
def _sqrtg_B_mn(params, transforms, profiles, data, **kwargs):
    nodes = jnp.array([data["rho"], data["theta_B"], data["zeta_B"]]).T
    norm = data["Boozer transform prefactor"]
    data["sqrt(g)_B_mn"] = (
        norm
        * (  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            transforms["B"].basis.evaluate(nodes).T
            @ (data["jac_Boozer_DESC"] * data["sqrt(g)_B"])
        )
        / transforms["B"].grid.num_nodes
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
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[
        "jac_Boozer_DESC",
        "|B|",
        "rho",
        "theta_B",
        "zeta_B",
        "Boozer transform prefactor",
    ],
)
def _B_mn(params, transforms, profiles, data, **kwargs):
    nodes = jnp.array([data["rho"], data["theta_B"], data["zeta_B"]]).T
    norm = data["Boozer transform prefactor"]
    data["|B|_mn"] = (
        norm
        * (  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            transforms["B"].basis.evaluate(nodes).T
            @ (data["jac_Boozer_DESC"] * data["|B|"])
        )
        / transforms["B"].grid.num_nodes
    )
    return data


@register_compute_fun(
    name="R_mn",
    label="R_{mn}^{\\mathrm{Boozer}}",
    units="m",
    units_long="meters",
    description="Boozer harmonics of radial toroidal coordinate of a flux surface",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[
        "jac_Boozer_DESC",
        "R",
        "rho",
        "theta_B",
        "zeta_B",
        "Boozer transform prefactor",
    ],
)
def _R_mn(params, transforms, profiles, data, **kwargs):
    nodes = jnp.array([data["rho"], data["theta_B"], data["zeta_B"]]).T
    norm = data["Boozer transform prefactor"]
    data["R_mn"] = (
        norm
        * (  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            transforms["B"].basis.evaluate(nodes).T
            @ (data["jac_Boozer_DESC"] * data["R"])
        )
        / transforms["B"].grid.num_nodes
    )
    return data


@register_compute_fun(
    name="Z_mn",
    label="Z_{mn}^{\\mathrm{Boozer}}",
    units="m",
    units_long="meters",
    description="Boozer harmonics of vertical toroidal coordinate of a flux surface",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[
        "jac_Boozer_DESC",
        "Z",
        "rho",
        "theta_B",
        "zeta_B",
        "Boozer transform prefactor",
    ],
)
def _Z_mn(params, transforms, profiles, data, **kwargs):
    nodes = jnp.array([data["rho"], data["theta_B"], data["zeta_B"]]).T
    norm = data["Boozer transform prefactor"]
    data["Z_mn"] = (
        norm
        * (  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
            transforms["B"].basis.evaluate(nodes).T
            @ (data["jac_Boozer_DESC"] * data["Z"])
        )
        / transforms["B"].grid.num_nodes
    )
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
)
def _B_modes(params, transforms, profiles, data, **kwargs):
    data["B modes"] = transforms["B"].basis.modes
    return data


@register_compute_fun(
    name="Boozer transform prefactor",
    label="\\nu_{mn} = (\\zeta_{B} - \\zeta)_{mn}",
    units="rad",
    units_long="radians",
    description="Prefactor in front of the boozer transform",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=[],
)
def _boozer_prefactor(params, transforms, profiles, data, **kwargs):
    norm = 2 ** (3 - jnp.sum((transforms["B"].basis.modes == 0), axis=1))
    data[
        "Boozer transform prefactor"
    ] = norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
    return data


@register_compute_fun(
    name="f_C",
    label="(M \\iota - N) (\\mathbf{B} \\times \\nabla \\psi) \\cdot \\nabla B"
    + " - (M G + N I) \\mathbf{B} \\cdot \\nabla B",
    units="T^{3}",
    units_long="Tesla cubed",
    description="Two-term quasisymmetry metric",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["iota", "B0", "B_theta", "B_zeta", "|B|_t", "|B|_z", "G", "I", "B*grad(|B|)"],
    helicity="helicity",
)
def _f_C(params, transforms, profiles, data, **kwargs):
    M, N = kwargs.get("helicity", (1, 0))
    data["f_C"] = (M * data["iota"] - N) * data["B0"] * (
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
    data=["B0", "|B|_t", "|B|_z", "(B*grad(|B|))_t", "(B*grad(|B|))_z"],
)
def _f_T(params, transforms, profiles, data, **kwargs):
    data["f_T"] = data["B0"] * (
        data["|B|_t"] * data["(B*grad(|B|))_z"]
        - data["|B|_z"] * data["(B*grad(|B|))_t"]
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
    data=["b", "grad(|B|)", "|B|", "grad(psi)"],
)
def _isodynamicity(params, transforms, profiles, data, **kwargs):
    data["isodynamicity"] = (
        dot(cross(data["b"], data["grad(|B|)"]), data["grad(psi)"]) / data["|B|"] ** 2
    )
    return data
