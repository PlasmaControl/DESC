"""Compute functions for quasisymmetry objectives."""

import numpy as np

from desc.backend import jnp, put, sign
from desc.vmec_utils import ptolemy_linear_transform

from .data_index import register_compute_fun
from .utils import expand, surface_averages, surface_integrals, surface_min, surface_max


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
    label="w_{Boozer,m,n}",
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
    ib, iw = np.where((Bm[:, None] == -wm) * (Bn[:, None] == wn) * (wm != 0))
    jb, jw = np.where(
        (Bm[:, None] == wm) * (Bn[:, None] == -wn) * (wm == 0) * (wn != 0)
    )
    w_mn = put(w_mn, iw, sign(wn[iw]) * data["B_theta_mn"][ib] / jnp.abs(wm[iw]))
    w_mn = put(w_mn, jw, sign(wm[jw]) * data["B_zeta_mn"][jb] / jnp.abs(NFP * wn[jw]))
    data["w_Boozer_mn"] = w_mn
    return data


@register_compute_fun(
    name="w_Boozer",
    label="w_{Boozer}",
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
    label="\\partial_{\\theta} w_{Boozer}",
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
    label="\\partial_{\\zeta} w_{Boozer}",
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
    name="sqrt(g)_B",
    label="\\sqrt{g}_{B}",
    units="~",
    units_long="None",
    description="Jacobian determinant of Boozer coordinates",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["lambda_t", "lambda_z", "nu_t", "nu_z", "iota"],
)
def _sqrtg_B(params, transforms, profiles, data, **kwargs):
    data["sqrt(g)_B"] = (1 + data["lambda_t"]) * (1 + data["nu_z"]) + (
        data["iota"] - data["lambda_z"]
    ) * data["nu_t"]
    return data


@register_compute_fun(
    name="|B|_mn",
    label="B_{mn}^{Boozer}",
    units="T",
    units_long="Tesla",
    description="Boozer harmonics of magnetic field",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["sqrt(g)_B", "|B|", "rho", "theta_B", "zeta_B"],
)
def _B_mn(params, transforms, profiles, data, **kwargs):
    nodes = jnp.array([data["rho"], data["theta_B"], data["zeta_B"]]).T
    norm = 2 ** (3 - jnp.sum((transforms["B"].basis.modes == 0), axis=1))
    data["|B|_mn"] = (
        norm  # 1 if m=n=0, 2 if m=0 or n=0, 4 if m!=0 and n!=0
        * jnp.matmul(
            transforms["B"].basis.evaluate(nodes).T, data["sqrt(g)_B"] * data["|B|"]
        )
        / transforms["B"].grid.num_nodes
    )
    return data


@register_compute_fun(
    name="B modes",
    label="Boozer modes",
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
    name="|B|_mn symmetrized",
    label="B_{mn}^{Boozer} symmetrized",
    units="T",
    units_long="Tesla",
    description="Boozer harmonics of magnetic field, keeping only quasisymmetric modes",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["|B|_mn"],
    helicity="helicity",
)
def _B_mn_symmetrized(params, transforms, profiles, data, **kwargs):
    matrix, modes, indices_of_nonsymmetric_modes = ptolemy_linear_transform(
        transforms["B"].basis.modes,
        helicity=kwargs.get("helicity", (1, 0)),
        NFP=transforms["B"].basis.NFP,
    )
    print("Condition number:", jnp.linalg.cond(matrix))
    print("NFP:", transforms["B"].basis.NFP)
    print("matrix shape:", matrix.shape)
    print("matrix:\n", matrix)
    print("helicity:", kwargs.get("helicity", (1, 0)))
    print("modes:\n", modes)
    print("indices_of_nonsymmetric_modes:", indices_of_nonsymmetric_modes)
    filter = jnp.ones(transforms["B"].basis.num_modes)
    filter = put(filter, indices_of_nonsymmetric_modes, 0)
    print("filter:", filter)
    # matrix_inv = jnp.linalg.inv(matrix)
    # print("matrix_inv:\n", matrix_inv)
    # data["|B|_mn symmetrized"] = matrix_inv @ (filter * (matrix @ data["|B|_mn"]))
    data["|B|_mn symmetrized"] = jnp.linalg.solve(
        matrix, filter * (matrix @ data["|B|_mn"])
    )
    return data


@register_compute_fun(
    name="|B| Boozer",
    label="|B| Boozer",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field as function of the Boozer angles",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["|B|_mn"],
)
def _B_Boozer(params, transforms, profiles, data, **kwargs):
    data["|B| Boozer"] = transforms["B"].transform(data["|B|_mn"])
    return data


@register_compute_fun(
    name="|B| Boozer symmetrized",
    label="|B| Boozer symmetrized",
    units="T",
    units_long="Tesla",
    description="Magnitude of magnetic field as function of the Boozer angles, keeping only quasisymmetric modes",
    dim=1,
    params=[],
    transforms={"B": [[0, 0, 0]]},
    profiles=[],
    coordinates="rtz",
    data=["|B|_mn symmetrized"],
)
def _B_Boozer_symmetrized(params, transforms, profiles, data, **kwargs):
    data["|B| Boozer symmetrized"] = transforms["B"].transform(
        data["|B|_mn symmetrized"]
    )
    return data


@register_compute_fun(
    name="|B|^2 Boozer symmetrized",
    label="|B|^2 Boozer symmetrized",
    units="T",
    units_long="Tesla",
    description="Squared magnitude of magnetic field as function of the Boozer angles, keeping only quasisymmetric modes",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B| Boozer symmetrized"],
)
def _B2_Boozer_symmetrized(params, transforms, profiles, data, **kwargs):
    data["|B|^2 Boozer symmetrized"] = data["|B| Boozer symmetrized"] ** 2
    return data


@register_compute_fun(
    name="sqrt(g) Boozer symmetrized",
    label="\\sqrt{g} Boozer symmetrized",
    units="T",
    units_long="Tesla",
    description="Boozer-coordinate Jacobian, keeping only quasisymmetric modes",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["|B|^2 Boozer symmetrized", "G", "iota", "I"],
)
def _sqrt_g_Boozer_symmetrized(params, transforms, profiles, data, **kwargs):
    data["sqrt(g) Boozer symmetrized"] = (data["G"] + data["iota"] * data["I"]) / data[
        "|B|^2 Boozer symmetrized"
    ]
    return data


@register_compute_fun(
    name="V_r(r) symmetrized",
    label="dV/d\\rho symmetrized",
    units="T",
    units_long="Tesla",
    description="Derivative of flux surface volume with respect to radius, keeping only quasisymmetric modes",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g) Boozer symmetrized"],
)
def _V_r_of_r_symmetrized(params, transforms, profiles, data, **kwargs):
    data["V_r(r) symmetrized"] = surface_integrals(
        transforms["grid"], jnp.abs(data["sqrt(g) Boozer symmetrized"])
    )
    return data


@register_compute_fun(
    name="<B^2> symmetrized",
    label="\\langle B^2 \\rangle",
    units="T^2",
    units_long="Tesla squared",
    description="Flux surface average magnetic field squared",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=[
        "sqrt(g) Boozer symmetrized",
        "|B|^2 Boozer symmetrized",
        "V_r(r) symmetrized",
    ],
)
def _B2_fsa_symmetrized(params, transforms, profiles, data, **kwargs):
    data["<B^2> symmetrized"] = surface_averages(
        transforms["grid"],
        data["|B|^2 Boozer symmetrized"],
        jnp.abs(data["sqrt(g) Boozer symmetrized"]),
        denominator=data["V_r(r) symmetrized"],
    )
    return data


@register_compute_fun(
    name="<1/|B|> symmetrized",
    label="\\langle 1/B \\rangle symmetrized",
    units="T^{-1}",
    units_long="1 / Tesla",
    description="Flux surface averaged inverse field strength",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g) Boozer symmetrized", "|B| Boozer symmetrized", "V_r(r) symmetrized"],
)
def _1_over_B_fsa_Boozer_symmetrized(params, transforms, profiles, data, **kwargs):
    data["<1/|B|> symmetrized"] = surface_averages(
        transforms["grid"],
        1 / data["|B| Boozer symmetrized"],
        jnp.abs(data["sqrt(g) Boozer symmetrized"]),
        denominator=data["V_r(r) symmetrized"],
    )
    return data


@register_compute_fun(
    name="max_tz |B| symmetrized",
    label="\\max_{\\theta \\zeta} |B|, symmetrized",
    units="T",
    units_long="Tesla",
    description="Maximum field strength on each flux surface, after filtering"
    "out non-quasi-symmetric modes",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|B| Boozer symmetrized"],
)
def _max_tz_modB_symmetrized(params, transforms, profiles, data, **kwargs):
    data["max_tz |B| symmetrized"] = expand(
        transforms["grid"],
        surface_max(transforms["grid"], data["|B| Boozer symmetrized"]),
    )
    return data


@register_compute_fun(
    name="min_tz |B| symmetrized",
    label="\\min_{\\theta \\zeta} |B|, symmetrized",
    units="T",
    units_long="Tesla",
    description="Minimum field strength on each flux surface, after filtering"
    "out non-quasi-symmetric modes",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["|B| Boozer symmetrized"],
)
def _min_tz_modB_symmetrized(params, transforms, profiles, data, **kwargs):
    data["min_tz |B| symmetrized"] = expand(
        transforms["grid"],
        surface_min(transforms["grid"], data["|B| Boozer symmetrized"]),
    )
    return data


@register_compute_fun(
    name="effective r/R0 symmetrized",
    label="(r / R_0)_{effective}, symmetrized",
    units="~",
    units_long="None",
    description="Effective local inverse aspect ratio, based on max and min |B|"
    "after filtering out non-quasi-symmetric modes",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="r",
    data=["max_tz |B| symmetrized", "min_tz |B| symmetrized"],
)
def _effective_r_over_R0_symmetrized(params, transforms, profiles, data, **kwargs):
    r"""
    Compute an effective local inverse aspect ratio.

    This effective local inverse aspect ratio epsilon is defined by

    .. math::
        \frac{Bmax}{Bmin} = \frac{1 + \epsilon}{1 - \epsilon}

    This definition is motivated by the fact that this formula would
    be true in the case of circular cross-section surfaces in
    axisymmetry with :math:`B \propto 1/R` and :math:`R = (1 +
    \epsilon \cos\theta) R_0`.
    """
    w = data["max_tz |B| symmetrized"] / data["min_tz |B| symmetrized"]
    data["effective r/R0 symmetrized"] = (w - 1) / (w + 1)
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
    data=[
        "iota",
        "psi_r",
        "sqrt(g)",
        "B_theta",
        "B_zeta",
        "|B|_t",
        "|B|_z",
        "G",
        "I",
        "B*grad(|B|)",
    ],
    helicity="helicity",
)
def _f_C(params, transforms, profiles, data, **kwargs):
    M, N = kwargs.get("helicity", (1, 0))
    data["f_C"] = (M * data["iota"] - N) * (data["psi_r"] / data["sqrt(g)"]) * (
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
    data=[
        "psi_r",
        "sqrt(g)",
        "|B|_t",
        "|B|_z",
        "(B*grad(|B|))_t",
        "(B*grad(|B|))_z",
    ],
)
def _f_T(params, transforms, profiles, data, **kwargs):
    data["f_T"] = (data["psi_r"] / data["sqrt(g)"]) * (
        data["|B|_t"] * data["(B*grad(|B|))_z"]
        - data["|B|_z"] * data["(B*grad(|B|))_t"]
    )
    return data
