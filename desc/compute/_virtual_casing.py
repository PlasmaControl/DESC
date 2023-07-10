"""Compute functions for magnetic field quantities."""

from scipy.constants import mu_0

from desc.backend import fori_loop, jnp, put
from desc.geometry.utils import rpz2xyz

from .data_index import register_compute_fun
from .utils import cross, dot


@register_compute_fun(
    name="K_vc",
    label="\\mathbf{K}_{VC} = \\mathbf{B} \\times \\mathbf{n}",
    units="A \\cdot m^{-1}",
    units_long="Amps / meter",
    description="Virtual casing sheet current",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["B", "n_rho"],
)
def _K_vc(params, transforms, profiles, data, **kwargs):
    data["K_vc"] = cross(data["B"], data["n_rho"]) / mu_0
    return data


@register_compute_fun(
    name="K_sc",
    label="\\mathbf{K}_{SC} = \\mathbf{n} \\times  \\nabla \\Phi_{SC}",
    units="A \\cdot m^{-1}",
    units_long="Amps / meter",
    description="Sheet current",
    dim=3,
    params=["IGPhi_mn"],
    transforms={"K": [[0, 1, 0], [0, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=["n_rho", "e^theta", "e^zeta"],
)
def _K_sc(params, transforms, profiles, data, **kwargs):
    I = params["IGPhi_mn"][0] / mu_0
    G = params["IGPhi_mn"][1] / mu_0
    Phi_mn = params["IGPhi_mn"][2:] / mu_0

    Phi_t = transforms["K"].transform(Phi_mn, dt=1) + I / (2 * jnp.pi)
    Phi_z = transforms["K"].transform(Phi_mn, dz=1) + G / (2 * jnp.pi)
    gradPhi = Phi_t[:, None] * data["e^theta"] + Phi_z[:, None] * data["e^zeta"]
    data["K_sc"] = cross(data["n_rho"], gradPhi)
    return data


@register_compute_fun(
    name="I_vc",
    label="I_{VC}",
    units="m^{-2}",
    units_long="inverse square meters",
    description="Virtual casing regularization integral",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["g_tt", "g_tz", "g_zz"],
)
def _I_vc(params, transforms, profiles, data, **kwargs):
    a = data["g_tt"] * 4 * jnp.pi**2
    b = data["g_tz"] * 4 * jnp.pi**2
    c = data["g_zz"] * 4 * jnp.pi**2
    rp = jnp.sqrt(a + 2 * b + c)
    rm = jnp.sqrt(a - 2 * b + c)
    Tp = 1 / rp * jnp.log((jnp.sqrt(c) * rp + c + b) / (jnp.sqrt(a) * rp - a - b))
    Tm = 1 / rm * jnp.log((jnp.sqrt(c) * rm + c - b) / (jnp.sqrt(a) * rm - a + b))
    data["I_vc"] = Tp + Tm
    return data


@register_compute_fun(
    name="A_vc",
    label="\\mathbf{A}_{VC}",
    units="T \\cdot m^{-1}",
    units_long="Tesla / meter",
    description="Virtual casing vector potential",
    dim=3,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=[
        "K_vc",
        "I_vc",
        "R",
        "phi",
        "Z",
        "theta",
        "zeta",
        "g_tt",
        "g_tz",
        "g_zz",
        "|e_theta x e_zeta|",
    ],
)
def _A_vc(params, transforms, profiles, data, **kwargs):
    assert transforms["grid"].num_rho == 1

    src_phi = data["phi"]
    src_zeta = data["zeta"]
    NFP = transforms["grid"].NFP

    ex = jnp.array([data["R"], data["phi"], data["Z"]]).T
    ex = rpz2xyz(ex)
    et = jnp.asarray(data["theta"])
    ez = jnp.asarray(data["zeta"])

    def body2(j, Adata):
        Areg1, Areg2, data = Adata
        data["phi"] = (src_phi + j * 2 * jnp.pi / NFP) % (2 * jnp.pi)
        data["zeta"] = (src_zeta + j * 2 * jnp.pi / NFP) % (2 * jnp.pi)
        sx = jnp.array([data["R"], data["phi"], data["Z"]]).T
        sx = rpz2xyz(sx)
        st = data["theta"]
        sz = data["zeta"]
        dS = (
            transforms["grid"].spacing[:, 1]
            / (2 * jnp.pi)
            * transforms["grid"].spacing[:, 2]
            / (2 * jnp.pi)
            / NFP
        )
        K = data["K_vc"] * data["|e_theta x e_zeta|"][:, None] * 4 * jnp.pi**2

        def body1(i, A):
            Areg1, Areg2 = A
            dx = jnp.linalg.norm(ex[i] - sx, axis=-1)
            dt = st - et[i]
            dz = sz - ez[i]
            # theta = 2pi * u -> dtheta = 2pi du
            tant = jnp.tan(dt / 2)  # original def as tan(pi*du) du in [0, 1]
            tanz = jnp.tan(dz / 2)  # original def as tan(pi*dv) dv in [0, 1]
            a = data["g_tt"][i] * 4 * jnp.pi**2
            b = data["g_tz"][i] * 4 * jnp.pi**2
            c = data["g_zz"][i] * 4 * jnp.pi**2
            regden = jnp.sqrt(a * tant**2 + 2 * b * tant * tanz + c * tanz**2)
            regnum = jnp.pi * data["K_vc"][i] * data["|e_theta x e_zeta|"][i]
            mask = (dx < 10 * jnp.finfo(dx.dtype).eps)[:, None]
            integrand1 = jnp.where(mask, 0, K / dx[:, None])
            integrand2 = jnp.where(mask, 0, regnum[None, :] / regden[:, None])

            A1i = mu_0 / (4 * jnp.pi) * jnp.sum(integrand1 * dS[:, None], axis=0)
            A2i = mu_0 / (4 * jnp.pi) * jnp.sum(integrand2 * dS[:, None], axis=0)
            Areg1 = put(Areg1, i, A1i.squeeze())
            Areg2 = put(Areg2, i, A2i.squeeze())
            return Areg1, Areg2

        A1 = jnp.zeros_like(Areg1)
        A2 = jnp.zeros_like(Areg2)
        A1, A2 = fori_loop(0, transforms["grid"].num_nodes, body1, (A1, A2))
        return Areg1 + A1, Areg2 + A2, data

    Asing = (
        mu_0
        / (4 * jnp.pi)
        * data["K_vc"]
        * (data["|e_theta x e_zeta|"] * data["I_vc"])[:, None]
    )
    Areg1 = jnp.zeros_like(Asing)
    Areg2 = jnp.zeros_like(Asing)
    Areg1, Areg2, _ = fori_loop(0, int(NFP), body2, (Areg1, Areg2, data))

    data["A_vc"] = Areg1 - Areg2 + Asing
    data["phi"] = src_phi
    data["zeta"] = src_zeta

    return data


@register_compute_fun(
    name="Bn_vc",
    label="(\\mathbf{B} \\cdot \\mathbf{n})_{VC}",
    units="T",
    units_long="Tesla",
    description="Virtual casing magnetic field normal to surface",
    dim=1,
    params=[],
    transforms={"A": [[0, 1, 0], [0, 0, 1]]},
    profiles=[],
    coordinates="rtz",
    data=["A_vc", "e_theta", "e_zeta", "|e_theta x e_zeta|"],
)
def _Bn_vc(params, transforms, profiles, data, **kwargs):
    At = dot(data["A_vc"], data["e_theta"])
    Az = dot(data["A_vc"], data["e_zeta"])
    Atmn = transforms["A"].fit(At)
    Azmn = transforms["A"].fit(Az)
    data["Bn_vc"] = (
        1
        / data["|e_theta x e_zeta|"]
        * (
            transforms["A"].transform(Azmn, dt=1)
            - transforms["A"].transform(Atmn, dz=1)
        )
    )
    return data
