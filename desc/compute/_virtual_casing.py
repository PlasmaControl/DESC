"""Compute functions for magnetic field quantities."""

from scipy.constants import mu_0

from desc.backend import jnp

from .data_index import register_compute_fun
from .utils import cross


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
