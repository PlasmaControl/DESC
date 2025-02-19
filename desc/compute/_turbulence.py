"""Compute functions for turbulent transport."""

from ..backend import jnp
from .data_index import register_compute_fun


@register_compute_fun(
    name="Kd",
    label="\\mathrm{cvdrift} = a^2\\nabla\\alpha\\cdot\\mathbf{b}\\times\\kappa",
    units="",
    units_long="",
    description="Dimensionless drift curvature",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="rtz",
    data=["cvdrift", "|B|", "a"],
)
def _Kd(params, transforms, profiles, data, **kwargs):
    # Exact definition of the dimensionless drift curvature can be found
    # in https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.L032028
    data["Kd"] = data["a"] ** 2 * jnp.multiply(data["|B|"], data["cvdrift"])
    return data


@register_compute_fun(
    name="xi",
    label="ξ",
    units="",
    units_long="",
    description="Target for spacing of flux surfaces",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["a", "|grad(rho)|", "Kd"],
)
def _xi(params, transforms, profiles, data, **kwargs):
    # square of gradrho in regions of bad curvature. Definition can be
    # found in https://arxiv.org/html/2405.19860v1
    mask = jnp.where(data["Kd"] < 0, 1.0, 0.0)
    xi = (data["a"] * mask * data["|grad(rho)|"]) ** 2
    data["xi"] = xi
    return data
