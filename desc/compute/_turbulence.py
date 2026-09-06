"""Compute functions for turbulent transport.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""

from ..backend import jnp, trapezoid
from ..compute.utils import cumtrapz
from ..integrals.critical_gradient import (
    extract_Kd_wells,
    extract_Kd_wells_and_peaks,
    fit_Kd_wells,
)
from .data_index import register_compute_fun

_doc = {
    "n_wells": (
        "int : Number of wells to detect for each pitch and field line. "
        "Default is 10 wells,"
    ),
    "curvature": (
        "str : good or bad curvature regions for the calculation of R_eff and L_par "
        "Default is bad curvature regions"
    ),
}


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
    # Exact definition of the dimenstionless drift curvature can be found
    # in https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.L032028
    data["Kd"] = data["a"] ** 2 * jnp.multiply(data["|B|"], data["cvdrift"])
    return data


@register_compute_fun(
    name="R_eff",
    label="R_eff",
    units="",
    units_long="",
    description="Effective radius of the drift curvature along the field line",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["Kd", "|e_zeta|r,a|"],
    **_doc,
)
def _R_eff(params, transforms, profiles, data, **kwargs):
    # Exact definition of the effective radius of curvature can be found
    # in https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.L032028
    grid = transforms["grid"].source_grid
    n_wells = kwargs.get("n_wells", 5)
    Kd_wells, _, masks = extract_Kd_wells(data["Kd"], n_wells=n_wells)
    l = cumtrapz(data["|e_zeta|r,a|"], x=grid.nodes[:, 2], initial=0)
    _, _, R_eff = fit_Kd_wells(l, Kd_wells, masks, n_wells=n_wells)
    data["R_eff"] = R_eff
    return data


@register_compute_fun(
    name="L_par",
    label="L_par",
    units="",
    units_long="",
    description="Width of Kd wells along the field line",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["Kd", "|e_zeta|r,a|"],
    **_doc,
)
def _L_par(params, transforms, profiles, data, **kwargs):
    # Parallel connection length defined as width of Kd wells
    grid = transforms["grid"].source_grid
    n_wells = kwargs.get("n_wells", 5)
    curvature = kwargs.get("curvature", "bad")
    _, masks, _ = extract_Kd_wells_and_peaks(data["Kd"], n_wells=n_wells)
    if curvature == "good":
        L_par = trapezoid(data["|e_zeta|r,a|"] * masks["masks_peaks"], grid.nodes[:, 2])
    else:
        L_par = trapezoid(data["|e_zeta|r,a|"] * masks["masks_wells"], grid.nodes[:, 2])
    data["L_par"] = L_par
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
    **_doc,
)
def _xi(params, transforms, profiles, data, **kwargs):
    # Parallel connection length defined as width of Kd wells
    mask = jnp.where(data["Kd"] < 0, 1.0, 0.0)
    xi = (data["a"] * mask * data["|grad(rho)|"]) ** 2
    data["xi"] = xi
    return data
