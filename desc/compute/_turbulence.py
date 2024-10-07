"""Compute functions for turbulent transport.

Notes
-----
Some quantities require additional work to compute at the magnetic axis.
A Python lambda function is used to lazily compute the magnetic axis limits
of these quantities. These lambda functions are evaluated only when the
computational grid has a node on the magnetic axis to avoid potentially
expensive computations.
"""
from .data_index import register_compute_fun
from ..backend import jnp
from ..integrals.critical_gradient import fit_drift_peaks


@register_compute_fun(
    name="K_d",
    # Exact definition of the dimenstionless drift curvature can be found
    # in https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.L032028
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

def _K_d(params, transforms, profiles, data, **kwargs):
    data["K_d"] = (
        data["a"]**2*jnp.multiply(data["|B|"],data["cvdrift"])
    )
    return data

@register_compute_fun(
    name="R_eff",
    # Exact definition of the effective radius of curvature can be found
    # in https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.L032028
    label="R_eff",
    units="",
    units_long="",
    description="Effective radius of the drift curvature along the field line",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["K_d"],
)

def _R_eff(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"].source_grid
    out = fit_drift_peaks(grid.nodes[:,2],data["K_d"])
    R_eff = [row[0] for row in out["values"]]
    data["R_eff"] = R_eff
    return data


