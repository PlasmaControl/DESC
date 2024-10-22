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
from ..integrals.critical_gradient import extract_Kd_wells, fit_Kd_wells


@register_compute_fun(
    name="Kd",
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

def _Kd(params, transforms, profiles, data, **kwargs):
    data["Kd"] = (
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
    data=["Kd"],
)

def _R_eff(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"].source_grid
    Kd_wells,_,masks = extract_Kd_wells(data["Kd"])
    _,_,R_eff = fit_Kd_wells(grid.nodes[:,2], Kd_wells, masks)
    data["R_eff"] = R_eff
    return data

@register_compute_fun(
    name="L_par",
    # Parallel connection length defined as width of Kd wells
    label="L_par",
    units="",
    units_long="",
    description="Width of Kd wells along the field line",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="rtz",
    data=["Kd"],
)

def _L_par(params, transforms, profiles, data, **kwargs):
    grid = transforms["grid"].source_grid
    _,length_wells,_ = extract_Kd_wells(data["Kd"],order=True)
    L_par = jnp.diff(grid.nodes[:,2])[0]*length_wells
    data["L_par"] = L_par
    return data
