#!/usr/bin/env python3

from .data_index import register_compute_fun


@register_compute_fun(
    name="UC",
    label="umbilic curve",
    units="~",
    units_long="None",
    description="angular difference between theta and phi on a surface"
    + "determines the shape of an umbilic curve",
    dim=1,
    params=["UC_n"],
    transforms={"UC": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="",
    data=[],
    parameterization="desc.geometry.umbiliccurve.FourierUmbilicCurve",
)
def _UC_FourierUmbilicCurve(params, transforms, profiles, data, **kwargs):
    # If the grid is non-uniform, the transform has to be modified
    data["UC"] = transforms["UC"].transform(params["UC_n"], dz=0)
    return data


@register_compute_fun(
    name="phi",
    label="phi",
    units="~",
    units_long="Radians",
    description="Toroidal phi position along curve",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="phi",
    data=[],
    parameterization="desc.geometry.umbiliccurve.FourierUmbilicCurve",
)
def _phi(params, transforms, profiles, data, **kwargs):
    data["phi"] = transforms["grid"].nodes[:]
    return data
