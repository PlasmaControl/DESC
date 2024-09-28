#!/usr/bin/env python3

from .data_index import register_compute_fun


@register_compute_fun(
    name="A",
    label="umbilic curve",
    units="~",
    units_long="None",
    description="angular difference between theta and phi on a surface"
    + "determines the shape of an umbilic curve",
    dim=1,
    params=["A_n"],
    transforms={"A": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="",
    data=[],
    parameterization="desc.geometry.umbiliccurve.FourierUmbilicCurve",
)
def _A_FourierUmbilicCurve(params, transforms, profiles, data, **kwargs):
    data["A"] = transforms["A"].transform(params["A_n"], dz=0)
    return data
