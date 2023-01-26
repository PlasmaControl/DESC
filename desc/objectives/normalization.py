"""Utility function for computing scaling factors for non-dimensionalization."""

import numpy as np
from scipy.constants import elementary_charge, mu_0


def compute_scaling_factors(eq):
    """Compute dimensional quantities for normalizations."""
    scales = {}
    R10 = eq.Rb_lmn[eq.surface.R_basis.get_idx(M=1, N=0)]
    Z10 = eq.Zb_lmn[eq.surface.Z_basis.get_idx(M=-1, N=0)]
    R00 = eq.Rb_lmn[eq.surface.R_basis.get_idx(M=0, N=0)]

    scales["R0"] = R00
    scales["a"] = np.sqrt(np.abs(R10 * Z10))
    scales["A"] = np.pi * scales["a"] ** 2
    scales["V"] = 2 * np.pi * scales["R0"] * scales["A"]
    scales["B_T"] = abs(eq.Psi) / scales["A"]
    iota_avg = np.mean(np.abs(eq.get_profile("iota")(np.linspace(0, 1, 20))))
    if np.isclose(iota_avg, 0):
        scales["B_P"] = scales["B_T"]
    else:
        scales["B_P"] = scales["B_T"] * iota_avg
    scales["B"] = np.sqrt(scales["B_T"] ** 2 + scales["B_P"] ** 2)
    scales["I"] = scales["B_P"] * 2 * np.pi / mu_0
    scales["p"] = scales["B"] ** 2 / (2 * mu_0)
    scales["W"] = scales["p"] * scales["V"]
    scales["J"] = scales["B"] / scales["a"] / mu_0
    scales["F"] = scales["p"] / scales["a"]
    scales["f"] = scales["F"] * scales["V"]
    scales["Psi"] = abs(eq.Psi)
    scales["n"] = 1e19
    scales["T"] = scales["p"] / (scales["n"] * elementary_charge)
    # replace 0 scales to avoid normalizing by zero
    for scale in scales.keys():
        if np.isclose(scales[scale], 0):
            scales[scale] = 1

    return scales
