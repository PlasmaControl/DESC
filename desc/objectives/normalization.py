"""Utility function for computing scaling factors for non-dimensionalization."""

import numpy as np
from scipy.constants import elementary_charge, mu_0


def compute_scaling_factors(thing):
    """Compute dimensional quantities for normalizations."""
    # local import to avoid circular import
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface

    scales = {}

    if isinstance(thing, Equilibrium):
        R10 = thing.Rb_lmn[thing.surface.R_basis.get_idx(M=1, N=0)]
        Z10 = thing.Zb_lmn[thing.surface.Z_basis.get_idx(M=-1, N=0)]
        R00 = thing.Rb_lmn[thing.surface.R_basis.get_idx(M=0, N=0)]

        scales["R0"] = R00
        scales["a"] = np.sqrt(np.abs(R10 * Z10))
        # TODO: also implement for FourierRZToroidalSurface,
        # need to add "A" compute fxn for that FourierRZToroidalSurface
        if np.isclose(scales["a"], 0):
            # R10 and Z10 can be 0 for stellarator asymmetric equilibria,
            # just use the computed minor radius in this case
            scales["a"] = thing.compute("a")["a"]
        scales["A"] = np.pi * scales["a"] ** 2
        scales["V"] = 2 * np.pi * scales["R0"] * scales["A"]
        scales["B_T"] = abs(thing.Psi) / scales["A"]
        iota_avg = np.mean(np.abs(thing.get_profile("iota")(np.linspace(0, 1, 20))))
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
        scales["Psi"] = abs(thing.Psi)
        scales["n"] = 1e19
        scales["T"] = scales["p"] / (scales["n"] * elementary_charge)
    elif isinstance(thing, FourierRZToroidalSurface):
        R10 = thing.R_lmn[thing.R_basis.get_idx(M=1, N=0)]
        Z10 = thing.Z_lmn[thing.Z_basis.get_idx(M=-1, N=0)]
        R00 = thing.R_lmn[thing.R_basis.get_idx(M=0, N=0)]

        scales["R0"] = R00
        scales["a"] = np.sqrt(np.abs(R10 * Z10))
        if np.isclose(scales["a"], 0):
            # R10 and Z10 can be 0 for stellarator asymmetric equilibria,
            # use the other coefficients in this case
            R10 = thing.R_lmn[thing.R_basis.get_idx(M=-1, N=0)]
            Z10 = thing.Z_lmn[thing.Z_basis.get_idx(M=1, N=0)]

        scales["A"] = np.pi * scales["a"] ** 2
        scales["V"] = 2 * np.pi * scales["R0"] * scales["A"]

    # replace 0 scales to avoid normalizing by zero
    for scale in scales.keys():
        if np.isclose(scales[scale], 0):
            scales[scale] = 1

    return scales
