"""Utility function for computing scaling factors for non-dimensionalization."""

import numpy as np
from scipy.constants import elementary_charge, mu_0

from desc.geometry import Curve


def compute_scaling_factors(thing):
    """Compute dimensional quantities for normalizations."""
    # local import to avoid circular import
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface

    scales = {}

    def get_max_mode(basis, coeffs):
        """Return max boundary mode that is not M=0 N=0."""
        inds_not_00 = np.where(
            np.logical_or(np.abs(basis.modes[:, 1]) > 0, np.abs(basis.modes[:, 2]) > 0)
        )[0]

        val = np.max(np.abs(coeffs[inds_not_00]))
        if not np.isclose(val, 0, atol=1e-12):
            return val
        raise ValueError("No modes found, geometry is unphysical.")

    if isinstance(thing, Equilibrium):
        R00 = thing.Rb_lmn[thing.surface.R_basis.get_idx(M=0, N=0)]
        R10 = get_max_mode(thing.surface.R_basis, thing.surface.R_lmn)
        Z10 = get_max_mode(thing.surface.Z_basis, thing.surface.Z_lmn)

        scales["R0"] = R00
        scales["a"] = np.sqrt(np.abs(R10 * Z10))
        scales["Psi"] = abs(thing.Psi)
        scales["A"] = np.pi * scales["a"] ** 2
        scales["V"] = 2 * np.pi * scales["R0"] * scales["A"]
        scales["B"] = (
            scales["Psi"] / scales["A"] * 1.25
        )  # 1.25 is just a magic number that seems to help scaling
        B_pressure = scales["B"] ** 2 / (2 * mu_0)
        scales["I"] = scales["B"] * scales["a"] * 2 * np.pi / mu_0
        scales["W"] = B_pressure * scales["V"]
        scales["J"] = scales["B"] / scales["a"] / mu_0
        scales["F"] = B_pressure / scales["a"]
        scales["f"] = scales["F"] * scales["V"]

        if thing.pressure is not None:
            p0 = float(thing.pressure(0)[0])
        else:
            scales["n"] = float(
                ((thing.atomic_number(0) + 1) / 2 * thing.electron_density(0))[0]
            )
            scales["T"] = np.mean(
                [thing.electron_temperature(0), thing.ion_temperature(0)]
            )
            p0 = elementary_charge * 2 * scales["n"] * scales["T"]
        if p0 < 1:  # vacuum
            scales["p"] = B_pressure
        else:
            scales["p"] = p0
        scales["W_p"] = scales["p"] * scales["V"] / 2

    elif isinstance(thing, FourierRZToroidalSurface):
        R00 = thing.R_lmn[thing.R_basis.get_idx(M=0, N=0)]
        R10 = get_max_mode(thing.R_basis, thing.R_lmn)
        Z10 = get_max_mode(thing.Z_basis, thing.Z_lmn)

        scales["R0"] = R00
        scales["a"] = np.sqrt(np.abs(R10 * Z10))
        scales["A"] = np.pi * scales["a"] ** 2
        scales["V"] = 2 * np.pi * scales["R0"] * scales["A"]

    elif isinstance(thing, Curve):
        scales["a"] = thing.compute("length")["length"] / (2 * np.pi)

    # replace 0 scales to avoid normalizing by zero
    for scale in scales.keys():
        if np.isclose(scales[scale], 0):
            scales[scale] = 1

    return scales
