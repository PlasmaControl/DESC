"""Functions to create the O(rho) and O(rho^2) NAE constraints on a DESC equilibrium."""

import numpy as np

from desc.basis import FourierSeries
from desc.grid import LinearGrid
from desc.transform import Transform

from .linear_objectives import FixSumModesR, FixSumModesZ


def _calc_1st_order_NAE_coeffs(qsc, desc_eq, N=None):
    """Calculate 1st order NAE coefficients' toroidal Fourier representations.

    Uses the passed-in qsc object, and the desc_eq's stellarator symmetry is used.

    Parameters
    ----------
    qsc : Qsc equilibrium
        Qsc object to use as the NAE constraints on the DESC equilibrium.
    desc_eq : Equilibrium
        desc equilibrium to constrain.
    N : int,
        max toroidal resolution to constrain.
        If None, defaults to equilibrium's toroidal resolution

    Returns
    -------
    coeffs : dict
        dictionary of arrays with keys like 'X_L_M_n', where
        X is R or Z, L is 1 or 2, and M is 0,1, or 2, are the
        NAE Fourier (in toroidal angle phi) coeffs of
        radial order L and poloidal order M.
    bases : dict
        dictionary of Rbasis_cos, Rbasis_sin, Zbasis_cos, Zbasis_sin,
        the FourierSeries basis objects used to obtain the coefficients, where
        _cos or _sin denotes the symmetry of the (toroidal) Fourier series.
        symmetry is such that the R or Z coefficients is stellarator symmetric
        i.e. R_1_1_n uses the Rbasis_cos, since cos(theta)*cos(phi) is
        stellarator symmetric for R i.e. R(-theta,-phi) = R(theta,phi)
        and Z_1_1_n uses the Zbasis_sin as the term is cos(theta)*sin(phi)
        since Z(-theta,-phi) = - Z(theta,phi) for Z stellarator symmetry.
    """
    phi = qsc.phi

    R0 = qsc.R0_func(phi)
    dR0_dphi = qsc.R0p
    dZ0_dphi = qsc.Z0p
    if N is None:
        N = desc_eq.N
    else:
        N = np.min([desc_eq.N, N])
    assert N == int(N), "Toroidal Resolution must be an integer!"
    N = int(N)
    # normal and binormal vector components
    # Spline interpolants for the cylindrical components of the Frenet-Serret frame:
    # these are functions of phi (toroidal cylindrical angle)
    k_dot_R = qsc.normal_R_spline(phi)
    k_dot_phi = qsc.normal_phi_spline(phi)
    k_dot_Z = qsc.normal_z_spline(phi)
    tau_dot_R = qsc.binormal_R_spline(phi)
    tau_dot_phi = qsc.binormal_phi_spline(phi)
    tau_dot_Z = qsc.binormal_z_spline(phi)

    # use untwisted, which accounts for when NAE has QH symmetry,
    # and the poloidal angle is a helical angle.
    X1c = qsc.X1c_untwisted
    X1s = qsc.X1s_untwisted
    Y1c = qsc.Y1c_untwisted
    Y1s = qsc.Y1s_untwisted

    R_1_1 = X1c * (k_dot_R - k_dot_phi * dR0_dphi / R0) + Y1c * (
        tau_dot_R - tau_dot_phi * dR0_dphi / R0
    )
    R_1_neg1 = Y1s * (tau_dot_R - tau_dot_phi * dR0_dphi / R0) + X1s * (
        k_dot_R - k_dot_phi * dR0_dphi / R0
    )

    Z_1_1 = X1c * (k_dot_Z - k_dot_phi * dZ0_dphi / R0) + Y1c * (
        tau_dot_Z - tau_dot_phi * dZ0_dphi / R0
    )
    Z_1_neg1 = Y1s * (tau_dot_Z - tau_dot_phi * dZ0_dphi / R0) + X1s * (
        k_dot_Z - k_dot_phi * dZ0_dphi / R0
    )

    nfp = qsc.nfp
    if desc_eq.sym:
        Rbasis = FourierSeries(N=N, NFP=nfp, sym="cos")
        Zbasis = FourierSeries(N=N, NFP=nfp, sym="cos")
        Rbasis_sin = FourierSeries(N=N, NFP=nfp, sym="sin")
        Zbasis_sin = FourierSeries(N=N, NFP=nfp, sym="sin")
    else:
        Rbasis = FourierSeries(N=N, NFP=nfp, sym=False)
        Zbasis = FourierSeries(N=N, NFP=nfp, sym=False)
        Rbasis_sin = FourierSeries(N=N, NFP=nfp, sym=False)
        Zbasis_sin = FourierSeries(N=N, NFP=nfp, sym=False)

    grid = LinearGrid(M=0, L=0, zeta=phi, NFP=nfp)
    Rtrans = Transform(grid, Rbasis, build_pinv=True, method="auto")
    Ztrans = Transform(grid, Zbasis, build_pinv=True, method="auto")
    Rtrans_sin = Transform(grid, Rbasis_sin, build_pinv=True, method="auto")
    Ztrans_sin = Transform(grid, Zbasis_sin, build_pinv=True, method="auto")

    R_1_1_n = Rtrans.fit(R_1_1)
    R_1_neg1_n = Rtrans_sin.fit(R_1_neg1)

    Z_1_1_n = Ztrans_sin.fit(Z_1_1)
    Z_1_neg1_n = Ztrans.fit(Z_1_neg1)

    bases = {}
    bases["Rbasis_cos"] = Rbasis
    bases["Rbasis_sin"] = Rbasis_sin
    bases["Zbasis_cos"] = Zbasis
    bases["Zbasis_sin"] = Zbasis_sin

    coeffs = {}
    coeffs["R_1_1_n"] = R_1_1_n
    coeffs["R_1_neg1_n"] = R_1_neg1_n
    coeffs["Z_1_1_n"] = Z_1_1_n
    coeffs["Z_1_neg1_n"] = Z_1_neg1_n

    return coeffs, bases


def _make_RZ_cons_order_rho(qsc, desc_eq, coeffs, bases):
    """Create the linear constraints for constraining an eq with O(rho) NAE behavior.

    Parameters
    ----------
    qsc : Qsc equilibrium
        Qsc object to use as the NAE constraints on the DESC equilibrium.
    desc_eq : Equilibrium
        desc equilibrium to constrain.
    coeffs : dict
        dictionary of arrays with keys like 'X_L_M_n', where
        X is R or Z, L is 1 , and M is 1, are the
        NAE Fourier (in toroidal angle phi) coeffs of
        radial order L and poloidal order M.
    bases : dict
        dictionary of Rbasis_cos, Rbasis_sin, Zbasis_cos, Zbasis_sin,
        the FourierSeries basis objects used to obtain the coefficients, where
        _cos or _sin denotes the symmetry of the (toroidal) Fourier series.
        symmetry is such that the R or Z coefficients is stellarator symmetric
        i.e. R_1_1_n uses the Rbasis_cos, since cos(theta)*cos(phi) is
        stellarator symmetric for R i.e. R(-theta,-phi) = R(theta,phi)
        and Z_1_1_n uses the Zbasis_sin as the term is cos(theta)*sin(phi)
        since Z(-theta,-phi) = - Z(theta,phi) for Z stellarator symmetry.

    Returns
    -------
    Rconstraints : tuple of Objective
        tuple of constraints of type FixSumModesR, which enforce
        the O(rho) behavior of the equilibrium R coefficients to match the NAE.
    Zconstraints : tuple of Objective
        tuple of constraints of type FixSumModesZ, which enforce
        the O(rho) behavior of the equilibrium Z coefficients to match the NAE.
    """
    # r is the ratio  r_NAE / rho_DESC
    r = np.sqrt(2 * abs(desc_eq.Psi / qsc.Bbar) / 2 / np.pi)

    Rconstraints = ()
    Zconstraints = ()
    Rbasis_cos = bases["Rbasis_cos"]
    Zbasis_cos = bases["Zbasis_cos"]
    Rbasis_sin = bases["Rbasis_sin"]
    Zbasis_sin = bases["Zbasis_sin"]

    # R_1_1_n
    for n, NAEcoeff in zip(Rbasis_cos.modes[:, 2], coeffs["R_1_1_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, 1, n])
            sum_weights.append([(-1) ** k * k])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Rcon = FixSumModesR(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Rconstraints += (Rcon,)
    # Z_1_neg1_n
    for n, NAEcoeff in zip(Zbasis_cos.modes[:, 2], coeffs["Z_1_neg1_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, -1, n])
            sum_weights.append([(-1) ** k * k])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Zcon = FixSumModesZ(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Zconstraints += (Zcon,)

    # R_1_neg1_n
    for n, NAEcoeff in zip(Rbasis_sin.modes[:, 2], coeffs["R_1_neg1_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, -1, n])
            sum_weights.append([(-1) ** k * k])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Rcon = FixSumModesR(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Rconstraints += (Rcon,)
    # Z_1_1_n
    for n, NAEcoeff in zip(Zbasis_sin.modes[:, 2], coeffs["Z_1_1_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, 1, n])
            sum_weights.append([(-1) ** k * k])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Zcon = FixSumModesZ(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Zconstraints += (Zcon,)

    return Rconstraints, Zconstraints


def make_RZ_cons_1st_order(qsc, desc_eq, N=None):
    """Make the first order NAE constraints for a DESC equilibrium.

    Parameters
    ----------
    qsc : Qsc equilibrium
        Qsc object to use as the NAE constraints on the DESC equilibrium.
    desc_eq : Equilibrium
        desc equilibrium to constrain.
    N : int,
        max toroidal resolution to constrain.
        If None, defaults to equilibrium's toroidal resolution

    Returns
    -------
    Rconstraints : tuple of Objective
        tuple of FixSumModesR constraints corresponding to constraining the O(rho)
        DESC coefficients, to be used in constraining a DESC equilibrium solve.
    Zconstraints : tuple of Objective
        tuple of FixSumModesZ constraints corresponding to constraining the O(rho)
        DESC coefficients, to be used in constraining a DESC equilibrium solve.
    """
    Rconstraints = ()
    Zconstraints = ()

    coeffs, bases = _calc_1st_order_NAE_coeffs(qsc, desc_eq, N=N)
    Rconstraints, Zconstraints = _make_RZ_cons_order_rho(qsc, desc_eq, coeffs, bases)

    return Rconstraints + Zconstraints
