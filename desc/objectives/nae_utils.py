"""Functions to create the O(rho) and O(rho^2) NAE constraints on a DESC equilibrium."""

import numpy as np

from desc.basis import FourierSeries
from desc.grid import LinearGrid
from desc.transform import Transform

from .linear_objectives import FixSumModesLambda, FixSumModesR, FixSumModesZ


def calc_zeroth_order_lambda(qsc, desc_eq, N=None):
    """Calculate 0th order NAE lambda constraint.

    Uses the passed-in qsc object, and the desc_eq's stellarator symmetry is used.

    Parameters
    ----------
    qsc : Qsc or None
        Qsc object to use as the NAE constraints on the DESC equilibrium.
        if None is passed, will instead use the given DESC equilibrium's
        current near-axis behavior for the constraint.
    desc_eq : Equilibrium
        desc equilibrium to constrain.
    N : int,
        max toroidal resolution to constrain.
        If None, defaults to equilibrium's toroidal resolution

    Returns
    -------
    Lconstraints : tuple of Objective
        tuple of constraints of type FixSumModesLambda, which enforce
        the axis (O(rho^0)) behavior of the equilibrium lambda coefficents
        to match the NAE.
    L_0_n : array
        array of floats of the toroidal Fourier coefficients of the
        lambda on-axis behavior dictated by the NAE
    Lbasis : FourierSeries Basis
        FourierSeries basis corresponding to L_0_n coefficients

    """
    nfp = desc_eq.NFP
    if N is None:
        N = desc_eq.N
    else:
        N = np.min([desc_eq.N, N])
    if desc_eq.sym:
        Lbasis_sin = FourierSeries(N=N, NFP=nfp, sym="sin")
    else:
        Lbasis_sin = FourierSeries(N=N, NFP=nfp, sym=False)
    if qsc is not None:
        phi = qsc.phi
        dphi = phi[1] - phi[0]

        grid = LinearGrid(M=0, L=0, zeta=phi, NFP=nfp)
        Ltrans_sin = Transform(grid, Lbasis_sin, build_pinv=True, method="auto")

        # from integrating eqn A20 in
        # Constructing stellarators with quasisymmetry to high order 2019
        #  Landreman and Sengupta
        nu_0 = np.cumsum(qsc.B0 / qsc.G0 * qsc.d_l_d_phi - 1) * np.ones_like(phi) * dphi
        nu_0_n = Ltrans_sin.fit(nu_0)

        # lambda = -iota_0 * nu
        L_0_n = -qsc.iota * nu_0_n
    else:
        # we will fix to eq current on-axis lambda, so just set L_0_n to 0
        # as it will be unused
        L_0_n = np.zeros(Lbasis_sin.num_modes)

    Lconstraints = ()
    for n, NAEcoeff in zip(Lbasis_sin.modes[:, 2], L_0_n):
        sum_weights = []
        modes = []
        target = None if qsc is None else NAEcoeff
        for l in range(int(desc_eq.L + 1)):
            modes.append([l, 0, n])
            if (l // 2) % 2 == 0:
                sum_weights.append(1)
            else:
                sum_weights.append(-1)

        modes = np.atleast_2d(modes)
        Lcon = FixSumModesLambda(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )

        Lconstraints += (Lcon,)

    return Lconstraints, L_0_n, Lbasis_sin


def _calc_1st_order_NAE_coeffs(qsc, desc_eq, N=None):
    """Calculate 1st order NAE coefficients' toroidal Fourier representations.

    Uses the passed-in qsc object, and the desc_eq's stellarator symmetry is used.

    Parameters
    ----------
    qsc : Qsc or None
        Qsc object to use as the NAE constraints on the DESC equilibrium.
        if None is passed, will instead use the given DESC equilibrium's
        current near-axis behavior for the constraint.
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
    if qsc is None:
        nfp = desc_eq.NFP
        # we will set behavior to the eq's current near axis behavior
        # we dont need to calculate any NAE coeffs, we only need the
        # bases and arrays of appropriate sizes for the coefficients
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
        bases = {}
        bases["Rbasis_cos"] = Rbasis
        bases["Rbasis_sin"] = Rbasis_sin
        bases["Zbasis_cos"] = Zbasis
        bases["Zbasis_sin"] = Zbasis_sin
        bases["Lbasis_cos"] = Zbasis  # L has same basis as Z bc has the same symmetry
        bases["Lbasis_sin"] = Zbasis_sin

        coeffs = {}
        coeffs["R_1_1_n"] = np.zeros(Rbasis.num_modes)
        coeffs["R_1_neg1_n"] = np.zeros(Rbasis_sin.num_modes)
        coeffs["Z_1_1_n"] = np.zeros(Zbasis_sin.num_modes)
        coeffs["Z_1_neg1_n"] = np.zeros(Zbasis.num_modes)
        coeffs["L_1_1_n"] = np.zeros(Zbasis_sin.num_modes)
        coeffs["L_1_neg1_n"] = np.zeros(Zbasis.num_modes)
        return coeffs, bases

    phi = qsc.phi
    dphi = phi[1] - phi[0]

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

    # these are like X_l_m
    # where l is radial order (in rho)
    # and m is the poloidal modenumber (+m for cos(theta), -m for sin(theta))

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

    # from integrating eqn A20 in
    # Constructing stellarators with quasisymmetry to high order 2019
    #  Landreman and Sengupta
    # take derivative of that form d(nu0)_dphi
    nu_0 = np.cumsum(qsc.B0 / qsc.G0 * qsc.d_l_d_phi - 1) * np.ones_like(phi) * dphi
    nu0p = np.diff(np.append(nu_0, nu_0[0])) / dphi
    L_1_1 = qsc.iota * (X1c * (k_dot_phi) + Y1c * (tau_dot_phi)) / R0 * (nu0p + 1)
    L_1_neg1 = qsc.iota * (X1s * (k_dot_phi) + Y1s * (tau_dot_phi)) / R0 * (nu0p + 1)

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

    L_1_1_n = Ztrans_sin.fit(L_1_1)
    L_1_neg1_n = Ztrans.fit(L_1_neg1)

    bases = {}
    bases["Rbasis_cos"] = Rbasis
    bases["Rbasis_sin"] = Rbasis_sin
    bases["Zbasis_cos"] = Zbasis
    bases["Zbasis_sin"] = Zbasis_sin
    bases["Lbasis_cos"] = Zbasis  # L has same basis as Z bc has the same symmetry
    bases["Lbasis_sin"] = Zbasis_sin

    coeffs = {}
    coeffs["R_1_1_n"] = R_1_1_n
    coeffs["R_1_neg1_n"] = R_1_neg1_n
    coeffs["Z_1_1_n"] = Z_1_1_n
    coeffs["Z_1_neg1_n"] = Z_1_neg1_n
    coeffs["L_1_1_n"] = L_1_1_n
    coeffs["L_1_neg1_n"] = L_1_neg1_n

    return coeffs, bases


def _make_RZ_cons_order_rho(  # noqa: C901
    qsc, desc_eq, coeffs, bases, fix_lambda=False
):
    """Create the linear constraints for constraining an eq with O(rho) NAE behavior.

    Parameters
    ----------
    qsc : Qsc or None
        Qsc object to use as the NAE constraints on the DESC equilibrium.
        if None is passed, will instead use the given DESC equilibrium's
        current near-axis behavior for the constraint.
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
    fix_lambda : bool, default False
        whether to include first order constraints to fix the O(rho) behavior
        of lambda. Defaults to False.

    Returns
    -------
    Rconstraints : tuple of Objective
        tuple of constraints of type FixSumModesR, which enforce
        the O(rho) behavior of the equilibrium R coefficients to match the NAE.
    Zconstraints : tuple of Objective
        tuple of constraints of type FixSumModesZ, which enforce
        the O(rho) behavior of the equilibrium Z coefficients to match the NAE.
    Lconstraints : tuple of Objective
        tuple of constraints of type FixSumModesLambda, which enforce
        the O(rho) behavior of the equilibrium lambda coefficients to match the NAE.
        Tuple is empty if fix_lambda=False.

    """
    if qsc is not None:
        # r is the ratio  r_NAE / rho_DESC
        r = np.sqrt(2 * abs(desc_eq.Psi / qsc.Bbar) / 2 / np.pi)
    else:
        r = 1  # using DESC equilibrium's behavior, no conversion is needed

    Rconstraints = ()
    Zconstraints = ()
    Lconstraints = ()

    Rbasis_cos = bases["Rbasis_cos"]
    Zbasis_cos = bases["Zbasis_cos"]
    Lbasis_cos = bases["Lbasis_cos"]

    Rbasis_sin = bases["Rbasis_sin"]
    Zbasis_sin = bases["Zbasis_sin"]
    Lbasis_sin = bases["Lbasis_sin"]

    # R_1_1_n
    for n, NAEcoeff in zip(Rbasis_cos.modes[:, 2], coeffs["R_1_1_n"]):
        sum_weights = []
        modes = []
        target = None if qsc is None else NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, 1, n])
            sum_weights.append((-1) ** k * k)
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
        target = None if qsc is None else NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, -1, n])
            sum_weights.append((-1) ** k * k)
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Zcon = FixSumModesZ(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )

        Zconstraints += (Zcon,)
    if fix_lambda:
        # L_1_neg1_n
        for n, NAEcoeff in zip(Lbasis_cos.modes[:, 2], coeffs["L_1_neg1_n"]):
            sum_weights = []
            modes = []
            target = None if qsc is None else NAEcoeff * r
            for k in range(1, int((desc_eq.L + 1) / 2) + 1):
                modes.append([2 * k - 1, -1, n])
                sum_weights.append((-1) ** k * k)
            modes = np.atleast_2d(modes)
            sum_weights = -np.atleast_1d(sum_weights)
            Lcon = FixSumModesLambda(
                eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
            )
            Lconstraints += (Lcon,)
    # R_1_neg1_n
    for n, NAEcoeff in zip(Rbasis_sin.modes[:, 2], coeffs["R_1_neg1_n"]):
        sum_weights = []
        modes = []
        target = None if qsc is None else NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, -1, n])
            sum_weights.append((-1) ** k * k)
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
        target = None if qsc is None else NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, 1, n])
            sum_weights.append((-1) ** k * k)
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Zcon = FixSumModesZ(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )

        Zconstraints += (Zcon,)
    if fix_lambda:
        # L_1_1_n
        for n, NAEcoeff in zip(Lbasis_sin.modes[:, 2], coeffs["L_1_1_n"]):
            sum_weights = []
            modes = []
            target = None if qsc is None else NAEcoeff * r
            for k in range(1, int((desc_eq.L + 1) / 2) + 1):
                modes.append([2 * k - 1, 1, n])
                sum_weights.append((-1) ** k * k)
            modes = np.atleast_2d(modes)
            sum_weights = -np.atleast_1d(sum_weights)
            Lcon = FixSumModesLambda(
                eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
            )
            Lconstraints += (Lcon,)
    return Rconstraints, Zconstraints, Lconstraints


def make_RZ_cons_1st_order(qsc, desc_eq, fix_lambda=False, N=None):
    """Make the first order NAE constraints for a DESC equilibrium.

    Parameters
    ----------
    qsc : Qsc or None
        Qsc object to use as the NAE constraints on the DESC equilibrium.
        if None is passed, will instead use the given DESC equilibrium's
        current near-axis behavior for the constraint.
    desc_eq : Equilibrium
        desc equilibrium to constrain.
    fix_lambda : bool, default False
        whether to include first order constraints to fix the O(rho) behavior
        of lambda. Defaults to False.
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
    Lconstraints = ()

    coeffs, bases = _calc_1st_order_NAE_coeffs(qsc, desc_eq, N=N)
    Rconstraints, Zconstraints, Lconstraints = _make_RZ_cons_order_rho(
        qsc, desc_eq, coeffs, bases, fix_lambda
    )

    return Rconstraints + Zconstraints + Lconstraints


""" Order (rho^2)"""


def _calc_2nd_order_NAE_coeffs(qsc, desc_eq, N=None):
    """Calculate 2nd order NAE coefficients' Fourier representations.

    uses the passed-in qsc object, and the desc_eq's stellarator symmetry is used.

    Parameters
    ----------
    qsc :  Qsc
        Qsc object to use as the NAE constraints on the DESC equilibrium
    desc_eq : Equilibrium
        DESC equilibrium to constrain
    N : int
        max toroidal resolution to constrain.
        If None, defaults to equilibrium's toroidal resolution

    Returns
    -------
    coeffs: dict
        dictionary of arrays with keys like 'X_L_M_n', where
        X is R or Z, L is 1 or 2, and M is 0,1, or 2, are the
        NAE Fourier (in tor. phi) coeffs of radial order L and poloidal order M
    bases: dict
        dictionary of Rbasis_cos, Rbasis_sin, Zbasis_cos, Zbasis_sin,
        the FourierSeries basis objects used to obtain the coefficients, where
        _cos or _sin denotes the symmetry of the Fourier series.
        symmetry is such that the R or Z coefficients is stellarator symmetric
        i.e. R_1_1_n uses the Rbasis_cos, since cos(theta)*cos(phi) is
        stellarator symmetric for R i.e. R(-theta,-phi) = R(theta,phi)
        and Z_1_1_n uses the Zbasis_sin as the term is cos(theta)*sin(phi)
        since Z(-theta,-phi) = - Z(theta,phi) for Z stellarator symmetry
    """
    if qsc is None:
        nfp = desc_eq.NFP
        # we will set behavior to the eq's current near axis behavior
        # we dont need to calculate any NAE coeffs, we only need the
        # bases and arrays of appropriate sizes for the coefficients
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
        bases = {}
        bases["Rbasis_cos"] = Rbasis
        bases["Rbasis_sin"] = Rbasis_sin
        bases["Zbasis_cos"] = Zbasis
        bases["Zbasis_sin"] = Zbasis_sin

        coeffs = {}
        coeffs["R_2_0_n"] = np.zeros(Rbasis.num_modes)
        coeffs["R_2_2_n"] = np.zeros(Rbasis.num_modes)
        coeffs["R_2_neg2_n"] = np.zeros(Rbasis_sin.num_modes)

        coeffs["Z_2_0_n"] = np.zeros(Zbasis_sin.num_modes)
        coeffs["Z_2_2_n"] = np.zeros(Zbasis_sin.num_modes)
        coeffs["Z_2_neg2_n"] = np.zeros(Zbasis.num_modes)
        return coeffs, bases
    # get variables from qsc

    if N is None:
        N = desc_eq.N
    else:
        N = np.max([desc_eq.N, N])
    assert N == int(N), "Toroidal Resolution must be an integer!"
    N = int(N)

    phi = qsc.phi
    R0 = qsc.R0_func(phi)

    # 2nd order terms

    stel = qsc
    R0 = stel.R0
    dR0 = stel.R0p
    ddR0 = stel.R0pp
    dZ0 = stel.Z0p
    ddZ0 = stel.Z0pp

    x1_cos = (
        stel.X1c_untwisted * stel.normal_cylindrical.transpose()
        + stel.Y1c_untwisted * stel.binormal_cylindrical.transpose()
    )
    x1_sin = (
        stel.Y1s_untwisted * stel.binormal_cylindrical.transpose()
        + stel.X1s_untwisted * stel.normal_cylindrical.transpose()
    )

    x2_0 = (
        stel.X20_untwisted * stel.normal_cylindrical.transpose()
        + stel.Y20_untwisted * stel.binormal_cylindrical.transpose()
        + stel.Z20_untwisted * stel.tangent_cylindrical.transpose()
    )
    x2_cos = (
        stel.X2c_untwisted * stel.normal_cylindrical.transpose()
        + stel.Y2c_untwisted * stel.binormal_cylindrical.transpose()
        + stel.Z2c_untwisted * stel.tangent_cylindrical.transpose()
    )
    x2_sin = (
        stel.X2s_untwisted * stel.normal_cylindrical.transpose()
        + stel.Y2s_untwisted * stel.binormal_cylindrical.transpose()
        + stel.Z2s_untwisted * stel.tangent_cylindrical.transpose()
    )

    X1Rc = x1_cos[0, :]
    dX1Rc = np.matmul(stel.d_d_phi, X1Rc)
    X1Rs = x1_sin[0, :]
    dX1Rs = np.matmul(stel.d_d_phi, X1Rs)
    X1phic = x1_cos[1, :]
    dX1phic = np.matmul(stel.d_d_phi, X1phic)
    X1phis = x1_sin[1, :]
    dX1phis = np.matmul(stel.d_d_phi, X1phis)
    X1zc = x1_cos[2, :]
    dX1zc = np.matmul(stel.d_d_phi, X1zc)
    X1zs = x1_sin[2, :]
    dX1zs = np.matmul(stel.d_d_phi, X1zs)
    X2R0 = x2_0[0, :]
    X2phi0 = x2_0[1, :]
    X2z0 = x2_0[2, :]
    X2Rs = x2_sin[0, :]
    X2phis = x2_sin[1, :]
    X2zs = x2_sin[2, :]
    X2Rc = x2_cos[0, :]
    X2phic = x2_cos[1, :]
    X2zc = x2_cos[2, :]

    coeffs = {}

    R_2_0 = (
        (X1phic**2 + X1phis**2) / 4 / R0
        + X2R0
        + (X1Rc * X1phic * dR0) / 2 / R0**2
        + (X1Rs * X1phis * dR0) / 2 / R0**2
        - X2phi0 * dR0 / R0
        - X1phic**2 * dR0**2 / 2 / R0**3
        - X1phis**2 * dR0**2 / 2 / R0**3
        - X1phic * dX1Rc / 2 / R0
        - X1phis * dX1Rs / 2 / R0
        + X1phic * dR0 * dX1phic / 2 / R0**2
        + X1phis * dR0 * dX1phis / 2 / R0**2
        + X1phic**2 * ddR0 / 4 / R0**2
        + X1phis**2 * ddR0 / 4 / R0**2
    )

    R_2_neg2 = (
        X1phic * X1phis / 2 / R0
        + X2Rs
        + X1Rs * X1phic * dR0 / 2 / R0**2
        + X1Rc * X1phis * dR0 / 2 / R0**2
        - X2phis * dR0 / R0
        - X1phic * X1phis * dR0**2 / R0**3
        - X1phis * dX1Rc / 2 / R0
        - X1phic * dX1Rs / 2 / R0
        + X1phis * dR0 * dX1phic / 2 / R0**2
        + X1phic * dR0 * dX1phis / 2 / R0**2
        + X1phic * X1phis * ddR0 / 2 / R0**2
    )

    R_2_2 = (
        (X1phic**2 - X1phis**2) / 4 / R0
        + X2Rc
        + X1Rc * X1phic * dR0 / 2 / R0**2
        - X1Rs * X1phis * dR0 / 2 / R0**2
        - X2phic * dR0 / R0
        - X1phic**2 * dR0**2 / 2 / R0**3
        + X1phis**2 * dR0**2 / 2 / R0**3
        - X1phic * dX1Rc / 2 / R0
        + X1phis * dX1Rs / 2 / R0
        + X1phic * dR0 * dX1phic / 2 / R0**2
        - X1phis * dR0 * dX1phis / 2 / R0**2
        + X1phic**2 * ddR0 / 4 / R0**2
        - X1phis**2 * ddR0 / 4 / R0**2
    )

    Z_2_0 = (
        X2z0
        - (X1phic * dX1zc + X1phis * dX1zs) / 2 / R0
        + (X1Rc * X1phic + X1Rs * X1phis) * dZ0 / 2 / R0**2
        - X2phi0 * dZ0 / R0
        - (X1phic**2 + X1phis**2) * dR0 * dZ0 / 2 / R0**3
        + (X1phic * dX1phic + X1phis * dX1phis) * dZ0 / 2 / R0**2
        + (X1phic**2 + X1phis**2) * ddZ0 / 4 / R0**2
    )

    Z_2_neg2 = (
        X2zs
        - (X1phis * dX1zc + X1phic * dX1zs) / 2 / R0
        + (X1Rs * X1phic + X1Rc * X1phis + X1phis * dX1phic + X1phic * dX1phis)
        * dZ0
        / 2
        / R0**2
        - X2phis * dZ0 / R0
        - X1phic * X1phis * dR0 * dZ0 / R0**3
        + X1phic * X1phis * ddZ0 / 2 / R0**2
    )

    Z_2_2 = (
        X2zc
        + (-X1phic * dX1zc + X1phis * dX1zs) / 2 / R0
        + (X1Rc * X1phic - X1Rs * X1phis + X1phic * dX1phic - X1phis * dX1phis)
        * dZ0
        / 2
        / R0**2
        - X2phic * dZ0 / R0
        + (X1phis**2 - X1phic**2) * dR0 * dZ0 / 2 / R0**3
        + (X1phic**2 - X1phis**2) * ddZ0 / 4 / R0**2
    )

    # Fourier Transform in toroidal angle phi

    bases = {}

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

    bases["Rbasis_cos"] = Rbasis
    bases["Rbasis_sin"] = Rbasis_sin
    bases["Zbasis_cos"] = Zbasis
    bases["Zbasis_sin"] = Zbasis_sin

    grid = LinearGrid(M=0, L=0, zeta=phi, NFP=nfp)
    Rtrans = Transform(grid, Rbasis, build_pinv=True, method="auto")
    Ztrans = Transform(grid, Zbasis, build_pinv=True, method="auto")
    Rtrans_sin = Transform(grid, Rbasis_sin, build_pinv=True, method="auto")
    Ztrans_sin = Transform(grid, Zbasis_sin, build_pinv=True, method="auto")

    # if stell sym we should be able to figure out the phi sym for each term
    # R cos terms need a cos phi basis
    # R sin terms need sin phi basis
    # Z cos terms need sin phi basis
    # Z sin terms need cos phi basis

    R_2_0_n = Rtrans.fit(R_2_0)
    R_2_2_n = Rtrans.fit(R_2_2)
    R_2_neg2_n = Rtrans_sin.fit(R_2_neg2)

    coeffs["R_2_0_n"] = R_2_0_n
    coeffs["R_2_2_n"] = R_2_2_n
    coeffs["R_2_neg2_n"] = R_2_neg2_n

    Z_2_0_n = Ztrans_sin.fit(Z_2_0)
    Z_2_2_n = Ztrans_sin.fit(Z_2_2)
    Z_2_neg2_n = Ztrans.fit(Z_2_neg2)

    coeffs["Z_2_0_n"] = Z_2_0_n
    coeffs["Z_2_2_n"] = Z_2_2_n
    coeffs["Z_2_neg2_n"] = Z_2_neg2_n

    return coeffs, bases


def _calc_2nd_order_constraints(qsc, desc_eq, coeffs, bases):  # noqa: C901
    """Creates 2nd order NAE constraints for a DESC eq based off given qsc eq.

    Parameters
    ----------
    qsc : Qsc
        pyQsc Qsc object to use as the NAE constraints on the DESC eq
    desc_eq : Equilibrium
        DESC equilibrium to constrain
    coeffs : dict
        dictionary of arrays with keys like 'X_L_M_n', where
        X is R or Z, L is 1 or 2, and M is 0,1, or 2, are the
        NAE Fourier (in tor. phi) coeffs of radial order L and poloidal order M
    bases : dict
        dictionary of Rbasis_cos, Rbasis_sin, Zbasis_cos, Zbasis_sin,
        the FourierSeries basis objects used to obtain the coefficients, where
        _cos or _sin denotes the symmetry of the Fourier series.
        symmetry is such that the R or Z coefficients is stellarator symmetric
        i.e. R_1_1_n uses the Rbasis_cos, since cos(theta)*cos(phi) is
            stellarator symmetric for R i.e. R(-theta,-phi) = R(theta,phi)
        and Z_1_1_n uses the Zbasis_sin as the term is cos(theta)*sin(phi)
        since Z(-theta,-phi) = - Z(theta,phi) for Z stellarator symmetry

    Returns
    -------
    Rconstraints : tuple
        tuple of FixSumModesR constraints corresponding
        to constraining the O(rho) DESC coefficients, to be used in
        constraining a DESC equilibrium solve
    Zconstraints : tuple
        tuple of FixSumModesZ constraints corresponding
        to constraining the O(rho) DESC coefficients, to be used in
        constraining a DESC equilibrium solve
    Notes
    ----
        Uses 2nd order coefficients eqns in NAE document in docs/dev_notes folder.
    """
    if qsc is not None:
        # r is the ratio  r_NAE / rho_DESC
        # squared (bc is rho^2 and r^2 terms considering here)
        r = 2 * abs(desc_eq.Psi / qsc.Bbar) / 2 / np.pi
    else:
        r = 1  # using DESC equilibrium's behavior, no conversion is needed
    Rconstraints = ()
    Zconstraints = ()

    Rbasis_cos = bases["Rbasis_cos"]
    Rbasis_sin = bases["Rbasis_sin"]
    Zbasis_cos = bases["Zbasis_cos"]
    Zbasis_sin = bases["Zbasis_sin"]

    # R_20n i.e. L=2, M=0
    for n, NAEcoeff in zip(Rbasis_cos.modes[:, 2], coeffs["R_2_0_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.L / 2) + 1):
            modes.append([2 * k, 0, n])
            sum_weights.append((-1) ** k * k * (k + 1))
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Rcon = FixSumModesR(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Rconstraints += (Rcon,)
    # R_2_2n
    for n, NAEcoeff in zip(Rbasis_cos.modes[:, 2], coeffs["R_2_2_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.L / 2) + 1):
            modes.append([2 * k, 2, n])
            sum_weights.append((-1) ** k * k * (k + 1))
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights) / 2
        Rcon = FixSumModesR(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Rconstraints += (Rcon,)
    # R_2_neg2n
    for n, NAEcoeff in zip(Rbasis_sin.modes[:, 2], coeffs["R_2_neg2_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.L / 2) + 1):
            modes.append([2 * k, -2, n])
            sum_weights.append((-1) ** k * k * (k + 1))
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights) / 2
        Rcon = FixSumModesR(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Rconstraints += (Rcon,)
    # Z_2_0n
    for n, NAEcoeff in zip(Zbasis_sin.modes[:, 2], coeffs["Z_2_0_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.L / 2) + 1):
            modes.append([2 * k, 0, n])
            sum_weights.append((-1) ** k * k * (k + 1))
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Zcon = FixSumModesZ(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Zconstraints += (Zcon,)
    # Z_2_neg2n
    for n, NAEcoeff in zip(Zbasis_cos.modes[:, 2], coeffs["Z_2_neg2_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.M / 2) + 1):
            modes.append([2 * k, -2, n])
            sum_weights.append((-1) ** k * k * (k + 1))
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights) / 2
        Zcon = FixSumModesZ(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Zconstraints += (Zcon,)
    # Z_2_2n
    for n, NAEcoeff in zip(Zbasis_sin.modes[:, 2], coeffs["Z_2_2_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.M / 2) + 1):
            modes.append([2 * k, 2, n])
            sum_weights.append((-1) ** k * k * (k + 1))
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights) / 2
        Zcon = FixSumModesZ(
            eq=desc_eq, target=target, sum_weights=sum_weights, modes=modes
        )
        Zconstraints += (Zcon,)

    return Rconstraints, Zconstraints


def make_RZ_cons_2nd_order(qsc, desc_eq, N=None):
    """Make the second order NAE constraints for a DESC equilibrium.

    Parameters
    ----------
        qsc (Qsc): pyQsc Qsc object to use as the NAE constraints on the DESC eq
        desc_eq (Equilibrium): desc equilibrium to constrain

    Returns
    -------
        Rconstraints tuple: tuple of FixSumModesR constraints corresponding to
         constraining the O(rho^2) DESC coefficients,
         to be used in constraining a DESC equilibrium solve
        Zconstraints tuple: tuple of FixSumModesZ constraints corresponding to
         constraining the O(rho^2) DESC coefficients,
         to be used in constraining a DESC equilibrium solve
    """
    Rconstraints = ()
    Zconstraints = ()

    coeffs, bases = _calc_2nd_order_NAE_coeffs(qsc, desc_eq, N=N)
    Rconstraints, Zconstraints = _calc_2nd_order_constraints(
        qsc, desc_eq, coeffs, bases
    )

    return Rconstraints + Zconstraints
