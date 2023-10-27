"""Utilities for reading EFIT geqdsk files."""

import time

import numpy as np
from scipy.interpolate import CubicSpline

from desc.basis import DoubleFourierSeries
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.profiles import SplineProfile
from desc.transform import Transform


def efit_to_desc(g, M, N=0, L=None):
    """Create a DESC equilibrium from an EFIT geqdsk file.

    The equilibrium will not be solved, this just creates an equilibirum
    with the correct boundary and profiles.

    Parameters
    ----------
    g : path-like or dict
        Path to geqdsk file, or dict of already read parameters.
    M : int
        Poloidal resolution of the equilibrium. Used for fitting the boundary.
    N : int, optional
        Toroidal resolution of the equilibrium.
    L : int, optional
        Radial resolution of the equilibrium. Used for fitting profiles.
        If None, spline profiles will be used. Otherwise they are fit
        to even power series in rho.

    Returns
    -------
    eq : Equilibrium
        DESC equilibrium with boundary shape, pressure, iota, and total toroidal flux
        from geqdsk file.
    """
    if not isinstance(g, dict):
        g = read_gfile(g)
    ns = len(g["q"])
    chiN = np.linspace(0, 1, ns)
    chi_scale = g["chi_boundary"] - g["chi_axis"]
    # chi = poloidal flux
    # psi = toroidal flux

    # iota = dchi / dpsi = dchi / dchiN * dchiN / dpsi = dchiN / dpsi * (chi_scale)
    # dpsi = 1/iota dchi = q dchiN
    # psi = integral(q dchiN) * (chi_scale)

    # q * (chi(1) - chi(0)) as function of chiN
    q_spline = CubicSpline(chiN, g["q"] * chi_scale)
    # this gives psi(chiN)
    psi_spline = q_spline.antiderivative()

    psi = psi_spline(chiN)
    Psi_b = psi[-1]  # boundary flux
    psiN = psi / Psi_b
    rho = np.sqrt(psiN)

    pressure = SplineProfile(g["pres"], rho)
    iota = SplineProfile(1 / g["q"], rho)
    if L is not None:
        pressure = pressure.to_powerseries(order=L, sym=True)
        iota = iota.to_powerseries(order=L, sym=True)

    nbdry = g["nbbbs"]
    rbdry = g["rbbbs"]
    zbdry = g["zbbbs"]
    theta = np.linspace(0, 2 * np.pi, nbdry)  # rbbbs includes endpoints

    grid = LinearGrid(theta=theta)
    basis = DoubleFourierSeries(M=M, N=N, sym=False)
    transform = Transform(grid, basis, build_pinv=True)
    Rb_lmn = transform.fit(rbdry)
    Zb_lmn = transform.fit(zbdry)

    surf = FourierRZToroidalSurface(
        R_lmn=Rb_lmn,
        Z_lmn=Zb_lmn,
        modes_R=basis.modes[:, 1:],
        modes_Z=basis.modes[:, 1:],
    )

    eq = Equilibrium(
        L=L, M=M, N=N, surface=surf, pressure=pressure, iota=iota, Psi=Psi_b
    )
    return eq


def read_gfile(filename):
    """Read an EFIT geqdsk file into a dict of ndarray.

    Parameters
    ----------
    filename : path-like
        Path to geqdsk file.

    Returns
    -------
    g : dict
        dict of ndarray containing data from the geqdsk file.
    """
    with open(filename) as f:
        lines = f.readlines()
    g = {}

    has_spaces = all(
        [elem == 5 for elem in [len(lines[i].split()) for i in range(1, 5)]]
    )
    if has_spaces:

        def splitline(s):
            return s.split()

        def splitarr(l):
            return " ".join(l).split()

    else:

        def splitline(s):
            ss = s.split("\n")[0]
            idx = np.arange(0, len(ss) + 1, 16)
            return [ss[i:j] for i, j in zip(idx[:-1], idx[1:])]

        def splitarr(l):
            return splitline("".join([line.split("\n")[0] for line in l]))

    line0 = lines[0].split()
    g["date"] = time.strptime(line0[1], "%m/%d/%Y")
    g["shot"] = int("".join([c for c in line0[2] if c.isdigit()]))
    g["time"] = int("".join([c for c in line0[3] if c.isdigit()]))
    g["nw"] = int("".join([c for c in line0[-2] if c.isdigit()]))
    g["nh"] = int("".join([c for c in line0[-1] if c.isdigit()]))
    if len(line0) == 8:
        g["efit"] = line0[4]
    else:
        g["efit"] = None

    [g["rdim"], g["zdim"], g["rcentr"], g["rleft"], g["zmid"]] = [
        float(foo) for foo in splitline(lines[1])
    ]
    [g["rmaxis"], g["zmaxis"], g["chi_axis"], g["chi_boundary"], g["bcentr"]] = [
        float(foo) for foo in splitline(lines[2])
    ]
    [g["current"], g["chi_axis"], _, g["rmaxis"], _] = [
        float(foo) for foo in splitline(lines[3])
    ]
    [g["zmaxis"], _, g["chi_boundary"], _, _] = [
        float(foo) for foo in splitline(lines[4])
    ]
    lines_per_profile = int(np.ceil(g["nw"] / 5))
    lines_psirz = int(np.ceil(g["nw"] * g["nh"] / 5))
    lines = lines[5:]
    g["fpol"] = np.array([float(foo) for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]
    g["pres"] = np.array([float(foo) for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]
    g["ffprime"] = np.array([float(foo) for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]
    g["pprime"] = np.array([float(foo) for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]
    g["chirz"] = np.array(
        [float(foo) for foo in splitarr(lines[:lines_psirz])]
    ).reshape((g["nh"], g["nw"]))
    lines = lines[lines_psirz:]
    g["q"] = np.array([float(foo) for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]

    if len(lines) > 0:
        [g["nbbbs"], g["limitr"]] = [int(foo) for foo in lines[0].split()]
        lines = lines[1:]
        lines_nbbbs = int(np.ceil(2 * g["nbbbs"] / 5))
        lines_limitr = int(np.ceil(2 * g["limitr"] / 5))
        if len(lines) >= lines_nbbbs:
            rzbbbs = np.array(
                [float(foo) for foo in splitarr(lines[:lines_nbbbs])]
            ).reshape((g["nbbbs"], 2))
            g["rbbbs"], g["zbbbs"] = rzbbbs[:, 0], rzbbbs[:, 1]
            lines = lines[lines_nbbbs:]
        if len(lines) >= lines_limitr:
            rzlimitr = np.array(
                [float(foo) for foo in splitarr(lines[:lines_limitr])]
            ).reshape((g["limitr"], 2))
            g["rlimitr"], g["zlimitr"] = rzlimitr[:, 0], rzlimitr[:, 1]
        lines = lines[lines_limitr:]

    # fix zero pressure
    if np.max(np.abs(g["pres"])) == 0:
        chio = g["chi_boundary"] - g["chi_axis"]
        chiN = np.linspace(0, 1, g["nw"])
        pp = g["pprime"]
        g["pres"] = CubicSpline(chiN, pp).antiderivative()(chiN) / chio

    return g
