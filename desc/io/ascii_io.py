"""Legacy functions for reading and writing ascii format."""

import warnings

import numpy as np

from desc.utils import errorif


def write_ascii(fname, eq):  # noqa: C901
    """Print the equilibrium solution to a text file.

    This is not the recommended output file format, please
    use the HDF5 format instead.

    Note: will save the pressure and iota profile as
    power series profiles. If not an iota constrained
    equilibrium, the iota profile will be computed and
    fit with a power series of the equilibrium's L
    resolution.

    Parameters
    ----------
    fname : str or path-like
        filename of output file.
    eq : dict
        dictionary of equilibrium parameters.

    """
    warnings.warn(
        "text-based IO is not as well-tested or maintained as the .h5"
        " format and is therefore not recommended, please use the"
        " .h5 output format instead. text-based IO will be deprecated"
        " in a future release",
        DeprecationWarning,
    )
    eq = eq.copy()  # so don't mess up original one
    if eq.iota is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            eq.iota = eq.get_profile("iota", kind="power_series")
    if eq.pressure is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            eq.pressure = eq.get_profile("p", kind="power_series")
    from desc.profiles import FourierZernikeProfile, PowerSeriesProfile

    errorif(
        isinstance(eq.pressure, FourierZernikeProfile)
        or isinstance(eq.iota, FourierZernikeProfile),
        ValueError,
        "cannot save FourierZernikeProfile as ascii",
    )

    pressure = eq.pressure
    iota = eq.iota

    if not isinstance(iota, PowerSeriesProfile):
        iota = iota.to_powerseries(order=eq.L, sym=True)
    if not isinstance(pressure, PowerSeriesProfile):
        pressure = pressure.to_powerseries(order=eq.L, sym=True)
    L = max(pressure.basis.L, iota.basis.L)
    pressure.change_resolution(L)
    iota.change_resolution(L)
    # always save as a full mode basis for legacy codes
    # which expect this
    if pressure.sym:
        pressure = PowerSeriesProfile(
            params=pressure.params, modes=pressure.basis.modes[:, 0], sym=False
        )
    if iota.sym:
        iota = PowerSeriesProfile(
            params=iota.params, modes=iota.basis.modes[:, 0], sym=False
        )

    assert pressure.sym == iota.sym

    # open file
    file = open(fname, "w+")
    file.seek(0)

    # scaling factors
    file.write("NFP = {:3d}\n".format(int(eq.NFP)))
    file.write("Psi = {:16.8E}\n".format(eq.Psi))

    # boundary parameters
    if eq.sym:
        nbdry = len(np.nonzero(eq.Rb_lmn)[0]) + len(np.nonzero(eq.Zb_lmn)[0])
        file.write("Nbdry = {:3d}\n".format(nbdry))
        for k, (l, m, n) in enumerate(eq.surface.R_basis.modes):
            if eq.Rb_lmn[k] != 0:
                file.write(
                    "m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n".format(
                        m, n, eq.Rb_lmn[k], 0
                    )
                )
        for k, (l, m, n) in enumerate(eq.surface.Z_basis.modes):
            if eq.Zb_lmn[k] != 0:
                file.write(
                    "m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n".format(
                        m, n, 0, eq.Zb_lmn[k]
                    )
                )
    else:
        nbdry = eq.surface.R_basis.num_modes
        file.write("Nbdry = {:3d}\n".format(nbdry))
        for k, (l, m, n) in enumerate(eq.surface.R_basis.modes):
            file.write(
                "m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n".format(
                    m, n, eq.Rb_lmn[k], eq.Zb_lmn[k]
                )
            )

    # profile coefficients
    nprof = pressure.params.size
    file.write("Nprof = {:3d}\n".format(nprof))
    for k, (l, m, n) in enumerate(pressure.basis.modes):
        file.write(
            "l: {:3d} cP = {:16.8E} cI = {:16.8E}\n".format(
                l, pressure.params[k], iota.params[k]
            )
        )

    # R, Z & L Fourier-Zernike coefficients
    if eq.sym:
        nRZ = eq.R_basis.num_modes + eq.Z_basis.num_modes
        file.write("NRZ = {:5d}\n".format(nRZ))
        for k, (l, m, n) in enumerate(eq.R_basis.modes):
            file.write(
                "l: {:3d} m: {:3d} n: {:3d} ".format(l, m, n)
                + "cR = {:16.8E} cZ = {:16.8E} cL = {:16.8E}\n".format(
                    eq.R_lmn[k], 0, 0
                )
            )
        for k, (l, m, n) in enumerate(eq.Z_basis.modes):
            file.write(
                "l: {:3d} m: {:3d} n: {:3d} ".format(l, m, n)
                + "cR = {:16.8E} cZ = {:16.8E} cL = {:16.8E}\n".format(
                    0, eq.Z_lmn[k], eq.L_lmn[k]
                )
            )
    else:
        nRZ = eq.R_basis.num_modes
        file.write("NRZ = {:5d}\n".format(nRZ))
        for k, (l, m, n) in enumerate(eq.R_basis.modes):
            file.write(
                "l: {:3d} m: {:3d} n: {:3d} ".format(l, m, n)
                + "cR = {:16.8E} cZ = {:16.8E} cL = {:16.8E}\n".format(
                    eq.R_lmn[k], eq.Z_lmn[k], eq.L_lmn[k]
                )
            )

    # close file
    file.truncate()
    file.close()


def read_ascii(filename):
    """Read a previously generated DESC ascii output file.

    Parameters
    ----------
    filename : str or path-like
        path to file to read

    Returns
    -------
    eq : dict
        dictionary of equilibrium parameters.

    """
    from desc.backend import sign
    from desc.equilibrium import Equilibrium
    from desc.utils import copy_coeffs

    warnings.warn(
        "text-based IO is not as well-tested or maintained as the .h5"
        " format and is therefore not recommended, please use the"
        " .h5 output format instead. text-based IO will be deprecated"
        " in a future release",
        DeprecationWarning,
    )
    eq = {}
    f = open(filename)
    lines = list(f)
    f.close()
    eq["NFP"] = int(lines[0].strip("\n").split()[-1])
    eq["Psi"] = float(lines[1].strip("\n").split()[-1])
    lines = lines[2:]

    Nbdry = int(lines[0].strip("\n").split()[-1])
    bdry_idx = np.zeros((Nbdry, 2), dtype=int)
    bdryR = np.zeros(Nbdry)
    bdryZ = np.zeros(Nbdry)
    for i in range(Nbdry):
        bdry_idx[i, 0] = int(lines[i + 1].strip("\n").split()[1])
        bdry_idx[i, 1] = int(lines[i + 1].strip("\n").split()[3])
        bdryR[i] = float(lines[i + 1].strip("\n").split()[6])
        bdryZ[i] = float(lines[i + 1].strip("\n").split()[9])
    eq["surface"] = np.hstack(
        [
            np.zeros((Nbdry, 1)),
            bdry_idx,
            bdryR.reshape((-1, 1)),
            bdryZ.reshape((-1, 1)),
        ]
    )
    lines = lines[Nbdry + 1 :]

    Nprof = int(lines[0].strip("\n").split()[-1])
    pl = np.zeros(Nprof).astype(int)
    cP = np.zeros(Nprof)
    cI = np.zeros(Nprof)
    for i in range(Nprof):
        pl[i] = int(lines[i + 1].strip("\n").split()[1])
        cP[i] = float(lines[i + 1].strip("\n").split()[4])
        cI[i] = float(lines[i + 1].strip("\n").split()[7])
    eq["pressure"] = np.hstack([pl.reshape((-1, 1)), cP.reshape((-1, 1))])
    eq["iota"] = np.hstack([pl.reshape((-1, 1)), cI.reshape((-1, 1))])
    lines = lines[Nprof + 1 :]

    NRZ = int(lines[0].strip("\n").split()[-1])
    zern_idx = np.zeros((NRZ, 3), dtype=int)
    cR = np.zeros(NRZ)
    cZ = np.zeros(NRZ)
    cL = np.zeros(NRZ)
    for i in range(NRZ):
        zern_idx[i, 0] = int(lines[i + 1].strip("\n").split()[1])
        zern_idx[i, 1] = int(lines[i + 1].strip("\n").split()[3])
        zern_idx[i, 2] = int(lines[i + 1].strip("\n").split()[5])
        cR[i] = float(lines[i + 1].strip("\n").split()[8])
        cZ[i] = float(lines[i + 1].strip("\n").split()[11])
        cL[i] = float(lines[i + 1].strip("\n").split()[14])
    lines = lines[NRZ + 1 :]

    eq["L"] = np.max(abs(zern_idx[:, 0]))
    eq["M"] = np.max(abs(zern_idx[:, 1]))
    eq["N"] = np.max(abs(zern_idx[:, 2]))

    if np.all(
        cR[np.where(sign(zern_idx[:, 1]) != sign(zern_idx[:, 2]))] == 0
    ) and np.all(cZ[np.where(sign(zern_idx[:, 1]) == sign(zern_idx[:, 2]))] == 0):
        eq["sym"] = True
    else:
        eq["sym"] = False

    eq["spectral_indexing"] = "ansi" if eq["L"] == eq["M"] else "fringe"
    equil = Equilibrium(**eq)
    equil.R_lmn = copy_coeffs(cR, zern_idx, equil.R_basis.modes)
    equil.Z_lmn = copy_coeffs(cZ, zern_idx, equil.Z_basis.modes)
    equil.L_lmn = copy_coeffs(cL, zern_idx, equil.L_basis.modes)

    return equil
