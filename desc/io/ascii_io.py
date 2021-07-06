import numpy as np


def write_ascii(fname, eq):
    """Print the equilibrium solution to a text file.

    Parameters
    ----------
    fname : str or path-like
        filename of output file.
    eq : dict
        dictionary of equilibrium parameters.

    """

    # open file
    file = open(fname, "w+")
    file.seek(0)

    # scaling factors
    file.write("NFP = {:3d}\n".format(int(eq.NFP)))
    file.write("Psi = {:16.8E}\n".format(eq.Psi))

    # boundary paramters
    if eq.sym:
        nbdry = len(np.nonzero(eq.Rb_lmn)[0]) + len(np.nonzero(eq.Zb_lmn)[0])
        file.write("Nbdry = {:3d}\n".format(nbdry))
        for k, (l, m, n) in enumerate(eq.Rb_basis.modes):
            if eq.Rb_lmn[k] != 0:
                file.write(
                    "m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n".format(
                        m, n, eq.Rb_lmn[k], 0
                    )
                )
        for k, (l, m, n) in enumerate(eq.Zb_basis.modes):
            if eq.Zb_lmn[k] != 0:
                file.write(
                    "m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n".format(
                        m, n, 0, eq.Zb_lmn[k]
                    )
                )
    else:
        nbdry = eq.Rb_basis.num_modes
        file.write("Nbdry = {:3d}\n".format(nbdry))
        for k, (l, m, n) in enumerate(eq.Rb_basis.modes):
            file.write(
                "m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n".format(
                    m, n, eq.Rb_lmn[k], eq.Zb_lmn[k]
                )
            )

    # profile coefficients
    nprof = len(np.nonzero(np.abs(eq.p_l) + np.abs(eq.i_l))[0])
    file.write("Nprof = {:3d}\n".format(nprof))
    for k, (l, m, n) in enumerate(eq.pressure.basis.modes):
        if (eq.p_l[k] != 0) or (eq.i_l[k] != 0):
            file.write(
                "l: {:3d} cP = {:16.8E} cI = {:16.8E}\n".format(k, eq.p_l[k], eq.i_l[k])
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


# FIXME: this is all wrong now with the class structure
def read_ascii(filename):
    """reads a previously generated DESC ascii output file

    Parameters
    ----------
    filename : str or path-like
        path to file to read

    Returns
    -------
    eq : dict
        dictionary of equilibrium parameters.

    """

    eq = {}
    f = open(filename, "r")
    lines = list(f)
    eq["NFP"] = int(lines[0].strip("\n").split()[-1])
    eq["Psi_lcfs"] = float(lines[1].strip("\n").split()[-1])
    lines = lines[2:]

    Nbdry = int(lines[0].strip("\n").split()[-1])
    eq["bdry_idx"] = np.zeros((Nbdry, 2), dtype=int)
    eq["bdryR"] = np.zeros(Nbdry)
    eq["bdryZ"] = np.zeros(Nbdry)
    for i in range(Nbdry):
        eq["bdry_idx"][i, 0] = int(lines[i + 1].strip("\n").split()[1])
        eq["bdry_idx"][i, 1] = int(lines[i + 1].strip("\n").split()[3])
        eq["bdryR"][i] = float(lines[i + 1].strip("\n").split()[6])
        eq["bdryZ"][i] = float(lines[i + 1].strip("\n").split()[9])
    lines = lines[Nbdry + 1 :]

    Nprof = int(lines[0].strip("\n").split()[-1])
    eq["cP"] = np.zeros(Nprof)
    eq["cI"] = np.zeros(Nprof)
    for i in range(Nprof):
        eq["cP"][i] = float(lines[i + 1].strip("\n").split()[4])
        eq["cI"][i] = float(lines[i + 1].strip("\n").split()[7])
    lines = lines[Nprof + 1 :]

    NRZ = int(lines[0].strip("\n").split()[-1])
    eq["zern_idx"] = np.zeros((NRZ, 3), dtype=int)
    eq["cR"] = np.zeros(NRZ)
    eq["cZ"] = np.zeros(NRZ)
    for i in range(NRZ):
        eq["zern_idx"][i, 0] = int(lines[i + 1].strip("\n").split()[1])
        eq["zern_idx"][i, 1] = int(lines[i + 1].strip("\n").split()[3])
        eq["zern_idx"][i, 2] = int(lines[i + 1].strip("\n").split()[5])
        eq["cR"][i] = float(lines[i + 1].strip("\n").split()[8])
        eq["cZ"][i] = float(lines[i + 1].strip("\n").split()[11])
    lines = lines[NRZ + 1 :]

    NL = int(lines[0].strip("\n").split()[-1])
    eq["lambda_idx"] = np.zeros((NL, 2), dtype=int)
    eq["cL"] = np.zeros(NL)
    for i in range(NL):
        eq["lambda_idx"][i, 0] = int(lines[i + 1].strip("\n").split()[1])
        eq["lambda_idx"][i, 1] = int(lines[i + 1].strip("\n").split()[3])
        eq["cL"][i] = float(lines[i + 1].strip("\n").split()[6])
    lines = lines[NL + 1 :]

    return eq
