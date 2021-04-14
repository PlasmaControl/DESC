import numpy as np


def write_ascii(fname, equil):
    """Prints the equilibrium solution to a text file

    Parameters
    ----------
    fname : str or path-like
        filename of output file.
    equil : dict
        dictionary of equilibrium parameters.

    """

    cR = equil["cR"]
    cZ = equil["cZ"]
    cL = equil["cL"]
    bdryR = equil["bdryR"]
    bdryZ = equil["bdryZ"]
    cP = equil["cP"]
    cI = equil["cI"]
    Psi_lcfs = equil["Psi_lcfs"]
    NFP = equil["NFP"]
    zern_idx = equil["zern_idx"]
    lambda_idx = equil["lambda_idx"]
    bdry_idx = equil["bdry_idx"]

    # open file
    file = open(fname, "w+")
    file.seek(0)

    # scaling factors
    file.write("NFP = {:3d}\n".format(NFP))
    file.write("Psi = {:16.8E}\n".format(Psi_lcfs))

    # boundary paramters
    nbdry = len(bdry_idx)
    file.write("Nbdry = {:3d}\n".format(nbdry))
    for k in range(nbdry):
        file.write(
            "m: {:3d} n: {:3d} bR = {:16.8E} bZ = {:16.8E}\n".format(
                int(bdry_idx[k, 0]), int(bdry_idx[k, 1]), bdryR[k], bdryZ[k]
            )
        )

    # profile coefficients
    nprof = max(cP.size, cI.size)
    file.write("Nprof = {:3d}\n".format(nprof))
    for k in range(nprof):
        if k >= cP.size:
            file.write("l: {:3d} cP = {:16.8E} cI = {:16.8E}\n".format(k, 0, cI[k]))
        elif k >= cI.size:
            file.write("l: {:3d} cP = {:16.8E} cI = {:16.8E}\n".format(k, cP[k], 0))
        else:
            file.write("l: {:3d} cP = {:16.8E} cI = {:16.8E}\n".format(k, cP[k], cI[k]))

    # R & Z Fourier-Zernike coefficients
    nRZ = len(zern_idx)
    file.write("NRZ = {:5d}\n".format(nRZ))
    for k, lmn in enumerate(zern_idx):
        file.write(
            "l: {:3d} m: {:3d} n: {:3d} cR = {:16.8E} cZ = {:16.8E}\n".format(
                lmn[0], lmn[1], lmn[2], cR[k], cZ[k]
            )
        )

    # lambda Fourier coefficients
    nL = len(lambda_idx)
    file.write("NL = {:5d}\n".format(nL))
    for k, mn in enumerate(lambda_idx):
        file.write("m: {:3d} n: {:3d} cL = {:16.8E}\n".format(mn[0], mn[1], cL[k]))

    # close file
    file.truncate()
    file.close()


def read_ascii(filename):
    """reads a previously generated DESC ascii output file

    Parameters
    ----------
    filename : str or path-like
        path to file to read

    Returns
    -------
    equil : dict
        dictionary of equilibrium parameters.

    """

    equil = {}
    f = open(filename, "r")
    lines = list(f)
    equil["NFP"] = int(lines[0].strip("\n").split()[-1])
    equil["Psi_lcfs"] = float(lines[1].strip("\n").split()[-1])
    lines = lines[2:]

    Nbdry = int(lines[0].strip("\n").split()[-1])
    equil["bdry_idx"] = np.zeros((Nbdry, 2), dtype=int)
    equil["bdryR"] = np.zeros(Nbdry)
    equil["bdryZ"] = np.zeros(Nbdry)
    for i in range(Nbdry):
        equil["bdry_idx"][i, 0] = int(lines[i + 1].strip("\n").split()[1])
        equil["bdry_idx"][i, 1] = int(lines[i + 1].strip("\n").split()[3])
        equil["bdryR"][i] = float(lines[i + 1].strip("\n").split()[6])
        equil["bdryZ"][i] = float(lines[i + 1].strip("\n").split()[9])
    lines = lines[Nbdry + 1 :]

    Nprof = int(lines[0].strip("\n").split()[-1])
    equil["cP"] = np.zeros(Nprof)
    equil["cI"] = np.zeros(Nprof)
    for i in range(Nprof):
        equil["cP"][i] = float(lines[i + 1].strip("\n").split()[4])
        equil["cI"][i] = float(lines[i + 1].strip("\n").split()[7])
    lines = lines[Nprof + 1 :]

    NRZ = int(lines[0].strip("\n").split()[-1])
    equil["zern_idx"] = np.zeros((NRZ, 3), dtype=int)
    equil["cR"] = np.zeros(NRZ)
    equil["cZ"] = np.zeros(NRZ)
    for i in range(NRZ):
        equil["zern_idx"][i, 0] = int(lines[i + 1].strip("\n").split()[1])
        equil["zern_idx"][i, 1] = int(lines[i + 1].strip("\n").split()[3])
        equil["zern_idx"][i, 2] = int(lines[i + 1].strip("\n").split()[5])
        equil["cR"][i] = float(lines[i + 1].strip("\n").split()[8])
        equil["cZ"][i] = float(lines[i + 1].strip("\n").split()[11])
    lines = lines[NRZ + 1 :]

    NL = int(lines[0].strip("\n").split()[-1])
    equil["lambda_idx"] = np.zeros((NL, 2), dtype=int)
    equil["cL"] = np.zeros(NL)
    for i in range(NL):
        equil["lambda_idx"][i, 0] = int(lines[i + 1].strip("\n").split()[1])
        equil["lambda_idx"][i, 1] = int(lines[i + 1].strip("\n").split()[3])
        equil["cL"][i] = float(lines[i + 1].strip("\n").split()[6])
    lines = lines[NL + 1 :]

    return equil
