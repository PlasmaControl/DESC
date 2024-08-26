import functools
import multiprocessing as mp
import os
import shutil
import subprocess
import time

import numpy as np
from netCDF4 import Dataset, stringtochar

from desc.backend import execute_on_cpu, jnp
from desc.basis import DoubleFourierSeries
from desc.compute.utils import get_transforms
from desc.grid import LinearGrid
from desc.transform import Transform
from desc.utils import errorif, warnif
from desc.vmec_utils import ptolemy_identity_rev, zernike_to_fourier

from ._generic import ExternalObjective


@execute_on_cpu
def terpsichore(
    eq,
    processes=1,
    path="",
    exec="",
    mode_family=-1,
    surfs=32,
    lssl=1000,
    lssd=1000,
    M_max=8,
    N_min=-4,
    N_max=4,
    M_booz_max=19,
    N_booz_max=18,
    awall=2.0,
    xplo=1e-6,
    deltajp=1e-2,
    modelk=1,
    nev=1,
    al0=-5e-1,
    sleep_time=1,
    stop_time=60,
    data_transforms=None,
    fit_transform=None,
):
    """TERPSICHORE driver function."""
    idxs = list(range(len(eq)))  # equilibrium indices

    # create temporary directory to store I/O files
    tmp_path = os.path.join(path, "tmp-TERPS")
    os.mkdir(tmp_path)

    # write input files for each equilibrium in serial
    for k in idxs:
        idx_path = os.path.join(tmp_path, str(k))
        os.mkdir(idx_path)
        exec_path = os.path.join(idx_path, exec)
        input_path = os.path.join(idx_path, "input")
        wout_path = os.path.join(idx_path, "wout.nc")
        shutil.copy(os.path.join(path, exec), exec_path)
        _write_wout(
            eq=eq[k],
            path=wout_path,
            surfs=surfs,
            data_transforms=data_transforms,
            fit_transform=fit_transform,
        )
        _write_terps_input(
            path=input_path,
            mode_family=mode_family,
            surfs=surfs,
            lssl=lssl,
            lssd=lssd,
            M_max=M_max,
            N_min=N_min,
            N_max=N_max,
            M_booz_max=M_booz_max,
            N_booz_max=N_booz_max,
            awall=awall,
            xplo=xplo,
            deltajp=deltajp,
            modelk=modelk,
            nfp=eq[k].NFP,
            nev=nev,
            al0=al0,
        )

    # run TERPSICHORE on list of equilibria in parallel
    if len(eq) == 1:  # no multiprocessing if only one equilibrium
        result = jnp.atleast_1d(
            _pool_fun(
                0,
                path=tmp_path,
                exec=exec,
                al0=al0,
                sleep_time=sleep_time,
                stop_time=stop_time,
            )
        )
    else:
        with mp.Pool(processes=min(processes, len(eq))) as pool:
            results = pool.map(
                functools.partial(
                    _pool_fun,
                    path=tmp_path,
                    exec=exec,
                    al0=al0,
                    sleep_time=sleep_time,
                    stop_time=stop_time,
                ),
                idxs,
            )
            pool.close()
            pool.join()
            result = jnp.vstack(results, dtype=float)

    # remove temporary directory
    shutil.rmtree(tmp_path)

    return result


def _write_wout(eq, path, surfs, data_transforms, fit_transform):  # noqa: C901
    """Write the wout NetCDF file from the equilibrium."""
    # this is a lightweight version of VMECIO.save
    file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")

    Psi = eq.Psi
    NFP = eq.NFP
    M = eq.M
    N = eq.N
    M_nyq = M + 4
    N_nyq = N + 2 if N > 0 else 0

    # VMEC radial coordinate: s = rho^2 = Psi / Psi(LCFS)
    s_full = np.linspace(0, 1, surfs)
    s_half = s_full[0:-1] + 0.5 / (surfs - 1)
    r_full = np.sqrt(s_full)
    r_half = np.sqrt(s_half)

    # dimensions
    file.createDimension("radius", surfs)  # number of flux surfaces
    file.createDimension("mn_mode", (2 * N + 1) * M + N + 1)  # number of Fourier modes
    file.createDimension(
        "mn_mode_nyq", (2 * N_nyq + 1) * M_nyq + N_nyq + 1
    )  # number of Nyquist Fourier modes
    file.createDimension("n_tor", N + 1)  # number of toroidal Fourier modes
    file.createDimension("preset", 21)  # dimension of profile inputs
    file.createDimension("ndfmax", 101)  # used for am_aux & ai_aux
    file.createDimension("time", 100)  # used for fsq* & wdot
    file.createDimension("dim_00001", 1)
    file.createDimension("dim_00020", 20)
    file.createDimension("dim_00100", 100)
    file.createDimension("dim_00200", 200)

    grid_half = LinearGrid(M=M_nyq, N=N_nyq, NFP=NFP, rho=r_half)
    data_half = eq.compute(
        [
            "B_rho",
            "B_theta",
            "B_zeta",
            "G",
            "I",
            "J",
            "iota",
            "p",
            "sqrt(g)",
            "V_r(r)",
            "|B|",
            "<|B|^2>",
        ],
        grid=grid_half,
        transforms=data_transforms,
    )

    mgrid_file = file.createVariable("mgrid_file", "S1", ("dim_00200",))
    mgrid_file[:] = stringtochar(
        np.array(["none" + " " * 196], "S" + str(file.dimensions["dim_00200"].size))
    )

    lasym = file.createVariable("lasym__logical__", np.int32)
    lasym.long_name = "asymmetry logical (0 = stellarator symmetry)"
    lasym[:] = int(not eq.sym)

    nfp = file.createVariable("nfp", np.int32)
    nfp.long_name = "number of field periods"
    nfp[:] = NFP

    ns = file.createVariable("ns", np.int32)
    ns.long_name = "number of flux surfaces"
    ns[:] = surfs

    mpol = file.createVariable("mpol", np.int32)
    mpol.long_name = "number of poloidal Fourier modes"
    mpol[:] = M + 1

    ntor = file.createVariable("ntor", np.int32)
    ntor.long_name = "number of positive toroidal Fourier modes"
    ntor[:] = N

    mnmax = file.createVariable("mnmax", np.int32)
    mnmax.long_name = "total number of Fourier modes"
    mnmax[:] = file.dimensions["mn_mode"].size

    xm = file.createVariable("xm", np.float64, ("mn_mode",))
    xm.long_name = "poloidal mode numbers"
    xm[:] = np.tile(np.linspace(0, M, M + 1), (2 * N + 1, 1)).T.flatten()[
        -file.dimensions["mn_mode"].size :
    ]

    xn = file.createVariable("xn", np.float64, ("mn_mode",))
    xn.long_name = "toroidal mode numbers"
    xn[:] = np.tile(np.linspace(-N, N, 2 * N + 1) * NFP, M + 1)[
        -file.dimensions["mn_mode"].size :
    ]

    mnmax_nyq = file.createVariable("mnmax_nyq", np.int32)
    mnmax_nyq.long_name = "total number of Nyquist Fourier modes"
    mnmax_nyq[:] = file.dimensions["mn_mode_nyq"].size

    xm_nyq = file.createVariable("xm_nyq", np.float64, ("mn_mode_nyq",))
    xm_nyq.long_name = "poloidal Nyquist mode numbers"
    xm_nyq[:] = np.tile(
        np.linspace(0, M_nyq, M_nyq + 1), (2 * N_nyq + 1, 1)
    ).T.flatten()[-file.dimensions["mn_mode_nyq"].size :]

    xn_nyq = file.createVariable("xn_nyq", np.float64, ("mn_mode_nyq",))
    xn_nyq.long_name = "toroidal Nyquist mode numbers"
    xn_nyq[:] = np.tile(np.linspace(-N_nyq, N_nyq, 2 * N_nyq + 1) * NFP, M_nyq + 1)[
        -file.dimensions["mn_mode_nyq"].size :
    ]

    gamma = file.createVariable("gamma", np.float64)
    gamma.long_name = "compressibility index (0 = pressure prescribed)"
    gamma[:] = 0

    # half mesh quantities

    pres = file.createVariable("pres", np.float64, ("radius",))
    pres.long_name = "pressure on half mesh"
    pres.units = "Pa"
    pres[0] = 0
    pres[1:] = grid_half.compress(data_half["p"])

    mass = file.createVariable("mass", np.float64, ("radius",))
    mass.long_name = "mass on half mesh"
    mass.units = "Pa"
    mass[:] = pres[:]

    iotas = file.createVariable("iotas", np.float64, ("radius",))
    iotas.long_name = "rotational transform on half mesh"
    iotas.units = "None"
    iotas[0] = 0
    iotas[1:] = -grid_half.compress(data_half["iota"])  # - for negative Jacobian

    phips = file.createVariable("phips", np.float64, ("radius",))
    phips.long_name = "d(phi)/ds * -1/2pi: toroidal flux derivative, on half mesh"
    phips[0] = 0
    phips[1:] = -Psi * np.ones((surfs - 1,)) / (2 * np.pi)

    vp = file.createVariable("vp", np.float64, ("radius",))
    vp.long_name = "dV/ds normalized by 4*pi^2, on half mesh"
    vp.units = "m^3"
    vp[1:] = grid_half.compress(data_half["V_r(r)"]) / (
        8 * np.pi**2 * grid_half.compress(data_half["rho"])
    )
    vp[0] = 0

    # R
    rmnc = file.createVariable("rmnc", np.float64, ("radius", "mn_mode"))
    rmnc.long_name = "cos(m*t-n*p) component of cylindrical R, on full mesh"
    rmnc.units = "m"
    if not eq.sym:
        rmns = file.createVariable("rmns", np.float64, ("radius", "mn_mode"))
        rmns.long_name = "sin(m*t-n*p) component of cylindrical R, on full mesh"
        rmns.units = "m"
    r1 = np.ones_like(eq.R_lmn)
    r1[eq.R_basis.modes[:, 1] < 0] *= -1
    m, n, x_mn = zernike_to_fourier(r1 * eq.R_lmn, basis=eq.R_basis, rho=r_full)
    xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
    rmnc[:] = c
    if not eq.sym:
        rmns[:] = s

    # Z
    zmns = file.createVariable("zmns", np.float64, ("radius", "mn_mode"))
    zmns.long_name = "sin(m*t-n*p) component of cylindrical Z, on full mesh"
    zmns.units = "m"
    if not eq.sym:
        zmnc = file.createVariable("zmnc", np.float64, ("radius", "mn_mode"))
        zmnc.long_name = "cos(m*t-n*p) component of cylindrical Z, on full mesh"
        zmnc.units = "m"
    z1 = np.ones_like(eq.Z_lmn)
    z1[eq.Z_basis.modes[:, 1] < 0] *= -1
    m, n, x_mn = zernike_to_fourier(z1 * eq.Z_lmn, basis=eq.Z_basis, rho=r_full)
    xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
    zmns[:] = s
    if not eq.sym:
        zmnc[:] = c

    # derived quantities (approximate conversion)

    def fit(x):
        y = fit_transform.fit(x)
        return np.where(fit_transform.basis.modes[:, 1] < 0, -y, y)

    # Jacobian
    gmnc = file.createVariable("gmnc", np.float64, ("radius", "mn_mode_nyq"))
    gmnc.long_name = "cos(m*t-n*p) component of Jacobian, on half mesh"
    gmnc.units = "m"
    if not eq.sym:
        gmns = file.createVariable("gmns", np.float64, ("radius", "mn_mode_nyq"))
        gmns.long_name = "sin(m*t-n*p) component of Jacobian, on half mesh"
        gmns.units = "m"
    m = fit_transform.basis.modes[:, 1]
    n = fit_transform.basis.modes[:, 2]
    # d(rho)/d(s) = 1/(2*rho)
    data = (
        (data_half["sqrt(g)"] / (2 * data_half["rho"]))
        .reshape(
            (grid_half.num_theta, grid_half.num_rho, grid_half.num_zeta), order="F"
        )
        .transpose((1, 0, 2))
        .reshape((grid_half.num_rho, -1), order="F")
    )
    x_mn = np.zeros((surfs - 1, m.size))
    for i in range(surfs - 1):
        x_mn[i, :] = fit(data[i, :])
    xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
    gmnc[0, :] = 0
    gmnc[1:, :] = -c  # negative sign for negative Jacobian
    if not eq.sym:
        gmns[0, :] = 0
        gmns[1:, :] = -s

    # |B|
    bmnc = file.createVariable("bmnc", np.float64, ("radius", "mn_mode_nyq"))
    bmnc.long_name = "cos(m*t-n*p) component of |B|, on half mesh"
    bmnc.units = "T"
    if not eq.sym:
        bmns = file.createVariable("bmns", np.float64, ("radius", "mn_mode_nyq"))
        bmns.long_name = "sin(m*t-n*p) component of |B|, on half mesh"
        bmns.units = "T"
    m = fit_transform.basis.modes[:, 1]
    n = fit_transform.basis.modes[:, 2]
    data = (
        data_half["|B|"]
        .reshape(
            (grid_half.num_theta, grid_half.num_rho, grid_half.num_zeta), order="F"
        )
        .transpose((1, 0, 2))
        .reshape((grid_half.num_rho, -1), order="F")
    )
    x_mn = np.zeros((surfs - 1, m.size))
    for i in range(surfs - 1):
        x_mn[i, :] = fit(data[i, :])
    xm, xn, s, c = ptolemy_identity_rev(m, n, x_mn)
    bmnc[0, :] = 0
    bmnc[1:, :] = c
    if not eq.sym:
        bmns[0, :] = 0
        bmns[1:, :] = s

    file.close()


def _write_terps_input(  # noqa: C901
    path,
    mode_family,
    surfs,
    lssl,
    lssd,
    M_max,
    N_min,
    N_max,
    M_booz_max,
    N_booz_max,
    awall,
    xplo,
    deltajp,
    modelk,
    nfp,
    nev,
    al0,
):
    """Write TERPSICHORE input file."""
    ivac = surfs // 4
    if (N_max > 8) or (M_max > 16):
        nj = 150
        nk = 150
    elif (N_max > 4) or (M_max > 8):
        nj = 100
        nk = 100
    else:
        nj = 50
        nk = 50

    f = open(path, "w")

    f.write("               TERPSICHORE\n")
    f.write("C\n")
    f.write("C        MM  NMIN  NMAX   MMS NSMIN NSMAX NPROCS INSOL\n")
    f.write(
        "         {:>2d}   {:>3d}   {:>3d}    55   {:>3d}   {:>3d}    1     0\n".format(
            M_booz_max, -N_booz_max, N_booz_max, N_min, N_max
        )
    )
    f.write("C\n")
    f.write("C        NJ    NK  IVAC  LSSL  LSSD MMAXDF NMAXDF\n")
    f.write(
        "        {:>3d}   {:>3d}   {:>3d}  {:>4d}  {:>4d}    120     64\n".format(
            nj, nk, ivac, lssl, lssd
        )
    )
    f.write("C\n")
    f.write("C     TABLE OF FOURIER COEFFIENTS FOR BOOZER COORDINATES\n")
    f.write("C     EQUILIBRIUM SETTINGS ARE COMPUTED FROM FIT/VMEC\n")
    f.write("C\n")

    boz_str_title = "C M=  0"
    boz_str_neg = "      0"
    boz_str_pos = "      1"
    for m in range(1, 37):
        if m >= 10:
            boz_str_title += " " + str(m)[1]
        else:
            boz_str_title += " " + str(m)
        boz_str_neg += " 1"
        boz_str_pos += " 1"
    boz_str_title += "  N\n"
    f.write(boz_str_title)

    for n in range(-N_booz_max, N_booz_max + 1):
        if n < 0:
            f.write(boz_str_neg + "{:>3}\n".format(n))
        else:
            f.write(boz_str_pos + "{:>3}\n".format(n))

    f.write("C\n")
    f.write("      LLAMPR      LVMTPR      LMETPR      LFOUPR\n")
    f.write("           0           0           0           0\n")
    f.write("      LLHSPR      LRHSPR      LEIGPR      LEFCPR\n")
    f.write("           9           9           1           1\n")
    f.write("      LXYZPR      LIOTPL      LDW2PL      LEFCPL\n")
    f.write("           0           1           1           1\n")
    f.write("      LCURRF      LMESHP      LMESHV      LITERS\n")
    f.write("           1           1           2           1\n")
    f.write("      LXYZPL      LEFPLS      LEQVPL      LPRESS\n")
    f.write("           1           1           0           0\n")
    f.write("C\n")
    f.write(
        "C    PVAC        PARFAC      QONAX        QN         DSVAC       "
        + "QVAC    NOWALL\n"
    )
    f.write(
        "  1.0001e+00  0.0000e-00  0.6500e-00  0.0000e-00  1.0000e-00  "
        + "1.0001e+00     -2\n"
    )
    f.write("\n")
    f.write(
        "C    AWALL       EWALL       DWALL       GWALL       DRWAL       DZWAL   "
        + "NPWALL\n"
    )
    f.write(
        "  {:10.4e}  ".format(awall)
        + "1.5000e+00 -1.0000e-00  5.2000e-00 -0.0000e-00 +0.0000e-00      2\n"
    )
    f.write("C\n")
    f.write("C    RPLMIN       XPLO      DELTAJP       WCT        CURFAC\n")
    f.write(
        "  1.0000e-05  {:10.4e}  {:10.4e}  6.6667e-01  1.0000e-00\n".format(
            xplo, deltajp
        )
    )
    f.write("C\n")
    f.write(
        "C                                                             "
        + "MODELK =      {}\n".format(modelk)
    )
    f.write("C\n")
    f.write(
        "C     NUMBER OF EQUILIBRIUM FIELD PERIODS PER STABILITY PERIOD: "
        + "NSTA =      {}\n".format(nfp)
    )
    f.write("C\n")
    f.write("C     TABLE OF FOURIER COEFFIENTS FOR STABILITY DISPLACEMENTS\n")
    f.write("C\n")
    f.write(
        "C M=  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 "
        + "0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5  N\n"
    )

    for n in range(N_min, N_max + 1):
        mode_str = "     "
        for m in range(56):
            if mode_family < 0 or n % 2 == mode_family:  # FIXME: modify for nfp != 2,3
                if (m <= M_max) and (n >= N_min) and (n <= N_max):
                    if m == 0:
                        if n > 0:
                            mode_str += " 1"
                        else:
                            mode_str += " 0"
                    else:
                        mode_str += " 1"
                else:
                    mode_str += " 0"
            else:
                mode_str += " 0"

        mode_str += "{:>3}\n".format(n)
        f.write(mode_str)

    f.write("C\n")
    f.write("C   NEV NITMAX         AL0     EPSPAM IGREEN MPINIT\n")
    f.write("      {}   4500  {:10.3e}  1.000E-04      0      0\n".format(nev, al0))
    f.write("C\n")

    f.close()


def _pool_fun(k, path, exec, al0, sleep_time, stop_time):
    """Run TERPSICHORE and read growth rate for equilibrium with index k."""
    idx_path = os.path.join(path, str(k))
    exec_path = os.path.join(idx_path, exec)
    fort16_path = os.path.join(idx_path, "fort.16")
    input_path = os.path.join(idx_path, "input")
    wout_path = os.path.join(idx_path, "wout.nc")

    try:
        _run_terps(
            dir=idx_path,
            exec=exec_path,
            input=input_path,
            wout=wout_path,
            sleep_time=sleep_time,
            stop_time=stop_time,
        )
        growth_rate = _read_terps_output(path=fort16_path)
    except RuntimeError:
        warnif(
            True, UserWarning, "TERPSICHORE growth rate not found, using default value."
        )
        growth_rate = abs(al0)  # default growth rate if it was unable to find one

    return np.atleast_1d(growth_rate)


def _run_terps(dir, exec, input, wout, sleep_time, stop_time):
    """Run TERPSICHORE."""
    stdout_path = os.path.join(dir, "stdout.terps")
    stderr_path = os.path.join(dir, "stderr.terps")

    fout = open(stdout_path, "w")
    ferr = open(stderr_path, "w")

    cmd = exec + " < " + input + " " + wout
    terps_subprocess = subprocess.run(
        cmd, cwd=dir, shell=True, stdout=fout, stderr=ferr
    )

    run_time = 0.0
    tic = time.perf_counter()
    while not _is_terps_complete(
        path=stdout_path,
        run_time=run_time,
        stop_time=stop_time,
        terps_subprocess=terps_subprocess,
    ):
        time.sleep(sleep_time)
        toc = time.perf_counter()
        run_time = toc - tic

    fout.close()
    ferr.close()


def _is_terps_complete(path, run_time, stop_time, terps_subprocess):
    """Check if TERPSICHORE is finished running."""
    if not os.path.exists(path):
        return False

    errorif(
        run_time > stop_time, RuntimeError, "TERPS was unable to find a growth rate!"
    )

    file = open(path)
    terps_output = file.read()
    if "GROWTH RATE" in terps_output:
        if terps_subprocess.returncode is None:
            terps_subprocess.terminate()
        file.close()
        return True
    else:
        file.close()
        return False


def _read_terps_output(path):
    """Read TERPSICHORE output file and return the growth rate."""
    errorif(
        not os.path.exists(path), RuntimeError, "TERPS fort.16 output file not found!"
    )

    file = open(path)
    growth_rates = []
    for line in file:
        index = line.find("GROWTH RATE")
        if index != -1:
            growth_rates.append(float(line.strip().split("=")[1]))
    file.close()

    errorif(len(growth_rates) == 0, RuntimeError, "Growth rate not found!")
    errorif(
        len(growth_rates) > 1,
        NotImplementedError,
        "Not capable of handling multiple growth rates.",
    )

    return growth_rates[0]


class TERPSICHORE(ExternalObjective):
    """Computes linear MHD stability from calls to the code TERPSICHORE.

    Returns the linear growth rate of the fastest growing instability.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=0``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Has no effect for this objective.
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    name : str, optional
        Name of the objective function.

    # TODO: update Parameters docs
    # If mode_family < 0, includes all modes in desired range

    """

    _units = "(dimensionless)"  # normalized by Alfven frequency
    _print_value_fmt = "TERPSICHORE growth rate: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        abs_step=1e-4,
        rel_step=0,
        processes=1,
        path="",
        exec="",
        mode_family=-1,
        surfs=32,
        lssl=1000,
        lssd=1000,
        M_max=8,
        N_min=-4,
        N_max=4,
        M_booz_max=19,
        N_booz_max=18,
        awall=2.0,
        xplo=1e-6,
        deltajp=5e-1,
        modelk=0,
        nev=1,
        al0=-5e-1,
        sleep_time=1,
        stop_time=60,
        name="terpsichore",
    ):
        if target is None and bounds is None:
            bounds = (-np.inf, 0)
        super().__init__(
            eq=eq,
            fun=terpsichore,
            dim_f=1,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            vectorized=True,
            abs_step=abs_step,
            rel_step=rel_step,
            processes=processes,
            path=path,
            exec=exec,
            mode_family=mode_family,
            surfs=surfs,
            lssl=lssl,
            lssd=lssd,
            M_max=M_max,
            N_min=N_min,
            N_max=N_max,
            M_booz_max=M_booz_max,
            N_booz_max=N_booz_max,
            awall=awall,
            xplo=xplo,
            deltajp=deltajp,
            modelk=modelk,
            nev=nev,
            al0=al0,
            sleep_time=sleep_time,
            stop_time=stop_time,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        # transforms for writing the wout file
        surfs = self._kwargs.get("surfs")
        NFP = self._eq.NFP
        M = self._eq.M
        N = self._eq.N
        M_nyq = M + 4
        N_nyq = N + 2 if N > 0 else 0
        s_full = np.linspace(0, 1, surfs)
        s_half = s_full[0:-1] + 0.5 / (surfs - 1)
        r_half = np.sqrt(s_half)
        grid_lcfs = LinearGrid(M=M_nyq, N=N_nyq, rho=np.array([1.0]), NFP=NFP)
        grid_half = LinearGrid(M=M_nyq, N=N_nyq, NFP=NFP, rho=r_half)
        self._kwargs["data_transforms"] = get_transforms(
            keys=[
                "B_rho",
                "B_theta",
                "B_zeta",
                "G",
                "I",
                "J",
                "iota",
                "p",
                "sqrt(g)",
                "V_r(r)",
                "|B|",
                "<|B|^2>",
            ],
            obj=self._eq,
            grid=grid_half,
        )
        if self._eq.sym:
            fit_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym="cos")
        else:
            fit_basis = DoubleFourierSeries(M=M_nyq, N=N_nyq, NFP=NFP, sym=None)
        self._kwargs["fit_transform"] = Transform(
            grid=grid_lcfs, basis=fit_basis, build=False, build_pinv=True
        )

        super().build(use_jit=use_jit, verbose=verbose)
