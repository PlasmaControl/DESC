import multiprocessing as mp
import os
import shutil
import subprocess
import time

import numpy as np

from desc.utils import errorif

from ._generic import _ExternalObjective


def terpsichore(
    eq,
    path="",
    exec="",
    eq_id="",
    mode_family=0,
    surfs=16,
    lssl=200,
    lssd=100,
    M_max=2,
    N_min=-2,
    N_max=2,
    M_booz_max=4,
    N_booz_max=4,
    awall=2.0,
    xplo=1e-6,
    deltajp=1e-2,
    modelk=1,
    nev=1,
    al0=-5e-1,
    sleep_time=1,
    stop_time=60,
):
    """TERPSICHORE driver function."""
    process = mp.current_process()
    pid = str(process.pid)

    # create temporary directory to store I/O files
    pid_path = os.path.join(path, pid)
    os.mkdir(pid_path)

    exec_path = os.path.join(pid_path, exec)
    input_path = os.path.join(pid_path, "input")
    wout_path = os.path.join(pid_path, "wout.nc")
    fort16_path = os.path.join(pid_path, "fort.16")

    # copy executable to temporary directory
    shutil.copy(os.path.join(path, exec), exec_path)

    _write_wout(eq=eq, path=wout_path, surfs=surfs)
    _write_terps_input(
        path=input_path,
        eq_id=eq_id,
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
        nfp=eq.NFP,
        nev=nev,
        al0=al0,
    )
    _run_terps(
        dir=pid_path,
        exec=exec_path,
        input=input_path,
        wout=wout_path,
        sleep_time=sleep_time,
        stop_time=stop_time,
    )
    growth_rate = _read_terps_output(path=fort16_path)

    # remove temporary directory
    shutil.rmtree(pid_path)

    return np.atleast_1d(growth_rate)


def _write_wout(eq, path, surfs):
    """Write the wout NetCDF file from the equilibrium."""
    from desc.vmec import VMECIO

    VMECIO.save(eq=eq, path=path, surfs=surfs, verbose=0)


def _write_terps_input(
    path,
    eq_id,
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

    f.write("               {}\n".format(eq_id))
    f.write("C\n")
    f.write("C        MM  NMIN  NMAX   MMS NSMIN NSMAX NPROCS INSOL\n")
    f.write(
        "         {:>2d}   {:>3d}   {:>3d}    55    -8    10    1     0\n".format(
            M_booz_max, -N_booz_max, N_booz_max
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
    for _im in range(1, 37):
        if _im >= 10:
            boz_str_title += " " + str(_im)[1]
        else:
            boz_str_title += " " + str(_im)
        boz_str_neg += " 1"
        boz_str_pos += " 1"

    boz_str_title += "  N\n"
    f.write(boz_str_title)
    for _in in range(-N_booz_max, N_booz_max + 1):
        final_str_neg = boz_str_neg + "{:>3}\n".format(_in)
        final_str_pos = boz_str_pos + "{:>3}\n".format(_in)
        if _in < 0.0:
            f.write(final_str_neg)
        else:
            f.write(final_str_pos)

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

    for n in range(-8, 11):

        in_family = False
        if n % 2 == mode_family:  # FIXME: must be modified for nfp != 2,3
            in_family = True

        if n <= 0:
            mode_str = "      0"
        else:
            if (n <= N_max) and (n >= N_min) and (in_family):
                mode_str = "      1"
            else:
                mode_str = "      0"

        for m in range(1, 55 + 1):
            if (m <= M_max) and (in_family):
                if (n <= N_max) and (n >= N_min):
                    mode_str += " 1"
                else:
                    mode_str += " 0"
            else:
                mode_str += " 0"

        mode_str += "{:>3}\n".format(_in)
        f.write(mode_str)

    f.write("C\n")
    f.write("C   NEV NITMAX         AL0     EPSPAM IGREEN MPINIT\n")
    f.write("      {}   4500  {:10.3e}  1.000E-04      0      0\n".format(nev, al0))
    f.write("C\n")

    f.close()


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


class TERPSICHORE(_ExternalObjective):
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

    """

    _units = "(???)"
    _print_value_fmt = "TERPSICHORE growth rate: {:10.3e}"

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        fd_step=1e-4,
        vectorized=True,
        path="",
        exec="",
        eq_id="",
        mode_family=0,
        surfs=16,
        lssl=1,
        lssd=1,
        M_max=2,
        N_min=-2,
        N_max=2,
        M_booz_max=4,
        N_booz_max=4,
        awall=2.0,
        xplo=1e-6,
        deltajp=1e-2,
        modelk=1,
        nev=1,
        al0=-5e-1,
        sleep_time=1,
        stop_time=60,
        name="terpsichore",
    ):
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
            fd_step=fd_step,
            vectorized=vectorized,
            name=name,
            path=path,
            exec=exec,
            eq_id=eq_id,
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
        )
