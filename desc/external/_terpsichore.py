import datetime
import functools
import multiprocessing as mp
import os
import shutil
import subprocess

import numpy as np
from netCDF4 import Dataset, stringtochar

from desc.backend import jnp
from desc.basis import DoubleFourierSeries
from desc.compat import flip_theta
from desc.compute.utils import get_transforms
from desc.grid import LinearGrid
from desc.objectives import ExternalObjective
from desc.objectives.objective_funs import collect_docs
from desc.transform import Transform
from desc.utils import errorif, warnif
from desc.vmec_utils import ptolemy_identity_rev, zernike_to_fourier


def terpsichore(
    eq,
    *,
    processes=1,
    path,
    exec,
    mode_family=-1,
    surfs=101,
    M_nyq=None,
    N_nyq=None,
    M_booz_max=None,
    N_booz_max=None,
    M_max=8,
    N_min=-4,
    N_max=4,
    lssl=None,
    lssd=None,
    awall=2.0,
    deltaJp=1e-4,
    modelk=0,
    al0=-5e-1,
    timeout=60,
    tmp_dir="tmp_TERPS",
    save_tmp=False,
    data_transforms=None,
    fit_transform=None,
    theta0_outboard=True,
):
    """TERPSICHORE driver function.

    Parameters
    ----------
    eq : list
        A list of Equilibrium objects to run TERPSICHORE on.
    processes : int, optional
        Maximum number of CPU threads to use for multiprocessing. Default = 1.
    path : str
        Path to the directory where temporary files will be stored.
    exec : str
        File name of the TERPSICHORE executable. Must be located in the directory
        specified by ``path``.
    mode_family : int, optional
        The mode family of the instabilities to consider. The toroidal modes included
        in a mode family are n_i = i * eq.NFP ± k, where i = ..., -1, 0, +1, ... and
        k ∈ [0, mode_family]. Possible mode families are in the range [0, eq.NFP // 2].
        If mode_family < 0 then all mode families are considered. Default = -1.
    surfs : int, optional
        Number of surfaces to include in the equilibrium input. More surfaces provides
        more accuracy at the cost of longer compute times. Default = 101.
    M_nyq, N_nyq: int
        The max poloidal and toroidal mode numbers to use in the Nyquist spectrum of the
        equilibrium input. Defaults to ``eq.M + 4`` and ``eq.N + 2``.
    M_booz_max, N_booz_max : int, optional
        Maximum poloidal and toroidal mode numbers of Boozer spectrum. Will include
        modes with m ∈ [0, M_booz_max] and n ∈ [-N_booz_max, N_booz_max].
        Defaults to ``2 * eq.M`` and ``2 * eq.N``.
    M_max : int, optional
        Maximum poloidal mode number of stability modes to consider. Will include modes
        with m ∈ [0, M_max] (if ``mode_family < 0``). Default = 8.
    N_min : int, optional
        Minimum toroidal mode number of stability modes to consider. Will include modes
        with n ∈ [N_min, N_max] (if ``mode_family < 0``). Default = -4.
    N_max : int, optional
        Maximum toroidal mode number of stability modes to consider. Will include modes
        with n ∈ [N_min, N_max] (if ``mode_family < 0``). Default = 4.
    lssl : int, optional
        Minimum number of possible permutations of Boozer mode combinations
        (determined by ``M_booz_max`` and ``N_booz_max``). If TERPSICHORE fails to run,
        try increasing this parameter. Default = ``20 * M_booz_max * N_booz_max``.
    lssd : int, optional
        Minimum number of possible permutations of stability mode combinations
        (determined by ``M_max`` and ``N_max``). If TERPSICHORE fails to run,
        try increasing this parameter. Default = ``20 * M_max * N_max``.
    awall : float, optional
        Ratio of the radius of the conformal conducting wall to the plasma minor radius.
        The conducting wall is obtained by scaling the m ≠ 0 Fourier components of the
        plasma boundary by ``awall``. A shorter wall offset will help stabilize the
        plasma. If TERPSICHORE fails to run, try decreasing this parameter. Default = 2.
    deltaJp : float
        Resonance detuning parameter to resolve parallel current density singularities.
        A larger value can artificially improve the stability. Default = 1e-4.
    modelk : int, optional
        0 = Noninteracting anisotropic fast particle stability model with reduced
        kinetic energy. 1 = Kruskal-Oberman anisotropic energy principle model with
        reduced kinetic energy. 2 = Noninteracting anisotropic fast particle stability
        model with physical kinetic energy. 3 = Kruskal-Oberman anisotropic energy
        principle model with physical kinetic energy. Default = 0.
    al0 : float, optional
        Initial guess of the eigenvalue. Use a sufficiently negative value to find the
        most unstable growth rate. If TERPSICHORE fails to run, the objective will
        return a growth rate of ``abs(al0)``. Default = -0.5.
    timeout : float, optional
        Time in seconds to wait for TERPSICHORE to execute before terminating its run.
        See ``timeout`` argument of ``subprocess.run`` for more info. Default = 60.
    tmp_dir : str, optional
        Name of directory where temporary files used and generated by TERPSICHORE are
        stored. This will be a sub-directory within the directory specified by ``path``.
        Default = "tmp_TERPS".
    save_tmp : bool, optional
        If True, ``tmp_dir`` is preserved to record the TERPSICHORE I/O files.
        It will contain sub-directories named by a timestamp for each time the function
        was evaluated. Those contain another layer of sub-directories named by integers
        ranging from 0 to ``len(eq)`` with the files used in each TERPSICHORE call.
        If False, ``tmp_dir`` and all of its contents are removed at the end of the
        function evaluation. Default = False.
    data_transforms, fit_transform : dict, Transform
        Transform objects used for writing the wout files.
    theta0_outboard : bool
        Should be True if theta=0 is defined at the outboard midplane and False
        otherwise. TERPSICHORE assumes theta=0 is on the outboard midplane, and so the
        equilibria will be flipped correspondingly if this is not already the case.

    Returns
    -------
    result : ndarray
        The TERPSICHORE linear growth rate of the fastest growing instability,
        normalized by the Alfven frequency. A negative growth rate denotes stability and
        a positive growth rate denotes instability. If TERPSICHORE fails to run, a
        result of ``abs(al0)`` is returned.

    """
    # TERPSICHORE assumes theta=0 is on the outboard midplane
    if not theta0_outboard:
        eq = flip_theta(eq)

    # create a temporary directory to store I/O files
    tmp_path = os.path.join(path, tmp_dir)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)  # tmp_path = highest-level directory for all temporary files
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time_path = os.path.join(tmp_path, timestamp)
    os.mkdir(time_path)  # time_path = sub-dir for each time this function is called

    # write input files for each equilibrium in serial
    idxs = list(range(len(eq)))  # equilibrium indices
    for k in idxs:
        idx_path = os.path.join(time_path, str(k))
        os.mkdir(idx_path)  # idx_path = sub-sub-dir for each eq in the list of eqs
        exec_path = os.path.join(idx_path, exec)
        input_path = os.path.join(idx_path, "input")
        wout_path = os.path.join(idx_path, "wout.nc")
        shutil.copy(os.path.join(path, exec), exec_path)
        _write_wout(
            eq=eq[k],
            path=wout_path,
            surfs=surfs,
            M_nyq=M_nyq,
            N_nyq=N_nyq,
            data_transforms=data_transforms,
            fit_transform=fit_transform,
        )
        _write_terps_input(
            path=input_path,
            mode_family=mode_family,
            surfs=surfs,
            M_booz_max=2 * eq[k].M if M_booz_max is None else M_booz_max,
            N_booz_max=2 * eq[k].N if N_booz_max is None else N_booz_max,
            M_max=M_max,
            N_min=N_min,
            N_max=N_max,
            lssl=lssl,
            lssd=lssd,
            awall=awall,
            deltaJp=deltaJp,
            modelk=modelk,
            nfp=eq[k].NFP,
            al0=al0,
        )

    # run TERPSICHORE on list of equilibria in parallel
    if len(eq) == 1:  # no multiprocessing if only one equilibrium
        result = jnp.atleast_1d(
            _pool_fun(0, path=time_path, exec=exec, al0=al0, timeout=timeout)
        )
    else:
        with mp.Pool(processes=min(processes, len(eq))) as pool:
            results = pool.map(
                functools.partial(
                    _pool_fun, path=time_path, exec=exec, al0=al0, timeout=timeout
                ),
                idxs,
            )
            result = jnp.vstack(results, dtype=float)

    # remove temporary directory and all sub-directories
    if not save_tmp:
        shutil.rmtree(tmp_path)

    return result


def _write_wout(
    eq, path, surfs, M_nyq, N_nyq, data_transforms, fit_transform
):  # noqa: C901
    """Write the wout NetCDF file from the equilibrium."""
    # this is a lightweight version of VMECIO.save
    file = Dataset(path, mode="w", format="NETCDF3_64BIT_OFFSET")

    Psi = eq.Psi
    NFP = eq.NFP
    M = eq.M
    N = eq.N
    M_nyq = M + 4 if M_nyq is None else M_nyq
    N_nyq = N + 2 if N_nyq is None else N_nyq
    N_nyq = 0 if int(N) == 0 else N_nyq
    M_grid = eq.M_grid
    N_grid = eq.N_grid

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

    grid_half = LinearGrid(M=M_grid, N=N_grid, NFP=NFP, rho=r_half)
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
    M_booz_max,
    N_booz_max,
    M_max,
    N_min,
    N_max,
    lssl,
    lssd,
    awall,
    deltaJp,
    modelk,
    nfp,
    al0,
):
    """Write TERPSICHORE input file."""
    lssl = 20 * M_booz_max * N_booz_max if lssl is None else lssl
    lssd = 20 * M_max * N_max if lssd is None else lssd

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
        "  1.0000e-05  1.0000e-06  {:10.4e}  6.6667e-01  1.0000e-00\n".format(deltaJp)
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

    if mode_family < 0:
        n_modes = np.arange(N_min, N_max + 1)
    else:
        n_modes = np.arange(N_min, N_max + 1) * nfp
        n_modes = np.concatenate((n_modes - mode_family, n_modes + mode_family))
        n_modes = np.sort(np.unique(n_modes))

    for n in range(N_min, N_max + 1):
        mode_str = "     "
        for m in range(56):
            if n in n_modes:
                if m <= M_max:
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
    f.write("      1   4500  {:10.3e}  1.000E-04      0      0\n".format(al0))
    f.write("C\n")

    f.close()


def _pool_fun(k, path, exec, al0, timeout):
    """Run TERPSICHORE and read output for equilibrium with index k."""
    idx_path = os.path.join(path, str(k))
    fort16_path = os.path.join(idx_path, "fort.16")

    try:
        _run_terps(dir=idx_path, exec=exec, timeout=timeout)
        output = _read_terps_output(path=fort16_path)
    except OSError:
        warnif(
            True,
            UserWarning,
            "TERPSICHORE fort.16 output file not found, using default growth rate.",
        )
        output = abs(al0)
    except RuntimeError:
        warnif(
            True,
            UserWarning,
            "TERPSICHORE execution failed, using default growth rate.",
        )
        output = abs(al0)
    except subprocess.TimeoutExpired:
        warnif(
            True,
            UserWarning,
            f"TERPSICHORE execution timed out after {timeout:.0f} seconds, "
            + "using default growth rate.",
        )
        output = abs(al0)  # default value if a growth rate was not found

    return np.atleast_1d(output)


def _run_terps(dir, exec, timeout):
    """Run TERPSICHORE."""
    stdout_path = os.path.join(dir, "stdout.terps")
    stderr_path = os.path.join(dir, "stderr.terps")

    fout = open(stdout_path, "w")
    ferr = open(stderr_path, "w")

    cmd = "./" + exec + " < input wout.nc"
    subprocess.run(cmd, cwd=dir, shell=True, timeout=timeout, stdout=fout, stderr=ferr)

    fout.close()
    ferr.close()


def _read_terps_output(path):
    """Read TERPSICHORE output file and return the growth rate."""
    errorif(
        not os.path.exists(path),
        OSError,
        "TERPSICHORE fort.16 output file not found!",
    )

    file = open(path)
    growth_rate = []
    for line in file:
        index = line.find("GROWTH RATE")
        if index != -1:
            growth_rate.append(float(line.strip().split("=")[1]))
            continue
    file.close()

    errorif(len(growth_rate) != 1, RuntimeError, "TERPSICHORE execution failed!")
    return growth_rate[0]


class TERPSICHORE(ExternalObjective):
    """Computes ideal MHD linear stability from calls to the code TERPSICHORE.

    Returns the linear growth rate of the fastest growing instability, normalized by the
    Alfven frequency. A negative growth rate denotes stability and a positive growth
    rate denotes instability.

    TERPSICHORE reference: https://doi.org/10.1007/978-1-4613-0659-7_8

    TERPSICHORE documentation:
    https://princetonuniversity.github.io/STELLOPT/TERPSICHORE.html

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    abs_step : float, optional
        Absolute finite difference step size. Default = 1e-4.
        Total step size is ``abs_step + rel_step * mean(abs(x))``.
    rel_step : float, optional
        Relative finite difference step size. Default = 0.
        Total step size is ``abs_step + rel_step * mean(abs(x))``.
    processes : int, optional
        Maximum number of CPU threads to use for multiprocessing. Default = 1.
    path : str
        Path to the directory where temporary files will be stored.
    exec : str
        File name of the TERPSICHORE executable. Must be located in the directory
        specified by ``path``.
    mode_family : int, optional
        The mode family of the instabilities to consider. The toroidal modes included
        in a mode family are n_i = i * eq.NFP ± k, where i = ..., -1, 0, +1, ... and
        k ∈ [0, mode_family]. Possible mode families are in the range [0, eq.NFP // 2].
        If mode_family < 0 then all mode families are considered. Default = -1.
    surfs : int, optional
        Number of surfaces to include in the equilibrium input. More surfaces provides
        more accuracy at the cost of longer compute times. Default = 101.
    M_nyq, N_nyq: int
        The max poloidal and toroidal mode numbers to use in the Nyquist spectrum of the
        equilibrium input. Defaults to ``eq.M + 4`` and ``eq.N + 2``.
    M_booz_max, N_booz_max : int, optional
        Maximum poloidal and toroidal mode numbers of Boozer spectrum. Will include
        modes with m ∈ [0, M_booz_max] and n ∈ [-N_booz_max, N_booz_max].
        Defaults to ``2 * eq.M`` and ``2 * eq.N``.
    M_max : int, optional
        Maximum poloidal mode number of stability modes to consider. Will include modes
        with m ∈ [0, M_max] (if ``mode_family < 0``). Default = 8.
    N_min : int, optional
        Minimum toroidal mode number of stability modes to consider. Will include modes
        with n ∈ [N_min, N_max] (if ``mode_family < 0``). Default = -4.
    N_max : int, optional
        Maximum toroidal mode number of stability modes to consider. Will include modes
        with n ∈ [N_min, N_max] (if ``mode_family < 0``). Default = 4.
    lssl : int, optional
        Minimum number of possible permutations of Boozer mode combinations
        (determined by ``M_booz_max`` and ``N_booz_max``). If TERPSICHORE fails to run,
        try increasing this parameter. Default = ``20 * M_booz_max * N_booz_max``.
    lssd : int, optional
        Minimum number of possible permutations of stability mode combinations
        (determined by ``M_max`` and ``N_max``). If TERPSICHORE fails to run,
        try increasing this parameter. Default = ``20 * M_max * N_max``.
    awall : float, optional
        Ratio of the radius of the conformal conducting wall to the plasma minor radius.
        The conducting wall is obtained by scaling the m ≠ 0 Fourier components of the
        plasma boundary by ``awall``. A shorter wall offset will help stabilize the
        plasma. If TERPSICHORE fails to run, try decreasing this parameter. Default = 2.
    deltaJp : float
        Resonance detuning parameter to resolve parallel current density singularities.
        A larger value can artificially improve the stability. Default = 1e-4.
    modelk : int, optional
        0 = Noninteracting anisotropic fast particle stability model with reduced
        kinetic energy. 1 = Kruskal-Oberman anisotropic energy principle model with
        reduced kinetic energy. 2 = Noninteracting anisotropic fast particle stability
        model with physical kinetic energy. 3 = Kruskal-Oberman anisotropic energy
        principle model with physical kinetic energy. Default = 0.
    al0 : float, optional
        Initial guess of the eigenvalue. Use a sufficiently negative value to find the
        most unstable growth rate. If TERPSICHORE fails to run, the objective will
        return a growth rate of ``abs(al0)``. Default = -0.5.
    timeout : float, optional
        Time in seconds to wait for TERPSICHORE to execute before terminating its run.
        See ``timeout`` argument of ``subprocess.run`` for more info. Default = 60.
    tmp_dir : str, optional
        Name of directory where temporary files used and generated by TERPSICHORE are
        stored. This will be a sub-directory within the directory specified by ``path``.
        Default = "tmp_TERPS".
    save_tmp : bool, optional
        If True, ``tmp_dir`` is preserved to record the TERPSICHORE I/O files.
        It will contain sub-directories named by a timestamp for each time the function
        was evaluated. Those contain another layer of sub-directories named by integers
        ranging from 0 to ``len(eq)`` with the files used in each TERPSICHORE call.
        If False, ``tmp_dir`` and all of its contents are removed at the end of the
        function evaluation. Default = False.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``bounds=(-np.inf, 0)``",
        bounds_default="``bounds=(-np.inf, 0)``",
    )

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "TERPSICHORE growth rate: "

    def __init__(
        self,
        eq,
        *,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        abs_step=1e-4,
        rel_step=0,
        processes=1,
        path,
        exec,
        mode_family=-1,
        surfs=101,
        M_nyq=None,
        N_nyq=None,
        M_booz_max=None,
        N_booz_max=None,
        M_max=8,
        N_min=-4,
        N_max=4,
        lssl=None,
        lssd=None,
        awall=2.0,
        deltaJp=1e-4,
        modelk=0,
        al0=-5e-1,
        timeout=60,
        tmp_dir="tmp_TERPS",
        save_tmp=False,
        name="terpsichore",
    ):
        if target is None and bounds is None:
            bounds = (-np.inf, 0)
        super().__init__(
            eq=eq,
            fun=terpsichore,
            dim_f=1,
            fun_kwargs={
                "processes": processes,
                "path": path,
                "exec": exec,
                "mode_family": mode_family,
                "surfs": surfs,
                "M_nyq": M_nyq,
                "N_nyq": N_nyq,
                "M_booz_max": M_booz_max,
                "N_booz_max": N_booz_max,
                "M_max": M_max,
                "N_min": N_min,
                "N_max": N_max,
                "lssl": lssl,
                "lssd": lssd,
                "awall": awall,
                "deltaJp": deltaJp,
                "modelk": modelk,
                "al0": al0,
                "timeout": timeout,
                "tmp_dir": tmp_dir,
                "save_tmp": save_tmp,
            },
            vectorized=True,
            abs_step=abs_step,
            rel_step=rel_step,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
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
        errorif(
            self._fun_kwargs["mode_family"] > self._eq.NFP // 2,
            ValueError,
            "Invalid mode family, must be <= eq.NFP // 2.",
        )

        # check if theta needs to be flipped so that theta=0 is on the outboard midplane
        grid0 = LinearGrid(rho=0.0, M=0, N=0)
        grid1 = LinearGrid(rho=1.0, M=0, N=0)
        R0 = self._eq.compute("R", grid0)["R"][0]  # R(rho=0)
        R1 = self._eq.compute("R", grid1)["R"][0]  # R(rho=1,theta=0,phi=0)
        self._fun_kwargs["theta0_outboard"] = bool(R1 > R0)

        # transforms for writing the wout file
        surfs = self._fun_kwargs.get("surfs")
        NFP = self._eq.NFP
        M = self._eq.M
        N = self._eq.N
        M_nyq = self._fun_kwargs.get("M_nyq")
        N_nyq = self._fun_kwargs.get("N_nyq")
        M_nyq = M + 4 if M_nyq is None else M_nyq
        N_nyq = N + 2 if N_nyq is None else N_nyq
        N_nyq = 0 if int(N) == 0 else N_nyq
        M_grid = self._eq.M_grid
        N_grid = self._eq.N_grid
        s_full = np.linspace(0, 1, surfs)
        s_half = s_full[0:-1] + 0.5 / (surfs - 1)
        r_half = np.sqrt(s_half)
        grid_lcfs = LinearGrid(M=M_grid, N=N_grid, rho=np.array([1.0]), NFP=NFP)
        grid_half = LinearGrid(M=M_grid, N=N_grid, NFP=NFP, rho=r_half)
        self._fun_kwargs["data_transforms"] = get_transforms(
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
        self._fun_kwargs["fit_transform"] = Transform(
            grid=grid_lcfs, basis=fit_basis, build=False, build_pinv=True
        )

        super().build(use_jit=use_jit, verbose=verbose)
