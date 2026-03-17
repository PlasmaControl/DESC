"""Functions to run the GX turbulence code on a DESC equilibrium."""

import datetime
import os
import re
import shutil
import subprocess

import numpy as np
from scipy.interpolate import interp1d

from desc.backend import jnp
from desc.compute.utils import get_transforms
from desc.grid import Grid, QuadratureGrid
from desc.objectives import ExternalObjective
from desc.objectives.objective_funs import collect_docs
from desc.utils import cross, dot, errorif, warnif


def gx(
    eq,
    *,
    processes=1,
    path,
    exec="gx",
    npol=1,
    nzgrid=32,
    alpha=0.0,
    psi=0.5,
    gx_input_file=None,
    launch_cmd=None,
    gx_gpu=None,
    timeout=600,
    tmp_dir="tmp_GX",
    save_tmp=False,
    data_transforms=None,
    eq_data_transforms=None,
):
    """GX driver function.

    Computes flux-tube geometry from the equilibrium, writes input files,
    runs GX, and returns the time-averaged nonlinear turbulent heat flux.

    Parameters
    ----------
    eq : list
        A list of Equilibrium objects to run GX on.
    processes : int, optional
        Maximum number of concurrent GX runs. Default = 1.
        Note: GX is a GPU code, so this is limited by the number of available GPUs.
    path : str
        Path to the directory where the GX executable is located and temporary files
        will be stored.
    exec : str, optional
        File name of the GX executable. Must be located in the directory specified by
        ``path``. Default = "gx".
    npol : int, optional
        Number of poloidal turns (divided by 2*pi) that the flux tube travels.
        Default = 1.
    nzgrid : int, optional
        Number of grid points along the field line in each direction from center.
        Total points = 2 * nzgrid + 1. Default = 32.
    alpha : float, optional
        Field line label. Default = 0.
    psi : float, optional
        Normalized toroidal flux surface label on which to simulate. Default = 0.5.
    gx_input_file : str, optional
        Path to a template GX input file. If provided, the geometry file path in this
        template will be replaced with the generated geometry file.
    launch_cmd : list of str, optional
        Command prefix for launching GX, e.g.
        ``["srun", "-N", "1", "--gpus-per-task=1"]`` for SLURM GPU allocation or
        ``["mpirun", "-np", "1"]``. If None, GX is launched directly.
    gx_gpu : int or str, optional
        GPU device index to use for GX (sets ``CUDA_VISIBLE_DEVICES`` in the
        subprocess environment). Use this to pin GX to a specific GPU when multiple
        GPUs are available. If None, inherits the parent environment. Ignored if
        ``launch_cmd`` is specified (in which case the launcher controls GPU binding).
    timeout : float, optional
        Time in seconds to wait for GX to execute before terminating. Default = 600.
    tmp_dir : str, optional
        Name of temporary directory for GX I/O files, created inside ``path``.
        Default = "tmp_GX".
    save_tmp : bool, optional
        If True, preserve temporary files. Default = False.
    data_transforms : dict, optional
        Pre-computed transforms for field-line data.
    eq_data_transforms : dict, optional
        Pre-computed transforms for global equilibrium data (iota, etc.).

    Returns
    -------
    result : ndarray
        The time-averaged nonlinear turbulent heat flux for each equilibrium.

    Notes
    -----
    GX is a GPU-accelerated code. When running DESC with JAX on the same GPU, JAX
    will by default pre-allocate most of the GPU memory, leaving none for GX.
    Recommended strategies to handle this:

    1. **DESC on CPU, GX on GPU (simplest).** Set ``JAX_PLATFORM_NAME=cpu`` before
       importing JAX. GX inherits the GPU. No special ``launch_cmd`` needed. The
       geometry computation in DESC is not the bottleneck, so this has minimal impact.

       .. code-block:: bash

           #!/bin/bash
           #SBATCH --gpus=1
           export JAX_PLATFORM_NAME=cpu
           python my_optimization.py

    2. **Separate GPUs via SLURM.** Request 2+ GPUs. Pin JAX to one GPU
       (``CUDA_VISIBLE_DEVICES=0``) and set ``gx_gpu=1`` to pin GX to the other,
       or use ``launch_cmd`` with appropriate SLURM flags.

       .. code-block:: bash

           #!/bin/bash
           #SBATCH --gpus=2
           export CUDA_VISIBLE_DEVICES=0  # JAX gets GPU 0
           python my_optimization.py      # GX uses gx_gpu=1

    3. **Shared GPU with memory limits.** Set ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
       and ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.1`` before importing JAX so it uses
       minimal GPU memory. GX can then allocate the remainder on the same GPU.

    """
    # create a temporary directory to store I/O files
    tmp_path = os.path.join(path, tmp_dir)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time_path = os.path.join(tmp_path, timestamp)
    os.mkdir(time_path)

    results = []
    for k in range(len(eq)):
        idx_path = os.path.join(time_path, str(k))
        os.mkdir(idx_path)

        # compute geometry
        geo = _compute_gx_geometry(
            eq=eq[k],
            npol=npol,
            nzgrid=nzgrid,
            alpha=alpha,
            psi=psi,
            data_transforms=data_transforms,
            eq_data_transforms=eq_data_transforms,
        )

        # interpolate to uniform grid
        geo_uniform = _interpolate_to_uniform_grid(geo, nzgrid)

        # write geometry file
        geo_path = os.path.join(idx_path, "gx_geo.out")
        _write_gx_geometry(path=geo_path, geo=geo_uniform, npol=npol)

        # write input file
        if gx_input_file is not None:
            input_path = os.path.join(idx_path, "gx.in")
            _write_gx_input(
                template_path=gx_input_file,
                output_path=input_path,
                geo_path=geo_path,
            )
        else:
            input_path = None

        # copy executable
        exec_src = os.path.join(path, exec)
        exec_dst = os.path.join(idx_path, exec)
        if os.path.exists(exec_src) and exec_src != exec_dst:
            shutil.copy(exec_src, exec_dst)

        # run GX
        try:
            output_nc = _run_gx(
                dir=idx_path,
                exec=exec,
                input_path=input_path,
                launch_cmd=launch_cmd,
                gx_gpu=gx_gpu,
                timeout=timeout,
            )
            qflux = _read_gx_output(output_nc)
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as e:
            warnif(
                True,
                UserWarning,
                f"GX execution failed for equilibrium {k}: {e}. "
                "Returning NaN for heat flux.",
            )
            qflux = np.nan
        results.append(qflux)

    # remove temporary directory
    if not save_tmp:
        shutil.rmtree(tmp_path)

    return jnp.atleast_1d(jnp.array(results, dtype=float))


def _compute_gx_geometry(
    eq, npol, nzgrid, alpha, psi, data_transforms=None, eq_data_transforms=None
):
    """Compute flux-tube geometric quantities needed by GX.

    Returns a dict with all geometric quantities on the field-line grid.
    """
    rho = np.sqrt(psi)

    # compute iota and shear on the equilibrium grid
    grid_eq = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data_eq = eq.compute(
        ["iota", "iota_r", "a", "psi", "rho"],
        grid=grid_eq,
        transforms=eq_data_transforms,
    )
    fi = interp1d(data_eq["rho"], data_eq["iota"], kind="cubic")
    fs = interp1d(data_eq["rho"], data_eq["iota_r"], kind="cubic")
    iota_val = float(fi(rho))
    iota_r_val = float(fs(rho))

    # construct field-line coordinates
    zeta = np.linspace(
        (-np.pi * npol - alpha) / np.abs(iota_val),
        (np.pi * npol - alpha) / np.abs(iota_val),
        2 * nzgrid + 1,
    )
    iota = iota_val * np.ones(len(zeta))
    shear_val = iota_r_val * np.ones(len(zeta))
    theta_sfl = iota_val / np.abs(iota_val) * alpha * np.ones(len(zeta)) + iota * zeta
    zeta_center = zeta[nzgrid]

    # map from field-line (rho, alpha, zeta) to DESC (rho, theta, zeta) coords
    rhoa = rho * np.ones(len(zeta))
    nodes = np.vstack([rhoa, theta_sfl, zeta]).T
    coords = eq.map_coordinates(
        nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        tol=1e-10,
        maxiter=50,
    )
    grid = Grid(coords, sort=False)

    # compute all needed quantities on the field-line grid
    field_line_keys = [
        "|B|",
        "|grad(psi)|^2",
        "grad(|B|)",
        "grad(psi)",
        "B",
        "kappa",
        "B^zeta",
        "lambda_t",
        "lambda_z",
        "lambda_r",
        "p_r",
        "g^rr",
        "g^rt",
        "g^rz",
        "g^tz",
        "g^tt",
        "g^zz",
        "e^rho",
        "e^theta",
        "e^zeta",
    ]
    data = eq.compute(field_line_keys, grid=grid, transforms=data_transforms)

    # reference quantities for GX normalizations
    psib = data_eq["psi"][-1]
    Lref = float(data_eq["a"])
    Bref = float(2 * np.abs(psib) / Lref**2)
    sign_psi = float(np.sign(psib))
    sign_iota = float(np.sign(iota_val))

    modB = np.array(data["|B|"])
    bmag = modB / Bref

    # shat = -rho/iota * d(iota)/d(rho) (GX convention)
    shat = float(-rho / iota_val * iota_r_val)

    # gradpar = Lref * B^zeta / |B|
    gradpar = Lref * np.array(data["B^zeta"]) / modB

    # grad_alpha components in DESC coordinates
    lmbda_r = np.array(data["lambda_r"])
    lmbda_t = np.array(data["lambda_t"])
    lmbda_z = np.array(data["lambda_z"])
    grad_alpha_r = lmbda_r - (zeta - zeta_center) * shear_val
    grad_alpha_t = 1 + lmbda_t
    grad_alpha_z = -iota + lmbda_z

    # reconstruct grad_alpha vector
    grad_alpha = (
        grad_alpha_r * np.array(data["e^rho"]).T
        + grad_alpha_t * np.array(data["e^theta"]).T
        + grad_alpha_z * np.array(data["e^zeta"]).T
    ).T

    grad_psi = np.array(data["grad(psi)"])
    grad_psi_sq = np.array(data["|grad(psi)|^2"])

    # grho
    grho = np.sqrt(grad_psi_sq / (Lref**2 * Bref**2 * psi))

    # gds2 = |grad(alpha)|^2 * Lref^2 * psi
    gds2 = np.array(dot(grad_alpha, grad_alpha)) * Lref**2 * psi

    # gds21 = -sign(iota) * grad(psi) . grad(alpha) * shat / Bref # noqa: E800
    gds21 = -sign_iota * np.array(dot(grad_psi, grad_alpha)) * shat / Bref

    # gds22 = |grad(psi)|^2 / psi * (shat / (Lref * Bref))^2
    gds22 = grad_psi_sq / psi * (shat / (Lref * Bref)) ** 2

    # gbdrift = (B x grad|B|) . grad_alpha normalized
    B_vec = np.array(data["B"])
    grad_modB = np.array(data["grad(|B|)"])
    kappa = np.array(data["kappa"])

    gbdrift = np.array(dot(cross(B_vec, grad_modB), grad_alpha))
    gbdrift *= -sign_psi * 2 * Bref * Lref**2 / modB**3 * np.sqrt(psi)

    cvdrift = np.array(dot(cross(B_vec, kappa), grad_alpha))
    cvdrift *= -sign_psi * 2 * Bref * Lref**2 / modB**2 * np.sqrt(psi)

    gbdrift0 = np.array(dot(cross(B_vec, grad_modB), grad_psi))
    gbdrift0 *= sign_iota * sign_psi * shat * 2 / modB**3 / np.sqrt(psi)
    cvdrift0 = gbdrift0.copy()

    return {
        "zeta": zeta,
        "bmag": bmag,
        "grho": grho,
        "gradpar": gradpar,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "gbdrift": gbdrift,
        "gbdrift0": gbdrift0,
        "cvdrift": cvdrift,
        "cvdrift0": cvdrift0,
        "Lref": Lref,
        "Bref": Bref,
        "shat": shat,
        "iota": iota_val,
        "nzgrid": nzgrid,
    }


def _interpolate_to_uniform_grid(geo, nzgrid):
    """Interpolate geometric quantities from field-line grid to uniform grid.

    GX requires geometric coefficients on a uniformly-spaced parallel coordinate grid.
    This is constructed by reparametrizing the field line in terms of the parallel arc
    length and then interpolating to a uniform grid.
    """
    zeta = geo["zeta"]
    gradpar = np.copy(geo["gradpar"])
    dzeta = zeta[1] - zeta[0]

    # compute gradpar on half-grid for arc-length integration
    gradpar_half = np.zeros(2 * nzgrid)
    for i in range(2 * nzgrid - 1):
        gradpar_half[i] = 0.5 * (np.abs(gradpar[i]) + np.abs(gradpar[i + 1]))
    gradpar_half[2 * nzgrid - 1] = gradpar_half[0]

    # integrate to get arc-length coordinate
    arc_length = np.zeros(2 * nzgrid + 1)
    for i in range(2 * nzgrid):
        arc_length[i + 1] = arc_length[i] + dzeta / np.abs(gradpar_half[i])

    # shift so center is at 0
    arc_length -= arc_length[nzgrid]

    # rescale to [-pi, pi]
    desired_gradpar = np.pi / np.abs(arc_length[0])
    arc_length *= desired_gradpar
    gradpar_uniform = desired_gradpar * np.ones(2 * nzgrid + 1)

    # uniform grid in rescaled coordinate
    uniform_zgrid = np.linspace(arc_length[0], -arc_length[0], 2 * nzgrid + 1)

    # interpolate all quantities to uniform grid
    def interp_to_uniform(arr):
        f = interp1d(arc_length, arr, kind="cubic", fill_value="extrapolate")
        return f(uniform_zgrid)

    return {
        "zgrid": uniform_zgrid,
        "bmag": interp_to_uniform(geo["bmag"]),
        "grho": interp_to_uniform(geo["grho"]),
        "gradpar": gradpar_uniform,
        "gds2": interp_to_uniform(geo["gds2"]),
        "gds21": interp_to_uniform(geo["gds21"]),
        "gds22": interp_to_uniform(geo["gds22"]),
        "gbdrift": interp_to_uniform(geo["gbdrift"]),
        "gbdrift0": interp_to_uniform(geo["gbdrift0"]),
        "cvdrift": interp_to_uniform(geo["cvdrift"]),
        "cvdrift0": interp_to_uniform(geo["cvdrift0"]),
        "Lref": geo["Lref"],
        "Bref": geo["Bref"],
        "shat": geo["shat"],
        "iota": geo["iota"],
        "nzgrid": geo["nzgrid"],
    }


def _write_gx_geometry(path, geo, npol):
    """Write GX geometry file in GS2/GX format."""
    nzgrid = geo["nzgrid"]
    nperiod = 1
    kxfac = 1.0
    zgrid = geo["zgrid"]

    with open(path, "w") as f:
        # header
        f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale\n")
        f.write(
            f"{nzgrid} {nperiod} {2 * nzgrid} {1.0} "
            f"{1 / geo['Lref']} {geo['shat']} {kxfac} "
            f"{1 / geo['iota']} {2 * npol - 1}\n"
        )

        # section 1: gbdrift gradpar grho tgrid
        f.write("gbdrift gradpar grho tgrid\n")
        for i in range(len(zgrid)):
            f.write(
                f"{geo['gbdrift'][i]} {geo['gradpar'][i]} "
                f"{geo['grho'][i]} {zgrid[i]}\n"
            )

        # section 2: cvdrift gds2 bmag tgrid
        f.write("cvdrift gds2 bmag tgrid\n")
        for i in range(len(zgrid)):
            f.write(
                f"{geo['cvdrift'][i]} {geo['gds2'][i]} "
                f"{geo['bmag'][i]} {zgrid[i]}\n"
            )

        # section 3: gds21 gds22 tgrid
        f.write("gds21 gds22 tgrid\n")
        for i in range(len(zgrid)):
            f.write(f"{geo['gds21'][i]} {geo['gds22'][i]} {zgrid[i]}\n")

        # section 4: cvdrift0 gbdrift0 tgrid
        f.write("cvdrift0 gbdrift0 tgrid\n")
        for i in range(len(zgrid)):
            f.write(f"{geo['cvdrift0'][i]} {geo['gbdrift0'][i]} {zgrid[i]}\n")


def _write_gx_input(template_path, output_path, geo_path):
    """Write GX input file by updating the geometry file path in a template."""
    with open(template_path) as f:
        data = f.read()

    # replace any existing geo_file reference with the new path
    data = re.sub(
        r"(geo_file\s*=\s*)(['\"])[^'\"]*\2",
        rf"\1'{geo_path}'",
        data,
    )
    # also try without quotes
    data = re.sub(
        r"(geo_file\s*=\s*)(\S+)",
        rf"\1'{geo_path}'",
        data,
    )

    with open(output_path, "w") as f:
        f.write(data)


def _run_gx(dir, exec, input_path=None, launch_cmd=None, gx_gpu=None, timeout=300):
    """Run the GX executable.

    Parameters
    ----------
    dir : str
        Working directory for GX execution.
    exec : str
        Name of the GX executable (must be in ``dir``).
    input_path : str, optional
        Path to the GX input file.
    launch_cmd : list of str, optional
        Command prefix for launching GX (e.g. srun for SLURM GPU allocation).
    gx_gpu : int or str, optional
        GPU device index for GX. Sets CUDA_VISIBLE_DEVICES in the subprocess.
    timeout : float
        Maximum execution time in seconds.

    Returns
    -------
    str
        Path to the output NetCDF file.
    """
    stdout_path = os.path.join(dir, "stdout.gx")
    stderr_path = os.path.join(dir, "stderr.gx")

    exec_path = os.path.join(dir, exec)
    errorif(
        not os.path.exists(exec_path),
        OSError,
        f"GX executable not found at {exec_path}",
    )

    # build command: [launch_cmd...] executable [input_file]
    cmd = list(launch_cmd) if launch_cmd else []
    cmd.append(exec_path)
    if input_path is not None:
        cmd.append(input_path)

    # set up subprocess environment for GPU control
    env = os.environ.copy()
    if gx_gpu is not None and launch_cmd is None:
        env["CUDA_VISIBLE_DEVICES"] = str(gx_gpu)

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        subprocess.run(
            cmd,
            cwd=dir,
            timeout=timeout,
            stdout=fout,
            stderr=ferr,
            check=True,
            env=env,
        )

    # find the output netcdf file
    out_files = [f for f in os.listdir(dir) if f.endswith(".out.nc")]
    errorif(
        len(out_files) == 0,
        OSError,
        f"No GX output NetCDF file found in {dir}",
    )
    return os.path.join(dir, out_files[0])


def _read_gx_output(path):
    """Read GX output file and return the time-averaged heat flux."""
    from netCDF4 import Dataset

    errorif(
        not os.path.exists(path),
        OSError,
        f"GX output file not found: {path}",
    )

    ds = Dataset(path)
    try:
        qflux = np.array(ds["Diagnostics/HeatFlux_st"])
        # use second half of time series (after transient)
        qflux = qflux[len(qflux) // 2 :]
        qflux_avg = _weighted_birkhoff_average(qflux)
    finally:
        ds.close()

    return float(qflux_avg)


def _weighted_birkhoff_average(data):
    """Compute weighted Birkhoff average of a time series.

    Weights the middle of the series more heavily to reduce endpoint effects.
    """
    N = len(data)
    weights = np.zeros(N)
    for i in range(1, N - 1):
        weights[i] = np.exp(-1.0 / (i / N * (1 - i / N)))
    norm = np.sum(weights)
    if norm == 0:
        return np.mean(data)
    weights = weights.reshape((N, 1)) if data.ndim > 1 else weights
    return float(np.sum(weights * data / norm))


class GX(ExternalObjective):
    """Computes nonlinear turbulent heat flux from calls to the gyrokinetic code GX.

    Returns the time-averaged nonlinear turbulent heat flux computed by GX on a
    flux tube centered at the specified flux surface and field line.

    GX is a GPU-accelerated gyrokinetic turbulence code. The objective computes
    field-line geometry from the DESC equilibrium, writes GX input files, executes
    the GX solver, and reads back the heat flux from the output.

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
        Maximum number of concurrent GX runs. Default = 1.
    path : str
        Path to the directory where the GX executable is located and temporary files
        will be stored.
    exec : str, optional
        File name of the GX executable. Must be located in the directory specified by
        ``path``. Default = "gx".
    npol : int, optional
        Number of poloidal turns (divided by 2*pi) that the flux tube travels.
        Default = 1.
    nzgrid : int, optional
        Number of grid points along the field line in each direction from center.
        Total points = 2 * nzgrid + 1. Default = 32.
    alpha : float, optional
        Field line label. Default = 0.
    psi : float, optional
        Normalized toroidal flux surface label on which to simulate. Default = 0.5.
    gx_input_file : str, optional
        Path to a template GX input file. The geometry file path in this template
        will be replaced with the generated geometry file.
    launch_cmd : list of str, optional
        Command prefix for launching GX, e.g.
        ``["srun", "-N", "1", "--gpus-per-task=1"]`` for SLURM GPU allocation.
        If None, GX is launched directly. See notes on ``gx()`` for GPU memory
        management strategies.
    gx_gpu : int or str, optional
        GPU device index to use for GX. Sets ``CUDA_VISIBLE_DEVICES`` in the
        subprocess environment. If None, inherits the parent environment.
        Ignored if ``launch_cmd`` is specified.
    timeout : float, optional
        Time in seconds to wait for GX to execute before terminating. Default = 600.
    tmp_dir : str, optional
        Name of directory where temporary files used and generated by GX are
        stored. This will be a sub-directory within ``path``. Default = "tmp_GX".
    save_tmp : bool, optional
        If True, preserve temporary files to record the GX I/O. Default = False.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``",
        bounds_default="``target=0``",
    )

    _scalar = True
    _units = "(dimensionless)"
    _print_value_fmt = "GX heat flux: "

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
        exec="gx",
        npol=1,
        nzgrid=32,
        alpha=0.0,
        psi=0.5,
        gx_input_file=None,
        launch_cmd=None,
        gx_gpu=None,
        timeout=600,
        tmp_dir="tmp_GX",
        save_tmp=False,
        name="GX",
    ):
        if target is None and bounds is None:
            target = 0
        super().__init__(
            eq=eq,
            fun=gx,
            dim_f=1,
            fun_kwargs={
                "processes": processes,
                "path": path,
                "exec": exec,
                "npol": npol,
                "nzgrid": nzgrid,
                "alpha": alpha,
                "psi": psi,
                "gx_input_file": gx_input_file,
                "launch_cmd": launch_cmd,
                "gx_gpu": gx_gpu,
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

        Pre-computes transform matrices and grids that remain constant across
        optimization iterations.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        psi = self._fun_kwargs["psi"]
        errorif(
            psi <= 0 or psi >= 1,
            ValueError,
            f"psi must be in (0, 1), got {psi}.",
        )

        # pre-compute transforms for equilibrium-level data (iota, etc.)
        grid_eq = QuadratureGrid(
            L=self._eq.L_grid,
            M=self._eq.M_grid,
            N=self._eq.N_grid,
            NFP=self._eq.NFP,
        )
        self._fun_kwargs["eq_data_transforms"] = get_transforms(
            keys=["iota", "iota_r", "a", "psi", "rho"],
            obj=self._eq,
            grid=grid_eq,
        )

        super().build(use_jit=use_jit, verbose=verbose)
