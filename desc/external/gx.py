"""Functions to run the GX turbulence code on a DESC equilibrium."""

import datetime
import os
import re
import shutil
import subprocess
from collections import deque

import numpy as np
from interpax import interp1d
from quadax import cumulative_trapezoid
from scipy.constants import mu_0

from desc.backend import jnp
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import Grid, LinearGrid
from desc.objectives import ExternalObjective
from desc.objectives.objective_funs import collect_docs
from desc.utils import cross, dot, errorif, warnif

# Keys computed on the constant equilibrium grid
_EQ_KEYS = ["iota", "iota_r", "a", "R0", "p_r"]
# Keys computed on the field-line grid
_FL_KEYS = [
    "|B|",
    "|grad(psi)|^2",
    "grad(|B|)",
    "grad(alpha)",
    "grad(psi)",
    "B",
    "B^theta",
    "B^zeta",
    "lambda_t",
    "lambda_z",
    "iota_r",
    "e^rho",
]


def compute_gx_geometry(
    eq,
    psi,
    params=None,
    alpha=0.0,
    npol=1,
    nzgrid=32,
    sigma_Bxy=-1.0,
    eq_transforms=None,
    profiles=None,
):
    """Compute flux-tube geometry for the GX gyrokinetic turbulence code.

    Computes all geometric coefficients needed by GX on a single flux tube,
    interpolated onto a uniform arc-length parallel coordinate z in [-pi, pi].

    Supply ``params``, ``eq_transforms`` and ``profiles`` for proper AD and JIT.

    Parameters
    ----------
    eq : Equilibrium
        DESC Equilibrium object.
    psi : float
        Normalized toroidal flux surface label (= rho^2), in (0, 1).
    params : dict
        Equilibrium parameter dictionary. If None, retrieved internally from
        ``eq.params_dict``.
    alpha : float, optional
        Field line label, defined as alpha = theta_PEST - iota * zeta at the
        tube center. Default = 0.
    npol : int, optional
        Number of poloidal turns (divided by 2*pi) that the flux tube travels.
        Default = 1.
    nzgrid : int, optional
        Number of grid points along the field line in each direction from center.
        Total points = 2 * nzgrid + 1. Default = 32.
    sigma_Bxy : float, optional
        Sign convention parameter: (1/|B|^2) B . (grad x cross grad y). Usually -1
        for stellarators, +1 for tokamaks. This controls the orientation of the GX
        (x, y, z) coordinate system and is written as ``kxfac`` in the geometry file.
        Default = -1.
    eq_transforms : dict, optional
        Pre-computed transforms for the flux-surface equilibrium grid.
        Built from ``get_transforms`` on a ``LinearGrid`` at ``rho = sqrt(psi)``.
        If None, built internally.
    profiles : dict, optional
        Equilibrium profiles (iota, pressure, current, etc.). If None, retrieved
        internally.

    Returns
    -------
    geo : dict
        Dictionary containing all GX geometric quantities on a uniform
        arc-length grid. Keys include:

        - ``"z"`` : ndarray, shape (2*nzgrid+1,) -- parallel coordinate, [-pi, pi]
        - ``"bmag"`` : ndarray -- |B| / B_ref
        - ``"gradpar"`` : ndarray -- constant parallel gradient (2*pi / L)
        - ``"grho"`` : ndarray -- |grad(rho)| * L_ref
        - ``"gds2"`` : ndarray -- |grad(alpha)|^2 * L_ref^2 * s
        - ``"gds21"`` : ndarray -- sigma_Bxy * grad(alpha).grad(psi) * shat / B_ref
        - ``"gds22"`` : ndarray -- |grad(psi)|^2 * shat^2 / (L_ref^2 * B_ref^2 * s)
        - ``"gbdrift"`` : ndarray -- grad-B drift coefficient
        - ``"gbdrift0"`` : ndarray -- grad-B drift (psi-component, shear-free)
        - ``"cvdrift"`` : ndarray -- curvature drift coefficient
        - ``"cvdrift0"`` : ndarray -- curvature drift (psi-component, shear-free)
        - ``"L_ref"`` : float -- reference length (minor radius a)
        - ``"B_ref"`` : float -- reference field (2 * |psi_edge| / a^2)
        - ``"R0"`` : float -- major radius
        - ``"shat"`` : float -- magnetic shear, -(rho/iota) * d(iota)/d(rho)
        - ``"sigma_Bxy"`` : float -- sign convention (= kxfac)
        - ``"iota"`` : float -- rotational transform at the surface
        - ``"nzgrid"`` : int -- half-grid size
        - ``"npol"`` : int -- number of poloidal turns

    """
    errorif(
        psi <= 0 or psi >= 1,
        ValueError,
        f"psi must be in (0, 1), got {psi}.",
    )
    rho = jnp.sqrt(psi)

    if params is None:
        params = eq.params_dict
    if eq_transforms is None:
        grid_eq = LinearGrid(rho=float(rho), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        eq_transforms = get_transforms(_EQ_KEYS, obj=eq, grid=grid_eq)
    if profiles is None:
        profiles = get_profiles(_EQ_KEYS + _FL_KEYS, obj=eq, grid=eq_transforms["grid"])

    data_eq = compute_fun(eq, _EQ_KEYS, params, eq_transforms, profiles)
    grid_eq = eq_transforms["grid"]
    iota_val = grid_eq.compress(data_eq["iota"])[0]
    iota_r_val = grid_eq.compress(data_eq["iota_r"])[0]
    p_r_val = grid_eq.compress(data_eq["p_r"])[0]
    d_pressure_d_s = p_r_val / (2 * rho)  # dp/ds where s = rho^2

    # Reference quantities for GX normalizations
    L_ref = data_eq["a"]
    R0 = data_eq["R0"]
    edge_toroidal_flux_over_2pi = params["Psi"] / (2 * jnp.pi)
    B_ref = 2 * jnp.abs(edge_toroidal_flux_over_2pi) / L_ref**2
    sign_psi = jnp.sign(edge_toroidal_flux_over_2pi)
    shat = -rho / iota_val * iota_r_val

    # Construct field-line coordinates
    # The field line is parameterized in theta_PEST, spanning npol poloidal turns.
    nl = 2 * nzgrid + 1
    theta_pest = jnp.linspace(alpha - jnp.pi * npol, alpha + jnp.pi * npol, nl)
    zeta_center = -alpha / iota_val
    zeta = zeta_center + (theta_pest - alpha) / iota_val

    # Map from (rho, theta_PEST, zeta) to (rho, theta, zeta) coords
    rhoa = rho * jnp.ones(nl)
    nodes = jnp.vstack([rhoa, theta_pest, zeta]).T
    coords = eq.map_coordinates(
        nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        params=params,
        tol=1e-10,
        maxiter=50,
    )
    grid = Grid(
        coords,
        coordinates="rtz",
        sort=False,
        jitable=True,
        # we compute surface integral quantities on proper grid before
        spacing=jnp.zeros_like(coords).astype(float),
        weights=jnp.zeros_like(coords[:, 0]).astype(float),
    )
    # Copy flux-surface quantities from the equilibrium grid to the field-line grid
    # they cannot be computed directly on the field-line grid since they require
    # surface integrals.
    data = {
        key: grid.copy_data_from_other(data_eq[key], grid_eq)
        for key in ["iota", "iota_r", "p_r"]
    }

    # Compute field-line quantities
    fl_transforms = get_transforms(_FL_KEYS, obj=eq, grid=grid, jitable=True)
    data = compute_fun(eq, _FL_KEYS, params, fl_transforms, profiles, data=data)

    modB = data["|B|"]
    bmag = modB / B_ref

    # gradpar in theta_PEST parameterization:
    # gradpar = L_ref * B . grad(theta_PEST) / |B|
    #         = L_ref * (B^theta * (1 + lambda_t) + B^zeta * lambda_z) / |B|
    gradpar = (
        L_ref
        * (data["B^theta"] * (1 + data["lambda_t"]) + data["B^zeta"] * data["lambda_z"])
        / modB
    )

    # DESC's grad(alpha) uses alpha = theta_PEST - iota * zeta (no zeta_center shift).
    # For GX we want alpha' = theta_PEST - iota * (zeta - zeta_center), so we shift:
    # grad(alpha') = grad(alpha) + zeta_center * d(iota)/d(rho) * e^rho
    grad_alpha = (
        data["grad(alpha)"]
        + zeta_center * data["iota_r"][:, jnp.newaxis] * data["e^rho"]
    )

    grad_psi = data["grad(psi)"]
    grad_psi_sq = data["|grad(psi)|^2"]

    # Metric-related quantities
    grad_alpha_dot_grad_alpha = dot(grad_alpha, grad_alpha)
    grad_alpha_dot_grad_psi = dot(grad_alpha, grad_psi)

    grho = jnp.sqrt(grad_psi_sq / (L_ref**2 * B_ref**2 * psi))
    gds2 = grad_alpha_dot_grad_alpha * L_ref**2 * psi
    gds21 = sigma_Bxy * grad_alpha_dot_grad_psi * shat / B_ref
    gds22 = grad_psi_sq * shat**2 / (L_ref**2 * B_ref**2 * psi)

    # Drift-related quantities
    B_vec = data["B"]
    grad_modB = data["grad(|B|)"]

    B_cross_grad_B_dot_grad_alpha = dot(cross(B_vec, grad_modB), grad_alpha)
    B_cross_grad_B_dot_grad_psi = dot(cross(B_vec, grad_modB), grad_psi)

    gbdrift = (
        2
        * sigma_Bxy
        * sign_psi
        * B_ref
        * L_ref**2
        * jnp.sqrt(psi)
        * B_cross_grad_B_dot_grad_alpha
        / modB**3
    )

    # Pressure correction to curvature drift.
    # Uses the Clebsch identity: (B x nabla_s) . nabla_alpha = |B|^2 / psi_edge,
    # which simplifies the general expression to a 1/|B|^2 dependence.
    cvdrift = gbdrift + (
        2
        * mu_0
        * sigma_Bxy
        * sign_psi
        * B_ref
        * L_ref**2
        * jnp.sqrt(psi)
        * d_pressure_d_s
        / (edge_toroidal_flux_over_2pi * modB**2)
    )

    gbdrift0 = (
        2 * sign_psi * shat * B_cross_grad_B_dot_grad_psi / (modB**3 * jnp.sqrt(psi))
    )
    cvdrift0 = gbdrift0

    # --- Interpolate to uniform arc-length grid ---
    geo_raw = {
        "bmag": bmag,
        "grho": grho,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "gbdrift": gbdrift,
        "gbdrift0": gbdrift0,
        "cvdrift": cvdrift,
        "cvdrift0": cvdrift0,
    }
    geo = _interpolate_to_uniform_grid(theta_pest, gradpar, geo_raw, nzgrid)
    geo.update(
        {
            "L_ref": L_ref,
            "B_ref": B_ref,
            "R0": R0,
            "shat": shat,
            "sigma_Bxy": sigma_Bxy,
            "iota": iota_val,
            "nzgrid": nzgrid,
            "npol": npol,
        }
    )
    return geo


def _interpolate_to_uniform_grid(theta_pest, gradpar, geo_arrays, nzgrid):
    """Interpolate geometric quantities to a uniform arc-length grid.

    GX requires geometric coefficients on a uniformly-spaced parallel coordinate
    z in [-pi, pi]. This function:
    1. Integrates the physical arc-length along the field line using the trapezoidal
       rule on 1/|gradpar|.
    2. Rescales the arc-length to [-pi, pi].
    3. Cubic-interpolates all quantities onto a uniform z grid .

    Parameters
    ----------
    theta_pest : jnp.ndarray
        PEST theta coordinates along the field line.
    gradpar : jnp.ndarray
        Parallel gradient dl/dtheta_pest (normalized by L_ref).
    geo_arrays : dict of jnp.ndarray
        Field-line-varying geometric quantities to interpolate.
    nzgrid : int
        Half-grid size. Total grid points = 2 * nzgrid + 1.

    Returns
    -------
    result : dict
        Interpolated quantities plus ``"z"`` (uniform grid) and ``"gradpar"``
        (constant value).
    """
    nl = 2 * nzgrid + 1

    # Integrate arc-length: dl = d(theta_pest) / |gradpar|
    arc_length = cumulative_trapezoid(1.0 / jnp.abs(gradpar), x=theta_pest, initial=0.0)

    # Shift so center (index nzgrid) is at 0, then rescale to [-pi, pi]
    arc_length = arc_length - arc_length[nzgrid]
    L_total = arc_length[-1] - arc_length[0]
    gradpar_uniform_val = 2.0 * jnp.pi / L_total
    arc_length_scaled = arc_length * (2.0 * jnp.pi / L_total)

    uniform_z = jnp.linspace(-jnp.pi, jnp.pi, nl)

    result = {
        k: interp1d(uniform_z, arc_length_scaled, v, method="cubic")
        for k, v in geo_arrays.items()
    }
    result["z"] = uniform_z
    result["gradpar"] = jnp.full(nl, gradpar_uniform_val)
    return result


def write_gx_geometry(geo, path):
    """Write GX geometry to an eik file in plain-text format.

    Writes the geometry dictionary (as returned by ``compute_gx_geometry``)
    to a file that GX can read via ``geo_option = "eik"`` and ``geo_file``.

    Parameters
    ----------
    geo : dict
        Geometry dictionary as returned by ``compute_gx_geometry``.
    path : str
        Output file path.

    """
    nzgrid = int(geo["nzgrid"])
    nperiod = 1
    ntheta = 2 * nzgrid
    kxfac = float(geo["sigma_Bxy"])
    drhodpsi = 1.0
    rmaj = float(geo["R0"])
    shat = float(geo["shat"])
    q = float(1.0 / geo["iota"])
    scale = 1.0
    z = np.asarray(geo["z"])
    nl = len(z)
    # Convert jax arrays to numpy for file I/O.
    _g = {
        k: np.asarray(geo[k])
        for k in [
            "gbdrift",
            "gradpar",
            "grho",
            "cvdrift",
            "gds2",
            "bmag",
            "gds21",
            "gds22",
            "cvdrift0",
            "gbdrift0",
        ]
    }

    with open(path, "w") as f:
        # header
        f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale\n")
        f.write(
            f"{nzgrid} {nperiod} {ntheta} {drhodpsi} "
            f"{rmaj} {shat} {kxfac} {q} {scale}\n"
        )

        # section 1: gbdrift gradpar grho tgrid
        f.write("gbdrift gradpar grho tgrid\n")
        for i in range(nl):
            f.write(
                f"{_g['gbdrift'][i]:23} {_g['gradpar'][i]:23} "
                f"{_g['grho'][i]:23} {z[i]:23}\n"
            )

        # section 2: cvdrift gds2 bmag tgrid
        f.write("cvdrift gds2 bmag tgrid\n")
        for i in range(nl):
            f.write(
                f"{_g['cvdrift'][i]:23} {_g['gds2'][i]:23} "
                f"{_g['bmag'][i]:23} {z[i]:23}\n"
            )

        # section 3: gds21 gds22 tgrid
        f.write("gds21 gds22 tgrid\n")
        for i in range(nl):
            f.write(f"{_g['gds21'][i]:23} {_g['gds22'][i]:23} {z[i]:23}\n")

        # section 4: cvdrift0 gbdrift0 tgrid
        f.write("cvdrift0 gbdrift0 tgrid\n")
        for i in range(nl):
            f.write(f"{_g['cvdrift0'][i]:23} {_g['gbdrift0'][i]:23} {z[i]:23}\n")


# ---------------------------------------------------------------------------
# GX driver function and ExternalObjective wrapper
# ---------------------------------------------------------------------------


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
    sigma_Bxy=-1.0,
    gx_input_file=None,
    launch_cmd=None,
    gx_gpu=None,
    timeout=600,
    tmp_dir="tmp_GX",
    save_tmp=False,
    eq_transforms=None,
    profiles=None,
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
    sigma_Bxy : float, optional
        Sign convention parameter: (1/|B|^2) B . (grad x cross grad y). Usually -1
        for stellarators. This controls the orientation of the GX coordinate system
        and is written as ``kxfac`` in the geometry file. Default = -1.
    gx_input_file : str
        Path to a template GX TOML input file that specifies physics parameters
        (species, domain, time stepping, etc.). The ``geo_file`` path in this
        template will be replaced with the generated geometry file, and
        ``geo_option`` will be set to ``"eik"`` to match DESC's plain-text
        geometry output.
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
    eq_transforms : dict, optional
        Pre-computed transforms for flux-surface quantities. Built in
        ``GX.build()`` and reused across iterations.
    profiles : dict, optional
        Pre-computed profiles. Built in ``GX.build()`` and reused across iterations.

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

        geo = compute_gx_geometry(
            eq=eq[k],
            psi=psi,
            params=eq[k].params_dict,
            alpha=alpha,
            npol=npol,
            nzgrid=nzgrid,
            sigma_Bxy=sigma_Bxy,
            eq_transforms=eq_transforms,
            profiles=profiles,
        )
        geo_path = os.path.join(idx_path, "gx_geo.out")
        write_gx_geometry(geo=geo, path=geo_path)

        input_path = os.path.join(idx_path, "gx.in")
        _write_gx_input(
            template_path=gx_input_file,
            # new input file path (GX will read this)
            # geo_file path in this input file will be replaced with geo_path
            output_path=input_path,
            geo_path=geo_path,
        )

        # run GX
        exec_path = os.path.join(path, exec)
        try:
            output_nc = _run_gx(
                dir=idx_path,
                exec_path=exec_path,
                input_path=input_path,
                launch_cmd=launch_cmd,
                gx_gpu=gx_gpu,
                timeout=timeout,
            )
            qflux = _read_gx_output(output_nc)
        except (OSError, RuntimeError) as e:
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


# ---------------------------------------------------------------------------
# Internal helpers for GX I/O and execution
# ---------------------------------------------------------------------------


def _write_gx_input(template_path, output_path, geo_path):
    """Write GX input file by updating the geometry file path in a template."""
    with open(template_path) as f:
        data = f.read()

    # DESC writes GX geometry in the plain-text eik format, so normalize the
    # generated input even if the template was copied from an NC-based workflow.
    data, geo_option_subs = re.subn(
        r"(^\s*geo_option\s*=\s*)(['\"])[^'\"]*\2",
        r'\1"eik"',
        data,
        count=1,
        flags=re.MULTILINE,
    )
    if geo_option_subs == 0:
        data, geo_option_subs = re.subn(
            r"(^\s*geo_option\s*=\s*)(\S+)",
            r'\1"eik"',
            data,
            count=1,
            flags=re.MULTILINE,
        )
    if geo_option_subs == 0:
        data, geo_option_subs = re.subn(
            r"(^\s*geo_file\s*=.*$)",
            'geo_option = "eik"\n\\1',
            data,
            count=1,
            flags=re.MULTILINE,
        )

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


def _tail_file(path, max_lines=10):
    """Return the tail of a text file, skipping blank lines."""
    if not os.path.exists(path):
        return ""

    tail = deque(maxlen=max_lines)
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if line:
                tail.append(line)
    return "\n".join(tail)


def _format_subprocess_failure(cmd, stdout_path, stderr_path, *, returncode=None, timeout=None):
    """Build a concise GX subprocess failure message with log context."""
    parts = [f"Command failed: {' '.join(cmd)}"]
    if returncode is not None:
        parts.append(f"exit status: {returncode}")
    if timeout is not None:
        parts.append(f"timed out after {timeout:.0f} seconds")

    stdout_tail = _tail_file(stdout_path)
    if stdout_tail:
        parts.append(f"stdout tail:\n{stdout_tail}")

    stderr_tail = _tail_file(stderr_path)
    if stderr_tail:
        parts.append(f"stderr tail:\n{stderr_tail}")

    parts.append(f"logs: stdout={stdout_path}, stderr={stderr_path}")
    return "\n".join(parts)


def _run_gx(dir, exec_path, input_path=None, launch_cmd=None, gx_gpu=None, timeout=300):
    """Run the GX executable.

    Parameters
    ----------
    dir : str
        Working directory for GX execution (output files written here).
    exec_path : str
        Full path to the GX executable. Called in-place so that GX can find
        any internal dependencies (shared libraries, etc.) relative to its
        install location.
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
        try:
            subprocess.run(
                cmd,
                cwd=dir,
                timeout=timeout,
                stdout=fout,
                stderr=ferr,
                check=True,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                _format_subprocess_failure(
                    cmd,
                    stdout_path,
                    stderr_path,
                    returncode=e.returncode,
                )
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                _format_subprocess_failure(
                    cmd,
                    stdout_path,
                    stderr_path,
                    timeout=timeout,
                )
            ) from e

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


# ---------------------------------------------------------------------------
# GX ExternalObjective wrapper
# ---------------------------------------------------------------------------


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
    sigma_Bxy : float, optional
        Sign convention parameter: (1/|B|^2) B . (grad x cross grad y). Usually -1
        for stellarators. Default = -1.
    gx_input_file : str
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
        sigma_Bxy=-1.0,
        gx_input_file,
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
                "sigma_Bxy": sigma_Bxy,
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
        eq = self._eq

        # Pre-compute transforms for flux-surface quantities (constant grid).
        rho = np.sqrt(psi)
        grid_eq = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        self._fun_kwargs["eq_transforms"] = get_transforms(
            _EQ_KEYS, obj=eq, grid=grid_eq
        )
        # Pre-compute profiles for all keys (profiles don't depend on grid).
        self._fun_kwargs["profiles"] = get_profiles(
            _EQ_KEYS + _FL_KEYS, obj=eq, grid=grid_eq
        )

        super().build(use_jit=use_jit, verbose=verbose)
