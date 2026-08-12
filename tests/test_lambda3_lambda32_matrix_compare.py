"""Split-phase regression test comparing ``finite-n lambda3`` and ``lambda32``.

This file is intentionally written so the expensive comparison can be run in one
of several phases across separate Slurm jobs.  The central question is:

1. Does the reduced matrix reconstructed from saved ``lambda32`` block files
   match the reduced matrix seen by ``lambda3``?
2. Does that reconstructed matrix reproduce the same leading eigenvalue and
   eigenvector as ``lambda3`` at the same resolution?

Why the workflow is split
-------------------------
The three expensive tasks have different resource profiles:

- ``lambda32`` block assembly is the phase that is most natural to run on a GPU
  node and to save to disk block-by-block.
- ``lambda3`` forms the reduced matrix seen by the legacy path and then calls an
  eigensolver on that reduced matrix; this is often best run on a large-memory
  CPU node.
- The final comparison phase also benefits from CPU memory because it rebuilds
  the reduced ``A32`` matrix, performs chunked matrix comparisons, and can run a
  direct ``eigsh`` solve on the reconstructed matrix.

Artifacts shared between phases
-------------------------------
All phases communicate through ``AGNI_LAMBDA32_ARTIFACT_DIR``.  The main files
written there are:

- ``A3_keep.npy``: reduced matrix seen by ``lambda3``.
- ``w3.npy`` and ``v3.npy``: leading eigenvalue/eigenvector from ``lambda3``.
- ``lambda32_prefix.txt``: prefix identifying the saved ``lambda32`` block dump.
- ``A32_keep.npy``: reconstructed reduced matrix produced during compare mode.

Run modes
---------
``AGNI_LAMBDA32_MODE`` selects which phase to execute:

- ``lambda32``: build and save ``lambda32`` block files only.
- ``lambda3``: run ``lambda3`` and save the reduced matrix and leading eigenpair.
- ``compare``: rebuild ``A32`` from saved blocks and compare matrix/eigenpair.
- ``all``: run every phase in one process. This is mostly for debugging because
  it combines all expensive work into a single run.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from desc.backend import jax, jnp
from desc.compute import _stability as stability
from desc.diffmat_utils import DiffMat, fourier_diffmat, legendre_diffmat
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid
from desc.integrals.quad_utils import automorphism_staircase1, leggauss_lob


def _build_grid_diffmat(eq, n_rho, n_theta, n_zeta):
    """Build the shared grid and derivative operators for one resolution.

    This helper exists so ``lambda3`` and ``lambda32`` are forced to use the
    exact same radial remap, quadrature weights, Fourier/Legendre derivative
    matrices, and mapped DESC coordinates.  That keeps the test focused on the
    operator implementation rather than on mismatched discretizations.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium used to map from the testing grid to DESC coordinates.
    n_rho, n_theta, n_zeta : int
        Radial, poloidal, and toroidal resolutions used for both operators.

    Returns
    -------
    Grid, DiffMat
        Objects passed directly into ``eq.compute(...)`` for both ``lambda3``
        and ``lambda32`` so both paths use the same discretization.
    """
    x, _ = leggauss_lob(n_rho)
    rho = automorphism_staircase1(x, eps=1e-2, x_0=0.7, m_1=2.0, m_2=3.0)
    dx_f = jax.vmap(
        lambda x_val: jax.grad(automorphism_staircase1, argnums=0)(
            x_val, eps=1e-2, x_0=0.7, m_1=2.0, m_2=3.0
        )
    )
    d_rho, w_rho = legendre_diffmat(n_rho)
    d_rho = d_rho / (dx_f(x)[:, None])
    w_rho = w_rho * (dx_f(x)[:, None])

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    d_theta, w_theta = fourier_diffmat(n_theta)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / eq.NFP, n_zeta, endpoint=False)
    d_zeta, w_zeta = fourier_diffmat(n_zeta)
    d_zeta = d_zeta * eq.NFP
    w_zeta = w_zeta / eq.NFP

    diffmat = DiffMat(
        D_rho=d_rho,
        W_rho=jnp.diagonal(w_rho),
        D_theta=d_theta,
        W_theta=jnp.diagonal(w_theta),
        D_zeta=d_zeta,
        W_zeta=jnp.diagonal(w_zeta),
    )

    grid0 = LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=1, sym=False)
    nodes = jnp.reshape(
        grid0.meshgrid_reshape(grid0.nodes, order="rtz"), (n_rho * n_theta * n_zeta, 3)
    )
    rtz_nodes = eq.map_coordinates(
        nodes,
        inbasis=("rho", "theta_PEST", "zeta"),
        outbasis=("rho", "theta", "zeta"),
        period=(jnp.inf, 2 * jnp.pi, jnp.inf),
        tol=1e-12,
        maxiter=50,
    )
    return Grid(rtz_nodes), diffmat


def _run_lambda3_capture(eq, grid, diffmat, n_rho, n_theta, n_zeta, artifact_dir):
    """Run ``lambda3`` and persist the reduced matrix and leading eigenpair.

    The legacy ``lambda3`` path does not expose the reduced matrix as a direct
    return value, so this helper temporarily monkeypatches
    :func:`desc.compute._stability.eigsh` to intercept the matrix immediately
    before the solve.  The captured matrix and the leading eigenpair are then
    written into ``artifact_dir`` for later comparison.

    Parameters
    ----------
    eq, grid, diffmat
        Inputs passed directly into ``eq.compute("finite-n lambda3", ...)``.
    n_rho, n_theta, n_zeta : int
        Resolution used only for progress logging and expected keep-size.
    artifact_dir : pathlib.Path
        Shared directory where ``A3_keep.npy``, ``w3.npy``, and ``v3.npy`` are
        written for the compare phase.

    Notes
    -----
    ``stability.eigsh`` is monkeypatched only inside this function and restored
    immediately afterward. Solver behavior is intentionally unchanged because
    the original ``eigsh`` is still called with the original arguments.
    """
    old = stability.eigsh
    a3_path = artifact_dir / "A3_keep.npy"
    w3_path = artifact_dir / "w3.npy"
    v3_path = artifact_dir / "v3.npy"

    def _capture(A, *args, **kwargs):
        A = np.asarray(A)
        a3_mm = np.lib.format.open_memmap(
            str(a3_path), mode="w+", dtype=A.dtype, shape=A.shape
        )
        chunk = int(os.environ.get("AGNI_LAMBDA32_A3_DUMP_CHUNK", "512"))
        for i0 in range(0, A.shape[0], chunk):
            i1 = min(i0 + chunk, A.shape[0])
            a3_mm[i0:i1, :] = A[i0:i1, :]
        a3_mm.flush()
        del a3_mm

        w, v = old(A, *args, **kwargs)
        np.save(w3_path, np.asarray(w))
        vv = np.asarray(v[:, 0]) if np.asarray(v).ndim == 2 else np.asarray(v)
        np.save(v3_path, vv)
        return w, v

    stability.eigsh = _capture
    try:
        n_total = n_rho * n_theta * n_zeta
        n_keep = 3 * n_total - 2 * n_theta * n_zeta
        print(f"[lambda-compare] phase=lambda3 start n_keep={n_keep}", flush=True)
        eq.compute(
            "finite-n lambda3",
            grid=grid,
            diffmat=diffmat,
            incompressible=False,
            gamma=5.0 / 3.0,
            v_guess=np.ones(n_keep),
        )
    finally:
        stability.eigsh = old
    print(f"[lambda-compare] phase=lambda3 done artifacts={artifact_dir}", flush=True)


def _run_lambda32_dump(eq, grid, diffmat, n_rho, n_theta, n_zeta, artifact_dir):
    """Run ``lambda32`` in dump-only mode and record the block-file prefix.

    This phase is designed to be run independently from the final comparison.
    It asks DESC to assemble ``lambda32`` in matrix-dump mode, which writes the
    six block matrices and supporting arrays to disk rather than immediately
    reconstructing the dense reduced matrix in-memory.

    Parameters
    ----------
    eq, grid, diffmat
        Inputs passed directly into ``eq.compute("finite-n lambda32", ...)``.
    n_rho, n_theta, n_zeta : int
        Resolution used to locate the most recent dump matching the expected
        ``"{n_rho}x{n_theta}x{n_zeta}"`` shape tag.
    artifact_dir : pathlib.Path
        Shared directory that receives the dump files and
        ``lambda32_prefix.txt``.

    Output
    ------
    Writes ``lambda32_prefix.txt`` in ``artifact_dir`` so the compare stage
    knows exactly which block files to read.
    """
    base = os.environ.get("AGNI_LAMBDA32_DUMP_BASENAME", "lambda32_cmp")
    old_dump_dir = os.environ.get("AGNI_LAMBDA32_DUMP_DIR")
    old_dump_base = os.environ.get("AGNI_LAMBDA32_DUMP_BASENAME")
    old_progress = os.environ.get("AGNI_LAMBDA32_PROGRESS")
    try:
        os.environ["AGNI_LAMBDA32_DUMP_DIR"] = str(artifact_dir)
        os.environ["AGNI_LAMBDA32_DUMP_BASENAME"] = base
        os.environ["AGNI_LAMBDA32_PROGRESS"] = os.environ.get(
            "AGNI_LAMBDA32_PROGRESS", "0"
        )
        print("[lambda-compare] phase=lambda32_dump start", flush=True)
        eq.compute(
            "finite-n lambda32",
            grid=grid,
            diffmat=diffmat,
            incompressible=False,
            gamma=5.0 / 3.0,
            matrix_dump_only=True,
            memmap_blocks=True,
            keep_source_blocks=True,
            gpu_assembly=os.environ.get("AGNI_LAMBDA32_GPU_ASSEMBLY", "1").lower()
            not in {"", "0", "false", "no", "off"},
            gpu_chunk_size=int(os.environ.get("AGNI_LAMBDA32_GPU_CHUNK_SIZE", "8192")),
            gpu_chunk_rows=int(os.environ.get("AGNI_LAMBDA32_GPU_CHUNK_ROWS", "4000")),
            gpu_chunk_k=int(os.environ.get("AGNI_LAMBDA32_GPU_CHUNK_K", "8192")),
            node_chunk_size=int(os.environ.get("AGNI_LAMBDA32_NODE_CHUNK", "4000")),
            bc_rho_inner=True,
            bc_rho_outer=True,
        )
    finally:
        if old_dump_dir is None:
            os.environ.pop("AGNI_LAMBDA32_DUMP_DIR", None)
        else:
            os.environ["AGNI_LAMBDA32_DUMP_DIR"] = old_dump_dir
        if old_dump_base is None:
            os.environ.pop("AGNI_LAMBDA32_DUMP_BASENAME", None)
        else:
            os.environ["AGNI_LAMBDA32_DUMP_BASENAME"] = old_dump_base
        if old_progress is None:
            os.environ.pop("AGNI_LAMBDA32_PROGRESS", None)
        else:
            os.environ["AGNI_LAMBDA32_PROGRESS"] = old_progress

    shape_tag = f"{n_rho}x{n_theta}x{n_zeta}"
    keep_files = sorted(
        artifact_dir.glob(f"{base}_*_{shape_tag}_keep.npy"),
        key=lambda p: p.stat().st_mtime,
    )
    assert (
        keep_files
    ), f"No lambda32 dump files found for shape {shape_tag} in {artifact_dir}"
    prefix = str(keep_files[-1])[: -len("_keep.npy")]
    (artifact_dir / "lambda32_prefix.txt").write_text(prefix + "\n")
    print(f"[lambda-compare] phase=lambda32_dump done prefix={prefix}", flush=True)


def _compare_saved_outputs(artifact_dir, n_rho, n_theta, n_zeta):
    """Rebuild ``A32`` from disk and compare it against saved ``lambda3`` data.

    This is the CPU-heavy verification stage.  It reconstructs the reduced
    ``lambda32`` matrix from the saved block files and then compares:

    - reconstructed ``A32`` versus saved ``A3`` in row chunks,
    - the Rayleigh quotient of ``A32`` using the saved ``lambda3`` eigenvector,
    - the filtered ``(A v)/v - lambda`` residual requested during debugging,
    - and, optionally, a direct ``eigsh`` solve of ``A32`` for eigenvalue and
      eigenvector agreement.

    Parameters
    ----------
    artifact_dir : pathlib.Path
        Shared directory containing ``A3_keep.npy``, ``w3.npy``, ``v3.npy``,
        ``lambda32_prefix.txt``, and where ``A32_keep.npy`` is written.
    n_rho, n_theta, n_zeta : int
        Resolution used only for informative assertion messages and progress
        output.
    """
    a3_path = artifact_dir / "A3_keep.npy"
    w3_path = artifact_dir / "w3.npy"
    v3_path = artifact_dir / "v3.npy"
    prefix_file = artifact_dir / "lambda32_prefix.txt"

    assert a3_path.exists(), f"missing {a3_path}"
    assert w3_path.exists(), f"missing {w3_path}"
    assert v3_path.exists(), f"missing {v3_path}"
    assert prefix_file.exists(), f"missing {prefix_file}"

    prefix = prefix_file.read_text().strip()
    a_rr = np.load(prefix + "_A_rr.npy", mmap_mode="r")
    a_ru = np.load(prefix + "_A_ru.npy", mmap_mode="r")
    a_rz = np.load(prefix + "_A_rz.npy", mmap_mode="r")
    a_uu = np.load(prefix + "_A_uu.npy", mmap_mode="r")
    a_uz = np.load(prefix + "_A_uz.npy", mmap_mode="r")
    a_zz = np.load(prefix + "_A_zz.npy", mmap_mode="r")
    keep = np.asarray(np.load(prefix + "_keep.npy"), dtype=np.int64)
    linv = np.asarray(np.load(prefix + "_Linv.npy"))

    n = a_rr.shape[0]
    p = np.empty(3 * n, dtype=np.int64)
    k = np.arange(n, dtype=np.int64)
    p[3 * k + 0] = k
    p[3 * k + 1] = n + k
    p[3 * k + 2] = 2 * n + k
    pinv = np.empty_like(p)
    pinv[p] = np.arange(3 * n, dtype=np.int64)
    keep_node = pinv[keep]
    red_pos = np.full(3 * n, -1, dtype=np.int64)
    red_pos[keep_node] = np.arange(keep_node.size, dtype=np.int64)

    chunk = int(os.environ.get("AGNI_LAMBDA32_NODE_CHUNK", "4000"))
    a32_path = artifact_dir / "A32_keep.npy"
    A32 = np.lib.format.open_memmap(
        str(a32_path),
        mode="w+",
        dtype=np.result_type(a_rr.dtype, linv.dtype),
        shape=(keep.size, keep.size),
    )
    A32[:] = 0
    linv_t = np.conjugate(np.swapaxes(linv, 1, 2))

    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        rows = np.arange(i0, i1, dtype=np.int64)
        row_node = np.arange(3 * i0, 3 * i1, dtype=np.int64)
        row_red = red_pos[row_node]
        row_mask = row_red >= 0
        li = linv[i0:i1]

        for j0 in range(0, n, chunk):
            j1 = min(j0 + chunk, n)
            cols = np.arange(j0, j1, dtype=np.int64)
            col_node = np.arange(3 * j0, 3 * j1, dtype=np.int64)
            col_red = red_pos[col_node]
            col_mask = col_red >= 0
            if not row_mask.any() or not col_mask.any():
                continue

            rr = np.asarray(a_rr[np.ix_(rows, cols)])
            ru = np.asarray(a_ru[np.ix_(rows, cols)])
            rz = np.asarray(a_rz[np.ix_(rows, cols)])
            uu = np.asarray(a_uu[np.ix_(rows, cols)])
            uz = np.asarray(a_uz[np.ix_(rows, cols)])
            zz = np.asarray(a_zz[np.ix_(rows, cols)])

            ur = np.conjugate(np.asarray(a_ru[np.ix_(cols, rows)]).T)
            zr = np.conjugate(np.asarray(a_rz[np.ix_(cols, rows)]).T)
            zu = np.conjugate(np.asarray(a_uz[np.ix_(cols, rows)]).T)

            tile4 = np.empty((i1 - i0, 3, j1 - j0, 3), dtype=A32.dtype)
            tile4[:, 0, :, 0] = rr
            tile4[:, 0, :, 1] = ru
            tile4[:, 0, :, 2] = rz
            tile4[:, 1, :, 0] = ur
            tile4[:, 1, :, 1] = uu
            tile4[:, 1, :, 2] = uz
            tile4[:, 2, :, 0] = zr
            tile4[:, 2, :, 1] = zu
            tile4[:, 2, :, 2] = zz

            tile4 = np.einsum("aik,akbl->aibl", li, tile4, optimize=True)
            tile4 = np.einsum("aibl,blj->aibj", tile4, linv_t[j0:j1], optimize=True)
            tile = tile4.reshape(3 * (i1 - i0), 3 * (j1 - j0))
            A32[np.ix_(row_red[row_mask], col_red[col_mask])] = tile[
                np.ix_(row_mask, col_mask)
            ]

    A32.flat[:: A32.shape[1] + 1] += 1e-11
    A32.flush()

    A3 = np.load(a3_path, mmap_mode="r")
    w3 = np.asarray(np.load(w3_path)).reshape(-1)
    v3 = np.asarray(np.load(v3_path)).reshape(-1)

    assert A3.shape == A32.shape, f"shape mismatch: A3={A3.shape}, A32={A32.shape}"

    cmp_chunk = int(os.environ.get("AGNI_LAMBDA32_COMPARE_CHUNK", "256"))
    for i0 in range(0, A3.shape[0], cmp_chunk):
        i1 = min(i0 + cmp_chunk, A3.shape[0])
        np.testing.assert_allclose(
            np.asarray(A32[i0:i1, :]),
            np.asarray(A3[i0:i1, :]),
            rtol=1e-8,
            atol=1e-10,
            err_msg=(
                f"matrix mismatch at n_rho={n_rho}, n_theta={n_theta}, "
                f"n_zeta={n_zeta}, row_block={i0}:{i1}"
            ),
        )

    av = np.zeros_like(v3, dtype=np.result_type(A32.dtype, v3.dtype))
    mv_chunk = int(os.environ.get("AGNI_LAMBDA32_MV_CHUNK", "256"))
    for i0 in range(0, A32.shape[0], mv_chunk):
        i1 = min(i0 + mv_chunk, A32.shape[0])
        av[i0:i1] = np.asarray(A32[i0:i1, :]) @ v3
    lam3 = w3[0]
    rq = np.vdot(v3, av) / np.vdot(v3, v3)
    lam32_rq = np.real_if_close(rq)

    mask = np.abs(v3) > 1e-5
    assert np.any(
        mask
    ), "No reliable entries in eigenvector after |v| > 1e-5 filtering."
    ratio = av[mask] / v3[mask]
    ratio = ratio[np.isfinite(ratio)]
    assert ratio.size > 0, "All filtered (A v)/v entries are non-finite."
    ratio_err = np.max(np.abs(ratio - lam3))
    assert (
        ratio_err < 5e-4
    ), f"filtered (A v)/v - lambda mismatch too large: maxabs={ratio_err:.3e}"

    eval_mode = os.environ.get("AGNI_LAMBDA32_EIGVEC_COMPARE", "1").lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }
    overlap = np.nan
    vrel = np.nan
    lam32 = lam32_rq
    if eval_mode:
        from scipy.sparse.linalg import eigsh

        print("[lambda-compare] phase=compare eigsh(A32) start", flush=True)
        w32, v32 = eigsh(
            np.asarray(A32),
            k=1,
            sigma=-1e-3,
            which="LM",
            tol=1e-8,
            return_eigenvectors=True,
        )
        lam32 = np.asarray(w32).reshape(-1)[0]
        vv32 = np.asarray(v32[:, 0]).reshape(-1)
        v3n = v3 / (np.linalg.norm(v3) + 1e-300)
        v32n = vv32 / (np.linalg.norm(vv32) + 1e-300)
        phase = np.vdot(v32n, v3n)
        phase = phase / (np.abs(phase) + 1e-300)
        diff = v32n - phase * v3n
        overlap = np.abs(np.vdot(v32n, v3n))
        vrel = np.linalg.norm(diff)
        print("[lambda-compare] phase=compare eigsh(A32) done", flush=True)

    print(
        "lambda compare:",
        f"n_rho={n_rho}",
        f"n_theta={n_theta}",
        f"n_zeta={n_zeta}",
        f"lambda3={lam3:.12e}",
        f"lambda32_rq={lam32_rq:.12e}",
        f"lambda32={np.asarray(lam32).reshape(-1)[0]:.12e}",
        f"maxabs((Av)/v-lambda)={ratio_err:.3e}",
        f"eigvec_overlap={overlap:.6e}",
        f"eigvec_relerr={vrel:.6e}",
        flush=True,
    )

    np.testing.assert_allclose(
        np.asarray([lam32_rq]).reshape(-1),
        w3.reshape(-1),
        rtol=1e-6,
        atol=1e-8,
        err_msg=(
            f"eigenvalue mismatch (rq) at n_rho={n_rho}, n_theta={n_theta}, "
            f"n_zeta={n_zeta}"
        ),
    )
    if eval_mode:
        np.testing.assert_allclose(
            np.asarray([lam32]).reshape(-1),
            w3.reshape(-1),
            rtol=1e-6,
            atol=1e-8,
            err_msg=(
                f"eigenvalue mismatch (eigsh) at n_rho={n_rho}, "
                f"n_theta={n_theta}, n_zeta={n_zeta}"
            ),
        )
        assert overlap > 0.95, f"eigenvector overlap too low: {overlap:.6e}"


@pytest.mark.unit
@pytest.mark.slow
def test_lambda3_lambda32_final_matrix_and_eigenvalue(tmp_path):
    """Entry point for the split-phase ``lambda3``/``lambda32`` workflow.

    In normal CI this test is skipped because it is intentionally expensive.
    When enabled, it can behave like three separate tests depending on
    ``AGNI_LAMBDA32_MODE``:

    - build only the ``lambda3`` reference data,
    - build only the ``lambda32`` block dump,
    - compare previously saved outputs,
    - or run all three stages locally in one process.

    The same Python test is therefore usable for:

    - lightweight local debugging at reduced resolution,
    - split CPU/GPU Slurm workflows that share an artifact directory,
    - and final end-to-end regression checks once artifacts already exist.

    Environment controls
    --------------------
    - ``AGNI_LAMBDA32_ENABLE=1`` to run this heavy test.
    - ``AGNI_LAMBDA32_MODE`` in {``lambda3``, ``lambda32``, ``compare``, ``all``}.
    - ``AGNI_LAMBDA32_ARTIFACT_DIR`` shared directory used across separate jobs.
    """
    if os.environ.get("AGNI_LAMBDA32_ENABLE", "0").lower() in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        pytest.skip("Set AGNI_LAMBDA32_ENABLE=1 to run this heavy comparison test.")

    n_rho = int(os.environ.get("AGNI_LAMBDA32_COMPARE_N_RHO", "30"))
    n_theta = int(os.environ.get("AGNI_LAMBDA32_COMPARE_N_THETA", "36"))
    n_zeta = int(os.environ.get("AGNI_LAMBDA32_COMPARE_N_ZETA", "24"))
    eq_path = Path(
        os.environ.get(
            "AGNI_EQ_PATH",
            "/pscratch/sd/r/rgaur/AGNI_var/matrix-free/"
            "qh_beta1.5_imin1.02_modprof_221410.h5",
        )
    )
    mode = os.environ.get("AGNI_LAMBDA32_MODE", "all").strip().lower()
    artifact_dir = Path(os.environ.get("AGNI_LAMBDA32_ARTIFACT_DIR", str(tmp_path)))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[lambda-compare] mode={mode} n_rho={n_rho} n_theta={n_theta} "
        f"n_zeta={n_zeta} artifact_dir={artifact_dir}",
        flush=True,
    )

    if mode not in {"all", "lambda3", "lambda32", "compare"}:
        raise AssertionError(f"Unknown AGNI_LAMBDA32_MODE={mode!r}")

    if mode in {"all", "lambda3", "lambda32"}:
        if not eq_path.is_file():
            pytest.skip(f"AGNI equilibrium fixture not found: {eq_path}")
        eq = Equilibrium.load(str(eq_path))
        grid, diffmat = _build_grid_diffmat(eq, n_rho, n_theta, n_zeta)
        if mode in {"all", "lambda3"}:
            _run_lambda3_capture(
                eq, grid, diffmat, n_rho, n_theta, n_zeta, artifact_dir
            )
        if mode in {"all", "lambda32"}:
            _run_lambda32_dump(eq, grid, diffmat, n_rho, n_theta, n_zeta, artifact_dir)

    if mode in {"all", "compare"}:
        _compare_saved_outputs(artifact_dir, n_rho, n_theta, n_zeta)
