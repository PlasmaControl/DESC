"""Regression gates for the AGNI two-level preconditioner machinery.

These pin the numbers that ``AGNI_var/precond_stage2`` produces, so that changes
to ``desc/compute/_stability.py`` cannot silently move them. The reference values
are the ones recorded in ``precond_stage2/VERIFICATION.md`` after the 2026-08-10
clobber-and-restore, where ``shakeout2.py`` reproduced the pre-clobber run
bit-identically. They are exact reproductions, not tolerances that were fitted
afterwards, which is why the assertions below are tight.

The preconditioner modules (``transfer``, ``two_level``, ``ring_precond``,
``pcg_test``) currently live OUTSIDE the DESC package, under
``AGNI_var/precond_stage2`` and ``AGNI_var/precond_harmonic``. Until they are
vendored into ``desc/``, this test skips wherever they are absent -- which is
every machine but the developer's, including CI.

``shakeout2_new.py`` is the variant wired to the in-package ring assembler
(``_agni3_assemble(..., ring_nodes=...)``). The original ``shakeout2.py`` goes
through ``restricted_assemble``, which rewrites ``_agni3_assemble``'s SOURCE
TEXT and therefore stopped working the moment that function was parameterized.
Both produce identical numbers; only the ``_new`` one still runs.

It is invoked as a subprocess rather than imported: it is a linear
script, not a module of functions, and it installs its own ``sys.path`` entries
and calls ``set_device`` at import. Running it out-of-process keeps those side
effects out of the pytest session and tests the script exactly as it is actually
run. Once the modules are vendored, this should call them directly instead.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Device for the subprocess gates. CPU BY DEFAULT: nothing here needs an
# accelerator to be CORRECT, only to be fast, and a test that silently demands
# one cannot run on someone else's machine. Set AGNI_TEST_DEVICE=gpu to use one.
_DEVICE = os.environ.get("AGNI_TEST_DEVICE", "cpu").strip().lower()

# No absolute path is baked in. Point AGNI_VAR_DIR at the AGNI_var tree to run
# these; without it they skip, which is correct on any other machine.
_AGNI_VAR = (
    Path(os.environ.get("AGNI_VAR_DIR", ""))
    if os.environ.get("AGNI_VAR_DIR", "").strip()
    else None
)
_STAGE2 = (_AGNI_VAR / "precond_stage2") if _AGNI_VAR is not None else Path(".")
_SHAKEOUT = _STAGE2 / "shakeout2_new.py"
_REQUIRED = (
    [
        _SHAKEOUT,
        _STAGE2 / "transfer.py",
        _STAGE2 / "two_level.py",
        _AGNI_VAR / "precond_harmonic",
    ]
    if _AGNI_VAR is not None
    else []
)

_missing = (
    ["set AGNI_VAR_DIR to the AGNI_var tree"]
    if _AGNI_VAR is None
    else [str(p) for p in _REQUIRED if not p.exists()]
)
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "AGNI preconditioner modules are not in the DESC package yet; "
        f"missing: {_missing}"
    ),
)


def _num(pattern, text, what):
    """Pull one float out of the shakeout output, or fail loudly."""
    m = re.search(pattern, text)
    assert m, f"could not find {what} in shakeout2 output"
    return float(m.group(1))


@pytest.mark.unit
@pytest.mark.slow
def test_shakeout2_reproduces_recorded_values():
    """Two-level preconditioner reproduces precond_stage2/VERIFICATION.md.

    Runs the stage-2 shakeout at its documented tiny resolution
    (fine 6x12x6, coarse 4x8x4) and checks every number VERIFICATION.md
    recorded. Together these cover the prolongation and its adjoint, the ring
    block preconditioner, the additive two-level operator, the conditioning
    gain, and the soft-mode retention path -- i.e. the whole chain that
    ``finite-n lambda3 rayleigh``'s ``pcg_deflated`` solver stands on.

    A mismatch here means the operator or the assembly changed, and every
    number produced after the change has to be re-derived.
    """
    env = dict(os.environ)
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "DESC_DEVICE": "cpu",
            "JAX_ENABLE_X64": "1",
            "AGNI_FINE": "6 12 6",
            "AGNI_COARSE": "4 8 4",
            "MPOL": "3",
            "NTOR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    proc = subprocess.run(
        [sys.executable, str(_SHAKEOUT)],
        cwd=str(_STAGE2),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"shakeout2 exited {proc.returncode}\n{out[-3000:]}"
    assert "SHAKEOUT2 CLEAN" in out, f"shakeout2 did not report CLEAN\n{out[-3000:]}"

    # --- prolongation and its adjoint -------------------------------------
    adjoint = _num(
        r"ADJOINT GATE PASS: worst rel mismatch ([0-9.e+-]+)", out, "adjoint"
    )
    assert adjoint < 1e-13, f"adjoint mismatch {adjoint:.3e} (recorded 1.414e-14)"

    dense_p = _num(r"dense P matches callable ([0-9.e+-]+)", out, "dense P")
    assert dense_p < 1e-14, f"dense P mismatch {dense_p:.3e} (recorded 3.50e-16)"

    # --- ring block preconditioner ----------------------------------------
    ring_lmin = _num(r"M\^-1 SPD \(lam_min=([0-9.e+-]+)\)", out, "ring lam_min")
    assert ring_lmin == pytest.approx(3.062e-06, rel=1e-3), (
        f"ring M^-1 lam_min {ring_lmin:.4e} != recorded 3.062e-06; "
        "M^-1 must stay SPD or CG is not legal"
    )

    # --- additive two-level operator --------------------------------------
    asym = _num(r"two-level M\^-1 symmetric \(([0-9.e+-]+)\)", out, "two-level asym")
    assert asym == 0.0, f"two-level M^-1 lost exact symmetry: {asym:.3e}"

    tl_spd = _num(
        r"symmetric \([0-9.e+-]+\) and SPD \(([0-9.e+-]+)\)", out, "2-level SPD"
    )
    assert tl_spd == pytest.approx(
        3.866e-06, rel=1e-3
    ), f"two-level SPD lam_min {tl_spd:.4e} != recorded 3.866e-06"

    # --- conditioning gain -------------------------------------------------
    k_ring = _num(r"kappa ring=([0-9.e+-]+)", out, "kappa ring")
    k_two = _num(r"two-level=([0-9.e+-]+)", out, "kappa two-level")
    gain = _num(r"gain=([0-9.e+-]+)x", out, "gain")

    assert k_ring == pytest.approx(6.4793e04, rel=1e-4), (
        f"kappa(M_ring^-1 H) = {k_ring:.6e} != recorded 6.4793e+04. This single "
        "number validates block assembly, the keep-mask mapping and the group "
        "ordering together."
    )
    assert k_two == pytest.approx(
        2.2172e04, rel=1e-4
    ), f"kappa(M_twolevel^-1 H) = {k_two:.6e} != recorded 2.2172e+04"
    assert gain == pytest.approx(
        2.922, rel=1e-3
    ), f"two-level gain {gain:.4f}x != recorded 2.922x"

    # --- soft-mode retention ----------------------------------------------
    m = re.search(r"r_H=\[([0-9.eE+\- ]+)\]", out)
    assert m, "could not find retention r_H"
    r_h = [float(v) for v in m.group(1).split()]
    recorded = [0.1064, 0.1113, 0.0081, 0.0485, 0.0502]
    assert len(r_h) == len(recorded), f"retention rank changed: {r_h}"
    for got, want in zip(r_h, recorded):
        assert got == pytest.approx(want, abs=5e-4), f"retention {r_h} != {recorded}"

    # --- calibration: on range(P) the coarse correction must be exact ------
    m = re.search(r"calibration, x in range\(P\): r_H=\[([0-9.eE+\- ]+)\]", out)
    assert m, "could not find calibration r_H"
    cal = [float(v) for v in m.group(1).split()]
    assert cal == pytest.approx([1.0, 1.0, 1.0], abs=1e-9), (
        f"calibration r_H={cal} != [1, 1, 1]; on range(P) the two-level "
        "correction is exact by construction, so any drift is a real defect"
    )


# The T1 gate is opt-in on MEMORY, not on hardware: at GJ 32x32x12 the dense A is
# 10.12 GB with a ~54 GB assembly peak. It runs on CPU given enough RAM, and on a
# GPU only with 80 GB. Opt-in rather than auto-detected because a confusing OOM
# is worse than a skip.
_T1_EQ = (
    (_AGNI_VAR / "dense-eigsh-optimization" / "eq_lowres_L10M10N10.h5")
    if _AGNI_VAR is not None
    else Path("eq_lowres_L10M10N10.h5")
)
_t1_reason = None
if os.environ.get("AGNI_GPU_TESTS", "0").strip() not in ("1", "true", "True"):
    _t1_reason = (
        "set AGNI_GPU_TESTS=1 to run; GJ 32x32x12 assembles a 10.12 GB dense A "
        "with a ~54 GB peak, so it needs a large-memory node (CPU or GPU)"
    )
elif not _T1_EQ.is_file():
    _t1_reason = f"T1 equilibrium not found: {_T1_EQ}"


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.skipif(bool(_t1_reason), reason=str(_t1_reason))
def test_t1_jax_lanczos_jit_reproduces_recorded_value():
    """Jitted jax_lanczos through FinitenStability matches dense ARPACK.

    This is T1 of ``precond_stage2/OPTIMIZATION.md``: GJ basis, 32x32x12,
    ``USE_JIT=1``, ``AGNI_EIGENSOLVER=jax_lanczos``, 50 matvecs. It runs the
    FULL objective -- assembly, LU shift-invert, Lanczos, Rayleigh quotient --
    compiled and on device, then compares against an eager
    ``eq.compute("finite-n lambda3")`` + host ARPACK reference computed in the
    same process on the same GPU.

    The two share nothing but the operator: one is compiled JAX with an on-device
    LU, the other a host callback into scipy. Agreement to ~1e-9 is therefore a
    real check on the assembly and the shift, and it is the number the rest of
    stage 2 calibrates against.

    Reference: job 56666621, recorded in OPTIMIZATION.md section T1.
    """
    import json

    out_name = "t1_gj_pytest.json"
    env = dict(os.environ)
    env.update(
        {
            "DESC_DEVICE": _DEVICE,
            "JAX_PLATFORMS": "" if _DEVICE == "gpu" else "cpu",
            "JAX_ENABLE_X64": "1",
            "AGNI_EIGENSOLVER": "jax_lanczos",
            "AGNI_NUM_MATVECS": "50",
            "USE_JIT": "1",
            "EQ_PATH": str(_T1_EQ),
            "OUT": out_name,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    proc = subprocess.run(
        [sys.executable, "-u", str(_STAGE2 / "t1_jit_gj_value.py")],
        cwd=str(_STAGE2),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"T1 exited {proc.returncode}\n{out[-3000:]}"

    rec = json.loads((_STAGE2 / out_name).read_text())

    assert rec["jit_ok"] is True, f"jitted evaluation failed: {rec}"

    lam_jit = float(rec["lambda_jit"])
    lam_dense = float(rec["lambda_dense"])

    # Recorded to every digit that was printed; these are exact reproductions,
    # so the tolerance only absorbs run-to-run LU/Lanczos noise.
    assert lam_jit == pytest.approx(
        -1.6662378246e-04, rel=1e-9
    ), f"jitted lam_R = {lam_jit:.10e} != recorded -1.6662378246e-04"
    assert lam_dense == pytest.approx(
        -1.6662378258e-04, rel=1e-9
    ), f"dense ARPACK reference = {lam_dense:.10e} != recorded -1.6662378258e-04"

    rel = abs(lam_jit - lam_dense) / abs(lam_dense)
    assert rel < 5e-9, (
        f"|jit - dense|/|dense| = {rel:.3e}; recorded 7.16e-10. A jump here means "
        "the compiled path and the host-callback path no longer agree on the "
        "operator or the shift."
    )
    assert (
        lam_jit < 0.0 and lam_dense < 0.0
    ), "growth rate lost its sign: an unstable equilibrium reported as stable"


_T2_HISTORY_REF = _STAGE2 / "gj_opt_history_56670567.jsonl"
_t2_reason = None
if os.environ.get("AGNI_GPU_TESTS", "0").strip() not in ("1", "true", "True"):
    _t2_reason = (
        "set AGNI_GPU_TESTS=1 to run; a full optimization at GJ 32x32x12, "
        "~20 min on a GPU and longer on CPU"
    )
elif not _T2_HISTORY_REF.is_file():
    _t2_reason = f"reference trajectory not found: {_T2_HISTORY_REF}"


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.skipif(bool(_t2_reason), reason=str(_t2_reason))
def test_optimizer_reproduces_recorded_trajectory():
    """End-to-end optimization: FinitenStability actually drives lambda down.

    This is T2 of ``precond_stage2/OPTIMIZATION.md`` -- the only test that
    exercises the OPTIMIZER rather than a fixed-point eigenvalue. It covers
    ``FinitenStability.update_state`` (dense ``finite-n lambda3`` + ARPACK), the
    Hellmann-Feynman gradient through the ``custom_vjp``, ``ProximalProjection``
    holding force balance, sigma ``adapt``, and the dense postcheck after the
    accepted step.

    Two different standards, deliberately:

    * STEP 0 must be EXACT. It is a fixed-point evaluation of the assembly and
      eigensolve at the starting equilibrium, with no optimizer state involved,
      so any drift there is a real change in the operator.
    * STEP 1 is checked loosely (1e-3). The optimizer's path runs through an
      ``lsqtr`` line search and a ``ProximalProjection`` force-balance re-solve,
      both iterative, so it is not bit-reproducible. Measured 1.7e-05 on
      ``lambda_dense`` between two runs of identical configuration.

    Reference: job 56670567, ``gj_opt_history_56670567.jsonl``.
    """
    import json

    out_name = f"gj_opt_history_pytest_{os.getpid()}.jsonl"
    env = dict(os.environ)
    env.update(
        {
            "DESC_DEVICE": _DEVICE,
            "JAX_PLATFORMS": "" if _DEVICE == "gpu" else "cpu",
            "JAX_ENABLE_X64": "1",
            "AGNI_EIGENSOLVER": "jax_lanczos",
            "AGNI_NUM_MATVECS": "50",
            "AGNI_SIGMA": "-1.0e-3",
            "AGNI_SIGMA_MODE": "adapt",
            "USE_JIT": "1",
            "N_STEPS": "1",
            "OUTER_MAXITER": "20",
            "UNFIX_K": "1",
            "POSTCHECK": "1",
            "NR": "32",
            "NT": "32",
            "NZ": "12",
            "OUT": out_name,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    proc = subprocess.run(
        [sys.executable, "-u", str(_STAGE2 / "run_gj_opt.py")],
        cwd=str(_STAGE2),
        env=env,
        capture_output=True,
        text=True,
        timeout=10800,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"optimizer exited {proc.returncode}\n{out[-3000:]}"

    def _rows(path):
        return [
            json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()
        ]

    new = _rows(_STAGE2 / out_name)
    ref = _rows(_T2_HISTORY_REF)

    assert len(new) >= 2, (
        "no post-step row: eq.optimize() did not complete a step. The run is "
        f"only meaningful if it returns.\n{out[-2000:]}"
    )

    # --- step 0: exact ----------------------------------------------------
    for key in ("lambda_dense", "lambda_rayleigh", "growth", "force_balance_max"):
        got, want = float(new[0][key]), float(ref[0][key])
        rel = abs(got - want) / abs(want)
        assert rel < 1e-12, (
            f"step 0 {key} = {got:.12e} != recorded {want:.12e} (rel {rel:.2e}). "
            "Step 0 involves no optimizer state, so this is a change in the "
            "assembly or the eigensolve."
        )

    # --- step 1: the descent ----------------------------------------------
    lam0, lam1 = float(new[0]["lambda_dense"]), float(new[1]["lambda_dense"])
    assert abs(lam1) < abs(lam0), (
        f"|lambda| did not decrease: {abs(lam0):.6e} -> {abs(lam1):.6e}. The "
        "Hellmann-Feynman gradient is not pointing downhill."
    )
    assert np.sign(lam1) == np.sign(lam0), "growth rate changed sign across the step"

    for key in ("lambda_dense", "growth"):
        got, want = float(new[1][key]), float(ref[1][key])
        rel = abs(got - want) / abs(want)
        assert rel < 1e-3, (
            f"step 1 {key} = {got:.12e} vs recorded {want:.12e} (rel {rel:.2e}); "
            "the optimizer took a materially different step"
        )
