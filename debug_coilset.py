"""
Debug/validation script for CoilSet vectorized Biot-Savart.

Compares the new chunked-vmap CoilSet._compute_A_or_B against the old
scan/fori_loop implementation for correctness.

Tests:
  - SplineXYZCoil (HH path) and FourierPlanarCoil (quadrature path)
  - NFP/sym combinations: NFP=4 sym=True, NFP=1 sym=False
  - Various coil counts to exercise padding: 1, 3, 5, 17, 33
  - SumMagneticField combining both types

Usage:  conda run -n desc-thea-gpu python debug_coilset.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax
from desc.backend import jnp, fori_loop, tree_stack
from desc.coils import CoilSet, SplineXYZCoil, FourierPlanarCoil
from desc.magnetic_fields import SumMagneticField
from desc.compute import get_params
from desc.utils import rpz2xyz, xyz2rpz, rpz2xyz_vec, xyz2rpz_vec, reflection_matrix
from jax.lax import scan as jax_scan

print(f"Backend: {jax.default_backend()}")

# ── Evaluation points (toroidal distribution) ────────────────────────
rng = np.random.default_rng(42)
n_eval = 20
R = 10 + 0.5 * rng.standard_normal(n_eval)
phi = 2 * np.pi * rng.random(n_eval)
Z = 0.3 * rng.standard_normal(n_eval)
eval_xyz = np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1)

# ── Coil constructors ────────────────────────────────────────────────
def make_spline_coil(current=1e5, n_pts=32, R0=10.0, a=1.0, dz=0.0, seed=0):
    rng_c = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    X = (R0 + a * np.cos(theta)) + 0.05 * rng_c.standard_normal(n_pts)
    Y = a * np.sin(theta) + 0.05 * rng_c.standard_normal(n_pts)
    Z_arr = np.full(n_pts, dz) + 0.05 * rng_c.standard_normal(n_pts)
    return SplineXYZCoil(current, X=X, Y=Y, Z=Z_arr)


def make_fourier_planar_coil(current=1e5, seed=0):
    rng_c = np.random.default_rng(seed)
    center = [10 + 0.1 * rng_c.standard_normal(), 0.0, 0.0]
    return FourierPlanarCoil(current=current, center=center, r_n=1.0)


# ── Old CoilSet B (scan/fori_loop reference) ─────────────────────────
def old_coilset_compute_B(cs, coords, basis="xyz"):
    """Reimplementation of the OLD CoilSet._compute_A_or_B (scan/fori_loop)."""
    coords = jnp.atleast_2d(jnp.asarray(coords))
    if basis == "rpz":
        coords_xyz = rpz2xyz(coords)
    else:
        coords_xyz = coords

    params = [get_params(["x_s", "x", "s", "ds"], coil, basis="rpz") for coil in cs]
    for par, coil in zip(params, cs):
        par["current"] = coil.current

    if cs.sym:
        normal = jnp.array(
            [-jnp.sin(jnp.pi / cs.NFP), jnp.cos(jnp.pi / cs.NFP), 0]
        )
        coords_sym = (
            coords_xyz @ reflection_matrix(normal).T @ reflection_matrix([0, 0, 1]).T
        )
        coords_xyz_full = jnp.vstack((coords_xyz, coords_sym))
    else:
        coords_xyz_full = coords_xyz

    coords_rpz = xyz2rpz(coords_xyz_full)
    op = cs[0].compute_magnetic_field

    def nfp_loop(k, AB):
        coords_nfp = coords_rpz + jnp.array([0, 2 * jnp.pi * k / cs.NFP, 0])

        def body(AB, x):
            AB += op(coords_nfp, params=x, basis="rpz", source_grid=None, chunk_size=None)
            return AB, None

        AB += jax_scan(body, jnp.zeros(coords_nfp.shape), tree_stack(params))[0]
        return AB

    AB = fori_loop(0, cs.NFP, nfp_loop, jnp.zeros_like(coords_rpz))

    if cs.sym:
        n_orig = coords.shape[0]
        AB = AB[:n_orig] + AB[n_orig:] * jnp.array([-1, 1, 1])

    if basis == "xyz":
        AB = rpz2xyz_vec(AB, x=coords[:, 0], y=coords[:, 1])
    return AB


# ── Run tests ────────────────────────────────────────────────────────
PASS_COUNT = 0
FAIL_COUNT = 0


def test_old_vs_new(label, cs, rtol=1e-3):
    global PASS_COUNT, FAIL_COUNT
    B_new = np.asarray(cs.compute_magnetic_field(eval_xyz, basis="xyz"))
    B_old = np.asarray(old_coilset_compute_B(cs, eval_xyz, basis="xyz"))
    diff = np.max(np.abs(B_new - B_old))
    maxref = np.max(np.abs(B_old))
    rel = diff / maxref if maxref > 0 else diff
    status = "PASS" if rel < rtol else "*** FAIL ***"
    print(f"  {label}: |new-old|={diff:.2e}, rel={rel:.2e} {status}")
    if status == "PASS":
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
        worst = np.argmax(np.linalg.norm(B_new - B_old, axis=-1))
        print(f"    worst pt[{worst}]: new={B_new[worst]}, old={B_old[worst]}")


print("\n--- SplineXYZCoil (HH path), NFP=4 sym=True ---")
for n in [1, 3, 5, 17, 33]:
    coils = [make_spline_coil(current=1e5*(i+1), seed=i, dz=0.5*i) for i in range(n)]
    test_old_vs_new(f"SplineXYZ x{n}", CoilSet(*coils, NFP=4, sym=True))

print("\n--- FourierPlanarCoil (quadrature path), NFP=4 sym=True ---")
for n in [1, 3, 5, 17, 33]:
    coils = [make_fourier_planar_coil(current=1e5*(i+1), seed=i) for i in range(n)]
    test_old_vs_new(f"FP x{n}", CoilSet(*coils, NFP=4, sym=True))

print("\n--- NFP=1, sym=False ---")
test_old_vs_new("SplineXYZ x5 NFP=1",
    CoilSet(*[make_spline_coil(current=1e5, seed=i) for i in range(5)], NFP=1, sym=False))
test_old_vs_new("FP x5 NFP=1",
    CoilSet(*[make_fourier_planar_coil(current=1e5, seed=i) for i in range(5)], NFP=1, sym=False))

print("\n--- NFP=4, sym=False ---")
test_old_vs_new("SplineXYZ x5 NFP=4 nosym",
    CoilSet(*[make_spline_coil(current=1e5, seed=i) for i in range(5)], NFP=4, sym=False))
test_old_vs_new("FP x5 NFP=4 nosym",
    CoilSet(*[make_fourier_planar_coil(current=1e5, seed=i) for i in range(5)], NFP=4, sym=False))

print("\n--- SumMagneticField ---")
cs_hh = CoilSet(*[make_spline_coil(current=1e5, seed=i) for i in range(5)], NFP=4, sym=True)
cs_fp = CoilSet(*[make_fourier_planar_coil(current=5e4, seed=i) for i in range(3)], NFP=4, sym=True)
total_field = SumMagneticField([cs_hh, cs_fp])
B_sum = np.asarray(total_field.compute_magnetic_field(eval_xyz, basis="xyz"))
B_hh = np.asarray(cs_hh.compute_magnetic_field(eval_xyz, basis="xyz"))
B_fp = np.asarray(cs_fp.compute_magnetic_field(eval_xyz, basis="xyz"))
B_manual = B_hh + B_fp
diff = np.max(np.abs(B_sum - B_manual))
status = "PASS" if diff < 1e-12 else "*** FAIL ***"
print(f"  SumMagneticField: |sum - (hh+fp)| = {diff:.2e} {status}")
if status == "PASS":
    PASS_COUNT += 1
else:
    FAIL_COUNT += 1

print(f"\n{'='*50}")
print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
print(f"{'='*50}")
if FAIL_COUNT > 0:
    exit(1)
