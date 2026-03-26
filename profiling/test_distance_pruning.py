#!/usr/bin/env python
"""Test distance pruning for CoilSetMinDistance at full resolution.

The full pairwise computation OOMs at 424 effective coils, so we:
1. Validate correctness at lightweight resolution (where full fits)
2. Benchmark pruned vs DESC's chunked implementation at full resolution
3. Test memory limits — can pruned handle larger jac_chunk_size?
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
from desc import set_device
set_device("gpu")

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

sys.path.insert(0, os.path.expanduser("~/finding_better_basins"))
from configs import build_config, load_data, setup_jax
from desc.objectives.utils import softmin
from desc.utils import safenorm
from desc.objectives import CoilSetMinDistance, ObjectiveFunction, FixParameters
from desc.grid import LinearGrid
from desc.derivatives import Derivative


def compute_min_distance_pruned(pts, alpha, K):
    """Softmin distances with K-nearest-neighbor pruning."""
    n_coils = pts.shape[0]
    K = min(K, n_coils - 1)

    centroids = jnp.mean(pts, axis=1)  # (n_coils, 3)
    centroid_dists = jnp.linalg.norm(
        centroids[:, None] - centroids[None, :], axis=-1
    )
    centroid_dists = centroid_dists.at[jnp.diag_indices(n_coils)].set(jnp.inf)

    def body(k):
        dists_k = centroid_dists[k]
        neighbor_idx = jax.lax.stop_gradient(jnp.argsort(dists_k)[:K])
        coil_pts = pts[k]
        neighbor_pts = pts[neighbor_idx]
        diff = coil_pts[None, :, None, :] - neighbor_pts[:, None, :, :]
        dist = safenorm(diff, axis=-1)
        return softmin(dist, alpha)

    return jax.vmap(body)(jnp.arange(n_coils))


def compute_min_distance_full(pts, alpha):
    """Full pairwise (only works at small scale)."""
    n_coils = pts.shape[0]
    def body(k):
        coil_pts = pts[k]
        other_pts = jnp.delete(pts, k, axis=0, assume_unique_indices=True)
        dist = safenorm(coil_pts[None, :, None] - other_pts[:, None], axis=-1)
        return softmin(dist, alpha)
    return jax.vmap(body)(jnp.arange(n_coils))


# ============================================================
# Part 1: Correctness at lightweight resolution
# ============================================================
print("=== Part 1: Correctness (lightweight) ===")
config_lw = build_config(platform="perlmutter", lightweight=True, seed=0)
setup_jax(config_lw)
data_lw = load_data(config_lw)
coilset_lw = data_lw["shaping"]
grid_lw = LinearGrid(N=config_lw["grid"]["coil_grid_N"])
alpha = config_lw["softmin"]["softmin_alpha"]

pts_lw = coilset_lw._compute_position(
    params=coilset_lw.params_dict, grid=grid_lw, basis="xyz"
)
print(f"Lightweight: {pts_lw.shape[0]} effective coils, {pts_lw.shape[1]} nodes")

full_dists = compute_min_distance_full(pts_lw, alpha)
full_dists.block_until_ready()

for K in [5, 10, 20, 50, 100]:
    pruned_dists = compute_min_distance_pruned(pts_lw, alpha, K)
    pruned_dists.block_until_ready()
    diff = jnp.abs(pruned_dists - full_dists)
    max_diff = float(jnp.max(diff))
    n_wrong = int(jnp.sum(diff > 1e-6))
    print(f"  K={K:>3d}: max_diff={max_diff:.2e}, wrong={n_wrong}/{len(full_dists)}")

# Jacobian correctness at lightweight
print("\nJacobian correctness (lightweight, chunk=1):")
x_lw = coilset_lw.pack_params(coilset_lw.params_dict)

def jac_full_lw(x):
    params = coilset_lw.unpack_params(x)
    pts = coilset_lw._compute_position(params=params, grid=grid_lw, basis="xyz")
    return compute_min_distance_full(pts, alpha)

def jac_pruned_lw(x, K):
    params = coilset_lw.unpack_params(x)
    pts = coilset_lw._compute_position(params=params, grid=grid_lw, basis="xyz")
    return compute_min_distance_pruned(pts, alpha, K)

J_full = Derivative(jac_full_lw, 0, mode="fwd", chunk_size=1)(x_lw)
J_full.block_until_ready()
print(f"  J_full shape: {J_full.shape}")

for K in [10, 20, 50]:
    J_pruned = Derivative(lambda x: jac_pruned_lw(x, K), 0, mode="fwd", chunk_size=1)(x_lw)
    J_pruned.block_until_ready()
    diff = jnp.abs(J_pruned - J_full)
    max_diff = float(jnp.max(diff))
    print(f"  K={K:>3d}: max_jac_diff={max_diff:.2e}")

# ============================================================
# Part 2: Full resolution benchmarks
# ============================================================
print(f"\n=== Part 2: Full Resolution Benchmarks ===")
config_fr = build_config(platform="perlmutter", lightweight=False, seed=0)
data_fr = load_data(config_fr)
coilset_fr = data_fr["shaping"]
grid_fr = LinearGrid(N=config_fr["grid"]["coil_grid_N"])

pts_fr = coilset_fr._compute_position(
    params=coilset_fr.params_dict, grid=grid_fr, basis="xyz"
)
x_fr = coilset_fr.pack_params(coilset_fr.params_dict)
print(f"Full res: {pts_fr.shape[0]} effective coils, {pts_fr.shape[1]} nodes, "
      f"{len(x_fr)} params")

# Forward pass timing
print("\nForward pass timing:")
for K in [10, 20, 50, 100]:
    f = partial(compute_min_distance_pruned, alpha=alpha, K=K)
    _ = f(pts_fr); _.block_until_ready()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        d = f(pts_fr); d.block_until_ready()
        times.append(time.perf_counter() - t0)
    print(f"  K={K:>3d}: {np.median(times)*1000:.1f}ms")

# DESC's chunked implementation for comparison
print("\nDESC CoilSetMinDistance (dist_chunk_size):")
for dc in [10, 50]:
    obj = CoilSetMinDistance(
        coilset_fr, bounds=(0.3, np.inf),
        use_softmin=True, softmin_alpha=alpha,
        grid=grid_fr, dist_chunk_size=dc,
    )
    obj.build(use_jit=True, verbose=0)
    params = coilset_fr.params_dict
    const = obj.constants
    _ = obj.compute(params, const); _.block_until_ready()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        d = obj.compute(params, const); d.block_until_ready()
        times.append(time.perf_counter() - t0)
    label = f"dist_chunk={dc}" if dc else "dist_chunk=None"
    print(f"  {label:<20s}: {np.median(times)*1000:.1f}ms")

# Jacobian timing at full resolution
print("\nJacobian timing (full res):")

def jac_pruned_fr(x, K):
    params = coilset_fr.unpack_params(x)
    pts = coilset_fr._compute_position(params=params, grid=grid_fr, basis="xyz")
    return compute_min_distance_pruned(pts, alpha, K)

for K in [20, 50]:
    for chunk in [1, 4, 8, 16]:
        try:
            jac_fn = Derivative(lambda x: jac_pruned_fr(x, K), 0, mode="fwd", chunk_size=chunk)
            J = jac_fn(x_fr); J.block_until_ready()
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                J = jac_fn(x_fr); J.block_until_ready()
                times.append(time.perf_counter() - t0)
            print(f"  K={K:>3d}, chunk={chunk:>3d}: {np.median(times):.2f}s  shape={J.shape}")
        except Exception as e:
            print(f"  K={K:>3d}, chunk={chunk:>3d}: OOM")

# Compare to DESC's CoilSetMinDistance Jacobian
print("\nDESC CoilSetMinDistance Jacobian (batched mode):")
for chunk in [1, 4]:
    obj_fn = ObjectiveFunction(
        (CoilSetMinDistance(
            coilset_fr, bounds=(0.3, np.inf),
            use_softmin=True, softmin_alpha=alpha,
            grid=grid_fr,
        ),),
        deriv_mode="batched", jac_chunk_size=chunk,
    )
    obj_fn.build(verbose=0)
    x_desc = obj_fn.x(coilset_fr)
    J = obj_fn.jac_scaled(x_desc); J.block_until_ready()
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        J = obj_fn.jac_scaled(x_desc); J.block_until_ready()
        times.append(time.perf_counter() - t0)
    print(f"  DESC chunk={chunk}: {np.median(times):.2f}s  shape={J.shape}")
