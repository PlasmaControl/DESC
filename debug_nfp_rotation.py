"""Quick diagnostic: is NFP rotation actually changing the result?"""
import numpy as np
import jax
from desc.backend import jnp
from desc.coils import CoilSet, FourierPlanarCoil, SplineXYZCoil

# One coil, eval near it
coil = FourierPlanarCoil(current=1e5, center=[10, 0, 0], r_n=1.0)
eval_xyz = np.array([[10.0, 0.0, 0.5], [0.0, 10.0, 0.5], [10.0, 0.0, -0.5]])

# NFP=1: just 1 copy
cs1 = CoilSet(coil, NFP=1, sym=False)
B1 = np.asarray(cs1.compute_magnetic_field(eval_xyz, basis="xyz"))

# NFP=4: should have 4 copies
cs4 = CoilSet(coil, NFP=4, sym=False)
B4 = np.asarray(cs4.compute_magnetic_field(eval_xyz, basis="xyz"))

# Single coil directly (no CoilSet)
B_single = np.asarray(coil.compute_magnetic_field(eval_xyz, basis="xyz"))

print("Single coil (no CoilSet):")
print(B_single)
print()
print("CoilSet NFP=1:")
print(B1)
print()
print("CoilSet NFP=4:")
print(B4)
print()
print(f"B1 == B_single? max diff = {np.max(np.abs(B1 - B_single)):.2e}")
print(f"B4 == B1? max diff = {np.max(np.abs(B4 - B1)):.2e}")
print(f"B4 / B1 ratios:")
for i in range(3):
    print(f"  pt[{i}]: {B4[i] / B1[i]}")

# Now test with SplineXYZCoil
theta = np.linspace(0, 2*np.pi, 32, endpoint=False)
X = 10 + np.cos(theta)
Y = np.sin(theta)
Z = np.zeros(32)
spline_coil = SplineXYZCoil(1e5, X=X, Y=Y, Z=Z)

cs1_hh = CoilSet(spline_coil, NFP=1, sym=False)
cs4_hh = CoilSet(spline_coil, NFP=4, sym=False)
B1_hh = np.asarray(cs1_hh.compute_magnetic_field(eval_xyz, basis="xyz"))
B4_hh = np.asarray(cs4_hh.compute_magnetic_field(eval_xyz, basis="xyz"))
B_single_hh = np.asarray(spline_coil.compute_magnetic_field(eval_xyz, basis="xyz"))

print()
print("=== SplineXYZCoil ===")
print(f"Single coil: {B_single_hh}")
print(f"CoilSet NFP=1: {B1_hh}")
print(f"CoilSet NFP=4: {B4_hh}")
print(f"B1 == B_single? max diff = {np.max(np.abs(B1_hh - B_single_hh)):.2e}")
print(f"B4 == B1? max diff = {np.max(np.abs(B4_hh - B1_hh)):.2e}")

# Reference: manually compute NFP=4 by creating 4 coils at different angles
# Rotate coil geometry by k*90 degrees
print()
print("=== Manual NFP=4 reference ===")
B_manual = np.zeros_like(eval_xyz)
for k in range(4):
    angle = 2 * np.pi * k / 4
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # Rotate coil geometry
    X_rot = X * c - Y * s
    Y_rot = X * s + Y * c
    Z_rot = Z.copy()
    rotated_coil = SplineXYZCoil(1e5, X=X_rot, Y=Y_rot, Z=Z_rot)
    B_k = np.asarray(rotated_coil.compute_magnetic_field(eval_xyz, basis="xyz"))
    print(f"  k={k} (angle={np.degrees(angle):.0f}): B={B_k[0]}")
    B_manual += B_k

print(f"Manual NFP=4 sum: {B_manual}")
print(f"Vectorized NFP=4: {np.asarray(B4_hh)}")
print(f"Match? max diff = {np.max(np.abs(B_manual - np.asarray(B4_hh))):.2e}")
