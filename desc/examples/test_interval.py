#!/usr/bin/env python3
"""
Temporary testing file for the TriangleFiniteElement class.

Making a single triangle can be done with e.g.,
    vertices = np.array([[2 * np.pi / 3.0, np.pi / 3.0],
                         [2 * np.pi / 2.0, np.pi / 2.0], [0, np.pi / 4.0]])
    triangle = TriangleFiniteElement(vertices, K=2)
    plt.figure()
    triangle.plot_triangle()
    plt.show()
"""
import numpy as np
from matplotlib import pyplot as plt

from desc.basis import FiniteElementBasis, FiniteElementMesh1D, FourierZernikeBasis
from desc.geometry import convert_spectral_to_FE

M = 3  # Note M > 2 required
N = 0
L = 0
K = 1
mesh = FiniteElementMesh1D(M, K=K)
mesh.plot_intervals(plot_quadrature_points=True)
integral = mesh.integrate(np.ones((M * mesh.nquad, 10000)))
length_total = 0.0
for interval in mesh.intervals:
    length_total += interval.length
assert np.allclose(integral, 2 * np.pi)

quadpoints = mesh.return_quadrature_points()
integral = mesh.integrate(np.array([np.cos(quadpoints)]).T)
assert np.allclose(integral, 0.0)

# Make a surface in (R, phi=0, Z) plane.

## Number of modes for l here is probably wrong for the FourierZernike basis
R_lmn = np.zeros((L + 1, 2 * M + 1, 2 * N + 1))
R_lmn[0, 0, 0] = 2.0
R_lmn[0, 2, 0] = 1.0
Z_lmn = np.zeros((L + 1, 2 * M + 1, 2 * N + 1))
Z_lmn[0, 0, 0] = 2.0
Z_lmn[0, 1, 0] = 5.0
R_lmn = R_lmn.reshape((L + 1) * (2 * M + 1) * (2 * N + 1))
Z_lmn = Z_lmn.reshape((L + 1) * (2 * M + 1) * (2 * N + 1))
L_lmn = np.zeros(R_lmn.shape)

# Plot original boundary
fz = FourierZernikeBasis(L, M, N)
theta = np.linspace(0, 2 * np.pi, endpoint=True)
plt.figure()
plt.plot(2 + np.cos(theta), 2 + 5 * np.sin(theta))

# Define the bases
Rprime_basis = FiniteElementBasis(L=L, M=M, N=N, K=K)
Zprime_basis = FiniteElementBasis(L=L, M=M, N=N, K=K)
Lprime_basis = FiniteElementBasis(L=L, M=M, N=N, K=K)
R_basis = FourierZernikeBasis(
    L=L,
    M=M,
    N=N,
)
Z_basis = FourierZernikeBasis(
    L=L,
    M=M,
    N=N,
)
L_basis = FourierZernikeBasis(
    L=L,
    M=M,
    N=N,
)
R_basis.R_lmn = R_lmn
Z_basis.Z_lmn = Z_lmn
L_basis.L_lmn = L_lmn

# Replot original boundary using the Zernike polynomials
print("R_lmn, Z_lmn = ", R_basis.R_lmn, Z_basis.Z_lmn)
nodes = (
    np.array(np.meshgrid(np.ones(1), theta, np.zeros(1), indexing="ij"))
    .reshape(3, len(theta))
    .T
)
print(R_basis.modes)
print(nodes)

R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
plt.plot(R, Z, "ro")

Rprime_lmn, Zprime_lmn, Lprime_lmn = convert_spectral_to_FE(
    R_lmn,
    Z_lmn,
    L_lmn,
    R_basis,
    Z_basis,
    L_basis,
    Rprime_basis,
    Zprime_basis,
    Lprime_basis,
)
print("R_lmn, Z_lmn = ")
print(Rprime_lmn)
print(Zprime_lmn)
Rprime_basis.R_lmn = Rprime_lmn
Zprime_basis.Z_lmn = Zprime_lmn
R = Rprime_basis.evaluate(nodes=nodes) @ Rprime_lmn
Z = Zprime_basis.evaluate(nodes=nodes) @ Zprime_lmn
print(R, Z)
plt.plot(R, Z, "ko")
plt.show()
