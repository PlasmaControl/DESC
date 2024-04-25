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

from desc.basis import (
    FiniteElementBasis,
    FiniteElementMesh1D_scikit,
    FourierZernikeBasis,
)
from desc.geometry import convert_spectral_to_FE

plt.figure()
L = 10
M = 10
N = 0
K = 1

# Make a surface in (R, phi=0, Z) plane.
# Plot original boundary
theta = np.linspace(0, 2 * np.pi, 400, endpoint=True)

# Define the bases
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

# Count the number of modes in the FourierZernike basis
num_modes = 1  # Add in (L = 0, M = 0) mode
for lmode in range(1, M + 1):
    num_modes += min(lmode + 1, int(L / 2) * 2 + 2)

# Set the R_lmn and Z_lmn modes to reproduce the desired cross section
L_indexing = M + 1
R_lmn = np.zeros((num_modes, 2 * N + 1))
R_lmn[0, 0] = 2.0
R_lmn[2, 0] = 1.0
Z_lmn = np.zeros((num_modes, 2 * N + 1))
Z_lmn[0, 0] = 2.0
Z_lmn[1, 0] = 5.0
R_lmn = R_lmn.reshape(num_modes * (2 * N + 1))
Z_lmn = Z_lmn.reshape(num_modes * (2 * N + 1))
L_lmn = np.zeros(R_lmn.shape)
# amp = 2
# R_lmn[np.isclose(R_lmn, 0.0)] = (
#     (np.random.rand(np.sum(np.isclose(R_lmn, 0.0))) - 0.5)
#     * amp
#     / np.arange(1, len(R_lmn[np.isclose(R_lmn, 0.0)]) + 1)
# )
# Z_lmn[np.isclose(Z_lmn, 0.0)] = (
#     (np.random.rand(np.sum(np.isclose(Z_lmn, 0.0))) - 0.5)
#     * amp
#     / np.arange(1, len(R_lmn[np.isclose(Z_lmn, 0.0)]) + 1)
# )

# Set the coefficients in the basis class
R_basis.R_lmn = R_lmn
Z_basis.Z_lmn = Z_lmn
L_basis.L_lmn = L_lmn

# Replot original boundary using the Zernike polynomials
L_FE = 2
rho = np.linspace(0.1, 1, L_FE)
nodes = (
    np.array(np.meshgrid(rho, theta, np.zeros(1), indexing="ij"))
    .reshape(3, len(theta) * len(rho))
    .T
)
# print(R_basis.evaluate(nodes=nodes).shape, R_lmn.shape, nodes.shape)
R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
print("R_lmn, Z_lmn = ", R_lmn, Z_lmn)
plt.plot(R, Z, "ro", label="DESC rep")

print(M, L, N, K)
Rprime_basis = FiniteElementBasis(L=L_FE, M=M, N=N, K=K)
Zprime_basis = FiniteElementBasis(L=L_FE, M=M, N=N, K=K)
Lprime_basis = FiniteElementBasis(L=L_FE, M=M, N=N, K=K)

# Convert to the finite element basis
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
Rprime_basis.R_lmn = Rprime_lmn
Zprime_basis.Z_lmn = Zprime_lmn

# Replot the surface in the finite element basis
nmodes = len(Rprime_basis.modes)
R = Rprime_basis.evaluate(nodes=nodes) @ Rprime_lmn
Z = Zprime_basis.evaluate(nodes=nodes) @ Zprime_lmn
plt.plot(R, Z, "ko", label="FE rep")
plt.legend()
plt.grid()
plt.show()
