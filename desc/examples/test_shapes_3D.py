#!/usr/bin/env python3
"""
Test file for 3D FE class
"""
import numpy as np
from matplotlib import pyplot as plt
import time

from desc.basis import (
    FiniteElementBasis,
    FourierZernikeBasis,
)
from desc.geometry import convert_spectral_to_FE

L = 2
M = 5
N = 2
K = 2

# Make a surface in (R, phi=0, Z), (R, phi=pi / N, Z), ...
nt = 100
theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)
zeta = np.linspace(0, 2 * np.pi, nt, endpoint=False)

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
R_lmn = np.zeros(num_modes * (2 * N + 1))
R_lmn[num_modes * N] = 2.0
R_lmn[num_modes * N + 2] = 1.0
Z_lmn = np.zeros(num_modes * (2 * N + 1))
Z_lmn[num_modes * N] = 2.0
Z_lmn[num_modes * N + 1] = 5.0
L_lmn = np.zeros(R_lmn.shape)
# amp = 1
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
print(R_basis.modes, R_basis.modes[num_modes * N])
R_basis.R_lmn = R_lmn
Z_basis.Z_lmn = Z_lmn
L_basis.L_lmn = L_lmn

# Replot original boundary using the Zernike polynomials
M_FE = 10
L_FE = 2
rho = np.linspace(0.5, 1, L_FE, endpoint=True)
nodes = (
    np.array(np.meshgrid(rho, theta, np.zeros(1), indexing="ij"))
    .reshape(3, len(theta) * len(rho))
    .T
)
R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
plt.figure(10)
plt.plot(R, Z, "ro", label="DESC rep")

nodes = (
    np.array(np.meshgrid(rho, theta, np.ones(1) * np.pi, indexing="ij"))
    .reshape(3, len(theta) * len(rho))
    .T
)
R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
plt.figure(11)
plt.plot(R, Z, "ro", label="DESC rep")
plt.show()
# nodes = (
#     np.array(np.meshgrid(rho, theta, zeta, indexing="ij"))
#     .reshape(3, len(rho) * nt ** 2)
#     .T
# )
# R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
# Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
# plt.figure(12)
# plt.plot(R, Z, "ro", label="DESC rep")

Rprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K)
Zprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K)
Lprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K)

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
print('Done with conversion')
t1 = time.time()
Rprime_basis.R_lmn = Rprime_lmn
Zprime_basis.Z_lmn = Zprime_lmn

# Replot the surface in the finite element basis
nmodes = len(Rprime_basis.modes)
R = Rprime_basis.evaluate(nodes=nodes) @ Rprime_lmn
Z = Zprime_basis.evaluate(nodes=nodes) @ Zprime_lmn
t2 = time.time()
print('Time for R, Z conversion = ', t2 - t1)
plt.figure(10)
plt.plot(R, Z, "ko", label="FE rep")
plt.legend()
plt.grid()
plt.show()