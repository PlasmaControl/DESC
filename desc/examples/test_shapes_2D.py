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
import time

from desc.basis import (
    FiniteElementBasis,
    FourierZernikeBasis,
)
from desc.geometry import convert_spectral_to_FE

np.random.seed(1)

L = 2
M = 5
N = 0
K = 2

# Make a surface in (R, phi=0, Z) plane.
# Plot original boundary
theta = np.linspace(0.0, 2 * np.pi, 200, endpoint=True)

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
R_lmn = R_lmn.reshape(-1) 
Z_lmn = np.zeros((num_modes, 2 * N + 1))
Z_lmn[0, 0] = 2.0
Z_lmn[1, 0] = 5.0
Z_lmn = Z_lmn.reshape(-1)  # num_modes * (2 * N + 1))
L_lmn = np.zeros(R_lmn.shape)
amp = 5
R_lmn[np.isclose(R_lmn, 0.0)] = (
    (np.random.rand(np.sum(np.isclose(R_lmn, 0.0))) - 0.5)
    * amp
    / np.arange(1, len(R_lmn[np.isclose(R_lmn, 0.0)]) + 1)
)
Z_lmn[np.isclose(Z_lmn, 0.0)] = (
    (np.random.rand(np.sum(np.isclose(Z_lmn, 0.0))) - 0.5)
    * amp
    / np.arange(1, len(R_lmn[np.isclose(Z_lmn, 0.0)]) + 1)
)

# Set the coefficients in the basis class
R_basis.R_lmn = R_lmn
Z_basis.Z_lmn = Z_lmn
L_basis.L_lmn = L_lmn

# Replot original boundary using the Zernike polynomials
M_FE = 60
L_FE = 20
rho = np.linspace(0.0, 1, L_FE, endpoint=True)
nodes = (
    np.array(np.meshgrid(rho, theta, np.zeros(1), indexing="ij"))
    .reshape(3, len(theta) * len(rho))
    .T
)
R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
plt.figure(10)
plt.plot(R, Z, "ro", label="DESC rep")

Rprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K, rho_range=rho)
Zprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K, rho_range=rho)
Lprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K, rho_range=rho)

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
# print('node_evaluate = ', Rprime_basis.evaluate(nodes=nodes))
RR = Rprime_basis.evaluate(nodes=nodes) @ Rprime_lmn
ZZ = Zprime_basis.evaluate(nodes=nodes) @ Zprime_lmn
# print('R, Z = ', R, Z)
t2 = time.time()
print('Time for R, Z conversion = ', t2 - t1)
plt.figure(10)
plt.plot(RR, ZZ, "ko", label="FE rep")
# plt.scatter(R, Z, s=np.arange(len(R)), marker="o", label="FE rep")
plt.legend()
plt.grid()
# Rprime_basis.mesh.plot_triangles(True)
# plt.show()
points_fourier = np.array([R, Z]).T

Ks = np.arange(1, 3)
# Ls = np.arange(2, 8, 2)
Ms = np.arange(4, 10, 4)
errors = np.zeros((len(Ks), len(Ms)))
q = 0 
for k in Ks:
    qq = 0
    for m in Ms:
        Rprime_basis = FiniteElementBasis(L=L_FE, M=m, N=N, K=k, rho_range=rho)
        Zprime_basis = FiniteElementBasis(L=L_FE, M=m, N=N, K=k, rho_range=rho)
        Lprime_basis = FiniteElementBasis(L=L_FE, M=m, N=N, K=k, rho_range=rho)

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
        RR = Rprime_basis.evaluate(nodes=nodes) @ Rprime_lmn
        ZZ = Zprime_basis.evaluate(nodes=nodes) @ Zprime_lmn
        points_FE = np.array([RR, ZZ]).T
        E = np.linalg.norm(points_fourier - points_FE, 2) / np.linalg.norm(points_fourier, 2)
        errors[q, qq] = E 
        print(q, qq, E, np.log10(errors[q, qq]))
        qq += 1
    q += 1

plt.figure()
plt.contourf(Ks, Ms, np.log10(errors).T)
plt.xlabel('K')
plt.ylabel('M')
plt.title(r'$\log_{10}(E)$')
plt.colorbar()
plt.show()
