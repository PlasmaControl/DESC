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

M = 20  # Note M > 2 required
N = 0
L = 0
K = 1
mesh = FiniteElementMesh1D(M, K=K)
integral = mesh.integrate(np.ones((M * mesh.nquad, 1000)))
length_total = 0.0
for interval in mesh.intervals:
    length_total += interval.length
assert np.allclose(integral, 2 * np.pi)

quadpoints = mesh.return_quadrature_points()
integral = mesh.integrate(np.array([np.cos(quadpoints)]).T)
assert np.allclose(integral, 0.0)
integral = mesh.integrate(np.array([np.cos(quadpoints) ** 2]).T)
assert np.allclose(integral, np.pi)

# Define an interval in theta
theta = np.linspace(0, 2 * np.pi, endpoint=False)
# Initialize the FourierZernike bases
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

# Set the coefficients in the basis class
R_basis.R_lmn = R_lmn
Z_basis.Z_lmn = Z_lmn
L_basis.L_lmn = L_lmn

# Replot original boundary using the Zernike polynomials
nodes = (
    np.array(np.meshgrid(np.ones(1), theta, np.zeros(1), indexing="ij"))
    .reshape(3, len(theta))
    .T
)
R_fourier = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z_fourier = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
print("R_lmn, Z_lmn = ", R_lmn, Z_lmn)

# Initialize a FE basis
Rprime_basis = FiniteElementBasis(L=L, M=M, N=N, K=K)
Zprime_basis = FiniteElementBasis(L=L, M=M, N=N, K=K)
Lprime_basis = FiniteElementBasis(L=L, M=M, N=N, K=K)

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
R = Rprime_basis.evaluate(nodes=nodes) @ Rprime_lmn
Z = Zprime_basis.evaluate(nodes=nodes) @ Zprime_lmn

# Plot original boundary and difference between Fourier and FE
# representation of 1D surface in theta
plt.figure()
plt.plot(2 + np.cos(theta), 2 + 5 * np.sin(theta))
plt.plot(R_fourier, Z_fourier, "ro")
plt.plot(R, Z, "ko")
plt.grid()

# Plot intervals and basis functions
mesh.plot_intervals()
plt.show()
