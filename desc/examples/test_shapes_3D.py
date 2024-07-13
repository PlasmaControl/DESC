#!/usr/bin/env python3
"""
Test file for 3D FE class
"""
import time
import numpy as np
from matplotlib import pyplot as plt

from pyevtk.hl import gridToVTK, pointsToVTK, unstructuredGridToVTK
from pyevtk.vtk import VtkTetra

from desc.basis import FiniteElementBasis, FourierZernikeBasis
from desc.geometry import convert_spectral_to_FE

np.random.seed(1)
L = 2
M = 5
N = 4
K = 2

# Make a surface in (R, phi=0, Z), (R, phi=pi / N, Z), ...
nt = 30
theta = np.linspace(0.0, 2 * np.pi, nt)
zeta = np.linspace(0.0, 2 * np.pi, nt)

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
amp = 2
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
M_FE = 10
L_FE = 2
N_FE = 10
rho = np.linspace(0.0, 1, L_FE, endpoint=True)

# Set nodes
nodes = (
    np.array(np.meshgrid(rho, theta, zeta, indexing="ij"))
    .reshape(3, len(rho) * nt**2)
    .T
)
# Evaluate
R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
contig = np.ascontiguousarray
X = R * np.cos(nodes[:, -1])
Y = R * np.sin(nodes[:, -1])
X = X.reshape(1, 1, X.shape[0])
Y = Y.reshape(1, 1, Y.shape[0])
Z = Z.reshape(1, 1, Z.shape[0])
pointData = {"dummy_variable": np.ones((1, 1, X.shape[-1]))}
for ll in range(L_FE):
    XX = X[:, :, X.shape[-1] // L_FE * ll : X.shape[-1] // L_FE * (ll + 1)]
    YY = Y[:, :, Y.shape[-1] // L_FE * ll : Y.shape[-1] // L_FE * (ll + 1)]
    ZZ = Z[:, :, Z.shape[-1] // L_FE * ll : Z.shape[-1] // L_FE * (ll + 1)]
    gridToVTK(
        "RZ_Fourier_basis_surf" + str(ll),
        contig(XX),
        contig(YY),
        contig(ZZ),
        pointData=pointData
    )
pointsToVTK('RZ_Fourier_basis', contig(X), contig(Y), contig(Z))
gridToVTK('RZ_Fourier_basis', contig(X), contig(Y), contig(Z), pointData=pointData)

# Plot the DESC 3D case
ax.scatter(np.ravel(X), np.ravel(Y), np.ravel(Z), label="DESC rep")

# Next, move on to FE 3D case
Rprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N_FE, K=K, rho_range=rho)
Zprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N_FE, K=K, rho_range=rho)
Lprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N_FE, K=K, rho_range=rho)

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
print("Done with conversion")
t1 = time.time()
Rprime_basis.R_lmn = Rprime_lmn
Zprime_basis.Z_lmn = Zprime_lmn

# Replot the surface in the finite element basis
nmodes = len(Rprime_basis.modes)
t1 = time.time()
R_FE = Rprime_basis.evaluate(nodes=nodes) @ Rprime_lmn
Z_FE = Zprime_basis.evaluate(nodes=nodes) @ Zprime_lmn
t2 = time.time()
print('Time to compute R and Z (evaluate the basis functions at the nodes) = ',
      t2 - t1)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
contig = np.ascontiguousarray
X_FE = R_FE * np.cos(nodes[:, -1])
Y_FE = R_FE * np.sin(nodes[:, -1])
X_FE = X_FE.reshape(1, 1, X_FE.shape[0])
Y_FE = Y_FE.reshape(1, 1, Y_FE.shape[0])
Z_FE = Z_FE.reshape(1, 1, Z_FE.shape[0])
pointData = {"dummy_variable": np.ones((1, 1, X.shape[-1]))}
for ll in range(L_FE):
    XX = X_FE[:, :, X_FE.shape[-1] // L_FE * ll : X_FE.shape[-1] // L_FE * (ll + 1)]
    YY = Y_FE[:, :, Y_FE.shape[-1] // L_FE * ll : Y_FE.shape[-1] // L_FE * (ll + 1)]
    ZZ = Z_FE[:, :, Z_FE.shape[-1] // L_FE * ll : Z_FE.shape[-1] // L_FE * (ll + 1)]
    gridToVTK(
        "RZ_FE_basis_surf" + str(ll),
        contig(XX),
        contig(YY),
        contig(ZZ),
        pointData=pointData
    )
pointsToVTK('RZ_FE_basis', contig(X_FE), contig(Y_FE), contig(Z_FE))
gridToVTK('RZ_FE_basis', contig(X_FE), contig(Y_FE), contig(Z_FE), pointData=pointData)

ax.scatter(np.ravel(X_FE), np.ravel(Y_FE), np.ravel(Z_FE), marker='x', label="FE rep")
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')
ax.set_zlabel(r'$Z$')
plt.legend()
plt.grid()
# plt.show()

def _tetrahedra_to_vtk(
    filename,
    points,
    cellData=None,
    pointData=None,
    fieldData=None,
):
    """
    Write a VTK file showing the individual voxel cubes.

    Args:
    -------
    filename: Name of the file to write.
    points: Array of size ``(nvoxels, nquadpoints, 3)``
        Here, ``nvoxels`` is the number of voxels.
        The last array dimension corresponds to Cartesian coordinates.
        The max and min over the ``nquadpoints`` dimension will be used to
        define the max and min coordinates of each voxel.
    cellData: Data for each voxel to pass to ``pyevtk.hl.unstructuredGridToVTK``.
    pointData: Data for each voxel's vertices to pass to ``pyevtk.hl.unstructuredGridToVTK``.
    fieldData: Data for each voxel to pass to ``pyevtk.hl.unstructuredGridToVTK``.
    """
    # Adapted from a function from Matt L:
    # Some references Matt L. used while writing this function:
    # https://vtk.org/doc/nightly/html/classvtkVoxel.html
    # https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png
    # https://github.com/pyscience-projects/pyevtk/blob/v1.2.0/pyevtk/hl.py
    # https://python.hotexamples.com/examples/vtk/-/vtkVoxel/python-vtkvoxel-function-examples.html

    nvoxels = points.shape[0]
    nvertices = points.shape[1]

    cell_types = np.empty(nvoxels, dtype="uint8")
    cell_types[:] = VtkTetra.tid
    connectivity = np.arange(4 * nvoxels, dtype=np.int64)
    offsets = (np.arange(nvoxels, dtype=np.int64) + 1) * 4

    x = np.zeros(4 * nvoxels)
    y = np.zeros(4 * nvoxels)
    z = np.zeros(4 * nvoxels)

    for j in range(nvoxels):
        x[4 * j : 4 * (j + 1)] = points[j, :, 0]
        y[4 * j : 4 * (j + 1)] = points[j, :, 1]
        z[4 * j : 4 * (j + 1)] = points[j, :, 2]

    unstructuredGridToVTK(
        filename,
        x,
        y,
        z,
        connectivity,
        offsets,
        cell_types,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
    )


# First plot FE basis in the (rho, theta, zeta) space
nodes = np.array(Rprime_basis.mesh.vertices_final)
_tetrahedra_to_vtk("finite_element_mesh", nodes)

mesh = Rprime_basis.mesh
quadpoints = np.array(mesh.return_quadrature_points())
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
nodes = nodes.reshape(-1, 3)
ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], marker='x')
ax.scatter(quadpoints[:, 0], quadpoints[:, 1], quadpoints[:, 2], marker='o')
plt.grid()
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\theta$')
ax.set_zlabel(r'$\zeta$')
# plt.show()
# print(nodes)
# nodes_xyz = np.zeros(nodes.shape)
# nodes_xyz[:, 0] = nodes[:, 0] * np.cos(nodes[:, -1])
# nodes_xyz[:, 1] = nodes[:, 0] * np.sin(nodes[:, -1])
# nodes_xyz[:, 2] = nodes[:, 2]

R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
X = R * np.cos(nodes[:, -1])
Y = R * np.sin(nodes[:, -1])
points_fourier = np.array([X, Y, Z]).T
Ks = np.arange(1, 3)
# Ls = np.arange(2, 8, 2)
Ms = np.arange(2, 28, 18)
Ns = np.arange(2, 28, 18)
errors = np.zeros((len(Ks), len(Ms), len(Ns)))
q = 0 
for k in Ks:
    qq = 0
    for m in Ms:
        qqq = 0
        for n in Ns:
            Rprime_basis = FiniteElementBasis(L=L_FE, M=m, N=n, K=k, rho_range=rho)
            Zprime_basis = FiniteElementBasis(L=L_FE, M=m, N=n, K=k, rho_range=rho)
            Lprime_basis = FiniteElementBasis(L=L_FE, M=m, N=n, K=k, rho_range=rho)

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
            XX = RR * np.cos(nodes[:, -1])
            YY = RR * np.sin(nodes[:, -1])
            points_FE = np.array([XX, YY, ZZ]).T
            print(points_fourier.shape, points_FE.shape)
            E = np.linalg.norm(points_fourier - points_FE, 2) / np.linalg.norm(points_fourier, 2)
            errors[q, qq, qqq] = E
            print(k, m, n, q, qq, qqq, E, np.log10(E))
            qqq += 1
        qq += 1
    q += 1

q = 0
plt.figure()
for k in Ks:
    plt.subplot(1, len(Ks), q + 1)
    plt.contourf(Ms, Ns, np.log10(errors[q, :, :]).T)  #, label=r'$\log_{10}(E)$')
    plt.xlabel('M')
    if q == 0:
        plt.ylabel('N')
    # plt.legend()
    plt.colorbar()
    plt.title('K = ' + str(k))
    q += 1
plt.show()
