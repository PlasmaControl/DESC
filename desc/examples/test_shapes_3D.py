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
from pyevtk.hl import pointsToVTK, gridToVTK, unstructuredGridToVTK
from pyevtk.vtk import VtkTetra
from desc.geometry import convert_spectral_to_FE
from mpl_toolkits import mplot3d as p3

np.random.seed(1)
L = 2
M = 5
N = 2
K = 1

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
amp = 0
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
print(R_basis.modes, R_basis.modes[num_modes * N])
R_basis.R_lmn = R_lmn
Z_basis.Z_lmn = Z_lmn
L_basis.L_lmn = L_lmn

# Replot original boundary using the Zernike polynomials
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
M_FE = 10
L_FE = 2
rho = np.linspace(0.1, 1, L_FE, endpoint=True)
nodes = (
    np.array(np.meshgrid(rho, theta, np.zeros(1), indexing="ij"))
    .reshape(3, len(theta) * len(rho))
    .T
)
R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
X = R * np.cos(nodes[:, -1])
Y = R * np.sin(nodes[:, -1])
X = X.reshape(1, 1, X.shape[0])
Y = Y.reshape(1, 1, Y.shape[0])
Z = Z.reshape(1, 1, Z.shape[0])
#plt.figure(10) 
#ax = p3.Axes3D(fig)
plt.figure(10)
ax.scatter(np.ravel(X),np.ravel(Y),np.ravel(Z))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
#ax.scatter(R,nodes[:,-1], Z, label="DESC rep")
plt.legend()
plt.grid()
plt.show()

nodes = (
    np.array(np.meshgrid(rho, theta, np.ones(1) * np.pi, indexing="ij"))
    .reshape(3, len(theta) * len(rho))
    .T
)
# R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
# Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
# X = R * np.cos(nodes[:, -1])
# Y = R * np.sin(nodes[:, -1])
# X = X.reshape(1, 1, X.shape[0])
# Y = Y.reshape(1, 1, Y.shape[0])
# Z = Z.reshape(1, 1, Z.shape[0])
# plt.figure(11)
# plt.plot(X, Y, Z, label="DESC rep")
#"ro"

# Going to attempt to do 3d plot in rho,theta,zeta space.


nodes = (
    np.array(np.meshgrid(rho, theta, zeta, indexing="ij"))
    .reshape(3, len(rho) * nt ** 2)
    .T
)
R = R_basis.evaluate(nodes=nodes) @ R_basis.R_lmn
Z = Z_basis.evaluate(nodes=nodes) @ Z_basis.Z_lmn
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#Gonna pause this for now so I can find problem with finite element plot
#ax.scatter(R, nodes[:, -1], Z, label="DESC rep")
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
    gridToVTK('RZ_Fourier_basis_surf' + str(ll), contig(XX), contig(YY), contig(ZZ), pointData=pointData)
# pointsToVTK('RZ_Fourier_basis', X, Y, contig(Z))

Rprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K,rho_range=rho)
Zprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K,rho_range=rho)
Lprime_basis = FiniteElementBasis(L=L_FE, M=M_FE, N=N, K=K,rho_range=rho)

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

nt = 100
theta = np.linspace(0, 2 * np.pi, nt, endpoint=True)
zeta = np.linspace(0, 2 * np.pi, nt, endpoint=True)

# Replot the surface in the finite element basis
nmodes = len(Rprime_basis.modes)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
R = Rprime_basis.evaluate(nodes=nodes) @ Rprime_lmn
Z = Zprime_basis.evaluate(nodes=nodes) @ Zprime_lmn
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
    gridToVTK('RZ_FE_basis_surf' + str(ll), contig(XX), contig(YY), contig(ZZ), pointData=pointData)
# pointsToVTK('RZ_FE_basis', X, Y, contig(Z))



t2 = time.time()
#print(R.shape, Z.shape)
#print('Time for R, Z conversion = ', t2 - t1)
#plt.figure(11) 
#ax = p3.Axes3D(fig)
ax.scatter(np.ravel(X),np.ravel(Y),np.ravel(Z))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.add_axes(ax)
#ax.scatter(R,nodes[:,-1], Z, label="DESC rep")
plt.legend()
plt.grid()
plt.show()


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
    # Some references Matt L. used while writing this function:
    # https://vtk.org/doc/nightly/html/classvtkVoxel.html
    # https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png
    # https://github.com/pyscience-projects/pyevtk/blob/v1.2.0/pyevtk/hl.py
    # https://python.hotexamples.com/examples/vtk/-/vtkVoxel/python-vtkvoxel-function-examples.html

    # assert points.ndim == 2
    nvoxels = points.shape[0]
    nvertices = points.shape[1]
    # contig = np.ascontiguousarray
    # points = contig(points)
    # assert points.shape[2] == 3

    cell_types = np.empty(nvoxels, dtype="uint8")
    cell_types[:] = VtkTetra.tid
    connectivity = np.arange(4 * nvoxels, dtype=np.int64)
    offsets = (np.arange(nvoxels, dtype=np.int64) + 1) * 4
    print(offsets, nvoxels)

    # base_x = np.array([0, 1, 1, 0, 2, 3, 3, 2])
    # base_y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    # base_z = np.array([0, 0, 0, 1, 2, 2, 2, 3])
    
    x = np.zeros(4 * nvoxels)
    y = np.zeros(4 * nvoxels)
    z = np.zeros(4 * nvoxels)

    for j in range(nvoxels):
        x[4 * j: 4 * (j + 1)] = points[j, :, 0]
        y[4 * j: 4 * (j + 1)] = points[j, :, 1]
        z[4 * j: 4 * (j + 1)] = points[j, :, 2]

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
print(nodes.shape)
_tetrahedra_to_vtk(
    'finite_element_mesh',
    nodes
)
# print(nodes)
# nodes_xyz = np.zeros(nodes.shape)
# nodes_xyz[:, 0] = nodes[:, 0] * np.cos(nodes[:, -1])
# nodes_xyz[:, 1] = nodes[:, 0] * np.sin(nodes[:, -1])
# nodes_xyz[:, 2] = nodes[:, 2]