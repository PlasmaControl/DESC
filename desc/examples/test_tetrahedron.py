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

from desc.basis import FiniteElementMesh3D

# Try triangulating whole mesh
M = 5
L = 3
N = 5
K = 2
mesh = FiniteElementMesh3D(L, M, N, K=K)
# mesh.plot_tetrahedra()  # plot_quadrature_points=True)
integral = mesh.integrate(np.ones((6 * (M - 1) * (L - 1) * (N - 1) * mesh.nquad, 100)))
volume_total = 0.0
for tet in mesh.tetrahedra:
    volume_total += tet.volume6
print(integral[0], volume_total / 6.0)
assert np.allclose(integral, 4 * np.pi ** 2)

quadpoints = mesh.return_quadrature_points()
# print(quadpoints)
# print(np.array([np.cos(quadpoints[:, 1]), np.sin(quadpoints[:, 1])]).T.shape)
integral = mesh.integrate(
    np.array([np.cos(quadpoints[:, 1]), np.sin(quadpoints[:, 1])]).T
)
print(integral)
assert np.allclose(integral, 0.0)

integral = mesh.integrate(
    np.array([-0.5 + quadpoints[:, 0], 0.5 - quadpoints[:, 0]]).T
)
print(integral)
assert np.allclose(integral, 0.0)

integral = mesh.integrate(
    np.array([quadpoints[:, 0], quadpoints[:, 0] ** 2]).T
) / (4* np.pi ** 2)
print(integral)
assert np.isclose(integral[0], 0.5)
assert np.isclose(integral[1], 1.0 / 3.0)

print(quadpoints.shape)
integral = mesh.integrate(
    np.array([np.cos(quadpoints[:, 2]) ** 2, np.sin(quadpoints[:, 2])]).T
) / (2 * np.pi)
print(integral)
assert np.isclose(integral[0], np.pi)
assert np.isclose(integral[1], 0.0)

