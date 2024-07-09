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

from desc.basis import FiniteElementMesh2D

# Try triangulating whole mesh
M = 4
L = 3
mesh = FiniteElementMesh2D(L, M, K=1)
mesh.plot_triangles(plot_quadrature_points=True)
integral = mesh.integrate(np.ones((2 * (M - 1) * (L - 1) * mesh.nquad, 10000)))
area2_total = 0.0
for triangle in mesh.triangles:
    area2_total += triangle.area2
print(integral, area2_total)
assert np.allclose(integral, 2 * np.pi)

quadpoints = mesh.return_quadrature_points()
integral = mesh.integrate(
    np.array([np.cos(quadpoints[:, 1]), np.sin(quadpoints[:, 1])]).T
)
print(integral)
assert np.allclose(integral, 0.0)

integral = mesh.integrate(
    np.array([np.cos(quadpoints[:, 1]) ** 2, np.sin(quadpoints[:, 1]) ** 2]).T
)
print(integral)
assert np.allclose(integral, np.pi)

print(quadpoints, quadpoints.shape, np.unique(quadpoints[:, 0]),  np.unique(quadpoints[:, 0]).shape)
integral = mesh.integrate(
    np.array([quadpoints[:, 0] ** 2, quadpoints[:, 0] * np.sin(quadpoints[:, 1]) ** 2]).T
)
print(integral, integral[0] / (2 * np.pi))
assert np.isclose(integral[0] / (2 * np.pi), 1.0 / 3.0)
assert np.isclose(integral[1], np.pi / 2.0)
