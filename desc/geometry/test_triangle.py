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

from desc.geometry import TriangleFiniteElement

# Try triangulating whole mesh
M = 8
N = 8
theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
zeta = np.linspace(0, 2 * np.pi, N, endpoint=False)
Theta, Zeta = np.meshgrid(theta, zeta, indexing="ij")

# Compute the vertices for all 2NM triangles
vertices = np.zeros((2 * N * M, 3, 2))
triangles = []
for i in range(M):
    for j in range(N):
        # Deal with the periodic boundary conditions...
        if i + 1 != M and j + 1 != N:
            rect_x1 = np.array([Theta[i, j], Zeta[i, j]])
            rect_x2 = np.array([Theta[i + 1, j], Zeta[i, j]])
            rect_x3 = np.array([Theta[i, j], Zeta[i, j + 1]])
            rect_x4 = np.array([Theta[i + 1, j], Zeta[i, j + 1]])
        elif i + 1 == M and j + 1 != N:
            rect_x1 = np.array([Theta[i, j], Zeta[i, j]])
            rect_x2 = np.array([2 * np.pi, Zeta[i, j]])
            rect_x3 = np.array([Theta[i, j], Zeta[i, j + 1]])
            rect_x4 = np.array([2 * np.pi, Zeta[i, j + 1]])
        elif i + 1 != M and j + 1 == N:
            rect_x1 = np.array([Theta[i, j], Zeta[i, j]])
            rect_x2 = np.array([Theta[i + 1, j], Zeta[i, j]])
            rect_x3 = np.array([Theta[i, j], 2 * np.pi])
            rect_x4 = np.array([Theta[i + 1, j], 2 * np.pi])
        elif i + 1 == M and j + 1 == N:
            rect_x1 = np.array([Theta[i, j], Zeta[i, j]])
            rect_x2 = np.array([2 * np.pi, Zeta[i, j]])
            rect_x3 = np.array([Theta[i, j], 2 * np.pi])
            rect_x4 = np.array([2 * np.pi, 2 * np.pi])

        # Split the (i, j)-th rectangle into two equal-sized triangles
        vertices[i * j * 2, 0, :] = rect_x1
        vertices[i * j * 2, 1, :] = rect_x2
        vertices[i * j * 2, 2, :] = rect_x3
        vertices[i * j * 2 + 1, 0, :] = rect_x4
        vertices[i * j * 2 + 1, 1, :] = rect_x3
        vertices[i * j * 2 + 1, 2, :] = rect_x2
        plt.figure(1)
        triangle1 = TriangleFiniteElement(vertices[i * j * 2, :, :], K=2)
        triangle1.plot_triangle()
        triangle2 = TriangleFiniteElement(vertices[i * j * 2 + 1, :, :], K=2)
        triangle2.plot_triangle()
        triangles.append(triangle1)
        triangles.append(triangle2)
plt.show()
