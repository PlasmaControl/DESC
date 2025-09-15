#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from desc.geometry.surface import ChebyshevRZToroidalSurface
from desc.grid import ConcentricGrid, Grid, LinearGrid
from desc.plotting import *

# def chebygrid(N_grid):
#    return np.concatenate(
#        (
#            [0],
#            (-np.cos((2 * np.arange(N_grid) + 1) * np.pi / (2 * N_grid)) + 1) * np.pi,
#            [2 * np.pi],
#        )
#    )


# def grid_gen(L_grid, M_grid, N_grid, node_pattern="jacobi"):
#    LMnodes = ConcentricGrid(L=L_grid, M=M_grid, N=0, node_pattern=node_pattern).nodes[
#        :, :2
#    ]
#    Nnodes = chebygrid(N_grid)
#    lm = np.tile(LMnodes, (Nnodes.size, 1))
#    n = np.tile(Nnodes.reshape(-1, 1), (1, LMnodes.shape[0])).reshape(-1, 1)
#    nodes = np.concatenate((lm, n), axis=1)
#
#    # RG: weights and spacing defined here
#    # just for the sake of compilation. Must be checked
#    weights = np.ones(nodes.shape[0])
#    spacing = np.ones_like(nodes)
#
#    spacing[1:, 1] = np.diff(nodes[:, 1])
#    spacing[1:, 2] = np.diff(nodes[:, 2])
#
#    return Grid(nodes, spacing=spacing, weights=weights)

## test
surf = ChebyshevRZToroidalSurface(
    R_lmn=[10, 1, -0.5],
    modes_R=[[0, 0], [1, 0], [1, 2]],
    Z_lmn=[0, -1, 0.5],
    modes_Z=[[0, 0], [-1, 0], [-1, 2]],
    mirror=True,
)


grid = LinearGrid(
    rho=1.0,
    theta=np.linspace(0, 2 * np.pi, 10, endpoint=False),
    zeta=np.linspace(0, 2 * np.pi, 10, endpoint=False),
)

# fig, data = plot_3d(surf, name="R", return_data=True)
# fig.write_html("test.html")

fig, ax = plot_boundary(surf, plot_axis=False)
plt.show()

# import pdb
# pdb.set_trace()

# data = surf.compute(["R"], grid=grid)
# import numpy as np
#
#
# ntheta
#
# theta_1D = np.linspace(0, 2*np.pi, ntheta)
# zeta_1D = np.linspace(0, 2*np.pi, nzeta)
#
# theta_2D, zeta_2D =
