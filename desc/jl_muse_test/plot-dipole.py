import numpy as np
import pyvista as pv

import sys
fin = sys.argv[1]

data = np.genfromtxt(fin, delimiter=',', skip_header=1)
x,y,z,m,phi,theta = data.T


cloud = pv.PolyData(np.transpose([x,y,z]))

# optionally attach a scalar for coloring
#cloud["values"] = some_scalar_array  # shape (10000,)

pl = pv.Plotter()
pl.add_points(
    cloud,
    #    scalars="values",
    point_size=5,
    render_points_as_spheres=True,
    cmap="viridis",
)

pl.add_scalar_bar()
pl.show()
