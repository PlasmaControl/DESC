import numpy as np
import pyvista as pv
from desc.dipole import _Dipole, DipoleSet, export_dipoles, import_dipoles
from desc.grid import LinearGrid
from desc.integrals import compute_B_plasma
from desc.plotting import plot_dipoles, plot_3d, plot_comparison, plot_coils2
from desc.io import load
import sys


eq = load('input.muse-fixedb_output.h5')[-1]


def compute_average_normalized_field(
    field, coils, eq, p, vacuum=False, chunk_size=None, B_plasma_chunk_size=None
):
    
    # compute Bn as error field on surface
    if B_plasma_chunk_size is None:
        B_plasma_chunk_size = chunk_size
    grid = LinearGrid(M=20, N=20, NFP=eq.NFP, endpoint=True)
    
    surf_coords = eq.surface.compute(['R', 'phi', 'Z'], grid=grid)

    # asked chatgpt to make the surf coords array
    surf_coords_array = np.column_stack([surf_coords['R'], surf_coords['phi'], surf_coords['Z']])
    n_surf = eq.surface.compute(['n_rho'], grid=grid)['n_rho']
    
    B_PM = field.compute_magnetic_field(surf_coords_array)
    
    B_TF = coils.compute_magnetic_field(surf_coords_array)
    
    B_total = B_PM + B_TF
    
    Bn = np.sum(B_total * n_surf, axis=1)
    
    # compute normalization
    normalizing_field_vec = B_total
    if not vacuum:
        # add plasma field to the normalizing field
        normalizing_field_vec += compute_B_plasma(
            eq, eval_grid=grid, chunk_size=B_plasma_chunk_size
        )
    
    normalizing_field = np.mean(np.linalg.norm(normalizing_field_vec, axis=1))

    ### total objective: B_normal = [ (B_PM + B_TF) dot n_surf / norm]**2 <- make this zero

    # make plot
    data = eq.surface.compute(['X','Y','Z'], grid=grid)
    X = np.asarray(data['X']).reshape(41,41)
    Y = np.asarray(data['Y']).reshape(41,41)
    Z = np.asarray(data['Z']).reshape(41,41)

    pgrid = pv.StructuredGrid(X,Y,Z)

    B_PM_dot_n = np.sum(B_PM * n_surf, axis=1).reshape(41, 41)
    B_TF_dot_n = np.sum(B_TF * n_surf, axis=1).reshape(41, 41)
    pgrid["bn_error"] = (B_PM_dot_n+B_TF_dot_n).flatten()
    

    # surf = pgrid.extract_surface()
    # p.add_mesh(
    #     surf,
    #     scalars="bn_error",
    #     map="viridis"
    # )
    

    pl.add_mesh(pgrid, scalars="bn_error", cmap="viridis")

    return np.mean(np.abs(Bn)) / normalizing_field



fin = sys.argv[1]

data = np.genfromtxt(fin, delimiter=',', skip_header=1)
x,y,z,m,phi,theta = data.T

idx_nonzero = np.argwhere(np.array(m!=0, int))[:,0]
points = np.transpose([x,y,z])[idx_nonzero]
cloud = pv.PolyData(points)

# optionally attach a scalar for coloring
#cloud["values"] = some_scalar_array  # shape (10000,)

coilset = load('tf_coils_desc.h5')
one_period = import_dipoles(eq, fin)

pl = pv.Plotter()
b1 = compute_average_normalized_field(one_period, coilset, eq, pl)
pl.add_points(
    cloud,
    #    scalars="values",
    point_size=8,
    render_points_as_spheres=True,
    cmap="viridis",
)

pl.add_scalar_bar()
pl.show()
