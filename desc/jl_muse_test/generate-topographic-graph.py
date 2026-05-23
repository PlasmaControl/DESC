import sys
import os

from desc.dipole import _Dipole, DipoleSet, export_dipoles, import_dipoles
from desc.grid import LinearGrid
import numpy as np
import pyvista as pv
from desc.integrals import compute_B_plasma
from desc.io import load
import pyvista as pv
eq = load('input.muse-fixedb_output.h5')[-1]


def compute_average_normalized_field(
    field, coils, eq, p, vacuum=False, chunk_size=None, B_plasma_chunk_size=None
):
    if B_plasma_chunk_size is None:
        B_plasma_chunk_size = chunk_size
    grid = LinearGrid(M=20, N=20, NFP=eq.NFP, endpoint=True, sym=eq.sym)
    
    surf_coords = eq.surface.compute(['R', 'phi', 'Z'], grid=grid)

    surf_coords_array = np.column_stack([surf_coords['R'], surf_coords['phi'], surf_coords['Z']])
    n_surf = eq.surface.compute(['n_rho'], grid=grid)['n_rho']
    
    B_PM = field.compute_magnetic_field(surf_coords_array)
    
    B_TF = coils.compute_magnetic_field(surf_coords_array)
    
    B_total = B_PM + B_TF
    
    Bn = np.sum(B_total * n_surf, axis=1)
    
    normalizing_field_vec = B_total
    if not vacuum:
        normalizing_field_vec += compute_B_plasma(
            eq, eval_grid=grid, chunk_size=B_plasma_chunk_size
        )
    
    normalizing_field = np.mean(np.linalg.norm(normalizing_field_vec, axis=1))

    data = eq.surface.compute(['X','Y','Z'], grid=grid)
    X = np.asarray(data['X']).reshape(41,21)
    Y = np.asarray(data['Y']).reshape(41,21)
    Z = np.asarray(data['Z']).reshape(41,21)

    pgrid = pv.StructuredGrid(X,Y,Z)
    pgrid.point_data["bn"] = Bn / normalizing_field
    

    surf = pgrid.extract_surface()
    p.add_mesh(
        surf,
        scalars="bn",
        smooth_shading=True,
        show_edges=False,
    )
    
    import matplotlib.pyplot as plt

    theta = grid.nodes[:, 1].reshape(41,21)
    phi = grid.nodes[:, 2].reshape(41,21)

    B_PM_dot_n = np.sum(B_PM * n_surf, axis=1).reshape(41,21)
    B_TF_dot_n = np.sum(B_TF * n_surf, axis=1).reshape(41,21)
    B_total_dot_n = Bn.reshape(41,21)

    idx = np.unravel_index(np.argmax(np.abs(B_total_dot_n)), B_total_dot_n.shape)

    theta_max = theta[idx]
    phi_max = phi[idx]
    Bn_max = B_total_dot_n[idx]

    print(f"theta = {theta_max}")
    print(f"phi   = {phi_max}")
    print(f"Bn    = {Bn_max}")
    print(f"|Bn| normalized = {abs(Bn_max)/normalizing_field}")

    print(np.mean(np.abs(Bn)) / normalizing_field)

    topo_grid = pv.StructuredGrid(np.asarray(phi, dtype=float), np.asarray(theta, dtype=float), 70 * np.asarray(B_total_dot_n, dtype=float))
    topo_grid["Bn"] = B_total_dot_n.flatten(order="F")
    topo_plotter = pv.Plotter()
    topo_plotter.add_mesh(
        topo_grid,
        scalars="Bn",
        cmap="coolwarm",
        show_edges=False,
    )

    topo_plotter.add_axes()
    topo_plotter.show()


    return np.mean(np.abs(Bn)) / normalizing_field


coilset = load('tf_coils_desc.h5')
fig = pv.Plotter()
one_period = import_dipoles(eq, 'muse_dipoles_desc.csv')

b1 = compute_average_normalized_field(one_period, coilset, eq, fig)

