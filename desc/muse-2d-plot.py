import sys
import os

from desc.dipole import _Dipole, DipoleSet, export_dipoles, import_dipoles
from desc.grid import LinearGrid
import numpy as np
import pyvista as pv
from desc.integrals import compute_B_plasma
from desc.plotting import plot_dipoles, plot_3d, plot_comparison, plot_coils2
from desc.io import load
import pyvista as pv
eq = load('input.muse-fixedb_output.h5')[-1]


def compute_average_normalized_field(
    field, coils, eq, p, vacuum=False, chunk_size=None, B_plasma_chunk_size=None
):
    
    # compute Bn as error field on surface
    if B_plasma_chunk_size is None:
        B_plasma_chunk_size = chunk_size
    grid = LinearGrid(M=20, N=20, NFP=eq.NFP, endpoint=True)
    # if vacuum:
    #     # we can avoid the expensive plasma contribution calculation if we
    #     # just pass in the surface instead of the Equilibrium!
    #     Bn, surf_coords = field.compute_Bnormal(
    #         eq.surface,
    #         eval_grid=grid,
    #         chunk_size=chunk_size,
    #         B_plasma_chunk_size=B_plasma_chunk_size,
    #     )
    
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
    pgrid.point_data["bn"] = Bn / normalizing_field
    

    surf = pgrid.extract_surface()
    p.add_mesh(
        surf,
        scalars="bn",
        smooth_shading=True,
        show_edges=False,
    )
    
    import matplotlib.pyplot as plt

    theta = grid.nodes[:, 1].reshape(41, 41)
    zeta = grid.nodes[:, 2].reshape(41, 41)

    B_PM_dot_n = np.sum(B_PM * n_surf, axis=1).reshape(41, 41)
    B_TF_dot_n = np.sum(B_TF * n_surf, axis=1).reshape(41, 41)
    B_total_dot_n = Bn.reshape(41, 41)

    print(np.mean(np.abs(Bn)) / normalizing_field)

    plt.figure()
    plt.contourf(zeta, theta, B_PM_dot_n, levels=100)
    plt.colorbar()
    plt.title("PM")
    plt.xlabel("toroidal angle")
    plt.ylabel("poloidal angle")

    plt.figure()
    plt.contourf(zeta, theta, B_TF_dot_n, levels=100)
    plt.colorbar()
    plt.title("TF")
    plt.xlabel("toroidal angle")
    plt.ylabel("poloidal angle")

    plt.figure()
    plt.contourf(zeta, theta, B_total_dot_n, levels=100)
    plt.colorbar()
    plt.title("total")
    plt.xlabel("toroidal angle")
    plt.ylabel("poloidal angle")

    plt.show()

    return np.mean(np.abs(Bn)) / normalizing_field


coilset = load('tf_coils_desc.h5')
fig = pv.Plotter()
one_period = import_dipoles(eq, 'muse_dipoles_desc.csv')
#dipole_set = DipoleSet.from_symmetry(one_period, NFP=eq.NFP, sym=eq.sym)

b1 = compute_average_normalized_field(one_period, coilset, eq, fig)
#fig = plot_coils2(coilset, plotter=fig)
fig = plot_dipoles(one_period, plotter=fig)
fig.show()



