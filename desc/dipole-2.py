from desc.dipole import _Dipole, DipoleSet, export_dipoles
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

    return np.mean(np.abs(Bn)) / normalizing_field


surf = eq.surface.constant_offset_surface(
    offset=0.05,
    M=2,
    N=eq.N,
    grid=LinearGrid(M=8, N=2 * eq.N, NFP=eq.NFP)
)

grid = LinearGrid(
    L=1,       # 1 radial index for a surface
    M=5,      # number of poloidal points
    N=5,      # number of toroidal points
    NFP=surf.NFP,
    sym=surf.sym,
)

surface_points = surf.compute(names=["R", "phi", "Z"], grid=grid, override_grid=False)

R_vals   = surface_points["R"]  
phi_vals = surface_points["phi"] 
Z_vals   = surface_points["Z"]

# cylindrical to xyz
X = R_vals * np.cos(phi_vals)
Y = R_vals * np.sin(phi_vals)
Z = Z_vals

XYZ_2 = np.array([X,Y,Z]).reshape((3,11,12)) # 2D array
axis = XYZ_2.mean(axis=2)

Ux,Uy,Uz = axis[:,:,np.newaxis] - XYZ_2 # (3,T,P) array pointing TOWARD axis
Ur = np.sqrt(Ux**2 + Uy**2)
M_phi = np.atan2(Uy, Ux).ravel()
M_theta = np.atan2(Ur, Uz).ravel() # polar angle (theta = 0) points up at x-axis

dipoles = []

for i in range(len(surface_points["R"])):
    d = _Dipole(x=X[i], y=Y[i], z=Z[i], phi=M_phi[i], theta=M_theta[i])
    dipoles.append(d)

one_period = DipoleSet(dipoles, NFP=eq.NFP, sym=eq.sym)
# dipole_set = DipoleSet.from_symmetry(one_period, NFP=eq.NFP, sym=eq.sym)


from desc.coils import initialize_modular_coils


coilset = initialize_modular_coils(eq, num_coils=3, r_over_a=3.0).to_FourierXYZ()


#export_dipoles(dipole_set, "dipoles.csv")

plotter = pv.Plotter()
b1 = compute_average_normalized_field(one_period, coilset, eq, plotter)
print(b1)
plotter = plot_dipoles(one_period, plotter=plotter)
plotter = plot_coils2(coilset, plotter=plotter)
plotter.view_xz()
plotter.show()

# plot surface, color surface with error field