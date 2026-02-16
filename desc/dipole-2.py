from desc.dipole import _Dipole, DipoleSet, export_dipoles
from desc.grid import LinearGrid
import numpy as np
import pyvista as pv
from desc.integrals import compute_B_plasma
from desc.plotting import plot_dipoles, plot_3d, plot_comparison
from desc.io import load



eq = load('input.muse-fixedb_output.h5')[-1]


def compute_average_normalized_field(
    field, eq, vacuum=False, chunk_size=None, B_plasma_chunk_size=None
):
    
    # compute Bn as error field on surface
    if B_plasma_chunk_size is None:
        B_plasma_chunk_size = chunk_size
    grid = LinearGrid(M=20, N=20, NFP=eq.NFP)
    if vacuum:
        # we can avoid the expensive plasma contribution calculation if we
        # just pass in the surface instead of the Equilibrium!
        Bn, surf_coords = field.compute_Bnormal(
            eq.surface,
            eval_grid=grid,
            chunk_size=chunk_size,
            B_plasma_chunk_size=B_plasma_chunk_size,
        )
    else:
        Bn, surf_coords = field.compute_Bnormal(
            eq,
            eval_grid=grid,
            chunk_size=chunk_size,
            B_plasma_chunk_size=B_plasma_chunk_size,
        )

    # compute normalization
    normalizing_field_vec = field.compute_magnetic_field(surf_coords)
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

    import pyvista as pv
    pgrid = pv.StructuredGrid(X,Y,Z)
    pgrid.point_data["bn"] = Bn / normalizing_field

    import matplotlib.pyplot as plt
    import pdb
    pdb.set_trace()
    

    surf = pgrid.extract_surface()
    p = pv.Plotter()
    p.add_mesh(
        surf,
        scalars="bn",
        smooth_shading=True,
        show_edges=False,
    )
    p.show()




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
print(len(surface_points["R"]))

R_vals   = surface_points["R"]  
phi_vals = surface_points["phi"] 
Z_vals   = surface_points["Z"]

# cylindrical to xyz
X = R_vals * np.cos(phi_vals)
Y = R_vals * np.sin(phi_vals)
Z = Z_vals

# mag_axis = eq.axis.compute(names=["R", "phi", "Z"], grid=LinearGrid(L=1, M=1, N=5, NFP=eq.NFP, sym=eq.sym), override_grid=False)

# R_axis = mag_axis["R"]
# phi_axis = mag_axis["phi"]
# Z_axis = mag_axis["Z"]

# X_axis = R_axis * np.cos(phi_axis)
# Y_axis = R_axis * np.sin(phi_axis)
# Z_axis = Z_axis

# print(len(X_axis))

XYZ_2 = np.array([X,Y,Z]).reshape((3,11,12)) # 2D array
axis = XYZ_2.mean(axis=2)

Ux,Uy,Uz = axis[:,:,np.newaxis] - XYZ_2 # (3,T,P) array pointing TOWARD axis
Ur = np.sqrt(Ux**2 + Uy**2)
M_phi = np.atan2(Uy, Ux).ravel()
M_theta = np.atan2(Ur, Uz).ravel() # polar angle (theta = 0) points up at x-axis

dipoles = []

for i in range(len(surface_points["R"])):
    # get A to B distance from same phi (toroidal))
    # U_x = X_axis[i] - X[i]
    # U_y = Y_axis[i] - Y[i]
    # U_z = Z_axis[i] - Z[i]
    # U_r = np.sqrt( U_x**2 + U_y**2)
    # # inverse map from xyz to phi/theta
    # phi = np.atan2(U_y, U_x)
    # theta = np.atan2(U_z, U_r)
    d = _Dipole(x=X[i], y=Y[i], z=Z[i], phi=M_phi[i], theta=M_theta[i])
    dipoles.append(d)

one_period = DipoleSet(dipoles, NFP=eq.NFP, sym=eq.sym)

# dipole_set = DipoleSet.from_symmetry(one_period, NFP=eq.NFP, sym=eq.sym)
b1 = compute_average_normalized_field(one_period, eq)
print(b1)

# b = compute_average_normalized_field(dipole_set, eq)
# print(b)


dipole_grid = LinearGrid(N=50)
plot_grid = LinearGrid(M=20, N=40, NFP=1, endpoint=True)



#export_dipoles(dipole_set, "dipoles.csv")

plotter = pv.Plotter()
plotter = plot_dipoles(one_period, plotter=plotter)
plotter.show()

# plot surface, color surface with error field


