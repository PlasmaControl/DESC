from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot


from desc.potential_functions import G_w_pair, stick

def bn_res_vec(sdata1, sdata2, sdata3,
                sgrid, surface,
                N, d_0,
                coords,
                du_data, dd_data, dr_data, dl_data,
                index_i, index_f,
               ):

    _b_pol_dips, _b_tor_dips = B_dips_vec(sdata1, sdata2, sdata3, 
                                         sgrid, surface, 
                                         N, d_0, 
                                         coords,
                                         du_data, dd_data, dr_data, dl_data,
                                         index_i,index_f,
                                         )

    _b_pol_sticks, _b_tor_sticks = B_sticks_vec(sdata1,
                                                 sgrid,
                                                 N, d_0, 
                                                 coords,
                                                 du_data, dd_data, dr_data, dl_data,
                                                 index_i,index_f,
                                                 )

    
    B_pol_ = jnp.transpose((_b_pol_dips - _b_pol_sticks), (1, 2, 0))
    B_tor_ = jnp.transpose((_b_tor_dips - _b_tor_sticks), (1, 2, 0))

    return (jnp.concatenate((B_pol_[:,0,:],B_pol_[:,1,:],B_pol_[:,2,:])), 
            jnp.concatenate((B_tor_[:,0,:],B_tor_[:,1,:],B_tor_[:,2,:]))
           )

    
def B_sticks_vec(sdata, sgrid,
                 dt,dz,
                 coords,
                 dl_data, dr_data, dd_data, du_data,
                 index_i,index_f,
                 ):

    pls_points = rpz2xyz(coords)

    b_stick_pol = stick(du_data["x"][index_i:index_f], # u-pos of second dipole
                            dd_data["x"][index_i:index_f], # u-pos of first dipole
                            pls_points,
                            sgrid,
                            basis = "rpz",
                            )
    
    b_stick_tor = stick(dr_data["x"][index_i:index_f], # u-pos of second dipole
                            dl_data["x"][index_i:index_f], # u-pos of first dipole
                            pls_points,
                            sgrid,
                            basis = "rpz",
                            )
    
    return b_stick_pol, b_stick_tor
    

def B_dips_vec(
    sdata1, sdata2, sdata3,
    sgrid, surface,
    N, d_0,
    coords,
    du_data, dd_data, dr_data, dl_data,
    index_i,index_f,
):

    k_pol, k_tor = K_dips_vec(sdata1, sdata2, sdata3,
                              sgrid, surface,
                              du_data, dd_data, dr_data, dl_data,
                              N, d_0,
                              index_i,index_f,
                              )

    b_pol = _compute_magnetic_field_from_Current_vec(sgrid,
                                                    jnp.transpose(k_pol, (2, 0, 1)),
                                                    surface,
                                                    sdata1,
                                                    coords,
                                                    basis="rpz",
                                                    )
    b_tor = _compute_magnetic_field_from_Current_vec(sgrid,
                                                    jnp.transpose(k_tor, (2, 0, 1)),
                                                    surface, sdata1,
                                                    coords,
                                                    basis="rpz",
                                                    )
    
    return b_pol, b_tor


def K_dips_vec(sdata1, sdata2, sdata3,
               sgrid, surface,
               du_data, dd_data, dr_data, dl_data,
               N, d_0,
               index_i,index_f,):
    
    # Need to evlauate three omegas
    G_w_pol_total = ( G_w_pair(sdata1,
                               du_data["u_iso"][index_i:index_f], du_data["v_iso"][index_i:index_f],
                               dd_data["u_iso"][index_i:index_f], dd_data["v_iso"][index_i:index_f],
                               N, d_0)
                     + G_w_pair(sdata2,
                                du_data["u_iso"][index_i:index_f], du_data["v_iso"][index_i:index_f],
                                dd_data["u_iso"][index_i:index_f], dd_data["v_iso"][index_i:index_f],
                                N, d_0)
                     + G_w_pair(sdata3,
                                du_data["u_iso"][index_i:index_f], du_data["v_iso"][index_i:index_f],
                                dd_data["u_iso"][index_i:index_f], dd_data["v_iso"][index_i:index_f],
                                N, d_0)
                       )

    G_w_tor_total = ( G_w_pair(sdata1,
                               dr_data["u_iso"][index_i:index_f], dr_data["v_iso"][index_i:index_f],
                               dl_data["u_iso"][index_i:index_f], dl_data["v_iso"][index_i:index_f],
                               N, d_0)
                     + G_w_pair(sdata2,
                                dr_data["u_iso"][index_i:index_f], dr_data["v_iso"][index_i:index_f],
                                dl_data["u_iso"][index_i:index_f], dl_data["v_iso"][index_i:index_f],
                                N, d_0)
                     + G_w_pair(sdata3,
                                dr_data["u_iso"][index_i:index_f], dr_data["v_iso"][index_i:index_f],
                                dl_data["u_iso"][index_i:index_f], dl_data["v_iso"][index_i:index_f],
                                N, d_0)
                     )
    

    K_pol_total = ( ( - jnp.imag(G_w_pol_total)[:, None, :] * sdata1['e_v'][:, :, None]
                      + jnp.real(G_w_pol_total)[:, None, :] * sdata1['e_u'][:, :, None] ) * ( 2 * sdata1["lambda_iso"] ** (-2) )[:,None,None]
                   )

    K_tor_total = ( ( - jnp.imag(G_w_tor_total)[:, None, :] * sdata1['e_v'][:, :, None]
                      + jnp.real(G_w_tor_total)[:, None, :] * sdata1['e_u'][:, :, None] ) * ( 2 * sdata1["lambda_iso"] ** (-2) )[:,None,None]
                   )

    return K_pol_total, K_tor_total

def _compute_magnetic_field_from_Current_vec(
    Kgrid, K_at_grid, surface, data, coords,
    basis="rpz"
):
    """Compute magnetic field at a set of points.

    Parameters
    ----------
    K_at_grid : ndarray, shape (M, num_nodes,3) or (num_nodes,3)
        Surface current evaluated at points on a grid, which you want to calculate
        B from, should be in cartesian ("xyz") or cylindrical ("rpz") specifiec
        by "basis" argument
    surface : FourierRZToroidalSurface
        surface object upon which the surface current K_at_grid lies
    coords : array-like shape(N,3) or Grid
        cylindrical or cartesian coordinates to evlauate B at
    grid : Grid,
        source grid upon which to evaluate the surface current density K
    basis : {"rpz", "xyz"}
        basis for input coordinates and returned magnetic field

    Returns
    -------
    field : ndarray, shape(M, N,3) or (N,3)
        magnetic field at specified points

    """
    
    assert basis.lower() in ["rpz", "xyz"]
    if hasattr(coords, "nodes"):
        coords = coords.nodes
    coords = jnp.atleast_2d(coords)
    
    # Handle if K_at_grid is 2D (num_nodes,3) by adding a leading dimension
    if K_at_grid.ndim == 2:
        K_at_grid = K_at_grid[None, :, :]
        vectorized = False
    else:
        vectorized = True
    
    N = K_at_grid.shape[0]
    
    grid_rpz = data["x"]  # Assuming data["x"] is rpz coordinates of the grid (num_nodes, 3)
    grid_xyz = rpz2xyz(grid_rpz)
    
    if basis == "rpz":
        coords = rpz2xyz(coords)
    else:
        K_at_grid = xyz2rpz_vec(K_at_grid, x=grid_xyz[:, 0], y=grid_xyz[:, 1])

    surface_grid = Kgrid

    _rs = grid_rpz
    _K = K_at_grid

    # Surface element
    _dV = (
        surface_grid.weights
        * data["|e_theta x e_zeta|"]
        / surface_grid.NFP
    )

    def nfp_loop(j, f):
        phi = (surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (
            2 * jnp.pi
        )
        rs = jnp.vstack((_rs[:, 0], phi, _rs[:, 2])).T
        rs = rpz2xyz(rs)
        K = rpz2xyz_vec(_K, phi=phi)
        fj = biot_savart_general_vec(
            coords,
            rs,
            K,
            _dV,
        )
        f += fj
        return f
        
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros((N, 3, coords.shape[0])))
    
    if basis == "rpz":
        B = xyz2rpz_vec(jnp.transpose(B, (0, 2, 1)), x=coords[:, 0], y=coords[:, 1])
    
    if not vectorized:
        B = B[0]

    return B

def biot_savart_general_vec(re, rs, J, dV):
    """Compute Biot-Savart integral for magnetic field.

    Parameters
    ----------
    re : ndarray, shape(N_coords,3)
        Evaluation points
    rs : ndarray, shape(num_nodes,3)
        Source points
    J : ndarray, shape(N,num_nodes,3) or (num_nodes,3)
        Current density at source points
    dV : ndarray, shape(num_nodes,)
        Differential volume/area at source points

    Returns
    -------
    B : ndarray, shape(N,3,N_coords) or (3,N_coords)
        Magnetic field at evaluation points
    """
    
    if J.ndim == 2:
        J = J[None, :, :]
        vectorized = False
    else:
        vectorized = True
    
    N = J.shape[0]
    
    re, rs, J, dV = map(lambda x: jnp.asarray(x, dtype=jnp.float64), (re, rs, J, dV))
    JdV = J * dV[:, None]  # JdV becomes (N, num_nodes, 3)
    B = jnp.zeros((N, 3, re.shape[0]), dtype=jnp.float64)
    
    def body(i, B):
        r = re - rs[i, :]  # (N_coords, 3)
        JdV_i = JdV[:, i, :][:, None, :]  # (N, 1, 3)
        num = jnp.cross(JdV_i, r, axis=-1)  # (N, N_coords, 3)
        num = jnp.transpose(num, (0, 2, 1))  # (N, 3, N_coords)
        den = jnp.linalg.norm(r, axis=-1) ** 3  # (N_coords,)
        contrib = jnp.where(den[None, None, :] == 0, 0, num / den[None, None, :])
        B = B + contrib
        return B

    result = 1e-7 * fori_loop(0, rs.shape[0], body, B)
    
    if not vectorized:
        result = result[0]
    
    return result