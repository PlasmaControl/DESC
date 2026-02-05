from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

from desc.potential_functions import G_w_regularized, stick

def bn_res_vec2(sdata1,
               sgrid, surface,
               N, d_0,
               coords, n_rho,
                ss_data,
               contour_data, contour_grid,
               sign_vals,theta,zeta,
               stick_data,
               index_i,index_f,
              ):

    B_sour0 = B_sour_vec(sdata1,
                         sgrid, surface, 
                         N, d_0, 
                         coords,
                         ss_data,
                         index_i,index_f,
                         )

    #B_cont = B_theta_contours_vec(surface,
    #                              coords,
    #                              contour_data, contour_grid,
    #                              sign_vals,theta,zeta,
    #                              index_i,index_f,
    #                              )

    #B_wire_stick = B_sticks_vec(sgrid,
    #                            surface,
    #                            coords,
    #                            stick_data,
    #                            index_i,index_f,
    #                            )

    B_total = jnp.transpose((B_sour0
                             #+ B_cont 
                             #+ B_wire_stick
                            ), (1, 2, 0)
                           )

    
    return jnp.sum(B_total * n_rho[:, :, None], axis=1)


def bn_res_vec(sdata1,
               sgrid, surface,
               N, d_0,
               coords, ss_data,
               contour_data, contour_grid,
               sign_vals,theta,zeta,
               stick_data,
               index_i,index_f,
              ):

    B_sour0 = B_sour_vec(sdata1,
                         sgrid, surface, 
                         N, d_0, 
                         coords,
                         ss_data,
                         index_i,index_f,
                         )

    #B_cont = B_theta_contours_vec(surface,
    #                              coords,
    #                              contour_data, contour_grid,
    #                              sign_vals,theta,zeta,
    #                              index_i,index_f,
    #                              )

    #B_wire_stick = B_sticks_vec(sgrid,
    #                            surface,
    #                            coords,
    #                            stick_data,
    #                            index_i,index_f,
    #                            )

    B_total = jnp.transpose((B_sour0
                             #+ B_cont 
                             #+ B_wire_stick
                            ), (1, 2, 0)
                           )

    
    return jnp.concatenate((B_total[:,0,:],B_total[:,1,:],B_total[:,2,:]))
    

def B_sour_vec(sdata1,
               sgrid, surface,
                N, d_0,
                coords,
                ss_data,
                index_i,index_f,
              ):

    
    return _compute_magnetic_field_from_Current_vec(sgrid,
                                                    jnp.transpose(K_sour_vec(sdata1,
                                                                             sgrid, surface,
                                                                             N, d_0,
                                                                             ss_data,
                                                                             index_i,index_f,
                                                                             ), (2, 0, 1)),
                                                    surface,
                                                    sdata1,
                                                    coords,
                                                    basis="rpz",
                                                )


def K_sour_vec(
    sdata1,
    sgrid, surface,
    N,
    d_0,
    ss_data,
    index_i,index_f,
):

    # Initialize the omega
    G_w_regularized_ = G_w_regularized(sdata1, 
                                  ss_data["u_iso"][index_i:index_f], 
                                  ss_data["v_iso"][index_i:index_f], 
                                  N, d_0)

    w_shift = sdata1['lambda_ratio'] * 2*jnp.pi/sgrid.NFP
    
    # Need to evluate NFP omegas to take into account superposition of dipoles
    for i in range(1, sgrid.NFP):
    
        G_w_regularized_ += G_w_regularized(sdata1, 
                                  ss_data["u_iso"][index_i:index_f] + w_shift * i, 
                                  ss_data["v_iso"][index_i:index_f], 
                                  N, d_0)

    K_sour_total = ( ( - jnp.imag(G_w_regularized_)[:, None, :] * sdata1['e_v'][:, :, None]
                      + jnp.real(G_w_regularized_)[:, None, :] * sdata1['e_u'][:, :, None] ) * ( 2 * sdata1["lambda_iso"] ** (-2) )[:,None,None]
                    )


    jnp.save('K_indexi_' + str(index_i) + '_indexi_' + str(index_f) + '.npy', K_sour_total)
    
    return K_sour_total


MU0_OVER_4PI = 1e-7  # μ0/(4π)

def B_theta_contours_vec(surface,
                         coords,
                         contour_data, contour_grid,
                         sign_vals, theta, zeta,
                         index_i, index_f):
    """
    Computes B from the poloidal wire 'current contours' as TRUE 1D filament integrals (trapz in theta),
    using the SAME function signature you already use.

    Inputs (same as before):
      - contour_data, contour_grid: built on the refined theta_contour x zeta_contour grid
          contour_data must contain: "x" (rpz), "e_theta" (rpz), and optionally "theta","zeta"
      - theta, zeta: your coarse endpoint arrays defining one contour per (zeta_idx, theta_idx) column
      - sign_vals: multiplicative direction factor for each contour (NOT per node)
      - index_i,index_f: chunk of contours to evaluate
    """

    # --- evaluation points ---
    # coords is provided in rpz (per your note)
    re_xyz = rpz2xyz(jnp.atleast_2d(coords))
    Ne = re_xyz.shape[0]

    # --- infer refined theta_line and zeta_line from contour_grid (no new external variables) ---
    nodes = contour_grid.nodes  # (K*M, 3): [rho, theta, zeta]

    # count how many points per zeta "row" (M) by counting nodes with same zeta as first node
    # (works because your zeta_contour values are exact grid values)
    z0 = nodes[0, 2]
    M = jnp.sum(nodes[:, 2] == z0).astype(int)

    # extract theta_line (refined) and zeta_line in the same ordering used by alt_grid_sticks
    theta_line = nodes[:M, 1]          # (M,)
    zeta_line = nodes[::M, 2]          # (K,)
    K = zeta_line.shape[0]

    # reshape refined geometry + tangent data to (K, M, 3)
    x_rpz_full = contour_data["x"].reshape((K, M, 3))
    e_theta_rpz_full = contour_data["e_theta"].reshape((K, M, 3))

    # --- build the mask that selects the portion of each contour between theta=pi and theta_target ---
    # compute_mask returns (n_nodes, p) where p = (#contours in chunk)
    # We'll reuse it to avoid "new bullshit variables".
    A = compute_mask(contour_data, theta, zeta, index_i, index_f)  # (K*M, P)
    P = A.shape[1]

    # slice per-contour signs (must be length m*k, same convention as compute_mask columns)
    # sign_vals is assumed to be defined on the SAME global_col indexing as compute_mask:
    # global_col = zeta_idx * m + theta_idx
    sign_chunk = sign_vals[index_i:index_f]  # (P,)

    # --- build trapezoid weights in theta ---
    dtheta = (theta_line[-1] - theta_line[0]) / (M - 1)
    wtheta = jnp.ones((M,), dtype=theta_line.dtype).at[0].set(0.5).at[-1].set(0.5) * dtheta  # (M,)

    # --- convert refined source points/tangents to xyz ---
    # (do only once for all contours, then slice in chunk)
    x_rpz = x_rpz_full  # (K,M,3)
    rs_xyz = rpz2xyz(x_rpz.reshape((-1, 3))).reshape((K, M, 3))  # (K,M,3)

    phi = x_rpz[:, :, 1]  # (K,M)
    e_theta_xyz = rpz2xyz_vec(e_theta_rpz_full.reshape((-1, 3)), phi=phi.reshape((-1,))).reshape((K, M, 3))

    e_theta_norm = jnp.linalg.norm(e_theta_xyz, axis=-1)  # (K,M)
    t_hat = e_theta_xyz / (e_theta_norm[:, :, None] + 1e-30)  # (K,M,3)

    # --- Now do the filament Biot–Savart sum for each contour column in the chunk ---
    # Each contour column picks a specific zeta_idx (row-block) and a theta-interval within that row.
    # We'll compute per-node contributions for ALL refined nodes, then the A mask will select the nodes for each contour.

    # reshape A to (K, M, P) to apply it over refined theta nodes within each zeta row
    A_kmp = A.reshape((K, M, P)).astype(theta_line.dtype)  # (K,M,P)

    # line element magnitude per node: dℓ = |e_theta| * (trapz weight)
    dl_km = e_theta_norm * wtheta[None, :]  # (K,M)

    # direction per contour column (P): apply as scalar multiplier on tangent
    # sign affects direction; A selects which nodes are on that contour
    # Idl vector at each (k,m,p):
    #   Idl = (sign[p] * t_hat[k,m]) * (dl[k,m] * A[k,m,p])
    Idl_kmp3 = (
        t_hat[:, :, None, :] *
        (dl_km[:, :, None] * A_kmp)[:, :, :, None] *
        sign_chunk[None, None, :, None]
    )  # (K,M,P,3)

    # vector from source node to eval point: r = re - rs
    r_kmNe3 = re_xyz[None, None, :, :] - rs_xyz[:, :, None, :]  # (K,M,Ne,3)
    den_kmNe = jnp.linalg.norm(r_kmNe3, axis=-1) ** 3 + 1e-30    # (K,M,Ne)

    # cross(Idl, r): need broadcast over Ne
    num_kmPNe3 = jnp.cross(Idl_kmp3[:, :, :, None, :], r_kmNe3[:, :, None, :, :], axis=-1)  # (K,M,P,Ne,3)

    # sum over refined nodes (k,m) to get B for each contour p at each eval point
    B_xyz_pNe3 = MU0_OVER_4PI * jnp.sum(num_kmPNe3 / den_kmNe[:, :, None, :, None], axis=(0, 1))  # (P,Ne,3)

    # return in rpz because your downstream expects rpz
    B_rpz_pNe3 = xyz2rpz_vec(B_xyz_pNe3, x=re_xyz[:, 0], y=re_xyz[:, 1])
    return B_rpz_pNe3
    
def B_theta_contours_vec2(surface,
                         coords,
                         ss_data, ss_grid,
                         sign_vals,theta,zeta,
                         index_i,index_f,
                        ):

    A = compute_mask(ss_data, theta, zeta, index_i, index_f)
    col_norms = jnp.linalg.norm(A, axis=0)
    print("min/max col norm:", col_norms.min(), col_norms.max())
    print("num ~zero cols:", jnp.sum(col_norms < 1e-12))

    AA = A[:, None, :] * ss_data['e_theta'][:, :, None]
    AAA = AA * ( jnp.sum(ss_data["e_theta"] * ss_data["e_theta"], axis = 1 ) ** ( -1 / 2 ) * sign_vals )[:, None, None]

    
    return _compute_magnetic_field_from_Current_Contour_vec(ss_grid,
                                                            jnp.transpose(AAA, (2, 0, 1)),
                                                            surface,
                                                            ss_data,
                                                            coords,
                                                            basis="rpz",
                                                            )

def B_sticks_vec(sgrid,
                surface,
                coords,
                ss_data,
                 index_i,index_f,
            ):

    pls_points = rpz2xyz(coords)  # eq_surf.compute(["x"], grid=Bgrid, basis="xyz")["x"]

    b_stick_fun = stick(ss_data["x"][index_i:index_f],  # Location of the wire at the theta = pi cut, variable zeta position
                        0 * ss_data["x"][index_i:index_f],  # All wires at the center go to the origin
                        pls_points,
                        sgrid,
                        basis="rpz",
                        )

    
    return b_stick_fun


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


def _compute_magnetic_field_from_Current_Contour_vec(
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
    # ---- 1D arc-length weights along theta (trapz), no dζ ----
    nodes = surface_grid.nodes  # (K*M, 3): [rho, theta, zeta]
    
    # infer M = number of theta points per zeta row (theta varies fastest)
    z0 = nodes[0, 2]
    M = jnp.sum(nodes[:, 2] == z0).astype(int)
    
    theta_line = nodes[:M, 1]  # refined theta samples (length M)
    
    # trapezoid weights in theta
    dtheta = (theta_line[-1] - theta_line[0]) / (M - 1)
    wtheta = jnp.ones((M,), dtype=theta_line.dtype)
    wtheta = wtheta.at[0].set(0.5)
    wtheta = wtheta.at[-1].set(0.5)
    wtheta = wtheta * dtheta  # (M,)
    
    # repeat for each zeta row to match flattened node ordering
    wtheta_flat = jnp.tile(wtheta, reps=(surface_grid.nodes.shape[0] // M,))  # (K*M,)
    
    # arc-length element dℓ = |e_theta| dθ
    # NOTE: use dot row-wise, not global dot
    e_theta_mag = jnp.linalg.norm(data["e_theta"], axis=1)  # (K*M,)
    _dV = (wtheta_flat * e_theta_mag) / surface_grid.NFP

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

    #import pdb
    B = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros((N, 3, coords.shape[0])))

    
    #pdb.set_trace()
    
    if basis == "rpz":
        B = xyz2rpz_vec(jnp.transpose(B, (0, 2, 1)), x=coords[:, 0], y=coords[:, 1])
    
    #pdb.set_trace()
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


def compute_mask(contour_data, theta_coarse, zeta_coarse, start_idx, end_idx):
    """
    Compute a subset of the binary mask matrix for a range of global column indices [start_idx, end_idx).
    Assumes theta_coarse and zeta_coarse are 1D arrays of shape (m,) and (k,) respectively.
    For each column:
        - If theta_coarse[theta_idx] < pi, mask sets 1s where contour_data["theta"] is >= theta_val and < pi.
        - If theta_coarse[theta_idx] >= pi, mask sets 1s where contour_data["theta"] is >= pi and <= theta_val.
    Also requires contour_data["zeta"] equals zeta_coarse[zeta_idx]. The full mask would be of shape (n, m*k),
    with columns ordered such that theta varies fastest within each zeta block
    (global_col = zeta_idx * m + theta_idx).
    
    Parameters:
        contour_data (dict): Contains 1D arrays "theta" (shape (n,)) and "zeta" (shape (n,)).
        theta_coarse (ndarray): 1D array of shape (m,) containing theta position values in [0, 2pi].
        zeta_coarse (ndarray): 1D array of shape (k,) containing zeta position values.
        start_idx (int): Starting global column index (inclusive).
        end_idx (int): Ending global column index (exclusive).
    
    Returns:
        ndarray: Binary mask of shape (n, p), where p = end_idx - start_idx.
    """
    m = theta_coarse.shape[0]  # Number of theta positions
    k = zeta_coarse.shape[0]   # Number of zeta positions
    n = contour_data["theta"].shape[0]
    p = end_idx - start_idx
    mask = jnp.zeros((n, p), dtype=jnp.int32)

    def body(local_col, mask):
        global_col = start_idx + local_col
        zeta_idx = global_col // m
        theta_idx = global_col % m
        theta_val = theta_coarse[theta_idx]
        zeta_val = zeta_coarse[zeta_idx]
        # Set theta bounds based on theta_val
        theta_lower = jnp.where(theta_val < jnp.pi, theta_val, jnp.pi)
        theta_upper = jnp.where(theta_val < jnp.pi, jnp.pi, theta_val)
        cond = (contour_data["theta"] >= theta_lower) & (contour_data["theta"] <= theta_upper) & (contour_data["zeta"] == zeta_val)
        return mask.at[:, local_col].set(jnp.where(cond, 1, 0))

    mask = jax.lax.fori_loop(0, p, body, mask)

    
    return mask