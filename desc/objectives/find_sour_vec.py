from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

from .sources_dipoles_utils import (_compute_magnetic_field_from_Current, 
                                    _compute_magnetic_field_from_Current_Contour,
                                    omega_sour,
                                    compute_mask,
                                    )

def bn_res_vec(
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    N,
    d_0,
    coords,
    tdata,
    contour_data,
    stick_data,
    contour_grid,
    ss_data,
    AAA,
):

    #import pdb
    #pdb.set_trace()
    B_sour0 = B_sour_vec(sdata1, sdata2, sdata3, sgrid, surface, N, d_0, coords, tdata, ss_data,
    )

    #pdb.set_trace()
    B_wire_cont = B_theta_contours_vec(surface, coords, contour_data, contour_grid,
                                   AAA,
                                  )
    
    #pdb.set_trace()
    B_sticks0 = B_sticks_vec(
        sgrid,
        surface,
        coords,
        stick_data,
    )

    #pdb.set_trace()
    B_total = jnp.transpose((B_sour0 + B_wire_cont + B_sticks0), (1, 2, 0))

    return jnp.concatenate((B_total[:,0,:],B_total[:,1,:],B_total[:,2,:]))

def B_theta_contours_vec(surface,
                    coords,
                    ss_data,
                    ss_grid,
                    AAA,
                ):

    test = _compute_magnetic_field_from_Current_Contour_vec(ss_grid, 
                                                        jnp.transpose(AAA, (2, 0, 1)), #AAA, 
                                                        surface, ss_data, coords, basis="rpz"
                                                        )
    return test

def B_sticks_vec(sgrid,
    surface,
    #y,
    coords,
    ss_data,
):

    pls_points = rpz2xyz(coords)  # eq_surf.compute(["x"], grid=Bgrid, basis="xyz")["x"]

    r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    b_stick_fun = stick(ss_data["x"],  # Location of the wire at the theta = pi cut, variable zeta position
                                            0 * ss_data["x"],  # All wires at the center go to the origin
                                            pls_points,
                                            sgrid,
                                            basis="rpz",
                                            )

    return b_stick_fun

def stick(
    p2_,  # second point of the stick
    p1_,  # first point of the stick
    plasma_points,  # points on the plasma surface
    surface_grid,  # Kgrid,
    basis="rpz",
):
    """Computes the magnetic field on the plasma surface due to a unit current on the source wires.
    
        p2_: numpy.ndarray of dimension (N, 3)
        p1_: numpy.ndarray of dimension (N, 3)
        plasma_point: numpy.ndarray of dimension (M, 3)
    
    """
    
    #basis="rpz"
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi2 = (p2_[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)

        # TODO: Make sure p2s has the shape (N, 3)
        p2s = jnp.stack([p2_[:, 0], phi2, p2_[:, 2]], axis=1)

        #print(p2s.shape)
        p2s = rpz2xyz(p2s)

        # a_s.shape = b_s.shape = c_s.shape = (N, M, 3)
        a_s = p2s[:, None, :] - p1_[:, None, :]
        b_s = p1_[:, None, :] - plasma_points[None, :, :]
        c_s = p2s[:, None, :] - plasma_points[None, :, :]

        # if c_s and a_s are (N, 3), will work fine
        c_sxa_s = cross(c_s, a_s)

        f += (
            1e-7
            * ( #(
                (
                    jnp.clip(jnp.sum(c_sxa_s * c_sxa_s, axis=2), a_min=1e-8, a_max=None)
                    * jnp.sum(c_s * c_s, axis=2) ** (1 / 2)
                )
                ** (-1)
                * (jnp.sum(a_s * c_s, axis=2) - jnp.sum(a_s * b_s, axis=2)) )[:, :, None]
                * c_sxa_s#.T
            #).T
        ) # (N, M, 3)
        
        return f

    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros((p1_.shape[0], plasma_points.shape[0], plasma_points.shape[1])))

    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points[:, 0], y=plasma_points[:, 1])

    return b_stick
    
#@jax.jit
def B_sour_vec(
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    #y,
    N,
    d_0,
    coords,
    tdata,
    ss_data,
):

    return _compute_magnetic_field_from_Current_vec(
        sgrid,
        jnp.transpose(K_sour_vec(sdata1, sdata2, sdata3,
               sgrid, surface,
               #y,
               N, d_0,
               tdata,
               ss_data,
               ), (2, 0, 1)),
        surface,
        sdata1,
        coords,
        basis="rpz",
    )

def K_sour_vec(
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    #y,
    N,
    d_0,
    tdata,
    ss_data,
):
    
    omega_sour_fun = ( omega_sour(sdata1, 
                                ss_data["u_iso"], 
                                ss_data["v_iso"], 
                                N, 
                                d_0)
                      + omega_sour(sdata2, 
                                ss_data["u_iso"], 
                                ss_data["v_iso"], 
                                N, 
                                d_0) 
                      + omega_sour(sdata3, 
                                ss_data["u_iso"], 
                                ss_data["v_iso"], 
                                N, 
                                d_0)
                     )

    K_sour_total = ( ( - jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_v'][:, :, None]
                      + jnp.real(omega_sour_fun)[:, None, :] * sdata1['e_u'][:, :, None] ) * ( sdata1["lambda_iso"] ** (-1) )[:,None,None]
                   )

    return K_sour_total

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
    _dV = (
        surface_grid.weights
        * dot(data["e_theta"], data["e_theta"]) ** (1 / 2)
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