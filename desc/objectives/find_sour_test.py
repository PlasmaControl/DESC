from interpax import interp2d

from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
#from desc.find_dips import biot_savart_general

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

from .sources_dipoles_utils import (_compute_magnetic_field_from_Current, 
                                    _compute_magnetic_field_from_Current_Contour,
                                    omega_sour,
                                    compute_mask,
                                    )

#@jax.jit
def bn_res(
    #p_M,
    #p_N,
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    y,
    N,
    d_0,
    coords,
    tdata,
    contour_data,
    stick_data,
    contour_grid,
    ss_data,
    #theta_coarse,
    #zeta_coarse,
    AAA,
):

    #import pdb
    #pdb.set_trace()
    B_sour0 = B_sour(#p_M, p_N, 
                     sdata1, sdata2, sdata3, sgrid, surface, y, N, d_0, coords, tdata, ss_data,
    )

    #pdb.set_trace()
    B_wire_cont = B_theta_contours(surface, y, coords, contour_data, contour_grid,
                                   AAA,
                                  )
    
    #pdb.set_trace()
    B_sticks0 = B_sticks(
    #    p_M, p_N,
        sgrid,
        surface,
        y,
        coords,
        stick_data,
    )

    #pdb.set_trace()
    B_total = (B_sour0 + B_wire_cont + B_sticks0)

    return B_total

#@jax.jit
def B_sour(
    #p_M,
    #p_N,
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    y,
    N,
    d_0,
    coords,
    tdata,
    ss_data,
):

    return _compute_magnetic_field_from_Current(
        sgrid,
        K_sour(sdata1, sdata2, sdata3,
               sgrid, surface,
               y,
               N, d_0,
               tdata,
               ss_data,
               ),
        surface,
        sdata1,
        coords,
        #basis="rpz",
    )

#@jax.jit#(static_argnums=(0,1))
def B_theta_contours(
    #p_M,
    #p_N,
    #sdata,
    #sgrid,
    surface,
    y,
    coords,
    ss_data,
    ss_grid,
    #theta_coarse,
    #zeta_coarse,
    AAA,
):

    #r_t = p_M * 2 + 1  # theta_coarse.shape[0]
    #r_z = p_N * 2 + 1  # zeta_coarse.shape[0]

    #theta_coarse = jnp.linspace(2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1 / 2,
    #                            2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1 / 2),
    #                            p_M * 2 + 1,
    #                            )

    #zeta_coarse = jnp.linspace(2 * jnp.pi / surface.NFP * (1 / (p_N * 2 + 1)) * 1 / 2,
    #                           2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2 + 1) * 1 / 2),
    #                           p_N * 2 + 1,
    #                           )

    #sign_vals = jnp.where(ss_data["theta"] < jnp.pi, -1, 1) #+ jnp.where(ss_data["theta"] > jnp.pi, 1, 0)

    # Generate the matrix of coefficients for the contours
    #A = compute_mask(ss_data, theta_coarse, zeta_coarse)
    #AA = A[:, None, :] * ss_data['e_theta'][:, :, None]
    #AAA = AA * ( dot(ss_data["e_theta"], ss_data["e_theta"] ) ** (-1 / 2) * sign_vals )[:, None, None]
    
    # Add the contributions from all the wires extending out of the winding surface
    K_cont = jnp.sum(y[None, None,:] * AAA,axis = 2)
    
    return _compute_magnetic_field_from_Current_Contour(
        ss_grid, K_cont, surface, ss_data, coords, #basis="rpz"
    )

#@jax.jit#(static_argnums=(0,1))
def B_sticks(
    #p_M,
    #p_N,
    sgrid,
    surface,
    y,
    coords,
    ss_data,
):

    pls_points = rpz2xyz(coords)  # eq_surf.compute(["x"], grid=Bgrid, basis="xyz")["x"]

    r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    b_stick_fun = y[:, None, None] * stick(ss_data["x"],  # Location of the wire at the theta = pi cut, variable zeta position
                                            0 * ss_data["x"],  # All wires at the center go to the origin
                                            pls_points,
                                            sgrid,
                                            #basis="rpz",
                                            )

    sticks_total = jnp.sum(b_stick_fun, axis=0) # (M, 3)

    return sticks_total

# TODO: This function has a problem when jitted    
#@jax.jit
def stick(
    p2_,  # second point of the stick
    p1_,  # first point of the stick
    plasma_points,  # points on the plasma surface
    surface_grid,  # Kgrid,
    #basis="rpz",
):
    """Computes the magnetic field on the plasma surface due to a unit current on the source wires.
    
        p2_: numpy.ndarray of dimension (N, 3)
        p1_: numpy.ndarray of dimension (N, 3)
        plasma_point: numpy.ndarray of dimension (M, 3)
    
    """
    
    basis="rpz"
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi2 = (p2_[:, 2] + j * 2 * jnp.pi / surface_grid.NFP) % (2 * jnp.pi)
        # phi1 = ( p1_[2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        # (surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )

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

        #import pdb
        #pdb.set_trace()
        
        return f

    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros((p1_.shape[0], plasma_points.shape[0], plasma_points.shape[1])))

    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points[:, 0], y=plasma_points[:, 1])

    return b_stick


#@jax.jit#(static_argnums=(0,1))
def K_sour(
    #p_M,
    #p_N,
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    y,
    N,
    d_0,
    tdata,
    ss_data,
):
    
    #r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    # Assume periodicity on the values of the sources so
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
    
    K_sour_total = (sdata1["lambda_iso"] ** (-1) * jnp.sum( ( ( - jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_v'][:, :, None]
                                                              + jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_u'][:, :, None] ) * y[None,:])
                                                           , axis = 2
                                                           ).T
                   ).T

    return K_sour_total