from interpax import interp2d

from desc.backend import fori_loop, jax, jit, jnp, scan
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
#from desc.find_dips import biot_savart_general

# from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

from .sources_dipoles_utils import (_compute_magnetic_field_from_Current, 
                                    _compute_magnetic_field_from_Current_Contour, 
                                    v1_eval, chi_reg, v1_prime_eval, comp_loc, 
                                    comp_loc, add_extra, add_extra_periodic, add_extra_coords, alt_grid, alt_grid_sticks, 
                                    #iso_coords_interp,
                                    #f_sour, f_reg,
                                    omega_sour, v1_eval, v1_prime_eval, chi_reg, 
                                    compute_mask,
                                    )


def bn_res(
    p_M,
    p_N,
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
):

    B_sour0 = B_sour(
        p_M, p_N, sdata1, sdata2, sdata3, sgrid, surface, y, N, d_0, coords, tdata, ss_data,
    )

    #B_wire_cont = B_theta_contours(
    #    p_M, p_N, sdata1, sgrid, surface, y, coords, contour_data, contour_grid
    #)

    #B_sticks0 = B_sticks(
    #    p_M,
    #    p_N,
    #    sgrid,
    #    surface,
    #    y,
    #    coords,
    #    stick_data,
    #)

    B_total = B_sour0 #+  B_wire_cont + B_sticks0 # + 

    return B_total


def B_sour(
    p_M,
    p_N,
    sdata1,
    sdata2,
    sdata3,
    sgrid,
    surface,
    y,
    N,
    d_0,
    coords,  # eq,Bgrid,
    tdata,
    ss_data,
):

    return _compute_magnetic_field_from_Current(
        sgrid,
        K_sour(
            p_M,
            p_N,
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
        ),
        surface,
        sdata1,
        coords,
        basis="rpz",
    )

# @jax.jit
def B_theta_contours_old(
    p_M,
    p_N,
    sdata,
    sgrid,
    surface,
    y,
    coords,
    ss_data,
    ss_grid,
):

    r_t = p_M * 2 + 1  # theta_coarse.shape[0]
    r_z = p_N * 2 + 1  # zeta_coarse.shape[0]

    theta_coarse = jnp.linspace(
        2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1 / 2,
        2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1 / 2),
        p_M * 2 + 1,
    )

    zeta_coarse = jnp.linspace(
        2 * jnp.pi / surface.NFP * (1 / (p_N * 2 + 1)) * 1 / 2,
        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2 + 1) * 1 / 2),
        p_N * 2 + 1,
    )

    sign_vals = jnp.where(ss_data["theta"] < jnp.pi, -1, 0) + jnp.where(
        ss_data["theta"] > jnp.pi, 1, 0
    )

    def outer_body(i, K_cont):
        def inner_body(j, K_cont_inner):
            k_fix = jnp.where(
                (ss_data["zeta"] == zeta_coarse[i])
                & (ss_data["theta"] > theta_coarse[j]),
                1,
                0,
            )
            return (
                K_cont_inner
                + (
                    y[i * r_t + j]
                    * sign_vals
                    * k_fix
                    * dot(ss_data["e_theta"], ss_data["e_theta"]) ** (-1 / 2)
                    * ss_data["e_theta"].T
                ).T
            )

        return fori_loop(0, r_t, inner_body, K_cont)

    K_cont = fori_loop(0, r_z, outer_body, jnp.zeros_like(ss_data["e_theta"]))

    return _compute_magnetic_field_from_Current_Contour(
        ss_grid, K_cont, surface, ss_data, coords, basis="rpz"
    )

def B_theta_contours(
    p_M,
    p_N,
    sdata,
    sgrid,
    surface,
    y,
    coords,
    ss_data,
    ss_grid,
):

    r_t = p_M * 2 + 1  # theta_coarse.shape[0]
    r_z = p_N * 2 + 1  # zeta_coarse.shape[0]

    theta_coarse = jnp.linspace(
        2 * jnp.pi * (1 / (p_M * 2 + 1)) * 1 / 2,
        2 * jnp.pi * (1 - 1 / (p_M * 2 + 1) * 1 / 2),
        p_M * 2 + 1,
    )

    zeta_coarse = jnp.linspace(
        2 * jnp.pi / surface.NFP * (1 / (p_N * 2 + 1)) * 1 / 2,
        2 * jnp.pi / surface.NFP * (1 - 1 / (p_N * 2 + 1) * 1 / 2),
        p_N * 2 + 1,
    )

    sign_vals = jnp.where(ss_data["theta"] < jnp.pi, -1, 0) + jnp.where(
        ss_data["theta"] > jnp.pi, 1, 0
    )

    # Generate the matrix of coefficients for the contours
    A = compute_mask(ss_data, theta_coarse, zeta_coarse)
    AA = A[:, None, :] * ss_data['e_theta'][:, :, None]
    AAA = AA * ( dot(ss_data["e_theta"], ss_data["e_theta"] ) ** (-1 / 2) * sign_vals )[:, None, None]
    # TODO: AAA matrix is constant during iterations so might be better to construic it in build and pass it to compute. Storing it in memory migh not be that expensive (?)

    # Add the contributions from all the wires extending out of the winding surface
    K_cont = jnp.sum(y[None, None,:]*AAA,axis = 2)
    
    return _compute_magnetic_field_from_Current_Contour(
        ss_grid, K_cont, surface, ss_data, coords, basis="rpz"
    )


def B_sticks(
    p_M,
    p_N,
    sgrid,
    surface,
    y,
    coords,
    ss_data,
):

    pls_points = rpz2xyz(coords)  # eq_surf.compute(["x"], grid=Bgrid, basis="xyz")["x"]

    r = ss_data["theta"].shape[0]  # Make r a Python int for indexing

    b_stick_fun = y[:, None, None] * stick(
        ss_data["x"],  # Location of the wire at the theta = pi cut, variable zeta position
        0 * ss_data["x"],  # All wires at the center go to the origin
        pls_points,
        sgrid,
        basis="rpz",
        )

    sticks_total = jnp.sum(b_stick_fun, axis=0) # (M, 3)

    return sticks_total


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

def K_sour_old(
    p_M,
    p_N,
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
    
    r = ss_data["theta"].shape[0]  # Make r a Python int for indexing
    # @jax.jit
    def body_fun1(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = jax.lax.dynamic_index_in_dim(y, i, axis=0)

        omega_s1 = omega_sour(sdata1, ss_data["u_iso"][i], ss_data["v_iso"][i], N, d_0)

        omega_total_real += y_ * jnp.real(omega_s1)
        omega_total_imag += y_ * jnp.imag(omega_s1)
        return omega_total_real, omega_total_imag

    # @jax.jit
    def body_fun2(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = jax.lax.dynamic_index_in_dim(y, i, axis=0)

        omega_s2 = omega_sour(sdata2, ss_data["u_iso"][i], ss_data["v_iso"][i], N, d_0)

        omega_total_real += y_ * jnp.real(omega_s2)
        omega_total_imag += y_ * jnp.imag(omega_s2)
        return omega_total_real, omega_total_imag

    # @jax.jit
    def body_fun3(i, carry):
        omega_total_real, omega_total_imag = carry

        y_ = jax.lax.dynamic_index_in_dim(y, i, axis=0)

        # Need to evlauate three omegas
        omega_s3 = omega_sour(sdata3, ss_data["u_iso"][i], ss_data["v_iso"][i], N, d_0)

        omega_total_real += y_ * jnp.real(omega_s3)
        omega_total_imag += y_ * jnp.imag(omega_s3)
        return omega_total_real, omega_total_imag

    omega_total_real1, omega_total_imag1 = fori_loop(
        0,
        r,
        body_fun1,
        (jnp.zeros_like(sdata1["theta"]), jnp.zeros_like(sdata1["theta"])),
    )

    omega_total_real2, omega_total_imag2 = fori_loop(
        0,
        r,
        body_fun2,
        (jnp.zeros_like(sdata2["theta"]), jnp.zeros_like(sdata2["theta"])),
    )

    omega_total_real3, omega_total_imag3 = fori_loop(
        0,
        r,
        body_fun3,
        (jnp.zeros_like(sdata3["theta"]), jnp.zeros_like(sdata3["theta"])),
    )

    # Assume periodicity on the values of the sources so
    return (
        (sdata1["lambda_iso"] ** (-1))
        * (
            -omega_total_imag1
            * sdata1["e_v"].T  # - cross(sdata["n_rho"],sdata["e_u"]).T
            - omega_total_imag2 * sdata2["e_v"].T
            - omega_total_imag2 * sdata3["e_v"].T
            + omega_total_real1
            * sdata1["e_u"].T  # - cross(sdata["n_rho"],sdata["e_v"]).T
            + omega_total_real2 * sdata2["e_u"].T
            + omega_total_real2 * sdata3["e_u"].T
        )
    ).T


def K_sour(
    p_M,
    p_N,
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
    omega_sour_fun = omega_sour(sdata1, 
                                ss_data["u_iso"], 
                                ss_data["v_iso"], 
                                N, 
                                d_0)
                      #+ omega_sour(sdata2, 
                      #          ss_data["u_iso"], 
                      #          ss_data["v_iso"], 
                      #          N, 
                      #          d_0) 
                      #+ omega_sour(sdata3, 
                      #          ss_data["u_iso"], 
                      #          ss_data["v_iso"], 
                      #          N, 
                      #          d_0)
                     #)  #* y[None, :]

    
    K_sour_total = (sdata1["lambda_iso"] ** (-1) * jnp.sum( ( ( - jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_v'][:, :, None]
                                                              + jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_u'][:, :, None] ) * y[None,:])
                                                           , axis = 2
                                                           ).T
                   ).T

    #K_sour_total = jnp.sum(sdata1['e_v'][:, :, None] * y[None,:] , axis = 2) #works
    #import pdb
    #pdb.set_trace()
    #K_sour_total = jnp.sum( ( jnp.imag(omega_sour_fun)[:, None, :] * sdata1['e_v'][:, :, None] ) * y[None,:] , axis = 2)
    #jnp.sum(y)#sdata1['e_v']
    return K_sour_total