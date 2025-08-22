from interpax import interp2d

from desc.backend import fori_loop, jax, jit, jnp
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.find_dips import biot_savart_general

from desc.grid import Grid, LinearGrid
from desc.utils import cross, dot

from .find_sour_test import (_compute_magnetic_field_from_Current, 
                            v1_eval, chi_reg, v1_prime_eval, comp_loc, 
                            comp_loc, add_extra, add_extra_periodic, add_extra_coords, alt_grid, 
                            iso_coords_interp,
                            )

def bn_res(p_M, p_N, 
           sdata1,
           sdata2,
           sdata3,
           sgrid, surface,
           y, N, d_0, dt, dz, 
           coords,  
           iso_data,
           dl_data, dr_data, dd_data, du_data):
    
    B0_dips = B_dips(p_M, p_N, 
                     sdata1,
                     sdata2,
                     sdata3,
                     sgrid, surface,
                     y,  N, d_0, dt, dz,
                     coords,
                     dl_data, dr_data, dd_data, du_data
                    )

    B0_sticks = B_sticks(p_M, p_N,
                         sdata1,
                         sgrid,
                         surface,
                         y,
                         dt,dz,
                         coords, 
                         dl_data, dr_data, dd_data, du_data)

    # Minus sign for sticks to change the polarity of the current in the sticks
    return B0_dips - B0_sticks
    
def B_dips(p_M, p_N,
           sdata1,
           sdata2,
           sdata3,
           sgrid,
           surface,
           y,
           N,
           d_0, dt,dz,
           coords,
           dl_data, dr_data, dd_data, du_data,):


    return _compute_magnetic_field_from_Current(sgrid, 
                                                K_dips(p_M, p_N,
                                                       sdata1,
                                                       sdata2,
                                                       sdata3,
                                                       sgrid, surface,
                                                       y,
                                                       N, 
                                                       d_0, dt,dz,
                                                       dl_data, dr_data, dd_data, du_data), 
                                                surface, 
                                                sdata1,
                                                coords,
                                                basis="rpz")


def B_sticks(p_M, p_N,
                 sdata,
                 sgrid,
                 surface,
                 y,
             dt,dz,
             coords,
             dl_data, dr_data, dd_data, du_data,
            ):

    pls_points = rpz2xyz(coords)
    
    r = dl_data["theta"].shape[0]
    
    #b_sticks_total = 0
    #for i in range(0,r):
    def body_fun(i, carry):
        
        b_stick_pol = stick(du_data["x"][i], # u-pos of second dipole
                            dd_data["x"][i], # u-pos of first dipole
                            pls_points,
                            sgrid,
                            basis = "rpz",
                            )
        
        b_stick_tor = stick(dr_data["x"][i], # u-pos of second dipole
                            dl_data["x"][i], # u-pos of first dipole
                            pls_points,
                            sgrid,
                            basis = "rpz",
                            )

        carry += y[i] * b_stick_pol + y[i+r] * b_stick_tor
        return carry
    
    return fori_loop(0, r, body_fun, jnp.zeros_like(pls_points))


def K_dips(p_M, p_N,
           sdata1,
           sdata2,
           sdata3,
           sgrid,
           surface,
           y,
           N,
           d_0, dt, dz,
           dl_data, dr_data, dd_data, du_data):
    
    # This could be moved to the build component of the objective
    assert (p_M * 2+1)*(p_N * 2+1) == dl_data["theta"].shape[0] , "Check that the number of dipole locations coincide with the number of dipoles"
    r = dl_data["theta"].shape[0]  # Make r a Python int for indexing

    def body_fun1(i, carry):
        omega_total_real, omega_total_imag = carry

        omega_pol1 = omega_pair(
            sdata1,
            du_data["u_iso"][i], du_data["v_iso"][i],
            dd_data["u_iso"][i], dd_data["v_iso"][i],
            N, d_0
        )

        omega_tor1 = omega_pair(
            sdata1,
            dr_data["u_iso"][i], dr_data["v_iso"][i],
            dl_data["u_iso"][i], dl_data["v_iso"][i],
            N, d_0
        )

        omega_total_real += y[i] * jnp.real(omega_pol1) + y[i + r] * jnp.real(omega_tor1)
        omega_total_imag += y[i] * jnp.imag(omega_pol1) + y[i + r] * jnp.imag(omega_tor1)
        return omega_total_real, omega_total_imag

    def body_fun2(i, carry):
        omega_total_real, omega_total_imag = carry

        omega_pol2 = omega_pair(
            sdata2,
            du_data["u_iso"][i], du_data["v_iso"][i],
            dd_data["u_iso"][i], dd_data["v_iso"][i],
            N, d_0
        )

        omega_tor2 = omega_pair(
            sdata2,
            dr_data["u_iso"][i], dr_data["v_iso"][i],
            dl_data["u_iso"][i], dl_data["v_iso"][i],
            N, d_0
        )

        omega_total_real += y[i] * jnp.real(omega_pol2) + y[i + r] * jnp.real(omega_tor2)
        omega_total_imag += y[i] * jnp.imag(omega_pol2) + y[i + r] * jnp.imag(omega_tor2)
        return omega_total_real, omega_total_imag

    def body_fun3(i, carry):
        omega_total_real, omega_total_imag = carry

        omega_pol3 = omega_pair(
            sdata3,
            du_data["u_iso"][i], du_data["v_iso"][i],
            dd_data["u_iso"][i], dd_data["v_iso"][i],
            N, d_0
        )

        omega_tor3 = omega_pair(
            sdata3,
            dr_data["u_iso"][i], dr_data["v_iso"][i],
            dl_data["u_iso"][i], dl_data["v_iso"][i],
            N, d_0
        )

        omega_total_real += y[i] * jnp.real(omega_pol3) + y[i + r] * jnp.real(omega_tor3)
        omega_total_imag += y[i] * jnp.imag(omega_pol3) + y[i + r] * jnp.imag(omega_tor3)
        return omega_total_real, omega_total_imag

    omega_total_real1, omega_total_imag1 = fori_loop(0, r, body_fun1, (jnp.zeros_like(sdata1["theta"]),
                                                                       jnp.zeros_like(sdata1["theta"]))
                                                    )

    omega_total_real2, omega_total_imag2 = fori_loop(0, r, body_fun2, (jnp.zeros_like(sdata2["theta"]),
                                                                       jnp.zeros_like(sdata2["theta"]))
                                                    )

    omega_total_real3, omega_total_imag3 = fori_loop(0, r, body_fun3, (jnp.zeros_like(sdata3["theta"]),
                                                                       jnp.zeros_like(sdata3["theta"]))
                                                    )

    return ( ( ( sdata1["lambda_iso"] ** (-1) ) * ( omega_total_imag1 * cross(sdata1["n_rho"],sdata1["e_u"]).T
                                                   + omega_total_real1 * cross(sdata1["n_rho"],sdata1["e_v"]).T
                                                    + omega_total_imag2 * cross(sdata2["n_rho"],sdata2["e_u"]).T 
                                                    + omega_total_real2 * cross(sdata2["n_rho"],sdata2["e_v"]).T
                                                    + omega_total_imag3 * cross(sdata3["n_rho"],sdata3["e_u"]).T 
                                                    + omega_total_real3 * cross(sdata3["n_rho"],sdata3["e_v"]).T
                                                 )
              ).T
            )
    
def f_pair(data_or,
           u1_, v1_, # first dipole
           u2_, v2_, # second dipole
           N, d_0):
    
    # Evaluate the dipoles on the grid of vortices
    w1 = comp_loc(u1_, v1_,)
    w2 = comp_loc(u2_, v2_,)
    
    v_1_num = v1_eval(w1, N, d_0, data_or)
    v_1_den = v1_eval(w2, N, d_0, data_or)
    
    f1_reg = f_reg(w1, d_0, data_or)
    f2_reg = f_reg(w2, d_0, data_or)

    return ( jnp.log( v_1_num ) - jnp.log( v_1_den )
            - 2 * jnp.pi * jnp.real( w1 - w2 ) / ( data_or["omega_1"] ** 2 * data_or["tau_2"] ) * data_or["w"] 
            - 1/2 * ( f1_reg - f2_reg )
           )

def omega_pair(data_or,
               u1_, v1_, # first dipole
               u2_, v2_, # second dipole
               N,
               d_0,
              ):
    
    w1 = comp_loc(u1_, v1_,)
    w2 = comp_loc(u2_, v2_,)
    
    v_1_num = v1_eval(w1, N,  d_0, data_or)
    v_1_den = v1_eval(w2, N, d_0, data_or) 
    v_1_num_prime = v1_prime_eval(w1, N, d_0, data_or)
    v_1_den_prime = v1_prime_eval(w2, N, d_0, data_or)
    
    chi_reg_1 = chi_reg(w1, d_0, data_or)
    chi_reg_2 = chi_reg(w2, d_0, data_or)
    
    omega = ( ( v_1_num_prime / v_1_num - v_1_den_prime / v_1_den ) # Regularized near the vortex cores
             - 2 * jnp.pi * jnp.real( w1 - w2 ) / ( data_or["omega_1"] ** 2 * data_or["tau_2"]) 
             + 1 / 2 * ( chi_reg_1 - chi_reg_2 ) # Additional terms with regularization close to the vortex core
            )
    
    return omega

def stick(p2_, # second point of the stick
          p1_, # first point of the stick
          plasma_points_, # points on the plasma surface
          surface_grid,#Kgrid,
          basis = "rpz",
          ):
    
    def nfp_loop(j, f):
        # calculate (by rotating) rs, rs_t, rz_t
        phi2 = ( p2_[2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        phi1 = ( p1_[2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )
        #(surface_grid.nodes[:, 2] + j * 2 * jnp.pi / surface_grid.NFP ) % ( 2 * jnp.pi )

        p2s = jnp.vstack((p2_[0], phi2, p2_[2])).T
        p2s = rpz2xyz(p2s)
        p1s = jnp.vstack((p1_[0], phi1, p1_[2])).T
        p1s = rpz2xyz(p1s)
        
        a_s = p2s - p1s
        
        b_s = p1s - plasma_points_
        
        c_s = p2s - plasma_points_
        
        c_sxa_s = cross(c_s, a_s)
    
        f += 1e-7 * ( ( jnp.sum(c_sxa_s*c_sxa_s, axis = 1) * jnp.sum(c_s * c_s, axis = 1) ** (1/2)
                       ) ** (-1) * (jnp.sum(a_s * c_s) - jnp.sum(a_s * b_s)
                                    ) * c_sxa_s.T 
                     ).T
        return f
    
    b_stick = fori_loop(0, surface_grid.NFP, nfp_loop, jnp.zeros_like(plasma_points_))
    
    if basis == "rpz":
        b_stick = xyz2rpz_vec(b_stick, x=plasma_points_[:, 0], y=plasma_points_[:, 1])
        
    return b_stick

def shift_grid(theta, zeta, dt, dz, w_surface, name):
    
    l_zeta = zeta - dz/2
    r_zeta = zeta + dz/2
    d_theta = theta - dt/2
    u_theta = theta + dt/2
        
    # Find grids for dipoles
    l_grid = alt_grid(theta,l_zeta)
    r_grid = alt_grid(theta,r_zeta)

    d_grid = alt_grid(d_theta,zeta)
    u_grid = alt_grid(u_theta,zeta)
    
    # Evaluate data on grids of dipoles
    l_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = l_grid)
    r_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = r_grid)
    d_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = d_grid)
    u_data = w_surface.compute(["theta","zeta","e^theta_s","e^zeta_s",
                                "x",
                               ], grid = u_grid)
    
    return (iso_coords_interp(name, l_data, l_grid, w_surface),
            iso_coords_interp(name, r_data, r_grid, w_surface),
            iso_coords_interp(name, d_data, d_grid, w_surface),
            iso_coords_interp(name, u_data, u_grid, w_surface)
           )