"""Compute functions for discrete conductivity.

Notes
-----

"""

import jax
import jax.numpy as jnp
from jax import jacfwd, jit

from desc.backend import jnp
from desc.derivatives import Derivative
from desc.utils import cross, dot, flatten_list

from desc.grid import Grid, LinearGrid
from desc.fns_simp import _compute_magnetic_field_from_Current
from desc.potential_functions import add_extra_periodic

from desc.optimize import lsqtr, lsq_auglag
from scipy.optimize import NonlinearConstraint 
from scipy.interpolate import griddata

from desc.discrete_sigma import ( B_discrete_sigma, B_discrete_V, B_discrete,
                                K_discrete_sigma, K_discrete_V, K_discrete, 
                                dV, 
                                G_ye, G_ye_spe, G_ye_le_spe, sigma_from_y, 
                                divergence, divergence_sigma, divergence_V,
                                alt_grid, add_extra_coords, discrete_grids)

from desc.fns_simp import surf_int

def rounds(surf_winding,
           p_M,p_N,
           eq, egrid, edata, B_target,
           mu_div_y,mu_pen_y,
           mu_div_v,
           iters,
           n_rounds, n_start):

    counter = 0
    
    for i in range(n_start, n_start + n_rounds):
    
        if i == 0:

            # Pre-compute data
            ( grid_center, grid_theta, grid_zeta, 
            data_center, data_theta, data_zeta, 
            delta_hor_edge, delta_ver_edge,
            delta_l, delta_r, delta_d, delta_u,
            l_l, l_r, l_d, l_u, 
             data_center_mod, grid_center_mod) = discrete_grids(p_M,p_N, surf_winding)
            
            # Let's start with our guess for z_mn and find a solution for V_mn
            # Initial guess of edge conductivities
            #y0 = jnp.load('/scratch/gpfs/EKOLEMEN/fcastro/discrete/dec2025/prog/01_reg_nodiv_nopen/ymn_round_2.npy')
            #y0 = jnp.concatenate((data_zeta['sigma'],data_theta['sigma']))
            y0 = jnp.ones((grid_center.M*2+2) * (grid_center.N*2+1)*2)
            
            # Build matrices with voltage differences
            theta_mod = add_extra_coords(data_center["theta"], grid_center.N*2+1, grid_center.M*2+1)
            x = (theta_mod.T).flatten() # Initial guess for the voltage
            
            vmn_i = V_soln(x, y0,
                           surf_winding,
                           data_center, grid_center,
                           data_theta, data_zeta,
                           delta_hor_edge, delta_ver_edge,
                           delta_l, delta_r,
                           delta_d, delta_u,
                           l_l, l_r,
                           l_d, l_u,
                           eq, egrid, edata, B_target,
                           grid_theta, grid_zeta,
                           mu_div_v,
                           )
            
            # Now let's solve for y with our previous solution for vmn_i
            # Now let's solve for y with our previous solution for vmn_i
            ymn_i = y_soln(y0,vmn_i,
                           surf_winding,
                           data_center, grid_center,
                           data_theta, data_zeta,
                           delta_hor_edge, delta_ver_edge,
                           delta_l, delta_r, delta_d, delta_u,
                           l_l, l_r, l_d, l_u,
                           data_center_mod, grid_center_mod,
                           eq, egrid, edata, B_target,
                           grid_theta, grid_zeta,
                           mu_div_y, mu_pen_y,
                           iters,
                           )

            jnp.save('vmn_round_' + str(i) + '.npy',vmn_i)
            jnp.save('ymn_round_' + str(i) + '.npy',ymn_i)

            counter = counter + 1
            
        else:

            if counter == 0:

                # Pre-compute data
                (grid_center, grid_theta, grid_zeta,
                 data_center, data_theta, data_zeta,
                 delta_hor_edge, delta_ver_edge,
                 delta_l, delta_r, delta_d, delta_u,
                 l_l, l_r, l_d, l_u, 
                 data_center_mod, grid_center_mod) = discrete_grids(p_M,p_N, surf_winding)
            
                ymn_i = jnp.load('ymn_round_' + str(i-1) + '.npy')
                vmn_i = jnp.load('vmn_round_' + str(i-1) +'.npy')

                counter = counter + 1

            vmn_i = V_soln(vmn_i, ymn_i,
                           surf_winding,
                           data_center, grid_center,
                           data_theta, data_zeta,
                           delta_hor_edge, delta_ver_edge,
                           delta_l, delta_r,
                           delta_d, delta_u,
                           l_l, l_r, l_d, l_u,
                           eq, egrid, edata, B_target,
                           grid_theta, grid_zeta,
                           mu_div_v,
                           )
            
            # Now let's solve for y with our previous solution for vmn_i
            ymn_i = y_soln(ymn_i,vmn_i,
                           surf_winding,
                           data_center, grid_center,
                           data_theta, data_zeta,
                           delta_hor_edge, delta_ver_edge,
                           delta_l, delta_r,
                           delta_d, delta_u,
                           l_l, l_r, l_d, l_u,
                           data_center_mod, grid_center_mod,
                           eq, egrid, edata, B_target,
                           grid_theta, grid_zeta,
                           mu_div_y, mu_pen_y,
                           iters,
                           )

            counter = counter + 1
            
            jnp.save('vmn_round_' + str(i) + '.npy',vmn_i)
            jnp.save('ymn_round_' + str(i) + '.npy',ymn_i)

    return vmn_i, ymn_i#, ymn_i_fixed
    
    
def y_soln(x,V_mn,
           surf_winding,
           data_center, grid_center,
           data_theta, data_zeta,
           delta_hor_edge, delta_ver_edge,
           delta_hor_edge_l, delta_hor_edge_r,
           delta_ver_edge_d, delta_ver_edge_u,
           l_e_ver_l, l_e_ver_r,
           l_e_hor_d, l_e_hor_u,
           data_center_mod, grid_center_mod,
           eq, egrid, edata, B_target,
           grid_theta, grid_zeta,
           mu_div_y, mu_pen_y,
           iters,
           ):

    y_res = lambda x: eqn_residual_y(x, V_mn,
                                     surf_winding,
                                     data_center, grid_center,
                                     data_theta, data_zeta,
                                     delta_hor_edge, delta_ver_edge,
                                     delta_hor_edge_l, delta_hor_edge_r,
                                     delta_ver_edge_d, delta_ver_edge_u,
                                     l_e_ver_l, l_e_ver_r,
                                     l_e_hor_d, l_e_hor_u,
                                     data_center_mod, grid_center_mod,
                                     eq, egrid, edata, B_target,
                                     grid_theta, grid_zeta,
                                     mu_div_y, mu_pen_y,
                                    )

        
    # Run the solver
    res = lsq_auglag(y_res,
                     x,
                     jac = jax.jacfwd(y_res),
                     bounds = (-jnp.inf, jnp.inf),
                     #constraint = nlc,
                     constraint = None,
                     args=(),
                     x_scale="jac",
                     ftol=1e-14,
                     xtol=1e-14,
                     gtol=1e-14,
                     ctol=1e-14,
                     verbose=1,
                     maxiter=iters,
                     callback=None,
                     options={},
                    )
    
    return res.x

def eqn_residual_y(z,V_mn,
                   surf_winding,
                   data_center, grid_center,
                   data_theta, data_zeta,
                   delta_hor_edge, delta_ver_edge,
                   delta_hor_edge_l, delta_hor_edge_r,
                   delta_ver_edge_d, delta_ver_edge_u,
                   l_e_ver_l, l_e_ver_r,
                   l_e_hor_d, l_e_hor_u,
                   data_center_mod, grid_center_mod,
                   eq, egrid, edata, B_target,
                   grid_theta, grid_zeta,
                   mu_div_y, mu_pen_y,
                   ):

    #import pdb
    #pdb.set_trace()
    error_sv = bsv_residual_y(z,V_mn,
                              surf_winding,
                             data_center, grid_center,
                             data_theta, data_zeta,
                             delta_hor_edge, delta_ver_edge,
                             delta_hor_edge_l, delta_hor_edge_r,
                             delta_ver_edge_d, delta_ver_edge_u,
                             l_e_ver_l, l_e_ver_r,
                             l_e_hor_d, l_e_hor_u,
                             eq, egrid, edata, B_target,
                             grid_theta, grid_zeta,
                            )

    #import pdb
    #pdb.set_trace()
    error_div = divergence_sigma(z, V_mn,
                               data_center, grid_center,
                               delta_hor_edge, delta_ver_edge,
                               delta_hor_edge_l, delta_hor_edge_r,
                               delta_ver_edge_d, delta_ver_edge_u,
                               l_e_ver_l, l_e_ver_r,
                               l_e_hor_d, l_e_hor_u,
                               )

    error_y_val = pen_y(z)

    error_B = surf_int(error_sv, edata, egrid) #jnp.sum(error_sv) # Error in the target magnetic field
    error_div_y = surf_int(error_div ** 2, data_center_mod, grid_center_mod) # Error in divergence of current
    error_y = jnp.sum(error_y_val)
    #surf_int(error_y_val, data_center_mod, grid_center_mod) # jnp.sum(error_y_val) # Error in values of discrete conductivity

    #error_B = jnp.sum(error_sv)
    #surf_int(error_sv, edata, egrid)
    #error_div_y = jnp.sum(error_div ** 2)
    #error_y = jnp.sum(error_y_val ** 2)
    #surf_int(error_div ** 2, data_center, grid_center)
    #error_pen_y = surf_int(error_div, data_center, grid_center)
    
    return jnp.asarray([ error_B 
                        + mu_div_y * error_div_y 
                        + mu_pen_y * error_y ])

def bsv_residual_y(z,V_mn,
                   surf_winding,
                 data_center, grid_center,
                 data_theta, data_zeta,
                 delta_hor_edge, delta_ver_edge,
                 delta_hor_edge_l, delta_hor_edge_r,
                 delta_ver_edge_d, delta_ver_edge_u,
                 l_e_ver_l, l_e_ver_r,
                 l_e_hor_d, l_e_hor_u,
                 eq, egrid, edata, B_target,
                 grid_theta, grid_zeta,
                 #tau, eps,
                ):

    B_sv = B_discrete_sigma(z, V_mn,
                            surf_winding,
                          data_center, grid_center,
                          data_theta, data_zeta,
                          delta_hor_edge, delta_ver_edge,
                          delta_hor_edge_l, delta_hor_edge_r,
                          delta_ver_edge_d, delta_ver_edge_u,
                          l_e_ver_l, l_e_ver_r,
                          l_e_hor_d, l_e_hor_u,
                          eq, egrid, edata,
                          grid_theta, grid_zeta,
                          #tau, eps,
                         )

    error = B_sv - B_target
    
    return dot(error, error)

    
# Penalty to force y to be closer to either 0 or
def pen_y(y):

    # Reshape the array to match the number of entries
    temp = y.reshape(int(y.shape[0]/2),2)
    return ( (temp[:,0] - 1) ** 2 ) * temp[:,0] ** 2 + ( (temp[:,1] - 1) ** 2 ) * temp[:,1] ** 2


def V_soln(x, z_mn,
                   surf_winding,
                 data_center, grid_center,
                 data_theta, data_zeta,
                 delta_hor_edge, delta_ver_edge,
                 delta_hor_edge_l, delta_hor_edge_r,
                 delta_ver_edge_d, delta_ver_edge_u,
                 l_e_ver_l, l_e_ver_r,
                 l_e_hor_d, l_e_hor_u,
                 eq, egrid, edata, B_target,
                 grid_theta, grid_zeta,
           mu_div_v,
                ):

    v_res = lambda x : bsv_residual_V(x, z_mn,
                                      surf_winding,
                                     data_center, grid_center,
                                     data_theta, data_zeta,
                                     delta_hor_edge, delta_ver_edge,
                                     delta_hor_edge_l, delta_hor_edge_r,
                                     delta_ver_edge_d, delta_ver_edge_u,
                                     l_e_ver_l, l_e_ver_r,
                                     l_e_hor_d, l_e_hor_u,
                                     eq, egrid, edata, B_target,
                                     grid_theta, grid_zeta,
                                     # tau,
                                    )

    error_div_v = lambda x : divergence_V(x,z_mn,
                                           data_center, grid_center,
                                           delta_hor_edge, delta_ver_edge,
                                           delta_hor_edge_l, delta_hor_edge_r,
                                           delta_ver_edge_d, delta_ver_edge_u,
                                           l_e_ver_l, l_e_ver_r,
                                           l_e_hor_d, l_e_hor_u,
                                           #tau, eps,
                                           )

    A = jax.jacfwd(v_res)(x)
    D = jax.jacfwd(error_div_v)(x)
    
    #soln = jnp.linalg.pinv(A.T@A + mu_div_v * D.T@D) @  (A.T @ B_target.flatten())
    return jnp.linalg.pinv(A.T@A + mu_div_v * D.T@D) @  (A.T @ B_target.flatten())
    
    
def bsv_residual_V(x, z_mn,
                   surf_winding,
                     data_center, grid_center,
                     data_theta, data_zeta,
                     delta_hor_edge, delta_ver_edge,
                     delta_hor_edge_l, delta_hor_edge_r,
                     delta_ver_edge_d, delta_ver_edge_u,
                     l_e_ver_l, l_e_ver_r,
                     l_e_hor_d, l_e_hor_u,
                     eq, egrid, edata, B_target,
                     grid_theta, grid_zeta,
                     # tau,
                    ):

    # This function defines a problem that is linear in V
    B_sv = B_discrete_V(x,z_mn,
                        surf_winding,
                      data_center, grid_center,
                      data_theta, data_zeta,
                      delta_hor_edge, delta_ver_edge,
                      delta_hor_edge_l, delta_hor_edge_r,
                      delta_ver_edge_d, delta_ver_edge_u,
                      l_e_ver_l, l_e_ver_r,
                      l_e_hor_d, l_e_hor_u,
                      eq, egrid, edata,
                      grid_theta, grid_zeta,
                      # tau,
                     )
    
    return B_sv.flatten()