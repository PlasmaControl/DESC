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

def rounds(surf_winding,
           p_M,p_N,
           eq, egrid, edata, B_target,
           grid_theta, grid_zeta,
           mu_div_y, mu_div_v,
           mu_y,
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
            y0 = jnp.load('/home/fcastro/discr/dec2025/clean/45x45/nodiv/ymn_round_45.npy')
            #y0 = jnp.ones((grid_center.M*2+2) * (grid_center.N*2+1)*2)
            
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
                           mu_div_v
                            )
            
            # Now let's solve for y with our previous solution for vmn_i
            ymn_i = y_soln(y0,vmn_i,
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
           mu_div_y, mu_y,
                           )

            jnp.save('vmn_round_' + str(i) + '.npy',vmn_i)
            jnp.save('ymn_round_' + str(i) + '.npy',ymn_i)

            counter = counter + 1
            
        else:

            if counter == 0:

                # Pre-compute data
                ( grid_center, grid_theta, grid_zeta, 
                data_center, data_theta, data_zeta, 
                delta_hor_edge, delta_ver_edge,
                delta_l, delta_r, delta_d, delta_u,
                l_l, l_r, l_d, l_u, 
                 data_center_mod, grid_center_mod) = discrete_grids(p_M,p_N, surf_winding)
            
                ymn_i = jnp.load('ymn_round_' + str(i-1) + '.npy')
                vmn_i = jnp.load('vmn_round_' + str(i-1) +'.npy')

                counter = counter + 1

            vmn_i = V_soln(vmn_i, ymn_i,#_fixed,
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
                           mu_div_v
                            )

            # Now let's solve for sigma with our previous solution for vmn_i
            ymn_i = y_soln(ymn_i, vmn_i,
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
                           mu_div_y, mu_y,
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
                 eq, egrid, edata, B_target,
                 grid_theta, grid_zeta,
           mu_div_y, mu_y,
                ):

    y_res = lambda x: bsv_residual_y(x, V_mn,
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

    error_div_y = lambda x : divergence_sigma(x,V_mn,
                                           data_center, grid_center,
                                           delta_hor_edge, delta_ver_edge,
                                           delta_hor_edge_l, delta_hor_edge_r,
                                           delta_ver_edge_d, delta_ver_edge_u,
                                           l_e_ver_l, l_e_ver_r,
                                           l_e_hor_d, l_e_hor_u,
                                           )
    
    A = jax.jacfwd(y_res)(x)
    D = jax.jacfwd(error_div_y)(x)
   
    mat = A.T @ A + mu_div_y * D.T@D + mu_y * jnp.eye(A.shape[1],A.shape[1])
    
    soln = jnp.linalg.pinv(mat) @ (A.T @ B_target.flatten() + 1/2 * mu_y)

    return soln

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
                                    )

    error_div_v = lambda x : divergence_V(x,z_mn,
                                           data_center, grid_center,
                                           delta_hor_edge, delta_ver_edge,
                                           delta_hor_edge_l, delta_hor_edge_r,
                                           delta_ver_edge_d, delta_ver_edge_u,
                                           l_e_ver_l, l_e_ver_r,
                                           l_e_hor_d, l_e_hor_u,
                                           )

    A = jax.jacfwd(v_res)(x)
    D = jax.jacfwd(error_div_v)(x)
    
    soln = jnp.linalg.pinv(A.T@A 
                           + mu_div_v * D.T@D
                          ) @  (A.T @ B_target.flatten())
    return soln
    

def bsv_residual_y(x,V_mn,
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

    B_sv = B_discrete_sigma(x, V_mn,
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
                         )

    
    
    return B_sv.flatten()

    
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

    error_BV = B_sv - B_target
    
    
    return error_BV.flatten()