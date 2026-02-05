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

#from desc.optimize import lsqtr, lsq_auglag
#from scipy.optimize import NonlinearConstraint 
from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid
from scipy.interpolate import griddata

from desc.fns_simp import surf_int

def rounds(surf_winding,
           p_M, p_N,
           eq, egrid, edata, B_target,
           mu_div_y, mu_pen_y,
           mu_div_v,
           #tau, eps, 
           iters, tol, K_batch,
           y_name, v_name,
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
            m_size = grid_center.M * 2 + 1
            n_size = grid_center.N * 2 + 1
            
            y0 = jnp.load(y_name)
            v0 = jnp.load(v_name)
            
            # Find y (sigma) with a fixed solution for V
            ymn_i = y_soln_discrete(y0,v0,
                                    surf_winding,
                                    data_center, grid_center,
                                    data_theta, data_zeta,
                                    delta_hor_edge, delta_ver_edge,
                                    delta_l, delta_r,
                                    delta_d, delta_u,
                                    l_l, l_r,
                                    l_d, l_u,
                                    data_center_mod, grid_center_mod,
                                    eq, egrid, edata, B_target,
                                    grid_theta, grid_zeta,
                                   mu_div_y, mu_pen_y,
                                   #tau, eps,
                                   iters, tol, K_batch,
                           )
            
            # Now let's solve for V with our previous solution for y (sigma)
            vmn_i = V_soln(v0, ymn_i,
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
                           #tau, eps,
                           )
            
            jnp.save('vmn_round_' + str(i) + '.npy',vmn_i)
            jnp.save('ymn_round_' + str(i) + '.npy',ymn_i)

            counter = counter + 1
            
        else:

            if counter == 0:

                # Compute the data needed. This is necessary to continue the optimization using a previous solution.
                ( grid_center, grid_theta, grid_zeta, 
                 data_center, data_theta, data_zeta, 
                 delta_hor_edge, delta_ver_edge,
                 delta_l, delta_r, delta_d, delta_u,
                 l_l, l_r, l_d, l_u,
                 data_center_mod, grid_center_mod) = discrete_grids(p_M,p_N, surf_winding)
                
                ymn_i = jnp.load('ymn_round_' + str(i-1) + '.npy')
                vmn_i = jnp.load('vmn_round_' + str(i-1) +'.npy')

                counter = counter + 1

            ymn_i = y_soln_discrete(ymn_i, vmn_i,
                                   surf_winding,
                                   data_center, grid_center,
                                   data_theta, data_zeta,
                                   delta_hor_edge, delta_ver_edge,
                                   delta_l, delta_r,
                                   delta_d, delta_u,
                                   l_l, l_r,
                                   l_d, l_u,
                                    data_center_mod, grid_center_mod,
                                   eq, egrid, edata, B_target,
                                   grid_theta, grid_zeta,
                                   mu_div_y, mu_pen_y,
                                   #tau, eps,
                                   iters,tol,
                                    K_batch,
                                   )

            
            # Now let's solve for V with our previous solution for y (sigma)
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
                           mu_div_v,
                           #tau, eps
                           )
            
            counter = counter + 1
            
            jnp.save('vmn_round_' + str(i) + '.npy',vmn_i)
            jnp.save('ymn_round_' + str(i) + '.npy',ymn_i)
            
    return vmn_i, ymn_i


def y_soln_discrete(z0, V_mn,
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
           #tau, eps,
           max_iter, tol,
           K_batch, 
           #K_batch=12000,
          ):
    
    """
    Batch greedy solver:
    - Evaluate all single-bit flips
    - Select K_batch best improving flips
    - Apply them simultaneously
    """

    z = z0

    # Error function (scalar)
    error_func = lambda z: eqn_residual_y(z, V_mn,
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
                                        #tau, eps,
                                    )[0]

    N = z.shape[0]
    idx = jnp.arange(N)

    
    def eval_flip(i):
        z0 = z.at[i].set(0.0)
        z1 = z.at[i].set(1.0)
        return jnp.stack([ error_func(z0), error_func(z1), ])

    
    for k in range(max_iter):

        base_error = error_func(z)

        # Evaluate all flips (N, 2)
        errors = jax.vmap(eval_flip)(idx)

        # Best value and error per index
        best_val = jnp.argmin(errors, axis=1)      # (N,)
        best_err = jnp.min(errors, axis=1)         # (N,)

        # Improvement per index
        improvement = base_error - best_err        # (N,)

        # Keep only improving flips
        #improving = improvement > tol

        # Keep only improving indices
        improving_idx = jnp.where(improvement > 0)[0]
        
        if improving_idx.size == 0:
            break
        
        # Sort only improving indices
        sorted_idx = improving_idx[jnp.argsort(-improvement[improving_idx])]
        
        # Select top-K improving flips
        topk_idx = sorted_idx[:K_batch]
        
        # Apply flips
        z = z.at[topk_idx].set(best_val[topk_idx].astype(z.dtype))

    return z
    

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
                   #tau, eps,
                   ):
    
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
                             #tau, eps,
                            )

    error_div = divergence_sigma(z, V_mn,
                               data_center, grid_center,
                               delta_hor_edge, delta_ver_edge,
                               delta_hor_edge_l, delta_hor_edge_r,
                               delta_ver_edge_d, delta_ver_edge_u,
                               l_e_ver_l, l_e_ver_r,
                               l_e_hor_d, l_e_hor_u,
                               #tau, eps,
                               )

    error_y_val = pen_y(z)

    error_B = surf_int(error_sv, edata, egrid) #jnp.sum(error_sv) # Error in the target magnetic field
    error_div_y = surf_int(error_div ** 2, data_center_mod, grid_center_mod) # Error in divergence of current
    #error_y = surf_int(error_y_val, data_center_mod, grid_center_mod) # jnp.sum(error_y_val) # Error in values of discrete conductivity
    
    return jnp.asarray([ error_B 
                        + mu_div_y * error_div_y 
                        #+ mu_pen_y * error_y 
                       ])


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
def pen_y(y,):#grid_center, data_center):
    return ( (y - 1) ** 2 ) * y ** 2
    #return surf_int( ( (y - 1) ** 2 ) * y ** 2,data_center,grid_center)


def V_soln(x, y_mn,
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
           #tau, eps
           ):
    
    v_res = lambda x : bsv_residual_V(x, y_mn,
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
                                    )
    error_div_v = lambda x : divergence_V(x,y_mn,
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
    
    soln = jnp.linalg.pinv(A.T@A 
                           + mu_div_v * D.T@D
                          ) @  (A.T @ B_target.flatten())
    return soln


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
                     #tau, eps,
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
                      #tau, eps,
                     )

    #error_BV = B_sv - B_target
    
    return B_sv.flatten()

    
def B_discrete(x,
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
                ):

    K_theta, K_zeta = K_discrete(x,
                                 data_center, grid_center,
                                 data_theta, data_zeta,
                                 delta_hor_edge, delta_ver_edge,
                                 delta_hor_edge_l, delta_hor_edge_r,
                                 delta_ver_edge_d, delta_ver_edge_u,
                                 l_e_ver_l, l_e_ver_r,
                                 l_e_hor_d, l_e_hor_u,
                                 )

    B_sv = ( _compute_magnetic_field_from_Current( grid_theta, K_theta, surf_winding, eq, egrid, basis = "rpz" )
            + _compute_magnetic_field_from_Current( grid_zeta, K_zeta, surf_winding, eq, egrid, basis = "rpz" )
           )
    
    return B_sv


def B_discrete_sigma(x,V_mn,
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
                    ):

    K_theta, K_zeta = K_discrete_sigma(x,V_mn,
                                 data_center, grid_center,
                                 data_theta, data_zeta,
                                 delta_hor_edge, delta_ver_edge,
                                 delta_hor_edge_l, delta_hor_edge_r,
                                 delta_ver_edge_d, delta_ver_edge_u,
                                 l_e_ver_l, l_e_ver_r,
                                 l_e_hor_d, l_e_hor_u,
                                 #tau, eps,
                                 )

    B_sv = ( _compute_magnetic_field_from_Current( grid_theta, K_theta, surf_winding, eq, egrid, basis = "rpz" )
            + _compute_magnetic_field_from_Current( grid_zeta, K_zeta, surf_winding, eq, egrid, basis = "rpz" )
           )
    
    return B_sv


def B_discrete_V(x,z_mn,
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
                ):

    K_theta, K_zeta = K_discrete_V(x,z_mn,
                                   data_center, grid_center,
                                   data_theta, data_zeta,
                                   delta_hor_edge, delta_ver_edge,
                                   delta_hor_edge_l, delta_hor_edge_r,
                                   delta_ver_edge_d, delta_ver_edge_u,
                                   l_e_ver_l, l_e_ver_r,
                                   l_e_hor_d, l_e_hor_u,
                                   #tau, eps,
                                   )

    B_sv = ( _compute_magnetic_field_from_Current( grid_theta, K_theta, surf_winding, eq, egrid, basis = "rpz" )
            + _compute_magnetic_field_from_Current( grid_zeta, K_zeta, surf_winding, eq, egrid, basis = "rpz" )
           )
    
    return B_sv


def K_discrete(x,
               data_center, grid_center,
               data_theta, data_zeta,
               delta_hor_edge, delta_ver_edge,
               delta_hor_edge_l, delta_hor_edge_r,
               delta_ver_edge_d, delta_ver_edge_u,
               l_e_ver_l, l_e_ver_r,
               l_e_hor_d, l_e_hor_u,
               #tau, eps,
               ):

    y_mn = x[0 : data_theta['theta'].shape[0]*2]
    V_mn = x[data_theta['theta'].shape[0]*2:] # Size of V_mn is not the same as z_mn becuase of the additional values at the boundaries.
    
    # Convert the probability z to conductivity s. This step adds the conductivities at the boundaries of the coil    
    y_lr_mn = y_mn[0: int(y_mn.shape[0]/2)].reshape(grid_center.N * 2 + 2, grid_center.M * 2 + 1).T
    y_ud_mn = y_mn[int(y_mn.shape[0]/2): y_mn.shape[0]].reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 2).T
    
    g_l_l, g_r_l, g_d_l, g_u_l = G_ye_le_spe(y_lr_mn, y_ud_mn,
                                         grid_center,
                                 delta_hor_edge, delta_ver_edge,
                                 delta_hor_edge_l, delta_hor_edge_r,
                                 delta_ver_edge_d, delta_ver_edge_u,
                                 l_e_ver_l, l_e_ver_r,
                                 l_e_hor_d, l_e_hor_u,
                                 )

    dV_l, dV_r, dV_d, dV_u = dV(V_mn,grid_center)
    
    # Define fluxes
    j_l = g_l_l * dV_l 
    j_r = g_r_l * dV_r 
    j_d = g_d_l * dV_d 
    j_u = g_u_l * dV_u

    # Build the current passing through each edge
    e_sup_theta = ( ( dot( data_theta['e^theta_s'],data_theta['e^theta_s']
                          ) ** (-1/2) * data_theta['e^theta_s'].T 
                    ).T
                  ).reshape(grid_center.M * 2 + 2, grid_center.N * 2 + 1, 3)
    e_sup_zeta = ( ( dot( data_zeta['e^zeta_s'],data_zeta['e^zeta_s']
                         ) ** (-1/2) * data_zeta['e^zeta_s'].T 
                   ).T 
                 ).reshape(grid_center.M * 2 + 1, grid_center.N * 2 + 2, 3)
    
    K_l = j_l[:,:,None] * e_sup_zeta[:,0:-1,:]
    K_r = j_r[:,:,None] * e_sup_zeta[:,1:,:]
    K_d = j_d[:,:,None] * e_sup_theta[0:-1,:,:]
    K_u = j_u[:,:,None] * e_sup_theta[1:,:,:]
    
    K_theta = jnp.concatenate((K_l, K_r[:,-1][:, None, :]),axis = 1).reshape(-1, 3, order='F')
    K_zeta = jnp.concatenate((K_d, K_u[-1,:][None, :, :]),axis = 0).reshape(-1, 3, order='F')
    
    return K_theta, K_zeta

    
def K_discrete_sigma(y_, V_mn,
                   data_center, grid_center,
                   data_theta, data_zeta,
                   delta_hor_edge, delta_ver_edge,
                   delta_hor_edge_l, delta_hor_edge_r,
                   delta_ver_edge_d, delta_ver_edge_u,
                   l_e_ver_l, l_e_ver_r,
                   l_e_hor_d, l_e_hor_u,
                   #tau, eps,
                   ):

    
    y_lr_mn = y_[0: int(y_.shape[0]/2)].reshape(grid_center.N * 2 + 2, grid_center.M * 2 + 1).T
    y_ud_mn = y_[int(y_.shape[0]/2): y_.shape[0]].reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 2).T

    
    g_l_l, g_r_l, g_d_l, g_u_l = G_ye_le_spe(y_lr_mn, y_ud_mn,
                                             grid_center,
                                             delta_hor_edge, delta_ver_edge,
                                             delta_hor_edge_l, delta_hor_edge_r,
                                             delta_ver_edge_d, delta_ver_edge_u,
                                             l_e_ver_l, l_e_ver_r,
                                             l_e_hor_d, l_e_hor_u,
                                             )
    
    dV_l, dV_r, dV_d, dV_u = dV(V_mn,grid_center)
    
    # Define fluxes
    j_l = g_l_l * dV_l 
    j_r = g_r_l * dV_r 
    j_d = g_d_l * dV_d 
    j_u = g_u_l * dV_u

    # Build the current passing through each edge
    e_sup_theta = ( ( dot( data_theta['e^theta_s'],data_theta['e^theta_s']
                          ) ** (-1/2) * data_theta['e^theta_s'].T 
                    ).T
                  ).reshape(grid_center.M * 2 + 2, grid_center.N * 2 + 1, 3)
    e_sup_zeta = ( ( dot( data_zeta['e^zeta_s'],data_zeta['e^zeta_s']
                         ) ** (-1/2) * data_zeta['e^zeta_s'].T 
                   ).T 
                 ).reshape(grid_center.M * 2 + 1, grid_center.N * 2 + 2, 3)
    
    K_l = j_l[:,:,None] * e_sup_zeta[:,0:-1,:]
    K_r = j_r[:,:,None] * e_sup_zeta[:,1:,:]
    K_d = j_d[:,:,None] * e_sup_theta[0:-1,:,:]
    K_u = j_u[:,:,None] * e_sup_theta[1:,:,:]
    
    K_theta = jnp.concatenate((K_l, K_r[:,-1][:, None, :]),axis = 1).reshape(-1, 3, order='F')
    K_zeta = jnp.concatenate((K_d, K_u[-1,:][None, :, :]),axis = 0).reshape(-1, 3, order='F')
    
    return K_theta, K_zeta


def K_discrete_V(x,y_,
                   data_center, grid_center,
                   data_theta, data_zeta,
                   delta_hor_edge, delta_ver_edge,
                   delta_hor_edge_l, delta_hor_edge_r,
                   delta_ver_edge_d, delta_ver_edge_u,
                   l_e_ver_l, l_e_ver_r,
                   l_e_hor_d, l_e_hor_u,
                   #tau, eps,
                   ):
    

    y_lr_mn = y_[0: int(y_.shape[0]/2)].reshape(grid_center.N * 2 + 2, grid_center.M * 2 + 1).T
    y_ud_mn = y_[int(y_.shape[0]/2): y_.shape[0]].reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 2).T
    
    g_l_l, g_r_l, g_d_l, g_u_l = G_ye_le_spe(y_lr_mn, y_ud_mn,
                                             grid_center,
                                             delta_hor_edge, delta_ver_edge,
                                             delta_hor_edge_l, delta_hor_edge_r,
                                             delta_ver_edge_d, delta_ver_edge_u,
                                             l_e_ver_l, l_e_ver_r,
                                             l_e_hor_d, l_e_hor_u,
                                             )


    dV_l, dV_r, dV_d, dV_u = dV(x,grid_center)
    
    # Define fluxes
    j_l = g_l_l * dV_l 
    j_r = g_r_l * dV_r 
    j_d = g_d_l * dV_d 
    j_u = g_u_l * dV_u

    # Build the current passing through each edge
    e_sup_theta = ( ( dot( data_theta['e^theta_s'],data_theta['e^theta_s']
                          ) ** (-1/2) * data_theta['e^theta_s'].T 
                    ).T
                  ).reshape(grid_center.M * 2 + 2, grid_center.N * 2 + 1, 3)
    e_sup_zeta = ( ( dot( data_zeta['e^zeta_s'],data_zeta['e^zeta_s']
                         ) ** (-1/2) * data_zeta['e^zeta_s'].T 
                   ).T 
                 ).reshape(grid_center.M * 2 + 1, grid_center.N * 2 + 2, 3)
    
    K_l = j_l[:,:,None] * e_sup_zeta[:,0:-1,:]
    K_r = j_r[:,:,None] * e_sup_zeta[:,1:,:]
    K_d = j_d[:,:,None] * e_sup_theta[0:-1,:,:]
    K_u = j_u[:,:,None] * e_sup_theta[1:,:,:]
    
    K_theta = jnp.concatenate((K_l, K_r[:,-1][:, None, :]),axis = 1).reshape(-1, 3, order='F')
    K_zeta = jnp.concatenate((K_d, K_u[-1,:][None, :, :]),axis = 0).reshape(-1, 3, order='F')
    
    return K_theta, K_zeta

def divergence(x,
               data_center, grid_center,
               data_theta, data_zeta,
               delta_hor_edge, delta_ver_edge,
               delta_hor_edge_l, delta_hor_edge_r,
               delta_ver_edge_d, delta_ver_edge_u,
               l_e_ver_l, l_e_ver_r,
               l_e_hor_d, l_e_hor_u,
               #tau, eps,
               ):

    y_mn = x[0 : data_theta['theta'].shape[0]*2]
    V_mn = x[data_theta['theta'].shape[0]*2:] # Size of V_mn is not the same as z_mn becuase of the additional values at the boundaries.

    # Convert the probability z to conductivity s. This step adds the conductivities at the boundaries of the coil    
    y_lr_mn = y_mn[0: int(y_mn.shape[0]/2)].reshape(grid_center.N * 2 + 2, grid_center.M * 2 + 1).T
    y_ud_mn = y_mn[int(y_mn.shape[0]/2): y_mn.shape[0]].reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 2).T
    
    g_l, g_r, g_d, g_u = G_ye_spe(y_lr_mn, y_ud_mn,
                                  grid_center,
                                 delta_hor_edge, delta_ver_edge,
                                 delta_hor_edge_l, delta_hor_edge_r,
                                 delta_ver_edge_d, delta_ver_edge_u,
                                 l_e_ver_l, l_e_ver_r,
                                 l_e_hor_d, l_e_hor_u,
                                 # tau,
                                 )

    dV_l, dV_r, dV_d, dV_u = dV(V_mn,grid_center)
    
    # Define fluxes
    q_l = g_l * dV_l 
    q_r = g_r * dV_r 
    q_d = g_d * dV_d 
    q_u = g_u * dV_u
    
    #pdb.set_trace()
    # Sum the fluxes in each cell
    GV = (q_l + q_r + q_d + q_u).T.flatten()
    
    return GV

def divergence_normalized(x,
                          data_center, grid_center,
                          data_theta, data_zeta,
                          delta_hor_edge, delta_ver_edge,
                          delta_hor_edge_l, delta_hor_edge_r,
                          delta_ver_edge_d, delta_ver_edge_u,
                          l_e_ver_l, l_e_ver_r,
                          l_e_hor_d, l_e_hor_u,
                          #tau, eps,
                          ):

    y_mn = x[0 : data_theta['theta'].shape[0]*2]
    V_mn = x[data_theta['theta'].shape[0]*2:] # Size of V_mn is not the same as z_mn becuase of the additional values at the boundaries.

    # Convert the probability z to conductivity s. This step adds the conductivities at the boundaries of the coil    
    y_lr_mn = y_mn[0: int(y_mn.shape[0]/2)].reshape(grid_center.N * 2 + 2, grid_center.M * 2 + 1).T
    y_ud_mn = y_mn[int(y_mn.shape[0]/2): y_mn.shape[0]].reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 2).T
    
    g_l, g_r, g_d, g_u = G_ye_spe(y_lr_mn, y_ud_mn,
                                  grid_center,
                                 delta_hor_edge, delta_ver_edge,
                                 delta_hor_edge_l, delta_hor_edge_r,
                                 delta_ver_edge_d, delta_ver_edge_u,
                                 l_e_ver_l, l_e_ver_r,
                                 l_e_hor_d, l_e_hor_u,
                                 # tau,
                                 )

    dV_l, dV_r, dV_d, dV_u = dV(V_mn,grid_center)
    
    # Define fluxes
    q_l = g_l * dV_l 
    q_r = g_r * dV_r 
    q_d = g_d * dV_d 
    q_u = g_u * dV_u
    
    # Sum the fluxes in each cell
    #GV = (q_l + q_r + q_d + q_u).flatten()
    
    return ( jnp.abs( q_l + q_r + q_d + q_u ) * ( jnp.abs(q_l) + jnp.abs(q_r) + jnp.abs(q_d) + jnp.abs(q_u) + 1e-3) ** (-1) ).T.flatten()


def divergence_sigma(y_, V_mn,
                   data_center, grid_center,
                   delta_hor_edge, delta_ver_edge,
                   delta_hor_edge_l, delta_hor_edge_r,
                   delta_ver_edge_d, delta_ver_edge_u,
                   l_e_ver_l, l_e_ver_r,
                   l_e_hor_d, l_e_hor_u,
               #tau, eps,
                   ):
    
    y_lr_mn = y_[0: int(y_.shape[0]/2)].reshape(grid_center.N * 2 + 2, grid_center.M * 2 + 1).T
    y_ud_mn = y_[int(y_.shape[0]/2): y_.shape[0]].reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 2).T
    
    g_l, g_r, g_d, g_u = G_ye_spe(y_lr_mn, y_ud_mn,
                                  grid_center,
                                 delta_hor_edge, delta_ver_edge,
                                 delta_hor_edge_l, delta_hor_edge_r,
                                 delta_ver_edge_d, delta_ver_edge_u,
                                 l_e_ver_l, l_e_ver_r,
                                 l_e_hor_d, l_e_hor_u,
                                 )

    
    dV_l, dV_r, dV_d, dV_u = dV(V_mn,grid_center)
    
    # Define fluxes
    q_l = g_l * dV_l 
    q_r = g_r * dV_r 
    q_d = g_d * dV_d 
    q_u = g_u * dV_u
    
    # Sum the fluxes in each cell
    GV = (q_l + q_r + q_d + q_u).flatten()
    
    return GV
    
    
def divergence_V(x,y_,
               data_center, grid_center,
               delta_hor_edge, delta_ver_edge,
               delta_hor_edge_l, delta_hor_edge_r,
               delta_ver_edge_d, delta_ver_edge_u,
               l_e_ver_l, l_e_ver_r,
               l_e_hor_d, l_e_hor_u,
               #tau, eps,
               ):

    
    # Convert the probability z to conductivity s
    y_lr_mn = y_[0: int(y_.shape[0]/2)].reshape(grid_center.N * 2 + 2, grid_center.M * 2 + 1).T
    y_ud_mn = y_[int(y_.shape[0]/2): y_.shape[0]].reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 2).T
    
    g_l, g_r, g_d, g_u = G_ye_spe(y_lr_mn, y_ud_mn,
                              grid_center,
                                 delta_hor_edge, delta_ver_edge,
                                 delta_hor_edge_l, delta_hor_edge_r,
                                 delta_ver_edge_d, delta_ver_edge_u,
                                 l_e_ver_l, l_e_ver_r,
                                 l_e_hor_d, l_e_hor_u,
                                 )

    dV_l, dV_r, dV_d, dV_u = dV(x,grid_center)

    
    # Define fluxes
    q_l = g_l * dV_l 
    q_r = g_r * dV_r 
    q_d = g_d * dV_d 
    q_u = g_u * dV_u
    
    #pdb.set_trace()
    # Sum the fluxes in each cell
    GV = (q_l + q_r + q_d + q_u).flatten()
    
    return GV


def dV(V_mn,grid_center):

    V_m = V_mn.reshape( grid_center.N * 2 + 1 + 2, grid_center.M * 2 + 1 + 2 ).T
    V_l = V_m[ 1 : grid_center.M * 2 + 2, 0 : grid_center.N * 2 + 1 ] # Left-values of voltage
    V_c = V_m[ 1 : grid_center.M * 2 + 2, 1 : grid_center.N * 2 + 2 ] # Center-values of voltage
    V_r = V_m[ 1 : grid_center.M * 2 + 2, 2 : grid_center.N * 2 + 3 ] # Right-values of voltage
    
    V_d = V_m[ 0 : grid_center.M * 2 + 1, 1 : grid_center.N * 2 + 2 ] # Left-values of voltage
    V_u = V_m[ 2 : grid_center.M * 2 + 3, 1 : grid_center.N * 2 + 2 ] # Right-values of voltage
    
    dV_l = - V_l + V_c # Left edge voltage-difference
    dV_r = V_r - V_c # Right edge voltage-difference
    dV_d = - V_d + V_c # Left edge voltage-difference
    dV_u = V_u - V_c # Right edge voltage-difference

    return dV_l, dV_r, dV_d, dV_u


def G_ye(y_lr_mn, y_ud_mn,
         grid_center, 
         delta_hor_edge, delta_ver_edge,
         delta_hor_edge_l, delta_hor_edge_r,
         delta_ver_edge_d, delta_ver_edge_u,
         l_e_ver_l, l_e_ver_r,
         l_e_hor_d, l_e_hor_u,
         ):

    y_l = y_lr_mn[ 0 : grid_center.M * 2 + 1, 0 : grid_center.N * 2 + 1 ] # Left-values of conductivity
    y_r = y_lr_mn[ 0 : grid_center.M * 2 + 1, 1 : grid_center.N * 2 + 2 ] # Right-values of conductivity
    
    y_d = y_ud_mn[ 0 : grid_center.M * 2 + 1, 0 : grid_center.N * 2 + 1 ] # Left-values of conductivity
    y_u = y_ud_mn[ 1 : grid_center.M * 2 + 2, 0 : grid_center.N * 2 + 1 ] # Right-values of conductivity

    g_l = jnp.where( y_l == 0,
                    0,
                    l_e_ver_l * ( 1 ) / ( 1 * delta_hor_edge_l + 1 * delta_hor_edge )
                   ) # Left edge conductivity

    
    g_r = jnp.where( y_r == 0,
                    0,
                    l_e_ver_r * ( 1 ) / ( 1 * delta_hor_edge_r + 1 * delta_hor_edge ) 
                   ) # Right edge conductivity
    
    g_d = jnp.where( y_d == 0,
                    0,
                    l_e_hor_d * ( 1 ) / ( 1 * delta_ver_edge_d + 1 * delta_ver_edge ) 
                   ) # Left edge conductivity
    
    g_u = jnp.where( y_u == 0,
                    0,
                    l_e_hor_u * ( 1 ) / ( 1 * delta_ver_edge_u + 1 * delta_ver_edge ) 
                   ) # Right edge conductivity
    
    return g_l, g_r, g_d, g_u


def G_ye_spe(y_lr_mn, y_ud_mn,
             grid_center, 
             delta_hor_edge, delta_ver_edge,
             delta_hor_edge_l, delta_hor_edge_r,
             delta_ver_edge_d, delta_ver_edge_u,
             l_e_ver_l, l_e_ver_r,
             l_e_hor_d, l_e_hor_u,
             ):

    y_l = y_lr_mn[ 0 : grid_center.M * 2 + 1, 0 : grid_center.N * 2 + 1 ] # Left-values of conductivity
    y_r = y_lr_mn[ 0 : grid_center.M * 2 + 1, 1 : grid_center.N * 2 + 2 ] # Right-values of conductivity
    
    y_d = y_ud_mn[ 0 : grid_center.M * 2 + 1, 0 : grid_center.N * 2 + 1 ] # Left-values of conductivity
    y_u = y_ud_mn[ 1 : grid_center.M * 2 + 2, 0 : grid_center.N * 2 + 1 ] # Right-values of conductivity

    g_l = l_e_ver_l * ( 1 ) / ( 1 * delta_hor_edge_l + 1 * delta_hor_edge ) * y_l # Left edge conductivity
    
    g_r = l_e_ver_r * ( 1 ) / ( 1 * delta_hor_edge_r + 1 * delta_hor_edge ) * y_r # Right edge conductivity
    
    g_d = l_e_hor_d * ( 1 ) / ( 1 * delta_ver_edge_d + 1 * delta_ver_edge ) * y_d # Left edge conductivity
    
    g_u = l_e_hor_u * ( 1 ) / ( 1 * delta_ver_edge_u + 1 * delta_ver_edge ) * y_u # Right edge conductivity
    
    return g_l, g_r, g_d, g_u


def G_ye_le_spe(y_lr_mn, y_ud_mn,
                grid_center, 
                delta_hor_edge, delta_ver_edge,
                delta_hor_edge_l, delta_hor_edge_r,
                delta_ver_edge_d, delta_ver_edge_u,
                l_e_ver_l, l_e_ver_r,
                l_e_hor_d, l_e_hor_u,
               ):

    y_l = y_lr_mn[ 0 : grid_center.M * 2 + 1, 0 : grid_center.N * 2 + 1 ] # Left-values of conductivity
    y_r = y_lr_mn[ 0 : grid_center.M * 2 + 1, 1 : grid_center.N * 2 + 2 ] # Right-values of conductivity
    
    y_d = y_ud_mn[ 0 : grid_center.M * 2 + 1, 0 : grid_center.N * 2 + 1 ] # Left-values of conductivity
    y_u = y_ud_mn[ 1 : grid_center.M * 2 + 2, 0 : grid_center.N * 2 + 1 ] # Right-values of conductivity

    g_l = 1 * ( 1 ) / ( 1 * delta_hor_edge_l + 1 * delta_hor_edge ) * y_l # Left edge conductivity
    
    g_r = 1 * ( 1 ) / ( 1 * delta_hor_edge_r + 1 * delta_hor_edge ) * y_r # Right edge conductivity
    
    g_d = 1 * ( 1 ) / ( 1 * delta_ver_edge_d + 1 * delta_ver_edge ) * y_d # Left edge conductivity
    
    g_u = 1 * ( 1 ) / ( 1 * delta_ver_edge_u + 1 * delta_ver_edge ) * y_u # Right edge conductivity
    
    return g_l, g_r, g_d, g_u


def sigma_from_y(ye_lr_m, ye_ud_m, grid_center,
                  ):

    # Build the G matrix
    ye_l = ye_lr_m[ 0 : grid_center.M * 2 + 1, 0 : grid_center.N * 2 + 1 ] # Left-values of conductivity
    ye_r = ye_lr_m[ 0 : grid_center.M * 2 + 1, 1 : grid_center.N * 2 + 2 ] # Right-values of conductivity
    
    ye_d = ye_ud_m[ 0 : grid_center.M * 2 + 1, 0 : grid_center.N * 2 + 1 ] # Left-values of conductivity
    ye_u = ye_ud_m[ 1 : grid_center.M * 2 + 2, 0 : grid_center.N * 2 + 1 ] # Right-values of conductivity
    
    sigma_m = jnp.where( ye_l + ye_r + ye_u + ye_d == 0, #(ye_l == 0) & (ye_r == 0) & (ye_u == 0) & (ye_d == 0),
                        0,
                        1
                       ) # Left edge conductivity
    
    return sigma_m


def count_non_binary(var):
    """
    Counts how many entries are NOT exactly 0 or 1.
    """
    return jnp.sum((var != 0) & (var != 1))


def discrete_grids(p_M,p_N,surf_winding):

    dt_2 = ( 2 * jnp.pi * ( 1 / (p_M * 2+1) ) ) * 1/2 # Half the width of a cell
    dz_2 = ( 2 * jnp.pi / surf_winding.NFP * ( 1 / (p_M * 2+1) ) ) * 1/2 # Half the height of a cell

    # Center-locations of the cells
    theta_center = jnp.linspace(dt_2, 2 * jnp.pi - dt_2, p_M * 2+1)
    zeta_center = jnp.linspace(dz_2, 2 * jnp.pi / surf_winding.NFP - dz_2, p_N * 2+1)
    
    # Create a grid for the center-locations
    grid_center = alt_grid(theta_center,zeta_center)
    grid_center_mod = LinearGrid(M = p_M, N = p_N, NFP = surf_winding.NFP)

    dt_2 = ( 2 * jnp.pi * ( 1 / (p_M * 2+1) ) ) * 1/2 # Half the width of a cell
    dz_2 = ( 2 * jnp.pi / surf_winding.NFP * ( 1 / (p_M * 2+1) ) ) * 1/2 # Half the height of a cell
    
    # Center-locations of the cells
    theta_center = jnp.linspace(dt_2, 2 * jnp.pi - dt_2, p_M * 2+1)
    zeta_center = jnp.linspace(dz_2, 2 * jnp.pi / surf_winding.NFP - dz_2, p_N * 2+1)
    
    # Create a grid for the center-locations
    grid_center = alt_grid(theta_center,zeta_center)

    # Shifted-locations of the cells
    theta_shift = jnp.linspace(0, 2 * jnp.pi, p_M * 2 + 2)
    zeta_shift = jnp.linspace(0, 2 * jnp.pi / surf_winding.NFP, p_N * 2 + 2)
    
    grid_theta = alt_grid(theta_shift,zeta_center)
    grid_zeta = alt_grid(theta_center,zeta_shift)

    data_center = surf_winding.compute(['theta','zeta', 'e_theta','e_zeta','e^theta_s','e^zeta_s',], grid = grid_center)
    data_center_mod = surf_winding.compute(['theta','zeta','|e_theta x e_zeta|',], grid = grid_center_mod)
    data_theta = surf_winding.compute(['theta','zeta', 'e_theta','e_zeta','e^theta_s','e^zeta_s',], grid = grid_theta)
    data_zeta = surf_winding.compute(['theta','zeta', 'e_theta','e_zeta','e^theta_s','e^zeta_s',], grid = grid_zeta)

    # Matrices with the length of the edges of each cell
    l_u_d = ( dot( data_theta['e_zeta'], data_theta['e_zeta'] ) ** (1/2) * ( 2 * dz_2 )
             ).reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 2).T
    
    l_l_r = ( dot(data_zeta['e_theta'], data_zeta['e_theta'] ) ** (1/2) * ( 2 * dt_2 )
             ).reshape(grid_center.N * 2 + 2, grid_center.M * 2 + 1).T
    
    l_l = l_l_r[:,0:-1]
    l_r = l_l_r[:,1:]
    l_d = l_u_d[0:-1,:]
    l_u = l_u_d[1:,:]

    # Matrices with the half length of each cell
    delta_ver_edge = ( dot( data_center['e_theta'], data_center['e_theta'] ) ** (1/2) * (1 * dt_2) 
                      ).reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 1).T

    delta_hor_edge = ( dot( data_center['e_zeta'], data_center['e_zeta'] ) ** (1/2) * (1 * dz_2) 
                      ).reshape(grid_center.N * 2 + 1, grid_center.M * 2 + 1).T
    
    delta_d = jnp.vstack(( delta_ver_edge[0,:]/1, delta_ver_edge[0:-1,:] 
                          ))

    delta_u = jnp.vstack(( delta_ver_edge[1:,:], delta_ver_edge[-1,:]/1 
                          ))

    delta_l = jnp.column_stack(( delta_hor_edge[:,0]/1, delta_hor_edge[:,0:-1] 
                               ))

    delta_r = jnp.column_stack(( delta_hor_edge[:,1:], delta_hor_edge[:,-1]/1 
                                ))

    #name = './'
    #data_center = iso_coords_interp2(name, data_center, grid_center, surf_winding)
    #data_theta = iso_coords_interp2(name, data_theta, grid_theta, surf_winding)
    #data_zeta = iso_coords_interp2(name, data_zeta, grid_zeta, surf_winding)

    return (grid_center, grid_theta, grid_zeta, 
            data_center, data_theta, data_zeta, 
            delta_hor_edge, delta_ver_edge,
            delta_l, delta_r, delta_d, delta_u,
            l_l, l_r, l_d, l_u,
            data_center_mod, grid_center_mod)


def alt_grid(theta,zeta):

    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta)
    theta_flat = theta_grid.flatten()
    zeta_flat = zeta_grid.flatten() 
    
    return Grid(nodes = jnp.stack( ( jnp.ones_like(theta_flat),
                                    theta_flat, zeta_flat ) 
                                  ).T,
                weights = jnp.ones_like(theta_flat),
                #jitable = True,
                NFP = 3,
                )


def add_extra_coords_old(data_, n_size,m_size,ind):

    _mod = data_.reshape( (n_size, m_size) ).T
    #_mod = jnp.vstack( [ _mod, _mod[0:m_size,0] ] )

    if ind == 0:
        _mod = jnp.column_stack( [_mod, _mod[:,0]] )
        _mod = jnp.vstack( [ _mod, 2 * jnp.pi * jnp.ones( _mod.shape[1] ) ] )

    if ind == 1:
        _mod = jnp.column_stack( [_mod, 2 * jnp.pi / 3 * jnp.ones_like(_mod[:,0])] )
        _mod = jnp.vstack( [ _mod, _mod[0,:] ] )
    
    return _mod

    
def add_extra_coords(data_, n_size,m_size):

    _mod = data_.reshape( (n_size, m_size) ).T
    
    _mod = jnp.column_stack( [_mod[:,0],_mod, _mod[:,-1]] )
    _mod = jnp.vstack( [_mod[0,:], 
                        _mod, 
                        _mod[-1,:] ] )

    return _mod