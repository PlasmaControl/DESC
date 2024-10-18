import numpy as np
import os
# os.environ["JAX_LOG_COMPILES"] = "True"
from scipy.io import netcdf_file
import copy

import jax
import jax.numpy as jnpå
from jax import jit, jacfwd

from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, jnp, sign

from desc.utils import flatten_list

import numpy as np

from numpy import ndarray

@jax.jit
def first_derivative_t(a_mn,
                       data,
                       grid,):

    n_size = int(np.sqrt(data["Z"].shape[0]))
    
    # Rearrange A as a matrix
    A = a_mn.reshape((n_size, n_size)).T
    
    # theta-step
    dt = data["theta"][1] - data["theta"][0]
                                     
    # d(sigma)/dt
    A_t = jnp.zeros_like(A)
    
    # i = 0
    #(-3 * A[0, :] + 4 * A[1, :] - A[2, :]) * (2 * dt) ** (-1)
    A_t = A_t.at[0, :].set( (A[1, :] - A[n_size-1, :]) * (2 * dt) ** (-1)
                          )
    
    # i = n_size
    #(3 * A[n_size - 1, :] - 4 * A[n_size - 2, :] + A[n_size - 3, :]) * (2 * dt) ** (-1)
    A_t = A_t.at[n_size-1, :].set((A[0, :] - A[n_size-2, :]) * (2 * dt) ** (-1)
                                   )
    
    # Intermediate steps
    A_t = A_t.at[1:n_size - 1, :].set((A[2:n_size, :] - A[0:n_size - 2, :]) * (2 * dt) ** (-1))
    
    # Go back to the list-format used in DESC
    A_ta = ((A_t.T).reshape(-1))
    
    return A_ta

@jax.jit
def first_derivative_z(a_mn,
                       data,
                       grid,):

    n_size = int(np.sqrt(data["Z"].shape[0]))
    
    # Rearrange A as a matrix
    A = a_mn.reshape((n_size, n_size)).T
    
    # dz-step
    dz = data["zeta"][n_size] - data["zeta"][0]
    # d(V)/dz
    A_z = jnp.zeros_like(A)
    
    # at i = 0
    #(-3 * A[:, 0] + 4 * A[:, 1] - A[:, 2]) * (2 * dz) ** (-1)
    A_z = A_z.at[:, 0].set( (A[:, 1] - A[:, n_size - 1]) * (2 * dz) ** (-1)
                          )
    # at i = n_size
    #(3 * A[:, n_size - 1] - 4 * A[:, n_size - 2] + A[:, n_size - 3]) * (2 * dz) ** (-1)
    A_z = A_z.at[:, n_size - 1].set((A[:, 0] - A[:, n_size - 2]) * (2 * dz) ** (-1)
                                   )
    
    # Intermediate steps
    A_z = A_z.at[:, 1:n_size - 1].set((A[:, 2:n_size] - A[:, 0:n_size - 2]) * (2 * dz) ** (-1)
                                     )
    
    # Go back to the list-format used in DESC
    A_za = (A_z.T).reshape(-1)
    
    return A_za

@jax.jit
def second_derivative_t(a_mn,
                       data,
                       grid,):
    
    n_size = int(np.sqrt(data["Z"].shape[0]))
    
    # Rearrange A as a matrix
    A = a_mn.reshape((n_size, n_size))
    
    # theta-step
    dt = data["theta"][1] - data["theta"][0]
                                     
    # Second derivatives
    
    # d2(V)/dt2
    A_tt = jnp.zeros_like(A)
    
    # at i = 0
    A_tt = A_tt.at[0,:].set((2*A[0,:] - 5*A[1,:] + 4*A[2,:] - A[3,:])*(dt)**(-2)
                                )
    # at i = n_size
    A_tt = A_tt.at[n_size-1,:].set((2*A[n_size-1,:] - 5*A[n_size-1-1,:] + 4*A[n_size-1-2,:] - A[n_size-1-3,:])*(dt)**(-2)
                                       )
    # Intermediate steps
    ##
    #A_tt = A_t.at[1:n_size - 1, :].set((A[2:n_size, :] - A[0:n_size - 2, :]) * (2 * dt) ** (-1))
    
    #for i in range(1,n_size-1):
    A_tt = A_tt.at[1:n_size - 1, :].set((A[2:n_size, :] - 2*A[1:n_size - 1, :] + A[0:n_size - 2, :])*(dt)**(-2)
                                    )
        
    # Go back to the list-format used in DESC
    A_tta = A_tt.reshape(-1)
        
    return A_tta

@jax.jit
def second_derivative_z(a_mn,
                       data,
                       grid,):
    
    n_size = int(np.sqrt(data["Z"].shape[0]))
    
    # Rearrange A as a matrix
    # A_ij
    for i in range(0,n_size):
    
        if i == 0:
            A = a_mn[0:n_size]
        
        if i > 0:
            A = jnp.column_stack((A, a_mn[n_size*i:n_size*(i+1)]))
   
    # zeta-step
    dz = data["zeta"][n_size] - data["zeta"][0]
    
    # Second derivatives
    
    # d2(V)/dz2
    A_zz = jnp.zeros_like(A)
    # at i = 0
    A_zz = A_zz.at[:,0].set((2*A[:,0] - 5*A[:,1] + 4*A[:,2] - A[:,3])*(dz)**(-2)
                                )
    # at i = n_size
    A_zz = A_zz.at[:,n_size-1].set((2*A[:,n_size-1] - 5*A[:,n_size-1-1] + 4*A[:,n_size-1-2] - A[:,n_size-1-3])*(dz)**(-2)
                                       )
    
    # Intermediate steps
    #A_z = A_z.at[:, 1:n_size - 1].set((A[:, 2:n_size] - A[:, 0:n_size - 2]) * (2 * dz) ** (-1))
    #for i in range(1,n_size-1):
    A_zz = A_zz.at[:,i].set((A[:, 2:n_size] -2*A[:,1:n_size - 1] + A[:, 0:n_size - 2])*(dz)**(-2)
                                    )
        
    # Go back to the list-format used in DESC
    A_zza = A_zz.reshape(-1)

    return A_zza

@jax.jit
def second_derivative_tz(a_mn,
                         data,
                         grid,):
    
    n_size = int(np.sqrt(data["Z"].shape[0]))
    
    # Rearrange A as a matrix
    A = a_mn.reshape((n_size, n_size))
    
    # theta-step
    dt = data["theta"][1] - data["theta"][0]
    # dz-step
    dz = data["zeta"][n_size] - data["zeta"][0]
    
    # d(sigma)/dt
    A_t = jnp.zeros_like(A)
    # i = 0
    A_t = A_t.at[0, :].set((-3 * A[0, :] + 4 * A[1, :] - A[2, :]) * (2 * dt) ** (-1))
    
    # i = n_size
    A_t = A_t.at[n_size - 1, :].set((3 * A[n_size - 1, :] - 4 * A[n_size - 2, :] + A[n_size - 3, :]) * (2 * dt) ** (-1))
    
    # Intermediate steps
    A_t = A_t.at[1:n_size - 1, :].set((A[2:n_size, :] - A[0:n_size - 2, :]) * (2 * dt) ** (-1))
    
    # d2(V)/dtz
    A_tz = jnp.zeros_like(A)
    
    # at i = 0
    A_tz = A_tz.at[:, 0].set((-3 * A_t[:, 0] + 4 * A_t[:, 1] - A_t[:, 2]) * (2 * dz) ** (-1))
    
    # at i = n_size
    A_tz = A_tz.at[:, n_size - 1].set((3 * A_t[:, n_size - 1] - 4 * A_t[:, n_size - 2] + A_t[:, n_size - 3]) * (2 * dz) ** (-1))
    
    # Intermediate steps
    A_tz = A_tz.at[:, 1:n_size - 1].set((A_t[:, 2:n_size] - A_t[:, 0:n_size - 2]) * (2 * dz) ** (-1))
        
    # Go back to the list-format used in DESC
    A_tza = A_tz.reshape(-1)

    return A_tza