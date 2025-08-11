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
def first_derivative_t2(a_mn,
                       data,
                       grid,):

    n_size = int(np.sqrt(data["theta"].shape[0]))
    n_t = n_size
    # Rearrange A as a matrix
    A = a_mn.reshape((n_size, n_size)).T
    
    # theta-step
    dt = data["theta"][1] - data["theta"][0]
                                     
    # d(sigma)/dt
    A_t = jnp.zeros_like(A)
    
    # i = 0
    A_t = A_t.at[0,:].set( (-3*A[0,:] + 4*A[1,:] - A[2,:])*(2*dt)**(-1)
                         )
    
    # i = n_size
    A_t = A_t.at[n_t-1,:].set( (3*A[n_t-1,:] - 4*A[n_t-1-1,:] + A[n_t-1-2,:])*(2*dt)**(-1)
                             )
    
    # Intermediate steps
    A_t = A_t.at[1:n_size - 1, :].set((A[2:n_size, :] - A[0:n_size - 2, :]) * (2 * dt) ** (-1))
    
    # Go back to the list-format used in DESC
    A_ta = ((A_t.T).reshape(-1))
    
    return A_ta

@jax.jit
def first_derivative_z2(a_mn,
                       data,
                       grid,):

    n_size = int(np.sqrt(data["theta"].shape[0]))
    n_z = n_size
    # Rearrange A as a matrix
    A = a_mn.reshape((n_size, n_size)).T
    
    # dz-step
    dz = data["zeta"][n_size] - data["zeta"][0]
    # d(V)/dz
    A_z = jnp.zeros_like(A)
    
    # at i = 0
    A_z = A_z.at[:,0].set((-3*A[:,0] + 4*A[:,1] - A[:,2])*(2*dz)**(-1)
                         )
    
    # at i = n_size
    A_z = A_z.at[:,n_z-1].set((3*A[:,n_z-1] - 4*A[:,n_z-1-1] + A[:,n_z-1-2])*(2*dz)**(-1)
                                )
    
    # Intermediate steps
    A_z = A_z.at[:, 1:n_size - 1].set((A[:, 2:n_size] - A[:, 0:n_size - 2]) * (2 * dz) ** (-1)
                                     )
    
    # Go back to the list-format used in DESC
    A_za = (A_z.T).reshape(-1)
    
    return A_za