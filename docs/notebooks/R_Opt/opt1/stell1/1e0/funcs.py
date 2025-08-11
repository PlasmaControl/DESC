import numpy as np
import os

from scipy.io import netcdf_file
import copy
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.constants import mu_0
import sys
import functools
import pickle

import jax
import jax.numpy as jnpå
from jax import jit, jacfwd

from netCDF4 import Dataset
import h5py

from desc.backend import put, fori_loop, jnp, sign

from desc.basis import FourierZernikeBasis, DoubleFourierSeries, FourierSeries
#from desc.basis import DoubleChebyshevSeries

from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid
from desc.io import InputReader, load
from desc.objectives import *
from desc.objectives.objective_funs import _Objective
from desc.plotting import plot_1d, plot_2d, plot_3d, plot_section, plot_surfaces, plot_comparison

from desc.plotting import *

from desc.transform import Transform
from desc.vmec import VMECIO
from desc.derivatives import Derivative
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import SplineProfile, PowerSeriesProfile

from desc.magnetic_fields import ( SplineMagneticField, 
                                  FourierCurrentPotentialField, ToroidalMagneticField,
                                  field_line_integrate)

import desc.examples

from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.coils import *

from desc.utils import flatten_list
from desc.utils import cross
from desc.utils import dot

from desc.optimize import lsqtr, lsq_auglag

from scipy.optimize import NonlinearConstraint 

from desc.magnetic_fields import FourierCurrentPotentialField

import time

import numpy as np
from numpy.linalg import eig

from scipy.linalg import null_space
from numpy import ndarray

from desc.fns_simp import (data_eval, 
                           plot_figure,
                           surf_int,
                           _compute_magnetic_field_from_Current,)

#from green import green_eval 

from desc.finite_diff2 import (first_derivative_t, first_derivative_z,
                               # second_derivative_t, second_derivative_tz,
                               # second_derivative_z
                              )

from desc.finite_diff4 import (first_derivative_t2, first_derivative_z2,
                               # second_derivative_t, second_derivative_tz,
                               # second_derivative_z
                              )

# Define function to find A and its respective derivatives
def eqn_residual(grid, data, x_,y_,y,):

    b_t, b_z = v_1(grid, data, y,)
    
    return x_*b_t - y_*b_z

# Define function to find A and its respective derivatives
def dev_t(grid, data, y,):

    b_t, _ = v_1(grid, data, y,)
    
    return b_t

# Define function to find A and its respective derivatives
def dev_z(grid, data, y,):

    _, b_z = v_1(grid, data, y,)
    
    return b_z

# Define function to find A and its respective derivatives
def v_1(grid, data, y,):

    f_t = first_derivative_t(y,
                             data,
                             grid,)
    
    f_z = first_derivative_z(y,
                             data,
                             grid,)
    
    return f_t,f_z

# Define function to find A and its respective derivatives
def V_inv(grid, data, y,):

    ft,fz = v_2(grid, data, y,)
    
    return jnp.concatenate((ft,
                            fz)
                          )


# Define function to find A and its respective derivatives
def v_2(grid, data, y,):

    f_t = first_derivative_t2(y,
                             data,
                             grid,)
    
    f_z = first_derivative_z2(y,
                             data,
                             grid,)
    
    return f_t,f_z