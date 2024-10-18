import numpy as np
import os
# os.environ["JAX_LOG_COMPILES"] = "True"
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
                                  #biot_savart_general, 
                                  FourierCurrentPotentialField, ToroidalMagneticField,
                                  field_line_integrate)

import desc.examples

from desc.backend import fori_loop, jit, jnp, odeint, sign
from desc.compute import rpz2xyz, rpz2xyz_vec, xyz2rpz, xyz2rpz_vec
from desc.coils import *

from desc.utils import flatten_list
from desc.compute.utils import cross
from desc.compute.utils import dot

from desc.optimize import lsqtr, lsq_auglag

from scipy.optimize import NonlinearConstraint 

from desc.magnetic_fields import ( SplineMagneticField, 
                                  #biot_savart_general, 
                                  FourierCurrentPotentialField, ToroidalMagneticField,
                                  field_line_integrate)

import time

import numpy as np
from numpy.linalg import eig

from scipy.linalg import null_space
from numpy import ndarray


# Find normal component from solution to confirm it is actually zero
def phi_sv_eval(grid, field, basis):
    
    transform = Transform(grid, basis, derivs = 2)
    phi_trans = desc.transform.Transform(grid, basis, derivs=2, 
                                         rcond='auto', build=True, build_pinv=False, method='auto')
    
    
    phi_mn = field.Phi_mn

    fs = {#"phi": transform.transform(phi_mn),
          #"phi_t": transform.transform(phi_mn, dt = 1),
          #"phi_z": transform.transform(phi_mn, dz = 1),
          "phi_tt": transform.transform(phi_mn, dt = 2),
          "phi_tz": transform.transform(phi_mn, dt = 1, dz = 1),
          "phi_zz": transform.transform(phi_mn, dz = 2),
            }

    #fs["phi"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=0)
    
    #fs["phi_t"] = phi_trans.transform(phi_mn, dr=0, dt=1, dz=0)
    #fs["phi_z"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=1)
    
    fs["phi_tt"] = phi_trans.transform(phi_mn, dr=0, dt=2, dz=0)
    fs["phi_tz"] = phi_trans.transform(phi_mn, dr=0, dt=1, dz=1)
    fs["phi_zz"] = phi_trans.transform(phi_mn, dr=0, dt=0, dz=2)
                                   
    return fs