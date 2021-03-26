# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 00:52:18 2021

@author: Dario Panici
"""
import numpy as np
from netCDF4 import Dataset
from desc.grid import  LinearGrid
from desc.equilibrium import EquilibriaFamily


def test_Bpressure_Components(DSHAPE):
    """Tests that the components of the unscaled Bpressure (i.e. grad(|B|^2)) match with numerical gradients for DSHAPE example."""

    eq = EquilibriaFamily.load(
        load_from=str(DSHAPE["output_path"]), file_format="hdf5"
    )[-1]
    # numerically calc grad B^2

    ## vs rho
    L=100
    grid = LinearGrid(L=L)
    mag_P = eq.compute_magnetic_pressure_gradient(grid)
    B = eq.compute_magnetic_field(grid)
    
    mag_B_sq = B['|B|']**2
    
    d_B2_d_rho = np.ones_like(mag_B_sq)
    drho = grid.nodes[1,0] 
    
    d_B2_d_rho[0] = (mag_B_sq[1] - mag_B_sq[0]) / drho
    for i in range(1,L-1):
        d_B2_d_rho[i] = (mag_B_sq[i+1] - mag_B_sq[i-1]) / (2*drho)
    d_B2_d_rho[-1] = (mag_B_sq[-1] - mag_B_sq[-2]) / drho

    np.testing.assert_allclose(mag_P['Bpressure_rho'][2:-2], d_B2_d_rho[2:-2],rtol=1e-2)

    # vs theta
    M=240
    theta_grid = LinearGrid(M=M)
    grid = theta_grid
    B = eq.compute_magnetic_field(grid)
    mag_P = eq.compute_magnetic_pressure_gradient(grid)
    mag_B_sq = B['|B|']**2
    
    d_B2_d_t = np.ones_like(mag_B_sq)
    
    dt = grid.nodes[1,1] 
    
    d_B2_d_t[0] = (mag_B_sq[1] - mag_B_sq[-1]) / 2/dt
    for i in range(1,M-1):
        d_B2_d_t[i] = (mag_B_sq[i+1] - mag_B_sq[i-1]) / (2*dt)
    d_B2_d_t[-1] = (mag_B_sq[0] - mag_B_sq[-2]) / 2/dt
    
    np.testing.assert_allclose(mag_P['Bpressure_theta'][1:-1], d_B2_d_t[1:-1],rtol=1e-2)

