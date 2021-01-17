# -*- coding: utf-8 -*-
"""

Demonstrating using quadrature to get magnetic energy, and the problem with symmetry

Created on Sun Jan 17 00:16:07 2021

@author: Dario Panici
"""
from desc.equilibrium import EquilibriaFamily
from desc.plotting import Plot
from desc.grid import ConcentricGrid
from desc.transform import Transform
from scipy import special
import numpy as np

filename = "examples/DESC/DSHAPE.output" # just the output of the example DSHAPE run on the vmec_coords branch
# minimized with force on the cheb1 patterned grid
eq_fam = EquilibriaFamily(load_from=filename)
eq = eq_fam[-1]
plotter = Plot()
WBs = []
WB_syms = []


########## example with a non-symmetric grid, so we have theta = [0,2*pi) with equispaced nodes ############
# Energy is correct in this case
M = 15
Nr=M
Nr = M+1
N=0
grid = ConcentricGrid(M,N,1,surfs='quad',sym=False) # create quadrature grid with (M+1) radial nodes and 2*(M+1) angular nodes
B= eq.compute_magnetic_field(grid)
g_abs = np.abs(eq.compute_jacobian(grid)['g'])

mu0 = 4 * np.pi * 1e-7

mag_B_sq = B['|B|']**2

rho = grid.nodes[:,0]
theta = grid.nodes[:,1]
rs,weights = special.js_roots(Nr,2,2)

if N == 0:
    ang_fac = 1
else:
    ang_fac = 2

dim_angle = (2*Nr) **ang_fac # num_poloidal * num_toroidal nodes , num_toroidal nodes = 1 if Ntor=0
i=0
W_B = 0
for j in range(len(rho)):
    W_B += mag_B_sq[j] * weights[i] * (np.pi / Nr) ** ang_fac * g_abs[j] / rho[j]
    if j % (dim_angle) == 0 and j != 0:
        i += 1

if N==0:
    W_B = W_B * 2*np.pi
W_B = W_B /2 / mu0
WBs.append(W_B)
print("Magnetic Energy W_B = %e"%(W_B))


########## example with a symmetric grid, so we have theta = [0,pi) with equispaced nodes ############
# Energy is NOT correct in this case, as the quadrature formula assumes the integration is from 0 to 2pi, not 0 to pi
# simply multiplying by 2 would not solve issue

grid = ConcentricGrid(M,N,1,surfs='quad',sym=True) # create grid with (M+1) radial nodes and 2*(M+1) angular nodes
B= eq.compute_magnetic_field(grid)
g_abs = np.abs(eq.compute_jacobian(grid)['g'])

mu0 = 4 * np.pi * 1e-7

mag_B_sq = B['|B|']**2

rho = grid.nodes[:,0]
theta = grid.nodes[:,1]
rs,weights = special.js_roots(Nr,2,2)

if N == 0:
    ang_fac = 1
else:
    ang_fac = 2

dim_angle = (2*Nr) **ang_fac # num_poloidal * num_toroidal nodes , num_toroidal nodes = 1 if Ntor=0
i=0
W_B_sym = 0
for j in range(len(rho)):
    W_B_sym += mag_B_sq[j] * weights[i] * (np.pi / Nr) ** ang_fac * g_abs[j] / rho[j]
    if j % (dim_angle) == 0 and j != 0:
        i += 1

if N==0:
    W_B_sym = W_B_sym * 2*np.pi
W_B_sym = W_B_sym /2 / mu0
WB_syms.append(W_B_sym)
print("Magnetic Energy sym W_B = %e"%(W_B_sym))
print("Ratio Magnetic Energy W_B / W_B_sym = %e"%(W_B/W_B_sym)) 
# this ratio changes with M used for quadrature grid, this gives reason to believe
# quadrature used requires the function to be evaluated from 0 to 2pi, not just from 0 to pi
# Need to figure out how to tweak formula to deal with symmetry




