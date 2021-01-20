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

# filename = "examples/DESC/DSHAPE.output" # just the output of the example DSHAPE run on the vmec_coords branch
filename = "examples/DESC/HELIOTRON.output" # just the output of the example HELIOTRON run on the vmec_coords branch
# filename = "examples/DESC/SOLOVEV.output" # just the output of the example SOLOVEV run on the vmec_coords branch


# minimized with force on the cheb1 patterned grid
eq_fam = EquilibriaFamily(load_from=filename)
eq = eq_fam[-1]
plotter = Plot()
WBs = []
WB_syms = []


########## example with a non-symmetric grid, so we have theta = [0,2*pi) with equispaced nodes ############
# Energy is correct in this case
M = 12
Mrange = range(10,20,1)
for M in Mrange:
    Nr = M+1
    if eq.NFP == 1:
        N=0
    else:
        N=6
    grid = ConcentricGrid(M,N,eq.NFP,surfs='quad',sym=False) # create quadrature grid with (M+1) radial nodes and 2*(M+1) poloidal nodes, and 2*N+1 toroidal nodes
    W = eq.compute_energy(grid)
    WBs.append(W['W_B'])
    print('non sym grid W  = %e'%W['W'])
    print('non sym grid W_B = %e'%W['W_B'])
    
############## example with a symmetric grid, so we have theta = [0,pi) with equispaced nodes ############
    # Energy for DSHAPE case is not correct here, the magnetic energy is too low by ~2%
    
    grid = ConcentricGrid(M,N,eq.NFP,surfs='quad',sym=True) # create symmetric grid with (M+1) radial nodes and (M+1) angular nodes, and 2*N+1 toroidal nodes
    
    W_sym = eq.compute_energy(grid)
    WB_syms.append(W_sym['W_B'])
    print('sym grid W  = %e'%W_sym['W'])
    print('sym grid W_B = %e'%W_sym['W_B'])

    
    
    print("Ratio Pressure Energy W_p / W_p_sym = %e"%(W['W_p']/W_sym['W_p']))
    print("Ratio Magnetic Energy W_B / W_B_sym = %e"%(W['W_B']/W_sym['W_B'])) 
    # this ratio becomes closer to 1 with increasing M 

import matplotlib.pyplot as plt

plt.plot(Mrange,WBs,label='non sym')
plt.plot(Mrange,WB_syms,label='sym')
plt.legend()
plt.xlabel('M')
plt.figure()
plt.plot(Mrange,np.array(WBs) / np.array(WB_syms),label='WB non sym / WB sym')

plt.legend()





