#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:38:14 2022

@author: pk123
"""
#matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
from desc.plotting import plot_1d, plot_2d, plot_3d, plot_section, plot_surfaces

surface = FourierRZToroidalSurface(R_lmn=[10, 1],
                                   modes_R=[[0, 0], [1,0]],
                                   Z_lmn=[0, 1],
                                   modes_Z=[[0,0], [-1, 0]],
                                  )

pressure = PowerSeriesProfile(params=[0, 0, 0]) # this is a constant pressure of 0
iota = PowerSeriesProfile(params=[1, 0, 1.5]) # iota = 1 + 1.5 r^2

eq = Equilibrium(surface=surface,
                 pressure=pressure,
                 iota=iota,
                 Psi=1.0, # flux (in Webers) within the last closed flux surface
                 NFP=1, # number of field periods
                 L=6, # radial spectral resolution
                 M=6, # poloidal spectral resolution
                 N=0, # toroidal spectral resolution (it's a tokamak so we don't need any toroidal modes)
                 L_grid=12, # real space radial resolution, slightly oversampled
                 M_grid=9, # real space poloidal resolution, slightly oversampled
                 N_grid=0, # real space toroidal resolution
                 sym=True, # explicitly enforce stellarator symmetry
                )

# plot_surfaces generates poincare plots of the flux surfaces
plot_surfaces(eq);

# plot_section plots various quantities on a toroidal cross section
# the second argument is a string telling it what to plot, in this case the force error density
# we could also look at B_z, (the toroidal magnetic field), or g (the coordinate jacobian) etc,
# here we also tell it to normalize the force error (relative to the magnetic pressure gradient or thermal pressure if present)
plot_section(eq,"|F|", norm_F=True, log=True);

eq.objective = "force"
eq.optimizer = "lsq-exact"
eq.solve(verbose=2, ftol=1e-8);
plot_surfaces(eq);
plot_section(eq,"|F|", norm_F=True, log=True);


#%%
delta_p = np.zeros_like(eq.p_l)
delta_p[0] = 1000.
delta_p[2] = -1000.
eq1 = eq.perturb(dp=delta_p, order=2)
plot_surfaces(eq1);
plot_section(eq1,"|F|", norm_F=True, log=True);

#%%
eq1.change_resolution(N=2,N_grid=3)
delta_R = np.zeros_like(eq1.Rb_lmn)
delta_Z = np.zeros_like(eq1.Zb_lmn)
delta_R[eq1.surface.R_basis.get_idx(M=1,N=1)] = -0.4
delta_Z[eq1.surface.Z_basis.get_idx(M=1,N=-1)] = -0.4

eq2 = eq1.perturb(dRb=delta_R, dZb=delta_Z,order=2)
plot_surfaces(eq2);
plot_section(eq2,"|F|", norm_F=True, log=True);
#%%
eq2.solve(verbose=2,ftol=1e-2);
plot_surfaces(eq2);
plot_section(eq2,"|F|", norm_F=True, log=True);