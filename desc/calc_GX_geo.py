#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:24:39 2022

@author: pk123
"""
import sys
import desc.io
import numpy as np
from scipy.interpolate import interp1d
from desc.grid import LinearGrid, Grid
from desc.backend import jnp
from desc.compute._core import (
    dot)

from scipy.constants import mu_0



#%%
# path = sys.argv[1]
# psi = sys.argv[2]
# alpha = sys.argv[3]

#path = '/home/pk123/DESC/examples/DESC/SOLOVEV_output.h5'
path = '/home/pk123/DESC/docs/notebooks/tutorials/HELIOTRON_output.h5'
psi = 0.5
alpha = 0
npol = 1
ntheta = 64


eq = desc.io.load(path)[-1]
grid_1d = LinearGrid(L = 100, theta=0, zeta=0)
iota_data = eq.compute('iota', grid=grid_1d)
fi = interp1d(grid_1d.nodes[:,0],iota_data['iota'])

psib = eq.compute('psi')["psi"][-1]
rho = np.sqrt(psi)
iota = fi(rho)
zeta = np.linspace(-np.pi*npol,np.pi*npol,ntheta)
thetas = alpha*np.ones(len(zeta)) + iota*zeta

rhoa = rho*np.ones(len(zeta))
c = np.vstack([rhoa,thetas,zeta]).T
coords = eq.compute_theta_coords(c)
#%%
#normalizations
grid = Grid(coords)
Lref = eq.compute('a',grid=grid)['a']
Bref = 2*psib/Lref**2

#calculate bmag
bmag = eq.compute('|B|',grid=grid)['|B|']/Bref

#calculate gradpar
gradpar  = Lref*eq.compute('B^zeta',grid=grid)['B^zeta']/Bref

#calculate grad_psi and grad_alpha
grad = eq.compute('|grad(rho)|',grid=grid)
grad_psi = 2*psib*rho* grad['|grad(rho)|']

lmbda = eq.compute('lambda',grid=grid)['lambda']
lmbda_r = eq.compute('lambda_r',grid=grid)['lambda_r']
lmbda_t = eq.compute('lambda_t',grid=grid)['lambda_t']
lmbda_z = eq.compute('lambda_z',grid=grid)['lambda_z']
iota_data = eq.compute('iota',grid=grid)
shear = iota_data['iota_r']

grad_alpha_r = (lmbda_r - zeta*shear)*grad['|grad(rho)|'] 
grad_alpha_t = (1 + lmbda_t)*grad['|grad(theta)|'] 
grad_alpha_z = (-iota_data['iota']+lmbda_z)*grad['|grad(zeta)|']
grad_alpha = np.sqrt(grad_alpha_r**2 * grad['g^rr'] + grad_alpha_t**2 * grad['g^tt'] + grad_alpha_z**2 * grad['g^zz'] + 2*grad_alpha_r*grad_alpha_t*grad['g^rt'] + 2*grad_alpha_r*grad_alpha_z*grad['g^rz']
                     + 2*grad_alpha_t*grad_alpha_z*grad['g^tz'])

grad_psi_dot_grad_alpha = grad_psi * grad_alpha_r * grad['g^rr'] + grad_psi * grad_alpha_t * grad['g^rt'] + grad_psi * grad_alpha_z * grad['g^rz']

#calculate gds*
x = Lref * rho
shat = -x/iota_data['iota'][0] * shear[0]
gds2 = grad_alpha**2 * Lref**2 *psi/psib
gds21 = shat/Bref * grad_psi_dot_grad_alpha
gds22 = (shat/(Lref*Bref))**2 * psib/psi * grad_psi**2*grad['g^rr']

#calculate gbdrift0 and cvdrift0
B_t = eq.compute('B_theta',grid=grid)['B_theta']
B_z = eq.compute('B_zeta',grid=grid)['B_zeta']
dB_t = eq.compute('|B|_t',grid=grid)['|B|_t']
dB_z = eq.compute('|B|_z',grid=grid)['|B|_z']
jac = eq.compute('sqrt(g)',grid=grid)['sqrt(g)']
gbdrift0 = (B_t*dB_z - B_z*dB_t)*2*rho*psib/jac
cvdrift0 = gbdrift0
#%%
#calculate gbdrift and cvdrift
B_r = eq.compute('B_rho',grid=grid)['B_rho']
#dB_r = eq.compute('|B|_r',grid=grid)['|B|_r']

data = eq.compute('|B|',grid=grid)
data.update(eq.compute('B^zeta_r',grid=grid))
data.update(eq.compute('B^theta_r',grid=grid))

data["|B|_r"] = (
    data["B^theta"]
    * (
        data["B^zeta_r"] * data["g_tz"]
        + data["B^theta_r"] * data["g_tt"]
        + data["B^theta"] * dot(data["e_theta_r"], data["e_theta"])
    )
    + data["B^zeta"]
    * (
        data["B^theta_r"] * data["g_tz"]
        + data["B^zeta_r"] * data["g_zz"]
        + data["B^zeta"] * dot(data["e_zeta_r"], data["e_zeta"])
    )
    + data["B^theta"]
    * data["B^zeta"]
    * (
        dot(data["e_theta_r"], data["e_zeta"])
        + dot(data["e_zeta_r"], data["e_theta"])
    )
) / data["|B|"]

dB_r = data['|B|_r']

iota = iota_data['iota'][0]
gbdrift = 1/jac*(B_r*dB_t*(lmbda_z - iota) + B_t*dB_z*(lmbda_r - zeta*shear[0]) + B_z*dB_r*(1+lmbda_t) - B_z*dB_t*(lmbda_r - zeta*shear[0]) - B_t*dB_r*(lmbda_z - iota) - B_r*dB_z*(1+lmbda_t))
Bsa = 1/jac * (B_z*(1+lmbda_t) - B_t*(lmbda_z - iota))
B = eq.compute('|B|',grid=grid)['|B|']
p_r = eq.compute('p_r',grid=grid)['p_r']
cvdrift = gbdrift + 2*Bref*Lref**2/B**2 * np.sqrt(psi/psib)*mu_0/B**2*p_r*Bsa
