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
#path = '/home/pk123/DESC/docs/notebooks/tutorials/HELIOTRON_output.h5'
path = '/home/pk123/DESC/examples/DESC/DSHAPE_output.h5'
psi = 0.5
alpha = 0
npol = 1
nzgrid = 32


eq = desc.io.load(path)[-1]
grid_1d = LinearGrid(L = 100, theta=0, zeta=0)
iota_data = eq.compute('iota', grid=grid_1d)
fi = interp1d(grid_1d.nodes[:,0],iota_data['iota'])

psib = eq.compute('psi')["psi"][-1]
rho = np.sqrt(psi)
iota = fi(rho)
zeta = np.linspace(-np.pi*npol,np.pi*npol,2*nzgrid+1)
thetas = alpha*np.ones(len(zeta)) + iota*zeta

rhoa = rho*np.ones(len(zeta))
c = np.vstack([rhoa,thetas,zeta]).T
coords = eq.compute_theta_coords(c)
#%%
#normalizations
grid = Grid(coords)
#Lref = eq.compute('a',grid=grid)['a']
Lref = 1.19836555357704
Bref = 2*psib/Lref**2

#calculate bmag
modB = eq.compute('|B|',grid=grid)['|B|']
bmag = modB/Bref

#calculate gradpar and grho
gradpar  = -Lref*eq.compute('B^zeta',grid=grid)['B^zeta']/modB
grho = eq.compute('|grad(rho)|',grid=grid)['|grad(rho)|']*Lref

#calculate grad_psi and grad_alpha
grad = eq.compute('|grad(rho)|',grid=grid)
# grad_psi = 2*psib*rho* grad['|grad(rho)|']
grad_psi = 2*psib*rho

lmbda = eq.compute('lambda',grid=grid)['lambda']
lmbda_r = eq.compute('lambda_r',grid=grid)['lambda_r']
lmbda_t = eq.compute('lambda_t',grid=grid)['lambda_t']
lmbda_z = eq.compute('lambda_z',grid=grid)['lambda_z']
iota_data = eq.compute('iota',grid=grid)
shear = iota_data['iota_r']

# grad_alpha_r = (lmbda_r - zeta*shear)*grad['|grad(rho)|'] 
# grad_alpha_t = (1 + lmbda_t)*grad['|grad(theta)|'] 
# grad_alpha_z = (-iota_data['iota']+lmbda_z)*grad['|grad(zeta)|']
grad_alpha_r = (lmbda_r - zeta*shear)
grad_alpha_t = (1 + lmbda_t)
grad_alpha_z = (-iota_data['iota']+lmbda_z)
grad_alpha = np.sqrt(grad_alpha_r**2 * grad['g^rr'] + grad_alpha_t**2 * grad['g^tt'] + grad_alpha_z**2 * grad['g^zz'] + 2*grad_alpha_r*grad_alpha_t*grad['g^rt'] + 2*grad_alpha_r*grad_alpha_z*grad['g^rz']
                     + 2*grad_alpha_t*grad_alpha_z*grad['g^tz'])

grad_psi_dot_grad_alpha = grad_psi * grad_alpha_r * grad['g^rr'] + grad_psi * grad_alpha_t * grad['g^rt'] + grad_psi * grad_alpha_z * grad['g^rz']

#calculate gds*
x = Lref * rho
shat = -x/iota_data['iota'][0] * shear[0]/Lref
gds2 = grad_alpha**2 * Lref**2 *psi
#gds21 with negative sign?
gds21 = -shat/Bref * grad_psi_dot_grad_alpha
gds22 = (shat/(Lref*Bref))**2 /psi * grad_psi**2*grad['g^rr']

#calculate gbdrift0 and cvdrift0
B_t = eq.compute('B_theta',grid=grid)['B_theta']
B_z = eq.compute('B_zeta',grid=grid)['B_zeta']
dB_t = eq.compute('|B|_t',grid=grid)['|B|_t']
dB_z = eq.compute('|B|_z',grid=grid)['|B|_z']
jac = eq.compute('sqrt(g)',grid=grid)['sqrt(g)']
#gbdrift0 = (B_t*dB_z - B_z*dB_t)*2*rho*psib/jac
#gbdrift0 with negative sign?
gbdrift0 = -shat * 2 / modB**3 / rho*(B_t*dB_z - B_z*dB_t)*psib/jac * 2 * rho
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
gbdrift_norm = 2*Bref*Lref**2/modB**3*rho
gbdrift = gbdrift_norm/jac*(B_r*dB_t*(lmbda_z - iota) + B_t*dB_z*(lmbda_r - zeta*shear[0]) + B_z*dB_r*(1+lmbda_t) - B_z*dB_t*(lmbda_r - zeta*shear[0]) - B_t*dB_r*(lmbda_z - iota) - B_r*dB_z*(1+lmbda_t))
Bsa = 1/jac * (B_z*(1+lmbda_t) - B_t*(lmbda_z - iota))
B = eq.compute('|B|',grid=grid)['|B|']
p_r = eq.compute('p_r',grid=grid)['p_r']
cvdrift = gbdrift + 2*Bref*Lref**2/modB**2 * rho*mu_0/B**2*p_r*Bsa

#%%Project onto GX grid

def interp_to_new_grid(geo_array,zgrid,uniform_grid):
    #l = 2*nzgrid + 1
    geo_array_gx = np.zeros(len(geo_array))
    
    f = interp1d(zgrid,geo_array,kind='cubic')
    
    for i in range(len(geo_array_gx)):
        geo_array_gx[i] = f(uniform_grid[i])
    
    return geo_array_gx

dzeta = zeta[1] - zeta[0]
dzeta_pi = np.pi / nzgrid
index_of_middle = nzgrid

gradpar_half_grid = np.zeros(2*nzgrid)
temp_grid = np.zeros(2*nzgrid+1)
z_on_theta_grid = np.zeros(2*nzgrid+1)
uniform_zgrid = np.zeros(2*nzgrid+1)

gradpar_temp = np.zeros(len(gradpar))

for i in range(2*nzgrid - 1):
    gradpar_half_grid[i] = 0.5*(np.abs(gradpar[i]) + np.abs(gradpar_temp[i+1]))    
gradpar_half_grid[2*nzgrid - 1] = gradpar_half_grid[0]

for i in range(2*nzgrid):
    temp_grid[i+1] = temp_grid[i] + dzeta * (1 / np.abs(gradpar_half_grid[i]))

for i in range(2*nzgrid+1):
    z_on_theta_grid[i] = temp_grid[i] - temp_grid[index_of_middle]
desired_gradpar = np.pi/np.abs(z_on_theta_grid[0])

for i in range(2*nzgrid+1):
    z_on_theta_grid[i] = z_on_theta_grid[i] * desired_gradpar
    gradpar_temp[i] = desired_gradpar

for i in range(2*nzgrid+1):
    uniform_zgrid[i] = z_on_theta_grid[0] + i*dzeta_pi

final_theta_grid = uniform_zgrid

bmag_gx = interp_to_new_grid(bmag,zeta,uniform_zgrid)
grho_gx = interp_to_new_grid(grho,zeta,uniform_zgrid)
gds2_gx = interp_to_new_grid(gds2,zeta,uniform_zgrid)
gds21_gx = interp_to_new_grid(gds21,zeta,uniform_zgrid)
gds22_gx = interp_to_new_grid(gds22,zeta,uniform_zgrid)
gbdrift_gx = interp_to_new_grid(gbdrift,zeta,uniform_zgrid)
gbdrift0_gx = interp_to_new_grid(gbdrift0,zeta,uniform_zgrid)
cvdrift_gx = interp_to_new_grid(cvdrift,zeta,uniform_zgrid)
cvdrift0_gx = interp_to_new_grid(cvdrift0,zeta,uniform_zgrid)

