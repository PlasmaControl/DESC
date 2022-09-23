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
import netCDF4 as nc


#%%
# path = sys.argv[1]
# psi = sys.argv[2]
# alpha = sys.argv[3]

#path = '/home/pk123/DESC/examples/DESC/SOLOVEV_output.h5'
#path = '/home/pk123/DESC/docs/notebooks/tutorials/HELIOTRON_output.h5'
path = '/scratch/gpfs/pk2354/DESC/desc/examples/W7X_output.h5'
#path = '/scratch/gpfs/pk2354/DESC/desc/examples/DSHAPE_output.h5'
#path = '/scratch/gpfs/pk2354/DESC/desc/examples/ESTELL_output.h5'
#path = "/scratch/gpfs/pk2354/DESC/docs/notebooks/tutorials/qs_initial_guess.h5"
psi = 0.5
alpha = 0
npol = 1.0
nzgrid = 32


eq = desc.io.load(path)[-1]
#eq.change_resolution(M=4,L=4,M_grid=8,L_grid=8)
grid_1d = LinearGrid(L = 500, theta=0, zeta=0)
iota_data = eq.compute('iota', grid=grid_1d)
fi = interp1d(grid_1d.nodes[:,0],iota_data['iota'])

psib = eq.compute('psi')["psi"][-1]
if psib < 0:
    print("psib < 0")
    sgn = False
    psib = np.abs(psib)
else:
    sgn = True

rho = np.sqrt(psi)
iota = fi(rho)
print(psib)
zeta = np.linspace(-np.pi*npol/iota,np.pi*npol/iota,2*nzgrid+1)
thetas = alpha*np.ones(len(zeta)) + iota*zeta

rhoa = rho*np.ones(len(zeta))
c = np.vstack([rhoa,thetas,zeta]).T
coords = eq.compute_theta_coords(c)
#%%
#normalizations
grid = Grid(coords)
Lref = eq.compute('a')['a']
#Lref = 1.19836555357704
#psib = 0.3394774936
#Lref = 0.505870915155837
Bref = 2*psib/Lref**2
print("Bref is " + str(Bref))
print('psib is ' + str(psib))
#print(eq.compute('sqrt(g)',grid=grid)['sqrt(g)'])

#calculate bmag
modB = eq.compute('|B|',grid=grid)['|B|']
#print("modB is " + str(modB))
bmag = modB/Bref

#calculate gradpar and grho
gradpar  = Lref*eq.compute('B^zeta',grid=grid)['B^zeta']/modB
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
iota = iota_data['iota']
#grad_alpha_r = (lmbda_r - zeta*shear)*grad['|grad(rho)|'] 
#grad_alpha_t = (1 + lmbda_t)*grad['|grad(theta)|'] 
#grad_alpha_z = (-iota_data['iota']+lmbda_z)*grad['|grad(zeta)|']
grad_alpha_r = (lmbda_r - zeta*shear)
grad_alpha_t = (1 + lmbda_t)
grad_alpha_z = (-iota+lmbda_z)

#print("shear is" + str(shear))
#print("iota is " + str(iota_data['iota']))

grad_alpha = np.sqrt(grad_alpha_r**2 * grad['g^rr'] + grad_alpha_t**2 * grad['g^tt'] + grad_alpha_z**2 * grad['g^zz'] + 2*grad_alpha_r*grad_alpha_t*grad['g^rt'] + 2*grad_alpha_r*grad_alpha_z*grad['g^rz']
                     + 2*grad_alpha_t*grad_alpha_z*grad['g^tz'])

grad_psi_dot_grad_alpha = grad_psi * grad_alpha_r * grad['g^rr'] + grad_psi * grad_alpha_t * grad['g^rt'] + grad_psi * grad_alpha_z * grad['g^rz']

#calculate gds*
x = Lref * rho
shat = -x/iota_data['iota'][0] * shear[0]/Lref
gds2 = grad_alpha**2 * Lref**2 *psi
#gds21 with negative sign?
gds21 = shat/Bref * grad_psi_dot_grad_alpha
gds22 = (shat/(Lref*Bref))**2 /psi * grad_psi**2*grad['g^rr']

#calculate gbdrift0 and cvdrift0
B_t = eq.compute('B_theta',grid=grid)['B_theta']
B_z = eq.compute('B_zeta',grid=grid)['B_zeta']
dB_t = eq.compute('|B|_t',grid=grid)['|B|_t']
dB_z = eq.compute('|B|_z',grid=grid)['|B|_z']
jac = eq.compute('sqrt(g)',grid=grid)['sqrt(g)']
#gbdrift0 = (B_t*dB_z - B_z*dB_t)*2*rho*psib/jac
#gbdrift0 with negative sign?
gbdrift0 = shat * 2 / modB**3 / rho*(B_t*dB_z + B_z*dB_t)*psib/jac * 2 * rho
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
    #print("The old grid is " + str(zgrid))
    #print("The new grid is " + str(uniform_grid))

    for i in range(len(geo_array_gx)):
        #print("zeta old is " + str(zgrid[i]))
        #print("zeta new is " + str(uniform_grid[i]))
        
        geo_array_gx[i] = f(np.round(uniform_grid[i],5))
    
    return geo_array_gx

dzeta = zeta[1] - zeta[0]
dzeta_pi = np.pi / nzgrid
index_of_middle = nzgrid

gradpar_half_grid = np.zeros(2*nzgrid)
temp_grid = np.zeros(2*nzgrid+1)
z_on_theta_grid = np.zeros(2*nzgrid+1)
uniform_zgrid = np.zeros(2*nzgrid+1)

gradpar_temp = np.copy(gradpar)

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

bmag_gx = interp_to_new_grid(bmag,z_on_theta_grid,uniform_zgrid)
grho_gx = interp_to_new_grid(grho,z_on_theta_grid,uniform_zgrid)
gds2_gx = interp_to_new_grid(gds2,z_on_theta_grid,uniform_zgrid)
gds21_gx = interp_to_new_grid(gds21,z_on_theta_grid,uniform_zgrid)
gds22_gx = interp_to_new_grid(gds22,z_on_theta_grid,uniform_zgrid)
gbdrift_gx = interp_to_new_grid(gbdrift,z_on_theta_grid,uniform_zgrid)
gbdrift0_gx = interp_to_new_grid(gbdrift0,z_on_theta_grid,uniform_zgrid)
cvdrift_gx = interp_to_new_grid(cvdrift,z_on_theta_grid,uniform_zgrid)
cvdrift0_gx = interp_to_new_grid(cvdrift0,z_on_theta_grid,uniform_zgrid)
gradpar_gx = gradpar_temp

if sgn:
    gds21_gx = -gds21_gx
    gbdrift_gx = -gbdrift_gx
    gbdrift0_gx = -gbdrift0_gx
    cvdrift_gx = -cvdrift_gx
    cvdrift0_gx = -cvdrift0_gx

#%% Make gx input file
gxgridNA  = "gxinput2.out"

# ## GX geometric quantities
# zMax= self.zscale * self.pi
# zMin=-self.zscale * self.pi

# iotaN=self.iota-self.nNormal
# nz=self.ntgrid*2

# thetamax = self.tgridmax
# thetamin = -self.tgridmax

# # gradparGX = 2*pi*(2*iotaN*pi)/(Laxis*(iotaN*(phiMax - phiMin) + etabar*rVMEC*(-np.sin(iotaN*phiMax) + np.sin(iotaN*phiMin))))
# # z0 = pi - 2*pi*(iotaN*phiMax-rVMEC*etabar*np.sin(iotaN*phiMax))/(iotaN*(phiMax - phiMin) + etabar*rVMEC*(-np.sin(iotaN*phiMax) + np.sin(iotaN*phiMin)))
# self.gradparGX = self.zscale * 4*self.pi**2*iotaN*self.Aminor/(self.Laxis*(thetamax-thetamin-self.rVMEC*self.etabar*(np.sin(thetamax)-np.sin(thetamin))))
# self.z0 = self.zscale * self.pi*(thetamax + thetamin - self.rVMEC*self.etabar*(np.sin(thetamax)+np.sin(thetamin)))/((-thetamax+thetamin+self.rVMEC*self.etabar*(np.sin(thetamax)-np.sin(thetamin))))

# zGXgrid=np.linspace(zMin, zMax, 2*self.ntgrid+1)
# paramThetaGX = [self.thetaGXgrid(zz) for zz in zGXgrid]

#Output to GX grid (bottleneck, thing that takes longer to do, needs to be more pythy)

#ds = nc.Dataset('/scratch/gpfs/pk2354/GX/ESTELL_desc/linear/cyc_nl.nc')
#g = ds['/Geometry/gds2'][:]
#g = np.append(g,np.array([g[len(g)-1]]))
#gds2_gx = g
#print(len(gds2_gx))
#print(len(uniform_zgrid))


nperiod = 1
rmaj = eq.compute('R0')['R0']
kxfac = 1.0
open(gxgridNA, 'w').close()
f = open(gxgridNA, "w")
f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale")
f.write("\n"+str(nzgrid)+" "+str(nperiod)+" "+str(2*nzgrid)+" "+str(1.0)+" "+ str(1/Lref)+" "+str(shat)+" "+str(kxfac)+" "+str(1/iota) + " " + str(2*npol-1))

f.write("\ngbdrift gradpar grho tgrid")
for i in range(len(uniform_zgrid)):
    f.write("\n"+str(gbdrift_gx[i])+" "+str(gradpar_gx[i])+ " " + str(grho_gx[i]) + " " + str(uniform_zgrid[i]))
    
f.write("\ncvdrift gds2 bmag tgrid")
for i in range(len(uniform_zgrid)):
    f.write("\n"+str(cvdrift_gx[i])+" "+str(gds2_gx[i])+ " " + str(bmag_gx[i]) + " " + str(uniform_zgrid[i]))

f.write("\ngds21 gds22 tgrid")
for i in range(len(uniform_zgrid)):
    f.write("\n"+str(gds21_gx[i])+" "+str(gds22_gx[i])+  " " + str(uniform_zgrid[i]))

f.write("\ncvdrift0 gbdrift0 tgrid")
for i in range(len(uniform_zgrid)):
    f.write("\n"+str(-cvdrift0_gx[i])+" "+str(-gbdrift0_gx[i])+ " " + str(uniform_zgrid[i]))
    
f.close()
