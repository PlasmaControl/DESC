import numpy as np
import netCDF4 as nc

from desc.backend import use_jax, put, jnp
from desc.utils import Timer
from desc.grid import QuadratureGrid, ConcentricGrid, LinearGrid, Grid
from desc.transform import Transform
from desc.compute import (
    data_index,
    compute_force_error,
    compute_energy,
    compute_contravariant_current_density,
)
from .objective_funs import _Objective
#from desc.objectives.utils import factorize_linear_constraints, get_fixed_boundary_constraints
if use_jax:
    import jax
from desc.derivatives import Derivative

from scipy.interpolate import interp1d
from desc.backend import jnp
from desc.compute._core import (
    dot)

from scipy.constants import mu_0


class GX_Wrapper():

    def __init__(self,eq=None,npol=1,nzgrid=32,alpha=0,psi=0.5):
        self.eq = eq
        self.npol = npol
        self.nzgrid = nzgrid
        self.alpha = alpha
        self.psi = psi
   
    def calc_geo(self):
        grid_1d = LinearGrid(L = 500, theta=0, zeta=0)
        iota_data = self.eq.compute('iota', grid=grid_1d)
        fi = interp1d(grid_1d.nodes[:,0],iota_data['iota'])

        psib = self.eq.compute('psi')["psi"][-1]
        if psib < 0:
            sgn = False
            psib = np.abs(psib)
        else:
            sgn = True
        
        #get coordinate system
        rho = np.sqrt(self.psi)
        iota = fi(rho)
        zeta = np.linspace(-np.pi*self.npol/iota,np.pi*self.npol/iota,2*self.nzgrid+1)
        thetas = self.alpha*np.ones(len(zeta)) + iota*zeta

        rhoa = rho*np.ones(len(zeta))
        c = np.vstack([rhoa,thetas,zeta]).T
        coords = self.eq.compute_theta_coords(c)
        

        #normalizations
        grid = Grid(coords)
        Lref = self.eq.compute('a')['a']
        Bref = 2*psib/Lref**2

        #calculate bmag
        modB = self.eq.compute('|B|',grid=grid)['|B|']
        bmag = modB/Bref

        #calculate gradpar and grho
        gradpar  = Lref*self.eq.compute('B^zeta',grid=grid)['B^zeta']/modB
        grho = self.eq.compute('|grad(rho)|',grid=grid)['|grad(rho)|']*Lref

        #calculate grad_psi and grad_alpha
        grad = self.eq.compute('|grad(rho)|',grid=grid)
        # grad_psi = 2*psib*rho* grad['|grad(rho)|']
        grad_psi = 2*psib*rho

        lmbda = self.eq.compute('lambda',grid=grid)['lambda']
        lmbda_r = self.eq.compute('lambda_r',grid=grid)['lambda_r']
        lmbda_t = self.eq.compute('lambda_t',grid=grid)['lambda_t']
        lmbda_z = self.eq.compute('lambda_z',grid=grid)['lambda_z']
        iota_data = self.eq.compute('iota',grid=grid)
        shear = iota_data['iota_r']
        iota = iota_data['iota']

        grad_alpha_r = (lmbda_r - zeta*shear)
        grad_alpha_t = (1 + lmbda_t)
        grad_alpha_z = (-iota+lmbda_z)

        grad_alpha = np.sqrt(grad_alpha_r**2 * grad['g^rr'] + grad_alpha_t**2 * grad['g^tt'] + grad_alpha_z**2 * grad['g^zz'] + 2*grad_alpha_r*grad_alpha_t*grad['g^rt'] + 2*grad_alpha_r*grad_alpha_z*grad['g^rz']
                         + 2*grad_alpha_t*grad_alpha_z*grad['g^tz'])

        grad_psi_dot_grad_alpha = grad_psi * grad_alpha_r * grad['g^rr'] + grad_psi * grad_alpha_t * grad['g^rt'] + grad_psi * grad_alpha_z * grad['g^rz']

        #calculate gds*
        x = Lref * rho
        shat = -x/iota_data['iota'][0] * shear[0]/Lref
        gds2 = grad_alpha**2 * Lref**2 *self.psi
        #gds21 with negative sign?
        gds21 = shat/Bref * grad_psi_dot_grad_alpha
        gds22 = (shat/(Lref*Bref))**2 /self.psi * grad_psi**2*grad['g^rr']

        #calculate gbdrift0 and cvdrift0
        B_t = self.eq.compute('B_theta',grid=grid)['B_theta']
        B_z = self.eq.compute('B_zeta',grid=grid)['B_zeta']
        dB_t = self.eq.compute('|B|_t',grid=grid)['|B|_t']
        dB_z = self.eq.compute('|B|_z',grid=grid)['|B|_z']
        jac = self.eq.compute('sqrt(g)',grid=grid)['sqrt(g)']
        #gbdrift0 = (B_t*dB_z - B_z*dB_t)*2*rho*psib/jac
        #gbdrift0 with negative sign?
        gbdrift0 = shat * 2 / modB**3 / rho*(B_t*dB_z + B_z*dB_t)*psib/jac * 2 * rho
        cvdrift0 = gbdrift0
        #%%
        #calculate gbdrift and cvdrift
        B_r = self.eq.compute('B_rho',grid=grid)['B_rho']
        #dB_r = self.eq.compute('|B|_r',grid=grid)['|B|_r']

        data = self.eq.compute('|B|',grid=grid)
        data.update(self.eq.compute('B^zeta_r',grid=grid))
        data.update(self.eq.compute('B^theta_r',grid=grid))

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
        B = self.eq.compute('|B|',grid=grid)['|B|']
        p_r = self.eq.compute('p_r',grid=grid)['p_r']
        cvdrift = gbdrift + 2*Bref*Lref**2/modB**2 * rho*mu_0/B**2*p_r*Bsa

        self.Lref = Lref
        self.shat = shat
        self.iota = iota

        self.get_gx_arrays(zeta,bmag,grho,gradpar,gds2,gds21,gds22,gbdrift,gbdrift0,cvdrift,cvdrift0,sgn)

    def interp_to_new_grid(self,geo_array,zgrid,uniform_grid):
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

    def get_gx_arrays(self,zeta,bmag,grho,gradpar,gds2,gds21,gds22,gbdrift,gbdrift0,cvdrift,cvdrift0,sgn):
        dzeta = zeta[1] - zeta[0]
        dzeta_pi = np.pi / self.nzgrid
        index_of_middle = self.nzgrid

        gradpar_half_grid = np.zeros(2*self.nzgrid)
        temp_grid = np.zeros(2*self.nzgrid+1)
        z_on_theta_grid = np.zeros(2*self.nzgrid+1)
        self.uniform_zgrid = np.zeros(2*self.nzgrid+1)

        gradpar_temp = np.copy(gradpar)

        for i in range(2*self.nzgrid - 1):
            gradpar_half_grid[i] = 0.5*(np.abs(gradpar[i]) + np.abs(gradpar_temp[i+1]))    
        gradpar_half_grid[2*self.nzgrid - 1] = gradpar_half_grid[0]

        for i in range(2*self.nzgrid):
            temp_grid[i+1] = temp_grid[i] + dzeta * (1 / np.abs(gradpar_half_grid[i]))

        for i in range(2*self.nzgrid+1):
            z_on_theta_grid[i] = temp_grid[i] - temp_grid[index_of_middle]
        desired_gradpar = np.pi/np.abs(z_on_theta_grid[0])

        for i in range(2*self.nzgrid+1):
            z_on_theta_grid[i] = z_on_theta_grid[i] * desired_gradpar
            gradpar_temp[i] = desired_gradpar

        for i in range(2*self.nzgrid+1):
            self.uniform_zgrid[i] = z_on_theta_grid[0] + i*dzeta_pi

        final_theta_grid = self.uniform_zgrid
        
        self.bmag_gx = self.interp_to_new_grid(bmag,z_on_theta_grid,self.uniform_zgrid)
        self.grho_gx = self.interp_to_new_grid(grho,z_on_theta_grid,self.uniform_zgrid)
        self.gds2_gx = self.interp_to_new_grid(gds2,z_on_theta_grid,self.uniform_zgrid)
        self.gds21_gx = self.interp_to_new_grid(gds21,z_on_theta_grid,self.uniform_zgrid)
        self.gds22_gx = self.interp_to_new_grid(gds22,z_on_theta_grid,self.uniform_zgrid)
        self.gbdrift_gx = self.interp_to_new_grid(gbdrift,z_on_theta_grid,self.uniform_zgrid)
        self.gbdrift0_gx = self.interp_to_new_grid(gbdrift0,z_on_theta_grid,self.uniform_zgrid)
        self.cvdrift_gx = self.interp_to_new_grid(cvdrift,z_on_theta_grid,self.uniform_zgrid)
        self.cvdrift0_gx = self.interp_to_new_grid(cvdrift0,z_on_theta_grid,self.uniform_zgrid)
        self.gradpar_gx = gradpar_temp

        if sgn:
            self.gds21_gx = -self.gds21_gx
            self.gbdrift_gx = -self.gbdrift_gx
            self.gbdrift0_gx = -self.gbdrift0_gx
            self.cvdrift_gx = -self.cvdrift_gx
            self.cvdrift0_gx = -self.cvdrift0_gx


    def write_geo(self):
        self.calc_geo()
        nperiod = 1
        rmaj = self.eq.compute('R0')['R0']
        kxfac = 1.0
        open('gxinput_wrap.out', 'w').close()
        f = open('gxinput_wrap.out', "w")
        f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale")
        f.write("\n"+str(self.nzgrid)+" "+str(nperiod)+" "+str(2*self.nzgrid)+" "+str(1.0)+" "+ str(1/self.Lref)+" "+str(self.shat)+" "+str(kxfac)+" "+str(1/self.iota) + " " + str(2*self.npol-1))

        f.write("\ngbdrift gradpar grho tgrid")
        for i in range(len(self.uniform_zgrid)):
            f.write("\n"+str(self.gbdrift_gx[i])+" "+str(self.gradpar_gx[i])+ " " + str(self.grho_gx[i]) + " " + str(self.uniform_zgrid[i]))
            
        f.write("\ncvdrift gds2 bmag tgrid")
        for i in range(len(self.uniform_zgrid)):
            f.write("\n"+str(self.cvdrift_gx[i])+" "+str(self.gds2_gx[i])+ " " + str(self.bmag_gx[i]) + " " + str(self.uniform_zgrid[i]))

        f.write("\ngds21 gds22 tgrid")
        for i in range(len(self.uniform_zgrid)):
            f.write("\n"+str(self.gds21_gx[i])+" "+str(self.gds22_gx[i])+  " " + str(self.uniform_zgrid[i]))

        f.write("\ncvdrift0 gbdrift0 tgrid")
        for i in range(len(self.uniform_zgrid)):
            f.write("\n"+str(-self.cvdrift0_gx[i])+" "+str(-self.gbdrift0_gx[i])+ " " + str(self.uniform_zgrid[i]))
            
        f.close()
