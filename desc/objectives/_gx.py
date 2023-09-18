
import numpy as np
import subprocess
from scipy.interpolate import interp1d
import os
import time
from desc.backend import jnp,put
from desc.compute import arg_order

from ._equilibrium import CurrentDensity
from .objective_funs import ObjectiveFunction, _Objective
from .utils import (
    align_jacobian,
    factorize_linear_constraints,
)
from desc.utils import Timer
from desc.compute import compute as compute_fun
from desc.compute import get_profiles, get_transforms, get_params
from desc.compute.utils import dot

from scipy.constants import mu_0, pi
from desc.grid import LinearGrid, Grid, ConcentricGrid, QuadratureGrid
from jax import core
from jax.interpreters import ad, batching
from desc.derivatives import FiniteDiffDerivative
import netCDF4 as nc
from shutil import copyfile

class GX(_Objective):
    r"""Calls the gyrokinetc code GX to compute the nonlinear
    turbulent heat flux.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    npol : int
        Number of poloidal turns (divided by 2pi) that the flux tube travels.
    nzgrid : int
        Number of grid points along the field line.
    alpha : float,
        Field line label.
    psi : float,
        Normalized toroidal flux on which to simulate.
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False
    _units = "Q"
    _print_value_fmt = "Total heat flux: {:10.3e} "
 
    def __init__(self, eq=None, target=0, weight=1, grid=None, name="GX", npol=1, nzgrid=32, alpha=0, psi=0.5, path='/home/pk2354/src/gx_latest',path_in='/scratch/gpfs/pk2354/DESC_GX_2/GX/gx_nl',path_geo='/scratch/gpfs/pk2354/DESC_GX_2/GX/gxinput_wrap',t=0,bounds=None,normalize=False,normalize_target=False):
        
        if target is None and bounds is None:
            target = 0
        self.eq = eq
        self.npol = npol
        self.nzgrid = nzgrid
        self.alpha = alpha
        self.psi = psi
        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

        units = "Q"
        self._callback_fmt = "Total heat flux: {:10.3e} " + units
        self._print_value_fmt = "Total heat flux: {:10.3e} " + units
        
        self.path = path
        self.path_in = path_in
        self.path_geo = path_geo
        self.t = t


    def build(self, eq, use_jit=False, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives
            (MUST be False to run GX).
        verbose : int, optional
            Level of output.

        """
        if self.grid is None:
            self.grid_eq = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
            
            #Construct flux-tube geometry
            data = eq.compute('iota')
            rhoa = eq.compute('rho')
            iotad = data['iota']
            fi = interp1d(rhoa['rho'],iotad)
            
            rho = np.sqrt(self.psi)
            iota = fi(rho)
            zeta = np.linspace((-np.pi*self.npol-self.alpha)/np.abs(iota),(np.pi*self.npol-self.alpha)/np.abs(iota),2*self.nzgrid+1)
            thetas = iota/np.abs(iota)*self.alpha*np.ones(len(zeta)) + iota*zeta
            
            zeta_center = zeta[self.nzgrid]
            rhoa = rho*np.ones(len(zeta))
            c = np.vstack([rhoa,thetas,zeta]).T
            coords = eq.compute_theta_coords(c,tol=1e-10,maxiter=50)
            self.grid = Grid(coords)

        self._dim_f = 1
        timer = Timer()

        self._data_eq_keys = [
            "iota",
            "iota_r",
            "a",
            "rho",
            "psi",
        ]
        self._data_keys = [
            "B", "|B|",
            "lambda", "lambda_r", "lambda_t", "lambda_z",
            "|grad(rho)|",
            "g^rr", "g^tt", "g^zz", "g^rt", "g^rz", "g^tz",
            "g_tz", "g_tt", "g_zz",
            "B_theta", "B_zeta", "B_rho", "|B|_t", "|B|_z",
            "B^theta", "B^zeta_r", "B^theta_r", "B^zeta",
            "e_theta", "e_theta_r", "e_zeta_r", "e_zeta",
            "p_r","grad(psi)",
        ]

#        self._args = get_params(self._data_keys,obj="desc.equilibrium.equilibrium.Equilibrium")
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=self.grid_eq.axis.size,
        )

        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        
        #Need separate transforms and profiles for the equilibrium and flux-tube
        self.eq = eq
        self._profiles = get_profiles(self._data_keys, obj=eq, grid=self.grid)
        self._profiles_eq = get_profiles(self._data_eq_keys, obj=eq, grid=self.grid_eq)
        self._transforms = get_transforms(self._data_keys, obj=eq, grid=self.grid)
        self._transforms_eq = get_transforms(self._data_eq_keys, obj=eq, grid=self.grid_eq)

        self._constants = {
            "transforms": self._transforms_eq,
            "profiles": self._profiles_eq,
        }


        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self.gx_compute = core.Primitive("gx")
        self.gx_compute.def_impl(self.compute_impl)
        ad.primitive_jvps[self.gx_compute] = self.compute_gx_jvp
        batching.primitive_batchers[self.gx_compute] = self.compute_gx_batch

#        self._check_dimensions()
#        self._set_dimensions(eq)
        super().build(eq=self.eq, use_jit=use_jit, verbose=verbose)
    
    def compute(self, *args, **kwargs):
        """Computes flux-tube geometric coefficients and calls GX.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile (Pa).
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile (A).
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).
        Returns
        -------
        Q : float
            The time-averaged nonlinear turbulent heat flux.

        """

        return self.gx_compute.bind(*args,**kwargs)

    def compute_impl(self, *args, **kwargs):

        params, constants = self._parse_args(*args, **kwargs)
        if constants is None:
            constants = self.constants
        rho = np.sqrt(self.psi)       
        data_eq = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_eq_keys,
            params=params,
            transforms=self._transforms_eq,
            profiles=self._profiles_eq,
        )
        #if data_eq['iota'][0] < 0:
        #    data_eq['iota'] = -data_eq['iota']
        #    data_eq['iota_r'] = -data_eq['iota_r']
        fi = interp1d(data_eq['rho'],data_eq['iota'])
        fs = interp1d(data_eq['rho'],data_eq['iota_r'])

        iotas = fi(np.sqrt(self.psi))
        shears = fs(np.sqrt(self.psi))

        
        zeta = np.linspace((-np.pi*self.npol-self.alpha)/np.abs(iotas),(np.pi*self.npol-self.alpha)/np.abs(iotas),2*self.nzgrid+1)
        iota = iotas * np.ones(len(zeta))
        shear = shears * np.ones(len(zeta))
        thetas = iotas/np.abs(iotas)*self.alpha*np.ones(len(zeta)) + iota*zeta
        zeta_center = zeta[self.nzgrid]

        rhoa = rho*np.ones(len(zeta))
        c = np.vstack([rhoa,thetas,zeta]).T
        coords = self.eq.compute_theta_coords(c,tol=1e-10,maxiter=50)
        th = coords[:,1]

        if self._profiles_eq["iota"] is None:
            self.grid = Grid(coords)
            self._transforms = get_transforms(self._data_keys, obj=self.eq, grid=self.grid)
            self._profiles = get_profiles(self._data_keys, obj=self.eq, grid=self.grid)
            data = {}
        
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=self._transforms,
            profiles=self._profiles,
        )

        psib = data_eq['psi'][-1]
        
        #normalizations       
        Lref = data_eq['a']
        Bref = 2*np.abs(psib)/Lref**2
        #calculate bmag
        modB = data['|B|']
        bmag = modB/Bref

        #calculate gradpar and grho
        gradpar = Lref*data['B^zeta']/modB
        grho = data['|grad(rho)|']*Lref

        #calculate grad_psi and grad_alpha
        grad_psi = 2*psib*rho
        lmbda = data['lambda']
        lmbda_r = data['lambda_r']
        lmbda_t = data['lambda_t']
        lmbda_z = data['lambda_z']
      
        

        grad_alpha_r = (lmbda_r - (zeta-zeta_center)*shear)
        grad_alpha_t = (1 + lmbda_t)
        grad_alpha_z = (-iota+lmbda_z)

        grad_alpha = np.sqrt(grad_alpha_r**2 * data['g^rr'] + grad_alpha_t**2 * data['g^tt'] + grad_alpha_z**2 * data['g^zz'] + 2*grad_alpha_r*grad_alpha_t*data['g^rt'] + 2*grad_alpha_r*grad_alpha_z*data['g^rz']
                         + 2*grad_alpha_t*grad_alpha_z*data['g^tz'])

        grad_psi_dot_grad_alpha = grad_psi * grad_alpha_r * data['g^rr'] + grad_psi * grad_alpha_t * data['g^rt'] + grad_psi * grad_alpha_z * data['g^rz']

        #calculate gds*
        x = Lref * rho
        shat = -x/iotas * shear[0]/Lref
        gds2 = grad_alpha**2 * Lref**2 *self.psi
        #gds21 with negative sign?

        gds21 = -iotas/np.abs(iotas) * shat/Bref * grad_psi_dot_grad_alpha
        gds22 = (shat/(Lref*Bref))**2 /self.psi * grad_psi**2*data['g^rr']

        #calculate gbdrift0 and cvdrift0
        B_t = data['B_theta']
        B_z = data['B_zeta']
        dB_t = data['|B|_t']
        dB_z = data['|B|_z']
        jac = data['sqrt(g)']
        #gbdrift0 = (B_t*dB_z - B_z*dB_t)*2*rho*psib/jac
        #gbdrift0 with negative sign?
        gbdrift0 = -iotas/np.abs(iotas) * -psib/np.abs(psib)*shat * 2 / modB**3 / rho*(B_t*dB_z - B_z*dB_t)*psib/jac * 2 * rho
        cvdrift0 = gbdrift0

        #calculate gbdrift and cvdrift
        B_r = data['B_rho'] 
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

        #iota = iota_data['iota'][0]
        gbdrift_norm = 2*Bref*Lref**2/modB**3*rho
        gbdrift = -psib/np.abs(psib) * gbdrift_norm/jac*(B_r*dB_t*(lmbda_z - iota) + B_t*dB_z*(lmbda_r - (zeta - zeta_center)*shear[0]) + B_z*dB_r*(1+lmbda_t) - B_z*dB_t*(lmbda_r - (zeta - zeta_center)*shear[0]) - B_t*dB_r*(lmbda_z - iota) - B_r*dB_z*(1+lmbda_t))
        Bsa = 1/jac * (B_z*(1+lmbda_t) - B_t*(lmbda_z - iota))
        p_r = data['p_r']
        cvdrift = gbdrift + 2*Bref*Lref**2/modB**2 * rho*mu_0/modB**2*p_r*Bsa

        self.Lref = Lref
        self.shat = shat
        self.iota = iota


        self.get_gx_arrays(zeta,bmag,grho,gradpar,gds2,gds21,gds22,gbdrift,gbdrift0,cvdrift,cvdrift0)
        t = str(self.t)
        path_geo_old = self.path_geo + '.out'
        path_in_old = self.path_in + '.in'
        path_geo_new = self.path_geo + '_' + t + '.out'
        path_in_new = self.path_in + '_' + t + '.in'
        self.write_geo(path_geo_new)
        self.write_input(path_in_old,path_geo_old,path_in_new,path_geo_new)
        #self.write_nc(t) 
                
        stdout = 'stdout.out_' + t
        stderr = 'stderr.out_' + t
        self.run_gx(t)

        out_file = self.path_in + '_' + t + '.nc'
        ds = nc.Dataset(out_file)

        
        qflux = ds['Fluxes/qflux']
        qflux = qflux[int(len(qflux)/2):]
        
        weighted_birkhoff = np.zeros(len(qflux))
        N = len(weighted_birkhoff)
        for i in range(1,N-1):
            weighted_birkhoff[i] = np.exp(-1/(i/N*(1-i/N)))
        norm = np.sum(weighted_birkhoff)
        
        weighted_birkhoff = weighted_birkhoff.reshape((len(qflux),1))
        qflux_avg = np.sum(weighted_birkhoff*qflux/norm)
        print(qflux_avg)
        ds.close() 

        
        return jnp.atleast_1d(qflux_avg)

    def compute_gx_jvp(self,values,tangents):
        
        R_lmn, Z_lmn, L_lmn, i_l, c_l, p_l, Psi = values
        primal_out = jnp.atleast_1d(0.0)

        n = len(values) 
        argnum = np.arange(0,n,1)
        
        jvp = FiniteDiffDerivative.compute_jvp(self.compute,argnum,tangents,*values,rel_step=1e-2)
        
        return (primal_out, jvp)

    def compute_gx_batch(self, values, axis):
        numdiff = len(values[0])
        res = jnp.array([0.0])

        for i in range(numdiff):
            R_lmn = values[0][i]
            Z_lmn = values[1][i]
            L_lmn = values[2][i]
            i_l = values[3][i]
            p_l = values[4][i]
            Psi = values[5][i]
            
            res = jnp.vstack([res,self.compute(R_lmn,Z_lmn,L_lmn,i_l,p_l,Psi)])

        res = res[1:]


        return res, axis[0]


    def interp_to_new_grid(self,geo_array,zgrid,uniform_grid):
        geo_array_gx = np.zeros(len(geo_array))
        f = interp1d(zgrid,geo_array,kind='cubic')
        for i in range(len(uniform_grid)-1):
            if uniform_grid[i] > zgrid[-1]:
                geo_array_gx[i] = geo_array_gx[i-1]
            else:
                geo_array_gx[i] = f(np.round(uniform_grid[i],5))
        
        geo_array_gx[-1] = geo_array[-1]

        return geo_array_gx

    def get_gx_arrays(self,zeta,bmag,grho,gradpar,gds2,gds21,gds22,gbdrift,gbdrift0,cvdrift,cvdrift0):
        dzeta = zeta[1] - zeta[0]
        dzeta_pi = np.pi / self.nzgrid
        index_of_middle = self.nzgrid

        gradpar_half_grid = np.zeros(2*self.nzgrid)
        temp_grid = np.zeros(2*self.nzgrid+1)
        z_on_theta_grid = np.zeros(2*self.nzgrid+1)
        self.uniform_zgrid = np.zeros(2*self.nzgrid+1)

        gradpar_temp = np.copy(gradpar) 
        for i in range(2*self.nzgrid - 1):
            gradpar_half_grid[i] = 0.5*(np.abs(gradpar_temp[i]) + np.abs(gradpar_temp[i+1]))    
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

    def write_nc(self,t):
        f = 'geo_' + str(t) + '.nc'
        ncfile = nc.Dataset(f,mode='w',format='NETCDF4_CLASSIC')
        geo_dim = ncfile.createDimension('dim', len(self.bmag_gx))
        shat_dim = ncfile.createDimension('shat_dim',1)

        bmag = ncfile.createVariable('bmag', np.float64, ('dim',))
        grho = ncfile.createVariable('grho', np.float64, ('dim',))
        gds2 = ncfile.createVariable('gds2', np.float64, ('dim',))
        gds21 = ncfile.createVariable('gds21', np.float64, ('dim',))
        gds22 = ncfile.createVariable('gds22', np.float64, ('dim',))
        gbdrift = ncfile.createVariable('gbdrift', np.float64, ('dim',))
        gbdrift0 = ncfile.createVariable('gbdrift0', np.float64, ('dim',))
        cvdrift = ncfile.createVariable('cvdrift', np.float64, ('dim',))
        cvdrift0 = ncfile.createVariable('cvdrift0', np.float64, ('dim',))
        gradpar = ncfile.createVariable('gradpar', np.float64, ('dim',))
        shat = ncfile.createVariable('shat',np.float64,('shat_dim',))

        bmag[:] = self.bmag_gx
        grho[:] = self.grho_gx
        gds2[:] = self.gds2_gx
        gds21[:] = self.gds21_gx
        gds22[:] = self.gds22_gx
        gbdrift[:] = self.gbdrift_gx
        gbdrift0[:] = self.gbdrift0_gx
        cvdrift[:] = self.cvdrift_gx
        cvdrift0[:] = self.cvdrift0_gx
        gradpar[:] = self.gradpar_gx
        shat[:] = self.shat


        ncfile.close()


    def write_geo(self,path_geo):
        nperiod = 1
        kxfac = 1.0
        f = open(path_geo, "w")
        f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale")
        f.write("\n"+str(self.nzgrid)+" "+str(nperiod)+" "+str(2*self.nzgrid)+" "+str(1.0)+" "+ str(1/self.Lref)+" "+str(self.shat)+" "+str(kxfac)+" "+str(1/self.iota[0]) + " " + str(2*self.npol-1))

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
            f.write("\n"+str(self.cvdrift0_gx[i])+" "+str(self.gbdrift0_gx[i])+ " " + str(self.uniform_zgrid[i]))
            
        f.close()

    def write_input(self,path_in_temp,geo_temp,path_in,geo):
        copyfile(path_in_temp,path_in)

        f = open(path_in,"r")
        data = f.read()
        
        data = data.replace(geo_temp,geo)
        f.close()

        f = open(path_in,"w")
        f.write(data)
        f.close()

    

    def run_gx(self,t):
        fs = open('stdout.out_' + str(t),'w')
        path = '/home/pk2354/gx_latest/'
        path_in = self.path_in + "_" + str(t) + '.in'
        cmd = ['srun', '-N', '1', '-t', '00:10:00', '--ntasks=1', '--gpus-per-task=1', '--exact','--overcommit',path+'./gx',path_in]
        subprocess.run(cmd,stdout=fs)
        fs.close()

