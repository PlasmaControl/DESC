import numpy as np
import subprocess
from scipy.interpolate import interp1d
import os
import time
from desc.backend import jnp,put
from desc.compute import arg_order
from .utils import (
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
    factorize_linear_constraints,
)

from desc.utils import Timer
from desc.transform import Transform
from .objective_funs import ObjectiveFunction, _Objective
from ._equilibrium import CurrentDensity

from desc.compute._core import (
    compute_rotational_transform,
    compute_toroidal_flux,    
    compute_geometry,
    compute_contravariant_metric_coefficients,
    compute_lambda,
    dot,
    compute_pressure,
    data_index,
    compute_flux_coords
)

from desc.compute._field import (
    compute_magnetic_field_magnitude,        
    compute_covariant_magnetic_field
)

from scipy.constants import mu_0, pi

from desc.grid import LinearGrid, Grid, ConcentricGrid, QuadratureGrid
from jax import core
from jax.interpreters import ad, batching
from desc.derivatives import FiniteDiffDerivative
import netCDF4 as nc

from shutil import copyfile
#from desc.configuration import compute_theta_coords

class WrappedEquilibriumObjective(ObjectiveFunction):
    """Evaluate an objective subject to equilibrium constraint.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    eq_objective : ObjectiveFunction
        Equilibrium objective to enforce.
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the objectives.
    use_jit : bool, optional
        Whether to just-in-time compile the objectives and derivatives.
    verbose : int, optional
        Level of output.

    """

    def __init__(
        self,
        objective,
        eq_objective=None,
        linear_objective=None,
        eq=None,
        use_jit=False,
        verbose=1,
        perturb_options={},
        solve_options={},
    ):

        self._objective = objective
        self._eq_objective = eq_objective
        self._linear_objective = linear_objective
        self._perturb_options = perturb_options
        self._solve_options = solve_options
        self._use_jit = use_jit
        self._built = False
        self._compiled = True

        if eq is not None:
            self.build(eq, use_jit=self._use_jit, verbose=verbose)

    # TODO: add timing and verbose statements
    def build(self, eq, use_jit=False, verbose=1):
        """Build the objective.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        self._eq = eq.copy()
        if self._eq_objective is None:
            self._eq_objective = get_equilibrium_objective()
        self._constraints = get_fixed_boundary_constraints(
            profiles=not isinstance(self._eq_objective.objectives[0], CurrentDensity),
            iota=eq.iota is not None,
        )

        if self._linear_objective is None:
            self._linear_objective = ObjectiveFunction(self._constraints)


        self._objective.build(self._eq, use_jit=self.use_jit, verbose=verbose)
        self._eq_objective.build(self._eq, use_jit=self.use_jit, verbose=verbose)
        self._linear_objective.build(self._eq, use_jit=self.use_jit, verbose=verbose)
        for constraint in self._constraints:
            constraint.build(self._eq, use_jit=self.use_jit, verbose=verbose)
        self._objectives = self._objective.objectives

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        # set_state_vector
        self._args = ["p_l", "i_l", "c_l", "Psi", "Rb_lmn", "Zb_lmn"]
        if isinstance(self._eq_objective.objectives[0], CurrentDensity):
            self._args.remove("p_l")
        self._dimensions = self._objective.dimensions
        self._dim_x = 0
        self._x_idx = {}
        for arg in self.args:
            self.x_idx[arg] = np.arange(self._dim_x, self._dim_x + self.dimensions[arg])
            self._dim_x += self.dimensions[arg]

        self._full_args = np.concatenate((self.args, self._eq_objective.args))
        self._full_args = [arg for arg in arg_order if arg in self._full_args]

        (
            xp,
            A,
            self._Ainv,
            b,
            self._Z,
            self._unfixed_idx,
            project,
            recover,
        ) = factorize_linear_constraints(
            self._constraints, extra_args=self._eq_objective.args
        )

        (
            xpl,
            Al,
            self._Ainvl,
            bl,
            self._Zl,
            self._unfixed_idxl,
            self._projectl,
            self._recoverl,
        ) = factorize_linear_constraints(
            self._linear_objective.objectives, extra_args=self._args
        )
        

        self._x_old = np.zeros((self._dim_x,))
        for arg in self.args:
            self._x_old[self.x_idx[arg]] = getattr(eq, arg)

        self.history = {}
        for arg in self._full_args:
            self.history[arg] = [np.atleast_1d(getattr(self._eq, arg))]

        self._built = True

    def _update_equilibrium(self, x):
        """Update the internal equilibrium with new boundary, profile etc."""
        if jnp.allclose(x, self._x_old, rtol=1e-14, atol=1e-14):
            pass
        else:
            x_dict = self.unpack_state(x)
            x_dict_old = self.unpack_state(self._x_old)
            deltas = {
                "d" + str(key).split("_")[0]: x_dict[key] - x_dict_old[key]
                for key in x_dict
            }
            self._eq = self._eq.perturb(
                objective=self._eq_objective,
                constraints=self._constraints,
                **deltas,
                **self._perturb_options
            )
            self._eq.solve(
                objective=self._eq_objective,
                constraints=self._constraints,
                **self._solve_options
            )
            self._x_old = x
            for arg in self._full_args:
                self.history[arg].append(getattr(self._eq, arg))

    def compute(self, x):
        """Compute the objective function.

        Parameters
        ----------
        x : ndarray
            State vector.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        self._update_equilibrium(x)
        x_obj = self._objective.x(self._eq)
        return self._objective.compute(x_obj)

    def grad(self, x):

        f = jnp.atleast_1d(self.compute(x))
        J = self.jac(x)
        return f.T @ J

    def jac(self, x):

        self._update_equilibrium(x)
        
        
        # dx/dc
        x_idx = np.concatenate(
            [
                self._eq_objective.x_idx[arg]
                for arg in ["p_l", "i_l", "c_l", "Psi"]
                if arg in self._eq_objective.args
            ]
        )
        x_idx.sort(kind="mergesort")
        dxdc = np.eye(self._eq_objective.dim_x)[:, x_idx]
        dxdRb = (
            np.eye(self._eq_objective.dim_x)[:, self._eq_objective.x_idx["R_lmn"]]
            @ self._Ainv["R_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdRb))
        dxdZb = (
            np.eye(self._eq_objective.dim_x)[:, self._eq_objective.x_idx["Z_lmn"]]
            @ self._Ainv["Z_lmn"]
        )
        dxdc = np.hstack((dxdc, dxdZb))
        
                 
        # state vectors
        xf = self._eq_objective.x(self._eq)
        xg = self._objective.x(self._eq)

        # Jacobian matrices wrt combined state vectors
        Fx = self._eq_objective.jac(xf)
#        Gx = self._objective.jac(xg)
        Fx = {
            arg: Fx[:, self._eq_objective.x_idx[arg]] for arg in self._eq_objective.args
        }
#        Gx = {arg: Gx[:, self._objective.x_idx[arg]] for arg in self._objective.args}
        for arg in self._eq_objective.args:
            if arg not in Fx.keys():
                Fx[arg] = jnp.zeros((self._eq_objective.dim_f, self.dimensions[arg]))
#            if arg not in Gx.keys():
#                Gx[arg] = jnp.zeros((self._objective.dim_f, self.dimensions[arg]))
        Fx = jnp.hstack([Fx[arg] for arg in arg_order if arg in Fx])
#        Gx = jnp.hstack([Gx[arg] for arg in arg_order if arg in Gx])
        print("The shape of Fx is " + str(Fx.shape))
        print("The shape of Z is " + str(self._Z.shape))
        print("unfixed idx is " + str(self._unfixed_idx.shape))
        Fx_reduced = Fx[:, self._unfixed_idx] @ self._Z
#        Gx_reduced = Gx[:, self._unfixed_idx] @ self._Z

        Fc = Fx @ dxdc
        Fx_reduced_inv = jnp.linalg.pinv(Fx_reduced, rcond=1e-6)

#        Gc = Gx @ dxdc
#        GxFx = Gx_reduced @ Fx_reduced_inv

#        LHS  = GxFx @ Fc - Gc
        print("The shape of dxdc is " + str(dxdc.shape))
        print("unfixed idxl is " + str(self._unfixed_idxl))
        tang = jnp.array([])
        for i in range(len(dxdc)):
            ti = self._projectl(dxdc[i,:])
            if np.abs(ti[0]) > 0.00001:
                print("ti is " + str(ti))
            tang = jnp.hstack([tang,ti])
        tang = jnp.atleast_2d(0.001*tang).T
        #print("tangents are " + str(tang))
       

        print("computing fx")
        fx = self._objective.compute(xg)
        args = (fx,)
        Gc = jnp.array([])
        for i in range(len(tang[0])):
            print("tangent is " + str(tang[:,i]))
            Gc = jnp.hstack([Gc,self._objective.jvp(tang[:,i],xg)])
        
        print("Gc new proj is " + str(Gc))
        Gc = self._recoverl(Gc)
        print("Gc new is " + str(Gc))
        
        #Gc = jnp.array([])
        #for i in range(len(dxdc[0])):
        #    Gc = jnp.hstack([Gc,self._objective.jvp(dxdc[:,i],xg)])
        #print("Gc old is " + str(Gc))
        #print("Gc old proj is " + str(self._projectl(Gc)))


        I = jnp.eye(len(xg),len(self._Z))
        #print("The shape of I is " + str(I.shape))
        #print("The shape of Z is " + str(self._Z.shape))
        #print("The shape of Fx_reduced_inv is " + str(Fx_reduced_inv.shape))
        #print("The shape of Fc is " + str(Fc.shape))
        #print("The shape of Fx is " + str(Fx.shape))
        t = I @ self._Z @ Fx_reduced_inv @ Fc
        print("The shape of t is " + str(t.shape))
        
        num_singular_values = 9

        tnorm = jnp.linalg.norm(t,axis=0)
        #print("t is " + str(t))
        #print("tnorm is " + str(tnorm))
        #tnp = np.array(t)
        #tnorm_np = np.array(tnorm)
        #nonzero_col = tnorm_np > 0
        #tnp[:,nonzero_col] /= tnorm_np[nonzero_col]
        #t = jnp.array(tnp)
        
        #t = jnp.asarray(t).at([:,nonzero_col]).set(t[:,nonzero_col]/tnorm[nonzero_col])
        

        tU, tS, tV = jnp.linalg.svd(t)
        #print("The singular values are of t are " + str(tS))
        tu1 = tU[:,:num_singular_values+1]
        tv1 = tV[:num_singular_values+1,:]
        ts1 = tS[:num_singular_values+1]

        #print("The shape of tu1 is " + str(tu1.shape))
        #print("The shape of tv1 is " + str(tv1.shape))
        print("ts1 is " + str(ts1))
        GxFxFc = jnp.array([])
        gx_eval = 0
        print("tu1 is " + str(tu1))
        for i in range(len(tu1[0])):
            GxFxFc = jnp.hstack([GxFxFc,self._objective.jvp(tu1[:,i],xg)])
            gx_eval = gx_eval + 1
        GxFxFc = GxFxFc @ jnp.diag(ts1) @ tv1
        print("GxFxFc proj is " + str(self._projectl(GxFxFc)))
        #GxFxFc = self._objective.jvp(tu1.T,xg)*ts1*tv1

        #GxFxFc = jnp.array([])
        #for i in range(len(t[0])):
        #    GxFxFc = jnp.hstack([GxFxFc,self._objective.jvp(t[:,i],xg)])
        #GxFxFc = self._objective.jvp(t,xg)

#        num_singular_values = dxdc.shape[1]
#        dxdc_norm = jnp.linalg.norm(dxdc,axis=0)
#        dxdc_np = np.array(dxdc)
#        dxdc_norm_np = np.array(dxdc_norm)
#        nonzero_col = dxdc_norm_np > 0
#        dxdc_np[:,nonzero_col] /= dxdc_norm_np[nonzero_col]
#        dxdc = np.array(dxdc_np)
#
#        dxdc_U, dxdc_S, dxdc_V = jnp.linalg.svd(dxdc)
#        #print("The singular values of dxdc are " + str(dxdc_S))
#        dxdc_u1 = dxdc_U[:,:num_singular_values]
#        dxdc_v1 = dxdc_V[:num_singular_values,:]
#        dxdc_s1 = dxdc_S[:num_singular_values]
#        Gc = jnp.array([])
#        for i in range(len(dxdc_u1[0])):
#            Gc = jnp.hstack([Gc,self._objective.jvp(dxdc_u1[:,i],xg)])
#            gx_eval = gx_eval + 1
#        Gc = Gc @ jnp.diag(dxdc_s1) @ dxdc_v1
        #Gc = self._objective.jvp(dxdc_u1,xg)*dxdc_s1*dxdc_v1
        


        #Gc = jnp.array([])
        #for i in range(len(dxdc[0])):
        #    Gc = jnp.hstack([Gc,self._objective.jvp(dxdc[:,i],xg)])
        #Gc = self._objective.jvp(dxdc,xg)

        
        #print("Gc is " + str(Gc))
        print("GxFxFc is " + str(GxFxFc))
        LHS = jnp.atleast_2d(GxFxFc - Gc)
        #print("The shape of LHS is " + str(LHS))
        #LHS = jnp.atleast_2d(-Gc)
        return -LHS

    def hess(self, x):

        J = self.jac(x)
        return J.T @ J



class GXWrapper(_Objective):

    _scalar = True
    _linear = False
    
    
    def __init__(self, eq=None, target=0, weight=1, grid=None, name="GK", npol=1, nzgrid=32, alpha=0, psi=0.5, equality=True, lb=False):
        self.eq = eq
        self.npol = npol
        self.nzgrid = nzgrid
        self.alpha = alpha
        self.psi = psi

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "Q"
        self._callback_fmt = "Total heat flux: {:10.3e} " + units
        self.lb = lb
        self.equality = equality
        self._print_value_fmt = "Total heat flux: {:10.3e} " + units


    def build(self, eq, use_jit=False, verbose=1):
        if self.grid is None:
            self.grid_eq = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                #sym=eq.sym,
                #axis=False,
                #rotation="sin",
                #node_pattern=eq.node_pattern,
            )

            #grid_1d = LinearGrid(L = 500, theta=0, zeta=0)
            #data = eq.compute('iota',grid=grid_1d)
            data = eq.compute('iota')
            rhoa = eq.compute('rho')
            iotad = np.abs(data['iota'])
            fi = interp1d(rhoa['rho'],iotad)
            
            #get coordinate system
            rho = np.sqrt(self.psi)
            iota = fi(rho)
            #print("IOTA IS " + str(iota))
            zeta = np.linspace(-np.pi*self.npol/iota,np.pi*self.npol/iota,2*self.nzgrid+1)
            thetas = self.alpha*np.ones(len(zeta)) + iota*zeta

            rhoa = rho*np.ones(len(zeta))
            c = np.vstack([rhoa,thetas,zeta]).T
            coords = eq.compute_theta_coords(c)
            self.grid = Grid(coords)

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        
        self.eq = eq

        self._pressure = eq.pressure.copy()
        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota_eq = eq.iota.copy()
            self._iota.grid = self.grid
            self._iota_eq.grid = self.grid_eq
        else:
            self._iota = None
            self._iota_eq = None
        if eq.current is not None:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._current_eq = eq.current.copy()
            self._current_eq.grid = self.grid_eq
        else:
            self._current = None
            self._current_eq = None

        self._pressure.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=3, build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=3, build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=3, build=True
        )

        self._R_transform_eq = Transform(
            self.grid_eq, eq.R_basis, derivs=3, build=True
        )
        self._Z_transform_eq = Transform(
            self.grid_eq, eq.Z_basis, derivs=3, build=True
        )
        self._L_transform_eq = Transform(
            self.grid_eq, eq.L_basis, derivs=3, build=True
        )


        
        self._args = ["R_lmn","Z_lmn","L_lmn","i_l","c_l", "p_l","Psi"]
        
        self.gx_compute = core.Primitive("gx")
        self.gx_compute.def_impl(self.compute_impl)
        ad.primitive_jvps[self.gx_compute] = self.compute_gx_jvp
        batching.primitive_batchers[self.gx_compute] = self.compute_gx_batch

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        #self._set_derivatives(use_jit=use_jit)
        self._built = True
    
    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, p_l, Psi):
        #print("At the beginning of compute: " + str(R_lmn[0]) + " " +str(Z_lmn[0]) + " " + str(L_lmn[0]))
        args = (R_lmn, Z_lmn, L_lmn, i_l, c_l, p_l, Psi)
        #self.gx_compute.bind(R_lmn,Z_lmn,L_lmn,i_l,p_l,Psi)
        return self.gx_compute.bind(*args)

    def compute_impl(self, *args):
        (R_lmn, Z_lmn, L_lmn, i_l, c_l, p_l, Psi) = args
        rho = np.sqrt(self.psi)
 
        if self._iota is None:
            data_eq =compute_rotational_transform(R_lmn,Z_lmn,L_lmn,i_l,c_l,Psi,self._R_transform_eq,self._Z_transform_eq,self._L_transform_eq,self._iota_eq,self._current_eq)
            data_eq = compute_flux_coords(grid=self.grid_eq,data=data_eq)
            fi = interp1d(data_eq['rho'],data_eq['iota'])
            fs = interp1d(data_eq['rho'],data_eq['iota_r'])

            iotas = fi(np.sqrt(self.psi))
            shears = fs(np.sqrt(self.psi))
            #if iotas < 0:
            #    iotas = np.abs(iotas)
            #    shears = -shears

            zeta = np.linspace(-np.pi*self.npol/np.abs(iotas),np.pi*self.npol/np.abs(iotas),2*self.nzgrid+1)
            iota = iotas * np.ones(len(zeta))
            shear = shears * np.ones(len(zeta))
            thetas = self.alpha*np.ones(len(zeta)) + iota*zeta

            rhoa = rho*np.ones(len(zeta))
            c = np.vstack([rhoa,thetas,zeta]).T
            coords = self.eq.compute_theta_coords(c,L_lmn=L_lmn)
            self.grid = Grid(coords)
            self._R_transform = Transform(
                self.grid, self.eq.R_basis, derivs=3, build=True
            )
            self._Z_transform = Transform(
                self.grid, self.eq.Z_basis, derivs=3, build=True
            )
            self._L_transform = Transform(
                self.grid, self.eq.L_basis, derivs=3, build=True
            )
            data = {}
        else:
            data = compute_rotational_transform(R_lmn,Z_lmn,L_lmn,i_l,c_l,Psi,self._R_transform,self._Z_transform,self._L_transform,self._iota,self._current)

            iota= np.abs(data['iota'])
            iotas = iota[0]
            if data['iota'][0] < 0:
                shear = -data['iota_r']
            else:
                shear = data['iota_r']
            data_eq = {}
            zeta = np.linspace(-np.pi*self.npol/iotas,np.pi*self.npol/iotas,2*self.nzgrid+1)


        #print("iota is " + str(iota))
        
        if self._iota is not None:
            data = compute_toroidal_flux(Psi,self._iota.grid,data=data)
        else:
            data = compute_toroidal_flux(Psi,self._current.grid,data=data)
        psib = data['psi'][-1]*2
        if psib < 0:
            sgn = False
            psib = np.abs(psib)
        else:
            sgn = True
        
        
        #normalizations
        #grid = Grid(coords)
        data_eq = compute_geometry(R_lmn,Z_lmn,self._R_transform_eq,self._Z_transform_eq,data=data_eq)
        Lref = data_eq['a']
        Bref = 2*psib/Lref**2
        #print('psib is ' + str(psib))
        #print("Bref is " + str(Bref))

        #calculate bmag
        data = compute_magnetic_field_magnitude(R_lmn,Z_lmn,L_lmn,i_l,c_l,Psi,self._R_transform,self._Z_transform,self._L_transform,self._iota,self._current,data=data)
        modB = data['|B|']
        #print("modB is " + str(modB))
        bmag = modB/Bref

        #calculate gradpar and grho
        gradpar  = Lref*data['B^zeta']/modB
        data = compute_contravariant_metric_coefficients(R_lmn,Z_lmn,self._R_transform,self._Z_transform,data=data)
        grho = data['|grad(rho)|']*Lref
        #print("gradpar is " + str(gradpar))

        #calculate grad_psi and grad_alpha
        grad_psi = 2*psib*rho
        data = compute_lambda(L_lmn,self._L_transform,data=data)

        lmbda = data['lambda']
        lmbda_r = data['lambda_r']
        lmbda_t = data['lambda_t']
        lmbda_z = data['lambda_z']
        #iota_data = self.eq.compute('iota')
       
        grad_alpha_r = (lmbda_r - zeta*shear)
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
        gds21 = shat/Bref * grad_psi_dot_grad_alpha
        gds22 = (shat/(Lref*Bref))**2 /self.psi * grad_psi**2*data['g^rr']

        #calculate gbdrift0 and cvdrift0
        data = compute_covariant_magnetic_field(R_lmn,Z_lmn,L_lmn,i_l,c_l,Psi,self._R_transform,self._Z_transform,self._L_transform,self._iota,self._current,data=data)
        B_t = data['B_theta']
        B_z = data['B_zeta']
        dB_t = data['|B|_t']
        dB_z = data['|B|_z']
        jac = data['sqrt(g)']
        #gbdrift0 = (B_t*dB_z - B_z*dB_t)*2*rho*psib/jac
        #gbdrift0 with negative sign?
        gbdrift0 = shat * 2 / modB**3 / rho*(B_t*dB_z + B_z*dB_t)*psib/jac * 2 * rho
        cvdrift0 = gbdrift0

        #calculate gbdrift and cvdrift
        B_r = data['B_rho']
        #dB_r = self.eq.compute('|B|_r',grid=grid)['|B|_r']

        #data = self.eq.compute('|B|',grid=grid)
        #data.update(self.eq.compute('B^zeta_r',grid=grid))
        #data.update(self.eq.compute('B^theta_r',grid=grid))

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
        gbdrift = gbdrift_norm/jac*(B_r*dB_t*(lmbda_z - iota) + B_t*dB_z*(lmbda_r - zeta*shear[0]) + B_z*dB_r*(1+lmbda_t) - B_z*dB_t*(lmbda_r - zeta*shear[0]) - B_t*dB_r*(lmbda_z - iota) - B_r*dB_z*(1+lmbda_t))
        Bsa = 1/jac * (B_z*(1+lmbda_t) - B_t*(lmbda_z - iota))
        data = compute_pressure(p_l,self._pressure,data=data)
        p_r = data['p_r']
        cvdrift = gbdrift + 2*Bref*Lref**2/modB**2 * rho*mu_0/modB**2*p_r*Bsa

        self.Lref = Lref
        self.shat = shat
        self.iota = iota


        self.get_gx_arrays(zeta,bmag,grho,gradpar,gds2,gds21,gds22,gbdrift,gbdrift0,cvdrift,cvdrift0,sgn)
        self.write_geo()
        self.run_gx()

        ds = nc.Dataset('/scratch/gpfs/pk2354/DESC/GX/gx_nl.nc')
        #ds = nc.Dataset('/scratch/gpfs/pk2354/DESC/GX/gx.nc')
        
        qflux = ds['Fluxes/qflux']
        qflux_avg = jnp.mean(qflux[int(len(qflux)/2):]) 
        print(qflux_avg)
        
        #om = ds['Special/omega_v_time'][len(ds['Special/omega_v_time'])-1][:]
        #gamma = np.zeros(len(om))
        #for i in range(len(gamma)):
        #    gamma[i] = om[i][0][1]
        #print(max(gamma))     
        

        os.rename('/scratch/gpfs/pk2354/DESC/GX/gx_nl.nc','/scratch/gpfs/pk2354/DESC/GX/gx_old.nc')
        os.rename('/scratch/gpfs/pk2354/DESC/GX/gxinput_wrap.out','/scratch/gpfs/pk2354/DESC/GX/gxinput_wrap_old.out')

        
        #print(gamma)
        #return self._shift_scale(jnp.atleast_1d(max(gamma)))
        return self._shift_scale(jnp.atleast_1d(qflux_avg))
    
    def compute_gx_jvp(self,values,tangents):
        
        R_lmn, Z_lmn, L_lmn, i_l, c_l, p_l, Psi = values
        primal_out = jnp.atleast_1d(0.0)

        n = len(values) 
        argnum = np.arange(0,n,1)
        
        jvp = FiniteDiffDerivative.compute_jvp(self.compute,argnum,tangents,*values,rel_step=1e-2)
        
        return (primal_out, jvp)

    def compute_gx_batch(self, values, axis):
        print("AT BATCH!!!")
        print("VALUES IS " + str(values))
        
        numdiff = len(values[0])
        print("NUMDIFF IS " + str(numdiff))
        
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




    def compute_gx_batch_old(self,values,axis):
        print("AT BATCH!!!")
        if not np.iscalar(axis):
            raise Exception('axis should be a scalar.')
        res = jnp.array([])
        if axis == 0:
            for i in range(len(values)):
                R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi = values[i,:]
                res = jnp.vstack([res,self.compute(r_lmn, z_lmn, l_lmn, i_l, p_l, psi)])

        else:
            for i in range(len(values[0])):
                R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi = values[:,i]
                res = jnp.vstack([res,self.compute(r_lmn, z_lmn, l_lmn, i_l, p_l, psi)])

        return res, axis

    def compute_gx_batch_old2(self, values, axis):
        print("AT BATCH!!!")
        print("axis is " + str(axis))
        ind = 0
        for i in range(len(axis)):
            if axis[i] != None:
                ind = i

        for i in range(len(values)):
            if i == ind:
                continue
            elif i == 0:
                R_lmn = values[i]
            elif i == 1:
                Z_lmn = values[i]
            elif i == 2:
                L_lmn = values[i]
            elif i == 3:
                i_l = values[i]
            elif i == 4:
                p_l = values[i]
            else:
                Psi = values[i]
        res = jnp.array([0.0])

        for i in range(len(values[ind])):
            if ind == 0:
                R_lmn = values[ind][i]
            elif ind == 1:
                Z_lmn = values[ind][i]
            elif ind == 2:
                L_lmn = values[ind][i]
            elif ind == 3:
                i_l = values[ind][i]
            elif ind == 4:
                p_l = values[ind][i]
            else:
                Psi = values[ind][i]
            res = jnp.vstack([res,self.compute(R_lmn,Z_lmn,L_lmn,i_l,p_l,Psi)])

        res = res[1:]
        #print("res is " + str(res))

        return res, axis[ind]


    def interp_to_new_grid(self,geo_array,zgrid,uniform_grid):
        geo_array_gx = np.zeros(len(geo_array))
        
        f = interp1d(zgrid,geo_array,kind='cubic')

        for i in range(len(geo_array_gx)):
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
        nperiod = 1
        #rmaj = self.eq.compute('R0')['R0']
        kxfac = 1.0
        #print("At write geo: " + str(self.gbdrift_gx[0]) + str(self.gds2_gx[0]) + str(self.bmag_gx[0]))
        #open('gxinput_wrap.out', 'w').close()
        fname = '/scratch/gpfs/pk2354/DESC/GX/gxinput_wrap.out'
        f = open(fname, "w")
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
            f.write("\n"+str(-self.gds21_gx[i])+" "+str(self.gds22_gx[i])+  " " + str(self.uniform_zgrid[i]))

        f.write("\ncvdrift0 gbdrift0 tgrid")
        for i in range(len(self.uniform_zgrid)):
            f.write("\n"+str(self.cvdrift0_gx[i])+" "+str(self.gbdrift0_gx[i])+ " " + str(self.uniform_zgrid[i]))
            
        f.close()

    def write_input(self,t):
        fname = '/scratch/gpfs/pk2354/DESC/GX/gx_nl.in'
        fname_new = '/scratch/gpfs/pk2354/DESC/GX/gx_nl_' + t + '.in'

        geo = 'gxinput_wrap_old.out'
        geo_new = 'gxinput_wrap_old_' + t + '.out'
        copyfile(fname,fname_new)

        f = open(fname_new,"r")
        data = f.read()

        data = data.replace(geo,geo_new)
        f.close()

        f = open(fname_new,"r")
        f.write(data)

    
    

    def run_gx(self):
        fs = open('stdout.out','w')
        path = '/home/pk2354/src/gx/'
        path_in = '/scratch/gpfs/pk2354/DESC/GX/gx_nl.in'
        cmd = ['srun', '-N', '1', '-t', '00:10:00', '--ntasks=1', '--gpus-per-task=1', path+'./gx',path_in]
        #cmd = [path+'./gx','/scratch/gpfs/pk2354/DESC/GX/gx_nl.in']
        #process = []
        #print(cmd)
        p = subprocess.run(cmd,stdout=fs)
        #p.wait()


