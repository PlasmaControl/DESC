import numpy as np
import os

# os.environ["JAX_LOG_COMPILES"] = "True"
import pickle
import copy
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.constants import mu_0
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
from desc import set_device

set_device("gpu")


import jax
import jax.numpy as jnp
from jax import jit, jacfwd

from netCDF4 import Dataset
import h5py


from desc.backend import put
from desc.basis import FourierZernikeBasis, DoubleFourierSeries, FourierSeries
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.grid import ConcentricGrid, LinearGrid, Grid, QuadratureGrid
from desc.io import InputReader, load
from desc.plotting import (
    plot_1d,
    plot_2d,
    plot_3d,
    plot_section,
    plot_surfaces,
    plot_comparison,
)
from desc.transform import Transform
from desc.vmec import VMECIO
from desc.derivatives import Derivative
from desc.profiles import SplineProfile
from desc.magnetic_fields import SplineMagneticField

from desc.utils import flatten_list

os.getcwd()


from desc.objectives.objective_funs import _Objective
from desc.compute import (
    compute_contravariant_basis,
    compute_contravariant_magnetic_field,
    compute_pressure,
    compute_magnetic_field_magnitude,
    compute_covariant_magnetic_field    
)
from desc.geometry.utils import rpz2xyz, rpz2xyz_vec, xyz2rpz_vec, rotation_matrix
from desc.nestor import Nestor



def compute_I(R_lmn, Z_lmn, L_lmn, i_l, Psi, R_transform, Z_transform, L_transform, i_profile, orientation):
    data = compute_covariant_magnetic_field(R_lmn, Z_lmn, L_lmn, i_l, Psi, R_transform, Z_transform, L_transform, i_profile, orientation)
    w = R_transform.grid.weights
    return jnp.sum(data['B_theta']*w)/(mu_0*2*np.pi)

def bsq_vac(R_lmn, Z_lmn, L_lmn, i_l, Psi, R_transform, Z_transform, L_transform, i_profile, orientation, nest):
    
    ctor = compute_I(R_lmn, Z_lmn, L_lmn, i_l, Psi, R_transform, Z_transform, L_transform, i_profile, orientation)
    out = nest.compute(R_lmn, Z_lmn, ctor)
    bsq = out[1]['|B|^2']
    w = nest._Rb_transform.grid.weights
    return (bsq*w/mu_0).flatten()

def bsq_plasma(R_lmn, Z_lmn, L_lmn, i_l, Psi, R_transform, Z_transform, L_transform, i_profile, orientation):
    
    data = compute_magnetic_field_magnitude(R_lmn, Z_lmn, L_lmn, i_l, Psi, R_transform, Z_transform, L_transform, i_profile, orientation)
    bsq = data['|B|']**2
    g = data['|e_theta x e_zeta|']
    w = R_transform.grid.weights
    return (bsq*g*w/2/mu_0)



class BoundaryErrorNESTOR(_Objective):
    _scalar = False
    _linear = False

    def __init__(self,         
                 ext_field,
                 eq=None,
                 target=0,
                 weight=1,
                 mf=None,
                 nf=None,
                 ntheta=None,
                 nzeta=None,
                 name="NESTOR Boundary",
): 
        
        self.mf = mf
        self.nf = nf
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.ext_field = ext_field
        self._callback_fmt = "Boundary Pressure Imbalance: {:10.3e} " + "Pa"
        super().__init__(eq, target, weight, name)


    def build(self, eq, use_jit=True, verbose=1):
        
        self.mf = eq.M+1 if self.mf is None else self.mf
        self.nf = eq.N if self.nf is None else self.nf
        self.ntheta = 4*eq.M+1 if self.ntheta is None else self.ntheta
        self.nzeta = 4*eq.N+1 if self.nzeta is None else self.nzeta
        
        eq._sym = False
        self.nest = Nestor(eq, ext_field, self.mf, self.nf, self.ntheta, self.nzeta)
        eq._sym = True
        Bgrid = LinearGrid(rho=np.array([1]), theta=np.linspace(0,2*np.pi,self.ntheta, endpoint=False), zeta=np.linspace(0,2*np.pi/eq.NFP,self.nzeta, endpoint=False), NFP=eq.NFP) 
        self.BR_transform = Transform(Bgrid, eq.R_basis, derivs=1)
        self.BZ_transform = Transform(Bgrid, eq.Z_basis, derivs=1)
        self.BL_transform = Transform(Bgrid, eq.L_basis, derivs=1)

        Igrid = LinearGrid(M=self.ntheta,N=0, rho=1, NFP=eq.NFP) 
        self.IR_transform = Transform(Igrid, eq.R_basis, derivs=1)
        self.IZ_transform = Transform(Igrid, eq.Z_basis, derivs=1)
        self.IL_transform = Transform(Igrid, eq.L_basis, derivs=1)

        self._dim_f = Bgrid.num_nodes
        self.orientation = eq.orientation
        
        self.Bp_profile = eq.pressure.copy()
        self.Bi_profile = eq.iota.copy()
        self.Ip_profile = eq.pressure.copy()
        self.Ii_profile = eq.iota.copy()
        self.Bp_profile.grid = Bgrid
        self.Bi_profile.grid = Bgrid
        self.Ip_profile.grid = Igrid
        self.Ii_profile.grid = Igrid
        
                
        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True
        
        


    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi):
    
        bv = bsq_vac(R_lmn, Z_lmn, L_lmn, i_l, Psi, self.IR_transform, self.IZ_transform, self.IL_transform, self.Ii_profile, self.orientation, self.nest)
        bp = bsq_plasma(R_lmn, Z_lmn, L_lmn, i_l, Psi, self.BR_transform, self.BZ_transform, self.BL_transform, self.Bi_profile, self.orientation)
        bp = bp.reshape((self.nzeta, self.ntheta)).T.flatten()
        return mu_0*(bv - bp - self.Bp_profile([1.0]))
                 
                 




f = Dataset("wout_test_iota.vmec.nc")
print(f["mpol"][:])
print(f["ntor"][:])


veq = VMECIO.load("wout_test_iota.vmec.nc", spectral_indexing="fringe")
veq.change_resolution(L=20, M=10, N=10, L_grid=30, M_grid=15, N_grid=15)

pres = np.asarray(f.variables["presf"])
sp = np.linspace(0, 1, pres.size)
rp = np.sqrt(sp)
pressure = SplineProfile(pres, rp)
pressure.params *= 0

iot = np.asarray(f.variables["iotaf"])
si = np.linspace(0, 1, iot.size)
ri = np.sqrt(si)
iota = SplineProfile(iot, ri)


veq.pressure = pressure
veq.iota = iota

NFP = veq.NFP
mgrid = "tests/inputs/nestor/mgrid_test.nc"
extcur = f["extcur"][:]
folder = os.getcwd()
mgridFilename = os.path.join(folder, mgrid)
ext_field = SplineMagneticField.from_mgrid(
    mgridFilename, extcur, extrap=True, period=(2 * np.pi / NFP)
)


veq.resolution_summary()
veq.solve(ftol=1e-2, xtol=1e-6, gtol=1e-6, maxiter=100, verbose=3)


surf = veq.get_surface_at(1)
surf.change_resolution(M=1, N=0)
eq = Equilibrium(
    Psi=veq.Psi,
    pressure=veq.pressure,
    iota=veq.iota,
    spectral_indexing=veq.spectral_indexing,
    sym=veq.sym,
    NFP=veq.NFP,
)

surf.Z_lmn = surf.R_lmn[-1:]
eq.set_initial_guess(surf)
eq.surface = surf

eq.change_resolution(
    veq.L // 2,
    veq.M // 2,
    veq.N // 2,
    veq.L_grid // 2,
    veq.M_grid // 2,
    veq.N_grid // 2,
)
eq.solve(ftol=1e-2, verbose=3)

from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    FixLambdaGauge,
    FixPressure,
    FixIota,
    FixPsi,
)

bc_objective = BoundaryErrorNESTOR(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
    FixIota(),
    FixPsi(),
)

fb_objective.build(eq)
bc_objective.build(eq)


eq1 = eq.copy()
out = eq1._optimize(
    ObjectiveFunction(bc_objective),
    ObjectiveFunction(fb_objective),
    maxiter=60,
    verbose=3,
    perturb_options={"order":2, "dZb":True, "dRb":True, "tr_ratio":[0.01, 0.01]}
)

eq1.save("run__nestor_vac_out1.h5")
with open("run__nestor_vac_out1.pkl", "wb+") as f:
    pickle.dump(out, f)


eq2 = eq1.copy()

eq2.change_resolution(veq.L, veq.M, veq.N, veq.L_grid, veq.M_grid, veq.N_grid)
eq2.solve(ftol=1e-2, verbose=3)


bc_objective = BoundaryErrorNESTOR(ext_field)
fb_objective = ForceBalance()

objective = ObjectiveFunction(bc_objective)
constraints = (
    fb_objective,
    FixPressure(),
    FixIota(),
    FixPsi(),
)

fb_objective.build(eq2)
bc_objective.build(eq2)


out = eq2._optimize(
    ObjectiveFunction(bc_objective),
    ObjectiveFunction(fb_objective),
    maxiter=60,
    verbose=3,
    perturb_options={"order":2, "dZb":True, "dRb":True, "tr_ratio":[0.01, 0.01]}
)


eq2.save("run__nestor_vac_out2.h5")
with open("run__nestor_vac_out2.pkl", "wb+") as f:
    pickle.dump(out, f)


out = veq._optimize(
    ObjectiveFunction(bc_objective),
    ObjectiveFunction(fb_objective),
    maxiter=60,
    verbose=3,
    perturb_options={"order":2, "dZb":True, "dRb":True, "tr_ratio":[0.01, 0.01]}
)

veq.save("run__nestor_vac_outv.h5")
with open("run__nestor_vac_outv.pkl", "wb+") as f:
    pickle.dump(out, f)
