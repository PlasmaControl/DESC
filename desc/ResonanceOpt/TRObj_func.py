import desc.io
from desc.objectives import (
    TrappedResonance
)
import numpy as np
import time

# Run objective function
def TrappedResonanceObj(rhos,pitch_invs,KE_frac,alphas,N,eq=None):
    eq = desc.io.load("equil_G1600_DESC_fixed.h5")
    eq_periodicity = (np.inf,np.inf,np.inf) # periodicity in zeta for these equilibrium to make rtz grid
    grid = eq._get_rtz_grid( # returns rho, theta, zeta coordinate grid
        rhos, # radial
        np.array([0]), # poloidal (alpha in this case)
        np.array([0]), # toroidal (zeta in this case)
        coordinates="raz", # rho, alpha, zeta input coordinates
        period=eq_periodicity, # periodicity of coordinate (rho,alpha,zeta)
    )
    Psi = eq.compute("Psi",grid=grid)
    obj = TrappedResonance(eq,rho=rhos,pitch_invs=pitch_invs,KE_frac=KE_frac,alpha=alphas,N=N,Psi=Psi['Psi'])
    obj.build()
    out = obj.compute(eq.params_dict) # when not flattened, this shape is (rho,pitch,energy)

    # Save objective function values to the firm3d directory for plotting with Poincare plots
    # np.savetxt('/Users/paullab/codes/firm3d_fork_10132025/firm3d/examples/trapped_map/obj_DESC.txt',out['obj'][:,0,0]) # value at all surfaces and one pitch (and one energy)
    return out