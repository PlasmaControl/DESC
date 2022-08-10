from desc import set_device
set_device('gpu')
import desc.io
from desc.objectives import GXWrapper
from desc.derivatives import nested_zeros_like
import numpy as np
import jax
eq = desc.io.load("/scratch/gpfs/pk2354/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
gxw = GXWrapper(eq=eq)
#gxw.compute(R_lmn=eq.R_lmn,Z_lmn=eq.Z_lmn,L_lmn=eq.L_lmn,i_l=eq.i_l,p_l=eq.p_l,Psi=eq.Psi)
values = (eq.R_lmn,eq.Z_lmn,eq._L_lmn,eq.i_l,eq.p_l,eq.Psi)
#tangents = nested_zeros_like(values)
#tangents[0] = tangents[0].at[0].set(1)
#tangents = (np.random.random(len(x) for x in values[:-1])) + (0.1,)
tangents = ()
for x in values[:-1]:
    print(len(x))
    tangents = tangents + (np.random.random(len(x)),)
tangents = tangents + (0.1,)
#gxw.compute_gx_jvp(values,tangents)
p,t = jax.jvp(gxw.compute,values,tangents)
