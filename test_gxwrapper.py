from desc import set_device
set_device('gpu')
import desc.io
from desc.objectives import GXWrapper
from desc.derivatives import nested_zeros_like
import numpy as np
import jax
from desc.vmec import VMECIO
import dill
#eq = desc.io.load("/scratch/gpfs/pk2354/DESC/docs/notebooks/tutorials/qs_initial_guess.h5")
#eq = desc.io.load("/scratch/gpfs/pk2354/DESC/desc/examples/DSHAPE_output.h5")[-1]
#eq = desc.io.load("/scratch/gpfs/pk2354/DESC/test_equilibria/unconstrained_qs.h5")
#eq.change_resolution(M=6,L=6,M_grid=12,L_grid=12)
#eq = desc.io.load("/scratch/gpfs/pk2354/DESC/examples/VMEC/input.set_of_configs_for_gx_output.h5")[-1]
#eq = VMECIO.load("/scratch/gpfs/pk2354/DESC/examples/VMEC/wout_set_of_configs_for_gx.nc",L=8,M=8,N=8)
#eq = desc.io.load("/scratch/gpfs/pk2354/DESC/examples/VMEC/OPT_output.h5")[-1]
#eq = desc.io.load("/scratch/gpfs/pk2354/DESC/examples/DESC/DSHAPE_current_output.h5")[-1]
#eq = desc.io.load("/scratch/gpfs/pk2354/DESC/examples/VMEC/OPT_scans/OPT_z_m0_n-1_-0.00_output.h5")[-1]
#eq = desc.io.load("/scratch/gpfs/pk2354/DESC/desc/examples/NCSX_output.h5")[-1]
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/intermediate8_3.h5')
eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/input.nfp4_QH_output.h5')[-1]
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_gx_end_case16_n1_all_4.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_gx_end_case18_n1_all_6.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_gx_end_case20_n1_all_3.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_end4_4.h5')
#eq.change_resolution(M=6,L=6,N=4,M_grid=12,L_grid=12,N_grid=8)
surfaces = [0.2, 0.4, 0.6, 0.8]
fluxes = []
count = 0
for rho in surfaces:
    print("rho = " + str(rho))
    gxw = GXWrapper(eq=eq,psi=rho**2,t=count,nzgrid=64,npol=2)

    #filename = "test_dill"

    #filehandler = open(filename, 'wb')
    #dill.dump(gxw, filehandler)
    #filehandler.close()

    qflux = gxw.compute(R_lmn=eq.R_lmn,Z_lmn=eq.Z_lmn,L_lmn=eq.L_lmn,i_l=eq.i_l,c_l=eq.c_l,p_l=eq.p_l,Psi=eq.Psi,ne_l=eq.ne_l, Ti_l=eq.Ti_l, Zeff_l=eq.Zeff_l, Te_l=eq.Te_l)
    fluxes.append(qflux)
    count = count + 1
f = open('original_fluxes_npol2_2.txt','w')
f.write(str(fluxes))
f.close()
    #values = (eq.R_lmn,eq.Z_lmn,eq._L_lmn,eq.i_l,eq.p_l,eq.Psi)
    #tangents = nested_zeros_like(values)
    #tangents[0] = tangents[0].at[0].set(1)
    #tangents = (np.random.random(len(x) for x in values[:-1])) + (0.1,)
    #tangents = ()
    #for x in values[:-1]:
    #    print(len(x))
    #    tangents = tangents + (np.random.random(len(x)),)
    #tangents = tangents + (0.1,)
    #gxw.compute_gx_jvp(values,tangents)
    #p,t = jax.jvp(gxw.compute,values,tangents)
