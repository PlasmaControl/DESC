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
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/input.nfp4_QH_output.h5')[-1]
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_gx_end_case16_n1_all_4.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_gx_end_case18_n1_all_6.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_gx_end_case20_n1_all_3.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_gx_end_case21_n_2_i_4.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_end4_4.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/precise_qh2.h5')
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/test_equilibria/swap_gx_intermediate_case2_n_3_i_31.h5')
#eq.change_resolution(M=6,L=6,N=4,M_grid=12,L_grid=12,N_grid=8)
eq = desc.io.load('/scratch/gpfs/pk2354/DESC/W7X_output.h5')[-1]
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/examples/VMEC/input.BEER_output.h5')[-1]
#eq = desc.io.load('/scratch/gpfs/pk2354/DESC/desc/examples/W7-X_output.h5')[-1]
print('loaded eq')
gxw = GXWrapper(eq=eq,psi=0.5,nzgrid=64,t=1,alpha=np.pi/2,npol=2)

qflux = gxw.compute(R_lmn=eq.R_lmn,Z_lmn=eq.Z_lmn,L_lmn=eq.L_lmn,i_l=eq.i_l,c_l=eq.c_l,p_l=eq.p_l,Psi=eq.Psi,ne_l=eq.ne_l, Ti_l=eq.Ti_l, Zeff_l=eq.Zeff_l, Te_l=eq.Te_l)

