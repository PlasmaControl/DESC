import numpy as np
import booz_xform as bx
from desc.grid import LinearGrid
from desc.equilibrium import EquilibriaFamily
from desc.plotting import plot_boozer_modes
import matplotlib.pyplot as plt
from desc.utils import Timer
from desc.basis import FourierZernikeBasis

timer = Timer()

plt.close('all')
##################################################### DSHAPE ############################
# fam = EquilibriaFamily.load("examples/DESC/DSHAPE_output.h5")
# # fam = EquilibriaFamily.load("examples/DESC/input.DSHAPE_M16_N16_ansi_output.h5")
# eq = fam[-1]
# timer.start('Booz Xform')
# b = eq.run_booz_xform()
# bx.modeplot(b[0], nmodes=6, sqrts=True, log=True, B0=True)
# timer.stop('Booz Xform')
# timer.disp('Booz Xform')
# B_mn = np.array([[]])
# rho = np.linspace(1,0,num=100,endpoint=False)
# rho = [0.5]
# # for i, r in enumerate(rho):
# #     grid = LinearGrid(M=2 * eq.M_grid + 1, N=2 * eq.N_grid + 1, NFP=eq.NFP, rho=r)
# #     data = eq.compute("|B|_mn", grid)
# #     b_mn = np.atleast_2d(data["|B|_mn"])
# #     B_mn = np.vstack((B_mn, b_mn)) if B_mn.size else b_mn
# timer.start('Booz Xform In DESC')
# fig,ax,d= plot_boozer_modes(eq, log=True, B0=True, num_modes=6, L=100,ax=plt.gca(),linestyle='--')
# timer.stop('Booz Xform In DESC')
# timer.disp('Booz Xform In DESC')

# plt.ylim([2.5e-5,1])
# plt.title('Boozer Harmonics from DESC Implementation for DSHAPE')
###########################################################################################
## ################################ do same for HELIOTRON #############
fam = EquilibriaFamily.load("examples/DESC/HELIOTRON_output.h5")
# fam = EquilibriaFamily.load("examples/DESC/input.DSHAPE_M16_N16_ansi_output.h5")
eq = fam[-1]


timer.start('Booz Xform')
# b,debug_data,quant_datas = eq.run_booz_xform()
b = eq.run_booz_xform()
bx.modeplot(b[0], nmodes=6, sqrts=True, log=True, B0=True)
timer.stop('Booz Xform')

nrho=100
B_mn = np.array([[]])
rho = np.linspace(0.01,1,num=100,endpoint=False)
# rho = [0.5]

### this block below runs at each surface and gets I,G, and other data 
# to plot later
Is=[]
Gs=[]
ds=[]
for i, r in enumerate(rho):
    if eq.N>0:
        grid = LinearGrid(M=2 * eq.M + 1, N=2 * eq.N + 1, NFP=eq.NFP, rho=r)
    else:
        grid = LinearGrid(M=2 * eq.M + 1, N=1, NFP=eq.NFP, rho=r)
    eq.M=2
    eq.N=2
    eq.L_basis.change_resolution(eq.L,eq.M,eq.N)
    data = eq.compute(name="|B|_mn", grid=grid)
    Is.append(data['I'])
    Gs.append(data['G'])
    ds.append(data)
    print(data['transform'])
    b_mn = np.atleast_2d(data["|B|_mn"])
    B_mn = np.vstack((B_mn, b_mn)) if B_mn.size else b_mn
###########################################################
    
timer.start('Booz Xform In DESC')
plot_boozer_modes(eq,ax=plt.gca(), log=True, B0=True, num_modes=6, L=20)
timer.stop('Booz Xform In DESC')
timer.disp('Booz Xform')
timer.disp('Booz Xform In DESC')
plt.title('Boozer Harmonics from DESC Implementation for HELIOTRON')



############# check I,G #######################
# 
# they seem to be off, I think issue is here, esp. with G
# check fitting of B_zeta_mn, maybe we need to provide whole zeta? then
# dont need to worry about mn stuff

I_b = b[0].bsubumnc[0,:]
G_b = b[0].bsubvmnc[0,:]
# Is.reverse()
# Gs.reverse()

rho = np.linspace(0.01,1,num=nrho,endpoint=False)

rho_b = np.linspace(0.01,1,num=len(I_b))

plt.figure()
plt.plot(rho_b,I_b,label='BOOZERXFORM')
plt.plot(rho,(Is),label='DESC',linestyle='--')

plt.ylabel('I')
plt.xlabel('rho')
plt.legend()

plt.figure()
plt.plot(rho_b,G_b,label='BOOZERXFORM')
plt.plot(rho,(Gs),label='DESC',linestyle='--')

plt.ylabel('G')
plt.xlabel('rho')
plt.legend()
################################################
'''
# # compare Bt_c at first rho
# r_idx=0
# plt.figure()
# plt.plot(ds[r_idx]['Bt_c'][0,:],label='DESC fxn')
# plt.plot(debug_data['Bt_c'][r_idx,:],'--',label='DESC into booz xform')
# plt.legend()
# plt.title(f'B_theta_c components at rho={rho[r_idx]}')

# plt.figure()
# plt.plot(ds[r_idx]['Bz_c'][0,:],label='DESC fxn')
# plt.plot(debug_data['Bz_c'][r_idx,:],'--',label='DESC into booz xform')
# plt.legend()
# plt.title(f'B_zeta_c components at rho={rho[r_idx]}')

# plt.figure()
# plt.plot(ds[r_idx]['Bt_mn'],label='DESC fxn')
# plt.plot(debug_data['Bt_mn'][r_idx,:],'--',label='DESC into booz xform')
# plt.legend()
# plt.title(f'B_theta_mn components at rho={rho[r_idx]}')

# plt.figure()
# plt.plot(ds[r_idx]['Bz_mn'],label='DESC fxn')
# plt.plot(debug_data['Bz_mn'][r_idx,:],'--',label='DESC into booz xform')
# plt.legend()
# plt.title(f'B_zeta_mn components at rho={rho[r_idx]}')

# plt.figure()

# control_data = eq.compute(name='B_theta',grid=LinearGrid(M=2 * eq.M + 1, N=2 * eq.N + 1, NFP=eq.NFP, rho=rho[r_idx]))
# plt.plot(control_data['B_theta'],'x',label='control')
# plt.plot(ds[r_idx]['B_theta'],label='DESC fxn')
# plt.plot(quant_datas[r_idx]['B_theta'],'--',label='DESC into booz xform')
# plt.plot(quant_datas[r_idx]['B_theta']-ds[r_idx]['B_theta'],'--',label='diff')

# plt.legend()
# plt.title(f'B_theta at rho={rho[r_idx]}')

# plt.figure()

# control_data = eq.compute(name='B_zeta',grid=LinearGrid(M=2 * eq.M + 1, N=2 * eq.N + 1, NFP=eq.NFP, rho=rho[r_idx]))
# plt.plot(control_data['B_zeta'],'x',label='control')
# plt.plot(ds[r_idx]['B_zeta'],label='DESC fxn')
# plt.plot(quant_datas[r_idx]['B_zeta'],'--',label='DESC into booz xform')
# plt.plot(quant_datas[r_idx]['B_zeta']-ds[r_idx]['B_zeta'],'--',label='diff')

# plt.legend()
# plt.title(f'B_zeta at rho={rho[r_idx]}')

# d_trans = ds[r_idx]['transform']
# b_trans = debug_data['transform']
# ### check fits of the same underlying data with the two transforms
# d_B_t_mn = d_trans.fit(control_data['B_theta'])
# b_B_t_mn = b_trans.fit(control_data['B_theta'])

# plt.figure()
# plt.plot(d_B_t_mn,label='Fit with DESC trans')
# plt.plot(b_B_t_mn,label='Fit with booz trans')
# plt.legend()
# plt.title('B_t_mn fitted from two diff transforms of the same data')
# print(np.isclose(d_B_t_mn,b_B_t_mn)) # they are the same
# ### check fits of the desc fxn data with the two transforms
# d_B_t_mn = d_trans.fit(ds[r_idx]['B_theta'])
# b_B_t_mn = b_trans.fit(ds[r_idx]['B_theta'])

# plt.figure()
# plt.plot(d_B_t_mn,label='Fit with DESC trans')
# plt.plot(b_B_t_mn,label='Fit with booz trans')
# plt.legend()
# plt.title('B_t_mn fitted from two diff transforms of the same data')
# print(np.isclose(d_B_t_mn,b_B_t_mn)) # they are the same




# print(f"booz transf:\n{debug_data['transform']}")
# print(f"\nDESC fxn transf:\n{ds[0]['transform']}")
# # booz transform uses fft but DESC one uses direct?


# print(f"booz basis:\n{debug_data['four_basis']}")
# print(f"\nDESC fxn basis:\n{ds[0]['four_basis']}")
# ## why is the basis NFP 1 for the DESc compute fxn but 19 for the other one??
'''

# compare nu components

# just need to evaluate the nu_s compoennets as a double angle fourier series

# below may be useless
r_idx=0
fig,ax2= plt.subplots(ncols=4,nrows=4)
for i in range(len(ds[r_idx]['nu_s'][0,:])):
    desc_coefs = []
    for j in range(nrho):
        desc_coefs.append(ds[j]['nu_s'][0,i])
    ax2[i%4,i//4].plot(rho,desc_coefs,label='DESC fxn')
    ax2[i%4,i//4].plot(b.numns_b[:,r_idx],'--',label='DESC into booz xform')
    ax2[i%4,i//4].set_title(f"DESC m={ds[r_idx]['nu_xm'][i]} n={ds[r_idx]['nu_xn'][i]}, bx m={b.xm_b[i]} n={b.xn_b[i]}")
plt.legend()
plt.title(f'nu_s components at rho={rho[r_idx]}')
#meaningless if they are different num of modes...
# interestingly, at the 0th index mode, they are off by a negative sign