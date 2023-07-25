from desc.grid import LinearGrid
from desc.examples import get
import matplotlib.pyplot as plt
import numpy as np
import desc.io

# running a job array with SLURM
import os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
s = ((idx % 50)+1) / 50

if idx < 50:
    name = "QA1"
elif idx < 100 and idx >= 50:
    name = "QA2"
elif idx < 150 and idx >= 100:
    name = "QA3"
elif idx < 200 and idx >= 150:
    name = "QA4"
elif idx < 250 and idx >= 200:
    name = "QA4n"
elif idx < 300 and idx >= 250:
    name = "QA5"
elif idx < 350 and idx >= 300:
    name = "QA4-1"
elif idx < 400 and idx >= 350:
    name = "QA4-2"
elif idx < 450 and idx >= 400:
    name = "QA4-3"
elif idx < 500 and idx >= 450:
    name = "QA4-4"
elif idx < 550 and idx >= 500:
    name = "QA5-1"
elif idx < 600 and idx >= 550:
    name = "QA5-2"
elif idx < 650 and idx >= 600:
    name = "QA5-3"
elif idx < 700 and idx >= 650:
    name = "QA5-4"

eq = desc.io.load(name + "_solved.h5")

# from desc.vmec import VMECIO
# eq = VMECIO.load("wout_QA4n.nc") 
# eq.save("wout_QA4n.h5")
# print('loaded eq')

# eq = get("W7-X")

grid = LinearGrid(rho=1,M=40,N=41,axis=False,NFP=eq.NFP,sym=False)
alpha = eq.compute("alpha",grid=grid)["alpha"] % (2*np.pi)



from desc.compute.utils import dot, cross
from desc.grid import Grid

stepswithin2pi = 100
nfulltransits = 100
        
coords = np.ones((stepswithin2pi*nfulltransits,3))
coords[:,0] = coords[:,0] * np.sqrt(s)
coords[:,1] = coords[:,1] * 2
coords[:,2] = np.arange(0, nfulltransits*2*np.pi,2*np.pi/stepswithin2pi)

coords1 = eq.map_coordinates(coords = coords, inbasis = ["rho", "alpha", "zeta"], 
                             outbasis = ["rho", "theta", "zeta"], period = [np.inf, (2*np.pi), (2*np.pi/eq.NFP)])
#print(coords1)

coords1 = coords1.at[:,2].set(coords[:,2])

#print('mapped coords')

# print(np.any(np.isnan(coords1)))
# print(coords1)
# print(np.where(np.isnan(coords1)))

grid2 = Grid(coords1)
#print(grid2)

B = eq.compute('|B|',grid2)['|B|']
# print(B)

# plt.plot(coords1[:,2],B)
# plt.ylabel("|B|")
# plt.xlabel("zeta")

maxB = np.nanmax(B)
#print(maxB)
minB = np.nanmin(np.abs(B))
#print(minB)

bpstep = 80   # iterations through b'
nsteps = len(B)  # steps along each field line (equal to number of B values we have, for now)
bp = np.zeros(bpstep)
deltabp = (maxB - minB) / (minB*bpstep)

wellGamma_c = 0
bigGamma_c = 0

# compute important quantities in DESC. 
grad_psi_mag = eq.compute('|grad(psi)|', grid2)['|grad(psi)|']
grad_psi = eq.compute('grad(psi)', grid2)['grad(psi)']
grad_zeta_mag = eq.compute('|grad(zeta)|', grid2)['|grad(zeta)|']
grad_zeta = eq.compute('e^zeta', grid2)['e^zeta']
grad_B = eq.compute('grad(|B|)', grid2)['grad(|B|)']
e_theta = np.linalg.norm(eq.compute('e_theta', grid2)['e_theta'], axis = -1)
kappa_g = eq.compute('kappa_g', grid2)['kappa_g']
psi = eq.compute('psi', grid2)['psi']
Bsupz = eq.compute('B^zeta', grid2)['B^zeta']
dBsupzdpsi = grad_B[:,2]*2*np.pi/psi
dBdpsi = grad_B[:,2]*2*np.pi/psi

Br = eq.compute('B_R', grid2)['B_R']
Bphi = eq.compute('B_phi', grid2)['B_phi']
zeta = coords1[:,2]
Bxyz = np.zeros((len(B),3))
Bxyz[:,0] = Br*np.cos(zeta) - Bphi*np.sin(zeta)
Bxyz[:,1] = Br*np.sin(zeta) + Bphi*np.cos(zeta)
Bxyz[:,2] = eq.compute('B_Z', grid2)['B_Z']

dVdb_t1 = eq.compute('iota_r', grid2)['iota_r'] * dot(cross(grad_psi, Bxyz), grad_zeta) / B

# finding basic arc length of each segment
x = eq.compute('X', grid2)['X']
y = eq.compute('Y', grid2)['Y']
z = eq.compute('Z', grid2)['Z']
ds = np.sqrt(np.add(np.square(np.diff(x)), np.square(np.diff(y)), np.square(np.diff(z))))

# integrating dl/b
dloverb = 0
for j in range(0, nsteps - 1):
    dloverb += ds[j]/B[j]

# making the b prime array
for i in range(0, bpstep):
    bp[i] = 1+ ((maxB-minB) * (i-0.5) / (minB*bpstep))
    
for i in range(0, bpstep):
    B_reflect = minB*bp[i]
    in_well = 0
    well_start = [0]
    well_end = [0]
    cur_well = 0

    grad_psi_min = 1E10
    grad_psi_i = np.ones(len(B)) * 1E10
    e_theta_min = 0
    e_theta_i = np.zeros(len(B))
    curB_min = B_reflect

    for j in range(0, nsteps):   
        if not(in_well) and B_reflect < B[j]:   # not in well and shouldn't be
            continue 
            
        if in_well and B_reflect < B[j]:    # in well, but just exited
            in_well = 0
            well_end.append(j-2)    # add well end location to well_end (in stellopt they use j-2 instead of j-1, not sure why)

            grad_psi_i[well_start[cur_well]:well_end[cur_well]] = grad_psi_min
            e_theta_i[well_start[cur_well]:well_end[cur_well]] = e_theta_min
            
            curB_min = B_reflect
            e_theta_min = 0
            grad_psi_min = 1E10
            
        if not(in_well) and B_reflect > B[j]:   # not in well but entering one
            in_well = 1
            well_start.append(j+1)     # add well start location to well_start (in stellopt they use j+1 instead of j, not sure why)
            cur_well +=1

        if in_well and B_reflect > B[j]:    # in well and should be there. This always runs if the previous 'if' runs
            if B[j] < curB_min:
                curB_min = B[j]
                grad_psi_min = grad_psi_mag[j]   # grad_psi_mag replaces grad_psi_norm from Stellopt
                e_theta_min = e_theta[j]

    # if we ended in a well, decrease cur_well by 1 so that total_wells is one smaller and the ending well is avoided
    if in_well:
        cur_well -= 1         
    
    total_wells = cur_well

    #print(total_wells)
    #print(B_reflect)
    # print(well_start)
    # print(well_end)
    # print(grad_psi_i)

    vrovervt = 0

    # loop to compute important quantities at each step of b'
    for k in range(1,total_wells+1):
        dIdb = 0
        dgdb = 0
        dbigGdb = 0
        dVdb = 0

        # loop to sum over each well for each quantity
        for j in range(well_start[k],well_end[k]):
            # additional check that we are in a valid well
            if grad_psi_i[j] == 1E10 or e_theta_i[j] == 0: continue

            # intermedite quantity to make other calculations easier
            sqrt_bbb = np.sqrt(1 - B[j]/B_reflect)

            # dIdb
            temp = ds[j]/2/minB/bp[i]/bp[i] / sqrt_bbb
            dIdb = dIdb + temp
                
            # dgdb
            temp = ds[j] * grad_psi_mag[j] * kappa_g[j]
            temp = temp/bp[i]/bp[i]/2/B[j]
            temp = temp*(sqrt_bbb + 1/sqrt_bbb)
            dgdb = dgdb + temp

            # dbigGdb 
            temp = dBdpsi[j] *ds[j] /B_reflect / bp[i] / B[j] / 2
            temp = temp*(sqrt_bbb + 1/sqrt_bbb)
            dbigGdb = dbigGdb + temp

            # dVdb
            temp = dVdb_t1[j] - (2 * dBdpsi[j] - B[j]/Bsupz[j]*dBsupzdpsi[j]) 
            temp = temp * 1.5 * ds[j] / B[j] / B_reflect * sqrt_bbb
            dVdb = dVdb + temp
     
        # print('dIdb: ' + str(dIdb))
        # print('dgdb: ' + str(dgdb))
        # print('dbigGdb: ' + str(dbigGdb))
        # print('dVdb: ' + str(dVdb))

        # if the well start and well end are not the same point, compute vr/vt (radial drift / torodial drift)
        if well_start[k] < well_end[k]:
            temp = dgdb/grad_psi_i[well_start[k]]/dIdb / minB / e_theta_i[well_start[k]]
            temp = temp / (dbigGdb/dIdb + 2/3 * dVdb/dIdb)
            vrovervt = temp
        else: 
            vrovervt = 0


        gamma_c = 2 * np.arctan(vrovervt) / np.pi
        
        wellGamma_c += gamma_c * gamma_c * dIdb

    bigGamma_c += wellGamma_c * np.pi/2/np.sqrt(2)*deltabp

bigGamma_c  = bigGamma_c / dloverb

print(bigGamma_c)

file = name + '_10kSolved.txt'
f=open(file,'a')
f.write(f"{s:1.2f}, {bigGamma_c:1.3e}\n")
f.close()