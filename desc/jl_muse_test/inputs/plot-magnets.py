import numpy as np
import matplotlib.pyplot as plt

'''
Test plot for visualizing MUSE magnet layers

TQ 10 June 2026
'''

fin = "muse-magnets.focus"

datain = np.genfromtxt(fin, delimiter=',', skip_header=3)


X = datain[:,3]
Y = datain[:,4]
Z = datain[:,5]

pho = datain[:,8]
idx = np.argwhere(pho != 0)[:,0]

R = np.sqrt(X*X + Y*Y)
R0 = 0.3048

r = R-R0
rho = np.sqrt(r*r + Z*Z)

phi = np.arctan2(Y,X)
theta = np.arctan2(Z,r)

N_surf = len(R) // 18

Phi = phi.reshape(18, N_surf)
Theta = theta.reshape(18, N_surf)
Pho = pho.reshape(18, N_surf)

fig,axs = plt.subplots(3,6,figsize=(14,8))

for i,a in enumerate(axs.flatten()):
    j = np.argwhere(Pho[i] != 0)[:,0]
    a.scatter(Phi[i,j], Theta[i,j] % (2*np.pi) , c=Pho[i,j], s=3, cmap='jet')
    #a.scatter(Phi[i,j], Theta[i,j], c=Pho[i,j], s=3, cmap='jet')
    #a.scatter(Phi[i], Theta[i], c=Pho[i], s=3)
    a.set_title(i-4)

fig.tight_layout()
plt.show()
