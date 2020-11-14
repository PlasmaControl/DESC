"""
Quasi-Symmetry testing script
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from desc.backend import get_needed_derivatives, sign
from desc.zernike import ZernikeTransform, symmetric_x
from desc.zernike import get_zern_basis_idx_dense, get_double_four_basis_idx_dense
from desc.boundary_conditions import format_bdry
from desc.objective_funs import get_equil_obj_fun, get_qisym_obj_fun
from desc.nodes import get_nodes_pattern, get_nodes_surf
from desc.perturbations import get_system_derivatives
from desc.input_output import read_desc
from desc.continuation import expand_resolution


def vec2mat(c, pol, tor, M, N):
    CC = np.zeros((2*N+1, 2*M+1))
    for i in range(c.size):
        CC[tor[i]+N, pol[i]+M] = c[i]
    return CC


equil_fname = 'examples/DESC/ITER_fake202.output_1.out'
bdry3_fname = 'examples/DESC/ITER_fake202_pert.output_1.out'

equil_fname = str(pathlib.Path(equil_fname).resolve())
bdry3_fname = str(pathlib.Path(bdry3_fname).resolve())

equil = read_desc(equil_fname)
bdry3 = read_desc(bdry3_fname)

cR = equil['cR']
cZ = equil['cZ']
cL = equil['cL']
cP = equil['cP']
cI = equil['cI']
NFP = equil['NFP']
Psi_lcfs = equil['Psi_lcfs']
bdry_idx = equil['bdry_idx']
bdryR = equil['bdryR']
bdryZ = equil['bdryZ']
zern_idx = equil['zern_idx']
lambda_idx = equil['lambda_idx']
stell_sym = True
errr_mode = 'force'
bdry_mode = 'spectral'
zern_mode = 'ansi'
node_mode = 'cheb2'
bdry_ratio = 1
pres_ratio = 1
zeta_ratio = 1
errr_ratio = 1e-5

M0 = 10
N0 = 0
M0nodes = 15
N0nodes = 0
M = 10
N = 10
Mnodes = 15
Nnodes = 15

# original resolution
bdry = np.concatenate([bdry_idx, np.atleast_2d(
    bdryR).T, np.atleast_2d(bdryZ).T], axis=1)
equil_nodes, equil_volumes = get_nodes_pattern(
    M0nodes, N0nodes, NFP, index=zern_mode, surfs=node_mode, sym=stell_sym, axis=False)
bdry_nodes, _ = get_nodes_surf(
    M0nodes, N0nodes, NFP, surf=1.0, sym=stell_sym)
derivatives = get_needed_derivatives('all')
equil_zernike_transform = ZernikeTransform(
    equil_nodes, zern_idx, NFP, derivatives, equil_volumes, method='fft')
bdry_zernike_transform = ZernikeTransform(
    bdry_nodes, zern_idx, NFP, [0, 0, 0], method='direct')
x0 = np.concatenate([cR, cZ, cL])

# expanded resolution
equil_nodes, equil_volumes = get_nodes_pattern(
    Mnodes, Nnodes, NFP, index=zern_mode, surfs=node_mode, sym=stell_sym, axis=False)
bdry_nodes, _ = get_nodes_surf(
    Mnodes, Nnodes, NFP, surf=1.0, sym=stell_sym)
equil_zernike_transform.expand_nodes(equil_nodes, equil_volumes)
bdry_zernike_transform.expand_nodes(bdry_nodes)
zern_idx_old = zern_idx
lambda_idx_old = lambda_idx
zern_idx = get_zern_basis_idx_dense(M, N, None, zern_mode)
lambda_idx = get_double_four_basis_idx_dense(M, N)
bdry_pol, bdry_tor, bdryR, bdryZ = format_bdry(
    M, N, NFP, bdry, bdry_mode, bdry_mode)
x0, equil_zernike_transform, bdry_zernike_transform = expand_resolution(x0, equil_zernike_transform, bdry_zernike_transform,
                                                                        zern_idx_old, zern_idx, lambda_idx_old, lambda_idx)
if stell_sym:
    sym_mat = symmetric_x(zern_idx, lambda_idx)
else:
    sym_mat = np.eye(2*zern_idx.shape[0] + lambda_idx.shape[0])
x0 = np.matmul(sym_mat.T, x0)

# equilibrium objective function
equil_obj, _ = get_equil_obj_fun(stell_sym, errr_mode, bdry_mode, M, N,
                                 NFP, equil_zernike_transform, bdry_zernike_transform, zern_idx, lambda_idx,
                                 bdry_pol, bdry_tor)
equil_args = [x0, bdryR, bdryZ, cP, cI, Psi_lcfs,
              bdry_ratio, pres_ratio, zeta_ratio, errr_ratio]

# quasisymmetry objective function
qisym_nodes, qisym_volumes = get_nodes_surf(
    Mnodes, Nnodes, NFP, surf=1.0, sym=stell_sym)
qisym_zernike_transform = ZernikeTransform(
    qisym_nodes, zern_idx, NFP, derivatives, qisym_volumes, method='fft')
qisym_obj = get_qisym_obj_fun(
    stell_sym, M, N, NFP, qisym_zernike_transform, zern_idx, lambda_idx, bdry_pol, bdry_tor)
qisym_args = [x0, cI, Psi_lcfs]

# boundary modes to perturb
bdryR_modes = np.where(np.logical_and(
    bdry_tor != 0, sign(bdry_pol) == sign(bdry_tor)))[0]
bdryZ_modes = np.where(np.logical_and(
    bdry_tor != 0, sign(bdry_pol) != sign(bdry_tor)))[0]
equil_arg_dict = {1: bdryR_modes, 2: bdryZ_modes}
qisym_arg_dict = {}

# jacobian matrices
dFdx, dFdc = get_system_derivatives(
    equil_obj, equil_args, equil_arg_dict, pert_order=1, verbose=2)
dGdx, dGdc = get_system_derivatives(
    qisym_obj, qisym_args, qisym_arg_dict, pert_order=1, verbose=2)

# singular value decomposition
QSmat = dGdx @ np.linalg.pinv(dFdx) @ dFdc
u, s, vh = np.linalg.svd(QSmat, full_matrices=False)

# Plunk's result
bdry_Plunk = np.concatenate([bdry3['bdry_idx'], np.atleast_2d(bdry3['bdryR']).T,
                             np.atleast_2d(bdry3['bdryZ']).T], axis=1)
_, _, bdryR_Plunk, bdryZ_Plunk = format_bdry(
    M, N, NFP, bdry_Plunk, bdry_mode, bdry_mode)
dR = (bdryR_Plunk - bdryR)[bdryR_modes]
dZ = (bdryZ_Plunk - bdryZ)[bdryZ_modes]
dc_Plunk = np.concatenate([dR, dZ])
dc_Plunk = dc_Plunk / np.linalg.norm(dc_Plunk)  # normalize

dc_dots = np.zeros((vh.shape[0],))
for i in range(vh.shape[0]):
    dc_dots[i] = np.abs(np.dot(dc_Plunk, vh[i, :]))
idx = np.argmax(dc_dots)

# our result
dc = vh[idx, :]  # right-singular vector
dc = dc / np.linalg.norm(dc)  # normalize
dx = -np.linalg.pinv(dFdx) @ dFdc @ dc

epsilon = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
QSerrs = np.zeros_like(epsilon)
for i in range(epsilon.size):
    QSerrs[i] = np.sum(qisym_obj(x0+dx*epsilon[i], cI, Psi_lcfs)**2) * 1e3

DC = vec2mat(dc, np.concatenate([bdry_pol[bdryR_modes], bdry_pol[bdryZ_modes]]),
             np.concatenate([bdry_tor[bdryR_modes], bdry_tor[bdryZ_modes]]), M, N)

DC_Plunk = vec2mat(dc_Plunk, np.concatenate([bdry_pol[bdryR_modes], bdry_pol[bdryZ_modes]]),
                   np.concatenate([bdry_tor[bdryR_modes], bdry_tor[bdryZ_modes]]), M, N)

plt.plot(epsilon, QSerrs, 'ro')
plt.plot(epsilon, epsilon**2, 'k-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\epsilon$')
plt.ylabel('$\Sigma g(x)^2$ (arbitrary units)')
plt.show()

plt.imshow(DC, origin='lower', extent=(-M, M, -N, N))
plt.xlabel('M')
plt.ylabel('N')
plt.title('$\Delta c$')
plt.show()

plt.imshow(DC_Plunk, origin='lower', extent=(-M, M, -N, N))
plt.xlabel('M')
plt.ylabel('N')
plt.title('$\Delta c$ (Plunk 2020)')
plt.show()
