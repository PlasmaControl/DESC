import numpy as np
import scipy.optimize

from backend import jnp, conditional_decorator, jit, use_jax
from backend import get_needed_derivatives, unpack_x, rms, jacfwd, jacrev
from zernike import ZernikeTransform, get_zern_basis_idx_dense, get_double_four_basis_idx_dense, symmetric_x
from init_guess import get_initial_guess_scale_bdry
from boundary_conditions import format_bdry
from objective_funs import get_equil_obj_fun
from nodes import get_nodes
from input_output import read_input, output_to_file



def perturb_pres(x,obj_jac,pres_jac,delta_pres,args):
    """perturbs an equilibrium by increasing pressure by delta_pres"""
    
    J = obj_jac(x,*args)
    Jp = pres_jac(x,*args)
    if Jp.ndim <2:
        Jp = Jp[:,np.newaxis]

    # 0 = J*dx + Jp*dp
    
    dx = np.linalg.lstsq(J,np.matmul(Jp,np.atleast_1d(delta_pres)),rcond=1e-6)[0]
    return x + dx


def perturb_iota(x,obj_jac,iota_jac,delta_iota,args):
    """perturbs an equilibrium by changing iota by delta_iota"""
    
    J = obj_jac(x,*args)
    Ji = iota_jac(x,*args)
    if Ji.ndim <2:
        Ji = Ji[:,np.newaxis]
    # 0 = J*dx + Ji*di
    
    dx = np.linalg.lstsq(J,np.matmul(Ji,np.atleast_1d(delta_iota)),rcond=1e-6)[0]
    return x + dx

def perturb_zeta(x,obj_jac,zeta_jac,delta_zeta,args):
    """perturbs an equilibrium by changing zeta derivative coef by delta_zeta"""
    
    J = obj_jac(x,*args)
    Jz = zeta_jac(x,*args)
    if Jz.ndim <2:
        Jz = Jz[:,np.newaxis]
    # 0 = J*dx + Jz*dz
    
    dx = np.linalg.lstsq(J,np.matmul(Jz,np.atleast_1d(delta_zeta)),rcond=1e-6)[0]
    return x + dx

def perturb_psi(x,obj_jac,psi_jac,delta_psi,args):
    """perturbs an equilibrium by changing total flux"""
    
    J = obj_jac(x,*args)
    Js = iota_jac(x,*args)
    if Js.ndim <2:
        Js = Js[:,np.newaxis]
    # 0 = J*dx + Js*ds
    
    dx = np.linalg.lstsq(J,np.matmul(Js,np.atleast_1d(delta_psi)),rcond=1e-6)[0]
    return x + dx    

def perturb_bdry(x,obj_jac,bdryR_jac, bdryZ_jac,delta_Rbdry,delta_Zbdry,args):
    """perturbs an equilibrium by changing the boundary surface"""
    
    J = obj_jac(x,*args)
    JR = bdryR_jac(x,*args)
    JZ = bdryZ_jac(x,*args)
    if JR.ndim <2:
        JR = JR[:,np.newaxis]
    if JZ.ndim <2:
        JZ = JZ[:,np.newaxis]
    # 0 = J*dx + JR*dR + JZ*dZ
    
    dx = np.linalg.lstsq(J,np.matmul(JR,np.atleast_1d(delta_Rbdry))
                         +np.matmul(JZ,np.atleast_1d(delta_Zbdry)),rcond=1e-6)[0]
    return x + dx       
    

def expand_resolution(x,zernt,zern_idx_old,zern_idx_new,lambda_idx_old,lambda_idx_new,nodes_new=None):
    """Expands solution to a higher resolution by filling with zeros
    Also modifies zernike transform object to work at higher resolution
    
    Args:
        x (ndarray): solution at initial resolution
        zernt (ZernikeTransform): zernike transform object corresponding to initial x
        zern_idx_old (ndarray of int, size(nRZ_old,3)): mode indices corresponding to initial R,Z
        zern_idx_new (ndarray of int, size(nRZ_new,3)): mode indices corresponding to new R,Z
        lambda_idx_old (ndarray of int, size(nL_old,2)): mode indices corresponding to initial lambda
        lambda_idx_new (ndarray of int, size(nL_new,2)): mode indices corresponding to new lambda
        nodes_new (ndarray of float, size(Nn_new,3))(optional): new node locations
    Returns:
        x_new (ndarray): solution expanded to new resolution
    """
    
    
    cR,cZ,cL = unpack_x(x,len(zern_idx_old))
    cR_new = np.zeros(len(zern_idx_new))
    cZ_new = np.zeros(len(zern_idx_new))
    cL_new = np.zeros(len(lambda_idx_new))
    old_in_new = np.where((zern_idx_new[:, None] == zern_idx_old).all(-1))[0]
    cR_new[old_in_new] = cR
    cZ_new[old_in_new] = cZ
    old_in_new = np.where((lambda_idx_new[:, None] == lambda_idx_old).all(-1))[0]
    cL_new[old_in_new] = cL
    x_new = np.concatenate([cR_new,cZ_new,cL_new])

    if nodes_new is not None:
        zernt.expand_nodes(nodes_new)
    zernt.expand_spectral_resolution(zern_idx_new)
    
    return x_new, zernt
    
    
    
def solve_eq_continuation(inputs,verbose):
    """Solves for an equilibrium by continuation method
    
    Steps up resolution, perturbs pressure, 3d bdry etc.
    
    Args:
        inputs (dict): dictionary with input parameters defining problem setup and solver options
        verbose (int): 0 = no text output, 1 = some text output, 2 = detailed text output
    
    Returns:
        equil (dict): dictionary of solution values
    """
    
    
    # broadcast parameter arrays to the same size
    arrs = ['Mpol','Ntor','Mnodes','Nnodes','bdry_ratio','pres_ratio','zeta_ratio',
            'errr_ratio','ftol','xtol','gtol','max_nfev']
    max_len = 0
    for a in arrs:
        max_len = max(max_len,len(inputs[a]))
    for a in arrs:
        inputs[a] = np.broadcast_to(inputs[a],max_len,subok=True).copy()

    stell_sym  = inputs['stell_sym']
    NFP        = inputs['NFP']
    Psi_total  = inputs['Psi_total']
    M          = inputs['Mpol'] #arr
    N          = inputs['Ntor'] #arr
    Mnodes     = inputs['Mnodes'] #arr
    Nnodes     = inputs['Nnodes'] #arr
    bdry_ratio = inputs['bdry_ratio'] #arr
    pres_ratio = inputs['pres_ratio'] #arr
    zeta_ratio = inputs['zeta_ratio'] #arr
    errr_ratio = inputs['errr_ratio'] #arr
    errr_mode  = inputs['errr_mode']
    bdry_mode  = inputs['bdry_mode']
    node_mode  = inputs['node_mode']
    cP         = inputs['presfun_params']
    cI         = inputs['iotafun_params']
    axis       = inputs['axis']
    bdry       = inputs['bdry']
    ftol       = inputs['ftol']
    xtol       = inputs['xtol']
    gtol       = inputs['gtol']
    max_nfev   = inputs['max_nfev']
    
    if np.unique(bdry_ratio).size >1 or np.unique(pres_ratio).size >1 or np.unique(zeta_ratio).size >1:
        assert use_jax, "Can't do perturbation method without JAX for derivatives"

    
    # weights
    weights = {'F' : 1.0, # force balance error
               'B' : 1.0, # error in bdry
               'L' : 1.0} # error in sum lambda_mn

    for ii in range(max_len):
        
        if verbose:
            print('====================')
            print('Iteration {}/{}'.format(ii+1,max_len))
            print('====================')
        # initial solution
        if ii == 0: 
    
            # interpolation nodes
            nodes,volumes = get_nodes(Mnodes[ii],Nnodes[ii],NFP,surfs=node_mode,nr=25,nt=25,nz=0)

            # interpolator
            if verbose:
                print('precomputing Fourier-Zernike basis')
            derivatives = get_needed_derivatives('all')
            zern_idx = get_zern_basis_idx_dense(M[ii],N[ii])
            lambda_idx = get_double_four_basis_idx_dense(M[ii],N[ii])
            zernt = ZernikeTransform(nodes,zern_idx,NFP,derivatives)

            # format boundary shape
            bdry_poloidal, bdry_toroidal, bdryR, bdryZ = format_bdry(M[ii], N[ii], NFP, bdry, bdry_mode, bdry_mode)

            # stellarator symmetry
            if stell_sym:
                sym_mat = symmetric_x(M[ii],N[ii])
            else:
                sym_mat = np.eye(2*len(zern_idx) + len(lambda_idx))

            # initial guess
            if verbose:
                print('computing initial guess')
            cR_init,cZ_init = get_initial_guess_scale_bdry(axis,bdry,zern_idx,NFP,mode=bdry_mode,rcond=1e-6)
            cL_init = np.zeros(len(lambda_idx))
            x_init = jnp.concatenate([cR_init,cZ_init,cL_init])
            x_init = jnp.matmul(sym_mat.T,x_init)
            x = x_init
        
        # continuing from prev soln
        else: 
            # figure out what changed:
            if M[ii] != M[ii-1] or N[ii] != N[ii-1]:
                if verbose:
                    print('Expanding spectral resolution from (M,N)=({},{}) to ({},{})'.format(
                        M[ii-1],N[ii-1],M[ii],N[ii]))
                zern_idx_old = zern_idx
                lambda_idx_old = lambda_idx
                zern_idx = get_zern_basis_idx_dense(M[ii],N[ii])
                lambda_idx = get_double_four_basis_idx_dense(M[ii],N[ii])
                bdry_poloidal, bdry_toroidal, bdryR, bdryZ = format_bdry(M[ii], N[ii], NFP, bdry, bdry_mode, bdry_mode)
                
                x, zent = expand_resolution(jnp.matmul(sym_mat,x),zernt,zern_idx_old,zern_idx,lambda_idx_old,lambda_idx)
                if stell_sym:
                    sym_mat = symmetric_x(M[ii],N[ii])
                else:
                    sym_mat = np.eye(2*len(zern_idx) + len(lambda_idx))

                x = jnp.matmul(sym_mat.T,x)
                
            if Mnodes[ii] != Mnodes[ii-1] or Nnodes[ii] != Nnodes[ii-1]:
                if verbose:
                    print('Expanding node resolution from (M,N)=({},{}) to ({},{})'.format(
                        Mnodes[ii-1],Nnodes[ii-1],Mnodes[ii],Nnodes[ii]))

                nodes,volumes = get_nodes(Mnodes[ii],Nnodes[ii],NFP,surfs=node_mode,nr=25,nt=25,nz=0)
                zernt.expand_nodes(nodes)
                
            if pres_ratio[ii] != pres_ratio[ii-1]:
                if verbose:
                    print('Increasing pressure ratio from {} to {}'.format(pres_ratio[ii-1],pres_ratio[ii]))
                equil_obj,callback = get_equil_obj_fun(M[ii],N[ii],zern_idx,lambda_idx,stell_sym,errr_mode,bdry_mode)
                # use params from previous iter because they haven't been perturbed yet
                args = (bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio[ii-1],pres_ratio[ii-1],zeta_ratio[ii-1],errr_ratio[ii-1],
                        NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)
                
                obj_jac = jacfwd(equil_obj,argnums=0)
                pres_jac = jacfwd(equil_obj,argnums=3)                
                delta_pres = (pres_ratio[ii] - pres_ratio[ii-1])*cP
                
                x = perturb_pres(x,obj_jac,pres_jac,delta_pres,args)
            
            if zeta_ratio[ii] != zeta_ratio[ii-1]:
                if verbose:
                    print('Increasing zeta ratio from {} to {}'.format(zeta_ratio[ii-1],zeta_ratio[ii]))
                equil_obj,callback = get_equil_obj_fun(M[ii],N[ii],zern_idx,lambda_idx,stell_sym,errr_mode,bdry_mode)
                # use params from previous iter because they haven't been perturbed yet
                args = (bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio[ii-1],pres_ratio[ii],zeta_ratio[ii-1],errr_ratio[ii-1],
                        NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)
                
                obj_jac = jacfwd(equil_obj,argnums=0)
                zeta_jac = jacfwd(equil_obj,argnums=8)                
                delta_zeta = (zeta_ratio[ii] - zeta_ratio[ii-1])
                
                x = perturb_zeta(x,obj_jac,zeta_jac,delta_zeta,args)
            
            if bdry_ratio[ii] != bdry_ratio[ii-1]:
                if verbose:
                    print('Increasing bdry ratio from {} to {}'.format(bdry_ratio[ii-1],bdry_ratio[ii]))
                equil_obj,callback = get_equil_obj_fun(M[ii],N[ii],zern_idx,lambda_idx,stell_sym,errr_mode,bdry_mode)
                # use params from previous iter because they haven't been perturbed yet
                args = (bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio[ii-1],pres_ratio[ii],zeta_ratio[ii],errr_ratio[ii-1],
                        NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights)
                
                obj_jac = jacfwd(equil_obj,argnums=0)
                Rb_jac = jacfwd(equil_obj,argnums=1)
                Zb_jac = jacfwd(equil_obj,argnums=2)
                delta_Rbdry = (bdry_ratio[ii] - bdry_ratio[ii-1])*bdryR
                delta_Zbdry = (bdry_ratio[ii] - bdry_ratio[ii-1])*bdryZ

                x = perturb_bdry(x,obj_jac,Rb_jac, Zb_jac,delta_Rbdry,delta_Zbdry,args)
            if errr_ratio[ii] != errr_ratio[ii-1]:
                if verbose:
                    print('Increasing err ratio from {} to {}'.format(errr_ratio[ii-1],errr_ratio[ii]))
        # equilibrium objective function
        equil_obj,callback = get_equil_obj_fun(M[ii],N[ii],zern_idx,lambda_idx,stell_sym,errr_mode,bdry_mode)
        args = [bdryR,bdryZ,cP,cI,Psi_total,bdry_ratio[ii],pres_ratio[ii],zeta_ratio[ii],errr_ratio[ii],
                NFP,bdry_poloidal,bdry_toroidal,zernt,zern_idx,lambda_idx,nodes,volumes,weights]
        static_args = (10,11,12,13,14,15,16,17)

        equil_obj_jit = jit(equil_obj, static_argnums=static_args)
        if verbose:
            print('compiling objective function')
        loss = equil_obj_jit(x,*args).block_until_ready() 
        if verbose:
            print('compiled objective')
            
        # normalize weights
        loss_rms = jnp.sum(loss**2)
        weights = {'F' : 1.0/np.sqrt(loss_rms),
                   'B' : 1.0/np.sqrt(loss_rms),
                   'L' : 1.0/np.sqrt(loss_rms)}
        args[-1] = weights

        if verbose:
            print('starting optimization')
        x_init = x
        out = scipy.optimize.least_squares(equil_obj_jit,
                                           x0=x_init,
                                           args=args,
                                           jac='2-point',
                                           method='trf',
                                           x_scale='jac',
                                           ftol=ftol[ii],
                                           xtol=xtol[ii],
                                           gtol=gtol[ii],
                                           max_nfev=max_nfev[ii],
                                           verbose=verbose)
        x = out['x']
        if verbose:
            print('Start of Iteration {}:'.format(ii+1))
            callback(x_init, *args)
            print('End of Iteration {}:'.format(ii+1))
            callback(x, *args)
        # TODO: checkpoint saving after each iteration
    
    cR,cZ,cL = unpack_x(np.matmul(sym_mat,x),len(zern_idx))
    equil = {'r_coef':cR,
             'z_coef':cZ,
             'lambda_idx':cL,
             'zern_idx':zern_idx,
             'lambda_idx':lambda_idx,
             'NFP':NFP,
             'Psi_total':Psi_total,
             'bdry_idx':bdry[:,:2],
             'r_bdry_coef': bdry[:,2],
             'z_bdry_coef': bdry[:,3],
             'pres_coef': cP,
             'iota_coef': cI}
    
    
    print('====================')
    print('Done')
    print('====================')
    return equil