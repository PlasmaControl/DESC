#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:39:48 2022

@author: pk123
"""

import numpy as np
from termcolor import colored
from desc.backend import jnp
from .derivative import CholeskyHessian
from .utils import (
    check_termination,
    evaluate_quadratic_form,
    print_header_nonlinear,
    print_iteration_nonlinear,
    status_messages,
)
from .tr_subproblems import (
    solve_trust_region_dogleg,
    solve_trust_region_2d_subspace,
    update_tr_radius,
)
from scipy.optimize import OptimizeResult

from desc.optimize.fmin_scalar import fmintr
from desc.optimize.least_squares import lsqtr

#from desc.objective_funs import AugLagrangian
from desc.derivatives import Derivative
#from desc.objective_funs import AugLagrangianLS
from desc.objectives.auglagrangian_objectives import AugLagrangianLS2
# def gradL(x,fun,lmbda,mu,c,gradc):
#     return

# def L(x,fun,lmbda,mu,c,gradc):    
#     return fun(x) - np.dot(lmbda,c(x)) + mu/2*np.dot(c(x),c(x))

# def proj(x,l,u):
#     if all(x - l > np.zeros(len(x))) and all(u - x > np.zeros(len(x))):
#         return x
#     else:
#         if not (all(x - l > np.zeros(len(x)))):
#             return l
#         else:
#             return u
        
# def conv_test(x,gL,l,u):
#     if not (l and u):
#         return np.linalg.norm(gL)
#     else:
#         return np.linalg.norm(x - proj(x - gL, l, u))
    

def conv_test(x,L,gL):
    return np.linalg.norm(jnp.dot(gL.T,L))

# def bound_constr(x,lmbda,mu,gradL)

def fmin_lag_ls_stel(
    fun,
    x0,
    lmbda0,
    mu0,
    grad,
    eq_constr,
    ineq_constr,
    bounds,
    hess="bfgs",
    args=(),
    method="dogleg",
    x_scale=1,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    ctol=1e-6,
    verbose=1,
    maxiter=None,
    callback=None,
    options={},
):
    nfev = 0
    ngev = 0
    nhev = 0
    iteration = 0

    N = x0.size
    x = x0.copy()
    f = fun(x, *args)
    nfev += 1
    g = grad(x, *args)
    ngev += 1
    
    
    ineq_dim = 0
    for i in range(len(ineq_constr)):
        ineq_dim = ineq_dim + len(np.array([ineq_constr[i](x)]).flatten())
    
    x = np.append(x,1.0*np.ones(ineq_dim))
    
    
    def recover(x):
        return x[0:len(x)-ineq_dim]
    
    def wrapped_constraint(x):
        c = np.array([])
        slack = x[len(recover(x)):]**2

        for i in range(len(eq_constr)):
            eq = eq_constr[i](recover(x))
            c = jnp.append(c,eq)
            slack = jnp.append(jnp.zeros(len(jnp.array(eq).flatten())),slack)
            
        for i in range(len(ineq_constr)):
            ineq = ineq_constr[i](recover(x))
            c = jnp.append(c,ineq)
        c = c - bounds + slack
        return c
    
    def wrapped_obj(x):
        return fun(recover(x))
        
    constr = np.array([wrapped_constraint])         

    L = AugLagrangianLS2(wrapped_obj, constr)
    gradL = Derivative(L.compute,0,"fwd")
    hessL = Derivative(L.compute,argnum=0,mode="hess")
    
    # constr = np.concatenate((constr,ineq),axis=None)
    # gradconstr = np.concatenate((gradconstr,gradineq),axis=None)
    
    # L = AugLagrangianLS(fun, constr)
    # gradL = Derivative(L.compute,0,"fwd")
    
    mu = mu0
    lmbda = lmbda0
    gtolk = 1/(10*np.linalg.norm(mu0))
    ctolk = 1/(np.linalg.norm(mu0)**(0.1))    
    xold = x
    f = fun(recover(x))
    fold = f
    print("mu is " + str(mu)) 
    while iteration < maxiter:
        #print(gtolk)
        xk = lsqtr(L.compute,x,gradL,args=(lmbda,mu,),gtol=gtolk,maxiter = int(10),verbose=2)
        x = xk['x']

        c = 0
        
        # for i in range(len(constr)):
        #     c = c + constr[i](x)**2
        
        # c = np.sqrt(c)
        
        f = fun(recover(x))
        cv = L.compute_constraints(x)
        c = np.linalg.norm(cv)
        print("The slack variable is now: " + str(x[len(x)-1]))

        if np.linalg.norm(xold - x) < xtol:
            print("xtol satisfied\n")
            break        
        

        if (np.linalg.norm(f) - np.linalg.norm(fold))/np.linalg.norm(fold) > 0.1:
            mu = mu / 2
            print("Decreasing mu. mu is now " + str(np.mean(mu)))
        elif c < ctolk:

            if c < ctol and conv_test(x,L.compute(x,lmbda,mu),gradL(x,lmbda,mu)) < gtol:
                break

            else:
                print("Updating lambda")
                lmbda = lmbda - mu*cv
                #ctolk = ctolk/(np.linalg.norm(mu)**(0.9))
                #gtolk = gtolk/np.linalg.norm(mu)
                ctolk = ctolk/(np.max(mu)**(0.9))
                gtolk = gtolk/(np.max(mu))
        else:
             mu = 5 * mu
             #ctolk = 1/(np.linalg.norm(mu)**(0.1))
             #gtolk = gtolk/np.linalg.norm(mu)
             ctolk = ctolk/(np.max(mu)**(0.1))
             gtolk = gtolk/np.max(mu)
        
        
        iteration = iteration + 1
        xold = x
        fold = f
        #print(fun(x))
        #print("\n")
        #print(x)

        
        print("The objective function is " + str(np.linalg.norm(f)))
        print("The constraints are " + str(c))
        print("The aspect ratio constraint is " + str(cv[-1]))
        print("x is " + str(recover(x)))
        print("The iteration is " + str(iteration))
        #xr = recover(x)
    #return [fun(xr),xr,mu,c,gradL(x,mu)]
    g = gradL(x,lmbda,mu)
    f = fun(recover(x))
    success = True
    message = "successful"
    result = OptimizeResult(
        x=x,
        success=success,
        fun=f,
        grad=g,
        optimality=jnp.linalg.norm(g),
        nfev=nfev,
        ngev=ngev,
        nhev=nhev,
        nit=iteration,
        message=message,
    )
    result["allx"] = [recover(x)]
    return result
