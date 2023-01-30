#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:39:48 2022

@author: pk123
"""

import numpy as np
from termcolor import colored
from desc.backend import jnp
from .utils import (
    check_termination,
    print_header_nonlinear,
    print_iteration_nonlinear,
    STATUS_MESSAGES,
    compute_jac_scale,
    evaluate_quadratic_form_jac,
)

from .tr_subproblems import (
    solve_trust_region_dogleg,
    solve_trust_region_2d_subspace,
    update_tr_radius,
)
from scipy.optimize import OptimizeResult, least_squares

from desc.optimize.fmin_scalar import fmintr
from desc.optimize.least_squares import lsqtr

from desc.derivatives import Derivative
from desc.objectives._auglagrangian import AugLagrangianLS
    

def conv_test(x,L,gL):
    return np.linalg.norm(jnp.dot(gL.T,L))

def fmin_lag_ls_stel(
    fun,
    x0,
    grad,
    eq_constr,
    ineq_constr,
    args=(),
    x_scale=1,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    ctol=1e-6,
    verbose=1,
    maxiter=None,
    tr_method="svd",
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
    
    eq = eq_constr(x) if eq_constr is not None else jnp.array([])
    ineq = ineq_constr(x) if ineq_constr is not None else jnp.array([])
    eq_dim = len(eq.flatten())
    ineq_dim = len(ineq.flatten())
    x = np.append(x,1.0*np.ones(ineq_dim))
    
    if maxiter is None:
        maxiter = N * 100

    mu = options.pop("mu", 10*jnp.ones(eq_dim + ineq_dim))
    lmbda = options.pop("lmbda", 10*jnp.ones(eq_dim + ineq_dim))
    bounds = options.pop("bounds", jnp.zeros(eq_dim + ineq_dim))
    
    def recover(x):
        return x[0:len(x)-ineq_dim]
    
    def wrapped_constraint(x):
        c = np.array([])
        slack = x[len(recover(x)):]**2
        x_recov = recover(x)
        eq = eq_constr(x_recov) if eq_constr is not None else jnp.array([])
        ineq = ineq_constr(x_recov) if ineq_constr is not None else jnp.array([])

        c = jnp.append(c,eq_constr(x_recov))
        slack = jnp.append(jnp.zeros(len(eq.flatten())),slack)
        c = jnp.append(c,ineq)
        c = c - bounds + slack
        return c
    
    def wrapped_obj(x):
        return fun(recover(x))
        
    constr = np.array([wrapped_constraint])         
    print("constraint is " + str(np.mean(wrapped_constraint(x))))
    L = AugLagrangianLS(wrapped_obj, constr)
    gradL = Derivative(L.compute,0,"fwd")
    hessL = Derivative(L.compute,argnum=0,mode="hess")
    
    gtolk = 1/(10*np.linalg.norm(mu))
    ctolk = 1/(np.linalg.norm(mu)**(0.1))    
    xold = x
    f = fun(recover(x))
    fold = f
    
    while iteration < maxiter:
        xk = lsqtr(L.compute,x,gradL,args=(lmbda,mu,),gtol=gtolk,maxiter=10,verbose=2)

        x = xk['x']
        f = fun(recover(x))
        cv = L.compute_constraints(x)
        c = np.max(cv)
        
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
                ctolk = ctolk/(np.max(mu)**(0.9))
                gtolk = gtolk/(np.max(mu))
        else:
            mu = 5.0 * mu
            ctolk = ctolk/(np.max(mu)**(0.1))
            gtolk = gtolk/np.max(mu)
        
        
        iteration = iteration + 1
        xold = x
        fold = f
        
        print("mu is now " + str(mu)) 
        print("The average c is " + str(np.mean(cv)))
        print("The max c is " + str(np.max(cv)))
        print("The objective function is " + str(np.linalg.norm(f)))
        print("The constraints are " + str(np.linalg.norm(cv)))
        penalty = mu[0]*np.dot(cv,cv)/2
        print("The penalty term is " + str(penalty))
        print("The aspect ratio constraint is " + str(cv[-1]))
        print("x is " + str(recover(x)))
        print("The iteration is " + str(iteration))
    
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
