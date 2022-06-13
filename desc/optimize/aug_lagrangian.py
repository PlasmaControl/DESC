#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:25:26 2022

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

from desc.objective_funs import AugLagrangian
from desc.derivatives import Derivative

# def gradL(x,fun,lmbda,mu,c,gradc):
#     return

# def L(x,fun,lmbda,mu,c,gradc):    
#     return fun(x) - np.dot(lmbda,c(x)) + mu/2*np.dot(c(x),c(x))

def proj(x,l,u):
    if all(x - l > np.zeros(len(x))) and all(u - x > np.zeros(len(x))):
        return x
    else:
        if not (all(x - l > np.zeros(len(x)))):
            return l
        else:
            return u

def fmin_lag(
    fun,
    x0,
    lmbda0,
    mu0,
    grad,
    constr,
    gradconstr,
    ineq,
    gradineq,
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
    lmbda = lmbda0
    
    L = AugLagrangian(fun, constr)
    gradL = Derivative(L.compute,argnum=0)
    
    mu = mu0
    gtolk = 1/mu0
    ctolk = 1/(mu0**(0.1))    
    
    
    while iteration < maxiter:
        xk = fmintr(L.compute,x,gradL,args=(lmbda,mu),gtol=gtolk,maxiter = maxiter)

        x = xk['x']
        c = 0
        for i in range(len(constr)):
            c = c + constr[i](x)**2
        
        c = np.sqrt(c)
        
        if c < ctolk:
            if c < ctol and gradL(x,lmbda,mu)[0] < gtol:
                break
            else:
                for i in range(len(lmbda)):
                    lmbda[i] = lmbda[i] - mu * constr[i](x)
                    ctolk = ctolk/(mu**(0.9))
                    gtolk = gtolk/mu
        else:
            mu = 100 * mu
            ctolk = 1/(mu**(0.1))
            gtolk = gtolk/mu
        iteration = iteration + 1
    return [xk,lmbda,c]