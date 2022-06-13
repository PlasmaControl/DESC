#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:00:40 2022

@author: pk123
"""
import numpy as np
from desc.optimize.aug_lagrangian import fmin_lag
from desc.derivatives import Derivative
from desc.backend import jnp

def obj_func(x):
    return jnp.dot(np.ones(2),x)

def constr_func(x):
    return jnp.dot(x,x) - 2

def ineq_constr_func(x):
    return -(jnp.dot(x,x) - 1)


grad = Derivative(obj_func, argnum=0)
gradc = Derivative(constr_func, argnum=0)
gradineq = Derivative(ineq_constr_func,argnum=0)

x0 = np.array([-0.75, -0.75])
lmbda0 = np.array([-0.4])
mu0 = 2
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

out,lmbdaf,ctolf = fmin_lag(obj_func,x0,lmbda0,mu0,grad,np.array([constr_func]),np.array([gradc]),np.array([ineq_constr_func]),np.array([gradineq]),maxiter = 50)

    