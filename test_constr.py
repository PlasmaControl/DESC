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

grad = Derivative(obj_func, argnum=0)
gradc = Derivative(constr_func, argnum=0)

x0 = np.array([1.0, 1.0])
lmbda0 = np.array([-0.1])
mu = np.array([1, 10, 100])
tau = 1/mu*10**(-4)

out = fmin_lag(obj_func,x0,lmbda0,mu,tau,grad,np.array([constr_func]),np.array([gradc]))

    