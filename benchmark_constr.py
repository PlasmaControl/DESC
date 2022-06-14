#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:27:48 2022

@author: pk123
"""
import numpy as np
from desc.optimize.aug_lagrangian import fmin_lag
from desc.derivatives import Derivative
from desc.backend import jnp

#%% G06
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return (x[0]-10)**3 + (x[1] - 20)**3

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return -(x[0]-5)**2 - (x[1]-5)**2 + 100 + x[2]**2

def ineq_constr_func2(x):
    return (x[0]-6)**2 - (x[1]-5)**2 -82.81 + x[3]**2

def bound_constr1(x):
    return -x[0] + 13 + x[4]**2

def bound_constr2(x):
    return x[0] - 100 + x[5]**2

def bound_constr3(x):
    return -x[1] + x[6]**2

def bound_constr4(x):
    return x[1] - 100 + x[7]**2

grad = Derivative(obj_func, argnum=0)
gradineq1 = Derivative(ineq_constr_func1,argnum=0)
gradineq2 = Derivative(ineq_constr_func2,argnum=0)
gradbound1 = Derivative(bound_constr1,argnum=0)
gradbound2 = Derivative(bound_constr2,argnum=0)
gradbound3 = Derivative(bound_constr3,argnum=0)
gradbound4 = Derivative(bound_constr4,argnum=0)

ic = np.array([ineq_constr_func1,ineq_constr_func2,bound_constr1,bound_constr2,bound_constr3,bound_constr4])
gic = np.array([gradineq1,gradineq2,gradbound1,gradbound2,gradbound3,gradbound4])

#x0 = np.array([15, 50, 45, 45.02, 1.41, 7.07, 3.87, 7.07])
#x0 = np.array([15, 50, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
x0 = np.array([15, 50, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
lmbda0 = 1.0*np.ones(6)
mu0 = 10
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

fopt,xopt,lmbdaf,ctolf = fmin_lag(obj_func,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 50)

#%% G08
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return jnp.sin(2*jnp.pi*x[0])**3 * jnp.sin(2*jnp.pi*x[1])/(x[0]**3*(x[0]+x[1]))

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return x[0]**2 - x[1] + 1 + x[2]**2

def ineq_constr_func2(x):
    return 1 - x[0] + (x[1] - 4)**2 + x[3]**2

def bound_constr1(x):
    return -x[0] + x[4]**2

def bound_constr2(x):
    return x[0] - 10 + x[5]**2

def bound_constr3(x):
    return -x[1] + x[6]**2

def bound_constr4(x):
    return x[1] - 10 + x[7]**2

grad = Derivative(obj_func, argnum=0)
gradineq1 = Derivative(ineq_constr_func1,argnum=0)
gradineq2 = Derivative(ineq_constr_func2,argnum=0)
gradbound1 = Derivative(bound_constr1,argnum=0)
gradbound2 = Derivative(bound_constr2,argnum=0)
gradbound3 = Derivative(bound_constr3,argnum=0)
gradbound4 = Derivative(bound_constr4,argnum=0)

ic = np.array([ineq_constr_func1,ineq_constr_func2,bound_constr1,bound_constr2,bound_constr3,bound_constr4])
gic = np.array([gradineq1,gradineq2,gradbound1,gradbound2,gradbound3,gradbound4])

#x0 = np.array([15, 50, 45, 45.02, 1.41, 7.07, 3.87, 7.07])
#x0 = np.array([15, 50, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
x0 = 5.0*np.ones(8)
lmbda0 = 1.0*np.ones(6)
mu0 = 10
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

fopt,xopt,lmbdaf,ctolf = fmin_lag(obj_func,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 50)



#%% G11
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return x[0]**2 + (x[1] - 1)**2

def eq_constr_func(x):
    return x[1] - x[0]**2

def bound_constr1(x):
    return -x[0] + 1 + x[2]**2

def bound_constr2(x):
    return x[0] - 1 + x[3]**2

def bound_constr3(x):
    return -x[1] + 1 + x[4]**2

def bound_constr4(x):
    return x[1] - 1 + x[5]**2

grad = Derivative(obj_func, argnum=0)
gradeq = Derivative(eq_constr_func,argnum=0)
gradbound1 = Derivative(bound_constr1,argnum=0)
gradbound2 = Derivative(bound_constr2,argnum=0)
gradbound3 = Derivative(bound_constr3,argnum=0)
gradbound4 = Derivative(bound_constr4,argnum=0)

eq = np.array([eq_constr_func])
geq = np.array([gradeq])
ic = np.array([bound_constr1,bound_constr2,bound_constr3,bound_constr4])
gic = np.array([gradbound1,gradbound2,gradbound3,gradbound4])

#x0 = np.array([15, 50, 45, 45.02, 1.41, 7.07, 3.87, 7.07])
#x0 = np.array([15, 50, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
#x0 = np.array([0.5, 0.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
x0 = 0.5*np.ones(6)

lmbda0 = 1.0*np.ones(5)
mu0 = 10
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

fopt,xopt,lmbdaf,ctolf = fmin_lag(obj_func,x0,lmbda0,mu0,grad,eq,gradeq,ic,gic,maxiter = 50)
