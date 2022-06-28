import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored
import warnings
import copy
from desc.backend import jnp, jit, use_jax, put
from desc.utils import unpack_state, Timer
from desc.io import IOAble
from desc.derivatives import Derivative
from desc.compute import (
    compute_force_error,
    compute_energy,
    compute_quasisymmetry_error,
)

from .objective_funs import ObjectiveFunction

class AugLagrangian(ObjectiveFunction):
    
    def __init__(self, func, constr):
        self.func = func
        self.constr = constr
    
    def scalar(self):
        return False
    
    def name(self):
        return "augmented lagrangian"
    
    def derivatives(self):
        return
    
    def compute(self, x, lmbda, mu):
        L = self.func(x)
        
        for i in range(len(self.constr)):
            L = L - lmbda[i]*self.constr[i](x) + mu/2*self.constr[i](x)**2
        return L
    
    def compute_scalar(self,x,lmbda,mu):
        return self.compute(x,lmbda,mu)
    
    def callback(self, x, lmbda, mu):
        L = self.compute(x,lmbda,mu)
        print("The Lagrangian is " + str(L))
        
class AugLagrangianLS(ObjectiveFunction):
    
    def __init__(self, func, constr):
        self.func = func
        self.constr = constr
    
    def scalar(self):
        return False
    
    def name(self):
        return "least squares augmented lagrangian"
    
    def derivatives(self):
        return
    
    def compute(self, x, mu):
        L = self.func(x)
        c = jnp.zeros(len(self.constr))
        for i in range(len(self.constr)):
            c = put(c,i,mu[i]/2*self.constr[i](x)**2)
        L = jnp.concatenate((L,c),axis=None)
        return L
    
    def compute_scalar(self,x,mu):
        return np.linalg.norm(self.compute(x,mu))
    
    def callback(self, x, mu):
        L = self.compute(x,mu)
        print("The Least Squares Lagrangian is " + str(L))
        
        
class ExLagrangian(ObjectiveFunction):
    def __init__(self, func, eqconstr, ineqconstr):
        self.func = func
        self.eqconstr = eqconstr
        self.ineqconstr = ineqconstr
    
    def scalar(self):
        return False
    
    def name(self):
        return "exact lagrangian"
    
    def derivatives(self):
        return
    
    def compute(self, x, mu):
        L = self.func(x)
        
        for i in range(len(self.eqconstr)):
            L = L + mu*self.eqconstr[i](x)
        for j in range(len(self.ineqconstr)):
            L = L + mu*jnp.maximum(0, self.ineqconstr[i](x))
            
        return L
    
    def compute_scalar(self,x,mu):
        return self.compute(x,mu)
    
    def callback(self, x, mu):
        L = self.compute(x,mu)
        print("The Exact Lagrangian is " + str(L))