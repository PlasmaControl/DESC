import numpy as np
from desc.backend import jnp
from desc.derivatives import Derivative
from .objective_funs import ObjectiveFunction
from desc.backend import put

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
    
    def compute(self, x, lmbda, mu):
        L = self.func(x)
        c = self.compute_constraints(x)
        for i in range(len(c)):
            c = put(c,i,-lmbda[i]*c[i] + mu[i]/2*c[i]**2)
        L = jnp.concatenate((L,c),axis=None)
        print("L is evaluated")
        return L
    
    def compute_scalar(self,x,lmbda,mu):
        return np.linalg.norm(self.compute(x,lmbda,mu))
    
    def callback(self, x, lmbda,mu):
        L = self.compute(x,lmbda,mu)
        print("The Least Squares Lagrangian is " + str(L))
        
    def compute_constraints(self,x):
        c = jnp.array([])
        for i in range(len(self.constr)):
            c = jnp.concatenate((c,self.constr[i](x)),axis=None)
        return c


