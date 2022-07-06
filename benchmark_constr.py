#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:27:48 2022

@author: pk123
"""
import numpy as np
from desc.optimize.aug_lagrangian import fmin_lag
from desc.optimize.exact_lagrangian import fmin_exlag
from desc.optimize.aug_lagrangian_ls import fmin_lag_ls
from desc.optimize.aug_lagrangian_stel import fmin_lag_stel
from desc.derivatives import Derivative
from desc.backend import jnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.random import default_rng

#%% G06
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return (x[0]-10)**3 + (x[1] - 20)**3

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return -(x[0]-5)**2 - (x[1]-5)**2 + 100 + x[2]**2

def ineq_constr_func2(x):
    return (x[0]-6)**2 + (x[1]-5)**2 -82.81 + x[3]**2

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

x0 = np.array([15, 50, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
lmbda0 = 1.0*np.ones(6)
mu0 = 10
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

fopt,xopt,lmbdaf,ctolf,gradopt = fmin_lag(obj_func,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)

#%% G06 wrapped
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return (x[0]-10)**3 + (x[1] - 20)**3

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return -(x[0]-5)**2 - (x[1]-5)**2

def ineq_constr_func2(x):
    return (x[0]-6)**2 + (x[1]-5)**2

def bound_constr1(x):
    return -x[0]

def bound_constr2(x):
    return x[0]

def bound_constr3(x):
    return -x[1]

def bound_constr4(x):
    return x[1]

grad = Derivative(obj_func, argnum=0)
gradineq1 = Derivative(ineq_constr_func1,argnum=0)
gradineq2 = Derivative(ineq_constr_func2,argnum=0)
gradbound1 = Derivative(bound_constr1,argnum=0)
gradbound2 = Derivative(bound_constr2,argnum=0)
gradbound3 = Derivative(bound_constr3,argnum=0)
gradbound4 = Derivative(bound_constr4,argnum=0)

ic = np.array([ineq_constr_func1,ineq_constr_func2,bound_constr1,bound_constr2,bound_constr3,bound_constr4])
gic = np.array([gradineq1,gradineq2,gradbound1,gradbound2,gradbound3,gradbound4])

x0 = np.array([15.0, 50.0])
lmbda0 = 1.0*np.ones(6)
mu0 = 10
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

#fopt,xopt,lmbdaf,ctolf,gradopt = fmin_lag(obj_func,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)
#fopt,xopt,lmbdaf,ctolf,gradopt = fmin_lag_stel(obj_func,x0,lmbda0,mu0,grad,np.array([]),ic,bounds=np.array([-100,82.81,-13,100,0,100]),maxiter = 100)
result = fmin_lag_stel(obj_func,x0,lmbda0,mu0,grad,np.array([]),ic,bounds=np.array([-100,82.81,-13,100,0,100]),maxiter = 100)

#%%G06 scipy
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return (x[0]-10)**3 + (x[1] - 20)**3

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return -(x[0]-5)**2 - (x[1]-5)**2 + 100

def ineq_constr_func2(x):
    return (x[0]-6)**2 - (x[1]-5)**2 -82.81

x1 = np.linspace(13, 100,num=1000)
x2 = np.linspace(0, 100,num=1000)
X1, X2 = meshgrid(x1,x2)
f = (X1 - 10)**3 + (X2 - 20)**3
i1 = -(X1 - 5)**2 - (X2 - 5)**2 + 100
i2 = (X1 - 6)**2 - (X2 - 5)**2 - 82.81

plt.figure(1)
plt.contourf(X1,X2,f,50,cmap='RdGy')
plt.colorbar()

plt.figure(2)
plt.contourf(X1,X2,i1,50,cmap='RdGy')
plt.colorbar()

plt.figure(3)
plt.contourf(X1,X2,i2,50,cmap='RdGy')
plt.colorbar()
plt.show()

# im = imshow(f,cmap=cm.RdBu) # drawing the function
# # adding the Contour lines with labels
# # cset = contour(f,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
# # clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# colorbar(im) # adding the colobar on the right
# show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X1, X2, f, rstride=1, cstride=1, 
#                       cmap=cm.RdBu,linewidth=0, antialiased=False)

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

out = minimize(obj_func,np.array([25,25]),method = "trust-constr",bounds=((13,100),(0,100)),constraints=[{"fun": ineq_constr_func1, "type": "ineq"},{"fun": ineq_constr_func2, "type": "ineq"}])

#%% G08
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return -jnp.sin(2*jnp.pi*x[0])**3 * jnp.sin(2*jnp.pi*x[1])/(x[0]**3*(x[0]+x[1]))

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
x0 = 3.5*np.ones(8)
lmbda0 = 1.0*np.ones(6)
mu0 = 100
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

fopt,xopt,lmbdaf,ctolf,grad = fmin_lag(obj_func,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)



#%% G11
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return x[0]**2 + (x[1] - 1)**2

def eq_constr_func(x):
    return x[1] - x[0]**2

def bound_constr1(x):
    return -x[0] - 1 + x[2]**2

def bound_constr2(x):
    return x[0] - 1 + x[3]**2

def bound_constr3(x):
    return -x[1] - 1 + x[4]**2

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
# ic = np.array([])
# gic = np.array([])

x0 = np.array([np.sqrt(0.1), 0.1, 1.0, 1.0, 1.0, 1.0])

lmbda0 = -0.1*np.ones(5)
# x0 = np.array([np.sqrt(0.1),0.1])
# lmbda0 = np.array([0.1])
mu0 = 100

fopt,xopt,lmbdaf,ctolf,gradopt = fmin_lag(obj_func,x0,lmbda0,mu0,grad,eq,gradeq,ic,gic,maxiter = 50)

#%% G11 wrapped
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return x[0]**2 + (x[1] - 1)**2

def eq_constr_func(x):
    return x[1] - x[0]**2

def bound_constr1(x):
    return -x[0]

def bound_constr2(x):
    return x[0]

def bound_constr3(x):
    return -x[1]

def bound_constr4(x):
    return x[1]

eq = np.array([eq_constr_func])
ineq = np.array([bound_constr1,bound_constr2,bound_constr3,bound_constr4])
# ic = np.array([])
# gic = np.array([])

x0 = np.array([np.sqrt(0.1), 0.1])

lmbda0 = -0.1*np.ones(5)
# x0 = np.array([np.sqrt(0.1),0.1])
# lmbda0 = np.array([0.1])
mu0 = 100

#fopt,xopt,lmbdaf,ctolf,gradopt = fmin_lag_stel(obj_func,x0,lmbda0,mu0,grad,eq,ineq,bounds=np.array([0,1,1,1,1]),maxiter = 100)
result = fmin_lag_stel(obj_func,x0,lmbda0,mu0,grad,eq,ineq,bounds=np.array([0,1,1,1,1]),maxiter = 100)
#%% G11 exact
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
# ic = np.array([])
# gic = np.array([])

x0 = np.array([np.sqrt(0.5), 0.5, 1.0, 1.0, 1.0, 1.0])

lmbda0 = -0.1*np.ones(5)
# x0 = np.array([np.sqrt(0.1),0.1])
# lmbda0 = np.array([0.1])
mu0 = 100

fopt,xopt,ctolf = fmin_exlag(obj_func,x0,lmbda0,mu0,grad,eq,gradeq,ic,gic,maxiter = 50)

#%% G24
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return -x[0] - x[1]

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return -2*x[0]**4 + 8*x[0]**3 - 8*x[0]**2 + x[1] - 2 + x[2]**2

def ineq_constr_func2(x):
    return -4*x[0]**4 + 32*x[0]**3 - 88*x[0]**2 + 96*x[0] + x[1] - 36 + x[3]**2

def bound_constr1(x):
    return -x[0] + x[4]**2

def bound_constr2(x):
    return x[0] - 2.5 + x[5]**2

def bound_constr3(x):
    return -x[1] + x[6]**2

def bound_constr4(x):
    return x[1] - 4 + x[7]**2

grad = Derivative(obj_func, argnum=0)
gradineq1 = Derivative(ineq_constr_func1,argnum=0)
gradineq2 = Derivative(ineq_constr_func2,argnum=0)
gradbound1 = Derivative(bound_constr1,argnum=0)
gradbound2 = Derivative(bound_constr2,argnum=0)
gradbound3 = Derivative(bound_constr3,argnum=0)
gradbound4 = Derivative(bound_constr4,argnum=0)

ic = np.array([ineq_constr_func1,ineq_constr_func2,bound_constr1,bound_constr2,bound_constr3,bound_constr4])
gic = np.array([gradineq1,gradineq2,gradbound1,gradbound2,gradbound3,gradbound4])

x0 = np.array([2.5, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
lmbda0 = -1.0*np.ones(6)
mu0 = 100
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

fopt,xopt,lmbdaf,ctolf,gradopt = fmin_lag(obj_func,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)
#%%G24 scipy
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return -x[0] - x[1]

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return -(-2*x[0]**4 + 8*x[0]**3 - 8*x[0]**2 + x[1] - 2)

def ineq_constr_func2(x):
    return -(-4*x[0]**4 + 32*x[0]**3 - 88*x[0]**2 + 96*x[0] + x[1] - 36)

out = minimize(obj_func,np.array([2.5,3.0]),bounds=((0,3),(0,4)),constraints=[{"fun": ineq_constr_func1, "type": "ineq"},{"fun": ineq_constr_func2, "type": "ineq"}])

#%%Least Squares No Constraints

def fun(x, p):
    a0 = x * p[0]
    a1 = jnp.exp(-(x ** 2) * p[1])
    a2 = jnp.cos(jnp.sin(x * p[2] - x ** 2 * p[3]))
    a3 = jnp.sum(
        jnp.array([(x + 2) ** -(i * 2) * pi ** (i + 1) for i, pi in enumerate(p[3:])]),
        axis=0,
    )
    return a0 + a1 + 3 * a2 + a3

p = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
x = np.linspace(-1, 1, 100)
y = fun(x, p)

def res(p):
    return fun(x, p) - y

def constraint1(x):
    return x[0] - 5 + x[6]**2
gradc = Derivative(constraint1,argnum=0)

rando = default_rng(seed=0)
p0 = p + 0.5 * (rando.random(p.size) - 0.5)

jac = Derivative(res, 0, "fwd")

lmbda0 = np.array([])
mu0 = 10
ic = np.array([])
gic = np.array([])

fopt,xopt,lmbdaf,ctolf,gradopt = fmin_lag_ls(res,p0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)


#%% Least squares with constraints 

def res(x):
    return x - jnp.array([1.5,2.0,0.0,0.0,0.0])

def ineq_constr_func1(x):
    return -x[0] + x[2]**2

def ineq_constr_func2(x):
    return -x[1] + x[3]**2

def ineq_constr_func3(x):
    return x[0] + x[1] - 1 + x[4]**2

gradc1 = Derivative(ineq_constr_func1,argnum=0)
gradc2 = Derivative(ineq_constr_func2,argnum=0)
gradc3 = Derivative(ineq_constr_func3,argnum=0)

x0 = np.ones(5)
mu0 = 10*np.ones(3)
ic = np.array([ineq_constr_func1,ineq_constr_func2,ineq_constr_func3])
gic = np.array([gradc1,gradc2,gradc3])

fopt,xopt,muf,ctolf,gradopt = fmin_lag_ls(res,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)


#%% G06 square
def obj_func(x):
    #return jnp.dot(np.ones(2),x)
    return (x[0]-10)**2 + (x[1] - 20)**2

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return -(x[0]-5)**2 - (x[1]-5)**2 + 100 + x[2]**2

def ineq_constr_func2(x):
    return (x[0]-6)**2 + (x[1]-5)**2 -82.81 + x[3]**2

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

x0 = np.array([15, 50, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
lmbda0 = 1.0*np.ones(6)
mu0 = 10
# mu = np.array([1, 10, 100])
# tau = 1/mu*10**(-4)

fopt,xopt,lmbdaf,ctolf,gradopt = fmin_lag(obj_func,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)
#%%G06 least squares

def res(x):
    return x - jnp.array([10.0,20.0,0.0,0.0,0.0,0.0,0.0,0.0])

def ineq_constr_func1(x):
    #return jnp.dot(x,x) - 2
    return -(x[0]-5)**2 - (x[1]-5)**2 + 100 + x[2]**2

def ineq_constr_func2(x):
    return (x[0]-6)**2 + (x[1]-5)**2 -82.81 + x[3]**2

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

x0 = np.ones(8)
mu0 = 10*np.ones(6)


fopt,xopt,muf,ctolf,gradopt = fmin_lag_ls(res,x0,lmbda0,mu0,grad,np.array([]),np.array([]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)

#%%G11 least squares

def res(x):
    return x - jnp.array([0.0,1.0,0.0,0.0,0.0,0.0,0.0])

def eq_constr_func(x):
    return x[1] - x[0]**2

def bound_constr1(x):
    return -x[0] - 1 + x[2]**2

def bound_constr2(x):
    return x[0] - 1 + x[3]**2

def bound_constr3(x):
    return -x[1] - 1 + x[4]**2

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

x0 = np.ones(7)
mu0 = 10*np.ones(5)


fopt,xopt,muf,ctolf,gradopt = fmin_lag_ls(res,x0,lmbda0,mu0,grad,np.array([eq]),np.array([geq]),ic,gic,l=np.array([13,0]),u=np.array([100,100]),maxiter = 100)