#!/usr/bin/env python3
"""
Created on Wed Jul 13 10:39:48 2022

@author: pk123
"""

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from desc.backend import jnp
from desc.derivatives import Derivative
from desc.objectives._auglagrangian import AugLagrangian
from desc.optimize.fmin_scalar import fmintr


def conv_test(x, L, gL):
    return np.linalg.norm(jnp.dot(gL.T, L))


def fmin_lag_stel(
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
    x = np.append(x, 1.0 * np.ones(ineq_dim))

    if maxiter is None:
        maxiter = N * 100

    mu = options.pop("mu", 1.0)
    lmbda = options.pop("lmbda", 0.0 * jnp.ones(eq_dim + ineq_dim))
    bounds = options.pop("bounds", jnp.zeros(eq_dim + ineq_dim))

    def recover(x):
        return x[0 : len(x) - ineq_dim]

    def wrapped_constraint(x):
        c = np.array([])
        slack = x[len(recover(x)) :] ** 2
        x_recov = recover(x)
        eq = eq_constr(x_recov) if eq_constr is not None else jnp.array([])
        ineq = ineq_constr(x_recov) if ineq_constr is not None else jnp.array([])

        c = jnp.append(c, eq_constr(x_recov))
        slack = jnp.append(jnp.zeros(len(eq.flatten())), slack)
        c = jnp.append(c, ineq)
        c = c - bounds + slack
        return c

    def wrapped_obj(x):
        return fun(recover(x))

    constr = np.array([wrapped_constraint])
    L = AugLagrangian(wrapped_obj, constr)
    gradL = Derivative(L.compute, 0, "rev")
    hessL = Derivative(L.compute, argnum=0, mode="hess")

    gtolk = 1 / (10 * mu)
    c = np.linalg.norm(L.compute_constraints(x))
    ctolk = c / (mu ** (0.1))
    xold = x
    f = fun(recover(x))
    fold = f
    cv = L.compute_constraints(x)
    # lmbda = lmbda - mu * cv
    print("f is " + str(f))
    print("lmbda term is " + str(np.dot(lmbda, cv)))
    print("mu term is " + str(mu / 2 * np.dot(cv, cv)))
    print("The constraints are " + str(c))

    while iteration < maxiter:
        # xk = fmintr(
        #    L.compute,
        #    x,
        #    grad=gradL,
        #    hess=hessL,
        #    args=(
        #        lmbda,
        #        mu,
        #    ),
        #    gtol=gtolk,
        #    maxiter=20,
        #    verbose=2,
        # )
        xk = minimize(
            L.compute,
            x,
            args=(lmbda, mu),
            method="trust-exact",
            jac=gradL,
            hess=hessL,
            options={"maxiter": 20},
        )
        x = xk["x"]
        f = fun(recover(x))
        print("f is " + str(f))
        cv = L.compute_constraints(x)
        print("The constraints are " + str(np.linalg.norm(cv)))
        print("lmbda term is " + str(np.dot(lmbda, cv)))
        print("mu term is " + str(mu / 2 * np.dot(cv, cv)))
        print("\n")
        c = np.linalg.norm(cv)

        if np.linalg.norm(xold - x) < xtol:
            print("xtol satisfied\n")
            break

        # if (np.linalg.norm(f) - np.linalg.norm(fold)) / np.linalg.norm(fold) > 0.1:
        #    mu = mu / 2
        #    print("Decreasing mu. mu is now " + str(np.mean(mu)))

        if c < ctolk:
            if c < ctol and gradL(x, lmbda, mu) < gtol:
                break

            else:
                print("Updating lambda")
                lmbda = lmbda - mu * cv
                ctolk = ctolk / (mu ** (0.9))
                gtolk = gtolk / (mu)
        else:
            mu = 2 * mu
            ctolk = c / (mu ** (0.1))
            gtolk = gtolk / mu

        iteration = iteration + 1
        xold = x
        fold = f

    g = gradL(x, lmbda, mu)
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
