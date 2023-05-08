#!/usr/bin/env python3
"""
Created on Wed Jul 13 10:39:48 2022

@author: pk123
"""

import numpy as np
from scipy.optimize import OptimizeResult

from desc.backend import jnp
from desc.derivatives import Derivative
from desc.objectives._auglagrangian import AugLagrangian
from desc.optimize.fmin_scalar import fmintr


def conv_test(x, L, gL):
    return np.linalg.norm(jnp.dot(gL.T, L))


def fmin_lag_stel(
    fun,
    constraint,
    x0,
    bounds,
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

    x = x0.copy()
    f = fun(x, *args)
    c = constraint.fun(x)
    nfev += 1

    mu = options.pop("mu", 1)
    lmbda = options.pop("lmbda", jnp.zeros(len(c)))

    constr = np.array([constraint])
    L = AugLagrangian(fun, constr)
    gradL = Derivative(L.compute, 0, "fwd")
    hessL = Derivative(L.compute, argnum=0, mode="hess")

    gtolk = 1 / (10 * np.linalg.norm(mu))
    ctolk = 1 / (np.linalg.norm(mu) ** (0.1))
    xold = x

    while iteration < maxiter:
        xk = fmintr(
            L.compute,
            x,
            grad=gradL,
            hess=hessL,
            args=(
                lmbda,
                mu,
            ),
            gtol=gtolk,
            maxiter=20,
            verbose=2,
        )
        # xk = minimize(
        #     L.compute,
        #     x,
        #     args=(lmbda, mu),
        #     method="trust-exact",
        #     jac=gradL,
        #     hess=hessL,
        #     options={"maxiter": 20},
        #     verbose=2
        # )
        x = xk["x"]
        f = fun(x)
        cv = L.compute_constraints(x)
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
    f = fun(x)
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
    result["allx"] = [x]
    return result
