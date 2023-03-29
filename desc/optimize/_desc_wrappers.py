from desc.backend import jnp

from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .optimizer import register_optimizer
from .stochastic import sgd
import numpy as np

@register_optimizer(
    name="lsq-exact",
    scalar=False,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=False,
)
def _optimize_desc_least_squares(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.lsqtr.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use. Any of the options in scipy.optimize.least_squares.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol",
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_radius", 1e-3)
        options.setdefault("max_trust_radius", 1.0)

    result = lsqtr(
        objective.compute,
        x0=x0,
        jac=objective.jac,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


@register_optimizer(
    name=["dogleg", "subspace", "dogleg-bfgs", "subspace-bfgs"],
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=[True, True, False, False],
)
def _optimize_desc_fmin_scalar(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.fmintr.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use. Any of the options in scipy.optimize.least_squares.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol",
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    hess = objective.hess if "bfgs" not in method else "bfgs"
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_ratio", 1e-3)
        options.setdefault("max_trust_radius", 1.0)

    result = fmintr(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
        hess=hess,
        args=(),
        method=method.replace("-bfgs", ""),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


@register_optimizer(
    name="sgd",
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=True,
    hessian=False,
)
def _optimize_desc_stochastic(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.sgd.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use. Any of the options in scipy.optimize.least_squares.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol",
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    
    def grad_fd(x_reduced,x0):
        x = x_reduced
        fx = objective.compute(x)
#        dx = 0.1*np.abs(x0)
        dx = 0.1*np.power(10,np.floor(np.log10(np.abs(x))))

        tang = np.eye(len(dx))
        jac = np.zeros((len(fx),len(tang)))
        for i in range(len(tang)):
            tang[i][i] = dx[i]
        for i in range(len(tang)):
            df = (objective.compute(x_reduced+tang[:,i].T)-fx)/np.linalg.norm(objective.recover(tang[:,i].T))
            jac[:,i] = df
        return fx.T @ jac


    result = sgd(
        objective,
        objective.compute_scalar,
        x0=x0,
        grad=grad_fd,
        args=(),
        method=method,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
#    objective._update_equilibrium(result['x'],store=True)
    return result


def grad_spsa(x_reduced,x0):
    x = recover(x_reduced)
    fx = objective.compute(x)
    h = 0.1*np.abs(x0)
    
    jac = np.zeros((len(fx),len(x0)))
    
    N = 4
    for j in range(N):
        djac = np.zeros((len(fx),len(x0)))
        dx = (np.random.binomial(1,0.5,x0.shape)*2-1)*h
        df = objective.compute(recover(x_reduced+dx)) - objective.compute(recover(x_reduced-dx))
        for i in range(len(x0)):
            Zx = recover(dx)
            print("norm of Zx is " + str(np.linalg.norm(Zx)))
            djac[:,i] = df/(2*np.linalg.norm(Zx)*dx[i])
        jac = jac + djac
    jac = jac/N

    return fx.T @ jac

def grad_spsa_rprop(x_reduced,x0):
    x = recover(x_reduced)
    fx = objective.compute(x)
    h = 0.1*np.abs(x0)
    
    jac = np.zeros((len(fx),len(x0)))
    
    N = 2
    for j in range(N):
        djac = np.zeros((len(fx),len(x0)))
        dx = (np.random.binomial(1,0.5,x0.shape)*2-1)*h
        df = objective.compute(recover(x_reduced+dx)) - objective.compute(recover(x_reduced-dx))
        mag = np.power(10,np.floor(np.log10(np.abs(x_reduced))))
        for i in range(len(x0)):
            djaci =  np.abs(df)/df * np.abs(dx[i])/dx[i]*mag[i]
            print("djaci is " + str(djaci))
            djac[:,i] = djaci
        jac = jac + djac
    jac = jac/N

    return fx.T @ jac


def grad_spsa_wrapped(x_reduced,x0_reduced):


    x = recover(x_reduced)
    x0 = recover(x0_reduced)
    fx = objective.compute(x)
    jac = np.zeros((len(fx),len(x)))
#        h = 0.1*np.abs(x0_reduced)
#        dx_reduced = (np.random.binomial(1,0.5,x0_reduced.shape)*2-1)*h

    print("x is " + str(x))
    sign_x = (x >= 0).astype(float) * 2 - 1
    x = put(x,np.where(x==0)[0],1e-6) 

    #        print("dx is " + str(dx))
#        print("unfixed idx is " + str(unfixed_idx))
#        h = 0.0001 * sign_x * np.maximum(1.0, np.abs(x))
#        dx = (np.random.binomial(1,0.5,x0.shape)*2-1)*h

#        h = 0.02*np.abs(x0)
#        dx = (np.random.binomial(1,0.5,x0.shape)*2-1)*h

#        print("h containts 0s: " + str(np.all(h)))
    N = 4
    for j in range(N):
        djac = np.zeros((len(fx),len(x)))
#            h = 1e-6 * sign_x * np.maximum(1.0, np.abs(x))
        h = 0.01*np.power(10,np.floor(np.log10(np.abs(x))))
        dx = (np.random.binomial(1,0.5,x0.shape)*2-1)*h

        df = objective.compute(x+dx) - objective.compute(x-dx)
        for i in range(len(x)):
            djac[:,i] = df/(2*len(dx)*dx[i])

        jac = jac + djac
    jac = jac/N
    return (fx.T @ jac)[unfixed_idx] @ Z

