import numpy as np
from termcolor import colored
from desc.backend import jnp
from .utils import (
    check_termination,
    print_header_nonlinear,
    print_iteration_nonlinear,
    status_messages,
    compute_jac_scale,
    evaluate_quadratic,
)
from .tr_subproblems import (
    trust_region_step_exact_svd,
    trust_region_step_exact_cho,
    update_tr_radius,
)
from scipy.optimize import OptimizeResult
import multiprocess as mp
#from pathos.multiprocessing import Pool

def calc_grad(fun,x,dx):
    #print("x + dx is " + str(x+dx))
    #print("type of x+dx is " + str(type(x+dx)))
    #mp.set_start_method("spawn")
    #ctx = mp.get_context("spawn")
    #args = [([x+dx]),([x-dx])]
    #print(mp.get_start_method())
    #with mp.Pool(2) as pool:
    #    out = pool.starmap(fun,args)
    return (fun(x+dx) - fun(x-dx))/np.linalg.norm(dx)
    #return (out[0] - out[1])/np.linalg.norm(dx)

def stoch(
    fun,
    x0,
    jac,
    args=(),
    x_scale=1,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=1,
    maxiter=10,
    tr_method="svd",
    callback=None,
    options={},
):
    """Solve a least squares problem using a (quasi)-Newton trust region method

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> 1d array
    x0 : array-like
        initial guess
    jac : callable:
        function to compute jacobian matrix of fun
    args : tuple
        additional arguments passed to fun, grad, and jac
    x_scale : array_like or ``'jac'``, optional
        Characteristic scale of each variable. Setting ``x_scale`` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust region along jth
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting ``x_scale`` such that a step of a given size
        along any of the scaled variables has a similar effect on the cost
        function. If set to ``'jac'``, the scale is iteratively updated using the
        inverse norms of the columns of the jacobian matrix.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step. If None, the termination by this
        condition is disabled.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8. Optimization is stopped when
        ``norm(dx) < xtol * (xtol + norm(x))``. If None, the termination by
        this condition is disabled.
    gtol : float or None, optional
        Absolute tolerance for termination by the norm of the gradient. Default is 1e-8.
        Optimizer teriminates when ``norm(g) < gtol``, where
        If None, the termination by this condition is disabled.
    verbose : {0, 1, 2}, optional
        * 0 (default) : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    maxiter : int, optional
        maximum number of iterations. Defaults to size(x)*100
    tr_method : {'cho', 'svd'}
        method to use for solving the trust region subproblem. 'cho' uses a sequence of
        cholesky factorizations (generally 2-3), while 'svd' uses one singular value
        decomposition. 'cho' is generally faster for large systems, especially on GPU,
        but may be less accurate in some cases.
    callback : callable, optional
        Called after each iteration. Should be a callable with
        the signature:

            ``callback(xk, *args) -> bool``

        where ``xk`` is the current parameter vector. and ``args``
        are the same arguments passed to fun and jac. If callback returns True
        the algorithm execution is terminated.
    options : dict, optional
        dictionary of optional keyword arguments to override default solver settings.
        See the code for more details.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        ``OptimizeResult`` for a description of other attributes.

    """
    if tr_method not in ["cho", "svd"]:
        raise ValueError(
            "tr_method should be one of 'cho', 'svd', got {}".format(tr_method)
        )
    if isinstance(x_scale, str) and x_scale not in ["jac", "auto"]:
        raise ValueError(
            "x_scale should be one of 'jac', 'auto' or array-like, got {}".format(
                x_scale
            )
        )

    nfev = 0
    njev = 0
    iteration = 0

    n = x0.size
    x = x0.copy()
    step = 0.00001
    
    bound = 0.20
    J = 0
    delta = np.zeros(len(x))
    alpha = 0.75
    for i in range(maxiter):
        step = 0.0001
        f = fun(x, *args)
        m = f.size
        nfev += 1
        cost = 0.5 * jnp.dot(f, f)
        dx = 0.01*np.ones(len(x))
        g = calc_grad(fun,x,dx)
        #J = jac(x, *args)
        njev += 1
        #g = jnp.dot(J.T, f)
        
        gnorm = np.linalg.norm(g)
        if gnorm < gtol:
            success = True
            break
        print("x is " + str(x))
        print("g is " + str(g))
        print("g/gnorm is " + str(g/gnorm))
        
        #x_new = x - step*g/gnorm
        while np.any(np.abs(x - step*g) > np.abs(bound*np.ones(len(x)))):
            print("hit bound!")
            step = step / 5
        x_new = x - step*g + delta
        delta = alpha*(delta - step*g)

        #f_new = fun(x_new,*args)
        #if (f_new-f)/f < -0.05:
        #    x_new = x - 2*step*g/gnorm
        x = x_new
        print("x_new is " + str(x))
        print("delta is " + str(delta))
        iteration += 1

        print("outer loop iteration is " + str(iteration))

    success = True
    message = None
    result = OptimizeResult(
        x=x,
        success=success,
        cost=cost,
        fun=f,
        grad=g,
        jac=J,
        optimality=gnorm,
        nfev=nfev,
        njev=njev,
        nit=iteration,
        message=message,
    )
    result["message"] = ""
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["cost"]))
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Jacobian evaluations: {:d}".format(result["njev"]))

    return result
