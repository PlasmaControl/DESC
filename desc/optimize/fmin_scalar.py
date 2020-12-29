import numpy as np
from desc.backend import jnp
from .derivative import CholeskyHessian
from .utils import check_termination, OptimizeResult, evaluate_quadratic_form, print_header_nonlinear, print_iteration_nonlinear, status_messages
from .tr_subproblems import solve_trust_region_dogleg, solve_trust_region_2d_subspace, update_tr_radius


def fmin_scalar(fun, x0, grad,
                hess='bfgs',
                init_hess=None,
                args=(),
                method='dogleg',
                x_scale=1,
                ftol=1e-8,
                xtol=1e-8,
                agtol=1e-8,
                rgtol=1e-8,
                verbose=1,
                maxiter=None,
                callback=None,
                options={}):
    """Minimize a scalar function using a (quasi)-Newton trust region method

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> float
    x0 : array-like
        initial guess
    grad : callable
        function to compute gradient, df/dx. Should take the same arguments as fun
    hess : callable or 'bfgs', optional:
        function to compute hessian matrix of fun, or 'bfgs' in which case the BFGS method
        will be used to approximate the hessian.
    init_hess : array-like, optional
        initial value for hessian matrix, used if hess='bfgs'
    args : tuple
        additional arguments passed to fun, grad, and hess
    method : 'dogleg' or 'subspace'
        method to use for trust region subproblem
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting `x_scale` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust region along jth
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting `x_scale` such that a step of a given size
        along any of the scaled variables has a similar effect on the cost
        function. If set to 'hess', the scale is iteratively updated using the
        inverse norms of the columns of the hessian matrix.
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
    agtol : float or None, optional
        Absolute tolerance for termination by the norm of the gradient. Default is 1e-8.
        Optimizer teriminates when ``norm(g) < agtol``, where
        If None, the termination by this condition is disabled.
    rgtol : float or None, optional
        Relative tolerance for termination by the norm of the change in the gradient.
        Default is 1e-8. Optimizer teriminates when
        ``norm(dg) < rgtol * (rgtol * norm(g))``,
        If None, the termination by this condition is disabled.
   verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations
    callback : callable, optional
        Called after each iteration. Should be a callable with
        the signature:
            ``callback(xk, OptimizeResult state) -> bool``
        where ``xk`` is the current parameter vector. and ``state``
        is an `OptimizeResult` object, with the same fields
        as the ones from the return. If callback returns True
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
        `OptimizeResult` for a description of other attributes.

    """

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

    if method == 'dogleg':
        subproblem = solve_trust_region_dogleg
    elif method == 'subspace':
        subproblem = solve_trust_region_2d_subspace
    else:
        raise ValueError("method should be one of 'dogleg' or 'subspace'")

    if maxiter is None:
        maxiter = N * 100
    max_nfev = options.pop('max_nfev', maxiter)
    max_ngev = options.pop('max_ngev', max_nfev)
    max_nhev = options.pop('max_nhev', max_nfev)
    gnorm_ord = options.pop('gnorm_ord', np.inf)
    xnorm_ord = options.pop('xnorm_ord', 2)
    step_accept_threshold = options.pop('step_accept_threshold', 0.15)
    hess_recompute_freq = options.pop(
        'hessian_recompute_interval', 1 if callable(hess) else 0)
    hess_damp_ratio = options.pop('hessian_damping_ratio', 0.2)
    hess_exception_strategy = options.pop(
        'hessian_exception_strategy', 'damp_update')
    hess_min_curvature = options.pop('hessian_minimum_curvature', None)
    ga_fd_step = options.pop('ga_fd_step', 0.1)
    ga_accept_threshold = options.pop('ga_accept_threshold', 1)

    if np.size(hess_recompute_freq) == 1 and hess_recompute_freq > 0:
        hess_recompute_iters = np.arange(1, maxiter, hess_recompute_freq)
    elif np.size(hess_recompute_freq) == 1:  # never recompute
        hess_recompute_iters = []
    else:
        hess_recompute_iters = hess_recompute_freq

    if hess == 'bfgs':
        bfgs = True
        if init_hess is None:
            init_hess = 'auto'
        hess = CholeskyHessian(N, init_hess,
                               hessfun=None,
                               hessfun_args=(),
                               exception_strategy=hess_exception_strategy,
                               min_curvature=hess_min_curvature,
                               damp_ratio=hess_damp_ratio)

    elif callable(hess):
        hess = CholeskyHessian(N, init_hess,
                               hessfun=hess,
                               hessfun_args=tuple(args),
                               exception_strategy=hess_exception_strategy,
                               min_curvature=hess_min_curvature,
                               damp_ratio=hess_damp_ratio)
    else:
        raise ValueError("hess should either be a callable or 'bfgs'")

    hess_scale = isinstance(x_scale, str) and x_scale == 'hess'
    if hess_scale:
        scale, scale_inv = hess.get_scale()
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    # initial trust region radius
    g_norm = np.linalg.norm(g, ord=gnorm_ord)
    x_norm = np.linalg.norm(x, ord=xnorm_ord)
    trust_radius = options.pop(
        'initial_trust_radius', np.linalg.norm(x * scale_inv))
    max_trust_radius = options.pop('max_trust_radius', trust_radius * 1000.0)
    min_trust_radius = options.pop('min_trust_radius', 0)
    tr_increase_threshold = options.pop('tr_increase_threshold', 0.75)
    tr_decrease_threshold = options.pop('tr_decrease_threshold', 0.25)
    tr_increase_ratio = options.pop('tr_increase_ratio', 2)
    tr_decrease_ratio = options.pop('tr_decrease_ratio', 0.25)

    if trust_radius == 0:
        trust_radius = 1.0
    if len(options) > 0:
        raise ValueError("Unknown options: {}".format(
            [key for key in options]))

    success = None
    step_norm = np.inf
    actual_reduction = np.inf
    ratio = 1   # ratio between actual reduction and predicted reduction
    dg_norm = np.inf

    if verbose > 1:
        print_header_nonlinear()

    while True:

        if iteration in hess_recompute_iters:
            hess.recompute(x)
            nhev += 1

        success, message = check_termination(actual_reduction, f, step_norm, x_norm, dg_norm, g_norm, ratio,
                                             ftol, xtol, rgtol, agtol, iteration, maxiter, nfev, max_nfev, ngev, max_ngev)
        if success is not None:
            result = OptimizeResult(x=x, success=success, fun=f, jac=g, hess=hess.get_matrix(),
                                    inv_hess=hess.get_inverse(), optimality=g_norm, nfev=nfev,
                                    ngev=ngev, nhev=nhev, nit=iteration, message=message)
            break

        # Solve the sub-problem.
        # This gives us the proposed step relative to the current position
        # and it tells us whether the proposed step
        # has reached the trust region boundary or not.
        try:
            step_h, hits_boundary = subproblem(g, hess, scale, trust_radius)

        except np.linalg.linalg.LinAlgError:
            result = OptimizeResult(x=x, success=False, fun=f, jac=g, hess=hess.get_matrix(),
                                    inv_hess=hess.get_inverse(), optimality=g_norm, nfev=nfev,
                                    ngev=ngev, nhev=nhev, nit=iteration, message=status_messages['err'])
            break

        # geodesic acceleration
        if ga_accept_threshold > 0:
            g0 = g
            g1 = grad(x + ga_fd_step*step_h*scale, *args)
            ngev += 1
            dg = (g1-g0)/ga_fd_step**2
            ga_step_h = -scale_inv*hess.solve(dg) + 1/ga_fd_step*step_h
            ga_ratio = np.linalg.norm(
                scale*ga_step_h, ord=xnorm_ord)/np.linalg.norm(scale*step_h, ord=xnorm_ord)
            if ga_ratio < ga_accept_threshold:
                step_h += ga_step_h
        else:
            ga_ratio = -1
            ga_step_h = np.zeros_like(step_h)

        # calculate the predicted value at the proposed point
        predicted_reduction = f - \
            evaluate_quadratic_form(step_h, f, g, hess, scale=scale, sqr=False)

#         if predicted_reduction <= 0:
#             result = OptimizeResult(x=x, success=False, fun=f, jac=g, hess=hess.get_matrix(),
#                                     inv_hess=hess.get_inverse(), optimality=g_norm, nfev=nfev,
#                                     ngev=ngev, nhev=nhev, nit=iteration, message=status_messages['approx'])
#             break

        # calculate actual reduction and step norm
        step = scale * step_h
        step_norm = np.linalg.norm(step, ord=xnorm_ord)
        step_h_norm = np.linalg.norm(step_h, ord=xnorm_ord)
        x_new = x + step
        f_new = fun(x_new, *args)
        nfev += 1
        actual_reduction = f - f_new

        # update the trust radius according to the actual/predicted ratio
        trust_radius, ratio = update_tr_radius(trust_radius, actual_reduction, predicted_reduction,
                                               step_h_norm, hits_boundary, max_trust_radius, min_trust_radius,
                                               tr_increase_threshold, tr_increase_ratio,
                                               tr_decrease_threshold, tr_decrease_ratio, ga_ratio, ga_accept_threshold)

        # if reduction was enough, accept the step
        if ratio > step_accept_threshold:
            x_old = x
            x = x_new
            f = f_new
            g_old = g
            g = grad(x, *args)
            ngev += 1
            dg = g - g_old
            dg_norm = np.linalg.norm(dg, ord=gnorm_ord)
            g_norm = np.linalg.norm(g, ord=gnorm_ord)
            x_norm = np.linalg.norm(x, ord=xnorm_ord)
            hess.update(x_new, x_old, g, g_old)

            if hess_scale:
                scale, scale_inv = hess.get_scale()
            if verbose > 1:
                print_iteration_nonlinear(iteration, nfev, f, actual_reduction,
                                          step_norm, g_norm)

            if callback is not None:
                callback(np.copy(x), result)

            iteration += 1

    if verbose > 0:
        if result['success']:
            print(result['message'])
        else:
            print('Warning: ' + result['message'])
        print("         Current function value: {:.3e}".format(result['fun']))
        print("         Iterations: {:d}".format(result['nit']))
        print("         Function evaluations: {:d}".format(result['nfev']))
        print("         Gradient evaluations: {:d}".format(result['ngev']))
        print("         Hessian evaluations: {:d}".format(result['nhev']))

    return result
