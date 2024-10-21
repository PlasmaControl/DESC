.. _adding-optimizers:

=====================
Adding new optimizers
=====================

This guide walks through adding an interface to a new optimizer. As an example, we will
write an interface to the popular open source ``ipopt`` interior point method.

We will first need to install the python interface to ``ipopt``, called ``cyipopt`` from
https://github.com/mechmotum/cyipopt

The main steps are to define a wrapper function to handle the interface, and decorate
the wrapper with the ``@register_optimizer`` decorator to tell DESC that the optimizer
exists and how to use it.

The ``register_optimizer`` decorator takes 6 arguments. In all cases the values can either
be single entries or lists, to register multiple versions of the same basic algorithm.
For example, here we register a standard version of ``ipopt`` that uses all derivative
information, and ``ipopt-bfgs`` which only uses approximate Hessians. The necessary fields
are:


- ``name`` (``str``) : Name you want to give the optimization method. This is what you will
  pass to ``desc.optimize.Optimizer`` to select the given method.
- ``description`` (``str``) : A short description of the method, with relevant links
- ``scalar`` (``bool``): Whether the method expects a scalar objective or a vector (for least squares).
- ``equality_constraints`` (``bool``) : Whether the method can handle equality constraints.
- ``inequality_constraints`` (``bool``) : Whether the method can handle inequality constraints.
- ``stochastic`` (``bool``) : Whether the optimizer can be used for stochastic/noisy objectives.
- ``hessian`` (``bool``) : Whether the optimzer uses hessian information.
- ``GPU`` (``bool``) : Whether the optimizer can run on GPUs


The wrapper function itself should take the following arguments:


- ``objective`` (``ObjectiveFunction``) : Function to minimize.
- ``constraint`` (``ObjectiveFunction``) : Constraint to satisfy.
- ``x0`` (``ndarray``) : Starting point.
- ``method`` (``str``) : Name of the method to use (this will be the same as the name
  the method was registered with)
- ``x_scale`` (``array_like``) : Characteristic scale of each variable.
- ``verbose`` (``int``) : level of output to console - 0  : work silently,
  1 : display a termination report, 2 : display progress during iterations
- ``stoptol`` (``dict``) : Dictionary of stopping tolerances, with keys ``{"xtol", "ftol",
  "gtol", "ctol", "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}``
- ``options`` (``dict``) : Optional dictionary of additional keyword arguments to override
  default solver settings.


The wrapper should return a ``scipy.optimize.OptimizeResult`` object.

A full listing of the wrapper function is shown below, with comments to explain the basic
procedure.

::

    from desc.backend import jnp
    from desc.optimize import register_optimizer
    from scipy.optimize import NonlinearConstraint, BFGS
    import cyipopt
    from scipy.optimize._constraints import new_constraint_to_old
    from desc.derivatives import Derivative

    @register_optimizer(
        name=[
            "ipopt",
            "ipopt-bfgs",
        ],
        description="Interior point optimizer, see https://cyipopt.readthedocs.io/en/latest/"
        scalar=True,
        equality_constraints=True,
        inequality_constraints=True,
        stochastic=False,
        hessian=[True, False],
        GPU=False
    )
    def _optimize_ipopt(
        objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
    ):
        """Wrapper for ipopt interior point optimizer.

        Parameters
        ----------
        objective : ObjectiveFunction
            Function to minimize.
        constraint : ObjectiveFunction
            Constraint to satisfy
        x0 : ndarray
            Starting point.
        method : str
            Name of the method to use.
        x_scale : array_like
            Characteristic scale of each variable. Setting x_scale is equivalent to
            reformulating the problem in scaled variables xs = x / x_scale. An alternative
            view is that the size of a trust region along jth dimension is proportional to
            x_scale[j]. Improved convergence may be achieved by setting x_scale such that
            a step of a given size along any of the scaled variables has a similar effect
            on the cost function.
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
        # first set some default behavior and some error checking
        options = {} if options is None else options
        options.setdefault("disp", False)
        options["max_iter"] = stoptol['maxiter']
        if verbose > 2:
            options.set_default("disp", 5)
        x_scale = 1 if x_scale == "auto" else x_scale
        assert x_scale == 1, "ipopt scaling hasn't been implemented"

        # the function and derivative information is contained in the `objective` object
        fun, grad, hess = objective.compute_scalar, objective.grad, objective.hess

        # similarly, the constraint and derivatives are in the `constraint` object
        if constraint is not None:
            # some error checking
            num_equality = jnp.count_nonzero(constraint.bounds[0] == constraint.bounds[1])
            if num_equality > len(x0):
                raise ValueError(
                    "ipopt cannot handle systems with more equality constraints "
                    + "than free variables. Suggest reducing the grid "
                    + "resolution of constraints"
                )
            # do we want to use the full derivative information, or approximate some of it
            if "bfgs" in method:
                conhess_wrapped = BFGS()
            else:
                # define a wrapper function to compute the constraint hessian in the way
                # ipopt expects it
                def confun(y):
                    x = y[:len(x0)]
                    lmbda = y[len(x0):]
                    return jnp.dot(lmbda, constraint.compute_scaled(x))
                conhess = Derivative(confun, mode="hess")
                conhess_wrapped = lambda x, lmbda: conhess(jnp.concatenate([x, lmbda]))
            # we make use of the scipy.optimize.NonlinearConstraint object here to
            # simplify the interface. cyipopt expects things in the same format as
            # scipy.optimize.minimize
            constraint_wrapped = NonlinearConstraint(
                constraint.compute_scaled,
                constraint.bounds_scaled[0],
                constraint.bounds_scaled[1],
                constraint.jac_scaled,
                conhess_wrapped,
            )
            # ipopt expects old style scipy constraints
            constraint_wrapped = new_constraint_to_old(constraint_wrapped, x0)

        else:
            constraint_wrapped = None

        # its helpful to keep a record of all the steps in the optimization.
        # need to use some "global" variables here
        # the function gets called with xs that are not accepted, but usually the
        # gradient is called only with accepted xs so we store those.
        grad_allx = []

        def grad_wrapped(x):
            grad_allx.append(x)
            g = grad(x)
            return g

        # do we want to use the full hessian or only approximate?
        hess_wrapped = None if method in ["ipopt-bfgs"] else hess

        # Now that everything is set up, we call the actual optimizer function
        result = cyipopt.minimize_ipopt(
            fun,
            x0=x0,
            args=(),
            jac=grad_wrapped,
            hess=hess_wrapped,
            constraints=constraint_wrapped,
            tol=stoptol['gtol'],
            options=options,
        )

        # cyipopt already returns a scipy.optimize.OptimizeResult object, so we just
        # need to add some extra information to it
        result["allx"] = grad_allx
        result['allx'].append(result['x'])
        result['message'] = result['message'].decode()

        # finally, we print some info to the console if requested
        if verbose > 0:
            if result["success"]:
                print(result["message"])
            else:
                print("Warning: " + result["message"])
            print("         Current function value: {:.3e}".format(result["fun"]))
            print(
                "         Max constraint violation: {:.3e}".format(
                    0
                if constraint is None
                else jnp.max(jnp.abs(constraint.compute_scaled(result['x']))),
                )
            )
            print("         Total delta_x: {:.3e}".format(jnp.linalg.norm(x0 - result["x"])))
            print("         Iterations: {:d}".format(result["nit"]))
            print("         Function evaluations: {:d}".format(result["nfev"]))
            print("         Gradient evaluations: {:d}".format(result["njev"]))

        return result
