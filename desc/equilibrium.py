import numpy as np
import warnings
import numbers
from termcolor import colored
from collections.abc import MutableSequence

from desc.backend import use_jax
from desc.utils import Timer, isalmostequal
from desc.configuration import _Configuration
from desc.io import IOAble
from desc.geometry import FourierRZToroidalSurface, ZernikeRZToroidalSection
from desc.optimize import Optimizer
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)
from desc.perturbations import perturb


class Equilibrium(_Configuration, IOAble):
    """Equilibrium is an object that represents a plasma equilibrium.

    It contains information about a plasma state, including the shapes of flux surfaces
    and profile inputs. It can compute additional information, such as the magnetic
    field and plasma currents, as well as "solving" itself by finding the equilibrium
    fields, and perturbing those fields to find nearby equilibria.

    Parameters
    ----------
    Psi : float (optional)
        total toroidal flux (in Webers) within LCFS. Default 1.0
    NFP : int (optional)
        number of field periods Default ``surface.NFP`` or 1
    L : int (optional)
        Radial resolution. Default 2*M for ``spectral_indexing=='fringe'``, else M
    M : int (optional)
        Poloidal resolution. Default surface.M or 1
    N : int (optional)
        Toroidal resolution. Default surface.N or 0
    L_grid : int (optional)
        resolution of real space nodes in radial direction
    M_grid : int (optional)
        resolution of real space nodes in poloidal direction
    N_grid : int (optional)
        resolution of real space nodes in toroidal direction
    node_pattern : str (optional)
        pattern of nodes in real space. Default is ``'jacobi'``
    pressure : Profile or ndarray shape(k,2) (optional)
        Pressure profile or array of mode numbers and spectral coefficients.
        Default is a PowerSeriesProfile with zero pressure
    iota : Profile or ndarray shape(k,2) (optional)
        Rotational transform profile or array of mode numbers and spectral coefficients
        Default is a PowerSeriesProfile with zero rotational transform
    surface: Surface or ndarray shape(k,5) (optional)
        Fixed boundary surface shape, as a Surface object or array of
        spectral mode numbers and coefficients of the form [l, m, n, R, Z].
        Default is a FourierRZToroidalSurface with major radius 10 and
        minor radius 1
    axis : Curve or ndarray shape(k,3) (optional)
        Initial guess for the magnetic axis as a Curve object or ndarray
        of mode numbers and spectral coefficints of the form [n, R, Z].
        Default is the centroid of the surface.
    sym : bool (optional)
        Whether to enforce stellarator symmetry. Default surface.sym or False.
    spectral_indexing : str (optional)
        Type of Zernike indexing scheme to use. Default ``'ansi'``
    objective : str or ObjectiveFunction (optional)
        function to solve for equilibrium solution
    optimizer : str or Optimzer (optional)
        optimizer to use
    """

    _io_attrs_ = _Configuration._io_attrs_ + [
        "_solved",
        "_L_grid",
        "_M_grid",
        "_N_grid",
        "_node_pattern",
    ]

    def __init__(
        self,
        Psi=1.0,
        NFP=None,
        L=None,
        M=None,
        N=None,
        L_grid=None,
        M_grid=None,
        N_grid=None,
        node_pattern=None,
        pressure=None,
        iota=None,
        surface=None,
        axis=None,
        sym=None,
        spectral_indexing=None,
        objective=None,
        optimizer=None,
        **kwargs,
    ):

        super().__init__(
            Psi,
            NFP,
            L,
            M,
            N,
            pressure,
            iota,
            surface,
            axis,
            sym,
            spectral_indexing,
            **kwargs,
        )

        assert (L_grid is None) or (
            isinstance(L_grid, numbers.Real)
            and (L_grid == int(L_grid))
            and (L_grid >= 0)
        ), "L_grid should be a non-negative integer or None, got {L_grid}"
        assert (M_grid is None) or (
            isinstance(M_grid, numbers.Real)
            and (M_grid == int(M_grid))
            and (M_grid >= 0)
        ), "M_grid should be a non-negative integer or None, got {M_grid}"
        assert (N_grid is None) or (
            isinstance(N_grid, numbers.Real)
            and (N_grid == int(N_grid))
            and (N_grid >= 0)
        ), "N_grid should be a non-negative integer or None, got {N_grid}"
        self._L_grid = L_grid if L_grid is not None else 2 * self.L
        self._M_grid = M_grid if M_grid is not None else 2 * self.M
        self._N_grid = N_grid if N_grid is not None else 2 * self.N
        self._node_pattern = node_pattern if node_pattern is not None else "jacobi"
        self._solved = False
        self.optimizer_results = {}

    def __repr__(self):
        """String form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, sym={}, spectral_indexing={})".format(
                self.L, self.M, self.N, self.NFP, self.sym, self.spectral_indexing
            )
        )

    @property
    def L_grid(self):
        """Radial resolution of grid in real space (int)."""
        if not hasattr(self, "_L_grid"):
            self._L_grid = (
                self.M_grid if self.spectral_indexing == "ansi" else 2 * self.M_grid
            )
        return self._L_grid

    @L_grid.setter
    def L_grid(self, L_grid):
        if self.L_grid != L_grid:
            self._L_grid = L_grid

    @property
    def M_grid(self):
        """Poloidal resolution of grid in real space (int)."""
        if not hasattr(self, "_M_grid"):
            self._M_grid = 1
        return self._M_grid

    @M_grid.setter
    def M_grid(self, M_grid):
        if self.M_grid != M_grid:
            self._M_grid = M_grid

    @property
    def N_grid(self):
        """Toroidal resolution of grid in real space (int)."""
        if not hasattr(self, "_N_grid"):
            self._N_grid = 0
        return self._N_grid

    @N_grid.setter
    def N_grid(self, N_grid):
        if self.N_grid != N_grid:
            self._N_grid = N_grid

    @property
    def node_pattern(self):
        """Pattern for placement of nodes in curvilinear coordinates (str)."""
        if not hasattr(self, "_node_pattern"):
            self._node_pattern = None
        return self._node_pattern

    @property
    def solved(self):
        """Whether the equilibrium has been solved (bool)."""
        return self._solved

    @solved.setter
    def solved(self, solved):
        self._solved = solved

    def resolution(self):
        return {
            "L": self.L,
            "M": self.M,
            "N": self.N,
            "L_grid": self.L_grid,
            "M_grid": self.M_grid,
            "N_grid": self.N_grid,
        }

    def resolution_summary(self):
        """Print a summary of the spectral and real space resolution."""
        print("Spectral indexing: {}".format(self.spectral_indexing))
        print("Spectral resolution (L,M,N)=({},{},{})".format(self.L, self.M, self.N))
        print("Node pattern: {}".format(self.node_pattern))
        print(
            "Node resolution (L,M,N)=({},{},{})".format(
                self.L_grid, self.M_grid, self.N_grid
            )
        )

    def change_resolution(
        self, L=None, M=None, N=None, L_grid=None, M_grid=None, N_grid=None, NFP=None
    ):
        """Set the spectral resolution and real space grid resolution.

        Parameters
        ----------
        L : int
            maximum radial zernike mode number.
        M : int
            maximum poloidal fourier mode number.
        N : int
            maximum toroidal fourier mode number.
        L_grid : int
            radial real space grid resolution.
        M_grid : int
            poloidal real space grid resolution.
        N_grid : int
            toroidal real space grid resolution.
        NFP : int
            number of field periods.

        """
        L_change = M_change = N_change = NFP_change = False
        if L is not None and L != self.L:
            L_change = True
        if M is not None and M != self.M:
            M_change = True
        if N is not None and N != self.N:
            N_change = True
        if NFP is not None and NFP != self.NFP:
            NFP_change = True

        if any([L_change, M_change, N_change, NFP_change]):
            super().change_resolution(L, M, N, NFP)

        if L_grid is not None and L_grid != self.L_grid:
            self._L_grid = L_grid
        if M_grid is not None and M_grid != self.M_grid:
            self._M_grid = M_grid
        if N_grid is not None and N_grid != self.N_grid:
            self._N_grid = N_grid

    def solve(
        self,
        objective="force",
        constraints=None,
        optimizer="lsq-exact",
        ftol=1e-2,
        xtol=1e-4,
        gtol=1e-6,
        maxiter=50,
        x_scale="auto",
        options={},
        verbose=1,
        copy=False,
    ):
        """Solve to find the equilibrium configuration.

        Parameters
        ----------
        objective : {"force", "force2", "energy"}
            Objective function to solve. Default = force balance on unified grid.
        constraints : Tuple
            set of constraints to enforce. Default = fixed boundary/profiles
        optimizer : string
            Optimization algorithm. Default = "lsq-exact".
        ftol : float
            Relative stopping tolerance on objective function value.
        xtol : float
            Stopping tolerance on step size.
        gtol : float
            Stopping tolerance on norm of gradient.
        maxiter : int
            Maximum number of solver steps.
        x_scale : array_like or ``'auto'``, optional
            Characteristic scale of each variable. Setting ``x_scale`` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting ``x_scale`` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to ``'auto'``, the scale is iteratively updated using the
            inverse norms of the columns of the jacobian or hessian matrix.
        options : dict
            Dictionary of additional options to pass to optimizer.
        verbose : int
            Level of output.
        copy : bool
            Whether to return the current equilibrium or a copy (leaving the original
            unchanged).

        Returns
        -------
        eq : Equilibrium
            Either this equilibrium or a copy, depending on "copy" argument.
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.


        """
        if not isinstance(objective, ObjectiveFunction):
            objective = get_equilibrium_objective(objective)
        if constraints is None:
            constraints = get_fixed_boundary_constraints()
        if not isinstance(optimizer, Optimizer):
            optimizer = Optimizer("lsq-exact")

        if copy:
            eq = self.copy()
        else:
            eq = self

        if eq.N > eq.N_grid or eq.M > eq.M_grid or eq.L > eq.L_grid:
            warnings.warn(
                colored(
                    "Equilibrium has one or more spectral resolutions "
                    + "less than the corresponding collocation grid resolution! "
                    + "This is not recommended and may result in poor convergence. "
                    + "Set grid resolutions to be higher,( i.e. like eq.N_grid=2*eq.N ) "
                    + "To avoid this warning. "
                    "yellow",
                )
            )
        if eq.bdry_mode == "poincare":
            raise NotImplementedError(
                f"Solving equilibrium with poincare XS as BC is not supported yet on master branch."
            )

        result = optimizer.optimize(
            eq,
            objective,
            constraints,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            verbose=verbose,
            maxiter=maxiter,
            options=options,
        )

        if verbose > 0:
            print("Start of solver")
            objective.callback(objective.x(eq))
        for key, value in result["history"].items():
            setattr(eq, key, value[-1])
        if verbose > 0:
            print("End of solver")
            objective.callback(objective.x(eq))

        eq.solved = result["success"]
        return eq, result

    def optimize(
        self,
        objective=None,
        constraints=None,
        optimizer=None,
        ftol=1e-2,
        xtol=1e-4,
        gtol=1e-6,
        maxiter=50,
        x_scale="auto",
        options={},
        verbose=1,
        copy=False,
    ):
        """Optimize an equilibrium for an objective.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to optimize.
        constraint : Objective or tuple of Objective
            Objective function to satisfy. Default = fixed-boundary force balance.
        optimizer : Optimizer
            Optimization algorithm. Default = lsq-exact.
        ftol : float
            Relative stopping tolerance on objective function value.
        xtol : float
            Stopping tolerance on step size.
        gtol : float
            Stopping tolerance on norm of gradient.
        maxiter : int
            Maximum number of solver steps.
        x_scale : array_like or ``'auto'``, optional
            Characteristic scale of each variable. Setting ``x_scale`` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting ``x_scale`` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to ``'auto'``, the scale is iteratively updated using the
            inverse norms of the columns of the jacobian or hessian matrix.
        options : dict
            Dictionary of additional options to pass to optimizer.
        verbose : int
            Level of output.
        copy : bool
            Whether to return the current equilibrium or a copy (leaving the original
            unchanged).

        Returns
        -------
        eq : Equilibrium
            Either this equilibrium or a copy, depending on "copy" argument.
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        """
        if optimizer is None:
            optimizer = Optimizer("lsq-exact")
        if constraints is None:
            constraints = get_fixed_boundary_constraints()
            constraints = (ForceBalance(), *constraints)

        if copy:
            eq = self.copy()
        else:
            eq = self

        result = optimizer.optimize(
            eq,
            objective,
            constraints,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale=x_scale,
            verbose=verbose,
            maxiter=maxiter,
            options=options,
        )

        if verbose > 0:
            print("Start of solver")
            objective.callback(objective.x(eq))
        for key, value in result["history"].items():
            setattr(eq, key, value[-1])
        if verbose > 0:
            print("End of solver")
            objective.callback(objective.x(eq))

        eq.solved = result["success"]
        return eq, result

    def _optimize(
        self,
        objective,
        constraint=None,
        ftol=1e-6,
        xtol=1e-6,
        maxiter=50,
        verbose=1,
        copy=False,
        solve_options={},
        perturb_options={},
    ):
        """Optimize an equilibrium for an objective.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to optimize.
        constraint : ObjectiveFunction
            Objective function to satisfy. Default = fixed-boundary force balance.
        ftol : float
            Relative stopping tolerance on objective function value.
        xtol : float
            Stopping tolerance on optimization step size.
        maxiter : int
            Maximum number of optimization steps.
        verbose : int
            Level of output.
        copy : bool, optional
            Whether to update the existing equilibrium or make a copy (Default).
        solve_options : dict
            Dictionary of additional options used in Equilibrium.solve().
        perturb_options : dict
            Dictionary of additional options used in Equilibrium.perturb().

        Returns
        -------
        eq_new : Equilibrium
            Optimized equilibrum.

        """
        import inspect
        from copy import deepcopy
        from desc.perturbations import optimal_perturb
        from desc.optimize.utils import check_termination
        from desc.optimize.tr_subproblems import update_tr_radius

        if constraint is None:
            constraint = get_equilibrium_objective()

        timer = Timer()
        timer.start("Total time")

        eq = self
        if not objective.built:
            objective.build(eq)
        if not constraint.built:
            constraint.build(eq)

        cost = objective.compute_scalar(objective.x(eq))
        perturb_options = deepcopy(perturb_options)
        tr_ratio = perturb_options.get(
            "tr_ratio",
            inspect.signature(optimal_perturb).parameters["tr_ratio"].default,
        )

        if verbose > 0:
            objective.callback(objective.x(eq))

        iteration = 1
        success = None
        while success is None:

            timer.start("Step {} time".format(iteration))
            if verbose > 0:
                print("====================")
                print("Optimization Step {}".format(iteration))
                print("====================")
                print("Trust-Region ratio = {:9.3e}".format(tr_ratio[0]))

            # perturb + solve
            (
                eq_new,
                predicted_reduction,
                dc_opt,
                dc,
                c_norm,
                bound_hit,
            ) = optimal_perturb(
                eq,
                constraint,
                objective,
                copy=True,
                **perturb_options,
            )
            eq_new.solve(objective=constraint, **solve_options)

            # update trust region radius
            cost_new = objective.compute_scalar(objective.x(eq_new))
            actual_reduction = cost - cost_new
            trust_radius, ratio = update_tr_radius(
                tr_ratio[0] * c_norm,
                actual_reduction,
                predicted_reduction,
                np.linalg.norm(dc_opt),
                bound_hit,
            )
            tr_ratio[0] = trust_radius / c_norm
            perturb_options["tr_ratio"] = tr_ratio

            timer.stop("Step {} time".format(iteration))
            if verbose > 0:
                objective.callback(objective.x(eq_new))
                print("Predicted Reduction = {:10.3e}".format(predicted_reduction))
                print("Reduction Ratio = {:+.3f}".format(ratio))
            if verbose > 1:
                timer.disp("Step {} time".format(iteration))

            # stopping criteria
            success, message = check_termination(
                actual_reduction,
                cost,
                np.linalg.norm(dc),
                c_norm,
                np.inf,  # TODO: add g_norm
                ratio,
                ftol,
                xtol,
                0,  # TODO: add gtol
                iteration,
                maxiter,
                0,
                np.inf,
                0,
                np.inf,
                0,
                np.inf,
            )
            if actual_reduction > 0:
                eq = eq_new
                cost = cost_new
            if success is not None:
                break

            iteration += 1

        timer.stop("Total time")
        print("====================")
        print("Done")
        if verbose > 0:
            print(message)
        if verbose > 1:
            timer.disp("Total time")

        if copy:
            return eq
        else:
            for attr in self._io_attrs_:
                setattr(self, attr, getattr(eq, attr))
            return self

    def perturb(
        self,
        objective=None,
        constraints=None,
        dR=None,
        dZ=None,
        dL=None,
        dRb=None,
        dZb=None,
        dp=None,
        di=None,
        dPsi=None,
        order=2,
        tr_ratio=0.1,
        verbose=1,
        copy=False,
    ):
        """Perturb an equilibrium.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to satisfy. Default = force balance.
        constraint : Objective or tuple of Objective
            Constraint function to satisfy. Default = fixed-boundary.
        dR, dZ, dL, dRb, dZb, dp, di, dPsi : ndarray or float
            Deltas for perturbations of R, Z, lambda, R_boundary, Z_boundary, pressure,
            rotational transform, and total toroidal magnetic flux.
            Setting to None or zero ignores that term in the expansion.
        order : {0,1,2,3}
            Order of perturbation (0=none, 1=linear, 2=quadratic, etc.)
        tr_ratio : float or array of float
            Radius of the trust region, as a fraction of ||x||.
            Enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||.
            If a scalar, uses the same ratio for all steps. If an array, uses the first
            element for the first step and so on.
        verbose : int
            Level of output.
        copy : bool, optional
            Whether to update the existing equilibrium or make a copy (Default).

        Returns
        -------
        eq_new : Equilibrium
            Perturbed equilibrum.

        """
        if objective is None:
            objective = get_equilibrium_objective()
        if constraints is None:
            constraints = get_fixed_boundary_constraints()

        if not objective.built:
            objective.build(self, verbose=verbose)
        for constraint in constraints:
            if not constraint.built:
                constraint.build(self, verbose=verbose)

        eq = perturb(
            self,
            objective,
            constraints,
            dR=dR,
            dZ=dZ,
            dL=dL,
            dRb=dRb,
            dZb=dZb,
            dp=dp,
            di=di,
            dPsi=dPsi,
            order=order,
            tr_ratio=tr_ratio,
            verbose=verbose,
            copy=copy,
        )
        eq.solved = False

        return eq


class EquilibriaFamily(IOAble, MutableSequence):
    """EquilibriaFamily stores a list of Equilibria.

    Has methods for solving complex equilibria using a multi-grid continuation method.

    Parameters
    ----------
    inputs : dict or list
        either a dictionary of inputs or list of dictionaries. For more information
        see inputs required by ``'Equilibrium'``.
        If solving using continuation method, a list should be given.

    """

    _io_attrs_ = ["_equilibria"]

    def __init__(self, inputs):
        # did we get 1 set of inputs or several?
        if isinstance(inputs, (list, tuple)):
            self.equilibria = [Equilibrium(**inputs[0])]
        else:
            self.equilibria = [Equilibrium(**inputs)]
        self.inputs = inputs

    @staticmethod
    def _format_deltas(inputs, equil):
        """Format the changes in continuation parameters.

        Parameters
        ----------
        inputs : dict
             Dictionary of continuation parameters for next step.
        equil : Equilibrium
            Equilibrium being perturbed.

        Returns
        -------
        deltas : dict
            Dictionary of changes in parameter values.

        """
        deltas = {}
        if equil.bdry_mode == "lcfs":
            s = FourierRZToroidalSurface(
                inputs["surface"][:, 3],
                inputs["surface"][:, 4],
                inputs["surface"][:, 1:3].astype(int),
                inputs["surface"][:, 1:3].astype(int),
                equil.NFP,
                equil.sym,
            )
            s.change_resolution(equil.L, equil.M, equil.N)
            Rb_lmn, Zb_lmn = s.R_lmn, s.Z_lmn
        elif equil.bdry_mode == "poincare":
            raise NotImplementedError(
                f"Specifying poincare XS as BC is not implemented yet on main branch."
            )

        p_l = np.zeros_like(equil.pressure.params)
        i_l = np.zeros_like(equil.iota.params)
        for l, p in inputs["pressure"]:
            idx_p = np.where(equil.pressure.basis.modes[:, 0] == int(l))[0]
            p_l[idx_p] = p
        for l, i in inputs["iota"]:
            idx_i = np.where(equil.iota.basis.modes[:, 0] == int(l))[0]
            i_l[idx_i] = i

        if not np.allclose(Rb_lmn, equil.Rb_lmn):
            deltas["dRb"] = Rb_lmn - equil.Rb_lmn
        if not np.allclose(Zb_lmn, equil.Zb_lmn):
            deltas["dZb"] = Zb_lmn - equil.Zb_lmn
        if not np.allclose(p_l, equil.p_l):
            deltas["dp"] = p_l - equil.p_l
        if not np.allclose(i_l, equil.i_l):
            deltas["di"] = i_l - equil.i_l
        if not np.allclose(inputs["Psi"], equil.Psi):
            deltas["dPsi"] = inputs["Psi"] - equil.Psi
        return deltas

    def _print_iteration(self, ii, equil):
        print("================")
        print("Step {}/{}".format(ii + 1, len(self.inputs)))
        print("================")
        equil.resolution_summary()
        print("Boundary ratio = {}".format(self.inputs[ii]["bdry_ratio"]))
        print("Pressure ratio = {}".format(self.inputs[ii]["pres_ratio"]))
        print("Perturbation Order = {}".format(self.inputs[ii]["pert_order"]))
        print("Objective: {}".format(self.inputs[ii]["objective"]))
        print("Optimizer: {}".format(self.inputs[ii]["optimizer"]))
        print("Function tolerance = {}".format(self.inputs[ii]["ftol"]))
        print("Gradient tolerance = {}".format(self.inputs[ii]["gtol"]))
        print("State vector tolerance = {}".format(self.inputs[ii]["xtol"]))
        print("Max function evaluations = {}".format(self.inputs[ii]["nfev"]))
        print("================")

    def solve_continuation(self, start_from=0, verbose=None, checkpoint_path=None):
        """Solve for an equilibrium by continuation method.

            1. Creates an initial guess from the given inputs
            2. Find equilibrium flux surfaces by minimizing the objective function.
            3. Step up to higher resolution and perturb the previous solution
            4. Repeat 2 and 3 until at desired resolution

        Parameters
        ----------
        start_from : integer
            start solution from the given index
        verbose : integer
            * 0: no output
            * 1: summary of each iteration
            * 2: as above plus timing information
            * 3: as above plus detailed solver output
        checkpoint_path : str or path-like
            file to save checkpoint data (Default value = None)

        """
        timer = Timer()
        if verbose is None:
            verbose = self.inputs[0]["verbose"]
        timer.start("Total time")

        if (
            not (
                isalmostequal([inp["bdry_ratio"] for inp in self.inputs])
                and isalmostequal([inp["pres_ratio"] for inp in self.inputs])
            )
            and not use_jax
        ):
            warnings.warn(
                colored(
                    "Computing perturbations with finite differences can be "
                    + "highly innacurate, consider using JAX or setting all "
                    + "perturbation ratios to 1",
                    "yellow",
                )
            )

        for ii in range(start_from, len(self.inputs)):
            timer.start("Iteration {} total".format(ii + 1))

            # TODO: make this more efficient (minimize re-building)
            optimizer = Optimizer(self.inputs[ii]["optimizer"])
            objective = get_equilibrium_objective(self.inputs[ii]["objective"])
            constraints = get_fixed_boundary_constraints(
                profiles=self.inputs[ii]["objective"] != "vacuum"
            )

            if ii == start_from:
                equil = self[ii]
                if verbose > 0:
                    self._print_iteration(ii, equil)

            else:
                equil = self[ii - 1].copy()
                self.insert(ii, equil)

                equil.change_resolution(
                    L=self.inputs[ii]["L"],
                    M=self.inputs[ii]["M"],
                    N=self.inputs[ii]["N"],
                )
                equil.L_grid = self.inputs[ii]["L_grid"]
                equil.M_grid = self.inputs[ii]["M_grid"]
                equil.N_grid = self.inputs[ii]["N_grid"]

                if verbose > 0:
                    self._print_iteration(ii, equil)

                # figure out if we we need perturbations
                deltas = self._format_deltas(self.inputs[ii], equil)

                if len(deltas) > 0:
                    if verbose > 0:
                        print("Perturbing equilibrium")
                    # TODO: pass Jx if available
                    equil.perturb(
                        objective=objective,
                        constraints=constraints,
                        **deltas,
                        order=self.inputs[ii]["pert_order"],
                        verbose=verbose,
                        copy=False,
                    )

            if not equil.is_nested():
                warnings.warn(
                    colored(
                        "WARNING: Flux surfaces are no longer nested, exiting early."
                        + "Consider taking smaller perturbation/resolution steps "
                        + "or reducing trust radius",
                        "yellow",
                    )
                )
                if checkpoint_path is not None:
                    if verbose > 0:
                        print("Saving latest state")
                    self.save(checkpoint_path)
                break

            equil.solve(
                optimizer=optimizer,
                objective=objective,
                constraints=constraints,
                ftol=self.inputs[ii]["ftol"],
                xtol=self.inputs[ii]["xtol"],
                gtol=self.inputs[ii]["gtol"],
                verbose=verbose,
                maxiter=self.inputs[ii]["nfev"],
            )

            if checkpoint_path is not None:
                if verbose > 0:
                    print("Saving latest iteration")
                self.save(checkpoint_path)
            timer.stop("Iteration {} total".format(ii + 1))
            if verbose > 1:
                timer.disp("Iteration {} total".format(ii + 1))

            if not equil.is_nested():
                warnings.warn(
                    colored(
                        "WARNING: Flux surfaces are no longer nested, exiting early."
                        + "Consider taking smaller perturbation/resolution steps "
                        + "or reducing trust radius",
                        "yellow",
                    )
                )
                break

        timer.stop("Total time")
        print("====================")
        print("Done")
        if verbose > 1:
            timer.disp("Total time")
        if checkpoint_path is not None:
            print("Output written to {}".format(checkpoint_path))
        print("====================")

    @property
    def equilibria(self):
        """List of equilibria contained in the family (list)."""
        return self._equilibria

    @equilibria.setter
    def equilibria(self, equil):
        if isinstance(equil, tuple):
            equil = list(equil)
        elif isinstance(equil, np.ndarray):
            equil = equil.tolist()
        elif not isinstance(equil, list):
            equil = [equil]
        if not np.all([isinstance(eq, Equilibrium) for eq in equil]):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria = list(equil)

    # dunder methods required by MutableSequence

    def __getitem__(self, i):
        return self._equilibria[i]

    def __setitem__(self, i, new_item):
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria[i] = new_item

    def __delitem__(self, i):
        del self._equilibria[i]

    def __len__(self):
        return len(self._equilibria)

    def insert(self, i, new_item):
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria.insert(i, new_item)
