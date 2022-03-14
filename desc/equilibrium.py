import numpy as np
import warnings
import numbers
import inspect
from termcolor import colored
from collections.abc import MutableSequence

from desc.backend import use_jax, put
from desc.utils import Timer, isalmostequal
from desc.configuration import _Configuration
from desc.io import IOAble
from desc.geometry import FourierRZToroidalSurface, ZernikeRZToroidalSection
from desc.optimize.utils import check_termination
from desc.optimize.tr_subproblems import update_tr_radius
from desc.optimize import Optimizer
from desc.objectives import (
    ObjectiveFunction,
    RadialForceBalance,
    HelicalForceBalance,
    Energy,
    get_fixed_boundary_constraints,
)
from desc.perturbations import perturb, optimal_perturb


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
        self._L_grid = L_grid if L_grid is not None else self.L
        self._M_grid = M_grid if M_grid is not None else self.M
        self._N_grid = N_grid if N_grid is not None else self.N
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

    def solve(
        self,
        optimizer=None,
        objective=None,
        ftol=1e-2,
        xtol=1e-4,
        gtol=1e-6,
        verbose=1,
        x_scale="auto",
        maxiter=50,
        options={},
    ):
        """Solve to find the equilibrium configuration.

        Parameters
        ----------
        optimizer : Optimizer
            Optimization algorithm. Default = lsq-exact.
        objective : ObjectiveFunction
            Objective function to solve. Default = fixed-boundary force balance.
        ftol : float
            Relative stopping tolerance on objective function value.
        xtol : float
            Stopping tolerance on step size.
        gtol : float
            Stopping tolerance on norm of gradient.
        verbose : int
            Level of output.
        maxiter : int
            Maximum number of solver steps.
        options : dict
            Dictionary of additional options to pass to optimizer.

        """
        if optimizer is None:
            optimizer = Optimizer("lsq-exact")
        if objective is None:
            objectives = (RadialForceBalance(), HelicalForceBalance())
            constraints = get_fixed_boundary_constraints()
            objective = ObjectiveFunction(objectives, constraints)
        if not objective.built:
            objective.build(self, verbose=verbose)

        x0 = objective.x(self)
        result = optimizer.optimize(
            objective,
            x_init=x0,
            args=(objective.y0,),
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
            objective.callback(x0)
            print("End of solver")
            objective.callback(result["x"])

        for key, value in objective.unpack_state(result["x"]).items():
            value = put(  # parameter values below threshold are set to 0
                value, np.where(np.abs(value) < 10 * np.finfo(value.dtype).eps)[0], 0
            )
            setattr(self, key, value)
        self.solved = result["success"]
        return result

    def perturb(
        self,
        objective=None,
        dR=None,
        dZ=None,
        dL=None,
        dRb=None,
        dZb=None,
        dp=None,
        di=None,
        dPsi=None,
        order=2,
        tr_ratio=[0.1, 0.25, 0.5],
        verbose=1,
        copy=True,
    ):
        """Perturb an equilibrium.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to satisfy.
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
        copy : bool
            Whether to perturb the input equilibrium or make a copy (Default).

        Returns
        -------
        eq_new : Equilibrium
            perturbed equilibrum, only returned if copy=True

        """
        if objective is None:
            objectives = (RadialForceBalance(), HelicalForceBalance())
            constraints = get_fixed_boundary_constraints()
            objective = ObjectiveFunction(objectives, constraints)

        eq = perturb(
            self,
            objective,
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

        if copy:
            return eq
        else:
            return None

    def optimize(
        self,
        objective,
        constraint=None,
        ftol=1e-6,
        xtol=1e-6,
        maxiter=50,
        verbose=1,
        copy=True,
        solve_options={},
        perturb_options={},
    ):
        """Optimize an equilibrium for an objective.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to optimize.
        constraint : ObjectiveFunction
            Objective function to satisfy.

        ftol : float
            Relative stopping tolerance on objective function value.
        xtol : float
            Stopping tolerance on optimization step size.
        maxiter : int
            Maximum number of optimization steps.

        verbose : int
            Level of output.
        copy : bool
            Whether to perturb the input equilibrium or make a copy (Default).

        solve_options : dict
            Dictionary of additional options used in Equilibrium.solve().
        perturb_options : dict
            Dictionary of additional options used in Equilibrium.perturb().

        Returns
        -------
        eq_new : Equilibrium
            perturbed equilibrum, only returned if copy=True

        """
        if constraint is None:
            objectives = (RadialForceBalance(), HelicalForceBalance())
            constraints = get_fixed_boundary_constraints()
            constraint = ObjectiveFunction(objectives, constraints)

        if copy:
            eq = self.copy()
        else:
            eq = self

        timer = Timer()
        timer.start("Total time")

        if not objective.built:
            objective.build(self)
        cost = objective.compute_scalar(objective.y(eq))

        tr_ratio = perturb_options.get(
            "tr_ratio",
            inspect.signature(optimal_perturb).parameters["tr_ratio"].default,
        )

        if verbose > 0:
            objective.callback(objective.y(eq))

        for iteration in range(maxiter):
            timer.start("Step {} time".format(iteration + 1))
            if verbose > 0:
                print("====================")
                print("Optimization Step {}".format(iteration + 1))
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
            cost_new = objective.compute_scalar(objective.y(eq_new))
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

            timer.stop("Step {} time".format(iteration + 1))
            if verbose > 0:
                objective.callback(objective.y(eq_new))
                print("Predicted Reduction = {:10.3e}".format(predicted_reduction))
                print("Reduction Ratio = {:+.3f}".format(ratio))
            if verbose > 1:
                timer.disp("Step {} time".format(iteration + 1))

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
            self = eq
            return None


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
            s = ZernikeRZToroidalSection(
                inputs["surface"][:, 3],
                inputs["surface"][:, 4],
                inputs["surface"][:, :2].astype(int),
                inputs["surface"][:, :2].astype(int),
                equil.spectral_indexing,
                equil.sym,
            )
            s.change_resolution(equil.L, equil.M, equil.N)
            Rb_lmn, Zb_lmn = s.R_lmn, s.Z_lmn

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
            if self.inputs[ii]["objective"] == "force":
                objectives = (RadialForceBalance(), HelicalForceBalance())
            elif self.inputs[ii]["objective"] == "energy":
                objectives = Energy()
            constraints = get_fixed_boundary_constraints()
            objective = ObjectiveFunction(objectives, constraints)

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
