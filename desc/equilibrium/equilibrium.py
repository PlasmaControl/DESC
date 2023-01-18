"""Core class representing MHD equilibrium."""

import numbers
import warnings
from collections.abc import MutableSequence

import numpy as np
from scipy import special
from scipy.constants import mu_0
from termcolor import colored

from desc.basis import FourierZernikeBasis
from desc.geometry import FourierRZCurve
from desc.grid import LinearGrid
from desc.io import IOAble
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    get_equilibrium_objective,
    get_fixed_boundary_constraints,
)
from desc.optimize import Optimizer
from desc.perturbations import perturb
from desc.transform import Transform
from desc.utils import Timer

from .configuration import _Configuration


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
    current : Profile or ndarray shape(k,2) (optional)
        Toroidal current profile or array of mode numbers and spectral coefficients
        Default is a PowerSeriesProfile with zero toroidal current
    surface: Surface or ndarray shape(k,5) (optional)
        Fixed boundary surface shape, as a Surface object or array of
        spectral mode numbers and coefficients of the form [l, m, n, R, Z].
        Default is a FourierRZToroidalSurface with major radius 10 and minor radius 1
    axis : Curve or ndarray shape(k,3) (optional)
        Initial guess for the magnetic axis as a Curve object or ndarray
        of mode numbers and spectral coefficients of the form [n, R, Z].
        Default is the centroid of the surface.
    sym : bool (optional)
        Whether to enforce stellarator symmetry. Default surface.sym or False.
    spectral_indexing : str (optional)
        Type of Zernike indexing scheme to use. Default ``'ansi'``
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
        current=None,
        surface=None,
        axis=None,
        sym=None,
        spectral_indexing=None,
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
            current,
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
        """int: Radial resolution of grid in real space."""
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
        """int: Poloidal resolution of grid in real space."""
        if not hasattr(self, "_M_grid"):
            self._M_grid = 1
        return self._M_grid

    @M_grid.setter
    def M_grid(self, M_grid):
        if self.M_grid != M_grid:
            self._M_grid = M_grid

    @property
    def N_grid(self):
        """int: Toroidal resolution of grid in real space."""
        if not hasattr(self, "_N_grid"):
            self._N_grid = 0
        return self._N_grid

    @N_grid.setter
    def N_grid(self, N_grid):
        if self.N_grid != N_grid:
            self._N_grid = N_grid

    @property
    def node_pattern(self):
        """str: Pattern for placement of nodes in curvilinear coordinates."""
        if not hasattr(self, "_node_pattern"):
            self._node_pattern = None
        return self._node_pattern

    @property
    def solved(self):
        """bool: Whether the equilibrium has been solved."""
        return self._solved

    @solved.setter
    def solved(self, solved):
        self._solved = solved

    @property
    def resolution(self):
        """dict: Spectral and real space resolution parameters of the Equilibrium."""
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

    @classmethod
    def from_near_axis(
        cls, na_eq, r=0.1, L=None, M=8, N=None, ntheta=None, spectral_indexing="ansi"
    ):
        """Initialize an Equilibrium from a near-axis solution.

        Parameters
        ----------
        na_eq : Qsc or Qic
            Near-axis solution generated by pyQSC or pyQIC.
        r : float
            Radius of the desired boundary surface (in meters).
        L : int (optional)
            Radial resolution. Default 2*M for `spectral_indexing`==fringe, else M
        M : int (optional)
            Poloidal resolution. Default is 8
        N : int (optional)
            Toroidal resolution. Default is M.
            If N=np.inf, the max resolution provided by na_eq.nphi is used.
        ntheta : int, optional
            Number of poloidal grid points used in the conversion. Default 2*M+1
        spectral_indexing : str (optional)
            Type of Zernike indexing scheme to use. Default ``'ansi'``

        Returns
        -------
        eq : Equilibrium
            Equilibrium approximation of the near-axis solution.

        """
        try:
            # default resolution parameters
            if L is None:
                if spectral_indexing == "ansi":
                    L = M
                elif spectral_indexing == "fringe":
                    L = 2 * M
            if N is None:
                N = M
            if N == np.inf:
                N = int((na_eq.nphi - 1) / 2)

            if ntheta is None:
                ntheta = 2 * M + 1

            inputs = {}
            inputs["Psi"] = np.pi * r**2 * na_eq.spsi * na_eq.Bbar
            inputs["NFP"] = na_eq.nfp
            inputs["L"] = L
            inputs["M"] = M
            inputs["N"] = N
            inputs["sym"] = not na_eq.lasym
            inputs["spectral_indexing "] = spectral_indexing
            inputs["pressure"] = np.array(
                [[0, -na_eq.p2 * r**2], [2, na_eq.p2 * r**2]]
            )
            inputs["iota"] = None
            inputs["current"] = np.array([[2, 2 * np.pi / mu_0 * na_eq.I2 * r**2]])
            inputs["axis"] = FourierRZCurve(
                R_n=np.concatenate((np.flipud(na_eq.rs[1:]), na_eq.rc)),
                Z_n=np.concatenate((np.flipud(na_eq.zs[1:]), na_eq.zc)),
                NFP=na_eq.nfp,
            )
            inputs["surface"] = None
        except AttributeError as e:
            raise ValueError("Input must be a pyQSC or pyQIC solution.") from e

        rho, _ = special.js_roots(L, 2, 2)
        grid = LinearGrid(rho=rho, theta=ntheta, zeta=na_eq.nphi, NFP=na_eq.nfp)
        basis_R = FourierZernikeBasis(
            L=L,
            M=M,
            N=N,
            NFP=na_eq.nfp,
            sym="cos" if not na_eq.lasym else False,
            spectral_indexing=spectral_indexing,
        )
        basis_Z = FourierZernikeBasis(
            L=L,
            M=M,
            N=N,
            NFP=na_eq.nfp,
            sym="sin" if not na_eq.lasym else False,
            spectral_indexing=spectral_indexing,
        )
        transform_R = Transform(grid, basis_R, build_pinv=True)
        transform_Z = Transform(grid, basis_Z, build_pinv=True)

        R_1D = np.zeros((grid.num_nodes,))
        Z_1D = np.zeros((grid.num_nodes,))

        for rho_i in rho:
            idx = idx = np.where(grid.nodes[:, 0] == rho_i)[0]
            R_2D, Z_2D, _ = na_eq.Frenet_to_cylindrical(r * rho_i, ntheta)
            R_1D[idx] = R_2D.flatten(order="F")
            Z_1D[idx] = Z_2D.flatten(order="F")

        inputs["R_lmn"] = transform_R.fit(R_1D)
        inputs["Z_lmn"] = transform_Z.fit(Z_1D)
        inputs["L_lmn"] = np.zeros_like(inputs["Z_lmn"])

        eq = Equilibrium(**inputs)
        eq.surface = eq.get_surface_at(rho=1)

        return eq

    def solve(
        self,
        objective="force",
        constraints=None,
        optimizer="lsq-exact",
        ftol=None,
        xtol=None,
        gtol=None,
        maxiter=50,
        x_scale="auto",
        options={},
        verbose=1,
        copy=False,
    ):
        """Solve to find the equilibrium configuration.

        Parameters
        ----------
        objective : {"force", "forces", "energy", "vacuum"}
            Objective function to solve. Default = force balance on unified grid.
        constraints : Tuple
            set of constraints to enforce. Default = fixed boundary/profiles
        optimizer : str or Optimizer (optional)
            optimizer to use
        ftol, xtol, gtol : float
            stopping tolerances. `None` will use defaults for given optimizer.
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
            inverse norms of the columns of the Jacobian or Hessian matrix.
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
        if constraints is None:
            constraints = get_fixed_boundary_constraints(
                iota=objective != "vacuum" and self.iota is not None
            )
        if not isinstance(objective, ObjectiveFunction):
            objective = get_equilibrium_objective(objective)
        if not isinstance(optimizer, Optimizer):
            optimizer = Optimizer(optimizer)

        if copy:
            eq = self.copy()
        else:
            eq = self

        if eq.N > eq.N_grid or eq.M > eq.M_grid or eq.L > eq.L_grid:
            warnings.warn(
                colored(
                    "Equilibrium has one or more spectral resolutions "
                    + "greater than the corresponding collocation grid resolution! "
                    + "This is not recommended and may result in poor convergence. "
                    + "Set grid resolutions to be higher, (i.e. eq.N_grid=2*eq.N) "
                    + "to avoid this warning.",
                    "yellow",
                )
            )
        if eq.bdry_mode == "poincare":
            raise NotImplementedError(
                "Solving equilibrium with poincare XS as BC is not supported yet "
                + "on master branch."
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
            objective.print_value(objective.x(eq))
        for key, value in result["history"].items():
            # don't set nonexistent profile (values are empty ndarrays)
            if not (key == "c_l" or key == "i_l") or value[-1].size:
                setattr(eq, key, value[-1])

        if verbose > 0:
            print("End of solver")
            objective.print_value(objective.x(eq))

        eq.solved = result["success"]
        return eq, result

    def optimize(
        self,
        objective=None,
        constraints=None,
        optimizer="lsq-exact",
        ftol=None,
        xtol=None,
        gtol=None,
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
        constraints : Objective or tuple of Objective
            Objective function to satisfy. Default = fixed-boundary force balance.
        optimizer : str or Optimizer (optional)
            optimizer to use
        ftol, xtol, gtol : float
            stopping tolerances. `None` will use defaults for given optimizer.
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
            inverse norms of the columns of the Jacobian or Hessian matrix.
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
        if not isinstance(optimizer, Optimizer):
            optimizer = Optimizer(optimizer)
        if constraints is None:
            constraints = get_fixed_boundary_constraints(iota=self.iota is not None)
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
            objective.print_value(objective.x(eq))
        for key, value in result["history"].items():
            # don't set nonexistent profile (values are empty ndarrays)
            if not (key == "c_l" or key == "i_l") or value[-1].size:
                setattr(eq, key, value[-1])
        if verbose > 0:
            print("End of solver")
            objective.print_value(objective.x(eq))

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
            Optimized equilibrium.

        """
        import inspect
        from copy import deepcopy

        from desc.optimize.tr_subproblems import update_tr_radius
        from desc.optimize.utils import check_termination
        from desc.perturbations import optimal_perturb

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
            objective.print_value(objective.x(eq))

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
                objective.print_value(objective.x(eq_new))
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
        dc=None,
        dPsi=None,
        order=2,
        tr_ratio=0.1,
        weight="auto",
        include_f=True,
        verbose=1,
        copy=False,
    ):
        """Perturb an equilibrium.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to satisfy. Default = force balance.
        constraints : Objective or tuple of Objective
            Constraint function to satisfy. Default = fixed-boundary.
        dR, dZ, dL, dRb, dZb, dp, di, dc, dPsi : ndarray or float
            Deltas for perturbations of R, Z, lambda, R_boundary, Z_boundary, pressure,
            rotational transform, toroidal current, and total toroidal magnetic flux.
            Setting to None or zero ignores that term in the expansion.
        order : {0,1,2,3}
            Order of perturbation (0=none, 1=linear, 2=quadratic, etc.)
        tr_ratio : float or array of float
            Radius of the trust region, as a fraction of ||x||.
            Enforces ||dx1|| <= tr_ratio*||x|| and ||dx2|| <= tr_ratio*||dx1||.
            If a scalar, uses the same ratio for all steps. If an array, uses the first
            element for the first step and so on.
        weight : ndarray, "auto", or None, optional
            1d or 2d array for weighted least squares. 1d arrays are turned into
            diagonal matrices. Default is to weight by (mode number)**2. None applies
            no weighting.
        include_f : bool, optional
            Whether to include the 0th order objective residual in the perturbation
            equation. Including this term can improve force balance if the perturbation
            step is large, but can result in too large a step if the perturbation
            is small.
        verbose : int
            Level of output.
        copy : bool, optional
            Whether to update the existing equilibrium or make a copy (Default).

        Returns
        -------
        eq_new : Equilibrium
            Perturbed equilibrium.

        """
        if objective is None:
            objective = get_equilibrium_objective()
        if constraints is None:
            constraints = get_fixed_boundary_constraints(iota=self.iota is not None)

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
            dc=dc,
            dPsi=dPsi,
            order=order,
            tr_ratio=tr_ratio,
            weight=weight,
            include_f=include_f,
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
    args : Equilibrium, dict or list of dict
        Should be either:
          * An Equilibrium (or several)
          * A dictionary of inputs (or several) to create a equilibria
          * A single list of dictionaries, one for each equilibrium in a continuation.
          * Nothing, to create an empty family.
        For more information see inputs required by ``'Equilibrium'``.
    """

    _io_attrs_ = ["_equilibria"]

    def __init__(self, *args):

        self.equilibria = []
        if len(args) == 1 and isinstance(args[0], list):
            for inp in args[0]:
                self.equilibria.append(Equilibrium(**inp))
        else:
            for arg in args:
                if isinstance(arg, Equilibrium):
                    self.equilibria.append(arg)
                elif isinstance(arg, dict):
                    self.equilibria.append(Equilibrium(**arg))
                else:
                    raise TypeError(
                        "Args to create EquilibriaFamily should either be "
                        + "Equilibrium or dictionary"
                    )

    def solve_continuation(
        self,
        objective="force",
        optimizer="lsq-exact",
        pert_order=2,
        ftol=None,
        xtol=None,
        gtol=None,
        nfev=100,
        verbose=1,
        checkpoint_path=None,
    ):
        """Solve for an equilibrium by continuation method.

        Steps through an EquilibriaFamily, solving each equilibrium, and uses
        pertubations to step between different profiles/boundaries.

        Uses the previous step as an initial guess for each solution.

        Parameters
        ----------
        eqfam : EquilibriaFamily or list of Equilibria
            Equilibria to solve for at each step.
        objective : str or ObjectiveFunction (optional)
            function to solve for equilibrium solution
        optimizer : str or Optimzer (optional)
            optimizer to use
        pert_order : int or array of int
            order of perturbations to use. If array-like, should be same length as
            family to specify different values for each step.
        ftol, xtol, gtol : float or array-like of float
            stopping tolerances for subproblem at each step. `None` will use defaults
            for given optimizer.
        nfev : int or array-like of int
            maximum number of function evaluations in each equilibrium subproblem.
        verbose : integer
            * 0: no output
            * 1: summary of each iteration
            * 2: as above plus timing information
            * 3: as above plus detailed solver output
        checkpoint_path : str or path-like
            file to save checkpoint data (Default value = None)

        Returns
        -------
        eqfam : EquilibriaFamily
            family of equilibria for the intermediate steps, where the last member is
            the final desired configuration,

        """
        from desc.continuation import solve_continuation

        return solve_continuation(
            self,
            objective,
            optimizer,
            pert_order,
            ftol,
            xtol,
            gtol,
            nfev,
            verbose,
            checkpoint_path,
        )

    @classmethod
    def solve_continuation_automatic(
        cls,
        eq,
        objective="force",
        optimizer="lsq-exact",
        pert_order=2,
        ftol=None,
        xtol=None,
        gtol=None,
        nfev=100,
        verbose=1,
        checkpoint_path=None,
        **kwargs,
    ):
        """Solve for an equilibrium using an automatic continuation method.

        By default, the method first solves for a no pressure tokamak, then a finite
        beta tokamak, then a finite beta stellarator. Currently hard coded to take a
        fixed number of perturbation steps based on conservative estimates and testing.
        In the future, continuation stepping will be done adaptively.

        Parameters
        ----------
        eq : Equilibrium
            Unsolved Equilibrium with the final desired boundary, profiles, resolution.
        objective : str or ObjectiveFunction (optional)
            function to solve for equilibrium solution
        optimizer : str or Optimzer (optional)
            optimizer to use
        pert_order : int
            order of perturbations to use.
        ftol, xtol, gtol : float
            stopping tolerances for subproblem at each step. `None` will use defaults
            for given optimizer.
        nfev : int
            maximum number of function evaluations in each equilibrium subproblem.
        verbose : integer
            * 0: no output
            * 1: summary of each iteration
            * 2: as above plus timing information
            * 3: as above plus detailed solver output
        checkpoint_path : str or path-like
            file to save checkpoint data (Default value = None)
        **kwargs : control continuation step sizes

            Valid keyword arguments are:

            mres_step: int, the amount to increase Mpol by at each continuation step
            pres_step: float, 0<=pres_step<=1, the amount to increase pres_ratio by
                            at each continuation step
            bdry_step: float, 0<=bdry_step<=1, the amount to increase pres_ratio by
                            at each continuation step
        Returns
        -------
        eqfam : EquilibriaFamily
            family of equilibria for the intermediate steps, where the last member is
            the final desired configuration,

        """
        from desc.continuation import solve_continuation_automatic

        return solve_continuation_automatic(
            eq,
            objective,
            optimizer,
            pert_order,
            ftol,
            xtol,
            gtol,
            nfev,
            verbose,
            checkpoint_path,
            **kwargs,
        )

    @property
    def equilibria(self):
        """list: Equilibria contained in the family."""
        return self._equilibria

    @equilibria.setter
    def equilibria(self, equil):
        if isinstance(equil, tuple):
            equil = list(equil)
        elif isinstance(equil, np.ndarray):
            equil = equil.tolist()
        elif not isinstance(equil, list):
            equil = [equil]
        if len(equil) and not all([isinstance(eq, Equilibrium) for eq in equil]):
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
        """Insert a new Equilibrium into the family at position i."""
        if not isinstance(new_item, Equilibrium):
            raise ValueError(
                "Members of EquilibriaFamily should be of type Equilibrium or subclass."
            )
        self._equilibria.insert(i, new_item)
