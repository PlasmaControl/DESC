"""Objectives for targeting geometrical quantities."""

import warnings

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_profiles, get_transforms, rpz2xyz, rpz2xyz_vec
from desc.compute.utils import safenorm
from desc.grid import LinearGrid
from desc.utils import Timer, warnif

from .objective_funs import _Objective


class QuadraticFlux(_Objective):
    """Target the quadratic flux on an equilibrium from a magnetic field.

    compute

    (B.n)^2

    where n is the normal vector to the plasma surface, and B is the magnetic field at
    the plasma surface.

    NOTE: Only works for vacuum equilibria currently

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    field : MagneticField
        MagneticField object, the parameters of this will be optimized
        to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
        FIXME: add normalization for the B part of this objective
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate field source at on
        the winding surface. (used if e.g. field is a CoilSet or
        FourierCurrentPotentialField)
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at.
    external_field : MagneticField, optional
        MagneticField object containing the external field to consider when
        minimizing the Bn errors. If None, the external field is assumed to be zero.
        e.g. this could be a 1/R field representing external TF coils, or
        it could be set of discrete TF coils so that coil ripple is considered during
        the optimization of the ``field`` object.
    external_field_source_grid : Grid, optional
        Grid object used to discretize the external field source.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = "rtz"
    _units = "T"
    _print_value_fmt = "Quadratic Flux: {:10.3e} "

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        source_grid=None,
        eval_grid=None,
        external_field=None,
        external_field_source_grid=None,
        name="quadratic-flux",
        eq_fixed=False,
    ):
        if target is None and bounds is None:
            target = 0
        self._field = field
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq_fixed = eq_fixed
        self._eq = eq if eq_fixed else None
        self._external_field = external_field
        self._external_field_source_grid = external_field_source_grid

        super().__init__(
            things=[field, eq] if not eq_fixed else [field],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self._eq if self._eq_fixed else self.things[1]
        field = self.things[0]
        # if field is different than self._field, update
        if field != self._field:
            self._field = field
        # if eq is different than self._eq, update
        if eq != self._eq:
            self._eq = eq
        if self._eval_grid is None:
            eval_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            eval_grid = self._eval_grid
        if not np.allclose(eval_grid.nodes[:, 0], 1):
            warnings.warn("Evaluation grid includes interior points, should be rho=1")

        # ensure vacuum eq, as we don't yet support finite beta
        pres = np.max(np.abs(eq.compute("p")["p"]))
        curr = np.max(np.abs(eq.compute("current")["current"]))
        warnif(
            pres > 1e-8,
            UserWarning,
            f"Pressure is non-zero (max {pres} Pa), "
            + "finite beta not supported yet.",
        )
        warnif(
            curr > 1e-8,
            UserWarning,
            f"Current is non-zero (max {curr} A), "
            + "finite plasma currents not supported yet.",
        )

        # eval_grid.num_nodes for quad flux cost,
        self._dim_f = eval_grid.num_nodes
        self._equil_data_keys = ["n_rho", "R", "phi", "Z"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        equil_profiles = get_profiles(
            self._equil_data_keys,
            obj=eq,
            grid=eval_grid,
            has_axis=eval_grid.axis.size,
        )
        equil_transforms = get_transforms(
            self._equil_data_keys,
            obj=eq,
            grid=eval_grid,
            has_axis=eval_grid.axis.size,
        )

        if self._eq_fixed:
            data = eq.compute(["R", "phi", "Z", "n_rho"], grid=eval_grid)

            plasma_coords = rpz2xyz(jnp.array([data["R"], data["phi"], data["Z"]]).T)
            data["n_rho"] = rpz2xyz_vec(
                data["n_rho"], x=plasma_coords[:, 0], y=plasma_coords[:, 1]
            )

        if not self._eq_fixed:
            self._constants = {
                "equil_transforms": equil_transforms,
                "equil_profiles": equil_profiles,
                "quad_weights": eval_grid.weights * jnp.sqrt(eval_grid.num_nodes),
            }
        else:
            self._constants = {
                "equil_transforms": equil_transforms,
                "equil_profiles": equil_profiles,
                "plasma_coords": plasma_coords,
                "equil_data": data,
                "quad_weights": eval_grid.weights * jnp.sqrt(eval_grid.num_nodes),
            }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, field_params=None, equil_params=None, constants=None):
        """Compute quadratic flux.

        Parameters
        ----------
        field_params : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotential.params_dict
        equil_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Normal field (B.n) on the plasma surface due to the contributions from
            the ``field`` being optimized and the ``external_field``.
            NOTE: This will then be squared to form the quadratic flux and minimized
            by the optimizer.

        """
        if constants is None:
            constants = self.constants
        if not self._eq_fixed:
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._equil_data_keys,
                params=equil_params,
                transforms=constants["equil_transforms"],
                profiles=constants["equil_profiles"],
            )
            plasma_coords = rpz2xyz(jnp.array([data["R"], data["phi"], data["Z"]]).T)
            data["n_rho"] = rpz2xyz_vec(
                data["n_rho"], x=plasma_coords[:, 0], y=plasma_coords[:, 1]
            )

        else:
            data = constants["equil_data"]
            plasma_coords = constants["plasma_coords"]

        B = self._field.compute_magnetic_field(
            plasma_coords, basis="xyz", grid=self._source_grid, params=field_params
        )

        Bn = jnp.sum(B * data["n_rho"], axis=-1)

        if self._external_field is not None:
            B_ext = self._external_field.compute_magnetic_field(
                plasma_coords,
                grid=self._external_field_source_grid,
                basis="xyz",
            )
            Bn += jnp.sum(B_ext * data["n_rho"], axis=-1)

        return Bn


class SurfaceCurrentRegularization(_Objective):
    """Target the surface current magnitude.

    compute::

        w*(|K|)^2

    where K is the winding surface current density, and w is the
    regularization parameter (the weight on this objective)

    This is intended to be used with a surface current::

        K = n x ∇ Φ
        Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    i.e. a FourierCurrentPotentialField

    Intended to be used with a QuadraticFlux objective, to form
    the REGCOIL algorithm described in [1]_.

    [1] Landreman, An improved current potential method for fast computation
        of stellarator coil shapes, Nuclear Fusion (2017)

    Parameters
    ----------
    surface_current_field : FourierCurrentPotentialField
        Surface current which is producing the magnetic field, the parameters
        of this will be optimized to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
        When used with QuadraticFlux objective, this acts as the regularization
        parameter, with 0 corresponding to no regularization. The larger this
        parameter is, the less complex the surface current will be, but the
        worse the normal field.
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate current source at on
        the winding surface. If used in conjunction with the QuadraticFlux objective,
        with the same ``source_grid``, this replicates the REGCOIL algorithm described
        in [1]_.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Surface Current Regularization: {:10.3e} "

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        source_grid=None,
        name="surface-current-regularization",
    ):
        if target is None and bounds is None:
            target = 0
        assert hasattr(
            surface_current_field, "Phi_mn"
        ), "surface_current_field must be a FourierCurrentPotentialField"
        self._surface_current_field = surface_current_field
        self._source_grid = source_grid

        super().__init__(
            things=[surface_current_field],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        surface_current_field = self.things[0]

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._surface_data_keys = ["K"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        surface_transforms = get_transforms(
            self._surface_data_keys,
            obj=surface_current_field,
            grid=source_grid,
            has_axis=source_grid.axis.size,
        )

        self._constants = {
            "surface_transforms": surface_transforms,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, surface_params=None, constants=None):
        """Compute surface current regularization.

        Parameters
        ----------
        surface_params : dict
            Dictionary of surface degrees of freedom,
            eg FourierCurrentPotential.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            The surface current density magnitude on the source surface.

        """
        if constants is None:
            constants = self.constants

        surface_data = compute_fun(
            self._surface_current_field,
            self._surface_data_keys,
            params=surface_params,
            transforms=constants["surface_transforms"],
            profiles={},
            basis="xyz",
        )

        K_mag = safenorm(surface_data["K"], axis=-1)
        return K_mag


class CoilsetMinDistance(_Objective):
    """Target the min distance btwn coils in the coilset.

    Parameters
    ----------
    coilset : CoilSet, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
        FIXME: add normalization for the B part of this objective
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate coils at.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = "rtz"
    _units = "m"
    _print_value_fmt = "CoilSet Minimum Distance: {:10.3e} "

    def __init__(
        self,
        coilset,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        source_grid=None,
        name="minimum-coil-distance",
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._coilset = coilset

        super().__init__(
            things=[coilset],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        coilset = self.things[0]
        # if field is different than self._field, update
        if coilset != self._coilset:
            self._coilset = coilset
        transforms = get_transforms(
            ["X", "Y", "Z"],
            obj=coilset[0],
            grid=self._source_grid,
        )
        # eval_grid.num_nodes for quad flux cost,
        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants = {
            "quad_weights": 1.0,
            "transforms": transforms,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, coil_params, constants=None):
        """Compute quadratic flux.

        Parameters
        ----------
        coil_params : dict
            Dictionary of coilset degrees of freedom,
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : scalar
            minimum distance between coils in the coilset

        """
        if constants is None:
            constants = self.constants

        return self.things[0].compute_minimum_distance(
            params=coil_params,
            method="jitable",
            grid=self._source_grid,
            transforms=constants["transforms"],
        )


class CoilsetCurvature(_Objective):
    """Target the curvatures in a coilset.

    Parameters
    ----------
    coilset : CoilSet, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
        FIXME: add normalization for the B part of this objective
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate coils at.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = "rtz"
    _units = "1/m"
    _print_value_fmt = "CoilSet Curvatures: {:10.3e} "

    def __init__(
        self,
        coilset,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        source_grid=None,
        name="coilset-curvature",
    ):
        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._coilset = coilset

        super().__init__(
            things=[coilset],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        coilset = self.things[0]
        # if field is different than self._field, update
        if coilset != self._coilset:
            self._coilset = coilset
        transforms = get_transforms(
            ["curvature"],
            obj=coilset[0],
            grid=self._source_grid,
        )
        # eval_grid.num_nodes for quad flux cost,
        self._dim_f = len(coilset) * self._source_grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants = {
            "quad_weights": jnp.ones(self._dim_f),
            "transforms": transforms,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, coil_params, constants=None):
        """Compute quadratic flux.

        Parameters
        ----------
        coil_params : dict
            Dictionary of coilset degrees of freedom,
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : scalar
            minimum distance between coils in the coilset

        """
        if constants is None:
            constants = self.constants

        fs = self.things[0].compute(
            "curvature",
            params=coil_params,
            method="jitable",
            grid=self._source_grid,
            transforms=constants["transforms"],
        )
        curvs = []
        for f in fs:
            curvs.append(f["curvature"])

        return jnp.concatenate(curvs)


class ToroidalFlux(_Objective):
    """Target the Toroidal flux in an equilibrium from a magnetic field.

    NOTE: Only works for vacuum equilibria currently

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    field : MagneticField
        MagneticField object, the parameters of this will be optimized
        to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect on this objective.
        FIXME: add normalization for the B part of this objective
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect on this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    source_grid : Grid, optional
        Collocation grid containing the nodes to evaluate field source at on
        the winding surface. (used if e.g. field is a CoilSet or
        FourierCurrentPotentialField)
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at.
    external_field : MagneticField, optional
        MagneticField object containing the external field to consider when
        minimizing the Bn errors. If None, the external field is assumed to be zero.
        e.g. this could be a 1/R field representing external TF coils, or
        it could be set of discrete TF coils so that coil ripple is considered during
        the optimization of the ``field`` object.
    external_field_source_grid : Grid, optional
        Grid object used to discretize the external field source.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = "rtz"
    _units = "T"
    _print_value_fmt = "Quadratic Flux: {:10.3e} "

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        source_grid=None,
        eval_grid=None,
        external_field=None,
        external_field_source_grid=None,
        name="toroidal-flux",
        eq_fixed=False,
    ):
        if target is None and bounds is None:
            target = eq.Psi
        self._field = field
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq_fixed = eq_fixed
        self._eq = eq if eq_fixed else None
        self._external_field = external_field
        self._external_field_source_grid = external_field_source_grid

        super().__init__(
            things=[field, eq] if not eq_fixed else [field],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self._eq if self._eq_fixed else self.things[1]
        field = self.things[0]
        # if field is different than self._field, update
        if field != self._field:
            self._field = field
        # if eq is different than self._eq, update
        if eq != self._eq:
            self._eq = eq
        if self._eval_grid is None:
            eval_grid = LinearGrid(
                L=eq.L_grid, M=eq.M_grid, zeta=jnp.array(0.0), NFP=eq.NFP
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid
        if not np.allclose(eval_grid.nodes[:, 2], eval_grid.nodes[0, 2]):
            warnings.warn("Evaluation grid should be at constant zeta")

        # ensure vacuum eq, as we don't yet support finite beta
        pres = np.max(np.abs(eq.compute("p")["p"]))
        curr = np.max(np.abs(eq.compute("current")["current"]))
        warnif(
            pres > 1e-8,
            UserWarning,
            f"Pressure is non-zero (max {pres} Pa), "
            + "finite beta not supported yet.",
        )
        warnif(
            curr > 1e-8,
            UserWarning,
            f"Current is non-zero (max {curr} A), "
            + "finite plasma currents not supported yet.",
        )

        # eval_grid.num_nodes for quad flux cost,
        self._dim_f = 1
        self._equil_data_keys = ["|e_rho x e_theta|", "R", "phi", "Z"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        equil_profiles = get_profiles(
            self._equil_data_keys,
            obj=eq,
            grid=eval_grid,
            has_axis=eval_grid.axis.size,
        )
        equil_transforms = get_transforms(
            self._equil_data_keys,
            obj=eq,
            grid=eval_grid,
            has_axis=eval_grid.axis.size,
        )

        if self._eq_fixed:
            data = eq.compute(["R", "phi", "Z", "|e_rho x e_theta|"], grid=eval_grid)

            plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T

        if not self._eq_fixed:
            self._constants = {
                "equil_transforms": equil_transforms,
                "equil_profiles": equil_profiles,
                "quad_weights": 1.0,
            }
        else:
            self._constants = {
                "equil_transforms": equil_transforms,
                "equil_profiles": equil_profiles,
                "plasma_coords": plasma_coords,
                "equil_data": data,
                "quad_weights": 1.0,
            }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, field_params=None, equil_params=None, constants=None):
        """Compute toroidal flux.

        Parameters
        ----------
        field_params : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotential.params_dict
        equil_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : float
            Toroidal flux from coils and external field

        """
        if constants is None:
            constants = self.constants
        if not self._eq_fixed:
            data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._equil_data_keys,
                params=equil_params,
                transforms=constants["equil_transforms"],
                profiles=constants["equil_profiles"],
            )

        else:
            data = constants["equil_data"]
            plasma_coords = constants["plasma_coords"]

        B = self._field.compute_magnetic_field(
            plasma_coords, basis="rpz", grid=self._source_grid, params=field_params
        )
        grid = self._eval_grid
        Psi = jnp.sum(
            grid.spacing[:, 0]
            * grid.spacing[:, 1]
            * data["|e_rho x e_theta|"]
            * B[:, 1]
        )

        if self._external_field is not None:
            B_ext = self._external_field.compute_magnetic_field(
                plasma_coords,
                grid=self._external_field_source_grid,
                basis="rpz",
            )
            Psi += jnp.sum(
                grid.spacing[:, 0]
                * grid.spacing[:, 1]
                * data["|e_rho x e_theta|"]
                * B_ext[:, 1]
            )

        return Psi
