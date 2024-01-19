"""Objectives for targeting geometrical quantities."""

import warnings

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_profiles, get_transforms, rpz2xyz, rpz2xyz_vec
from desc.compute.utils import safenorm
from desc.grid import LinearGrid
from desc.utils import Timer

from .objective_funs import _Objective


class SurfaceCurrentRegularizedQuadraticFlux(_Objective):
    """Target the quadratic flux from a surface current with regularization.

    compute

    (B.n)^2 + alpha*(|K|)^2

    where n is the normal vector to the ppasma surface, and B is the magnetic field at
    the plasma surface, K is the winding surface current density, and alpha is the
    regularization parameter.

    This is the quadratic flux from a magnetic field due to
    a surface current of the form

    K = n x ∇ Φ
    Φ(θ,ζ) = Φₛᵥ(θ,ζ) + Gζ/2π + Iθ/2π

    i.e. a FourierCurrentPotentialField

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
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
        the winding surface.
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at.
    alpha: float, optional
        Regularization parameter, 0 for no regularization. The larger this
        parameter is, the less complex the surface current will be, but the
        worse the normal field.
    external_field : MagneticField, optional
        MagneticField object containing the external field to consider when
        minimizing the Bn errors. If None, the external field is assumed to be zero.
        e.g. this could be a 1/R field representing external TF coils, or
        it could be set of TF coils so that coil ripple is considered during
        the optimization.
    external_field_source_grid : Grid, optional
        Grid object used to discretize the external field source.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = ""
    _units = ""
    _print_value_fmt = "Regularized Quadratic Flux: {:10.3e} "

    def __init__(
        self,
        eq,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        source_grid=None,
        eval_grid=None,
        alpha=0.0,
        external_field=None,
        external_field_source_grid=None,
        name="surface-current-regularized-quadratic-flux",
        eq_fixed=False,
    ):
        if target is None and bounds is None:
            target = 0
        self._surface_current_field = surface_current_field
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._alpha = alpha
        self._eq_fixed = eq_fixed
        self._eq = eq if eq_fixed else None
        self._external_field = external_field
        self._external_field_source_grid = external_field_source_grid

        super().__init__(
            things=[surface_current_field, eq]
            if not eq_fixed
            else [surface_current_field],
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
        surface_current_field = self.things[0]
        # if things[1] is different than self._surface_current_field, update
        if surface_current_field != self._surface_current_field:
            self._surface_current_field = surface_current_field

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=surface_current_field._M_Phi * 3 + 1,
                N=surface_current_field._N_Phi * 3 + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid
        if self._eval_grid is None:
            eval_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            eval_grid = self._eval_grid
        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")
        if not np.allclose(eval_grid.nodes[:, 0], 1):
            warnings.warn("Evaluation grid includes interior points, should be rho=1")

        # eval_grid.num_nodes for quad flux cost,
        # source_grid.num_nodes for the regularization cost
        self._dim_f = eval_grid.num_nodes + source_grid.num_nodes
        self._equil_data_keys = ["n_rho", "R", "phi", "Z", "|e_theta x e_zeta|"]
        self._surface_data_keys = ["K", "x", "|e_theta x e_zeta|"]
        # TODO: should check that G is set correctly
        # and is not an optimizable parameter?
        # since we know what G should be given the equilibrium.

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
        surface_transforms = get_transforms(
            self._surface_data_keys,
            obj=surface_current_field,
            grid=source_grid,
            has_axis=source_grid.axis.size,
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
                "surface_transforms": surface_transforms,
            }
        else:
            self._constants = {
                "equil_transforms": equil_transforms,
                "equil_profiles": equil_profiles,
                "surface_transforms": surface_transforms,
                "plasma_coords": plasma_coords,
                "equil_data": data,
            }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, surface_params=None, equil_params=None, constants=None):
        """Compute current-regularized quadratic flux.

        Parameters
        ----------
        equil_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        surface_params : dict
            Dictionary of surface degrees of freedom,
            eg FourierCurrentPotential.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : float
            Sum of the quadratic flux integrated on the plasma surface along
            with the regularization term times the surface current
            density magnitude integrated over the source surface.

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

        surface_data = compute_fun(
            self._surface_current_field,
            self._surface_data_keys,
            params=surface_params,
            transforms=constants["surface_transforms"],
            profiles={},
            basis="xyz",
        )
        B = self._surface_current_field.compute_magnetic_field(
            plasma_coords,
            grid=self._source_grid,
            params=surface_params,
            data=surface_data,
            basis="xyz",
        )
        Bn = jnp.sum(B * data["n_rho"], axis=-1)

        if self._external_field is not None:
            B_ext = self._external_field.compute_magnetic_field(
                plasma_coords,
                grid=self._external_field_source_grid,
                basis="xyz",
            )
            Bn += jnp.sum(B_ext * data["n_rho"], axis=-1)

        K_mag = safenorm(surface_data["K"], axis=-1)
        return jnp.concatenate([Bn, self._alpha * K_mag])
