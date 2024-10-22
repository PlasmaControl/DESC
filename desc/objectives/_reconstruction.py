import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.magnetic_fields._current_potential import (
    _compute_A_or_B_from_CurrentPotentialField,
)
from desc.utils import Timer, warnif

from .objective_funs import _Objective


# TODO: should allow a CoilSet of flux loops, and the target
# would be a pytree of the same structure, to avoid redundant
# calculations or needing multiple of these objectives
class FluxLoop(_Objective):
    """Target the flux through a loop given by the given coil.

    This objective will calculate the magnetic flux through a given coil,
    intended for use to compare to experimental measurements for reconstruction.

    Will use the vector potential method to calculate the magnetic flux
    (Φ = ∮ 𝐀 ⋅ 𝐝𝐥 over the coil loop) The vector potential method
    is much more efficient, however not every ``MagneticField`` object
    has a vector potential available to compute, so in those cases
    an error will be thrown.

    The equilibrium and possibly a coilset are allowed to vary, but the
    flux loop location is held fixed.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium for which the magnetic flux will be calculated,
        if not vacuum.
    coils : CoilSet
        Coilset that supports the equilibrium. If coils_fixed is True,
        their contribution to the flux loops will be pre-calculated.
    flux_loops : MagneticField
        CoilSet object, contains the diagnostic flux loop geometry.
        Assumed to be fixed
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Defaults to eq.Psi. Must be broadcastable to Objective.dim_f
        which is the number of coils in flux_loops.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source.
        Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    flux_loop_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the magnetic vector potential
        at in the flux loop geometry. Defaults to a LinearGrid(L=0, M=0,
        N=50).
    vc_source_grid : LinearGrid
        LinearGrid to use for the (non-singular) integral for the virtual casing
        principle to calculate the flux loop contribution from the
        plasma currents. Must have endpoint=False and sym=False and be linearly
        spaced in theta and zeta, with nodes only at rho=1.0
    name : str, optional
        Name of the objective function.
    jac_chunk_size : int , optional
        Will calculate the Jacobian for this objective ``jac_chunk_size``
        columns at a time, instead of all at once. The memory usage of the
        Jacobian calculation is roughly ``memory usage = m0 + m1*jac_chunk_size``:
        the smaller the chunk size, the less memory the Jacobian calculation
        will require (with some baseline memory usage). The time to compute the
        Jacobian is roughly ``t=t0 +t1/jac_chunk_size``, so the larger the
        ``jac_chunk_size``, the faster the calculation takes, at the cost of
        requiring more memory. A ``jac_chunk_size`` of 1 corresponds to the least
        memory intensive, but slowest method of calculating the Jacobian.
        If None, it will use the largest size i.e ``obj.dim_x``.
    vacuum : bool
        whether eq is vacuum, in which case plasma contribution to B won't be
        calculated.


    """

    _coordinates = "rtz"
    _units = "(Wb)"
    _print_value_fmt = "Toroidal Flux: "

    def __init__(
        self,
        eq,
        coilset,
        flux_loops,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        field_grid=None,
        flux_loop_grid=None,
        vc_source_grid=None,
        eval_grid=None,
        name="toroidal-flux",
        jac_chunk_size=None,
        coils_fixed=False,
        vacuum=False,
    ):
        if target is None and bounds is None:
            target = eq.Psi
        self._coils = coilset
        self._field_grid = field_grid
        self._eval_grid = eval_grid
        self._vc_source_grid = vc_source_grid
        self._flux_loop_grid = flux_loop_grid
        self._flux_loops = flux_loops
        self._eq = eq
        self._vacuum = vacuum
        self._sheet_current = hasattr(self._eq.surface, "Phi_mn")
        things = [eq]
        self._coils_fixed = coils_fixed
        if not coils_fixed:
            things.append(coilset)
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
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
        eq = self._eq

        if self._normalize:
            self._normalization = eq.Psi

        # ensure vacuum eq, as is unneeded for finite beta
        pres = np.max(np.abs(eq.compute("p")["p"]))
        curr = np.max(np.abs(eq.compute("current")["current"]))
        warnif(
            pres > 1e-8,
            UserWarning,
            f"Pressure appears to be non-zero (max {pres} Pa), "
            + "this objective is unneeded at finite beta.",
        )
        warnif(
            curr > 1e-8,
            UserWarning,
            f"Current appears to be non-zero (max {curr} A), "
            + "this objective is unneeded at finite beta.",
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        self._eq_vc_data_keys = ["K_vc", "R", "phi", "Z"]

        eq_transforms = get_transforms(
            self._eq_vc_data_keys, self._eq, grid=self._vc_source_grid
        )
        eq_profiles = get_profiles(
            self._eq_vc_data_keys, self._eq, grid=self._vc_source_grid
        )

        flux_loop_data = self._flux_loops.compute(
            ["x", "x_s"], grid=self._flux_loop_grid
        )

        # dim_f is number of flux loops we have
        # TODO: use coil utils and flatten this
        self._dim_f = self._flux_loops.num_coils

        # pre-calc coil contrib to flux loops if coils are fixed
        if self._coils_fixed:
            fluxes = []
            for i in range(self._dim_f):
                A = self._coils.compute_magnetic_vector_potential(
                    flux_loop_data[i]["x"],
                    basis="rpz",
                    source_grid=self._field_grid,
                )
                A_dot_dxds = jnp.sum(
                    A * flux_loop_data[i]["x_s"] * self._flux_loops[i].num_turns, axis=1
                )
                Psi = jnp.sum(self._flux_loop_grid.spacing[:, 2] * A_dot_dxds)
                fluxes.append(Psi)
            fluxes = jnp.array(fluxes)

        self._constants = {
            "quad_weights": 1.0,
            "field_grid": self._field_grid,
            "flux_loop_data": flux_loop_data,
            "equil_transforms": eq_transforms,
            "equil_profiles": eq_profiles,
            "vc_source_grid": self._vc_source_grid,
        }
        if self._coils_fixed:
            self._constants["flux_from_coils"] = fluxes
        if self._sheet_current:
            self._sheet_data_keys = ["K"]
            sheet_source_transforms = get_transforms(
                self._sheet_data_keys, obj=eq.surface, grid=self._vc_source_grid
            )
            self._constants["sheet_source_transforms"] = sheet_source_transforms

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params=None, field_params=None, constants=None):
        """Compute toroidal flux.

        Parameters
        ----------
        equil_params : dict
            Dictionary of eq degrees of freedom,
            eg Equilibrium.params_dict
        field_params : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotential.params_dict or CoilSet.params_dict
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

        flux_loop_data = constants["flux_loop_data"]
        grid = self._flux_loop_grid
        plasma_surf_data = compute_fun(
            self._eq,
            self._eq_vc_data_keys,
            params=eq_params,
            profiles=constants["equil_profiles"],
            transforms=constants["equil_transforms"],
        )
        plasma_surf_data["x"] = jnp.vstack(
            [plasma_surf_data["R"], plasma_surf_data["phi"], plasma_surf_data["Z"]]
        ).T
        if self._sheet_current:
            sheet_params = {
                "R_lmn": eq_params["Rb_lmn"],
                "Z_lmn": eq_params["Zb_lmn"],
                "I": eq_params["I"],
                "G": eq_params["G"],
                "Phi_mn": eq_params["Phi_mn"],
            }
            sheet_source_data = compute_fun(
                "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
                self._sheet_data_keys,
                params=sheet_params,
                transforms=constants["sheet_source_transforms"],
                profiles={},
            )
            plasma_surf_data["K_vc"] += sheet_source_data["K"]
        plasma_surf_data["K"] = plasma_surf_data["K_vc"]
        # loop over the flux loop coils
        fluxes = []
        for i in range(self._dim_f):
            if not self._coils_fixed:
                Acoil = self._coils.compute_magnetic_vector_potential(
                    flux_loop_data[i]["x"],
                    basis="rpz",
                    source_grid=constants["field_grid"],
                    params=field_params,
                )
            else:
                Acoil = jnp.zeros_like(self._flux_loop_grid.nodes)

            # get plasma contribution
            if not self._vacuum:
                Aplasma = _compute_A_or_B_from_CurrentPotentialField(
                    self._coils,
                    flux_loop_data[i]["x"],
                    source_grid=constants["vc_source_grid"],
                    compute_A_or_B="A",
                    data=plasma_surf_data,
                )
            else:
                Aplasma = jnp.zeros_like(self._flux_loop_grid.nodes)
            A = Aplasma + Acoil

            A_dot_dxds = jnp.sum(
                A * flux_loop_data[i]["x_s"] * self._flux_loops[i].num_turns, axis=1
            )
            Psi = jnp.sum(grid.spacing[:, 2] * A_dot_dxds)
            fluxes.append(Psi)
        fluxes = jnp.asarray(fluxes)
        if self._coils_fixed:
            fluxes += constants["flux_from_coils"]
        return fluxes


class RogowskiLoop(_Objective):
    """Target the net current enclosed by a loop given by the given coil.

    This objective will calculate the net current through a given coil,
    intended for use to compare to experimental measurements for reconstruction.

    Will use the Ampere's Law + stoke's theorem to calculate the current
    (mu_0 I = ∮ B ⋅ 𝐝𝐥 over the coil loop)

    The equilibrium and possibly a coilset are allowed to vary, but the
    flux loop location is held fixed.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium for which the magnetic flux will be calculated,
        if not vacuum.
    coils : CoilSet
        Coilset that supports the equilibrium. If coils_fixed is True,
        their contribution to the flux loops will be pre-calculated.
    flux_loops : MagneticField
        CoilSet object, contains the diagnostic flux loop geometry.
        Assumed to be fixed
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Defaults to eq.Psi. Must be broadcastable to Objective.dim_f
        which is the number of coils in flux_loops. Target should be in
        units of mu0*I
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source.
        Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    flux_loop_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the magnetic vector potential
        at in the flux loop geometry. Defaults to a LinearGrid(L=0, M=0,
        N=50).
    vc_source_grid : LinearGrid
        LinearGrid to use for the (non-singular) integral for the virtual casing
        principle to calculate the current contribution from the
        plasma currents. Must have endpoint=False and sym=False and be linearly
        spaced in theta and zeta, with nodes only at rho=1.0
    name : str, optional
        Name of the objective function.
    jac_chunk_size : int , optional
        Will calculate the Jacobian for this objective ``jac_chunk_size``
        columns at a time, instead of all at once. The memory usage of the
        Jacobian calculation is roughly ``memory usage = m0 + m1*jac_chunk_size``:
        the smaller the chunk size, the less memory the Jacobian calculation
        will require (with some baseline memory usage). The time to compute the
        Jacobian is roughly ``t=t0 +t1/jac_chunk_size``, so the larger the
        ``jac_chunk_size``, the faster the calculation takes, at the cost of
        requiring more memory. A ``jac_chunk_size`` of 1 corresponds to the least
        memory intensive, but slowest method of calculating the Jacobian.
        If None, it will use the largest size i.e ``obj.dim_x``.
    vacuum : bool
        whether eq is vacuum, in which case plasma contribution to B won't be
        calculated.

    """

    _coordinates = "rtz"
    _units = "(Wb)"
    _print_value_fmt = "Toroidal Flux: "

    def __init__(
        self,
        eq,
        coilset,
        flux_loops,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        field_grid=None,
        flux_loop_grid=None,
        vc_source_grid=None,
        eval_grid=None,
        name="toroidal-flux",
        jac_chunk_size=None,
        coils_fixed=False,
        vacuum=False,
    ):
        if target is None and bounds is None:
            target = eq.Psi
        self._coils = coilset
        self._field_grid = field_grid
        self._eval_grid = eval_grid
        self._vc_source_grid = vc_source_grid
        self._flux_loop_grid = flux_loop_grid
        self._flux_loops = flux_loops
        self._eq = eq
        self._vacuum = vacuum
        self._sheet_current = hasattr(self._eq.surface, "Phi_mn")
        things = [eq]
        self._coils_fixed = coils_fixed
        if not coils_fixed:
            things.append(coilset)
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
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
        eq = self._eq

        if self._normalize:
            self._normalization = eq.Psi
        if self._vc_source_grid is None:
            # for axisymmetry we still need to know about toroidal effects, so its
            # cheapest to pretend there are extra field periods
            self._vc_source_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP if eq.N > 0 else 64,
                sym=False,
            )

        # ensure vacuum eq, as is unneeded for finite beta
        pres = np.max(np.abs(eq.compute("p")["p"]))
        curr = np.max(np.abs(eq.compute("current")["current"]))
        warnif(
            pres > 1e-8,
            UserWarning,
            f"Pressure appears to be non-zero (max {pres} Pa), "
            + "this objective is unneeded at finite beta.",
        )
        warnif(
            curr > 1e-8,
            UserWarning,
            f"Current appears to be non-zero (max {curr} A), "
            + "this objective is unneeded at finite beta.",
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        self._eq_vc_data_keys = ["K_vc", "R", "phi", "Z"]

        eq_transforms = get_transforms(
            self._eq_vc_data_keys, self._eq, grid=self._vc_source_grid
        )
        eq_profiles = get_profiles(
            self._eq_vc_data_keys, self._eq, grid=self._vc_source_grid
        )

        flux_loop_data = self._flux_loops.compute(
            ["x", "x_s"], grid=self._flux_loop_grid
        )

        # dim_f is number of flux loops we have
        # TODO: use coil utils and flatten this
        self._dim_f = self._flux_loops.num_coils

        # pre-calc coil contrib to flux loops if coils are fixed
        if self._coils_fixed:
            currents = []
            for i in range(self._dim_f):
                B = self._coils.compute_magnetic_field(
                    flux_loop_data[i]["x"],
                    basis="rpz",
                    source_grid=self._field_grid,
                )
                B_dot_dxds = jnp.sum(B * flux_loop_data[i]["x_s"], axis=1)
                mu0_I = jnp.sum(self._flux_loop_grid.spacing[:, 2] * B_dot_dxds)
                currents.append(mu0_I)
            currents = jnp.array(currents)

        self._constants = {
            "quad_weights": 1.0,
            "field_grid": self._field_grid,
            "flux_loop_data": flux_loop_data,
            "equil_transforms": eq_transforms,
            "equil_profiles": eq_profiles,
            "vc_source_grid": self._vc_source_grid,
        }
        if self._coils_fixed:
            self._constants["mu0_I_from_coils"] = currents
        if self._sheet_current:
            self._sheet_data_keys = ["K"]
            sheet_source_transforms = get_transforms(
                self._sheet_data_keys, obj=eq.surface, grid=self._vc_source_grid
            )
            self._constants["sheet_source_transforms"] = sheet_source_transforms

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params=None, field_params=None, constants=None):
        """Compute toroidal flux.

        Parameters
        ----------
        equil_params : dict
            Dictionary of eq degrees of freedom,
            eg Equilibrium.params_dict
        field_params : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotential.params_dict or CoilSet.params_dict
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

        flux_loop_data = constants["flux_loop_data"]
        grid = self._flux_loop_grid
        plasma_surf_data = compute_fun(
            self._eq,
            self._eq_vc_data_keys,
            params=eq_params,
            profiles=constants["equil_profiles"],
            transforms=constants["equil_transforms"],
        )
        plasma_surf_data["x"] = jnp.vstack(
            [plasma_surf_data["R"], plasma_surf_data["phi"], plasma_surf_data["Z"]]
        ).T
        if self._sheet_current:
            sheet_params = {
                "R_lmn": eq_params["Rb_lmn"],
                "Z_lmn": eq_params["Zb_lmn"],
                "I": eq_params["I"],
                "G": eq_params["G"],
                "Phi_mn": eq_params["Phi_mn"],
            }
            sheet_source_data = compute_fun(
                "desc.magnetic_fields._current_potential.FourierCurrentPotentialField",
                self._sheet_data_keys,
                params=sheet_params,
                transforms=constants["sheet_source_transforms"],
                profiles={},
            )
            plasma_surf_data["K_vc"] += sheet_source_data["K"]
        plasma_surf_data["K"] = plasma_surf_data["K_vc"]
        # loop over the flux loop coils
        mu0_Is = []
        for i in range(self._dim_f):
            if not self._coils_fixed:
                Bcoil = self._coils.compute_magnetic_field(
                    flux_loop_data[i]["x"],
                    basis="rpz",
                    source_grid=constants["field_grid"],
                    params=field_params,
                )
            else:
                Bcoil = jnp.zeros_like(self._flux_loop_grid.nodes)

            # get plasma contribution
            if not self._vacuum:
                Bplasma = _compute_A_or_B_from_CurrentPotentialField(
                    self._coils,
                    flux_loop_data[i]["x"],
                    source_grid=constants["vc_source_grid"],
                    compute_A_or_B="B",
                    data=plasma_surf_data,
                )
            else:
                Bplasma = jnp.zeros_like(self._flux_loop_grid.nodes)
            B = Bplasma + Bcoil

            B_dot_dxds = jnp.sum(B * flux_loop_data[i]["x_s"], axis=1)
            mu0I = jnp.sum(grid.spacing[:, 2] * B_dot_dxds)
            mu0_Is.append(mu0I)
        mu0_Is = jnp.asarray(mu0_Is)
        if self._coils_fixed:
            mu0_Is += constants["mu0_I_from_coils"]
        out = jnp.atleast_1d(mu0_Is.squeeze())
        return out
