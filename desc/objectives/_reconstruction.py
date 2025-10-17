"""Objectives pertaining to experimental reconstruction."""

import numpy as np
from scipy.constants import mu_0

from desc.backend import jax, jnp
from desc.compute import get_profiles, get_transforms, xyz2rpz, xyz2rpz_vec
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, dot, errorif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs

# Magnetics


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
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    B_plasma_chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``bs_chunk_size``.

    """

    _coordinates = "rtz"
    _units = "(Wb)"
    _print_value_fmt = "Diamagnetic Loop Error : "
    _print_error = True
    _static_attrs = _Objective._static_attrs + [
        # TODO: should we add an intermediate flag to avoid using an array as static?
        "_sheet_current",
        "_vacuum",
        "_eq_vc_data_keys",
        # TODO: make it field fixed or ext field fixed
        "_coils_fixed",
        "_sheet_data_keys",
        "_compute_A_or_B_from_CurrentPotentialField",
    ]

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
        name="toroidal-flux",
        jac_chunk_size=None,
        coils_fixed=False,
        vacuum=False,
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = eq.Psi
        self._coils = coilset
        self._field_grid = field_grid
        self._vc_source_grid = vc_source_grid
        self._flux_loop_grid = flux_loop_grid
        self._flux_loops = flux_loops
        self._eq = eq
        self._vacuum = vacuum
        self._sheet_current = hasattr(self._eq.surface, "Phi_mn")
        things = [eq]
        self._coils_fixed = coils_fixed
        self._bs_chunk_size = bs_chunk_size
        if B_plasma_chunk_size == 0:
            B_plasma_chunk_size = None
        self._B_plasma_chunk_size = B_plasma_chunk_size
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
        from desc.magnetic_fields._current_potential import (
            _compute_A_or_B_from_CurrentPotentialField,
        )

        self._compute_A_or_B_from_CurrentPotentialField = (
            _compute_A_or_B_from_CurrentPotentialField
        )
        eq = self._eq

        if self._normalize:
            self._normalization = eq.Psi

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        self._eq_vc_data_keys = ["K_vc", "R", "phi", "Z"]

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
        if self._flux_loop_grid is None:
            # for axisymmetry we still need to know about toroidal effects, so its
            # cheapest to pretend there are extra field periods
            self._flux_loop_grid = LinearGrid(
                N=50,
            )

        eq_transforms = get_transforms(
            self._eq_vc_data_keys, self._eq, grid=self._vc_source_grid
        )
        eq_profiles = get_profiles(
            self._eq_vc_data_keys, self._eq, grid=self._vc_source_grid
        )

        # dim_f is number of flux loops we have
        # TODO: use coil utils and flatten this
        # or enforce no coilsets inside of it
        self._dim_f = self._flux_loops.num_coils

        flux_loop_data = self._flux_loops.compute(
            ["x", "x_s"], grid=self._flux_loop_grid
        )
        flux_loop_all_x = jnp.vstack([data["x"] for data in flux_loop_data])
        flux_loop_all_x_s = jnp.vstack([data["x_s"] for data in flux_loop_data])
        self._flux_loop_all_x = flux_loop_all_x
        self._flux_loop_all_x_s = flux_loop_all_x_s

        segment_ids = np.concatenate(
            [
                np.array(data["x"].shape[0] * [i])
                for i, data in enumerate(flux_loop_data)
            ]
        )
        self._flux_loop_all_x = flux_loop_all_x
        self._flux_loop_all_x_s = flux_loop_all_x_s
        # to be used later to do the A.dl integral
        self._flux_loop_all_x_s_time_spacing = (
            self._flux_loop_grid.spacing[:, 2][:, None]
            * flux_loop_all_x_s.reshape(
                (self._dim_f, self._flux_loop_grid.num_nodes, 3)
            )
        ).reshape(flux_loop_all_x_s.shape)
        self._segment_ids = segment_ids

        # pre-calc coil contrib to flux loops if coils are fixed
        if self._coils_fixed:
            A = self._coils.compute_magnetic_vector_potential(
                flux_loop_all_x,
                basis="rpz",
                source_grid=self._field_grid,
                chunk_size=self._bs_chunk_size,
            )
            A_dot_dxds = jnp.sum(A * self._flux_loop_all_x_s_time_spacing, axis=1)
            fluxes = jax.ops.segment_sum(
                A_dot_dxds,
                segment_ids=segment_ids,
                num_segments=self._dim_f,
                indices_are_sorted=True,
            )

        self._constants = {
            "quad_weights": 1.0,
            "field_grid": self._field_grid,
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

        # get plasma contribution
        if not self._vacuum:
            Aplasma = self._compute_A_or_B_from_CurrentPotentialField(
                self._coils,  # this is unused, just pass a dummy variable in
                self._flux_loop_all_x,
                source_grid=constants["vc_source_grid"],
                compute_A_or_B="A",
                data=plasma_surf_data,
                chunk_size=self._B_plasma_chunk_size,
            )
        else:
            Aplasma = jnp.zeros_like(self._flux_loop_all_x)
        A = Aplasma

        if not self._coils_fixed:
            Acoil = self._coils.compute_magnetic_vector_potential(
                self._flux_loop_all_x,
                basis="rpz",
                source_grid=constants["field_grid"],
                params=field_params,
                chunk_size=self._bs_chunk_size,
            )
            A += Acoil

        A_dot_dxds = jnp.sum(A * self._flux_loop_all_x_s_time_spacing, axis=1)
        fluxes = jax.ops.segment_sum(
            A_dot_dxds,
            segment_ids=self._segment_ids,
            num_segments=self._dim_f,
            indices_are_sorted=True,
        )

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
    flux_loops : CoilSet
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
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    B_plasma_chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``bs_chunk_size``.

    """

    _coordinates = "rtz"
    _units = "(Tm)"
    _print_value_fmt = "Net Enclosed Current Error : "
    _print_error = True
    _static_attrs = _Objective._static_attrs + [
        # TODO: should we add an intermediate flag to avoid using an array as static?
        "_sheet_current",
        "_vacuum",
        "_eq_vc_data_keys",
        # TODO: make it field fixed or ext field fixed
        "_coils_fixed",
        "_sheet_data_keys",
        "_compute_A_or_B_from_CurrentPotentialField",
    ]

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
        name="toroidal-flux",
        jac_chunk_size=None,
        coils_fixed=False,
        vacuum=False,
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = eq.Psi
        self._coils = coilset
        self._field_grid = field_grid
        self._vc_source_grid = vc_source_grid
        self._flux_loop_grid = flux_loop_grid
        # TODO: allow single coil to be passed in
        self._flux_loops = flux_loops
        self._eq = eq
        self._vacuum = vacuum
        self._sheet_current = hasattr(self._eq.surface, "Phi_mn")
        things = [eq]
        self._coils_fixed = coils_fixed
        self._bs_chunk_size = bs_chunk_size
        if B_plasma_chunk_size == 0:
            B_plasma_chunk_size = None
        self._B_plasma_chunk_size = B_plasma_chunk_size
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
        from desc.magnetic_fields._current_potential import (
            _compute_A_or_B_from_CurrentPotentialField,
        )

        self._compute_A_or_B_from_CurrentPotentialField = (
            _compute_A_or_B_from_CurrentPotentialField
        )

        eq = self._eq

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["I"] * mu_0
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
                    chunk_size=self._bs_chunk_size,
                )
            else:
                Bcoil = jnp.zeros_like(self._flux_loop_grid.nodes)

            # get plasma contribution
            if not self._vacuum:
                # FIXME: Is this missing |e_theta x e_zeta| inside of data?
                Bplasma = self._compute_A_or_B_from_CurrentPotentialField(
                    self._coils,
                    flux_loop_data[i]["x"],
                    source_grid=constants["vc_source_grid"],
                    compute_A_or_B="B",
                    data=plasma_surf_data,
                    chunk_size=self._B_plasma_chunk_size,
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


class PointBMeasurement(_Objective):
    """Target B at a point in real space, outside the plasma.

    This objective will calculate the magnetic field at a point,
    intended for use to compare to experimental measurements for reconstruction.

    The equilibrium and possibly a MagneticField are allowed to vary, but the
    measurement location in real space is held fixed.

    The measurement point should be at a point outside the plasma, otherwise
    the plasma contribution will be incorrectly calculated.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium from which the plasma constribution to the magnetic field will
        be calculated, if not vacuum.
    field : MagneticField
        External field produced by coils or other sources outside the plasma.
    measurement_coords : (n,3) ndarray
        Array of n points at which the magnetic field B is measured.
    target : {float, ndarray}
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f which is the
        number of measurement points if ``direction`` is passed in, or
        3 times the number of measurement points if ``direction`` is not passed in.
        If ``direction`` is not passed in, target is assumed to be the
        flattened array of the field vector at the measurement points i.e.
        ``B_array.flatten()``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to Objective.dim_f
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source.
        Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    vc_source_grid : LinearGrid
        LinearGrid to use for the (non-singular) integral over the virtual casing
        principle current to calculate the flux loop contribution from the
        plasma currents. Must have endpoint=False and sym=False and be linearly
        spaced in theta and zeta, with nodes only at rho=1.0
    basis : {"rpz","xyz"}
        the basis used for ``measurement_coords``, ``directions`` and ``target``,
        assumed to be "rpz" by default. if "xyz", will convert the input arrays
        into "rpz".
    field_fixed : bool, optional
        whether or not to fix the external field's DOFs during optimization.
        False by default. Set to True if the field is not changing during the
        optimization.
    vacuum : bool, optional
        whether the Equilibrium is vacuum, in which case plasma contribution to B won't
        be calculated.
    directions : array (n,3), optional
        the directions of the measured B field for each of the n sensors, i.e. if a
        sensor is measuring the poloidal field and is located at (R,phi,Z) = (1,0,0),
        its ``direction`` might be [0,0,+/- 1]. If not passed, will default to instead
        comparing the entire magnetic field vector at the points passed in.

    """

    __doc__ = __doc__.rstrip() + collect_docs()

    _coordinates = "rtz"
    _units = "(T)"
    _print_value_fmt = "Point B Measurement Error: "
    _print_error = True
    _static_attrs = _Objective._static_attrs + [
        "_use_directions",
        "_sheet_current",
        "_vacuum",
        "_eq_vc_data_keys",
        "_field_fixed",
        "_sheet_data_keys",
        "_compute_A_or_B_from_CurrentPotentialField",
    ]

    def __init__(
        self,
        eq,
        field,
        measurement_coords,
        target,
        *,
        field_grid=None,
        vc_source_grid=None,
        field_fixed=False,
        vacuum=False,
        directions=None,
        basis="rpz",
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="Magnetic-Point-Measurement-Error",
        jac_chunk_size=None,
    ):
        self._field = field
        self._field_grid = field_grid
        self._vc_source_grid = vc_source_grid
        measurement_coords = np.atleast_2d(measurement_coords)
        self._use_directions = False
        if directions is not None:
            self._use_directions = True
            directions = np.atleast_2d(directions)
            assert (
                directions.shape == measurement_coords.shape
            ), "Must pass in same number of direction vectors as measurements"
            # make the direction vectors unit norm
            directions = directions / jnp.linalg.norm(directions, axis=1)[:, None]
        errorif(
            basis not in ["xyz", "rpz"],
            ValueError,
            f"basis must be either rpz or xyz, instead got {basis}",
        )

        if basis == "rpz":
            pass
        elif basis == "xyz":
            if directions:
                # convert directions to rpz
                directions = xyz2rpz_vec(
                    directions, x=measurement_coords[:, 0], y=measurement_coords[:, 1]
                )
                # no need to change target as is already a scalar
            else:
                # convert target B field vectors to rpz
                target = target.reshape(measurement_coords.shape)
                target = xyz2rpz_vec(
                    target, x=measurement_coords[:, 0], y=measurement_coords[:, 1]
                )
                target = target.flatten()

            measurement_coords = xyz2rpz(measurement_coords)
        self._measurement_coords = measurement_coords
        self._directions = directions
        self._vacuum = vacuum
        self._sheet_current = hasattr(eq.surface, "Phi_mn")
        things = [eq]
        self._field_fixed = field_fixed
        if not field_fixed:
            things.append(field)
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
        from desc.magnetic_fields._current_potential import (
            _compute_A_or_B_from_CurrentPotentialField,
        )

        self._compute_A_or_B_from_CurrentPotentialField = (
            _compute_A_or_B_from_CurrentPotentialField
        )
        eq = self.things[0]

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        self._eq_vc_data_keys = ["K_vc", "R", "phi", "Z"]

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

        self._eq_transforms = get_transforms(
            self._eq_vc_data_keys, self.things[0], grid=self._vc_source_grid
        )
        self._eq_profiles = get_profiles(
            self._eq_vc_data_keys, self.things[0], grid=self._vc_source_grid
        )

        # dim_f is number of B components we  have,
        # which is coords.size, if no directions are given,
        # else it is equal to how many points we are evaluating the
        # B measurement at (coords.shape[0])
        self._dim_f = (
            self._measurement_coords.size
            if not self._use_directions
            else self._measurement_coords.shape[0]
        )

        # pre-calc field contrib to B if field are fixed
        if self._field_fixed:
            self._B_from_field = self._field.compute_magnetic_field(
                self._measurement_coords,
                basis="rpz",
                source_grid=self._field_grid,
            )
            # dot with directions if directions provided
            self._B_from_field = (
                self._B_from_field
                if not self._use_directions
                else dot(self._B_from_field, self._directions)
            )

        self._constants = {
            "quad_weights": 1.0,
        }

        if self._sheet_current:
            self._sheet_data_keys = ["K"]
            self._sheet_source_transforms = get_transforms(
                self._sheet_data_keys, obj=eq.surface, grid=self._vc_source_grid
            )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, eq_params=None, field_params=None, constants=None):
        """Compute point B measurements in "rpz" basis.

        Parameters
        ----------
        equil_params : dict
            Dictionary of eq degrees of freedom,
            eg Equilibrium.params_dict
        field_params : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotentialField.params_dict or CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : array
            B from plasma and external field at given measurement coordinates,
            These are always returned in rpz basis, and if ``directions`` is None, this
            is equal to ``B.flatten()`` where ``B`` is an ``(n,3)`` array corresponding
            to the magnetic field at the given measurement coordinates.

        """
        if constants is None:
            constants = self.constants

        plasma_surf_data = compute_fun(
            self.things[0],
            self._eq_vc_data_keys,
            params=eq_params,
            profiles=self._eq_profiles,
            transforms=self._eq_transforms,
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
                transforms=self._sheet_source_transforms,
                profiles={},
            )
            plasma_surf_data["K_vc"] += sheet_source_data["K"]
        plasma_surf_data["K"] = plasma_surf_data["K_vc"]

        ## calc B at measurement points
        # field contribution
        if not self._field_fixed:
            Bcoil = self._field.compute_magnetic_field(
                self._measurement_coords,
                basis="rpz",
                source_grid=self._field_grid,
                params=field_params,
            )
        else:
            Bcoil = jnp.zeros_like(self._measurement_coords)

        # get plasma contribution
        if not self._vacuum:
            Bplasma = self._compute_A_or_B_from_CurrentPotentialField(
                self._field,  # this is unused, just pass a dummy variable in
                self._measurement_coords,
                source_grid=self._vc_source_grid,
                compute_A_or_B="B",
                data=plasma_surf_data,
            )
        else:
            Bplasma = jnp.zeros_like(self._measurement_coords)
        B = Bplasma + Bcoil
        if self._use_directions:
            B = dot(B, self._directions)

        if self._field_fixed:
            # add fixed field contribution at end, after we've already
            # dotted Bplasma, as it is already dotted with the directions,
            # if passed in.
            B += self._B_from_field
        return B.flatten()
