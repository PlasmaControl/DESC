"""Objectives for experimental reconstruction problems."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, errorif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


class MeasurementError(_Objective):
    """Objective for signals from a set of diagnostics.

    This objective will calculate the magnetic field at the points needed for
    the set of diagnostics, then pass that field to each sub-objective
    for it to compute the diagnostic signal. This objective will then
    return a concatenated array of each of the diagnostic signals.

    This class exists to maximize the possible vectorization when computing
    the field from the external coils and the plasma current at the necessary
    evaluation points.

    The class may also accept a symmetric, pos-def matrix for its weight,
    which will be assumed to be the inverse covariance matrix of the
    measurement error noise. If so, then a generalized least squares
    loss function will be used, (error.T @ W @ error) rather than the
    usual assumption of uncorrelated errors, (W_ii * error.T @ error)

    NOTE: any grids that control discretization at the diagnostic-level, such
    as the discretization of a flux loop for calculating the flux loop signal,
    should be assigned to that diagnostic, and will not be passed to this objective.
    This grid is determined when initializing the diagnostic object, or can be
    changed by accessing that diagnostic's attributes after initialization.


    Parameters
    ----------
    eq : Equilibrium
        Equilibrium from which the plasma constribution to the magnetic field will
        be calculated, if not vacuum.
    field : MagneticField
        External field produced by coils or other sources outside the plasma.
    diagnostics: DiagnosticSet
        DiagnosticSet containing the diagnostics to compute synthetic signals,
        for example DiagnosticSet(PointBMeasurements, RogowskiCoilFourierXYZ).
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source.
        Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    vc_source_grid : LinearGrid
        LinearGrid to use for the (non-singular) integral over the virtual casing
        principle current to calculate the flux loop contribution from the
        plasma currents. Must have endpoint=False and sym=False and be linearly
        spaced in theta and zeta, with nodes only at rho=1.0. Defaults to
        LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP if eq.N > 0 else 64)
    field_fixed : bool, optional
        whether or not to fix the external field's DOFs during optimization.
        False by default. Set to True if the field is not changing during the
        optimization.
    vacuum : bool, optional
        whether the Equilibrium is vacuum, in which case the plasma contribution to B
        won't be calculated.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    B_plasma_chunk_size : int or None
        Size to split plasma current surface integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``bs_chunk_size``.

    """

    __doc__ = __doc__.rstrip() + collect_docs()

    _coordinates = "rtz"
    _units = "(~)"
    # TODO: units will be determined by sub objectives, I guess need to
    # define a custom print for this that goes
    # thru each sub objective and calls _compute or something
    _print_value_fmt = "Diagnostic Error: "
    _print_error = True
    _static_attrs = _Objective._static_attrs + [
        "_sheet_current",
        "_vacuum",
        "_eq_vc_data_keys",
        "_field_fixed",
        "_diag_fixed",
        "_sheet_data_keys",
        "_compute_A_or_B_from_CurrentPotentialField",
        "_diagnostics",
    ]

    def __init__(
        self,
        eq,
        field,
        diagnostics,
        *,
        field_grid=None,
        vc_source_grid=None,
        field_fixed=False,
        diag_fixed=True,
        vacuum=False,
        bounds=None,
        target=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="Diagnostic-Error",
        jac_chunk_size=None,
    ):
        from desc.diagnostics import DiagnosticSet  # local to avoid circular import

        # TODO: how best to handle the target and bounds? easiest is to
        # allow diagnostics to accept target and bounds and gather them up
        # here, but sort of makes them into objectives if we do that...
        # maybe some sort of utility for a DiagnosticSet which would accept
        # a list or dict corresponding to its submembers and return the
        # target in the correct order for the objective? or use
        # pytree stuff for this, but that is still not so easy. Probably
        # pytree is best bet though
        self._field = field
        self._field_grid = field_grid
        self._vc_source_grid = vc_source_grid
        if not isinstance(diagnostics, DiagnosticSet):
            diagnostics = DiagnosticSet(diagnostics)
        assert isinstance(diagnostics, DiagnosticSet)

        self._diagnostics = diagnostics
        self._vacuum = vacuum
        self._sheet_current = hasattr(eq.surface, "Phi_mn")
        things = [eq]
        self._field_fixed = field_fixed
        if not field_fixed:
            things.append(field)
        # TODO: probably want target bounds etc to
        # be pytree logic like in Coils stuff... annoying
        # but necessary
        errorif(
            diag_fixed is False,
            NotImplementedError,
            "Currently diag_fixed must be True, optimizing"
            " diagnostics not yet supported",
        )
        self._diag_fixed = diag_fixed
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
        # must "build" our diagnostic set to make sure
        # it gets populated with the necessary data
        self._diagnostics.build(verbose=0)

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = self._diagnostics.compute_normalization(scales)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")

        timer.start("Precomputing transforms")
        for diag in self._diagnostics:
            diag.build(verbose=0)
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

        # TODO: change when diag_fixed=False
        self._all_eval_x_rpz = self._diagnostics._all_eval_x_rpz

        # dim_f is number of signals we have across all diags
        self._dim_f = self._diagnostics._dim_f

        # pre-calc field contrib to B if field are fixed
        # TODO: can only do if diags are fixed too, must change when diag_fixed=False
        if self._field_fixed:
            self._B_from_field = self._field.compute_magnetic_field(
                self._all_eval_x_rpz,
                basis="rpz",
                source_grid=self._field_grid,
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

    def compute(self, eq_params, params_2=None, params_3=None, constants=None):
        """Compute B at each diagnostic, then compute each diagnostic signal.

        Parameters
        ----------
        equil_params : dict
            Dictionary of eq degrees of freedom,
            eg Equilibrium.params_dict
        params_2 : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotentialField.params_dict or CoilSet.params_dict
        params_3 : dict
            Dictionary of diagnostic degrees of freedom,
            eg PointBMeasurements.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : array
            concatenated array of each diagnostic's signal.

        """
        if constants is None:
            constants = self.constants
        if self._diag_fixed:
            diag_params = self._diagnostics.params_dict
            field_params = params_2
        if self._field_fixed:
            field_params = self._field.params_dict

        # compute needed preliminary plasma surface data
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

        ## calc B at points required by diagnostics
        # field contribution
        if self._field_fixed:
            Bcoil = self._B_from_field
        else:
            Bcoil = self._field.compute_magnetic_field(
                self._all_eval_x_rpz,
                basis="rpz",
                source_grid=self._field_grid,
                params=field_params,
            )
        #  plasma contribution
        if not self._vacuum:
            Bplasma = self._compute_A_or_B_from_CurrentPotentialField(
                self._field,  # this is unused, just pass a dummy variable in
                self._all_eval_x_rpz,
                source_grid=self._vc_source_grid,
                compute_A_or_B="B",
                data=plasma_surf_data,
            )
        else:
            Bplasma = jnp.zeros_like(self._all_eval_x_rpz)
        B = Bplasma + Bcoil

        # compute diagnostic signals using the _compute method of the DiagnosticSet,
        # accepts the total B at the diagnostic eval pts as input as well as any
        # auxiliary data needed, which may depend on the eq or field params.
        diag_datas = self._diagnostics._compute_data(
            eq_params, field_params, diag_params, constants
        )
        signals = self._diagnostics._compute(B, data=diag_datas)
        return signals
