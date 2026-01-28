"""Objectives for experimental reconstruction problems."""

from abc import ABC, abstractmethod

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms, xyz2rpz, xyz2rpz_vec
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, dot, errorif, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


class _MagneticMeasurement(_Objective, ABC):
    """Base class for magnetic diagnostics objectives.

    Subclasses of this class must implement a _compute method, which should
    compute the same values as the compute method, but _compute should accept
    the total B field at the necessary evaluation points as its only input,
    and so should not compute B.

    Subclasses must also define a ``self._all_eval_x_rpz`` attribute in their build
    method, which contains every R,phi,Z point at which the total B must be evaluated
    at. These are the points at which _compute should expect the B passed in to be
    computed at.
    """

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        super().build(use_jit=use_jit, verbose=verbose)
        assert hasattr(self, "_all_eval_x_rpz"), (
            "Subclasses of _MagneticMeasurement must define self._all_eval_x_rpz in"
            "  their build method!"
        )

    @abstractmethod
    def _compute(self, B):
        """Compute magnetic diagnostic signal using passed-in total field."""


class PointBMeasurement(_MagneticMeasurement):
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
        "_B_plasma_chunk_size",
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
        B_plasma_chunk_size=None,
    ):
        self._field = field
        self._field_grid = field_grid
        self._vc_source_grid = vc_source_grid
        self._B_plasma_chunk_size = B_plasma_chunk_size
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
        else:
            # FIXME: this is just because of laziness/reducing complexity
            assert bounds is None, "Cannot use bounds without specifying directions"
        errorif(
            basis not in ["xyz", "rpz"],
            ValueError,
            f"basis must be either rpz or xyz, instead got {basis}",
        )

        if basis == "rpz":
            pass
        elif basis == "xyz":
            if self._use_directions:
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

        self._all_eval_x_rpz = self._measurement_coords

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
                chunk_size=self._B_plasma_chunk_size,
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

    def _compute(self, B):
        """Same as above, but accepts pre-computed B at the points it needs."""
        if self._use_directions:
            B = dot(B, self._directions)
        return B.flatten()


class MagneticDiagnostics(_Objective):
    """Wrapper class for signals from a set of magnetic diagnostics.

    This objective will calculate the magnetic field at the points needed for
    the set of magnetic diagnostics, then pass that field to each sub-objective
    for it to compute the diagnostic signal. This objective will then
    return a concatenated array of each of the sub-objective signals.

    This class exists to maximize the possible vectorization when computing
    the field from the external coils and the plasma current at the necessary
    evaluation points.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium from which the plasma constribution to the magnetic field will
        be calculated, if not vacuum.
    field : MagneticField
        External field produced by coils or other sources outside the plasma.
    magnetic_diagnostics: tuple of _Objective
        Tuple of the magnetic diagnostic signals, for example
        (PointBMeasurement, FluxLoop). each of these should have a self._all_eval_x_rpz
        attribute, which will be used by this wrapper class to compute the B field
        over every evaluation point needed for the sub-objectives
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source.
        Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    vc_source_grid : LinearGrid
        LinearGrid to use for the (non-singular) integral over the virtual casing
        principle current to calculate the flux loop contribution from the
        plasma currents. Must have endpoint=False and sym=False and be linearly
        spaced in theta and zeta, with nodes only at rho=1.0
    field_fixed : bool, optional
        whether or not to fix the external field's DOFs during optimization.
        False by default. Set to True if the field is not changing during the
        optimization.
    vacuum : bool, optional
        whether the Equilibrium is vacuum, in which case plasma contribution to B won't
        be calculated.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    B_plasma_chunk_size : int or None
        Size to split singular integral computation into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``bs_chunk_size``.

    """

    __doc__ = __doc__.rstrip() + collect_docs()

    _coordinates = "rtz"
    _units = "(~)"
    # these will be determined by sub objectives, I guess need to
    # define a custom print for this that goes
    # thru each sub objective and calls _compute or maybe the sub prints... annoying
    _print_value_fmt = "Magnetic Diagnostic Error: "
    _print_error = True
    _static_attrs = _Objective._static_attrs + [
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
        magnetic_diagnostics,
        *,
        field_grid=None,
        vc_source_grid=None,
        field_fixed=False,
        vacuum=False,
        bounds=None,  # unused
        target=None,  # unused
        weight=1,
        normalize=True,
        normalize_target=True,
        # unused (sub-objective? how to deal with, or
        # maybe enforce all are the same? or allow a list? no that wont work)
        loss_function=None,
        deriv_mode="auto",
        name="Magnetic-Diagnostic-Error",
        jac_chunk_size=None,
    ):
        self._field = field
        self._field_grid = field_grid
        self._vc_source_grid = vc_source_grid
        if not isinstance(magnetic_diagnostics, (tuple, list)):
            magnetic_diagnostics = (magnetic_diagnostics,)
        assert len(
            magnetic_diagnostics
        ), "Must supply at least one diagnostic sub-objective"
        assert all(
            [isinstance(diag, _MagneticMeasurement) for diag in magnetic_diagnostics]
        ), "Must pass in only subclasses of _MagneticMeasurement"
        self._magnetic_diagnostics = magnetic_diagnostics
        self._vacuum = vacuum
        self._sheet_current = hasattr(eq.surface, "Phi_mn")
        things = [eq]
        self._field_fixed = field_fixed
        if not field_fixed:
            things.append(field)
        warnif(
            target is not None or bounds is not None,
            UserWarning,
            "MagneticDiagnostics class"
            "does not require target or bounds, these will be"
            "overridden by the sub-objectives' targets/bounds",
        )
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
        for diag in self._magnetic_diagnostics:
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

        self._all_eval_x_rpz = np.vstack(
            [diag._all_eval_x_rpz for diag in self._magnetic_diagnostics]
        )
        # make the indices of the overall eval_x_rpz array corresponding
        # to each sub-objective
        self._eval_x_idxs = [
            np.arange(self._magnetic_diagnostics[0]._all_eval_x_rpz.shape[0])
        ]
        for i in range(1, len(self._magnetic_diagnostics)):
            self._eval_x_idxs.append(
                np.arange(self._magnetic_diagnostics[i]._all_eval_x_rpz.shape[0])
                + self._eval_x_idxs[i - 1][-1]
                + 1
            )

        # dim_f is number of signals we have across all diags
        self._dim_f = np.sum([diag._dim_f for diag in self._magnetic_diagnostics])

        # to account for some sub-objectives having bounds and others having target,
        # just always use bounds here
        bounds_lower = np.hstack(
            [
                (
                    diag._target * np.ones(diag._dim_f)
                    if hasattr(diag, "_target")
                    else diag._bounds[0] * np.ones(diag._dim_f)
                )
                for diag in self._magnetic_diagnostics
            ]
        )
        bounds_upper = np.hstack(
            [
                (
                    diag._target * np.ones(diag._dim_f)
                    if hasattr(diag, "_target")
                    else diag._bounds[1] * np.ones(diag._dim_f)
                )
                for diag in self._magnetic_diagnostics
            ]
        )
        self._bounds = (bounds_lower, bounds_upper)

        self._weight = self._weight * np.hstack(
            [diag._weight * np.ones(diag._dim_f) for diag in self._magnetic_diagnostics]
        )

        # set normalizations to each sub objective normalization
        # these also I think must be expanded out for each
        if self._normalize:
            self._normalization = np.hstack(
                [
                    diag._normalization * np.ones(diag._dim_f)
                    for diag in self._magnetic_diagnostics
                ]
            )
        if self._normalize_target:
            # we need every sub diagnostic's normalize_target to be the same
            # otherwise the targets will be inconsistent with some normalized and
            # some not normalized
            assert np.all(
                [
                    diag._normalize_target == self._normalize_target
                    for diag in self._magnetic_diagnostics
                ]
            ), "if normalize_target is True, it must be true for every sub-objective"
        # pre-calc field contrib to B if field are fixed
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

    def compute(self, eq_params=None, field_params=None, constants=None):
        """Compute B at each sub-diagnostic, then compute each sub-diagnostic.

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
            concatenated array of each magnetic diagnostic's signal.

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

        # compute diagnostic signals using the _compute method of each sub, which
        # accepts the total B at the diagnostic eval pts as input.
        signals = jnp.hstack(
            [
                diag._compute(B[idx, :])
                for diag, idx in zip(self._magnetic_diagnostics, self._eval_x_idxs)
            ]
        )
        return signals
