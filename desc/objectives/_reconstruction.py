"""Objectives for experimental reconstruction problems."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms, xyz2rpz
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, dot

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class PointBMeasurement(_Objective):
    """Target B at a point in real space, outside the plasma.

    This objective will calculate the magnetic field at a point,
    intended for use to compare to experimental measurements for reconstruction.

    The equilibrium and possibly a coilset are allowed to vary, but the
    measurement location in real space is held fixed.

    The measurement point should be at a point outside the plasma, otherwise
    the plasma contribution will be incorrectly calculated.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium from which the plasma constribution to the magnetic field will
        be calculated, if not vacuum.
    coils : CoilSet
        Coilset that supports the equilibrium. If coils_fixed is True,
        their contribution to the magnetic field measurements will be pre-calculated.
    measurement_coords : (n,3) ndarray
        Array of points at which the magnetic field B is measured.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f which is the
        number of measurement points if ``direction`` is passed in, or
        3 times the number of measurement points if ``direction`` is not passed in.
        If ``direction`` is not passed in, target is assumed to be the
        flattened array of the field vector at the measurement points i.e.
        ``B_array.flatten()``.
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
    vc_source_grid : LinearGrid
        LinearGrid to use for the (non-singular) integral for the virtual casing
        principle to calculate the flux loop contribution from the
        plasma currents. Must have endpoint=False and sym=False and be linearly
        spaced in theta and zeta, with nodes only at rho=1.0
    basis : {"rpz","xyz"}
        the basis used for measurement_coords, assumed to be "rpz" by default.
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
    coils_fixed : bool, optional
        whether or not to fix the coilset DOFs during optimization. False by default
    vacuum : bool, optional
        whether eq is vacuum, in which case plasma contribution to B won't be
        calculated.
    directions : array (n,3), optional
        the directions of the measured B field for each of the n sensors, i.e. if a
        sensor is measuring the poloidal field and is located at (R,phi,Z) = (1,0,0),
        its ``direction`` might be [0,0,+/- 1]. If not passed, will default to instead
        comparing the entire magnetic field vector at the points passed in.

    """

    _coordinates = "rtz"
    _units = "(T)"
    _print_value_fmt = "Point B Measurement: "

    def __init__(
        self,
        eq,
        coilset,
        measurement_coords,
        field_grid=None,
        vc_source_grid=None,
        coils_fixed=False,
        vacuum=False,
        directions=None,
        basis="rpz",
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="Magnetic-Point-Measurement",
        jac_chunk_size=None,
    ):
        # TODO: I want the output to be like "measurement error", but that means
        # having the compute method subtract the measured values, and then the
        # target should be zero, but I'd like target to be the diagnostic targets.
        # could also just define a custom print_value to use
        # compute - target instead of just compute
        # TODO: change coils name to field and make naming consistent
        # TODO: use new docstring collect fxn
        if target is None and bounds is None:
            target = 0
        self._coils = coilset
        self._field_grid = field_grid
        self._vc_source_grid = vc_source_grid
        if basis == "rpz":
            self._measurement_coords = measurement_coords
        elif basis == "xyz":
            self._measurement_coords = xyz2rpz(measurement_coords)
        else:
            raise ValueError(f"basis must be either rpz or xyz, instead got {basis}")
        if directions is not None:
            assert (
                directions.shape == measurement_coords.shape
            ), "Must pass in same number of direction vectors as measurements"
        self._directions = directions
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
        from desc.magnetic_fields._current_potential import (
            _compute_A_or_B_from_CurrentPotentialField,
        )

        self._compute_A_or_B_from_CurrentPotentialField = (
            _compute_A_or_B_from_CurrentPotentialField
        )
        eq = self._eq

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

        eq_transforms = get_transforms(
            self._eq_vc_data_keys, self._eq, grid=self._vc_source_grid
        )
        eq_profiles = get_profiles(
            self._eq_vc_data_keys, self._eq, grid=self._vc_source_grid
        )

        # dim_f is number of B components we  have,
        # which is coords.size, if no directions are given,
        # else it is equal to how many points we are evaluating the
        # B measurement at (coords.shape[0])
        self._dim_f = (
            self._measurement_coords.size
            if self._directions is None
            else self._measurement_coords.shape[0]
        )

        # TODO: should probably call this "field" since doesn't need to
        # be coils
        # pre-calc coil contrib to B if coils are fixed
        if self._coils_fixed:
            B_from_coils = self._coils.compute_magnetic_field(
                self._measurement_coords,
                basis="rpz",
                source_grid=self._field_grid,
            )

        self._constants = {
            "quad_weights": 1.0,
            "field_grid": self._field_grid,
            "measurement_coords": self._measurement_coords,
            "equil_transforms": eq_transforms,
            "equil_profiles": eq_profiles,
            "vc_source_grid": self._vc_source_grid,
            "directions": self._directions,
        }
        if self._coils_fixed:
            self._constants["B_from_coils"] = (
                B_from_coils
                if self._directions is None
                else dot(B_from_coils, self._directions)
            )
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
        """Compute point B measurements.

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
        f : array
            B from plasma and external field at given measurement coordinates.

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

        # calc B at measurement points
        if not self._coils_fixed:
            Bcoil = self._coils.compute_magnetic_field(
                constants["measurement_coords"],
                basis="rpz",
                source_grid=constants["field_grid"],
                params=field_params,
            )
        else:
            Bcoil = jnp.zeros_like(self._measurement_coords)

        # get plasma contribution
        if not self._vacuum:
            Bplasma = self._compute_A_or_B_from_CurrentPotentialField(
                self._coils,  # this is unused, just pass a dummy variable in
                constants["measurement_coords"],
                source_grid=constants["vc_source_grid"],
                compute_A_or_B="B",
                data=plasma_surf_data,
            )
        else:
            Bplasma = jnp.zeros_like(self._measurement_coords)
        B = Bplasma + Bcoil
        if constants["directions"] is not None:
            B = dot(B, constants["directions"])

        if self._coils_fixed:
            B += constants["B_from_coils"]
        return B.flatten()
