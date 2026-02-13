"""Classes for computing magnetic diagnostic measurements."""

from abc import ABC

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.geometry import (
    FourierPlanarCurve,
    FourierRZCurve,
    FourierXYCurve,
    FourierXYZCurve,
    SplineXYZCurve,
)
from desc.grid import LinearGrid
from desc.magnetic_fields import FourierCurrentPotentialField
from desc.optimizable import optimizable_parameter
from desc.utils import dot, errorif, xyz2rpz, xyz2rpz_vec

from ._core import AbstractDiagnostic


class PointBMeasurements(AbstractDiagnostic):
    """Target B at a point in real space, outside the plasma.

    This diagnostic will calculate the magnetic field at given points,
    intended for use to compare to experimental measurements for reconstruction.

    The measurement points should outside the plasma, otherwise
    the plasma contribution will be incorrectly calculated.

    Parameters
    ----------
    measurement_coords : (n,3) ndarray
        Array of n points at which the magnetic field B is measured.
    diag_fixed : bool, optional
        whether or not to fix the diagnostic's DOFs during optimization.
        Currently must be True.
    field_fixed : bool, optional
        whether or not to fix the external field's DOFs during optimization.
        False by default. Set to True if the field is not changing during the
        optimization.
    eq_fixed : bool, optional
        whether or not to fix the equilibrium's DOFs during optimization.
        False by default. Set to True if the equilibrium is not changing during the
        optimization.
    vacuum : bool, optional
        whether the Equilibrium is vacuum, in which case plasma contribution to B won't
        be calculated.
    directions : array (n,3), optional
        the directions of the measured B field for each of the n sensors, i.e. if a
        sensor is measuring the poloidal field and is located at (R,phi,Z) = (1,0,0),
        its ``direction`` might be [0,0,+/- 1]. If not passed, will default to instead
        comparing the entire magnetic field vector at the points passed in.
    basis : {"rpz","xyz"}
        the basis used for ``measurement_coords``, ``directions`` and ``target``,
        assumed to be "rpz" by default. if "xyz", will convert the input arrays
        into "rpz".
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source.
        Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    vc_source_grid : LinearGrid
        LinearGrid to use for the (non-singular) integral over the virtual casing
        principle current to calculate the flux loop contribution from the
        plasma currents. Must have endpoint=False and sym=False and be linearly
        spaced in theta and zeta, with nodes only at rho=1.0


    """

    _coordinates = "rtz"
    _units = "(T)"
    _print_value_fmt = "Point B Measurement Error: "
    _print_error = True
    _static_attrs = [
        "_use_directions",
        "_sheet_current",
        "_vacuum",
        "_eq_vc_data_keys",
        "_field_fixed",
        "_sheet_data_keys",
        "_compute_A_or_B_from_CurrentPotentialField",
        "_name",
    ]

    def __init__(
        self,
        measurement_coords,
        *,
        directions=None,
        basis="rpz",
        name="Magnetic-Point-Measurement",
    ):
        measurement_coords = np.atleast_2d(measurement_coords)
        if directions is not None:
            self._use_directions = True
            directions = np.atleast_2d(directions)
            assert (
                directions.shape == measurement_coords.shape
            ), "Must pass in same number of direction vectors as measurements"
            # make the direction vectors unit norm
            directions = directions / jnp.linalg.norm(directions, axis=1)[:, None]
        else:
            self._use_directions = False
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
                # just fill the directions with dummy values
                directions = np.zeros_like(measurement_coords)

            measurement_coords = xyz2rpz(measurement_coords)
        self._R = measurement_coords[:, 0]
        self._phi = measurement_coords[:, 1]
        self._Z = measurement_coords[:, 2]
        self._direction_R = (
            directions[:, 0] if self._use_directions else jnp.zeros_like(self._R)
        )
        self._direction_phi = (
            directions[:, 1] if self._use_directions else jnp.zeros_like(self._phi)
        )
        self._direction_Z = (
            directions[:, 2] if self._use_directions else jnp.zeros_like(self._Z)
        )

        super().__init__(name=name)

    def build(self, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.
        """
        measurement_coords = np.vstack([self._R, self._phi, self._Z]).T
        self._all_eval_x_rpz = measurement_coords

        # dim_f is number of B components we have,
        # which is coords.size, if no directions are given,
        # else it is equal to how many points we are evaluating the
        # B measurement at (coords.shape[0])
        self._dim_f = (
            measurement_coords.size
            if not self._use_directions
            else measurement_coords.shape[0]
        )

        super().build()

    @optimizable_parameter
    @property
    def R(self):
        """R coordinates of the measurement points."""
        return self._R

    @R.setter
    def R(self, new):
        if len(new) == self._R.size:
            self._R = jnp.asarray(new)
        else:
            raise ValueError(
                f"R should have the same size as the existing array, got {len(new)}"
                + f"for existing array with {self._R.size} points."
            )

    @optimizable_parameter
    @property
    def phi(self):
        """Torodial phi coordinates of the measurement points."""
        return self._phi

    @phi.setter
    def phi(self, new):
        if len(new) == self._R.size:
            self._phi = jnp.asarray(new)
        else:
            raise ValueError(
                f"phi should have the same size as the existing array, got {len(new)}"
                + f"for existing array with {self._phi.size} points."
            )

    @optimizable_parameter
    @property
    def Z(self):
        """R coordinates of the measurement points."""
        return self._Z

    @Z.setter
    def Z(self, new):
        if len(new) == self._Z.size:
            self._Z = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z should have the same size as the existing array, got {len(new)}"
                + f"for existing array with {self._Z.size} points."
            )

    @optimizable_parameter
    @property
    def direction_R(self):
        """R component of the direction of measurement."""
        return self._direction_R

    @direction_R.setter
    def direction_R(self, new):
        if len(new) == self._direction_R.size:
            self._direction_R = jnp.asarray(new)
        else:
            raise ValueError(
                "direction_R should have the same size as the existing array"
                f", got {len(new)}"
                + f"for existing array with {self._direction_R.size}."
            )

    @optimizable_parameter
    @property
    def direction_phi(self):
        """Toroidal phi component of the direction of measurement."""
        return self._direction_Z

    @direction_phi.setter
    def direction_phi(self, new):
        if len(new) == self._direction_phi.size:
            self._direction_phi = jnp.asarray(new)
        else:
            raise ValueError(
                "direction_phi should have the same size as the existing array"
                f", got {len(new)}"
                + f"for existing array with {self._direction_phi.size}."
            )

    @optimizable_parameter
    @property
    def direction_Z(self):
        """Z component of the direction of measurement."""
        return self._direction_Z

    @direction_Z.setter
    def direction_Z(self, new):
        if len(new) == self._direction_Z.size:
            self._direction_Z = jnp.asarray(new)
        else:
            raise ValueError(
                "direction_Z should have the same size as the existing array"
                f", got {len(new)}"
                + f"for existing array with {self._direction_Z.size}."
            )

    def compute_normalization(self, scales):
        """Compute normalization for diagnostic from input scales dictionary."""
        return scales["B"]

    def compute_measurement(
        self,
        eq,
        field,
        *,
        vc_source_grid=None,
        field_grid=None,
        vacuum=False,
    ):
        """Compute the B measurement from a given equilibrium and external field.

        Parameters
        ----------
        eq : Equilibrium
            The equilibrium to compute the diagnostic from.
        field : MagneticField
            The external magnetic field to compute the diagnostic from.
        vc_source_grid : LinearGrid, optional
            LinearGrid to use for the (non-singular) integral over the virtual casing
            principle current to calculate the flux loop contribution from the
            plasma currents. Must have endpoint=False and sym=False and be linearly
            spaced in theta and zeta, with nodes only at rho=1.0. Defaults to
            a grid with same M,N as the equilibrium.
        field_grid : Grid, optional
            Grid containing the nodes to evaluate field source.
            Defaults to the default for the given field, see the docstring of the
            field object for the specific default.
        vacuum : bool, optional
            whether the Equilibrium is vacuum, in which case plasma
            contribution to B won't
            be calculated.

        Returns
        -------
        f : array
            B from plasma and external field at given measurement coordinates,
            These are always returned in rpz basis, and if ``directions`` is None, this
            is equal to ``B.flatten()`` where ``B`` is an ``(n,3)`` array
            corresponding to the magnetic field at the given measurement coordinates.
        """
        self._eq = eq
        self._sheet_current = hasattr(eq.surface, "Phi_mn")
        self._field = field
        self._vc_source_grid = vc_source_grid
        self._field_grid = field_grid
        self._vacuum = vacuum
        self.build()
        return self._compute_from_params(eq.params_dict, field.params_dict)

    def _compute_from_params(
        self, eq_params=None, field_params=None, diag_params=None, constants=None
    ):
        """Compute point B measurements in "rpz" basis.

        Parameters
        ----------
        diag_params : dict
            Dictionary of diagnostic degrees of freedom,
            eg PointBMeasurements.params_dict
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
        # TODO: change once diags can be optimized
        diag_params = self.params_dict

        measurement_coords = jnp.vstack(
            [diag_params["R"], diag_params["phi"], diag_params["Z"]]
        ).T

        measurement_directions = jnp.vstack(
            [
                diag_params["direction_R"],
                diag_params["direction_phi"],
                diag_params["direction_Z"],
            ]
        ).T

        plasma_surf_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
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
        # TODO if diag position allowed to vary...
        # also do we even want this compute method here??
        if not self._field_fixed:
            Bcoil = self._field.compute_magnetic_field(
                measurement_coords,
                basis="rpz",
                source_grid=self._field_grid,
                params=field_params,
            )
        else:
            Bcoil = jnp.zeros_like(measurement_coords)

        # get plasma contribution
        if not self._vacuum:
            Bplasma = self._compute_A_or_B_from_CurrentPotentialField(
                self._field,  # this is unused, just pass a dummy variable in
                measurement_coords,
                source_grid=self._vc_source_grid,
                compute_A_or_B="B",
                data=plasma_surf_data,
            )
        else:
            Bplasma = jnp.zeros_like(measurement_coords)
        B = Bplasma + Bcoil
        if self._use_directions:
            B = dot(B, measurement_directions)

        if self._field_fixed:
            # add fixed field contribution at end, after we've already
            # dotted Bplasma, as it is already dotted with the directions,
            # if passed in.
            B += self._B_from_field
        return B.flatten()

    def _compute_data(self, eq_params, field_params, diag_params=None, constants=None):
        """Compute necessary auxiliary data for diagnostic from params."""
        if constants is None:
            constants = self.constants
        if diag_params is None:
            diag_params = self.params_dict

        data = {
            "measurement_directions": jnp.vstack(
                [
                    diag_params["direction_R"],
                    diag_params["direction_phi"],
                    diag_params["direction_Z"],
                ]
            ).T
        }
        # we recompute this here for cases when the diagnostic is allowed to be change
        # during optimization
        data["all_eval_x_rpz"] = jnp.vstack(
            [diag_params["R"], diag_params["phi"], diag_params["Z"]]
        ).T
        return data

    def _compute(self, B, data=None):
        """Compute diagnostic, using pre-computed B at the points it needs."""
        if self._use_directions:
            B = dot(B, data["measurement_directions"])
        return B.flatten()


## RogowskiCoil classes


class AbstractRogowskiCoil(AbstractDiagnostic, ABC):
    """Base class for (Full) Rogowski coil diagnostics.

    return mu0*I, where I is the current enclosed by the coil,
    in units of T*m.

    """

    def build(self):
        """Build constant arrays."""
        # basically just need to define self._all_eval_x_rpz here
        self._all_eval_x_rpz = self.compute(
            "x",
            grid=self._eval_grid,
            basis="rpz",
        )["x"]
        self._dim_f = 1  # Rogowski coil returns a single value
        super().build()

    def compute_measurement(
        self,
        eq,
        field,
        *,
        eval_grid=None,
        vc_source_grid=None,
        field_grid=None,
        vacuum=False,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
    ):
        """Compute the diagnostic's measurement from a given eq and field."""
        from desc.magnetic_fields._current_potential import (
            _compute_A_or_B_from_CurrentPotentialField,
        )

        if vc_source_grid is None:
            vc_source_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP if eq.N > 0 else 64,
                sym=False,
            )
        if eval_grid is None:
            eval_grid = LinearGrid(N=2 * self.N * getattr(self, "NFP", 1) + 5)

        flux_loop_data = self.compute(["x", "x_s"], grid=eval_grid)

        # need to just compute B from coils then compute B from plasma
        Bcoil = field.compute_magnetic_field(
            flux_loop_data["x"],
            basis="rpz",
            source_grid=field_grid,
            chunk_size=bs_chunk_size,
        )

        if not vacuum:  # compute B plasma contribution
            plasma_surf_data = eq.compute(
                ["K_vc", "R", "phi", "Z"], grid=vc_source_grid
            )

            plasma_surf_data["x"] = jnp.vstack(
                [plasma_surf_data["R"], plasma_surf_data["phi"], plasma_surf_data["Z"]]
            ).T
            if isinstance(eq.surface, FourierCurrentPotentialField):
                sheet_source_data = eq.surface.compute(["K"], grid=vc_source_grid)
                plasma_surf_data["K_vc"] += sheet_source_data["K"]
            plasma_surf_data["K"] = plasma_surf_data["K_vc"]
            Bplasma = _compute_A_or_B_from_CurrentPotentialField(
                self._coils,  # this is unused since we pass in the source data
                flux_loop_data["x"],
                source_grid=vc_source_grid,
                compute_A_or_B="B",
                data=plasma_surf_data,
                chunk_size=B_plasma_chunk_size,
            )
        else:
            Bplasma = jnp.zeros_like(eval_grid.nodes)

        B = Bplasma + Bcoil

        flux_loop_data["ds"] = eval_grid.spacing[:, 2]

        return self._compute(B, flux_loop_data)

    def _compute_data(
        self, diag_params=None, eq_params=None, field_params=None, constants=None
    ):
        """Compute necessary auxiliary data for diagnostic."""
        data = self.compute(["x_s", "ds"], grid=constants["eval_grid"], basis="rpz")
        # we recompute this here for cases when the diagnostic is allowed to be change
        # during optimization
        data["all_eval_x_rpz"] = data["x"]
        return data

    def _compute(self, B, data):
        """Given the magnetic field, compute the line integral sum(B dot dl).

        A_or_B should be the field vector B of shape (num_points, 3).

        Data should be a dictionary containing 'x_s' (the tangent vector)
        and 'ds' (the differential length elements) at the points at which
        B is evaluated, ``self._all_eval_x_rpz``
        """
        B_dot_dl = jnp.sum(B * data["x_s"], axis=1)
        mu0I = jnp.sum(B_dot_dl * data["ds"])
        return mu0I


class RogowskiCoilFourierRZ(AbstractRogowskiCoil, FourierRZCurve):
    """Rogowski coil given by Fourier series for R,Z in terms of toroidal angle phi."""

    def __init__(
        self,
        R_n=10,
        Z_n=0,
        modes_R=None,
        modes_Z=None,
        eval_grid=None,
        NFP=1,
        sym="auto",
        name="",
    ):
        self._eval_grid = eval_grid
        # AbstractDiagnostic will forward these to FourierRZCurve.
        super().__init__(
            R_n=R_n,
            Z_n=Z_n,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=NFP,
            sym=sym,
            name=name,
        )


class RogowskiCoilFourierXYZ(AbstractRogowskiCoil, FourierXYZCurve):
    """Rogowski coil given by Fourier series for X,Y,Z in terms of an angle s."""

    def __init__(
        self,
        X_n=[0, 10, 2],
        Y_n=[0, 0, 0],
        Z_n=[-2, 0, 0],
        eval_grid=None,
        modes=None,
        name="",
    ):
        self._eval_grid = eval_grid
        super().__init__(
            X_n=X_n,
            Y_n=Y_n,
            Z_n=Z_n,
            modes=modes,
            name=name,
        )


class RogowskiCoilFourierPlanar(AbstractRogowskiCoil, FourierPlanarCurve):
    """Rogowski coil that lies in a plane."""

    def __init__(
        self,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        r_n=2,
        eval_grid=None,
        modes=None,
        basis="xyz",
        name="",
    ):
        self._eval_grid = eval_grid
        super().__init__(
            center=center,
            normal=normal,
            r_n=r_n,
            modes=modes,
            basis=basis,
            name=name,
        )


class RogowskiCoilFourierXY(AbstractRogowskiCoil, FourierXYCurve):
    """Rogowski coil that lies in a plane, given by Fourier series for X and Y."""

    def __init__(
        self,
        center=[10, 0, 0],
        normal=[0, 1, 0],
        X_n=[0, 2],
        Y_n=[2, 0],
        eval_grid=None,
        modes=None,
        basis="xyz",
        name="",
    ):
        self._eval_grid = eval_grid
        super().__init__(
            center=center,
            normal=normal,
            X_n=X_n,
            Y_n=Y_n,
            modes=modes,
            basis=basis,
            name=name,
        )


class RogowskiCoilSplineXYZ(AbstractRogowskiCoil, SplineXYZCurve):
    """Rogowski coil given by spline knots in X,Y,Z."""

    def __init__(
        self,
        X,
        Y,
        Z,
        knots=None,
        eval_grid=None,
        method="cubic",
        name="",
    ):
        self._eval_grid = eval_grid
        super().__init__(
            X=X,
            Y=Y,
            Z=Z,
            knots=knots,
            method=method,
            name=name,
        )
