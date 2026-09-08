"""Objectives for targeting quasisymmetry."""

import warnings

from desc.backend import jnp
from desc.batching import vmap_chunked
from desc.compute import get_profiles, get_transforms
from desc.compute._omnigenity import (
    _omnigenity_mapping,
    _omnigenity_mapping_LandremanForm,
    _omnigenity_mapping_OOPS,
)
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, errorif, warnif
from desc.vmec_utils import ptolemy_linear_transform

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


class QuasisymmetryBoozer(_Objective):
    """Quasi-symmetry Boozer harmonics error.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Must be a LinearGrid with sym=False.
        Defaults to ``LinearGrid(M=M_booz, N=N_booz)``.
    helicity : tuple, optional
        Type of quasi-symmetry (M, N). Default = quasi-axisymmetry (1, 0).
    M_booz : int, optional
        Poloidal resolution of Boozer transformation. Default = 2 * eq.M.
    N_booz : int, optional
        Toroidal resolution of Boozer transformation. Default = 2 * eq.N.
    surf_batch_size: int
        Number of flux surfaces to compute simultaneously. Defaults to
        computing all flux surfaces simultaneously. Decrease to reduce
        memory required for computation.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _units = "(T)"
    _print_value_fmt = "Quasi-symmetry Boozer error: "
    _static_attrs = _Objective._static_attrs + ["_helicity", "_surf_batch_size"]

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        helicity=(1, 0),
        M_booz=None,
        N_booz=None,
        name="QS Boozer",
        jac_chunk_size=None,
        surf_batch_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.helicity = helicity
        self.M_booz = M_booz
        self.N_booz = N_booz
        self._surf_batch_size = surf_batch_size
        super().__init__(
            things=eq,
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

        self._print_value_fmt = "Quasi-symmetry ({},{}) Boozer error: ".format(
            self.helicity[0], self.helicity[1]
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
        eq = self.things[0]
        M_booz = self.M_booz or 2 * eq.M
        N_booz = self.N_booz or 2 * eq.N

        if self._grid is None:
            grid = LinearGrid(M=2 * M_booz, N=2 * N_booz, NFP=eq.NFP, sym=False)
        else:
            grid = self._grid

        errorif(grid.sym, ValueError, "QuasisymmetryBoozer grid must be non-symmetric")
        warnif(
            grid.num_theta < 2 * eq.M,
            RuntimeWarning,
            "QuasisymmetryBoozer objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            RuntimeWarning,
            "QuasisymmetryBoozer objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._data_keys = ["|B|_mn_B"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(
            self._data_keys,
            obj=eq,
            grid=grid,
            M_booz=M_booz,
            N_booz=N_booz,
        )
        matrix, _, idx = ptolemy_linear_transform(
            transforms["B"].basis.modes,
            helicity=self.helicity,
            NFP=transforms["B"].basis.NFP,
        )

        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "matrix": matrix,
            "idx": idx,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._dim_f = idx.size * grid.num_rho

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute quasi-symmetry Boozer harmonics error.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Symmetry breaking harmonics of B (T).

        """
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            surf_batch_size=self._surf_batch_size,
        )
        B_mn = data["|B|_mn_B"].reshape((constants["transforms"]["grid"].num_rho, -1))
        B_mn = constants["matrix"] @ B_mn.T
        # output order = (rho, mn).flatten(), ie all the surfaces concatenated
        # one after the other
        return B_mn[constants["idx"]].T.flatten()

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @helicity.setter
    def helicity(self, helicity):
        assert (
            (len(helicity) == 2)
            and (int(helicity[0]) == helicity[0])
            and (int(helicity[1]) == helicity[1])
        )
        if hasattr(self, "_helicity") and self._helicity != helicity:
            self._built = False
            warnings.warn("Re-build objective after changing the helicity!")
        self._helicity = helicity
        if hasattr(self, "_print_value_fmt"):
            self._print_value_fmt = "Quasi-symmetry ({},{}) Boozer error: ".format(
                self.helicity[0], self.helicity[1]
            )


class QuasisymmetryTwoTerm(_Objective):
    """Quasi-symmetry two-term error.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    helicity : tuple, optional
        Type of quasi-symmetry (M, N).

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = "rtz"
    _units = "(T^3)"
    _print_value_fmt = "Quasi-symmetry two-term error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        helicity=(1, 0),
        name="QS two-term",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.helicity = helicity
        super().__init__(
            things=eq,
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

        self._print_value_fmt = "Quasi-symmetry ({},{}) two-term error: ".format(
            self.helicity[0], self.helicity[1]
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
        eq = self.things[0]
        if self._grid is None:
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            RuntimeWarning,
            "QuasisymmetryTwoTerm objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            RuntimeWarning,
            "QuasisymmetryTwoTerm objective grid requires toroidal "
            "resolution for surface averages",
        )

        self._dim_f = grid.num_nodes
        self._data_keys = ["f_C"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "helicity": self.helicity,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] ** 3

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute quasi-symmetry two-term errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^3).

        """
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            helicity=constants["helicity"],
        )
        return data["f_C"]

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @helicity.setter
    def helicity(self, helicity):
        assert (
            (len(helicity) == 2)
            and (int(helicity[0]) == helicity[0])
            and (int(helicity[1]) == helicity[1])
        )
        if hasattr(self, "_helicity") and self._helicity != helicity:
            self._built = False
        self._helicity = helicity
        if hasattr(self, "_print_value_fmt"):
            self._print_value_fmt = "Quasi-symmetry ({},{}) error: ".format(
                self.helicity[0], self.helicity[1]
            )


class QuasisymmetryTripleProduct(_Objective):
    """Quasi-symmetry triple product error.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = "rtz"
    _units = "(T^4/m^2)"
    _print_value_fmt = "Quasi-symmetry error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="QS triple product",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        super().__init__(
            things=eq,
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
        eq = self.things[0]
        if self._grid is None:
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["f_T"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] ** 4 / scales["a"] ** 2

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute quasi-symmetry triple product errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^4/m^2).

        """
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return data["f_T"]


class Omnigenity(_Objective):
    """Omnigenity error.

    Errors are relative to a target field that is perfectly omnigenous,
    and are computed on a collocation grid in (ρ,η,α) coordinates.

    This objective assumes that the collocation point (θ=0,ζ=0) lies on the contour of
    maximum field strength ||B||=B_max.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to be optimized to satisfy the Objective.
    field : OmnigenousField
        Omnigenous magnetic field to be optimized to satisfy the Objective.
    eq_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for equilibrium data.
        Defaults to a linearly space grid on the rho=1 surface.
        Must be without stellarator symmetry.
    field_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for omnigenous field data.
        The grid nodes are given in the usual (ρ,θ,ζ) coordinates (with θ ∈ [0, 2π),
        ζ ∈ [0, 2π/NFP)), but θ is mapped to η and ζ is mapped to α. Defaults to a
        linearly space grid on the rho=1 surface. Must be without stellarator symmetry.
    M_booz : int, optional
        Poloidal resolution of Boozer transformation. Default = 2 * eq.M.
    N_booz : int, optional
        Toroidal resolution of Boozer transformation. Default = 2 * eq.N.
    eta_weight : float, optional
        Magnitude of relative weight as a function of η:
        w(η) = (`eta_weight` + 1) / 2 + (`eta_weight` - 1) / 2 * cos(η)
        Default value of 1 weights all nodes equally.
    eq_fixed: bool, optional
        Whether the Equilibrium `eq` is fixed or not.
        If True, the equilibrium is fixed and its values are precomputed, which saves on
        computation time during optimization and only ``field`` is allowed to change.
        If False, the equilibrium is allowed to change during the optimization and its
        associated data are re-computed at every iteration (Default).
    field_fixed: bool, optional
        Whether the OmnigenousField `field` is fixed or not.
        If True, the field is fixed and its values are precomputed, which saves on
        computation time during optimization and only ``eq`` is allowed to change.
        If False, the field is allowed to change during the optimization and its
        associated data are re-computed at every iteration (Default).
    surf_batch_size: int
        Number of flux surfaces to compute simultaneously. Defaults to
        computing all flux surfaces simultaneously. Decrease to reduce
        memory required for computation.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _static_attrs = _Objective._static_attrs + [
        "_eq_data_keys",
        "_eq_fixed",
        "_field_data_keys",
        "_field_fixed",
        "_helicity",
        "_surf_batch_size",
    ]

    _coordinates = "rtz"
    _units = "(T)"
    _print_value_fmt = "Omnigenity error: "

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
        eq_grid=None,
        field_grid=None,
        M_booz=None,
        N_booz=None,
        eta_weight=1,
        eq_fixed=False,
        field_fixed=False,
        name="omnigenity",
        jac_chunk_size=None,
        surf_batch_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._eq = eq
        self._field = field
        self._eq_grid = eq_grid
        self._field_grid = field_grid
        self.helicity = field.helicity
        self.M_booz = M_booz
        self.N_booz = N_booz
        self.eta_weight = eta_weight
        self._eq_fixed = eq_fixed
        self._field_fixed = field_fixed
        self._surf_batch_size = surf_batch_size
        if not eq_fixed and not field_fixed:
            things = [eq, field]
        elif eq_fixed and not field_fixed:
            things = [field]
        elif field_fixed and not eq_fixed:
            things = [eq]
        else:
            raise ValueError("Cannot fix both the eq and field.")
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
        if self._eq_fixed:
            eq = self._eq
            field = self.things[0]
        elif self._field_fixed:
            eq = self.things[0]
            field = self._field
        else:
            eq = self.things[0]
            field = self.things[1]

        M_booz = self.M_booz or 2 * eq.M
        N_booz = self.N_booz or 2 * eq.N

        # default grids
        if self._eq_grid is None and self._field_grid is not None:
            rho = self._field_grid.nodes[self._field_grid.unique_rho_idx, 0]
        elif self._eq_grid is not None and self._field_grid is None:
            rho = self._eq_grid.nodes[self._eq_grid.unique_rho_idx, 0]
        elif self._eq_grid is None and self._field_grid is None:
            rho = 1.0
        if self._eq_grid is None:
            eq_grid = LinearGrid(
                rho=rho, M=2 * M_booz, N=2 * N_booz, NFP=eq.NFP, sym=False
            )
        else:
            eq_grid = self._eq_grid
        if self._field_grid is None:
            field_grid = LinearGrid(
                rho=rho, theta=2 * field.M_B, N=2 * field.N_x, NFP=field.NFP, sym=False
            )
        else:
            field_grid = self._field_grid

        self._dim_f = field_grid.num_nodes
        self._eq_data_keys = ["|B|_mn_B"]
        self._field_data_keys = ["|B|", "theta_B", "zeta_B"]

        errorif(
            eq_grid.NFP != field_grid.NFP,
            msg="eq_grid and field_grid must have the same number of field periods",
        )
        errorif(eq_grid.sym, msg="eq_grid must not be symmetric")
        errorif(field_grid.sym, msg="field_grid must not be symmetric")
        field_rho = field_grid.nodes[field_grid.unique_rho_idx, 0]
        eq_rho = eq_grid.nodes[eq_grid.unique_rho_idx, 0]
        errorif(
            any(eq_rho != field_rho),
            msg="eq_grid and field_grid must be the same surface(s), "
            + f"eq_grid has surfaces {eq_rho}, "
            + f"field_grid has surfaces {field_rho}",
        )
        errorif(
            jnp.any(field.B_lm[: field.M_B] < 0),
            msg="|B| on axis must be positive! Check B_lm input.",
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._eq_data_keys, obj=eq, grid=eq_grid)
        eq_transforms = get_transforms(
            self._eq_data_keys,
            obj=eq,
            grid=eq_grid,
            M_booz=M_booz,
            N_booz=N_booz,
        )
        field_transforms = get_transforms(
            self._field_data_keys,
            obj=field,
            grid=field_grid,
        )

        # compute returns points on the grid of the field (dim_f = field_grid.num_nodes)
        # so set quad_weights to the field grid
        # to avoid it being incorrectly set in the super build
        w = field_grid.weights
        w *= jnp.sqrt(field_grid.num_nodes)

        self._constants = {
            "eq_profiles": profiles,
            "eq_transforms": eq_transforms,
            "field_transforms": field_transforms,
            "quad_weights": w,
        }

        if self._eq_fixed:
            # precompute the eq data since it is fixed during the optimization
            eq_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._eq_data_keys,
                params=self._eq.params_dict,
                transforms=self._constants["eq_transforms"],
                profiles=self._constants["eq_profiles"],
                surf_batch_size=self._surf_batch_size,
            )
            self._constants["eq_data"] = eq_data
        if self._field_fixed:
            # precompute the field data since it is fixed during the optimization
            field_data = compute_fun(
                "desc.magnetic_fields._core.OmnigenousField",
                self._field_data_keys,
                params=self._field.params_dict,
                transforms=self._constants["field_transforms"],
                profiles={},
                helicity=self.helicity,
                surf_batch_size=self._surf_batch_size,
            )
            self._constants["field_data"] = field_data

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            # average |B| on axis
            self._normalization = jnp.mean(field.B_lm[: field.M_B])

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1=None, params_2=None, constants=None):
        """Compute omnigenity errors.

        Parameters
        ----------
        params_1 : dict
            If eq_fixed=True, dictionary of field degrees of freedom,
            eg OmnigenousField.params_dict. Otherwise, dictionary of equilibrium degrees
            of freedom, eg Equilibrium.params_dict.
        params_2 : dict
            If eq_fixed=False and field_fixed=False, dictionary of field degrees of
            freedom, eg OmnigenousField.params_dict. Otherwise None.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        omnigenity_error : ndarray
            Omnigenity error at each node (T).

        """
        constants = self._get_deprecated_constants(constants)

        # sort parameters
        if self._eq_fixed:
            field_params = params_1
        elif self._field_fixed:
            eq_params = params_1
        else:
            eq_params = params_1
            field_params = params_2

        eq_grid = constants["eq_transforms"]["grid"]
        field_grid = constants["field_transforms"]["grid"]

        # compute eq data
        if self._eq_fixed:
            eq_data = constants["eq_data"]
        else:
            eq_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._eq_data_keys,
                params=eq_params,
                transforms=constants["eq_transforms"],
                profiles=constants["eq_profiles"],
                surf_batch_size=self._surf_batch_size,
            )

        # compute field data
        if self._field_fixed:
            field_data = constants["field_data"]
            # update theta_B and zeta_B with new iota from the equilibrium
            M, N = self.helicity
            iota = eq_data["iota"][eq_grid.unique_rho_idx]
            theta_B, zeta_B = _omnigenity_mapping(
                M,
                N,
                iota,
                field_data["alpha"],
                field_data["h"],
                field_grid,
            )
        else:
            field_data = compute_fun(
                "desc.magnetic_fields._core.OmnigenousField",
                self._field_data_keys,
                params=field_params,
                transforms=constants["field_transforms"],
                profiles={},
                helicity=self.helicity,
                iota=eq_data["iota"][eq_grid.unique_rho_idx],
                surf_batch_size=self._surf_batch_size,
            )
            theta_B = field_data["theta_B"]
            zeta_B = field_data["zeta_B"]

        # additional computations that cannot be part of the regular compute API

        def _compute_B_eta_alpha(theta_B, zeta_B, B_mn):
            nodes = jnp.vstack(
                (
                    jnp.zeros_like(theta_B),
                    theta_B,
                    zeta_B,
                )
            ).T
            B_eta_alpha = jnp.matmul(
                constants["eq_transforms"]["B"].basis.evaluate(nodes), B_mn
            )
            return B_eta_alpha

        theta_B = field_grid.meshgrid_reshape(theta_B, "rtz").reshape(
            (field_grid.num_rho, -1)
        )
        zeta_B = field_grid.meshgrid_reshape(zeta_B, "rtz").reshape(
            (field_grid.num_rho, -1)
        )
        B_mn = eq_data["|B|_mn_B"].reshape((eq_grid.num_rho, -1))
        B_eta_alpha = vmap_chunked(
            _compute_B_eta_alpha,
            in_axes=(0, 0, 0),
            chunk_size=self._surf_batch_size,
        )(theta_B, zeta_B, B_mn)
        B_eta_alpha = B_eta_alpha.reshape(
            (field_grid.num_rho, field_grid.num_theta, field_grid.num_zeta)
        )
        B_eta_alpha = jnp.moveaxis(B_eta_alpha, 0, 1).flatten(order="F")
        omnigenity_error = B_eta_alpha - field_data["|B|"]
        weights = (self.eta_weight + 1) / 2 + (self.eta_weight - 1) / 2 * jnp.cos(
            field_data["eta"]
        )
        return omnigenity_error * weights


class Isodynamicity(_Objective):
    """Isodynamicity metric for cross field transport.

    Note: This is NOT the same as Quasi-isodynamicity (QI), which is a more general
    condition. This specifically penalizes the local cross field transport, rather than
    just the average.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = "rtz"
    _units = "(dimensionless)"
    _print_value_fmt = "Isodynamicity error: "

    def __init__(
        self,
        eq,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="Isodynamicity",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        super().__init__(
            things=eq,
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
        eq = self.things[0]
        if self._grid is None:
            grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["isodynamicity"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute isodynamicity errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        f : ndarray
            Isodynamicity error at each node (~).

        """
        constants = self._get_deprecated_constants(constants)
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return data["isodynamicity"]


class OmnigenityHarmonics(_Objective):
    """Omnigenity harmonics error.

    Errors are relative to a target field contour that is perfectly omnigenous,
    and are computed on a collocation grid in (η,α) coordinates.

    This objective assumes that the collocation point (θ=0,ζ=0) lies on the contour of
    maximum field strength ||B||=B_max.

    Note: Only handles single magnetic surface.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to be optimized to satisfy the Objective.
    field : OmnigenousField
        Omnigenous magnetic field to be optimized to satisfy the Objective.
    field_type : {'desc', 'oops', 'lcform'}, optional
        Type of omnigenous field. 'desc' uses DESC representation `OmnigenousField`,
        'oops' uses OOPS representation `OmnigenousFieldOOPS`, and 'lcform' uses
        Landreman-type representation `OmnigenousFieldLCForm`.
        Default is 'desc'.
    eq_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for equilibrium data.
        Defaults to a linearly space grid on the rho=1 surface.
        Must be without stellarator symmetry.
    field_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for omnigenous field data.
        The input nodes are given in (ρ,θ,ζ) coordinates. For ``field_type="desc"``,
        θ is mapped to η and ζ to α. For ``"oops"`` and ``"lcform"``, θ is α and
        ``NFP * ζ`` is η, with an additional ``-π`` shift for ``"oops"``.
        Defaults to a linearly spaced grid on the rho=1 surface without stellarator
        symmetry. The sampled values are fitted by index on a separate uniform
        periodic grid in (α,η), preserving the input samples and their ordering.
        A cropped input grid therefore defines a spectrum of those samples, rather
        than a Fourier spectrum over the full physical angular domain.
    M_booz : int, optional
        Poloidal resolution of Boozer transformation. Default = 2 * eq.M.
    N_booz : int, optional
        Toroidal resolution of Boozer transformation. Default = 2 * eq.N.
    M_harmonics : int, optional
        Maximum alpha harmonic. Defaults to the input grid's alpha resolution.
    N_harmonics : int, optional
        Maximum eta harmonic. Defaults to the input grid's eta resolution.
    eq_fixed: bool, optional
        Whether the Equilibrium `eq` is fixed or not.
        If True, the equilibrium is fixed and its values are precomputed, which saves on
        computation time during optimization and only ``field`` is allowed to change.
        If False, the equilibrium is allowed to change during the optimization and its
        associated data are re-computed at every iteration (Default).
    field_fixed: bool, optional
        Whether the OmnigenousField `field` is fixed or not.
        If True, the field is fixed and its values are precomputed, which saves on
        computation time during optimization and only ``eq`` is allowed to change.
        If False, the field is allowed to change during the optimization and its
        associated data are re-computed at every iteration (Default).

    Notes
    -----
    With ``normalize=True``, the reference field is fixed when the objective is built.
    The DESC representation uses the mean on-axis ``field.B_lm``; OOPS and LCForm use
    the equilibrium reference field from ``compute_scaling_factors(eq)``. Updating the
    optimization parameters does not update this reference. Use ``normalize=False``
    to retain the dimensional harmonic residuals, including earlier OOPS objectives
    that did not apply a reference-field normalization.

    For toroidally closed LCForm contours (``helicity[1] == 0``), the mapping uses
    a full-torus alpha chart. With ``NFP > 1``, periodicity over one field period
    is not guaranteed; building the objective emits a warning in this case. The
    second argument of the S callback must have period ``2*pi/NFP``, satisfying
    ``S(x, y + 2*pi/NFP) = S(x, y)``; for example, use ``sin(k*NFP*y)`` for integer k.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _static_attrs = _Objective._static_attrs + [
        "_eq_data_keys",
        "_eq_fixed",
        "_field_data_keys",
        "_field_fixed",
        "helicity",
        "M_booz",
        "N_booz",
        "M_harmonics",
        "N_harmonics",
        "_field_type",
        "_is_imag",
        "S_function",
        "D_function",
    ]
    _coordinates = "rtz"
    _units = "~"
    _print_value_fmt = "Omnigenity error: "

    def __init__(
        self,
        eq,
        field,
        field_type="desc",
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        eq_grid=None,
        field_grid=None,
        M_booz=None,
        N_booz=None,
        M_harmonics=None,
        N_harmonics=None,
        eq_fixed=False,
        field_fixed=False,
        name="omnigenity_harmonics",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._eq = eq
        self._field = field
        self._field_type = field_type
        if self._field_type not in ["desc", "oops", "lcform"]:
            raise ValueError(
                "field_type must be 'desc', 'oops' or 'lcform', "
                f"got {self._field_type} instead."
            )

        self._eq_grid = eq_grid
        self._field_grid = field_grid
        # HDF5 may restore helicity elements as zero-dimensional numpy arrays.
        # Keep topology as Python integers for the static mapping branches under JIT.
        self.helicity = tuple(map(int, field.helicity))
        self.M_booz = M_booz
        self.N_booz = N_booz
        self.M_harmonics = M_harmonics
        self.N_harmonics = N_harmonics
        self.S_function = None
        self.D_function = None
        if self._field_type == "lcform":
            # Ensure the provided field supplies the required callable attributes
            missing = []
            if not hasattr(self._field, "_S_func"):
                missing.append("_S_func")
            if not hasattr(self._field, "_D_func"):
                missing.append("_D_func")
            if missing:
                raise ValueError(
                    "field_type 'lcform' requires field to define attributes: "
                    + ", ".join(missing)
                    + "; bind field.S_func / field.D_func before constructing "
                    "the objective."
                )
            self.S_function = getattr(self._field, "_S_func")
            self.D_function = getattr(self._field, "_D_func")
            if not callable(self.S_function) or not callable(self.D_function):
                raise ValueError(
                    "field_type 'lcform' requires _S_func and _D_func to be callable; "
                    "bind field.S_func / field.D_func before constructing "
                    "the objective."
                )

        self._eq_fixed = eq_fixed
        self._field_fixed = field_fixed
        if not eq_fixed and not field_fixed:
            things = [eq, field]
        elif eq_fixed and not field_fixed:
            things = [field]
        elif field_fixed and not eq_fixed:
            things = [eq]
        else:
            raise ValueError("Cannot fix both the eq and field.")
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

    def build(self, use_jit=True, verbose=1):  # noqa: C901
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self._eq_fixed:
            eq = self._eq
            field = self.things[0]
        elif self._field_fixed:
            eq = self.things[0]
            field = self._field
        else:
            eq = self.things[0]
            field = self.things[1]

        M_booz = self.M_booz or 2 * eq.M
        N_booz = self.N_booz or 2 * eq.N

        # default grids
        if self._eq_grid is None and self._field_grid is not None:
            rho = self._field_grid.nodes[self._field_grid.unique_rho_idx, 0]
        elif self._eq_grid is not None and self._field_grid is None:
            rho = self._eq_grid.nodes[self._eq_grid.unique_rho_idx, 0]
        elif self._eq_grid is None and self._field_grid is None:
            rho = 1.0
        if self._eq_grid is None:
            eq_grid = LinearGrid(
                rho=rho, M=2 * M_booz, N=2 * N_booz, NFP=eq.NFP, sym=False
            )
        else:
            eq_grid = self._eq_grid
        if self._field_grid is None:
            field_grid = LinearGrid(
                rho=rho, M=2 * eq.M, N=2 * eq.N, NFP=field.NFP, sym=False
            )
        else:
            field_grid = self._field_grid

        if self._field_type == "desc":
            # DESC's input grid is (eta, alpha), while the harmonic basis is
            # (alpha, eta). Keep its dimensions consistent with the transpose below.
            num_alpha, num_eta = field_grid.num_zeta, field_grid.num_theta
            M_alpha, N_eta = field_grid.N, field_grid.M
        else:
            num_alpha, num_eta = field_grid.num_theta, field_grid.num_zeta
            M_alpha, N_eta = field_grid.M, field_grid.N
        M_harmonics = M_alpha if self.M_harmonics is None else self.M_harmonics
        N_harmonics = N_eta if self.N_harmonics is None else self.N_harmonics

        self._eq_data_keys = ["|B|_mn_B"]

        if self._field_type == "desc":
            self._is_imag = True
            self._dim_f = 2 * M_harmonics * (2 * N_harmonics + 1)
            self._field_data_keys = ["|B|", "theta_B", "zeta_B"]
            errorif(
                jnp.any(field.B_lm[: field.M_B] < 0),
                msg="|B| on axis must be positive! Check B_lm input.",
            )
            if self._normalize:
                # average |B| on axis
                self._normalization = jnp.mean(field.B_lm[: field.M_B])
        elif self._field_type == "oops":
            self._is_imag = False
            self._dim_f = 1 * M_harmonics * (2 * N_harmonics + 1)
            self._field_data_keys = ["theta_B_OOPS", "zeta_B_OOPS", "S_list", "D_list"]
            if self._normalize:
                self._normalization = compute_scaling_factors(eq)["B"]
        elif self._field_type == "lcform":
            warnif(
                self.helicity[1] == 0 and field.NFP > 1,
                UserWarning,
                "LCForm toroidally closed contours with NFP > 1 use a full-torus "
                "chart; periodicity over one field period requires the S callback's "
                "second argument to have period 2*pi/NFP: "
                "S(x, y + 2*pi/NFP) = S(x, y), e.g. sin(k*NFP*y) for integer k.",
            )
            if eq.sym:
                self._is_imag = False
                self._dim_f = 1 * M_harmonics * (2 * N_harmonics + 1)
            else:
                self._is_imag = True
                self._dim_f = 2 * M_harmonics * (2 * N_harmonics + 1)
            if self._normalize:
                scales = compute_scaling_factors(eq)
                self._normalization = scales["B"]
            self._field_data_keys = [
                "theta_B_LCForm",
                "zeta_B_LCForm",
                "S_list",
                "D_list",
            ]

        errorif(
            eq_grid.NFP != field_grid.NFP,
            msg="eq_grid and field_grid must have the same number of field periods",
        )
        errorif(eq_grid.sym, msg="eq_grid must not be symmetric")
        errorif(field_grid.sym, msg="field_grid must not be symmetric")
        errorif(
            eq_grid.num_rho != 1 or field_grid.num_rho != 1,
            msg="eq_grid and field_grid must each contain exactly one surface",
        )
        field_rho = field_grid.nodes[field_grid.unique_rho_idx, 0]
        eq_rho = eq_grid.nodes[eq_grid.unique_rho_idx, 0]
        errorif(
            eq_rho[0] != field_rho[0],
            msg="eq_grid and field_grid must be the same surface, "
            f"got rho={eq_rho[0]} and rho={field_rho[0]}",
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._eq_data_keys, obj=eq, grid=eq_grid)
        eq_transforms = get_transforms(
            self._eq_data_keys,
            obj=eq,
            grid=eq_grid,
            M_booz=M_booz,
            N_booz=N_booz,
        )
        field_transforms = get_transforms(
            self._field_data_keys,
            obj=field,
            grid=field_grid,
        )

        from desc.basis import DoubleFourierSeries
        from desc.transform import Transform

        grid_B = LinearGrid(theta=num_alpha, zeta=num_eta, NFP=1, sym=False)
        field_transforms["|B|_eta_alpha"] = Transform(
            grid_B,
            DoubleFourierSeries(
                M=M_harmonics,
                N=N_harmonics,
                NFP=1,
                sym="cos" if not self._is_imag else False,
            ),
            derivs=0,
            build=False,
            build_pinv=True,
            method="auto",
        )

        # I don't know why this is needed, but it is
        w = jnp.ones(self._dim_f)
        w *= 1

        self._constants = {
            "eq_profiles": profiles,
            "eq_transforms": eq_transforms,
            "field_transforms": field_transforms,
            "quad_weights": w,
        }

        if self._eq_fixed:
            # precompute the eq data since it is fixed during the optimization
            eq_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._eq_data_keys,
                params=self._eq.params_dict,
                transforms=self._constants["eq_transforms"],
                profiles=self._constants["eq_profiles"],
            )
            self._constants["eq_data"] = eq_data
        if self._field_fixed:
            field_data = compute_fun(
                self._field,
                self._field_data_keys,
                params=self._field.params_dict,
                transforms=self._constants["field_transforms"],
                profiles={},
                helicity=self.helicity,
                S_func=self.S_function,
                D_func=self.D_function,
            )
            self._constants["field_data"] = field_data

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1=None, params_2=None, constants=None):
        """Compute omnigenity errors.

        Parameters
        ----------
        params_1 : dict
            If eq_fixed=True, dictionary of field degrees of freedom,
            eg OmnigenousField.params_dict. Otherwise, dictionary of equilibrium degrees
            of freedom, eg Equilibrium.params_dict.
        params_2 : dict
            If eq_fixed=False and field_fixed=False, dictionary of field degrees of
            freedom, eg OmnigenousField.params_dict. Otherwise None.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants. (Deprecated)

        Returns
        -------
        omnigenity_error : ndarray
            Omnigenity error at each node (T).

        """
        constants = self._get_deprecated_constants(constants)

        # sort parameters
        if self._eq_fixed:
            field_params = params_1
        elif self._field_fixed:
            eq_params = params_1
        else:
            eq_params = params_1
            field_params = params_2

        eq_grid = constants["eq_transforms"]["grid"]
        field_grid = constants["field_transforms"]["grid"]

        # compute eq data
        if self._eq_fixed:
            eq_data = constants["eq_data"]
        else:
            eq_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._eq_data_keys,
                params=eq_params,
                transforms=constants["eq_transforms"],
                profiles=constants["eq_profiles"],
            )

        # compute field data
        if self._field_type == "desc":
            if self._field_fixed:
                field_data = constants["field_data"]
                # update theta_B and zeta_B with new iota from the equilibrium
                M, N = self.helicity
                iota = eq_data["iota"][eq_grid.unique_rho_idx]
                theta_B, zeta_B = _omnigenity_mapping(
                    M,
                    N,
                    iota,
                    field_data["alpha"],
                    field_data["h"],
                    field_grid,
                )
            else:
                field_data = compute_fun(
                    "desc.magnetic_fields._core.OmnigenousField",
                    self._field_data_keys,
                    params=field_params,
                    transforms=constants["field_transforms"],
                    profiles={},
                    helicity=self.helicity,
                    iota=eq_data["iota"][eq_grid.unique_rho_idx],
                )
                theta_B = field_data["theta_B"]
                zeta_B = field_data["zeta_B"]
        elif self._field_type == "oops":
            if self._field_fixed:
                field_data = constants["field_data"]
                # update theta_B and zeta_B with new iota from the equilibrium
                M, N = self.helicity
                iota = eq_data["iota"][eq_grid.unique_rho_idx]
                theta_B, zeta_B = _omnigenity_mapping_OOPS(
                    M,
                    N,
                    iota,
                    field_data["S_list"],
                    field_data["D_list"],
                    field_grid,
                )
            else:
                field_data = compute_fun(
                    "desc.magnetic_fields._core.OmnigenousFieldOOPS",
                    self._field_data_keys,
                    params=field_params,
                    transforms=constants["field_transforms"],
                    profiles={},
                    helicity=self.helicity,
                    iota=eq_data["iota"][eq_grid.unique_rho_idx],  # For test
                )
                theta_B = field_data["theta_B_OOPS"]
                zeta_B = field_data["zeta_B_OOPS"]
        if self._field_type == "lcform":
            if self._field_fixed:
                field_data = constants["field_data"]
                # update theta_B and zeta_B with new iota from the equilibrium
                M, N = self.helicity
                iota = eq_data["iota"][eq_grid.unique_rho_idx]
                theta_B, zeta_B = _omnigenity_mapping_LandremanForm(
                    M,
                    N,
                    iota,
                    field_data["S_list"],
                    field_data["D_list"],
                    self.S_function,
                    self.D_function,
                    field_grid,
                )
            else:
                field_data = compute_fun(
                    "desc.magnetic_fields._core.OmnigenousFieldLCForm",
                    self._field_data_keys,
                    params=field_params,
                    transforms=constants["field_transforms"],
                    profiles={},
                    helicity=self.helicity,
                    iota=eq_data["iota"][eq_grid.unique_rho_idx],  # For test
                    S_func=self.S_function,
                    D_func=self.D_function,
                )
                theta_B = field_data["theta_B_LCForm"]
                zeta_B = field_data["zeta_B_LCForm"]

        # additional computations that cannot be part of the regular compute API
        def _compute_B_eta_alpha(theta_B, zeta_B, B_mn):
            nodes = jnp.vstack(
                (
                    jnp.zeros_like(theta_B),
                    theta_B,
                    zeta_B,
                )
            ).T
            B_eta_alpha = jnp.matmul(
                constants["eq_transforms"]["B"].basis.evaluate(nodes), B_mn
            )
            return B_eta_alpha

        theta_B = field_grid.meshgrid_reshape(theta_B, "rtz").reshape(
            (field_grid.num_rho, -1)
        )
        zeta_B = field_grid.meshgrid_reshape(zeta_B, "rtz").reshape(
            (field_grid.num_rho, -1)
        )
        B_mn = eq_data["|B|_mn_B"].reshape((eq_grid.num_rho, -1))
        B_eta_alpha = vmap_chunked(
            _compute_B_eta_alpha, in_axes=(0, 0, 0), chunk_size=None
        )(theta_B, zeta_B, B_mn)

        B_eta_alpha = B_eta_alpha.reshape(
            (field_grid.num_rho, field_grid.num_theta, field_grid.num_zeta)
        )

        # TODO: Need to figure out how to handle the matrix
        if self._field_type == "desc":
            B_eta_alpha = B_eta_alpha[-1].T.flatten(order="F")
        elif self._field_type == "oops" or self._field_type == "lcform":
            B_eta_alpha = B_eta_alpha[-1].flatten(order="F")

        B_ea_mn = constants["field_transforms"]["|B|_eta_alpha"].fit(B_eta_alpha)

        modes_index_xm_non_zero = (
            constants["field_transforms"]["|B|_eta_alpha"].basis.modes[:, 1] != 0
        )
        B_ea_mn_non_zero = B_ea_mn[modes_index_xm_non_zero]
        return B_ea_mn_non_zero
