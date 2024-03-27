"""Objectives for targeting quasisymmetry."""

import warnings

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer, errorif, warnif
from desc.vmec_utils import ptolemy_linear_transform

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class QuasisymmetryBoozer(_Objective):
    """Quasi-symmetry Boozer harmonics error.

    Parameters
    ----------
    eq : Equilibrium
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
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Must be a LinearGrid with a single flux surface and sym=False.
    helicity : tuple, optional
        Type of quasi-symmetry (M, N). Default = quasi-axisymmetry (1, 0).
    M_booz : int, optional
        Poloidal resolution of Boozer transformation. Default = 2 * eq.M.
    N_booz : int, optional
        Toroidal resolution of Boozer transformation. Default = 2 * eq.N.
    name : str, optional
        Name of the objective function.

    """

    _units = "(T)"
    _print_value_fmt = "Quasi-symmetry Boozer error: {:10.3e} "

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
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.helicity = helicity
        self.M_booz = M_booz
        self.N_booz = N_booz
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
        )

        self._print_value_fmt = (
            "Quasi-symmetry ({},{}) Boozer error: ".format(
                self.helicity[0], self.helicity[1]
            )
            + "{:10.3e} "
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
        errorif(
            grid.num_rho != 1,
            ValueError,
            "QuasisymmetryBoozer grid must be on a single surface. "
            "To target multiple surfaces, use multiple objectives.",
        )
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

        self._data_keys = ["|B|_mn"]

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
        matrix, modes, idx = ptolemy_linear_transform(
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

        self._dim_f = idx.size

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
            self.constants

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^3).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        B_mn = constants["matrix"] @ data["|B|_mn"]
        return B_mn[constants["idx"]]

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
            units = "(T)"
            self._print_value_fmt = (
                "Quasi-symmetry ({},{}) Boozer error: ".format(
                    self.helicity[0], self.helicity[1]
                )
                + "{:10.3e} "
                + units
            )


class QuasisymmetryTwoTerm(_Objective):
    """Quasi-symmetry two-term error.

    Parameters
    ----------
    eq : Equilibrium
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
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    helicity : tuple, optional
        Type of quasi-symmetry (M, N).
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(T^3)"
    _print_value_fmt = "Quasi-symmetry two-term error: {:10.3e} "

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
        )

        self._print_value_fmt = (
            "Quasi-symmetry ({},{}) two-term error: ".format(
                self.helicity[0], self.helicity[1]
            )
            + "{:10.3e} "
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
            self.constants

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^3).

        """
        if constants is None:
            constants = self.constants
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
            units = "(T^3)"
            self._print_value_fmt = (
                "Quasi-symmetry ({},{}) error: ".format(
                    self.helicity[0], self.helicity[1]
                )
                + "{:10.3e} "
                + units
            )


class QuasisymmetryTripleProduct(_Objective):
    """Quasi-symmetry triple product error.

    Parameters
    ----------
    eq : Equilibrium
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
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(T^4/m^2)"
    _print_value_fmt = "Quasi-symmetry error: {:10.3e} "

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
            self.constants

        Returns
        -------
        f : ndarray
            Quasi-symmetry flux function error at each node (T^4/m^2).

        """
        if constants is None:
            constants = self.constants
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
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    eq_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for equilibrium data.
        Defaults to a linearly space grid on the rho=1 surface.
        Must be a single flux surface without stellarator symmetry.
    field_grid : Grid, optional
        Collocation grid containing the nodes to evaluate at for omnigenous field data.
        The grid nodes are given in the usual (ρ,θ,ζ) coordinates (with θ ∈ [0, 2π),
        ζ ∈ [0, 2π/NFP)), but θ is mapped to η and ζ is mapped to α. Defaults to a
        linearly space grid on the rho=1 surface. Must be a single flux surface without
        stellarator symmetry.
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
        computation time during optimization and self.things = [field] only.
        If False, the equilibrium is allowed to change during the optimization and its
        associated data are re-computed at every iteration (Default).
    field_fixed: bool, optional
        Whether the OmnigenousField `field` is fixed or not.
        If True, the field is fixed and its values are precomputed, which saves on
        computation time during optimization and self.things = [eq] only.
        If False, the field is allowed to change during the optimization and its
        associated data are re-computed at every iteration (Default).
    name : str
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(T)"
    _print_value_fmt = "Omnigenity error: {:10.3e} "

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
        deriv_mode="fwd",  # FIXME: get it working with rev mode (see GH issue #943)
        eq_grid=None,
        field_grid=None,
        M_booz=None,
        N_booz=None,
        eta_weight=1,
        eq_fixed=False,
        field_fixed=False,
        name="omnigenity",
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
            rho = self._field_grid.nodes[0, 0]
        elif self._eq_grid is not None and self._field_grid is None:
            rho = self._eq_grid.nodes[0, 0]
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
        self._eq_data_keys = ["|B|_mn"]
        self._field_data_keys = ["|B|", "theta_B", "zeta_B"]

        errorif(
            eq_grid.NFP != field_grid.NFP,
            msg="eq_grid and field_grid must have the same number of field periods",
        )
        errorif(eq_grid.sym, msg="eq_grid must not be symmetric")
        errorif(field_grid.sym, msg="field_grid must not be symmetric")
        errorif(eq_grid.num_rho != 1, msg="eq_grid must be a single surface")
        errorif(field_grid.num_rho != 1, msg="field_grid must be a single surface")
        errorif(
            eq_grid.nodes[eq_grid.unique_rho_idx, 0]
            != field_grid.nodes[field_grid.unique_rho_idx, 0],
            msg="eq_grid and field_grid must be the same surface",
        )
        errorif(
            jnp.any(field.B_lm[: field.M_B] < 0),
            "|B| on axis must be positive! Check B_lm input.",
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
            "helicity": self.helicity,
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
            # precompute the field data since it is fixed during the optimization
            field_data = compute_fun(
                "desc.magnetic_fields._core.OmnigenousField",
                self._field_data_keys,
                params=self._field.params_dict,
                transforms=self._constants["field_transforms"],
                profiles={},
                helicity=self._constants["helicity"],
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
            self.constants

        Returns
        -------
        omnigenity_error : ndarray
            Omnigenity error at each node (T).

        """
        if constants is None:
            constants = self.constants

        # sort parameters
        if self._eq_fixed:
            field_params = params_1
        elif self._field_fixed:
            eq_params = params_1
        else:
            eq_params = params_1
            field_params = params_2

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
        if self._field_fixed:
            field_data = constants["field_data"]
            # update theta_B and zeta_B with new iota from the equilibrium
            M, N = constants["helicity"]
            iota = jnp.mean(eq_data["iota"])
            matrix = jnp.where(
                M == 0,
                jnp.array([N, iota / N, 0, 1 / N]),  # OP
                jnp.where(
                    N == 0,
                    jnp.array([0, -1, M, -1 / iota]),  # OT
                    jnp.array(
                        [N, M * iota / (N - M * iota), M, M / (N - M * iota)]  # OH
                    ),
                ),
            ).reshape((2, 2))
            booz = matrix @ jnp.vstack((field_data["alpha"], field_data["h"]))
            field_data["theta_B"] = booz[0, :]
            field_data["zeta_B"] = booz[1, :]
        else:
            field_data = compute_fun(
                "desc.magnetic_fields._core.OmnigenousField",
                self._field_data_keys,
                params=field_params,
                transforms=constants["field_transforms"],
                profiles={},
                helicity=constants["helicity"],
                iota=jnp.mean(eq_data["iota"]),
            )

        # additional computations that cannot be part of the regular compute API
        nodes = jnp.vstack(
            (
                jnp.zeros_like(field_data["theta_B"]),
                field_data["theta_B"],
                field_data["zeta_B"],
            )
        ).T
        B_eta_alpha = jnp.matmul(
            constants["eq_transforms"]["B"].basis.evaluate(nodes), eq_data["|B|_mn"]
        )
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
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
        Has no effect for this objective.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(dimensionless)"
    _print_value_fmt = "Isodynamicity error: {:10.3e} "

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
            self.constants

        Returns
        -------
        f : ndarray
            Isodynamicity error at each node (~).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        return data["isodynamicity"]
