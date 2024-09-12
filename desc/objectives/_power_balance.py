"""Objectives for targeting energy confinement."""

from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import QuadratureGrid
from desc.utils import Timer, errorif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class FusionPower(_Objective):
    """Fusion power.

    P = e E ∫ n^2 ⟨σν⟩ dV (W)

    References
    ----------
    https://doi.org/10.1088/0029-5515/32/4/I07.
    Improved Formulas for Fusion Cross-Sections and Thermal Reactivities.
    H.-S. Bosch and G.M. Hale. Nucl. Fusion April 1992; 32 (4): 611-631.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=1e9``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=1e9``.
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
        normalization. Note: Has no effect for this objective.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    fuel : str, optional
        Fusion fuel, assuming a 50/50 mix. One of {'DT'}. Default = 'DT'.
    grid : Grid, optional
        Collocation grid used to compute the intermediate quantities.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)``.
    name : str, optional
        Name of the objective function.
    jac_chunk_size : int or "auto", optional
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
        Defaults to ``chunk_size="auto"`` which will use a conservative
        size of 1000.

    """

    _scalar = True
    _units = "(W)"
    _print_value_fmt = "Fusion power: "

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
        fuel="DT",
        grid=None,
        name="fusion power",
        jac_chunk_size="auto",
    ):
        errorif(
            fuel not in ["DT"], ValueError, f"fuel must be one of ['DT'], got {fuel}."
        )
        if target is None and bounds is None:
            target = 1e9
        self._fuel = fuel
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
        errorif(
            eq.electron_density is None,
            ValueError,
            "Equilibrium must have an electron density profile.",
        )
        errorif(
            eq.ion_temperature is None,
            ValueError,
            "Equilibrium must have an ion temperature profile.",
        )
        if self._grid is None:
            self._grid = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
        self._dim_f = 1
        self._data_keys = ["P_fusion"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants = {
            "profiles": get_profiles(self._data_keys, obj=eq, grid=self._grid),
            "transforms": get_transforms(self._data_keys, obj=eq, grid=self._grid),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["W_p"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute fusion power.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom, eg
            Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        P : float
            Fusion power (W).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            fuel=self.fuel,
        )
        return data["P_fusion"]

    @property
    def fuel(self):
        """str: Fusion fuel, assuming a 50/50 mix. One of {'DT'}. Default = 'DT'."""
        return self._fuel

    @fuel.setter
    def fuel(self, new):
        errorif(
            new not in ["DT"], ValueError, f"fuel must be one of ['DT'], got {new}."
        )
        self._fuel = new


class HeatingPowerISS04(_Objective):
    """Heating power required by the ISS04 energy confinement time scaling.

    τ_E = Wₚ / P = 0.134 H_ISS04 a²ᐧ²⁸ R⁰ᐧ⁶⁴ P⁻⁰ᐧ⁶¹ nₑ⁰ᐧ⁵⁴ B⁰ᐧ⁸⁴ 𝜄⁰ᐧ⁴¹ (s)

    References
    ----------
    https://doi.org/10.1088/0029-5515/45/12/024.
    Characterization of energy confinement in net-current free plasmas using the
    extended International Stellarator Database.
    H. Yamada et al. Nucl. Fusion 29 November 2005; 45 (12): 1684–1693.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``target=0``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``target=0``.
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
        normalization. Note: Has no effect for this objective.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    H_ISS04 : float, optional
        ISS04 confinement enhancement factor. Default = 1.
    gamma : float, optional
        Adiabatic (compressional) index. Default = 0.
    grid : Grid, optional
        Collocation grid used to compute the intermediate quantities.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)``.
    name : str, optional
        Name of the objective function.
    jac_chunk_size : int or "auto", optional
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
        Defaults to ``chunk_size="auto"`` which will use a conservative
        size of 1000.

    """

    _scalar = True
    _units = "(W)"
    _print_value_fmt = "Heating power: "

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
        H_ISS04=1,
        gamma=0,
        grid=None,
        name="heating power",
        jac_chunk_size="auto",
    ):
        if target is None and bounds is None:
            target = 0
        self._H_ISS04 = H_ISS04
        self._gamma = gamma
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
        errorif(
            eq.electron_density is None,
            ValueError,
            "Equilibrium must have an electron density profile.",
        )
        if self._grid is None:
            self._grid = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
        self._dim_f = 1
        self._data_keys = ["P_ISS04"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._constants = {
            "profiles": get_profiles(self._data_keys, obj=eq, grid=self._grid),
            "transforms": get_transforms(self._data_keys, obj=eq, grid=self._grid),
            "H_ISS04": self._H_ISS04,
            "gamma": self._gamma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["W_p"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute heating power.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium or surface degrees of freedom, eg
            Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        P : float
            Heating power required by the ISS04 energy confinement time scaling (W).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            gamma=constants["gamma"],
            H_ISS04=constants["H_ISS04"],
        )
        return data["P_ISS04"]

    @property
    def H_ISS04(self):
        """float: Confinement enhancement factor."""
        return self._H_ISS04

    @H_ISS04.setter
    def H_ISS04(self, new):
        self._H_ISS04 = new

    @property
    def gamma(self):
        """float: Adiabatic (compressional) index."""
        return self._gamma

    @gamma.setter
    def gamma(self, new):
        self._gamma = new
