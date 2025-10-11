"""Objectives for targeting energy confinement."""

from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.grid import QuadratureGrid
from desc.utils import Timer, errorif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


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
    fuel : str, optional
        Fusion fuel, assuming a 50/50 mix. One of {'DT'}. Default = 'DT'.
    grid : Grid, optional
        Collocation grid used to compute the intermediate quantities.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=1e9``.",
        bounds_default="``target=1e9``.",
        loss_detail=" Note: Has no effect for this objective.",
    )

    _static_attrs = _Objective._static_attrs + ["_fuel"]

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
        jac_chunk_size=None,
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
    H_ISS04 : float, optional
        ISS04 confinement enhancement factor. Default = 1.
    gamma : float, optional
        Adiabatic (compressional) index. Default = 0.
    grid : Grid, optional
        Collocation grid used to compute the intermediate quantities.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        loss_detail=" Note: Has no effect for this objective.",
    )

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
        jac_chunk_size=None,
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
