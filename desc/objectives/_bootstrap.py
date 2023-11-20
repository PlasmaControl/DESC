"""Objectives related to the bootstrap current profile."""

import warnings

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.grid import LinearGrid
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class BootstrapRedlConsistency(_Objective):
    """
    Promote consistency of the bootstrap current for axisymmetry or quasi-symmetry.

    This objective function penalizes the difference between the MHD
    and neoclassical profiles of parallel current, using the Redl
    formula for the bootstrap current. The scalar objective is defined as

    f = ½ ∫dρ [(⟨J⋅B⟩_MHD - ⟨J⋅B⟩_Redl) / (J_ref B_ref)]²

    where J_ref and B_ref are the reference magnitudes of current
    density and magnetic field strength.  This objective is treated as
    a sum of Nρ least-squares terms, where Nρ is the number of rho
    grid points.

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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
    helicity : tuple, optional
        Type of quasi-symmetry (M, N). Default = quasi-axisymmetry (1, 0).
        First entry must be M=1. Second entry is the toroidal mode number N,
        used for evaluating the Redl bootstrap current formula. Set to 0 for axisymmetry
        or quasi-axisymmetry; set to +/-NFP for quasi-helical symmetry.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = "r"
    _units = "(T A m^-2)"
    _print_value_fmt = "Bootstrap current self-consistency error: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        helicity=(1, 0),
        name="Bootstrap current self-consistency (Redl)",
    ):
        if target is None and bounds is None:
            target = 0
        assert (len(helicity) == 2) and (int(helicity[1]) == helicity[1])
        assert helicity[0] == 1, "Redl bootstrap current model assumes helicity[0] == 1"
        self._grid = grid
        self.helicity = helicity
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq=None, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = eq or self._eq
        if self._grid is None:
            grid = LinearGrid(
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=np.linspace(1 / eq.L, 1, eq.L) - 1 / (2 * eq.L),
            )
        else:
            grid = self._grid

        assert (
            self.helicity[1] == 0 or abs(self.helicity[1]) == eq.NFP
        ), "Helicity toroidal mode number should be 0 (QA) or +/- NFP (QH)"
        self._dim_f = grid.num_rho
        self._data_keys = ["<J*B>", "<J*B> Redl"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=grid.axis.size,
        )

        if eq.electron_temperature is None:
            raise RuntimeError(
                "Bootstrap current calculation requires an electron temperature "
                "profile."
            )
        if eq.electron_density is None:
            raise RuntimeError(
                "Bootstrap current calculation requires an electron density profile."
            )
        if eq.ion_temperature is None:
            raise RuntimeError(
                "Bootstrap current calculation requires an ion temperature profile."
            )

        # Try to catch cases in which density or temperatures are specified in the
        # wrong units. Densities should be ~ 10^20, temperatures are ~ 10^3.
        rho = eq.compute("rho", grid=grid)["rho"]
        if jnp.any(eq.electron_density(rho) > 1e22):
            warnings.warn(
                "Electron density is surprisingly high. It should have units of "
                "1/meters^3"
            )
        if jnp.any(eq.electron_temperature(rho) > 50e3):
            warnings.warn(
                "Electron temperature is surprisingly high. It should have units of eV"
            )
        if jnp.any(eq.ion_temperature(rho) > 50e3):
            warnings.warn(
                "Ion temperature is surprisingly high. It should have units of eV"
            )
        # Profiles may go to 0 at rho=1, so exclude the last few grid points from lower
        # bounds:
        rho = rho[rho < 0.85]
        if jnp.any(eq.electron_density(rho) < 1e17):
            warnings.warn(
                "Electron density is surprisingly low. It should have units 1/meters^3"
            )
        if jnp.any(eq.electron_temperature(rho) < 30):
            warnings.warn(
                "Electron temperature is surprisingly low. It should have units of eV"
            )
        if jnp.any(eq.ion_temperature(rho) < 30):
            warnings.warn(
                "Ion temperature is surprisingly low. It should have units of eV"
            )

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "helicity": self._helicity,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["J"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):
        """Compute the bootstrap current self-consistency objective.

        Returns
        -------
        obj : ndarray
            Bootstrap current self-consistency residual on each rho grid point.

        """
        params, constants = self._parse_args(*args, **kwargs)
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
        return constants["transforms"]["grid"].compress(
            data["<J*B>"] - data["<J*B> Redl"]
        )

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
        assert helicity[0] == 1, "Redl bootstrap current model assumes helicity[0] == 1"
        if hasattr(self, "_helicity") and self._helicity != helicity:
            self._built = False
        self._helicity = helicity
