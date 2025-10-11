"""Objectives related to the bootstrap current profile."""

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, errorif, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


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
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Requires poloidal and
        toroidal resolution as the objective must compute flux surface averages.
        Defaults to
        ``grid = LinearGrid(M=eq.M_grid,N=eq.N_grid,NFP=eq.NFP,``
        ``sym=eq.sym,rho=np.linspace(1 / eq.L, 1, eq.L) - 1 / (2 * eq.L),)``
    helicity : tuple, optional
        Type of quasi-symmetry (M, N). Default = quasi-axisymmetry (1, 0).
        First entry must be M=1. Second entry is the toroidal mode number N,
        used for evaluating the Redl bootstrap current formula. Set to 0 for
        axisymmetry or quasi-axisymmetry; set to +/-NFP for quasi-helical symmetry.
    degree : int, optional
        The `degree` kwarg to pass to the `<J*B>_Redl` compute call, which
        controls the degree of polynomial fit to the Redl current derivative
        before it is integrated.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = "r"
    _units = "(T A m^-2)"
    _print_value_fmt = "Bootstrap current self-consistency error: "

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
        degree=None,
        name="Bootstrap current self-consistency (Redl)",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        assert (len(helicity) == 2) and (int(helicity[1]) == helicity[1])
        assert helicity[0] == 1, "Redl bootstrap current model assumes helicity[0] == 1"
        self._grid = grid
        self.helicity = helicity
        self._degree = degree
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
            grid = LinearGrid(
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=np.linspace(1 / eq.L, 1, eq.L) - 1 / (2 * eq.L),
            )
        else:
            grid = self._grid

        warnif(
            (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
            RuntimeWarning,
            "BootstrapRedlConsistency objective grid requires poloidal "
            "resolution for surface averages",
        )
        warnif(
            grid.num_zeta < 2 * eq.N,
            RuntimeWarning,
            "BootstrapRedlConsistency objective grid requires toroidal "
            "resolution for surface averages",
        )

        errorif(
            not (self.helicity[1] == 0 or abs(self.helicity[1]) == eq.NFP),
            ValueError,
            "Helicity toroidal mode number should be 0 (QA) or +/- NFP (QH)",
        )
        rho = grid.nodes[grid.unique_rho_idx, 0]
        errorif(
            grid.axis.size,
            ValueError,
            "Redl formula is undefined at rho=0, but grid has grid points at rho=0",
        )

        self._dim_f = grid.num_rho
        self._data_keys = ["<J*B>", "<J*B> Redl"]

        errorif(
            eq.electron_temperature is None,
            RuntimeError,
            "Bootstrap current calculation requires an electron temperature "
            "profile.",
        )
        errorif(
            eq.electron_density is None,
            RuntimeError,
            "Bootstrap current calculation requires an electron density profile.",
        )
        errorif(
            eq.ion_temperature is None,
            RuntimeError,
            "Bootstrap current calculation requires an ion temperature profile.",
        )

        rho = grid.compress(grid.nodes[:, 0], "rho")

        # check if profiles may go to zero
        # if they are exactly zero this would cause NaNs since the profiles
        # vanish.
        errorif(
            np.any(np.isclose(eq.electron_density(rho), 0.0, atol=1e-8)),
            ValueError,
            "Redl formula is undefined where kinetic profiles vanish, "
            "but given electron density vanishes at at least one provided"
            "rho grid point.",
        )
        errorif(
            np.any(np.isclose(eq.electron_temperature(rho), 0.0, atol=1e-8)),
            ValueError,
            "Redl formula is undefined where kinetic profiles vanish, "
            "but given electron temperature vanishes at at least one provided"
            "rho grid point.",
        )
        errorif(
            np.any(np.isclose(eq.ion_temperature(rho), 0.0, atol=1e-8)),
            ValueError,
            "Redl formula is undefined where kinetic profiles vanish, "
            "but given ion temperature vanishes at at least one provided"
            "rho grid point.",
        )
        # Try to catch cases in which density or temperatures are specified in the
        # wrong units. Densities should be ~ 10^20, temperatures are ~ 10^3.
        warnif(
            jnp.any(eq.electron_density(rho) > 1e22),
            UserWarning,
            "Electron density is surprisingly high. It should have units of "
            "1/meters^3",
        )
        warnif(
            jnp.any(eq.electron_temperature(rho) > 50e3),
            UserWarning,
            "Electron temperature is surprisingly high. It should have units of eV",
        )
        warnif(
            jnp.any(eq.ion_temperature(rho) > 50e3),
            UserWarning,
            "Ion temperature is surprisingly high. It should have units of eV",
        )

        # Profiles may go to 0 at rho=1 (and we've already checked if our
        # grid has points there), so exclude the last few grid points from lower
        # bounds:
        rho = rho[rho < 0.85]
        warnif(
            jnp.any(eq.electron_density(rho) < 1e17),
            UserWarning,
            "Electron density is surprisingly low. It should have units 1/meters^3",
        )
        warnif(
            jnp.any(eq.electron_temperature(rho) < 30),
            UserWarning,
            "Electron temperature is surprisingly low. It should have units of eV",
        )
        warnif(
            jnp.any(eq.ion_temperature(rho) < 30),
            UserWarning,
            "Ion temperature is surprisingly low. It should have units of eV",
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
            "degree": self._degree,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["J"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the bootstrap current self-consistency objective.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        obj : ndarray
            Bootstrap current self-consistency residual on each rho grid point.

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
            degree=constants["degree"],
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
