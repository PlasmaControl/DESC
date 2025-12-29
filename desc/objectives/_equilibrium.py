"""Objectives for solving equilibrium problems."""

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import ConcentricGrid, QuadratureGrid
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


class ForceBalance(_Objective):
    r"""Radial and helical MHD force balance.

    Given force densities:

    Fᵨ = √g (J^θ B^ζ - J^ζ B^θ) - ∇ p

    Fₕₑₗᵢ √g J^ρ

    and helical basis vector:

    𝐞ʰᵉˡⁱ = B^ζ ∇ θ - B^θ ∇ ζ

    Minimizes the magnitude of the forces:

    fᵨ = Fᵨ ||∇ ρ|| dV  (N)

    fₕₑₗᵢ = Fₕₑₗᵢ ||𝐞ʰᵉˡⁱ|| dV  (N)

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _equilibrium = True
    _coordinates = "rtz"
    _units = "(N)"
    _print_value_fmt = "Force error: "

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
        name="force",
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
            grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        self._dim_f = 2 * grid.num_nodes
        self._data_keys = [
            "F_rho",
            "|grad(rho)|",
            "sqrt(g)",
            "F_helical",
            "|e^helical*sqrt(g)|",
        ]

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
            self._normalization = scales["f"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute MHD force balance errors.

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
            MHD force balance error at each node (N).

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
        fr = data["F_rho"] * data["|grad(rho)|"] * data["sqrt(g)"]
        fb = data["F_helical"] * data["|e^helical*sqrt(g)|"]

        return jnp.concatenate([fr, fb])


class ForceBalanceDeflated(_Objective):
    r"""Deflated radial and helical MHD force balance.

    Given force densities:

    Fᵨ = √g (J^θ B^ζ - J^ζ B^θ) - ∇ p

    Fₕₑₗᵢ √g J^ρ

    and helical basis vector:

    𝐞ʰᵉˡⁱ = B^ζ ∇ θ - B^θ ∇ ζ

    and deflation operator:

    M(𝐱;𝐱₁*) = ||𝐱−𝐱₁*||⁻ᵖ₂ + σ

    (if `deflation_type="power"`)

    or

    M(𝐱;𝐱₁*) = exp(-||𝐱−𝐱₁*||₂) + σ

    (if `deflation_type="exp"`)

    Minimizes the magnitude of the deflated forces:

    fᵨ = Fᵨ ||∇ ρ|| dV  (N)

    fₕₑₗᵢ = Fₕₑₗᵢ ||𝐞ʰᵉˡⁱ|| dV  (N)

    M(𝐱;𝐱₁*)[fᵨ, fₕₑₗᵢ]

    If multiple states are used for deflation, the

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    eqs: list of Equilibrium
        list of Equilibrium objects to use in deflation operator.
    sigma: float, optional
        shift parameter in deflation operator.
    power: float, optional
        power parameter in deflation operator, ignored if `deflation_type="exp"`.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid)``
    params_to_deflate_with: list of str
        Which params to use in the deflation operator to define the
        state to deflate, defaults to ["Rb_mn","Zb_mn"]
    deflation_type: {"power","exp"}
        What type of deflation to use. If `"power"`, uses the form
        pioneered by Farrell where M(𝐱;𝐱₁*) = ||𝐱−𝐱₁*||⁻ᵖ₂ + σ
        while `"exp"` uses the form from Riley 2024, where
        M(𝐱;𝐱₁*) = exp(-||𝐱−𝐱₁*||₂) + σ. Defaults to "power".
    multiple_deflation_type: {"prod","sum"}
        When deflating multiple states, how to reduce the individual deflation
        terms Mᵢ(𝐱;𝐱ᵢ*). `"prod"` will multiply each individual deflation term
        together, while `"sum"` will add each individual term.
    single_shift: bool,
        Whether to use a single shift or include the shift in each individual
        deflation term. i.e. whether to use M = σ + prod(||𝐱−𝐱_i*||⁻ᵖ₂) (if True)
        or to use M = prod( σ + ||𝐱−𝐱_i*||⁻ᵖ₂). Defaults to False.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    _static_attrs = _Objective._static_attrs + [
        "_params_to_deflate_with",
        "_deflation_type",
        "_multiple_deflation_type",
        "_single_shift",
    ]

    _equilibrium = True
    _coordinates = "rtz"
    _units = "(N)"
    _print_value_fmt = "Force error: "

    # TODO: use a pytree input for params_to_deflate_with
    def __init__(
        self,
        eq,
        eqs,
        sigma=0.05,
        power=2,
        # TODO: update to pytree like DeflationOperator is
        params_to_deflate_with=["Rb_lmn", "Zb_lmn"],
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="force-deflated",
        jac_chunk_size=None,
        deflation_type="power",
        multiple_deflation_type="prod",
        single_shift=False,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._eqs = eqs
        self._sigma = sigma
        self._power = power
        self._params_to_deflate_with = params_to_deflate_with
        assert deflation_type in ["power", "exp"]
        self._deflation_type = deflation_type
        assert multiple_deflation_type in ["prod", "sum"]
        self._multiple_deflation_type = multiple_deflation_type
        self._single_shift = single_shift

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
            grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        self._dim_f = 2 * grid.num_nodes
        self._data_keys = [
            "F_rho",
            "|grad(rho)|",
            "sqrt(g)",
            "F_helical",
            "|e^helical*sqrt(g)|",
        ]

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
            self._normalization = scales["f"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute MHD force balance errors.

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
            MHD force balance error at each node (N).

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
        fr = data["F_rho"] * data["|grad(rho)|"] * data["sqrt(g)"]
        fb = data["F_helical"] * data["|e^helical*sqrt(g)|"]
        keys = self._params_to_deflate_with
        # TODO: better to do only surface, or every key? seems like can obtain
        # very different results depending on if using only surf or every key, but
        # at same time, only the surface matters as far as closeness of solution
        # (the surface dictates the equilibrium, more or less)
        diffs = [
            jnp.concatenate([params[key] - eq.params_dict[key] for key in keys])
            for eq in self._eqs
        ]
        diffs = jnp.vstack(diffs)

        if self._deflation_type == "power":
            M_i = jnp.prod(
                1 / jnp.linalg.norm(diffs, axis=1) ** self._power
                + self._sigma * (not self._single_shift)
            )
        else:
            M_i = jnp.prod(
                jnp.exp(1 / jnp.linalg.norm(diffs, axis=1))
                + self._sigma * (not self._single_shift)
            )

        if self._multiple_deflation_type == "prod":
            deflation_parameter = jnp.prod(M_i) + self._sigma * (self._single_shift)
        else:
            deflation_parameter = jnp.sum(M_i) + self._sigma * (self._single_shift)

        return jnp.concatenate([fr, fb]) * deflation_parameter


class ForceBalanceAnisotropic(_Objective):
    """Force balance for anisotropic pressure equilibria.

    Solves for F = J × B − ∇ ⋅ Π = 0

    Where Π is the anisotropic pressure tensor of the form Π = (p_∥ - p_⊥)𝐛𝐛 + p_⊥𝕀

    Expanded out, this gives:

    F =  (1−βₐ)J × B − 1/μ₀ (B ⋅ ∇ βₐ)B − βₐ ∇(B²/2μ₀) − ∇(p_⊥)

    where βₐ is the anisotropy term: βₐ = μ₀ (p_∥ − p_⊥)/B²

    For this objective, the standard ``Equilibrium.pressure`` profile is used for p_⊥,
    and ``Equilibrium.anisotropy`` is used for βₐ. To get fully 3D anisotropy, these
    should be ``FourierZernikeProfile``, not the standard ``PowerSeriesProfile`` (which
    is only a function of rho).

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _units = "(N)"
    _coordinates = "rtz"
    _equilibrium = True
    _print_value_fmt = "Anisotropic force error: "

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
        name="force-anisotropic",
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
            grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        self._dim_f = 3 * grid.num_nodes
        self._data_keys = ["F_anisotropic", "sqrt(g)"]

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
            self._normalization = scales["f"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute MHD force balance errors.

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
            MHD force balance error at each node (N).

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
        f = (data["sqrt(g)"] * data["F_anisotropic"].T).T

        return f.flatten(order="F")  # to line up with quad weights


class RadialForceBalance(_Objective):
    r"""Radial MHD force balance.

    Fᵨ = √g (B^ζ J^θ - B^θ J^ζ) - ∇ p

    fᵨ = Fᵨ ||∇ ρ|| dV  (N)

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _equilibrium = True
    _coordinates = "rtz"
    _units = "(N)"
    _print_value_fmt = "Radial force error: "

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
        name="radial force",
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
            grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["F_rho", "|grad(rho)|", "sqrt(g)"]

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
            self._normalization = scales["f"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute radial MHD force balance errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f_rho : ndarray
            Radial MHD force balance error at each node (N).

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
        return data["F_rho"] * data["|grad(rho)|"] * data["sqrt(g)"]


class HelicalForceBalance(_Objective):
    r"""Helical MHD force balance.

    Fₕₑₗᵢ √g J^ρ

    𝐞ʰᵉˡⁱ = −B^ζ ∇ θ + B^θ ∇ ζ

    fₕₑₗᵢ = Fₕₑₗᵢ ||𝐞ʰᵉˡⁱ|| dV  (N)

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _equilibrium = True
    _coordinates = "rtz"
    _units = "(N)"
    _print_value_fmt = "Helical force error: "

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
        name="helical force",
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
            grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        self._data_keys = ["F_helical", "|e^helical|", "sqrt(g)"]

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
            self._normalization = scales["f"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute helical MHD force balance errors.

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
            Helical MHD force balance error at each node (N).

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
        return data["F_helical"] * data["|e^helical|"] * data["sqrt(g)"]


class Energy(_Objective):
    """MHD energy.

    W = integral( ||B||^2 / (2*mu0) + p / (gamma - 1) ) dV  (J)

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid)``
    gamma : float, optional
        Adiabatic (compressional) index. Default = 0.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _scalar = True
    _coordinates = ""
    _equilibrium = True
    _units = "(J)"
    _print_value_fmt = "Total MHD energy: "
    _io_attrs_ = _Objective._io_attrs_ + ["gamma"]

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
        gamma=0,
        name="energy",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self.gamma = gamma
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
            grid = QuadratureGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
            )
        else:
            grid = self._grid

        self._dim_f = 1
        self._data_keys = ["W"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "gamma": self._gamma,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["W"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute MHD energy.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        W : float
            Total MHD energy in the plasma volume (J).

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
        )
        return data["W"]

    @property
    def gamma(self):
        """float: Adiabatic (compressional) index."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma


class CurrentDensity(_Objective):
    """Radial, poloidal, and toroidal current density.

    Useful for solving vacuum equilibria.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _equilibrium = True
    _coordinates = "rtz"
    _units = "(A*m)"
    _print_value_fmt = "Current density: "

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
        name="current density",
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
            grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        self._dim_f = 3 * grid.num_nodes
        self._data_keys = ["J^rho", "J^theta", "J^zeta", "sqrt(g)"]

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
            self._normalization = scales["J"] * scales["V"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute toroidal current density.

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
            Toroidal current at each node (A*m).

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
        jr = data["J^rho"] * data["sqrt(g)"]
        jt = data["J^theta"] * data["sqrt(g)"]
        jz = data["J^zeta"] * data["sqrt(g)"]

        return jnp.concatenate([jr, jt, jz])
