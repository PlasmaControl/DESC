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


class CartesianForceBalance(_Objective):
    r"""Cartesian MHD force balance.
    Not sure about equations

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
    name : str, optional
        Name of the objective function.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _equilibrium = True
    _coordinates = "rtz"
    _units = "(N)"
    _print_value_fmt = "Cartesian force error:  "

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="cartesian force",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        super().__init__(
            eq=eq,
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
            if eq.node_pattern is None or eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
                grid = ConcentricGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                    node_pattern=eq.node_pattern,
                )
            elif eq.node_pattern == "quad":
                grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )
        else:
            grid = self._grid

        self._dim_f = grid.num_nodes
        # Change. Not sure e^x
        self._data_keys = ["F_cartesian", "|e^x|", "sqrt(g)"]
        # --no-verify self._args = get_params(
        # --no-verify     self._data_keys,
        # --no-verify     obj="desc.equilibrium.equilibrium.Equilibrium",
        # --no-verify     has_axis=grid.axis.size,
        # --no-verify )

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

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute cartesian MHD force balance errors.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile (Pa).
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile (A).
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).
        Te_l : ndarray
            Spectral coefficients of Te(rho) -- electron temperature profile (eV).
        ne_l : ndarray
            Spectral coefficients of ne(rho) -- electron density profile (1/m^3).
        Ti_l : ndarray
            Spectral coefficients of Ti(rho) -- ion temperature profile (eV).
        Zeff_l : ndarray
            Spectral coefficients of Zeff(rho) -- effective atomic number profile.

        Returns
        -------
        f : ndarray
            Cartesian MHD force balance error at each node (N).

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
        return data["F_cartesian"] * data["|e^x|"] * data["sqrt(g)"]


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
