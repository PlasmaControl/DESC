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


class ForceBalanceGalerkin(_Objective):
    """Projection of MHD force balance onto the basis functions.

    Given force densities:

    Fᵨ = √g (B^ζ J^θ - B^θ J^ζ) - ∇ p

    Fₕₑₗᵢ √g J^ρ

    and helical basis vector:

    𝐞ʰᵉˡⁱ = −B^ζ ∇ θ + B^θ ∇ ζ

    Minimizes the projection of the forces onto the Fourier-Zernike basis.

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
        target_default="``bounds=(-np.inf,0)``.",
        bounds_default="``bounds=(-np.inf,0)``.",
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
        basis=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="force galerkin",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._basis = basis
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
        from desc.basis import FourierZernikeBasis
        from desc.transform import Transform

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

        self._dim_f = eq.R_basis.num_modes + eq.Z_basis.num_modes
        self._data_keys = [
            "F_rho",
            "|grad(rho)|",
            "sqrt(g)",
            "F_helical",
            "|e^helical|",
        ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)
        if self._basis is None:
            transform_nyq_R = transforms["R"]
            transform_nyq_Z = transforms["Z"]
        else:
            L = self._basis.L
            M = self._basis.M
            N = self._basis.N
            Rsym = "cos" if eq.sym else False
            Zsym = "sin" if eq.sym else False
            transform_nyq_R = Transform(
                grid, FourierZernikeBasis(L, M, N, eq.NFP, Rsym)
            )
            transform_nyq_Z = Transform(
                grid, FourierZernikeBasis(L, M, N, eq.NFP, Zsym)
            )
            self._dim_f = (
                transform_nyq_R.basis.num_modes + transform_nyq_Z.basis.num_modes
            )
        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
            "quad_weights": jnp.ones(self._dim_f),
            "nyq_trans_R": transform_nyq_R,
            "nyq_trans_Z": transform_nyq_Z,
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
            MHD force balance error at each node (N) projected onto the FourierZernike
            basis.

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

        fr = data["F_rho"] * data["|grad(rho)|"]
        fr = fr * data["sqrt(g)"] * constants["transforms"]["grid"].weights

        fb = data["F_helical"] * data["|e^helical|"]
        fb = fb * data["sqrt(g)"] * constants["transforms"]["grid"].weights

        fr_proj = constants["nyq_trans_R"].project(fr)
        fb_proj = constants["nyq_trans_Z"].project(fb)

        return jnp.concatenate([fr_proj, fb_proj])


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
