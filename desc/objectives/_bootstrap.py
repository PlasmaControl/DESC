"""Objectives related to the bootstrap current profile."""

import numpy as np

from desc.backend import jax, jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer, errorif, warnif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


def _check_bootstrap_inputs(eq, grid, helicity, name):
    """Check the grid and kinetic profiles for the Redl bootstrap current."""
    warnif(
        (grid.num_theta * (1 + eq.sym)) < 2 * eq.M,
        RuntimeWarning,
        f"{name} objective grid requires poloidal resolution for surface averages",
    )
    warnif(
        grid.num_zeta < 2 * eq.N,
        RuntimeWarning,
        f"{name} objective grid requires toroidal resolution for surface averages",
    )

    errorif(
        not (helicity[1] == 0 or abs(helicity[1]) == eq.NFP),
        ValueError,
        "Helicity toroidal mode number should be 0 (QA) or +/- NFP (QH)",
    )
    errorif(
        grid.axis.size,
        ValueError,
        "Redl formula is undefined at rho=0, but grid has grid points at rho=0",
    )

    errorif(
        eq.electron_temperature is None,
        RuntimeError,
        "Bootstrap current calculation requires an electron temperature profile.",
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
    errorif(
        eq.ion_density is None and eq.atomic_number is None,
        RuntimeError,
        "Bootstrap current calculation requires an ion density profile.",
    )

    rho = grid.compress(grid.nodes[:, 0], "rho")

    # Check if profiles are exactly zero anywhere, since this would cause NaNs.
    errorif(
        np.any(np.isclose(eq.electron_temperature(rho), 0.0, atol=1e-8)),
        ValueError,
        "Redl formula is undefined where kinetic profiles vanish, "
        "but given electron temperature vanishes at at least one provided"
        "rho grid point.",
    )
    errorif(
        np.any(np.isclose(eq.electron_density(rho), 0.0, atol=1e-8)),
        ValueError,
        "Redl formula is undefined where kinetic profiles vanish, "
        "but given electron density vanishes at at least one provided"
        "rho grid point.",
    )
    errorif(
        np.any(np.isclose(eq.ion_temperature(rho), 0.0, atol=1e-8)),
        ValueError,
        "Redl formula is undefined where kinetic profiles vanish, "
        "but given ion temperature vanishes at at least one provided"
        "rho grid point.",
    )
    if eq.ion_density is not None:
        errorif(
            np.any(np.isclose(eq.ion_density(rho), 0.0, atol=1e-8)),
            ValueError,
            "Redl formula is undefined where kinetic profiles vanish, "
            "but given ion density vanishes at at least one provided"
            "rho grid point.",
        )
    # Try to catch cases in which density or temperatures are specified in the
    # wrong units. Densities should be ~ 10^20, temperatures are ~ 10^3.
    warnif(
        jnp.any(eq.electron_temperature(rho) > 50e3),
        UserWarning,
        "Electron temperature is surprisingly high. It should have units of eV.",
    )
    warnif(
        jnp.any(eq.electron_density(rho) > 1e22),
        UserWarning,
        "Electron density is surprisingly high. "
        + "It should have units of 1/meters^3.",
    )
    warnif(
        jnp.any(eq.ion_temperature(rho) > 50e3),
        UserWarning,
        "Ion temperature is surprisingly high. It should have units of eV.",
    )
    if eq.ion_density is not None:
        warnif(
            jnp.any(eq.ion_density(rho) > 1e22),
            UserWarning,
            "Ion density is surprisingly high. It should have units of 1/meters^3.",
        )
    # Profiles may go to 0 at rho=1 (and we have already checked if our grid has
    # points there), so exclude the edge from lower bounds:
    rho = rho[rho < 0.85]
    warnif(
        jnp.any(eq.electron_temperature(rho) < 30),
        UserWarning,
        "Electron temperature is surprisingly low. It should have units of eV.",
    )
    warnif(
        jnp.any(eq.electron_density(rho) < 1e17),
        UserWarning,
        "Electron density is surprisingly low. It should have units of 1/meters^3.",
    )
    warnif(
        jnp.any(eq.ion_temperature(rho) < 30),
        UserWarning,
        "Ion temperature is surprisingly low. It should have units of eV.",
    )
    if eq.ion_density is not None:
        warnif(
            jnp.any(eq.ion_density(rho) < 1e17),
            UserWarning,
            "Ion density is surprisingly low. It should have units of 1/meters^3.",
        )


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

        _check_bootstrap_inputs(eq, grid, self.helicity, type(self).__name__)
        self._dim_f = grid.num_rho
        self._data_keys = ["<J*B>", "<J*B> Redl"]

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
            self.constants. (Deprecated)

        Returns
        -------
        obj : ndarray
            Bootstrap current self-consistency residual on each rho grid point.

        """
        constants = self._get_deprecated_constants(constants)
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


class BootstrapRedlCurrentConsistency(_Objective):
    """Self-consistency of the current profile with the Redl bootstrap current.

    This is a residual for the bootstrap current self-consistency that has one entry
    for each current profile parameter it determines, so that it can be solved
    together with force balance in a single equilibrium solve, with the current
    profile as one more unknown. This replaces both optimizing the current profile
    for ``BootstrapRedlConsistency`` and iteratively re-solving the equilibrium
    with the ``"current Redl"`` profile, eg::

        objective = ObjectiveFunction(
            (ForceBalance(eq), BootstrapRedlCurrentConsistency(eq, helicity=(1, 0)))
        )
        constraints = (
            FixBoundaryR(eq), FixBoundaryZ(eq), FixPsi(eq), FixCurrent(eq, [0, 1]),
            FixAtomicNumber(eq), FixElectronDensity(eq), FixElectronTemperature(eq),
            FixIonTemperature(eq),
        )
        eq.solve(objective=objective, constraints=constraints)

    With ``quantity="<J*B>"`` (the default) the residual is the difference
    ⟨J⋅B⟩_MHD − ⟨J⋅B⟩_Redl projected onto the test functions
    T_j = B₀² ∂ρφⱼ / (2Ψρ), where φⱼ are the basis functions of the current profile
    parameters given by ``indices`` and B₀ is the reference magnetic field strength.
    To leading order T_j is the change of ⟨J⋅B⟩_MHD with the j-th parameter, so a
    zero residual is the least squares fit of ⟨J⋅B⟩_MHD to ⟨J⋅B⟩_Redl, as found by
    optimizing the current profile for ``BootstrapRedlConsistency``. The test
    functions are fixed so that the solution does not depend on the state of the
    equilibrium when the objective is built.

    With ``quantity="current"`` the residual is zero when those parameters are the
    least squares fit of the ``"current Redl"`` profile (Equation C3 in Landreman &
    Catto (2012)), which is the solution that iteratively re-solving the equilibrium
    with the ``"current Redl"`` profile converges to (as in STELLOPT VBOOT).

    In both cases the residual is scaled to be the error in ⟨J⋅B⟩, so that its
    normalization is the same as for ``BootstrapRedlConsistency``.

    Because the discretized force balance equations are not satisfied exactly, the
    least squares solution of both sets of equations can lower the force balance
    error slightly by not satisfying this residual exactly. A larger ``weight``
    makes this negligible at no cost to the force balance error, as there are as
    many residuals as unknowns that they determine.

    The Redl formula is only valid for quasi-symmetric (or axisymmetric)
    equilibria.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
        Must have a current profile (not a rotational transform profile) and the
        kinetic profiles.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at. Requires poloidal and
        toroidal resolution as the objective must compute flux surface averages,
        and no nodes at the magnetic axis or where the kinetic profiles vanish.
        Defaults to
        ``grid = LinearGrid(M=eq.M_grid,N=eq.N_grid,NFP=eq.NFP,``
        ``sym=eq.sym,rho=np.linspace(1 / eq.L, 1, eq.L) - 1 / (2 * eq.L),)``
    helicity : tuple, optional
        Type of quasi-symmetry (M, N). Default = quasi-axisymmetry (1, 0).
        First entry must be M=1. Second entry is the toroidal mode number N,
        used for evaluating the Redl bootstrap current formula. Set to 0 for
        axisymmetry or quasi-axisymmetry; set to +/-NFP for quasi-helical symmetry.
    quantity : {"<J*B>", "current"}, optional
        Whether to fit ⟨J⋅B⟩ (same solution as optimizing for
        ``BootstrapRedlConsistency``) or the current profile (same solution as
        iterating with ``"current Redl"``). Default ``"<J*B>"``.
    indices : array of int, optional
        Indices of the current profile parameters ``eq.c_l`` that are determined by
        the self-consistency condition. The other parameters should be fixed with
        ``FixCurrent``. There should be no more of them than there are radial grid
        points. Default is all parameters except those of a ``PowerSeriesProfile``
        with modes 0 and 1, or of a ``SplineProfile`` with a knot at rho=0, so that
        the current and its derivative are fixed at the magnetic axis (where the
        Redl formula does not apply).
    degree : int, optional
        The `degree` kwarg to pass to the `current Redl` compute call, which
        controls the degree of polynomial fit to the Redl current derivative
        before it is integrated. Only used with ``quantity="current"``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _static_attrs = _Objective._static_attrs + ["_quantity"]
    _scalar = False
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
        quantity="<J*B>",
        indices=None,
        degree=None,
        name="Bootstrap current self-consistency (Redl, current profile)",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        assert (len(helicity) == 2) and (int(helicity[1]) == helicity[1])
        assert helicity[0] == 1, "Redl bootstrap current model assumes helicity[0] == 1"
        errorif(
            quantity not in ("<J*B>", "current"),
            ValueError,
            f"quantity should be '<J*B>' or 'current', got {quantity}.",
        )
        self._grid = grid
        self._helicity = helicity
        self._quantity = quantity
        self._indices = indices
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
        from desc.profiles import PowerSeriesProfile, SplineProfile

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

        errorif(
            eq.current is None,
            ValueError,
            "BootstrapRedlCurrentConsistency requires an equilibrium with a current "
            "profile, but it has a rotational transform profile.",
        )
        _check_bootstrap_inputs(eq, grid, self._helicity, type(self).__name__)

        if self._indices is None:
            if isinstance(eq.current, PowerSeriesProfile):
                modes = np.asarray(eq.current.basis.modes[:, 0])
                indices = np.where(modes >= 2)[0]
            elif isinstance(eq.current, SplineProfile):
                indices = np.where(np.asarray(eq.current.knots) > 0)[0]
            else:
                indices = np.arange(eq.c_l.size)
        else:
            indices = np.atleast_1d(np.asarray(self._indices, dtype=int))
        indices = np.unique(indices)
        rho = grid.compress(grid.nodes[:, 0], "rho")
        errorif(
            indices.size > rho.size,
            ValueError,
            f"BootstrapRedlCurrentConsistency determines {indices.size} current "
            f"profile parameters, which needs at least as many radial grid points, "
            f"but the grid has {rho.size}.",
        )

        # Basis functions of the current profile parameters (and their derivatives)
        # at the grid points. The profile is linear in its parameters for the usual
        # profile types, so this is exact for them.
        c = jnp.asarray(eq.c_l)
        basis = np.asarray(jax.jacfwd(lambda c: eq.current(rho, params=c))(c))
        basis_r = np.asarray(jax.jacfwd(lambda c: eq.current(rho, params=c, dr=1))(c))
        basis, basis_r = basis[:, indices], basis_r[:, indices]
        # Test functions: to leading order the change in <J*B> ≈ B₀² I' / (2Ψρ)
        # of an equilibrium (in DESC units) with each of the parameters.
        scales = compute_scaling_factors(eq)
        test = scales["B"] ** 2 * basis_r / (2 * np.abs(eq.Psi) * rho[:, None])
        # radial quadrature weights
        w = np.sqrt(grid.compress(grid.spacing[:, 0], surface_label="rho"))

        def _orthonormalize(A, what):
            Q, R = np.linalg.qr(w[:, None] * A)
            d = np.abs(np.diag(R))
            errorif(
                np.any(d <= 1e-10 * np.max(d)),
                ValueError,
                f"The {what} of the current profile parameters given by indices are "
                "not linearly independent on the radial grid points. Note that a "
                "constant current (eg c_0 of a PowerSeriesProfile) is not determined "
                "by the bootstrap current, so it should be fixed.",
            )
            return Q, R

        Qt, Rt = _orthonormalize(test, "derivatives")
        if self._quantity == "<J*B>":
            self._data_keys = ["<J*B>", "<J*B> Redl"]
            # Galerkin projection onto the test functions
            projection = Qt.T * w
        else:
            self._data_keys = ["current", "current Redl"]
            # Least squares fit of the current profile difference (Δc), scaled so
            # its norm is the norm of the resulting change in <J*B>, |Rt Δc|.
            Qb, Rb = _orthonormalize(basis, "basis functions")
            projection = Rt @ np.linalg.solve(Rb, Qb.T * w)
        self._dim_f = indices.size

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
            "projection": jnp.asarray(projection),
            "quad_weights": 1.0,
        }
        if self._degree is not None:
            self._constants["degree"] = self._degree

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            self._normalization = scales["B"] * scales["J"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the bootstrap current self-consistency residual.

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
            Projection of the difference between the MHD and Redl profiles of
            ⟨J⋅B⟩ or current, one for each current profile parameter that is
            determined by the self-consistency condition (T A m^-2).

        """
        constants = self._get_deprecated_constants(constants)
        kwargs = {"helicity": constants["helicity"]}
        if "degree" in constants:
            kwargs["degree"] = constants["degree"]
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
            **kwargs,
        )
        mhd, redl = self._data_keys
        grid = constants["transforms"]["grid"]
        return constants["projection"] @ grid.compress(data[mhd] - data[redl])

    @property
    def helicity(self):
        """tuple: Type of quasi-symmetry (M, N)."""
        return self._helicity

    @property
    def quantity(self):
        """str: Whether ⟨J⋅B⟩ or the current profile is fit."""
        return self._quantity
