"""Objective for finite-n ideal MHD stability, computed by AGNI (``agnimhd``)."""

import numpy as np

from desc.backend import jax, jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid, QuadratureGrid
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import errorif, setdefault

# Flux functions are computed on a LinearGrid at the rho of the AGNI nodes (it
# has the quadrature weights the AGNI nodes lack) and copied onto the nodes.
_FLUX_KEYS = [
    "iota",
    "iota_r",
    "iota_den",
    "iota_den_r",
    "iota_num",
    "iota_num_r",
    "iota_num current",
    "iota_num_r current",
    "iota_num vacuum",
    "iota_num_r vacuum",
    "psi_r",
    "psi_rr",
    "p",
    "p_r",
]


def _import_agnimhd():
    try:
        import agnimhd
    except ImportError as e:
        raise ImportError(
            "AgniStability needs the AGNI solver: pip install agnimhd"
        ) from e
    return agnimhd


class AgniStability(_Objective):
    """Finite-n ideal MHD stability: squared growth rate of the most unstable mode.

    AGNI [1]_ discretizes the ideal MHD energy principle::

        δW = ∫ dV [ |C|² + Γ p |∇·ξ|² − F |ξ·∇ρ|² ],    δK = ∫ dV n |ξ|²

    on a straight-field-line (PEST) grid, where ξ is the plasma displacement,
    C the field-line bending, Γ the adiabatic index, F the instability drive
    (the only term that can be negative) and n the normalized density (1 by
    default). The discrete problem A x = λ B x has A from δW and B from δK.
    This objective returns the normalized squared growth rate γ² = −λ of the
    lowest eigenvalue. The unnormalized growth rate is γ (v_A / a) where
    v_A = B_N/sqrt(mu_0 n M) is the Alfven speed. The normalizing scalars
    ``a`` and ``B_N = |Ψ| / (π a²)`` are recomputed from the parameters at every
    iteration. More information about the normalization in the preprint [1]_


    The applied perturbations do not change the ρ=1 boundary (ξ·∇ρ = 0 there),
    so the modes are global internal modes:

    - γ² > 0: unstable to a global internal mode of the chosen toroidal family,
    - γ² ≤ 0: stable.

    The default target 0 asks for marginal stability. Unlike
    ``BallooningStability`` (infinite n, one field line at a time) the mode is
    global: it can couple all flux surfaces and poloidal harmonics.

    The nodes are fixed in PEST coordinates (ρ, ϑ, ζ) by ``basis``. At every
    call they are mapped to DESC's θ at the current parameters, so the value and
    its gradient follow the equilibrium as it changes. The gradient is exact at a
    converged eigenvector (Hellmann-Feynman): no derivative of the eigensolver
    is taken.

    Requires the ``agnimhd`` package (``pip install agnimhd``, Python ≥ 3.12).
    Its documentation explains the bases, solvers and options:
    https://rahulgaur104.github.io/AGNI/

    Parameters
    ----------
    eq : Equilibrium
        ``Equilibrium`` to be optimized.
    basis : agnimhd.Basis
        The AGNI grid and its spectral basis, e.g. ``agnimhd.Basis(24, 12, 8)``
        for 24 radial, 12 poloidal and 8 toroidal (per field period) nodes.
        Fixed during an optimization.
    family : int
        Toroidal mode family ``n = family + k NFP`` whose γ² is returned.
        ``family=0`` contains the n = 0, NFP, 2 NFP, ... modes. All families
        together are ``0, ..., NFP // 2``; use one objective per family.
        Default 0.
    assembly : agnimhd.AssemblyConfig, optional
        Physics options, such as the adiabatic index ``gamma`` (default 5/3).
    solver : agnimhd.SolverConfig, optional
        Eigensolver and its shift ``sigma``, which must lie above the largest γ²
        the optimizer will meet. Default ``agnimhd.SolverConfig()``: ``"eigsh"``
        (ARPACK on the host) with ``sigma=0.1``. ``"dense"`` runs on the GPU,
        ``"jd"`` is matrix-free for large bases.
    coarse : agnimhd.Basis, optional
        Only for ``solver.eigensolver="jd"``: the coarse level, which must be
        ``basis.coarse(n_theta, n_zeta)``. Default ``basis.coarse()``.
    density : bool
        Weight the kinetic energy with ``ni / max(ni)``. Needs
        ``eq.electron_density``. Default False.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        normalize_detail=" Note: Has no effect for this objective.",
        normalize_target_detail=" Note: Has no effect for this objective.",
    )
    __doc__ += """

    References
    ----------
    .. [1] R. Gaur et al., "AGNI: a differentiable MHD stability solver and
       optimizer for magnetic confinement fusion devices", arXiv:2608.01750 (2026).

    """

    _static_attrs = _Objective._static_attrs + [
        "_agnimhd",
        "_basis",
        "_family",
        "_assembly",
        "_solver",
        "_coarse_basis",
        "_density",
    ]
    _coordinates = ""
    _units = "(dimensionless)"
    _print_value_fmt = "Finite-n γ²: "

    def __init__(
        self,
        eq,
        basis,
        family=0,
        assembly=None,
        solver=None,
        coarse=None,
        density=False,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name="AGNI stability",
        jac_chunk_size=None,
        device_id=0,
        rank=None,
    ):
        if target is None and bounds is None:
            target = 0
        agnimhd = _import_agnimhd()
        self._agnimhd = agnimhd
        self._basis = basis
        self._family = int(family)
        self._assembly = setdefault(assembly, agnimhd.AssemblyConfig())
        self._solver = setdefault(solver, agnimhd.SolverConfig())
        self._density = bool(density)
        errorif(
            self._density and eq.electron_density is None,
            ValueError,
            "density=True needs a density profile, please make sure that "
            "eq.electron_density is not None.",
        )
        self._coarse_basis = None
        if self._solver.eigensolver == "jd":
            coarse = basis.coarse() if coarse is None else coarse
            errorif(
                coarse != basis.coarse(coarse.n_theta, coarse.n_zeta),
                ValueError,
                "coarse must be basis.coarse(n_theta, n_zeta): the same radial "
                "nodes, map, mpol and ntor on fewer angular nodes.",
            )
            self._coarse_basis = coarse

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
            device_id=device_id,
            rank=rank,
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
        rho, diffmat, nodes = self._pest_nodes(self._basis)
        flux_grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        # a is a cross-section area integral; it needs a QuadratureGrid.
        quad_grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)
        self._dim_f = 1
        self._constants = {
            "diffmat": diffmat,
            "nodes": nodes,
            "flux_grid": flux_grid,
            "flux_transforms": get_transforms(_FLUX_KEYS, obj=eq, grid=flux_grid),
            "flux_profiles": get_profiles(_FLUX_KEYS, obj=eq, grid=flux_grid),
            "a_transforms": get_transforms(["a"], obj=eq, grid=quad_grid),
            "a_profiles": get_profiles(["a"], obj=eq, grid=quad_grid),
            "quad_weights": 1.0,
        }
        if self._coarse_basis is not None:
            # same radial nodes as the fine level, so the same flux grid
            c = self._constants
            c["coarse_nodes"] = self._pest_nodes(self._coarse_basis)[2]
            eq_c = self._equilibrium_data(eq.params_dict, c, coarse=True)
            _, c["coarse_diffmat"], c["coarse_transfer"] = self._basis.coarse_level(
                eq_c, self._family
            )
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute the squared growth rate of the most unstable mode.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, e.g.
            ``Equilibrium.params_dict``.
        constants : dict
            Dictionary of constant data, e.g. transforms, profiles etc.
            Defaults to ``self.constants``. (Deprecated)

        Returns
        -------
        gamma2 : ndarray, shape (1,)
            Squared growth rate γ² = −λ, positive when unstable.

        """
        c = self._get_deprecated_constants(constants)
        coarse = None
        if self._coarse_basis is not None:
            eq_c = self._equilibrium_data(params, c, coarse=True)
            coarse = (eq_c, c["coarse_diffmat"], c["coarse_transfer"])

        def solve(p):
            return jnp.asarray(
                self._agnimhd.growth_rate(
                    p,
                    c["diffmat"],
                    self._assembly,
                    self._solver,
                    coarse=coarse,
                    equilibrium_map=lambda q: self._equilibrium_data(q, c),
                ),
                dtype=float,
            ).reshape(())

        # An optimizer's trial step can reach parameters where the equilibrium is
        # not nested on the AGNI nodes (the Jacobian sqrt(g) changes sign) or the
        # data is not finite. The assembly then produces NaN matrices, the host
        # eigensolver raises, and the whole run ends (10-25: a first lsq-exact step
        # on the QH test case took sqrt(g) negative at 56 of 192 nodes). Return NaN
        # instead, so the optimizer rejects the step and shrinks its trust region,
        # as it does for any non-finite trial cost.
        data = self._equilibrium_data(params, c)
        leaves = jax.tree_util.tree_leaves(data)
        finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(x)) for x in leaves]))
        sqrtg = data.sqrtg
        nested = jnp.all(sqrtg > 0) | jnp.all(sqrtg < 0)
        gamma2 = jax.lax.cond(
            finite & nested, solve, lambda p: jnp.array(jnp.nan), params
        )
        return jnp.atleast_1d(gamma2)

    def _pest_nodes(self, basis):
        """Return the rho values, DiffMat and PEST nodes (rho-major) of ``basis``."""
        eq = self.things[0]
        nodes, diffmat = basis.nodes_and_diffmat(eq.NFP, self._family)
        rho, theta, zeta = (np.asarray(nodes[k]) for k in ("rho", "theta", "zeta"))
        R, T, Z = np.meshgrid(rho, theta, zeta, indexing="ij")
        pest = np.stack([R.ravel(), T.ravel(), Z.ravel()], axis=-1)
        _, u_idx, i_idx = np.unique(pest[:, 0], return_index=True, return_inverse=True)
        nodes = {"pest": pest, "u_idx": u_idx, "i_idx": i_idx}
        return rho, diffmat, {k: jnp.asarray(v) for k, v in nodes.items()}

    def _equilibrium_data(self, params, c, coarse=False):
        """Evaluate the equilibrium on the AGNI nodes as an ``EquilibriumData``."""
        eq = self.things[0]
        agnimhd = self._agnimhd
        from agnimhd.adapters.desc import KEY_MAP  # DESC key -> AGNI field

        nodes = c["coarse_nodes"] if coarse else c["nodes"]
        rtz = eq.map_coordinates(
            nodes["pest"],
            inbasis=("rho", "theta_PEST", "zeta"),
            outbasis=("rho", "theta", "zeta"),
            period=(jnp.inf, 2 * jnp.pi, jnp.inf),
            tol=1e-12,
            maxiter=50,
            params=params,
        )
        grid = Grid(
            rtz,
            coordinates="rtz",
            sort=False,
            jitable=True,
            _unique_rho_idx=nodes["u_idx"],
            _inverse_rho_idx=nodes["i_idx"],
        )
        a = compute_fun(eq, ["a"], params, c["a_transforms"], c["a_profiles"])["a"]
        flux = compute_fun(
            eq,
            _FLUX_KEYS,
            params,
            c["flux_transforms"],
            c["flux_profiles"],
            data={"a": a},
        )
        data = {"a": a}
        for key in _FLUX_KEYS:
            data[key] = grid.copy_data_from_other(
                flux[key], c["flux_grid"], surface_label="rho"
            )
        keys = list(KEY_MAP) + (["ni"] if self._density else [])
        data = compute_fun(
            eq,
            keys,
            params,
            get_transforms(keys, eq, grid, jitable=True),
            get_profiles(keys, eq, grid),
            data=data,
        )
        basis = self._coarse_basis if coarse else self._basis
        return agnimhd.EquilibriumData(
            n_rho=basis.n_rho,
            n_theta=basis.n_theta,
            n_zeta=basis.n_zeta,
            NFP=eq.NFP,
            Psi=params["Psi"],
            a=a,
            density=data["ni"] / jnp.max(data["ni"]) if self._density else None,
            validate=False,
            **{field: data[key] for key, field in KEY_MAP.items()},
        )
