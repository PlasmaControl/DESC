import warnings

import numpy as np
from scipy.constants import mu_0

from desc.backend import jit, jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.objectives.normalization import compute_scaling_factors
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.utils import Timer, errorif

from ._quadcoil_utils import (
    _BCOIL_DATA_KEYS,
    _BPLASMA_DATA_KEYS,
    _compute_Bnormal,
    _compute_Bnormal_ext,
    _compute_Bnormal_plasma,
    _compute_Bplasma,
    _compute_eval_data_coils,
    _compute_G,
    _create_source,
    _ptolemy_identity_rev_compute,
    _ptolemy_identity_rev_precompute,
    _quadcoil_kwargs_to_field_kwargs,
    _quadcoil_phi_to_desc_phi_gather,
)

# ----- A QUADCOIL wrapper -----
# A list of all inputs of quadoil.quadcoil
# that can be extracted from DESC. The
# rest cannot. These variables should not
# show up in quadcoil_kwargs. If they do, they will be ignored.
_DESC_DERIVED_ARGNAMES = [
    "nfp",
    "stellsym",
    "plasma_mpol",
    "plasma_ntor",
    "plasma_quadpoints_phi",
    "plasma_quadpoints_theta",
    "plasma_dofs",
    "net_poloidal_current_amperes",
    "Bnormal_plasma",
    "metric_name",
    "value_only",
]

# A list of argnames that must be user-provided,
# but are considered differentiable by JAX.
# Variables here will be excluded when constructing
# nondiff_args, the non-differentiable argument of
# quadcoil.io.quadcoil_for_diff
_DIFF_USER_ARGNAMES = [
    "net_toroidal_current_amperes"
    "plasma_coil_distance"
    "objective_weight"
    "constraint_value"
]

_normalize_target_detail = (
    "Normalization options for a typical DESC Objective. Disabled by default. "
    "When enabled, overrides the ``<quantity>_unit`` in "
    "``quadcoil_kwargs`` with scaling constants calculated from the parameteres "
    "of the DESC equilibrium. Note that QUADCOIL usually works the best "
    "when ``<quantity>_unit`` are the same quantities measured from another "
    "winding surface solution (either the solution of the same problem with "
    "``<quantity>_unit=1``, or the solution of a REGCOIL problem). This is "
    "because coil metrics at a QUADCOIL optimum can differ by orders of "
    "magnitudes from the DESC auto-calculated values. Setting "
    "this to ``True`` may impact QUADCOIL's accuracy."
)


class QuadcoilProxy(_Objective):
    """
    A QUADCOIL-based coil complexity proxy.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    quadcoil_kwargs : dict
        A dictionary containing all inputs for ``quadcoil.quadcoil`` (see the
        `QUADCOIL documentation
        <https://quadcoil.readthedocs.io/en/latest/tutorial_outputs.html>`__).
        The following quantities are automatically extracted from DESC and
        will be ignored:

        .. code-block:: python

            nfp,
            stellsym,
            plasma_mpol, plasma_ntor,
            plasma_quadpoints_phi, plasma_quadpoints_theta,
            plasma_dofs,
            net_poloidal_current_amperes,
            Bnormal_plasma,
            metric_name,
            value_only,
            verbose,
    plasma_M_theta : int, optional
        The plasma poloidal quadrature resolution. Determines the
        resolution of QUADCOIL plasma surface integrals and point-wise
        functions.
        Unlike the winding surface quadrature points, which is a required input,
        the plasma surface quadpoints is evaluated from a linear grid to make
        sure that the grid points in DESC B calculations line up exactly
        with the QUADCOIL grids.
        Values lower than eq.M_grid will trigger interpolation truncation
        warnings. Default = eq.M_grid.
    plasma_N_phi : int, optional
        The plasma toroidal quadrature resolution. Default = eq.N_grid.
    metric_name : str or tuple of str
        The coil property(ies) to measure as the value of the proxy.
        We strongly advise using the default value to ensure accurate adjoint
        differentiation. Default = "f_obj", which uses the normalized QUADCOIL
        objective.
    metric_target : scalar or ndarray
        In addition to target, bounds and weight,
        The QUADCOIL proxy objective allows the user to set weights and
        targets for each objective terms individually besides using ``target``
        and ``bounds`` that comes with other DESC objectives.
        Targets of each property. Default = 0.0.
    metric_weight : scalar or ndarray
        Weights of each property. Default = 1.0.
    vacuum : bool, optional
        Whether to enable Bnormal contributions from plasma current.
        Default = False.
    verbose : int, optional
        Whether to enable verbose output. Default = 0.
    source_grid : Grid, optional
        Grid for evaluating vacuum casing and the required net poloidal coil
        current. Default = None, which uses a
        ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    field : list or CoilSet, optional
        Other coils to use in combinations with the winding surface.
        Can be optimized. For combined filament-dipole modeling/optimization.
        Default = [].
    field_grid : Grid, optional
        The grid for ``field``. Default = None.
    enable_net_current_plasma : bool, optional
        Whether to enable a net poloidal current in the winding surface.
        Default = True.
    eq_fixed : bool, optional
        Whether to fix ``eq``, or make it optimizable degrees of freedom.
        Default = False for quasi-single-stage optimization.
    field_fixed : bool, optional
        Whether to fix ``field``, or make it optimizable degrees of freedom.
        Default = False for quasi-single-stage dipole/PM optimization with known
        filament coils.
    B_plasma_chunk_size : int or None
        Size to split singular integral computation for B_plasma into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default = ``bs_chunk_size``.
    bs_chunk_size : int, optional
        Size to split Biot-Savart computation into chunks of evaluation points.
        Forwarded to QUADCOIL as ``quadcoil_kwargs["bs_chunk_size"]`` so the
        winding-surface and self-field kernels also walk evaluation points in
        batches. Default = None.
    """

    # ----- Setting and registering keyword arguments -----
    _static_attrs = _Objective._static_attrs + [
        # External-coils related
        "_enable_net_current_plasma",
        "_eq_fixed",
        "_field_fixed",
        # Free-boundary-related
        "_bplasma_chunk_size",
        "_vacuum",
        # QUADCOIL-related
        "metric_name",
        "nfp",
        "stellsym",
        "_plasma_M_theta",
        "_plasma_N_phi",
        "_quadcoil_for_diff",
        "_quadcoil_values",
        "_Bnormal_shape",
        # VMEC <=> DESC
        "_surf_R_A",
        "_surf_R_c_indices",
        "_surf_R_s_indices",
        "_surf_Z_A",
        "_surf_Z_c_indices",
        "_surf_Z_s_indices",
    ]

    # Most of the documentation is shared among all objectives, so we just
    # inherit the docstring from the base class and add a few details specific
    # to this objective.
    # See the documentation of `collect_docs` for more details.
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``bound=None``.",
        normalize_target_detail=_normalize_target_detail,
    )

    _coordinates = ""  # What coordinates is this objective a function of, with
    # r=rho, t=theta, z=zeta? i.e. if only a profile, it is "r" , while if all
    # 3 coordinates it is "rtz"
    _units = "N/A"  # units of the output
    # string with python string formatting for printing the value
    _print_value_fmt = "QUADCOIL subproblem: "
    # Subclasses can append extra equilibrium data keys for the eval grid.
    _extra_eval_data_keys = []

    def __init__(  # noqa: C901
        self,
        eq,
        quadcoil_kwargs,
        plasma_M_theta=None,
        plasma_N_phi=None,
        target=None,
        bounds=None,
        weight=1,
        metric_name="f_obj",
        metric_weight=1.0,
        metric_target=0.0,
        vacuum: bool = False,
        normalize=False,
        normalize_target=False,
        verbose=0,
        name="QUADCOIL Proxy",
        source_grid=None,
        # External coils - no external coils by default
        field=None,
        field_grid=None,
        enable_net_current_plasma=True,
        eq_fixed=False,  # Whether the equilibrium are fixed
        field_fixed=False,  # Whether the external fields are fixed
        # misc
        B_plasma_chunk_size=None,
        jac_chunk_size=None,
        bs_chunk_size=None,
    ):
        # Importing QUADCOIL
        try:
            from quadcoil.io import gen_quadcoil_for_diff
        except ModuleNotFoundError:
            raise ModuleNotFoundError("QuadcoilProxy requires a QUADCOIL installation.")

        self._enable_net_current_plasma = enable_net_current_plasma
        self._eq_fixed = eq_fixed
        self._eq = eq
        if field:  # To be also tolerant on `False` and `None` as an input
            self._field = [field] if not isinstance(field, list) else field
            self._field_fixed = field_fixed
        else:
            self._field = []
            self._field_fixed = True
        # Things initialization
        things = []
        if not (self._eq_fixed or self._field_fixed):
            warnings.warn(
                "Both eq_fixed and field_fixed are True. things will be empty."
            )
        if not self._eq_fixed:
            things += [eq]
        if not self._field_fixed:
            things += [field]

        if not (enable_net_current_plasma or field):
            warnings.warn(
                "enable_net_current_plasma is false and field is empty. "
                "The problem may be trivial."
            )

        if enable_net_current_plasma and field:
            warnings.warn(
                "There are both external coils and net current. "
                "This is very uncommon (windowpane filaments + "
                "winding surface with net current)."
            )

        quadcoil_kwargs = quadcoil_kwargs.copy()
        if target is None and bounds is None:
            target = 0  # default target value
        # Uses LSE to smooth non-smooth problems rather than slack variables
        # by default.
        quadcoil_kwargs.setdefault("smoothing", "approx")

        # ----- Checking inputs -----
        # Checking whether all metrics have a weight and a target provided.
        # By default, the metric is the quadcoil objective. This choice
        # empirically has the most accurate adjoint gradients.
        if isinstance(metric_name, str):
            if (not jnp.isscalar(metric_target)) or (not jnp.isscalar(metric_weight)):
                raise ValueError(
                    "When metric_name is a str, metric_target and "
                    "metric_target must both be scalar."
                )
            # Makign them into iterables will make things easier when
            # scaling in the end.
            metric_name = (metric_name,)
            metric_weight = jnp.array([metric_weight])
            metric_target = jnp.array([metric_target])
        elif isinstance(metric_name, tuple):
            if len(metric_target) != len(metric_name):
                raise KeyError(
                    "metric_name and metric_target have mismatching lengths!"
                )
            if len(metric_weight) != len(metric_name):
                raise KeyError(
                    "metric_name and metric_weight have mismatching lengths!"
                )
        else:
            raise ValueError("metric_name must be a tuple or a str.")
        # Detect if the user has provided any arguments
        # that will also-be extracted from DESC.
        # If there are, these objectives will be discarded.
        redundant_arg_names = set(_DESC_DERIVED_ARGNAMES) & quadcoil_kwargs.keys()
        if redundant_arg_names:
            warnings.warn(
                f"Redundant arguments detected: {redundant_arg_names}. "
                "These arguments are extracted from the equilibrium, "
                "or specified by other parameters. The provided values "
                "will be discarded."
            )

        # ----- Storing equilibrium-independent, differentiable variables -----
        # These are differentiable quantities that are not equilibrium-dependent.
        # They can be user-provided, but they also all have default values, so
        # we set them here. This is necessary because we are calling quadcoil
        # through quadcoil.io.quadcoil_for_diff, which cannot see their default
        # values in quadcoil.quadcoil.
        self.net_toroidal_current_amperes = quadcoil_kwargs.pop(
            "net_toroidal_current_amperes", 0.0
        )
        # A sign flip is necessary here since simsopt and DESC surfaces
        # have different handedness. QUADCOIL uses the simsopt convention.
        _plasma_coil_distance = quadcoil_kwargs.pop("plasma_coil_distance", None)
        self.plasma_coil_distance = (
            -_plasma_coil_distance if _plasma_coil_distance is not None else None
        )
        self.winding_dofs = quadcoil_kwargs.pop("winding_dofs", None)
        self.objective_weight = quadcoil_kwargs.pop("objective_weight", None)
        self.constraint_value = quadcoil_kwargs.pop("constraint_value", jnp.array([]))

        # ----- Setting attributes -----
        self.metric_name = metric_name
        self.metric_target = metric_target
        self.metric_weight = metric_weight
        self._verbose = verbose
        self._bplasma_chunk_size = B_plasma_chunk_size
        self._bs_chunk_size = bs_chunk_size
        self._vacuum = vacuum
        if not plasma_M_theta:
            plasma_M_theta = eq.M_grid
        elif plasma_M_theta <= eq.M_grid:
            warnings.warn(
                f"plasma_M_theta = {plasma_M_theta} <= eq.M_grid = {eq.M_grid}. "
                "An interpolation truncation warning may appear."
            )
        if not plasma_N_phi:
            plasma_N_phi = eq.N_grid
        elif plasma_N_phi <= eq.N_grid:
            warnings.warn(
                f"plasma_N_phi = {plasma_N_phi} <= eq.N_grid = {eq.N_grid}. "
                "An interpolation truncation warning may appear."
            )
        self._plasma_M_theta = plasma_M_theta
        self._plasma_N_phi = plasma_N_phi
        self._constants = {}
        # B_normal and G source grids
        if source_grid is None:
            self._constants["source_grid"] = LinearGrid(
                M=eq.M_grid,
                N=eq.N_grid,
                # for axisymmetry we still need to know about toroidal effects, so its
                # cheapest to pretend there are extra field periods
                NFP=eq.NFP if eq.N > 0 else 64,
                sym=False,
            )
        else:
            self._constants["source_grid"] = source_grid
        self._constants["field_grid"] = field_grid
        # These are differentiable quantities that are not equilibrium-dependent.
        # They can be user-provided, but they also all have default values, so
        # we set them here. This is necessary because we are calling quadcoil through
        # quadcoil.io.quadcoil_for_diff, which cannot see their default value in
        # quadcoil.quadcoil.

        # ----- Calculating DESC-derived, non-differentiable attrs -----
        # eval_grid is used to generate quadrature points.
        # It is the same as "eval_grid" in desc.integrals.compute_B_plasma
        # it is also used to calculate surface Bnormal_plasma
        # when vacuum=False, along with surface_grid.
        # because we the quadrature points must be calculated before generating
        # quadcoil callable, it will be constructed here, instead of in the build().
        eval_grid = LinearGrid(
            NFP=eq.NFP,
            # If we set this to sym it will only evaluate
            # theta from 0 to pi.
            sym=False,
            M=self._plasma_M_theta,  # Poloidal grid resolution.
            N=self._plasma_N_phi,
            rho=1.0,
        )
        eval_data_keys = []
        if self._field:
            eval_data_keys = eval_data_keys + _BCOIL_DATA_KEYS
        if not self._vacuum:
            eval_data_keys = eval_data_keys + _BPLASMA_DATA_KEYS
        eval_data_keys = list(
            dict.fromkeys(eval_data_keys + list(self._extra_eval_data_keys))
        )
        eval_profiles = get_profiles(eval_data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(eval_data_keys, obj=eq, grid=eval_grid)
        self._constants["eval_grid"] = eval_grid
        self._constants["eval_profiles"] = eval_profiles
        self._constants["eval_transforms"] = eval_transforms
        self.nfp = eq.NFP
        self.stellsym = eq.sym
        quadcoil_kwargs["metric_name"] = metric_name
        quadcoil_kwargs["nfp"] = eq.NFP
        quadcoil_kwargs["stellsym"] = eq.sym
        quadcoil_kwargs["plasma_mpol"] = eq.surface.M
        quadcoil_kwargs["plasma_ntor"] = eq.surface.N
        quadcoil_kwargs["plasma_quadpoints_phi"] = (
            eval_grid.nodes[eval_grid.unique_zeta_idx, 2] / jnp.pi / 2
        )
        quadcoil_kwargs["plasma_quadpoints_theta"] = (
            eval_grid.nodes[eval_grid.unique_theta_idx, 1] / jnp.pi / 2
        )
        self._Bnormal_shape = (
            len(quadcoil_kwargs["plasma_quadpoints_phi"]),
            len(quadcoil_kwargs["plasma_quadpoints_theta"]),
        )
        # ----- Generating quadcoil partial and its jvp rule -----
        # quadcoil_kwargs is a mixture of static and traced arguments.
        # Because we likely will not adjust quadcoil settings dynamically,
        # here we treat all of them like staic using
        # partial(quadcoil, **quadcoil_kwargs), implemented in gen_quadcoil_for_diff.
        # The function also generates the custom_jvp rule based on the static arguments.
        # We store the resulting function as a static attribute.
        # Forward DESC's eval-point Biot-Savart chunk size into Quadcoil so the
        # winding-surface / self-field kernels use the same knob.
        if bs_chunk_size is not None:
            quadcoil_kwargs["bs_chunk_size"] = bs_chunk_size
        _quadcoil_values, _quadcoil_for_diff = gen_quadcoil_for_diff(**quadcoil_kwargs)
        # Used later for Bnormal_plasma also
        self._quadcoil_for_diff = jit(_quadcoil_for_diff)
        self._quadcoil_values = jit(_quadcoil_values)

        # ----- Superclass -----
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def _build_quadcoil_constants(self, verbose=1):
        """Precompute ptolemy maps, transforms, and eq_fixed quantities.

        Shared by ``QuadcoilProxy.build`` and subclasses that set their own
        ``_dim_f`` / ``quad_weights`` before calling ``_Objective.build``.
        """
        eq = self._eq
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        # ----- Building the desc surf -> quadcoil (simsopt) surf map -----
        (
            self._surf_R_A,
            self._surf_R_c_indices,
            self._surf_R_s_indices,
        ) = _ptolemy_identity_rev_precompute(
            eq.surface.R_basis.modes[:, 1], eq.surface.R_basis.modes[:, 2]
        )
        (
            self._surf_Z_A,
            self._surf_Z_c_indices,
            self._surf_Z_s_indices,
        ) = _ptolemy_identity_rev_precompute(
            eq.surface.Z_basis.modes[:, 1], eq.surface.Z_basis.modes[:, 2]
        )

        # ----- Building grids and transforms -----
        # source_grid for Bnormal_plasma, and eval_grid.
        # Eval grid has a special role, in that it helps
        # generate plasma_quadpoint_phi and theta. Therefore,
        # it will be generated in init instead.
        if self._enable_net_current_plasma:
            net_poloidal_current_profiles = get_profiles(
                ["G"], obj=eq, grid=self._constants["source_grid"]
            )
            net_poloidal_current_transforms = get_transforms(
                ["G"], obj=eq, grid=self._constants["source_grid"]
            )
            # Storing transforms
            # Attributes inside and outside _constants are not really treated
            # differently, except that self._constants is traced. Because
            # quadcoil_arg is a mixture of traced and static inputs, we want to
            # individually register all the static inputs. Moreover, dicts are
            # not hashable, so the static arguments in quadcoil_kwargs must all
            # be stored as individual attributes. We might as well store
            # everything in quadcoil_kwargs as individual attributes,
            # and only store the transforms and profiles here in self._constants.
            self._constants["net_poloidal_current_profiles"] = (
                net_poloidal_current_profiles
            )
            self._constants["net_poloidal_current_transforms"] = (
                net_poloidal_current_transforms
            )

        # Mose DESC objectives are fields, so they
        # hard-coded the superclass to ask for a weight
        # to integrate the field over a quadrature...

        # source_grid will only be generated when self.vacuum == False.
        # Here, eval_grid is not only used to define Bnormal_plasma,
        # but also used to generate plasma_quadpoints_phi and theta.
        # Therefore, it will be greated regardless self.vacuum == True.
        if not self._vacuum:
            (
                source_profiles,
                source_transforms,
                interpolator,
            ) = _create_source(
                eq=eq,
                source_grid=self._constants["source_grid"],
                eval_grid=self._constants["eval_grid"],
            )
            self._constants["source_profiles"] = source_profiles
            self._constants["source_transforms"] = source_transforms
            self._constants["interpolator"] = interpolator

        if self._field:
            from desc.magnetic_fields import SumMagneticField

            self._constants["sum_field"] = SumMagneticField(self._field)

        # ----- Precomputing quantities -----

        # Now that all transforms are calculated, time to
        # precompute quantities where applicable.
        if self._eq_fixed:
            # Plasma dofs
            self._constants["plasma_dofs"] = self.compute_plasma_surface_dofs_simsopt(
                eq.params_dict
            )

            # Net plasma current
            if self._enable_net_current_plasma:
                self._constants["G"] = _compute_G(eq.params_dict, self._constants)

            # B plasma
            if not self._vacuum:
                self._constants["Bnormal_plasma"] = _compute_Bnormal_plasma(
                    self._constants, eq.params_dict, self._bplasma_chunk_size
                )

            # Part of external field
            if self._field:
                coils_x, coils_n_rho = _compute_eval_data_coils(
                    self._constants, eq.params_dict
                )
                self._constants["coils_x"] = coils_x
                self._constants["coils_n_rho"] = coils_n_rho
                if self._field_fixed:
                    self._constants["Bnormal_ext"] = _compute_Bnormal_ext(
                        self._constants,
                        self._constants["sum_field"].params_dict,
                        self._bs_chunk_size,
                    )

        # ----- Wrapping up and timing -----
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        # dim_f = size of the output vector returned by self.compute.
        # This is a scalar objective.
        self._dim_f = 1
        self._build_quadcoil_constants(verbose=verbose)

        # ----- Normalization scales -----
        # We try to normalize metrics to order(1) by dividing things by some
        # characteristic scale for a given quantity.
        # See ``desc.objectives.compute_scaling_factors`` for examples.
        # The unit for each quantity is implemented as the attribute ``desc_unit``
        # of the corresponding function. These attributes are lambda functions
        # that act on self.scales and returns a number. Example:
        # K.desc_unit = lambda scales: scales["B"] / mu_0 # noqa: E800
        # NOTE: the units of objectives and constraints still needs to be
        # provided in quadcoil_kwargs.
        if self._normalize:
            self.scales = compute_scaling_factors(self._eq)

        # ----- Fixing a key error -----
        # The QUADCOIL has np coordinates. To prevent a key error when DESC
        # tries to execute `grid = self._constants["transforms"]["grid"]`
        # during build() and cause a key error, we assign some dummy weights.
        # This prevents the whole
        # `if hasattr(self, "_constants") and ("quad_weights" not in self._constants):`
        # from triggering.
        self._constants["quad_weights"] = jnp.ones(self._dim_f)

        # finally, call ``super.build()``
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, *all_params, constants=None):
        """Computes the scalar value of the QUADCOIL proxy.

        Computes the scalar value of the QUADCOIL proxy. A wrapper for
        ``solve_quadcoil``.

        Parameters
        ----------
        *all_params : dict
            Dictionaries of equilibrium/coils degrees of freedom, depending on
            ``eq_fixed`` and ``field_fixed``.
        constants : dict
            (Dummy for now) Dictionary of constant data, eg transforms,
            profiles etc. Defaults to self.constants

        Returns
        -------
        The scalar quadcoil proxy.

        """
        _ = self._get_deprecated_constants(constants)
        # We prohibit the user from providing constants
        return self.solve_quadcoil(*all_params, full_mode=False)

    def _quadcoil_inputs(self, *all_params):
        """Resolve eq/field params, plasma dofs, and net poloidal current G.

        Returns
        -------
        params_eq : dict
        params_field : dict or tuple
        plasma_dofs : ndarray
        net_poloidal_current_amperes : float
        """
        constants = self._constants

        if self._eq_fixed:
            params_eq = self._eq.params_dict
            params_field = all_params
        else:
            params_eq = all_params[0]
            if self._field:
                if self._field_fixed:
                    params_field = constants["sum_field"].params_dict
                else:
                    params_field = all_params[1:]
            else:
                params_field = {}

        if self._eq_fixed:
            plasma_dofs = constants["plasma_dofs"]
        else:
            plasma_dofs = self.compute_plasma_surface_dofs_simsopt(params_eq)

        if self._enable_net_current_plasma:
            if self._eq_fixed:
                net_poloidal_current_amperes = constants["G"]
            else:
                net_poloidal_current_amperes = _compute_G(params_eq, constants)
        else:
            net_poloidal_current_amperes = 0.0

        return params_eq, params_field, plasma_dofs, net_poloidal_current_amperes

    def solve_quadcoil(self, *all_params, full_mode=True):
        """Calls QUADCOIL.

        Takes the same parameters as compute, but can either output the
        full quadcoil results, or do what compute() is supposed to do.
        compute() is a wrapper for solve_quadcoil.

        Parameters
        ----------
        *all_params : dict
            Dictionaries of equilibrium/coils degrees of freedom, depending on
            ``eq_fixed`` and ``field_fixed``.
        constants : dict
            Dictionary of constant data, eg transforms,
            profiles etc. Defaults to self.constants
        full_mode : bool
            When ``True``, returns the QUADCOIL standard outputs (see the
            [QUADCOIL documentation](
            https://quadcoil.readthedocs.io/en/latest/tutorial_outputs.html)
            ). When ``False``, returns the scalar QUADCOIL proxy.

        Returns
        -------
        f : scalar

        """
        # Importing QUADCOIL
        try:
            from quadcoil import get_quantity
        except ModuleNotFoundError:
            raise ModuleNotFoundError("QuadcoilProxy requires a QUADCOIL installation.")

        params_eq, params_field, plasma_dofs, net_poloidal_current_amperes = (
            self._quadcoil_inputs(*all_params)
        )

        Bnormal = _compute_Bnormal(
            field=self._field,
            constants=self._constants,
            Bnormal_shape=self._Bnormal_shape,
            vacuum=self._vacuum,
            eq_fixed=self._eq_fixed,
            field_fixed=self._field_fixed,
            params_eq=params_eq,
            params_field=params_field,
            bs_chunk_size=self._bs_chunk_size,
            bplasma_chunk_size=self._bplasma_chunk_size,
        )

        # ----- Calling the quadcoil wrapper with custom_vjp -----
        if full_mode:
            out_dict, qp, cp_mn, solve_results = self._quadcoil_values(
                plasma_dofs=plasma_dofs,
                net_poloidal_current_amperes=net_poloidal_current_amperes,
                net_toroidal_current_amperes=self.net_toroidal_current_amperes,
                Bnormal_plasma=Bnormal,  # Because DESC plasma surface is flipped.
                plasma_coil_distance=self.plasma_coil_distance,
                winding_dofs=self.winding_dofs,
                objective_weight=self.objective_weight,
                constraint_value=self.constraint_value,
            )
            return out_dict, qp, cp_mn, solve_results
        # ----- Calling the quadcoil wrapper with custom_vjp -----
        # If this can't show then the error is before this
        metric_dict = self._quadcoil_for_diff(
            plasma_dofs=plasma_dofs,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=self.net_toroidal_current_amperes,
            Bnormal_plasma=Bnormal,
            plasma_coil_distance=self.plasma_coil_distance,
            winding_dofs=self.winding_dofs,
            objective_weight=self.objective_weight,
            constraint_value=self.constraint_value,
        )

        # ----- Thresholding and weighing -----
        f_out = 0.0
        for i in range(len(self.metric_name)):
            # Set during the loop through quadcoil_kwargs
            f_name = self.metric_name[i]
            f_weight = self.metric_weight[i]
            f_target_eff = self.metric_target[i]
            f_val_eff = metric_dict[f_name]
            # MEY NOT BE LOWERABLE?
            if self._normalize or self._normalize_target:
                f_unit = get_quantity(f_name + "_desc_unit")(self.scales)
            if self._normalize:
                f_val_eff = f_val_eff / f_unit
            if self._normalize_target:
                f_target_eff = f_target_eff / f_unit
            f_out = f_out + f_weight * jnp.where(
                f_val_eff > f_target_eff, f_val_eff - f_target_eff, 0.0
            )

        return f_out

    def compute_plasma_surface_dofs_simsopt(self, params_eq):
        """Computes the plasma surface dofs in the Simsopt convention.

        Computes the plasma surface dofs in the Simsopt convention.

        Parameters
        ----------
        *all_params : dict
            Dictionaries of equilibrium/coils degrees of freedom, depending on
            ``eq_fixed`` and ``field_fixed``.

        Returns
        -------
        plasma_dofs : ndarray
            The plasma surface dofs in the Simsopt SurfaceRZFourier convention.
        """
        rs_raw, rc_raw = _ptolemy_identity_rev_compute(
            self._surf_R_A,
            self._surf_R_c_indices,
            self._surf_R_s_indices,
            params_eq["Rb_lmn"],
        )
        zs_raw, zc_raw = _ptolemy_identity_rev_compute(
            self._surf_Z_A,
            self._surf_Z_c_indices,
            self._surf_Z_s_indices,
            params_eq["Zb_lmn"],
        )
        # Stellsym SurfaceRZFourier dofs consists of
        # [rc, zs] # noqa: E800
        # Non-stellsym SurfaceRZFourier dofs consists of
        # [rc, rs, zc, zs] # noqa: E800
        # Because rs, zs from ptolemy_identity_rev shares the same m, n
        # arrays as rc, zc, they both have a zero as the first element
        # that need to be removed.
        rc = rc_raw.flatten()
        rs = rs_raw.flatten()[1:]
        zc = zc_raw.flatten()
        zs = zs_raw.flatten()[1:]
        if self.stellsym:
            plasma_dofs = jnp.concatenate([rc, zs])
        else:
            plasma_dofs = jnp.concatenate([rc, rs, zc, zs])
        return plasma_dofs

    def _build_field_from_quadcoil(self, quadcoil_qp, quadcoil_dofs):
        """Build a ``FourierCurrentPotentialField`` from a QUADCOIL solution."""
        # Prevents circular import
        from desc.magnetic_fields import FourierCurrentPotentialField

        quadcoil_kwargs_temp = {
            "winding_stellsym": quadcoil_qp.winding_surface.stellsym,
            "winding_mpol": quadcoil_qp.winding_surface.mpol,
            "winding_ntor": quadcoil_qp.winding_surface.ntor,
            "stellsym": quadcoil_qp.stellsym,
            "mpol": quadcoil_qp.mpol,
            "ntor": quadcoil_qp.ntor,
            "net_poloidal_current_amperes": quadcoil_qp.net_poloidal_current_amperes,
            "net_toroidal_current_amperes": quadcoil_qp.net_toroidal_current_amperes,
        }
        # This helper function converts information in a quadcoil object into
        # a kwargs for DESC FourierCurrentPotentialField.
        filtered = _quadcoil_kwargs_to_field_kwargs(
            quadcoil_kwargs_temp,
            quadcoil_dofs,
            self._eq.sym,
            FourierCurrentPotentialField,
            self._verbose,
        )
        winding_surface = quadcoil_qp.winding_surface.to_desc()
        R_lmn = winding_surface.R_lmn
        Z_lmn = winding_surface.Z_lmn
        modes_R = winding_surface._R_basis.modes[:, 1:]
        modes_Z = winding_surface._Z_basis.modes[:, 1:]
        return FourierCurrentPotentialField(
            # Phi_mn is already in filtered
            # modes_Phi is already in filtered
            # I is already in filtered
            # G is already in filtered
            # sym_Phi is already in filtered
            # M_Phi is already in filtered
            # N_Phi is already in filtered
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            NFP=self.nfp,
            sym=self._eq.sym,  # Symmetry of the plasma
            # M is already in filtered
            # N is already in filtered
            name="QUADCOIL Proxy Output",
            check_orientation=True,
            **filtered,
        )

    def solve_quadcoil_surface_current(self, *all_params):
        """Calls QUADCOIL and returns the solution as a FourierCurrentPotentialField.

        Calls QUADCOIL and returns the solution as a FourierCurrentPotentialField.
        For use with DESC's built-in REGCOIL and coil-cutting features.

        Parameters
        ----------
        params_eq : dict
            Dictionary of equilibrium degrees of freedom, eg
            Equilibrium.params_dict.

        Returns
        -------
        A FourierCurrentPotentialField containing the QUADCOIL solution.
        """
        _, quadcoil_qp, quadcoil_dofs, _ = self.solve_quadcoil(
            *all_params, full_mode=True
        )
        return self._build_field_from_quadcoil(quadcoil_qp, quadcoil_dofs)


class QuadcoilFreeBoundaryError(QuadcoilProxy):
    """Free-boundary residual with external field from a QUADCOIL solve.

    Solves QUADCOIL for the differentiable current potential ``phi_dofs`` and
    evaluates the same residual blocks as :class:`BoundaryError`, with the
    winding-surface field playing the role of the external coil field.

    The winding-surface geometry is frozen at ``build`` time; only ``Phi(eq)``
    and the evaluation geometry carry equilibrium gradients.
    """

    _static_attrs = QuadcoilProxy._static_attrs + [
        "_eq_data_keys",
        "_sheet_current",
        "_sheet_data_keys",
    ]

    _scalar = False
    _coordinates = "rtz"
    _units = "(T*m^2, T^2*m^2, T*m^2)"
    _print_value_fmt = "Quadcoil boundary error: "
    _extra_eval_data_keys = _BPLASMA_DATA_KEYS + ["p"]

    def __init__(
        self,
        eq,
        quadcoil_kwargs,
        plasma_M_theta=None,
        plasma_N_phi=None,
        target=None,
        bounds=None,
        weight=1,
        vacuum=False,
        normalize=True,
        normalize_target=True,
        verbose=0,
        name="Quadcoil free boundary error",
        source_grid=None,
        field=None,
        field_grid=None,
        winding_grid=None,
        enable_net_current_plasma=True,
        eq_fixed=False,
        field_fixed=False,
        s=None,
        q=None,
        B_plasma_chunk_size=None,
        jac_chunk_size=None,
        bs_chunk_size=None,
    ):
        errorif(
            eq_fixed,
            ValueError,
            "QuadcoilFreeBoundaryError does not support eq_fixed=True "
            "(the residual would be constant).",
        )
        obj_name = quadcoil_kwargs.get("objective_name")
        if isinstance(obj_name, str):
            has_f_B = "f_B" in obj_name
        elif isinstance(obj_name, (tuple, list)):
            has_f_B = any("f_B" in name for name in obj_name)
        else:
            has_f_B = False
        if not has_f_B:
            warnings.warn(
                "objective_name does not contain f_B. This is discouraged for "
                "free boundary quasi-single-stage optimization. "
                "QuadcoilFreeBoundary minimizes the free boundary error using "
                "external magnetic fields generated a QUADCOIL solve (and some "
                "other fields, if provided). If the QUADCOIL objective does not "
                "contain 'f_B', then QUADCOIL may not push f_B low enough for "
                "free boundary error to also be low."
            )
        self._sheet_current = hasattr(eq.surface, "Phi_mn")
        self._st, self._sz = s if isinstance(s, (tuple, list)) else (s, s)
        self._q = q
        super().__init__(
            eq=eq,
            quadcoil_kwargs=quadcoil_kwargs,
            plasma_M_theta=plasma_M_theta,
            plasma_N_phi=plasma_N_phi,
            target=target,
            bounds=bounds,
            weight=weight,
            metric_name="phi_dofs",
            metric_weight=1.0,
            metric_target=0.0,
            vacuum=vacuum,
            normalize=normalize,
            normalize_target=normalize_target,
            verbose=verbose,
            name=name,
            source_grid=source_grid,
            field=field,
            field_grid=field_grid,
            enable_net_current_plasma=enable_net_current_plasma,
            eq_fixed=eq_fixed,
            field_fixed=field_fixed,
            B_plasma_chunk_size=B_plasma_chunk_size,
            jac_chunk_size=jac_chunk_size,
            bs_chunk_size=bs_chunk_size,
        )
        self._constants["winding_grid"] = winding_grid
        self._eq_data_keys = list(self._extra_eval_data_keys)

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self._eq
        self._build_quadcoil_constants(verbose=verbose)

        # Source transforms from _create_source only cover _BPLASMA_DATA_KEYS;
        # rebuild with the full free-boundary keys (includes "p") when needed.
        if not self._vacuum:
            self._constants["source_profiles"] = get_profiles(
                self._eq_data_keys, obj=eq, grid=self._constants["source_grid"]
            )
            self._constants["source_transforms"] = get_transforms(
                self._eq_data_keys, obj=eq, grid=self._constants["source_grid"]
            )

        # Freeze winding geometry at a concrete QUADCOIL solution.
        _, qp, dofs, _ = self.solve_quadcoil(eq.params_dict, full_mode=True)
        # Building a static FourierCurrentPotentialField from the QUADCOIL solution
        # that serves to store its static attributes.
        field = self._build_field_from_quadcoil(qp, dofs)
        phi = np.asarray(dofs["phi"])
        # Calculating static operator that converts a quadcoil current potential
        # to a DESC current potential.
        # Works like this: Phi_mn_gather = np.sum(coef * phi[idx], axis=1)
        idx, coef = _quadcoil_phi_to_desc_phi_gather(
            len(phi), qp.stellsym, qp.mpol, qp.ntor, field.Phi_basis
        )
        self._constants["winding_field"] = field
        self._constants["phi_gather_idx"] = idx
        self._constants["phi_gather_coef"] = coef

        if self._sheet_current:
            self._sheet_data_keys = ["K"]
            self._constants["sheet_source_transforms"] = get_transforms(
                self._sheet_data_keys,
                obj=eq.surface,
                grid=self._constants["source_grid"],
            )
            self._constants["sheet_eval_transforms"] = get_transforms(
                self._sheet_data_keys,
                obj=eq.surface,
                grid=self._constants["eval_grid"],
            )

        eval_grid = self._constants["eval_grid"]
        neq = 3 if self._sheet_current else 2
        self._dim_f = neq * eval_grid.num_nodes
        self._constants["quad_weights"] = np.sqrt(np.tile(eval_grid.weights, neq))

        if self._normalize:
            scales = compute_scaling_factors(eq)
            Bn_norm = (
                np.ones(eval_grid.num_nodes) * scales["B"] * scales["R0"] * scales["a"]
            )
            B2_norm = (
                np.ones(eval_grid.num_nodes)
                * scales["B"] ** 2
                * scales["R0"]
                * scales["a"]
            )
            self._normalization = np.concatenate([Bn_norm, B2_norm])
            if self._sheet_current:
                self._normalization = np.concatenate([self._normalization, Bn_norm])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, *all_params, constants=None):
        """Compute free-boundary residual with QUADCOIL external field.

        Parameters
        ----------
        *all_params : dict
            Dictionaries of equilibrium/coils degrees of freedom.
        constants : dict
            Deprecated; ignored. Constants are taken from ``self._constants``.

        Returns
        -------
        f : ndarray
            Boundary error residuals. First block is √g B·n, second is
            √g[[B² + 2μ₀p]], and (if sheet current) third is
            √g||μ₀K − n × [B]||.
        """
        _ = self._get_deprecated_constants(constants)
        constants = self._constants

        params_eq, params_field, plasma_dofs, G = self._quadcoil_inputs(*all_params)

        sheet_source_data = None
        sheet_eval_data = None
        if self._sheet_current:
            sheet_params = {
                "R_lmn": params_eq["Rb_lmn"],
                "Z_lmn": params_eq["Zb_lmn"],
                "I": params_eq["I"],
                "G": params_eq["G"],
                "Phi_mn": params_eq["Phi_mn"],
            }
            sheet_source_data = compute_fun(
                self._eq.surface,
                self._sheet_data_keys,
                params=sheet_params,
                transforms=constants["sheet_source_transforms"],
                profiles={},
            )
            sheet_eval_data = compute_fun(
                self._eq.surface,
                self._sheet_data_keys,
                params=sheet_params,
                transforms=constants["sheet_eval_transforms"],
                profiles={},
            )

        if self._vacuum:
            eval_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                self._eq_data_keys,
                params=params_eq,
                transforms=constants["eval_transforms"],
                profiles=constants["eval_profiles"],
            )
            Bplasma = 0.0
        else:
            K_sheet = sheet_source_data["K"] if sheet_source_data is not None else None
            Bplasma, eval_data = _compute_Bplasma(
                constants,
                params_eq,
                self._bplasma_chunk_size,
                self._eq_data_keys,
                K_sheet=K_sheet,
            )

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T
        Bnormal = jnp.sum(Bplasma * eval_data["n_rho"], axis=-1).reshape(
            self._Bnormal_shape
        )
        if self._field:
            constants["coils_x"] = x
            constants["coils_n_rho"] = eval_data["n_rho"]
            Bnormal = Bnormal + _compute_Bnormal_ext(
                constants, params_field, self._bs_chunk_size
            ).reshape(self._Bnormal_shape)

        phi = self._quadcoil_for_diff(
            plasma_dofs=plasma_dofs,
            net_poloidal_current_amperes=G,
            net_toroidal_current_amperes=self.net_toroidal_current_amperes,
            Bnormal_plasma=Bnormal,
            plasma_coil_distance=self.plasma_coil_distance,
            winding_dofs=self.winding_dofs,
            objective_weight=self.objective_weight,
            constraint_value=self.constraint_value,
        )["phi_dofs"]

        Phi_mn = jnp.sum(
            constants["phi_gather_coef"] * phi[constants["phi_gather_idx"]],
            axis=1,
        )
        winding_field = constants["winding_field"]
        Bext = winding_field.compute_magnetic_field(
            x,
            params={
                **winding_field.params_dict,
                "Phi_mn": Phi_mn,
                "G": G,
                "I": self.net_toroidal_current_amperes,
            },
            basis="rpz",
            source_grid=constants["winding_grid"],
            chunk_size=self._bs_chunk_size,
        )
        if self._field:
            Bext = Bext + constants["sum_field"].compute_magnetic_field(
                x,
                source_grid=constants["field_grid"],
                basis="rpz",
                params=params_field,
                chunk_size=self._bs_chunk_size,
            )

        Bex_total = Bext + Bplasma
        Bin_total = eval_data["B"]
        Bn = jnp.sum(Bex_total * eval_data["n_rho"], axis=-1)

        bsq_out = jnp.sum(Bex_total * Bex_total, axis=-1)
        bsq_in = jnp.sum(Bin_total * Bin_total, axis=-1)

        g = eval_data["|e_theta x e_zeta|"]
        Bn_err = Bn * g
        Bsq_err = jnp.where(
            eval_data["p"] == 0,
            (bsq_in - bsq_out) * g,
            (bsq_in - bsq_out + eval_data["p"] * 2 * mu_0) * g,
        )
        Bjump = Bex_total - Bin_total
        if self._sheet_current:
            Kerr = mu_0 * sheet_eval_data["K"] - jnp.cross(eval_data["n_rho"], Bjump)
            Kerr = jnp.linalg.norm(Kerr, axis=-1) * g
            return jnp.concatenate([Bn_err, Bsq_err, Kerr])
        return jnp.concatenate([Bn_err, Bsq_err])
