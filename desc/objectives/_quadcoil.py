from desc.utils import Timer
from desc.grid import LinearGrid
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.objectives.normalization import compute_scaling_factors
from desc.compute import get_profiles, get_transforms
from desc.magnetic_fields import FourierCurrentPotentialField
from jax import jit
from desc.backend import jnp
import jax  # for printing
import warnings
from quadcoil import get_quantity
from quadcoil.io import gen_quadcoil_for_diff, generate_desc_scaling
from ._quadcoil_utils import (
    _BCOIL_DATA_KEYS,
    _BPLASMA_DATA_KEYS,
    _quadcoil_kwargs_to_field_kwargs,
    _ptolemy_identity_rev_precompute,
    _ptolemy_identity_rev_compute,
    _compute_G,
    _compute_Bnormal_plasma,
    _compute_eval_data_coils,
    _compute_Bnormal_ext,
    _compute_Bnormal,
    _create_source,
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
    "verbose",
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


class QuadcoilProxy(_Objective):
    """
    A QUADCOIL-based coil complexity proxy.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    quadcoil_kwargs : dict
        (Mixed, automatically handled) A dictionary containing all inputs for ``quadcoil.quadcoil``,
        except some that can be extracted from the equilibrium.
    metric_target : dict
        (Traced) Targets for each objectives. Keys must contain all strings in quadcoil_kwargs['metric_name']
    metric_weight : dict
        (Traced) Weights for each objectives.
    plasma_M_theta : int
        (static) The plasma poloidal quadrature resolution.
        Unlike the winding surface quadrature points, which is a required input, the plasma surface quadpoints
        is evaluated from a linear grid to make sure that the grid points in DESC B calculations line up exactly
        with
    plasma_N_phi : int
        (static) The plasma toroidal quadrature resolution.
    enable_Bnormal_plasma : bool, optional, default=False
        (Traced) Whether to enable Bnormal contributions from plasma current.
    Bnormal_plasma_chunk_size
        (static)
    source_grid

    Attributes
    ----------
    metric_target : tuple
        (Traced) targets for each objectives.
    metric_weight : tuple
        (Traced) weights for each objectives.
    _plasma_M_theta : int
        (static) The plasma poloidal quadrature resolution.
    _plasma_N_phi : int
        (static) The plasma toroidal quadrature resolution.
    _static_attrs : list
        (Static) : a list of all static variables. Generated based on
        ``quadcoil.QUADCOIL_STATIC_ARGNAMES`` and ``quadcoil_kwargs.keys()``.
        For use in the superclass.
    """

    # Most of the documentation is shared among all objectives, so we just inherit
    # the docstring from the base class and add a few details specific to this objective.
    # See the documentation of `collect_docs` for more details.
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = ""  # What coordinates is this objective a function of, with r=rho, t=theta, z=zeta?
    # i.e. if only a profile, it is "r" , while if all 3 coordinates it is "rtz"
    _units = "N/A"  # units of the output
    _print_value_fmt = "QUADCOIL subproblem: "  # string with python string formatting for printing the value

    def __init__(
        self,
        eq,
        quadcoil_kwargs,
        metric_name,
        metric_target,
        metric_weight,
        plasma_M_theta: int,
        plasma_N_phi: int,
        enable_Bnormal_plasma: bool = False,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        verbose=0,
        name="QUADCOIL Proxy",
        Bnormal_plasma_chunk_size=None,
        source_grid=None,
        # External coils - no external coils by default
        field=[],
        field_grid=None,
        enable_net_current_plasma=True,
        eq_fixed=False,  # Whether the equilibrium are fixed
        field_fixed=True,  # Whether the external fields are fixed
        # misc
        jac_chunk_size=None,
        bs_chunk_size=None,
    ):
        self._eq = eq
        if field:  # To be also tolerant on `False` and `None` as an input
            self._field = [field] if not isinstance(field, list) else field
        else:
            self._field = []
        # Coil and things initialization
        self._enable_net_current_plasma = enable_net_current_plasma
        self._eq_fixed = eq_fixed
        self._field_fixed = field_fixed
        things = []
        if not (eq_fixed or field_fixed):
            warnings.warn(
                "Both eq_fixed and field_fixed are True. things will be empty."
            )
        if not eq_fixed:
            things += [eq]
        if not field_fixed:
            if not field:
                raise AttributeError(
                    "When field_fixed is False, field must be an object or a non-empty list."
                )
            things += [field]

        if not (enable_net_current_plasma or field):
            warnings.warn(
                "enable_net_current_plasma is false and field is empty. The problem may be trivial."
            )

        if enable_net_current_plasma and field:
            warnings.warn(
                "There are both external coils and net current. "
                "This is very uncommon (windowpane filaments + winding surface with net current)."
            )

        # quadcoil_kwargs_bak = quadcoil_kwargs.copy()
        quadcoil_kwargs = quadcoil_kwargs.copy()
        if target is None and bounds is None:
            target = 0  # default target value

        # ----- Checking inputs -----
        # Checking whether all metrics have a weight and a target provided.
        if isinstance(metric_name, str):
            if metric_target is None:
                metric_target = 0.0
            if metric_weight is None:
                metric_weight = 1.0
            if (not jnp.isscalar(metric_target)) or (not jnp.isscalar(metric_weight)):
                raise ValueError(
                    "When metric_name is str, metric_target and metric_target must both be scalar."
                )
            # Makign them into iterables will make things easier when scaling in the end.
            metric_name = (metric_name,)
            metric_weight = jnp.array(
                [
                    metric_weight,
                ]
            )
            metric_target = jnp.array(
                [
                    metric_target,
                ]
            )
        elif isinstance(metric_name, tuple):
            if len(metric_target) != len(metric_name):
                raise KeyError(
                    "metric_name and metric_target have mismatching lengths!."
                )
            if len(metric_weight) != len(metric_name):
                raise KeyError(
                    "metric_name and metric_weight have mismatching lengths!."
                )
        else:
            raise ValueError("When metric_name must be a tuple or a str.")
        # Detect if the user has provided any arguments
        # that will also-be extracted from DESC.
        # If there are, these objectives will be discarded.
        if normalize:
            # When normalize is set to true, we
            # use quantities from DESC to perform
            # normalization instead.
            overridden_argnames = _DESC_DERIVED_ARGNAMES + [
                "objective_unit",
                "constraint_unit",
            ]
        else:
            overridden_argnames = _DESC_DERIVED_ARGNAMES
        redundant_arg_names = set(overridden_argnames) & quadcoil_kwargs.keys()
        if redundant_arg_names:
            warnings.warn(
                "Redundant arguments detected: "
                + str(redundant_arg_names)
                + ". These arguments are extracted from the equilibrium, "
                "or specified by other parameters. The provided values "
                "will be discarded."
            )

        # ----- Storing equilibrium-independent, differentiable variables -----
        # These are differentiable quantities that are not equilibrium-dependent.
        # They can be user-provided, but they also all have default values, so
        # we set them here. This is necessary because we're calling quadcoil through
        # quadcoil.io.quadcoil_for_diff, which cannot see their default value in
        # quadcoil.quadcoil.
        if "net_toroidal_current_amperes" in quadcoil_kwargs.keys():
            self.net_toroidal_current_amperes = quadcoil_kwargs.pop(
                "net_toroidal_current_amperes"
            )
        else:
            self.net_toroidal_current_amperes = 0.0
        if "plasma_coil_distance" in quadcoil_kwargs.keys():
            # A sign flip is necessary here!
            self.plasma_coil_distance = -quadcoil_kwargs.pop("plasma_coil_distance")
        else:
            self.plasma_coil_distance = None
        if "winding_dofs" in quadcoil_kwargs.keys():
            # A sign flip is necessary here!
            self.winding_dofs = quadcoil_kwargs.pop("winding_dofs")
        else:
            self.winding_dofs = None
        if "objective_weight" in quadcoil_kwargs.keys():
            self.objective_weight = quadcoil_kwargs.pop("objective_weight")
        else:
            self.objective_weight = None
        if "constraint_value" in quadcoil_kwargs.keys():
            self.constraint_value = quadcoil_kwargs.pop("constraint_value")
        else:
            self.constraint_value = jnp.array([])

        # ----- Setting attributes -----
        self.metric_name = metric_name
        self.metric_target = metric_target
        self.metric_weight = metric_weight
        self._verbose = verbose
        self._source_grid = source_grid  # B_normal grids
        self._bplasma_chunk_size = Bnormal_plasma_chunk_size
        self._bs_chunk_size = bs_chunk_size
        self._enable_Bnormal_plasma = enable_Bnormal_plasma
        self._plasma_M_theta = plasma_M_theta
        self._plasma_N_phi = plasma_N_phi
        self._constants = {}
        self._constants["field_grid"] = field_grid
        # These are differentiable quantities that are not equilibrium-dependent.
        # They can be user-provided, but they also all have default values, so
        # we set them here. This is necessary because we're calling quadcoil through
        # quadcoil.io.quadcoil_for_diff, which cannot see their default value in
        # quadcoil.quadcoil.

        # ----- Calculating DESC-derived, non-differentiable attrs -----
        # plasma_grid is used to generate quadrature points.
        # it is also used to calculate surface Bnormal_plasma
        # when enable_Bnormal_plasma=True, along with surface_grid.
        # because we the quadrature points must be calculated before generating
        # quadcoil callable, it will be constructed here, instead of in the build().
        plasma_grid = LinearGrid(
            NFP=eq.NFP,
            # If we set this to sym it'll only evaluate
            # theta from 0 to pi.
            sym=False,
            M=self._plasma_M_theta,  # Poloidal grid resolution.
            N=self._plasma_N_phi,
            rho=1.0,
        )
        eval_data_keys = []
        if self._field:
            eval_data_keys = eval_data_keys + _BCOIL_DATA_KEYS
        if self._enable_Bnormal_plasma:
            eval_data_keys = eval_data_keys + _BPLASMA_DATA_KEYS
        eval_profiles = get_profiles(eval_data_keys, obj=eq, grid=plasma_grid)
        eval_transforms = get_transforms(eval_data_keys, obj=eq, grid=plasma_grid)
        self._constants["eval_profiles"] = eval_profiles
        self._constants["eval_transforms"] = eval_transforms
        self._constants["plasma_grid"] = plasma_grid
        self.nfp = eq.NFP
        self.stellsym = eq.sym
        quadcoil_kwargs["metric_name"] = metric_name
        quadcoil_kwargs["nfp"] = eq.NFP
        quadcoil_kwargs["stellsym"] = eq.sym
        quadcoil_kwargs["plasma_mpol"] = eq.surface.M
        quadcoil_kwargs["plasma_ntor"] = eq.surface.N
        quadcoil_kwargs["plasma_quadpoints_phi"] = (
            plasma_grid.nodes[plasma_grid.unique_zeta_idx, 2] / jnp.pi / 2
        )
        quadcoil_kwargs["plasma_quadpoints_theta"] = (
            plasma_grid.nodes[plasma_grid.unique_theta_idx, 1] / jnp.pi / 2
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
        _quadcoil_values, _quadcoil_for_diff = gen_quadcoil_for_diff(**quadcoil_kwargs)
        # Used later for Bnormal_plasma also
        self._quadcoil_for_diff = jit(_quadcoil_for_diff)
        self._quadcoil_values = jit(_quadcoil_values)
        # self._quadcoil_kwargs_bak = quadcoil_kwargs_bak
        # ----- Setting and registering keyword arguments -----
        self._static_attrs = _Objective._static_attrs + [
            # External-coils related
            # '_quadcoil_kwargs_bak'
            "_enable_net_current_plasma",
            "_eq_fixed",
            "_field_fixed",
            "_bs_chunk_size",
            # Basics
            "_static_attrs",
            "_deriv_mode",
            "_verbose",
            # Free-boundary-related
            "_bplasma_chunk_size",
            "_enable_Bnormal_plasma",
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

        # ----- Superclass -----
        super().__init__(
            things=things,  # things is a list of things that will be optimized, in this case just the equilibrium
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
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
        # ----- Starting and timing-----
        # things is the list of things that will be optimized,
        # we assigned things to be just eq in the init, so we know that the
        # first (and only) element of things is the equilibrium
        eq = self._eq
        # dim_f = size of the output vector returned by self.compute.
        # This is a scalar objective.
        self._dim_f = 1
        # some helper code for profiling and logging
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
        # source_grid for Bnormal_plasma, and plasma_grid.
        # Eval grid has a special role, in that it helps
        # generate plasma_quadpoint_phi and theta. Therefore,
        # it will be generated in init instead.
        if self._enable_net_current_plasma:
            net_poloidal_current_grid = LinearGrid(rho=jnp.array(1.0))
            net_poloidal_current_profiles = get_profiles(
                ["G"], obj=eq, grid=net_poloidal_current_grid
            )
            net_poloidal_current_transforms = get_transforms(
                ["G"], obj=eq, grid=net_poloidal_current_grid
            )
            # Storing transforms
            # Attributes inside and outside _constants aren't really treated
            # differently, except that self._constants is traced. Because quadcoil_arg
            # is a mixture of traced and static inputs, we want to individually register
            # all the static inputs. Moreover, dicts are not hashable, so the static
            # arguments in quadcoil_kwargs must all be stored as individual attributes.
            # We might as well store everything in quadcoil_kwargs as individual attributes,\
            # and only store the transforms and profiles here in self._constants.
            self._constants[
                "net_poloidal_current_profiles"
            ] = net_poloidal_current_profiles
            self._constants[
                "net_poloidal_current_transforms"
            ] = net_poloidal_current_transforms

        # Mose DESC objectives are fields, so they
        # hard-coded the superclass to ask for a weight
        # to integrate the field over a quadrature...

        # source_grid will only be generated when self._enable_Bnormal_plasma == True.
        # Here, plasma_grid is not only used to define Bnormal_plasma,
        # but also used to generate plasma_quadpoints_phi and theta.
        # Therefore, it will be greated regardless self.valuum == True.
        if self._enable_Bnormal_plasma:
            if verbose:
                jax.debug.print(
                    "enable_Bnormal_plasma=True, QUADCOIL will no "
                    "longer assume zero Bnormal_plasma at the boundary."
                )
            (source_profiles, source_transforms, interpolator,) = _create_source(
                eq=eq,
                source_grid_in=self._source_grid,
                plasma_grid_in=self._constants["plasma_grid"],
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
            self._constants["plasma_dofs"] = self.compute_plasma_dofs(eq.params_dict)

            # Net plasma current
            if self._enable_net_current_plasma:
                self._constants["G"] = _compute_G(eq.params_dict, self._constants)

            # B plasma
            if self._enable_Bnormal_plasma:
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
                        self._constants["sum_field"].params_dict,  # params_field,
                        self._bs_chunk_size,
                    )

        # ----- Normalization scales -----
        # We try to normalize things to order(1) by dividing things by some
        # characteristic scale for a given quantity.
        # See ``desc.objectives.compute_scaling_factors`` for examples.
        # The unit for each objective is implemented as the attribute ``desc_unit``
        # of the corresponding function. These attributes are lambda functions
        # that act on self.scales and returns a number. Example:
        # K.desc_unit = lambda scales: scales["B"] / mu_0
        if self._normalize:
            self.scales = compute_scaling_factors(eq)
            obj_unit_new, cons_unit_new = generate_desc_scaling(
                self.objective_name, self.constraint_name, self.scales
            )
            self.objective_unit = obj_unit_new
            self.constraint_unit = cons_unit_new

        # ----- Wrapping up and timing -----
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")
        # finally, call ``super.build()``
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, *all_params, constants=None):
        # We prohibit the user from providing constants
        return self.compute_full(*all_params, full_mode=False)

    def compute_full(self, *all_params, full_mode=True):
        """
        Takes the same parameters as compute, but can either output the
        full quadcoil results, or do what compute() is supposed to do.
        compute() will be a wrapper for this.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            (Dummy for now) Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : scalar
        """
        # Loading constants
        constants = self._constants

        # Load fixed params
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

        # ----- Quantities with pre-computation -----

        # Plasma dofs
        if self._eq_fixed:
            plasma_dofs = constants["plasma_dofs"]
        else:
            plasma_dofs = self.compute_plasma_dofs(params_eq)

        Bnormal = _compute_Bnormal(
            field=self._field,
            constants=constants,
            Bnormal_shape=self._Bnormal_shape,
            enable_Bnormal_plasma=self._enable_Bnormal_plasma,
            eq_fixed=self._eq_fixed,
            field_fixed=self._field_fixed,
            params_eq=params_eq,
            params_field=params_field,
            bs_chunk_size=self._bs_chunk_size,
            bplasma_chunk_size=self._bplasma_chunk_size,
        )

        # ----- Calling the net poloidal current -----
        if self._enable_net_current_plasma:
            if self._eq_fixed:
                net_poloidal_current_amperes = constants["G"]
            else:
                net_poloidal_current_amperes = _compute_G(params_eq, constants)
        else:
            net_poloidal_current_amperes = 0.0

        # ----- Calling the quadcoil wrapper with custom_vjp -----
        if full_mode:
            out_dict, qp, cp_mn, solve_results = self._quadcoil_values(
                plasma_dofs=plasma_dofs,
                net_poloidal_current_amperes=net_poloidal_current_amperes,
                net_toroidal_current_amperes=self.net_toroidal_current_amperes,
                Bnormal_plasma=-Bnormal,  # Because DESC plasma surface is flipped.
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

    def compute_plasma_dofs(self, params_eq):
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
        # Stellsym SurfaceRZFourier's dofs consists of
        # [rc, zs]
        # Non-stellsym SurfaceRZFourier's dofs consists of
        # [rc, rs, zc, zs]
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

    def compute_FourierCurrentPotentialField(self, *all_params):
        _, quadcoil_qp, quadcoil_dofs, _ = self.compute_full(
            *all_params, full_mode=True
        )
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
        )
        winding_surface = quadcoil_qp.winding_surface.to_desc()
        R_lmn = winding_surface.R_lmn
        Z_lmn = winding_surface.Z_lmn
        modes_R = winding_surface._R_basis.modes[:, 1:]
        modes_Z = winding_surface._Z_basis.modes[:, 1:]
        return FourierCurrentPotentialField.__init__(
            # Phi_mn=Phi_mn, # already in filtered
            # modes_Phi=modes_Phi, # already in filtered
            # I=I, # already in filtered
            # G=G, # already in filtered
            # sym_Phi=sym_Phi, # already in filtered
            # M_Phi=M_Phi, # already in filtered
            # N_Phi=N_Phi, # already in filtered
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            # NFP=NFP, # already in filtered
            # sym=sym, # already in filtered
            # M=M, # already in filtered
            # N=N, # already in filtered
            name="QUADCOIL Proxy Output",
            check_orientation=True,
            **filtered
        )
