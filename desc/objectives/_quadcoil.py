from desc.utils import Timer, warnif
from desc.vmec_utils import ptolemy_linear_transform
from desc.grid import LinearGrid
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.objectives.normalization import compute_scaling_factors
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.integrals import DFTInterpolator, FFTInterpolator, virtual_casing_biot_savart
from ..integrals.singularities import best_params, best_ratio
from scipy.constants import mu_0
from functools import partial
from jax import jit
import jax.numpy as jnp
import jax # for printing
import warnings
import numpy as np
from quadcoil import QUADCOIL_STATIC_ARGNAMES, get_quantity
from quadcoil.io import gen_quadcoil_for_diff, generate_desc_scaling

# ----- A QUADCOIL wrapper -----
# A list of all inputs of quadoil.quadcoil
# that can be extracted from DESC. The 
# rest cannot. These variables should not 
# show up in quadcoil_args. If they do, they will be ignored.
_DESC_DERIVED_ARGNAMES = [
    'nfp',
    'stellsym',
    'plasma_mpol',
    'plasma_ntor',
    'plasma_quadpoints_phi',
    'plasma_quadpoints_theta',
    'plasma_dofs',
    'net_poloidal_current_amperes',
    'Bnormal_plasma',
    'winding_dofs',
    'metric_name',
    'value_only',
    'verbose',
] 

# A list of argnames that must be user-provided,
# but are considered differentiable by JAX. 
# Variables here will be excluded when constructing 
# nondiff_args, the non-differentiable argument of 
# quadcoil.io.quadcoil_for_diff
_DIFF_USER_ARGNAMES = [
    'net_toroidal_current_amperes'
    'plasma_coil_distance'
    'objective_weight'
    'constraint_value'
]

# Data keys needed to calculate Bnormal_plasma.
_BPLASMA_DATA_KEYS = ["K_vc", "B", "R", "phi", "Z", "e^rho", "n_rho", "|e_theta x e_zeta|"]

class QuadcoilProxy(_Objective):  
    """
    A QUADCOIL-based coil complexity proxy. 

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    quadcoil_args : dict
        (Mixed, automatically handled) A dictionary containing all inputs for ``quadcoil.quadcoil``,
        except some that can be extracted from the equilibrium.
    metric_target : dict
        (Traced) Targets for each objectives. Keys must contain all strings in quadcoil_args['metric_name']
    metric_weight : dict
        (Traced) Weights for each objectives.
    plasma_M_theta : int
        (static) The plasma poloidal quadrature resolution.
    plasma_N_phi : int
        (static) The plasma toroidal quadrature resolution.
    enable_Bnormal_plasma : bool, optional, default=False
        (Traced) By default, we assume :math:`\hat{\mathbf{n}} \cdot \mathbf{B}_{plasma}=0`. This is generally true
        for surfaces bounding a fixed boundary equilibrium. When ``True``, we will no longer 
        assume this for the ``eq.surface``. Here for the sake of generality.
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
        ``quadcoil.QUADCOIL_STATIC_ARGNAMES`` and ``quadcoil_args.keys()``.
        For use in the superclass.
    """
    # Most of the documentation is shared among all objectives, so we just inherit
    # the docstring from the base class and add a few details specific to this objective.
    # See the documentation of `collect_docs` for more details.
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )

    _coordinates = ""    # What coordinates is this objective a function of, with r=rho, t=theta, z=zeta?
                            # i.e. if only a profile, it is "r" , while if all 3 coordinates it is "rtz"
    _units = "N/A"    # units of the output
    _print_value_fmt = "QUADCOIL subproblem: "    # string with python string formatting for printing the value

    def __init__(
        self,
        eq,
        quadcoil_args,
        metric_name,
        metric_target,
        metric_weight,
        plasma_M_theta:int,
        plasma_N_phi:int,
        enable_Bnormal_plasma:bool=False,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        verbose=False,
        name="QUADCOIL Proxy",
        Bnormal_plasma_chunk_size=None,
        source_grid=None,
        jac_chunk_size=None,
    ):
        quadcoil_args = quadcoil_args.copy()
        if target is None and bounds is None:
            target = 0 # default target value
            
        # ----- Checking inputs -----
        # Checking whether all metrics have a weight and a target provided.
        if isinstance(metric_name, str):
            if metric_target is None:
                metric_target = 0.
            if metric_weight is None:
                metric_weight = 1.
            if (not jnp.isscalar(metric_target)) or (not jnp.isscalar(metric_weight)):
                raise ValueError('When metric_name is str, metric_target and metric_target must both be scalar.')
            # Makign them into iterables will make things easier when scaling in the end.
            metric_name = (metric_name,)
            metric_weight = jnp.array([metric_weight,])
            metric_target = jnp.array([metric_target,])
        elif isinstance(metric_name, tuple):
            if len(metric_target) != len(metric_name):
                raise KeyError('metric_name and metric_target have mismatching lengths!.')
            if len(metric_weight) != len(metric_name):
                raise KeyError('metric_name and metric_weight have mismatching lengths!.')
        else:
            raise ValueError('When metric_name must be a tuple or a str.')   
        # Detect if the user has provided any arguments 
        # that will also-be extracted from DESC. 
        # If there are, these objectives will be discarded.
        if normalize:
            # When normalize is set to true, we 
            # use quantities from DESC to perform 
            # normalization instead.
            overridden_argnames = _DESC_DERIVED_ARGNAMES + ['objective_unit', 'constraint_unit']
        else:
            overridden_argnames = _DESC_DERIVED_ARGNAMES
        redundant_arg_names = set(overridden_argnames) & quadcoil_args.keys()
        if redundant_arg_names:
            warnings.warn(
                'Redundant arguments detected: ' 
                + str(redundant_arg_names)
                + '. These arguments are extracted from the equilibrium, '\
                'or specified by other parameters. The provided values '\
                'will be discarded.'
            )

        # ----- Storing equilibrium-independent, differentiable variables -----
        # These are differentiable quantities that are not equilibrium-dependent. 
        # They can be user-provided, but they also all have default values, so 
        # we set them here. This is necessary because we're calling quadcoil through
        # quadcoil.io.quadcoil_for_diff, which cannot see their default value in 
        # quadcoil.quadcoil.
        if 'net_toroidal_current_amperes' in quadcoil_args.keys():
            self.net_toroidal_current_amperes = quadcoil_args.pop('net_toroidal_current_amperes')
        else:
            self.net_toroidal_current_amperes = 0.
        if 'plasma_coil_distance' in quadcoil_args.keys():
            # A sign flip is necessary here!
            self.plasma_coil_distance = - quadcoil_args.pop('plasma_coil_distance')
        else:
            self.plasma_coil_distance = None
        if 'winding_dofs' in quadcoil_args.keys():
            # A sign flip is necessary here!
            self.winding_dofs = quadcoil_args.pop('winding_dofs')
        else:
            self.winding_dofs = None
        if 'objective_weight' in quadcoil_args.keys():
            self.objective_weight = quadcoil_args.pop('objective_weight')
        else:
            self.objective_weight = None
        if 'constraint_value' in quadcoil_args.keys():
            self.constraint_value = quadcoil_args.pop('constraint_value')
        else:
            self.constraint_value = jnp.array([])
        
        # ----- Setting attributes -----
        self.metric_name = metric_name
        self.metric_target = metric_target
        self.metric_weight = metric_weight
        self._verbose = verbose  
        self._source_grid = source_grid # B_normal grids
        self._Bnormal_plasma_chunk_size = Bnormal_plasma_chunk_size
        self.enable_Bnormal_plasma = enable_Bnormal_plasma
        self._plasma_M_theta = plasma_M_theta
        self._plasma_N_phi = plasma_N_phi
        self._constants = {}
        # These are differentiable quantities that are not equilibrium-dependent. 
        # They can be user-provided, but they also all have default values, so 
        # we set them here. This is necessary because we're calling quadcoil through
        # quadcoil.io.quadcoil_for_diff, which cannot see their default value in 
        # quadcoil.quadcoil.

        # ----- Calculating DESC-derived, non-differentiable attrs -----
        # eval_grid is used to generate quadrature points. 
        # it is also used to calculate surface Bnormal_plasma
        # when enable_Bnormal_plasma=True, along with surface_grid.
        # because we the quadrature points must be calculated before generating
        # quadcoil callable, it will be constructed here, instead of in the 
        # 
        # quadrature points, and will still be computed
        # when self.enable_Bnormal_plasma=False.
        eval_grid = LinearGrid(
            NFP=eq.NFP,
            # If we set this to sym it'll only evaluate 
            # theta from 0 to pi.
            sym=False, 
            M=self._plasma_M_theta,#Poloidal grid resolution.
            N=self._plasma_N_phi,
            rho=1.0
        )
        eval_profiles = get_profiles(_BPLASMA_DATA_KEYS, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(_BPLASMA_DATA_KEYS, obj=eq, grid=eval_grid)
        self._constants['eval_profiles'] = eval_profiles
        self._constants['eval_transforms'] = eval_transforms
        # The grid used to calculate net toroidal current
        # desc_net_poloidal_current_amperes = (
        #     -desc_eq.compute("G", grid=LinearGrid(rho=jnp.array(1.0)))["G"][0]
        #     / mu_0
        #     * 2
        #     * jnp.pi
        # )
        self.nfp = eq.NFP
        self.stellsym = eq.sym
        quadcoil_args['metric_name'] = metric_name
        quadcoil_args['nfp'] = eq.NFP
        quadcoil_args['stellsym'] = eq.sym
        quadcoil_args['plasma_mpol'] = eq.surface.M
        quadcoil_args['plasma_ntor'] = eq.surface.N
        quadcoil_args['plasma_quadpoints_phi'] = eval_grid.nodes[eval_grid.unique_zeta_idx, 2]/jnp.pi/2
        quadcoil_args['plasma_quadpoints_theta'] = eval_grid.nodes[eval_grid.unique_theta_idx, 1]/jnp.pi/2
        self.plasma_quadpoints_phi = tuple(quadcoil_args['plasma_quadpoints_phi'].tolist())
        self.plasma_quadpoints_theta = tuple(quadcoil_args['plasma_quadpoints_theta'].tolist())
        # ----- Generating quadcoil partial and its jvp rule -----
        # quadcoil_args is a mixture of static and traced arguments.
        # Because we likely will not adjust quadcoil settings dynamically,
        # here we treat all of them like staic using 
        # partial(quadcoil, **quadcoil_args), implemented in gen_quadcoil_for_diff.
        # The function also generates the custom_jvp rule based on the static arguments.
        # We store the resulting function as a static attribute.
        _quadcoil_full, _quadcoil_for_diff = gen_quadcoil_for_diff(**quadcoil_args)
        # Used later for Bnormal_plasma also
        self._quadcoil_for_diff = jit(_quadcoil_for_diff)
        self._quadcoil_full = jit(_quadcoil_full)
        
        # ----- Setting and registering keyword arguments -----
        self._static_attrs = [
            '_verbose',
            '_Bnormal_plasma_chunk_size',
            'enable_Bnormal_plasma',
            'metric_name',
            'nfp',
            'stellsym',
            '_plasma_M_theta',
            '_plasma_N_phi',
            '_surf_R_A',
            '_surf_R_c_indices',
            '_surf_R_s_indices',
            '_surf_Z_A',
            '_surf_Z_c_indices',
            '_surf_Z_s_indices',
            '_quadcoil_for_diff',
            '_quadcoil_full',
            'plasma_quadpoints_phi',
            'plasma_quadpoints_theta'
        ]
        
        # ----- Superclass -----
        super().__init__(
            things=[eq], # things is a list of things that will be optimized, in this case just the equilibrium
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
            jac_chunk_size=jac_chunk_size
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
        eq = self.things[0]
        # dim_f = size of the output vector returned by self.compute.
        # This is a scalar objective.
        self._dim_f = 1
        # some helper code for profiling and logging
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        # ----- Building the desc surf -> quadcoil (simsopt) surf map -----
        # self._desc_to_vmec_surf_R = ptolemy_identity_rev_jit_precomputation(eq.surface.R_basis.modes[:,1], eq.surface.R_basis.modes[:,2])
        # self._desc_to_vmec_surf_Z = ptolemy_identity_rev_jit_precomputation(eq.surface.Z_basis.modes[:,1], eq.surface.Z_basis.modes[:,2])
        self._surf_R_A, self._surf_R_c_indices, self._surf_R_s_indices = ptolemy_identity_rev_precompute(
            eq.surface.R_basis.modes[:,1], 
            eq.surface.R_basis.modes[:,2]
        )
        self._surf_Z_A, self._surf_Z_c_indices, self._surf_Z_s_indices = ptolemy_identity_rev_precompute(
            eq.surface.Z_basis.modes[:,1], 
            eq.surface.Z_basis.modes[:,2]
        )
        # ----- Building grids and transforms -----
        # source_grid for Bnormal_plasma, and eval_grid.
        # Eval grid has a special role, in that it helps
        # generate plasma_quadpoint_phi and theta. Therefore,
        # it will be generated in init instead.
        net_poloidal_current_grid = LinearGrid(rho=jnp.array(1.0))
        net_poloidal_current_profiles = get_profiles(['G'], obj=eq, grid=net_poloidal_current_grid)
        net_poloidal_current_transforms = get_transforms(['G'], obj=eq, grid=net_poloidal_current_grid)
        # Storing transforms
        # Attributes inside and outside _constants aren't really treated
        # differently, except that self._constants is traced. Because quadcoil_arg
        # is a mixture of traced and static inputs, we want to individually register
        # all the static inputs. Moreover, dicts are not hashable, so the static 
        # arguments in quadcoil_args must all be stored as individual attributes. 
        # We might as well store everything in quadcoil_args as individual attributes,\
        # and only store the transforms and profiles here in self._constants.

        self._constants['net_poloidal_current_profiles'] = net_poloidal_current_profiles
        self._constants['net_poloidal_current_transforms'] = net_poloidal_current_transforms
        # Mose DESC objectives are fields, so they 
        # hard-coded the superclass to ask for a weight 
        # to integrate the field over a quadrature...

        # source_grid will only be generated when self.enable_Bnormal_plasma == True.
        # Here, eval_grid is not only used to define Bnormal_plasma, 
        # but also used to generate plasma_quadpoints_phi and theta.
        # Therefore, it will be greated regardless self.valuum == True.
        if self.enable_Bnormal_plasma:
            if verbose:
                jax.debug.print('enable_Bnormal_plasma=True, QUADCOIL will no '\
                                'longer assume zero Bnormal_plasma at the boundary.')
            if self._source_grid is None:
                # for axisymmetry we still need to know about toroidal effects, so its
                # cheapest to pretend there are extra field periods
                source_grid = LinearGrid(
                    rho=np.array([1.0]),
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP if eq.N > 0 else 64,
                    sym=False,
                )
                source_profiles = get_profiles(_BPLASMA_DATA_KEYS, obj=eq, grid=source_grid)
                source_transforms = get_transforms(_BPLASMA_DATA_KEYS, obj=eq, grid=source_grid)
            else:
                source_grid = self._source_grid
            # Creating interpolator for Bnormal_plasma
            ratio_data = eq.compute(
                ["|e_theta x e_zeta|", "e_theta", "e_zeta"], grid=source_grid
            )
            st, sz, q = best_params(source_grid, best_ratio(ratio_data))
            try:
                interpolator = FFTInterpolator(self._constants['eval_grid'], source_grid, st, sz, q)
            except AssertionError as e:
                warnif(
                    True,
                    msg="Could not build fft interpolator, switching to dft which is slow."
                    "\nReason: " + str(e),
                )
                interpolator = DFTInterpolator(self._constants['eval_grid'], source_grid, st, sz, q)
            self._constants['source_profiles'] = source_profiles
            self._constants['source_transforms'] = source_transforms
            self._constants['interpolator'] = interpolator
            
        
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
                self.objective_name, 
                self.constraint_name, 
                self.scales
            )
            self.objective_unit = obj_unit_new
            self.constraint_unit = cons_unit_new

        # ----- Wrapping up and timing -----
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")
        # finally, call ``super.build()``
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        # We prohibit the user from providing constants
        return self.compute_full(params=params, full_mode=False)
    
    def compute_full(self, params, full_mode=True):
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
        # ----- Constructing rz surface dofs in Simsopt convention -----
        rs_raw, rc_raw = ptolemy_identity_rev_compute(self._surf_R_A, self._surf_R_c_indices, self._surf_R_s_indices, params['Rb_lmn'])
        zs_raw, zc_raw = ptolemy_identity_rev_compute(self._surf_Z_A, self._surf_Z_c_indices, self._surf_Z_s_indices, params['Zb_lmn'])
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

        # ----- Computing Bnormal_plasma -----
        # Copied from DESC/desc/objectives/_free_boundary.py
        constants = self._constants
        
        if self.enable_Bnormal_plasma:
            # Using the stored transforms to calculate B_normal_plasma
            source_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                _BPLASMA_DATA_KEYS,
                params=params,
                transforms=constants["source_transforms"],
                profiles=constants["source_profiles"],
            )
            eval_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                _BPLASMA_DATA_KEYS,
                params=params,
                transforms=constants["eval_transforms"],
                profiles=constants["eval_profiles"],
            )
            Bplasma = virtual_casing_biot_savart(
                eval_data,
                source_data,
                constants["interpolator"],
                chunk_size=self._B_plasma_chunk_size,
            )
            # need extra factor of B/2 bc we're evaluating on plasma surface
            Bplasma = Bplasma + eval_data["B"] / 2
            Bnormal_plasma = jnp.sum(Bplasma * eval_data["n_rho"], axis=1)
        else:
            Bnormal_plasma = jnp.zeros((
                len(self.plasma_quadpoints_phi),
                len(self.plasma_quadpoints_theta)
            ))
        
        # ----- Calling the net poloidal current -----
        G_data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            ["G"],                
            params=params,                   
            transforms=constants['net_poloidal_current_transforms'],
            profiles=constants['net_poloidal_current_profiles'],
        )
        net_poloidal_current_amperes = - G_data["G"][0] / mu_0 * 2 * jnp.pi
        # ----- Calling the quadcoil wrapper with custom_vjp -----  
        if full_mode:
            out_dict, qp, cp_mn, solve_results = self._quadcoil_full(
                plasma_dofs=plasma_dofs,
                net_poloidal_current_amperes=net_poloidal_current_amperes,
                net_toroidal_current_amperes=self.net_toroidal_current_amperes, 
                Bnormal_plasma=Bnormal_plasma,
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
            Bnormal_plasma=Bnormal_plasma,
            plasma_coil_distance=self.plasma_coil_distance,
            winding_dofs=self.winding_dofs,
            objective_weight=self.objective_weight,
            constraint_value=self.constraint_value,
        )

        # ----- Thresholding and weighing -----
        f_out = 0.
        for i in range(len(self.metric_name)):
            # Set during the loop through quadcoil_args
            f_name = self.metric_name[i]
            f_weight = self.metric_weight[i]
            f_target_eff = self.metric_target[i]
            f_val_eff = metric_dict[f_name]
            # MEY NOT BE LOWERABLE?
            if self._normalize or self._normalize_target:
                f_unit = get_quantity(f_name + '_desc_unit')(self.scales)
            if self._normalize:
                f_val_eff = f_val_eff / f_unit
            if self._normalize_target:
                f_target_eff = f_target_eff / f_unit
            f_out = f_out + f_weight * jnp.where(f_val_eff > f_target_eff, f_val_eff-f_target_eff, 0.)

        return f_out

# ----- Helper functions -----
def ptolemy_identity_rev_precompute(m_1, n_1):
    """
    We have split ptolemy_identity_rev into two parts:
    ``ptolemy_identity_rev_precompute`` and
    ``ptolemy_identity_rev_compute``. The ``original ptolemy_identity_rev``
    relies on numpy boolean indexing. Even when we set m_1, n_1 to static, 
    they will still be converted to traced arrays once jit happens, and the 
    numpy boolean indexing will break. Because of that, we perform all numpy
    operations in ``ptolemy_identity_rev_precompute`` during ``build()``,
    store the results as static, and then perform all jaxable operations in
    ``ptolemy_identity_rev_compute`` during ``compute()``.

    .. code-block:: python
    
        desc_to_vmec_surf_R = ptolemy_identity_rev_jit_precomputation(
            tuple(eq.surface.R_basis.modes[:,1]), 
            tuple(eq.surface.R_basis.modes[:,2])
        )
        desc_to_vmec_surf_Z = ptolemy_identity_rev_jit_precomputation(
            tuple(eq.surface.Z_basis.modes[:,1]), 
            tuple(eq.surface.Z_basis.modes[:,2])
        )
        rs_raw, rc_raw = desc_to_vmec_surf_R(eq.surface.R_lmn)
        zs_raw, zc_raw = desc_to_vmec_surf_Z(eq.surface.Z_lmn)# Stellsym SurfaceRZFourier's dofs consists of 
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
        if eq.sym:
            dofs = jnp.concatenate([rc, zs])
        else:
            dofs = jnp.concatenate([rc, rs, zc, zs])
    
    Converts from a double Fourier series of the form:
        ss * sin(m𝛉) * sin(n𝛟) + sc * sin(m𝛉) * cos(n𝛟) +
        cs * cos(m𝛉) * sin(n𝛟) + cc * cos(m𝛉) * cos(n𝛟)
    to the double-angle form:
        s * sin(m𝛉-n𝛟) + c * cos(m𝛉-n𝛟)
    using Ptolemy's sum and difference formulas.

    Parameters
    ----------
    m_1 : ndarray, shape(num_modes,)
    n_1 : ndarray, shape(num_modes,)
        ``R_basis_modes[:,1], R_basis_modes[:,2]`` or ``Z_basis_modes[:,1], Z_basis_modes[:,2]`` 

    Returns
    -------
    A, c_indices, s_indices : tuples
        .. code-block:: python

            # For calculating rs_raw, rc_raw,
            x = np.atleast_2d(desc_surf.R_lmn)
            y = (A @ x.T).T
            rs_raw, rc_raw = _modes_x_to_mnsc(vmec_modes, y)

    """
    try:
        from desc.backend import jnp, sign
        # from desc.vmec_utils import ptolemy_linear_transform # , _modes_x_to_mnsc
    except:
        raise ModuleNotFoundError('desc.backend.jnp and desc.backend.sign unavailable.')

    # Precomputing linear operators
    m_1, n_1 = map(np.atleast_1d, (m_1, n_1))
    desc_modes = np.vstack([np.zeros_like(m_1), m_1, n_1]).T
    A, vmec_modes = ptolemy_linear_transform(desc_modes)
    cmask = vmec_modes[:, 0] == 1
    smask = vmec_modes[:, 0] == -1
    c_indices = np.where(cmask)[0]
    s_indices = np.where(smask)[0]
    # A is 2d, and this converts 2d arr to tuple.
    A = tuple(map(tuple, A.tolist()))
    c_indices = tuple(c_indices.tolist())
    s_indices = tuple(s_indices.tolist())
    return A, c_indices, s_indices

@partial(jit, static_argnames=['A', 'c_indices', 's_indices', ])
def ptolemy_identity_rev_compute(A, c_indices, s_indices, x):
    A = jnp.array(A)
    y = (A @ x.T).T
    if len(c_indices):
        c = (y.T[jnp.array(c_indices)]).T
    if len(s_indices):
        s = (y.T[jnp.array(s_indices)]).T
        # if there are sin terms, add a zero for the m=n=0 mode
        s = jnp.concatenate([jnp.zeros_like(s.T[:1]), s.T]).T
    
    if not len(s_indices):
        s = jnp.zeros_like(c) 
    if not len(c_indices):
        c = jnp.zeros_like(s)
    assert len(s.T) == len(c.T)
    return s, c



# def ptolemy_identity_rev_jit_precomputation(m_1, n_1):
#     """
#     Precompute a map that converts from double-Fourier to double-angle form using Ptolemy's identity.
#     These pre-computed maps, along with some array manipulation, convert desc surfaces
#     to simsopt surfaces. m_1, n_1 must be tuples

#     .. code-block:: python
    
#         desc_to_vmec_surf_R = ptolemy_identity_rev_jit_precomputation(
#             tuple(eq.surface.R_basis.modes[:,1]), 
#             tuple(eq.surface.R_basis.modes[:,2])
#         )
#         desc_to_vmec_surf_Z = ptolemy_identity_rev_jit_precomputation(
#             tuple(eq.surface.Z_basis.modes[:,1]), 
#             tuple(eq.surface.Z_basis.modes[:,2])
#         )
#         rs_raw, rc_raw = desc_to_vmec_surf_R(eq.surface.R_lmn)
#         zs_raw, zc_raw = desc_to_vmec_surf_Z(eq.surface.Z_lmn)# Stellsym SurfaceRZFourier's dofs consists of 
#         # [rc, zs]
#         # Non-stellsym SurfaceRZFourier's dofs consists of 
#         # [rc, rs, zc, zs]
#         # Because rs, zs from ptolemy_identity_rev shares the same m, n 
#         # arrays as rc, zc, they both have a zero as the first element 
#         # that need to be removed.
#         rc = rc_raw.flatten()
#         rs = rs_raw.flatten()[1:]
#         zc = zc_raw.flatten()
#         zs = zs_raw.flatten()[1:]
#         if eq.sym:
#             dofs = jnp.concatenate([rc, zs])
#         else:
#             dofs = jnp.concatenate([rc, rs, zc, zs])
    
#     Converts from a double Fourier series of the form:
#         ss * sin(m𝛉) * sin(n𝛟) + sc * sin(m𝛉) * cos(n𝛟) +
#         cs * cos(m𝛉) * sin(n𝛟) + cc * cos(m𝛉) * cos(n𝛟)
#     to the double-angle form:
#         s * sin(m𝛉-n𝛟) + c * cos(m𝛉-n𝛟)
#     using Ptolemy's sum and difference formulas.

#     Parameters
#     ----------
#     m_1 : ndarray, shape(num_modes,)
#     n_1 : ndarray, shape(num_modes,)
#         ``R_basis_modes[:,1], R_basis_modes[:,2]`` or ``Z_basis_modes[:,1], Z_basis_modes[:,2]`` 

#     Returns
#     -------
#     A, vec_modes : ndarray
#         .. code-block:: python

#             # For calculating rs_raw, rc_raw,
#             x = np.atleast_2d(desc_surf.R_lmn)
#             y = (A @ x.T).T
#             rs_raw, rc_raw = _modes_x_to_mnsc(vmec_modes, y)

#     """
#     try:
#         from desc.backend import jnp, sign
#         # from desc.vmec_utils import ptolemy_linear_transform # , _modes_x_to_mnsc
#     except:
#         raise ModuleNotFoundError('desc.backend.jnp and desc.backend.sign unavailable.')

#     # Precomputing linear operators
#     m_1, n_1 = map(np.atleast_1d, (m_1, n_1))
#     desc_modes = np.vstack([np.zeros_like(m_1), m_1, n_1]).T
#     A, vmec_modes = ptolemy_linear_transform(desc_modes)

#     # Precomputing map
#     def transform(x):
#         y = (A @ x.T).T
#         cmask = vmec_modes[:, 0] == 1
#         smask = vmec_modes[:, 0] == -1
    
#         c_indices = np.where(cmask)[0]
#         s_indices = np.where(smask)[0]
    
#         c = (y.T[c_indices]).T
#         s = (y.T[s_indices]).T
        
#         if not len(s.T):
#             s = jnp.zeros_like(c)
#         elif len(s.T):  # if there are sin terms, add a zero for the m=n=0 mode
#             s = jnp.concatenate([jnp.zeros_like(s.T[:1]), s.T]).T
#         if not len(c.T):
#             c = jnp.zeros_like(s)
#         assert len(s.T) == len(c.T)
#         return s, c
        
#     return transform
