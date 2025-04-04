from desc.utils import Timer
from desc.vmec_utils import ptolemy_linear_transform
from desc.grid import LinearGrid
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.objectives.normalization import compute_scaling_factors
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
import jax.numpy as jnp
import jax # for printing
import warnings
import numpy as np
import jax

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
    vacuum : bool
        (Traced) Whether the optimization problem is vacuum.
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
    _nondiff_args_name : tuple(str)
        (Static) : a list of non-differentiable variables. Generated 
        based on ``_DIFF_USER_ARGNAMES`` and ``quadcoil_args.keys()``.
        For generating nondiff_args, the non-differentiable component of 
        quadcoil.io.quadcoil_for_diff

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
        vacuum:bool,
        quadcoil_args,
        metric_name,
        metric_target,
        metric_weight,
        plasma_M_theta:int,
        plasma_N_phi:int,
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
        # ----- Importing -----
        try:
            # We will use QUADCOIL_STATIC_ARGNAMES to tell DESC which argument
            # in quadcoil_args is static. The rest are assumed traced.
            from quadcoil import QUADCOIL_STATIC_ARGNAMES, get_objective
            from quadcoil.io import quadcoil_for_diff, quadcoil_for_diff_full
        except:
            ModuleNotFoundError('quadcoil must be available for '\
                                'this wrapper to work. Please visit '\
                                'https://github.com/lankef/quadcoil.')
        
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
            if len(metric_target) != len(quadcoil_args['metric_name']):
                raise KeyError('metric_name and metric_target have mismatching lengths!.')
            if len(metric_weight) != len(quadcoil_args['metric_name']):
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
        
        # ----- Setting attributes -----
        self.metric_name = metric_name
        self.metric_target = metric_target
        self.metric_weight = metric_weight
        self._verbose = verbose
        self._plasma_M_theta = plasma_M_theta
        self._plasma_N_phi = plasma_N_phi       
        self._source_grid = source_grid # B_normal grids
        self._Bnormal_plasma_chunk_size = Bnormal_plasma_chunk_size
        self.vacuum = vacuum
        # These are differentiable quantities that are not equilibrium-dependent. 
        # They can be user-provided, but they also all have default values, so 
        # we set them here. This is necessary because we're calling quadcoil through
        # quadcoil.io.quadcoil_for_diff, which cannot see their default value in 
        # quadcoil.quadcoil.
        if 'net_toroidal_current_amperes' in quadcoil_args.keys():
            self.net_toroidal_current_amperes = quadcoil_args.pop('net_toroidal_current_amperes')
        else:
            self.net_toroidal_current_amperes = 0
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
            
        # ----- Setting and registering keyword arguments -----
        # loop over all keys in quadcoil_args, registering 
        # every one of them as a class attribute. 
        # - If user_key appear in _DESC_DERIVED_ARGNAMES, it will not be registered as an attribute. 
        # - If user_key appear in QUADCOIL_STATIC_ARGNAMES, it will be registered as static. 
        # - If user_key DOES NOT appear in _DIFF_USER_ARGNAMES, it will be added to the tuple of 
        #     strings, _nondiff_args_name, that build() will use to construct nondiff_args
        #     for quadcoil.io.quadcoil_for_diff
        # 
        # Frank to developers: This seems convoluted, but it will allow users to use default values
        # like passing a **kwargs.
        # It will also allow developers to add new non-differentiable arguments, static or traced,
        # to quadcoil.quadcoil without breaking this objective. Adding new differentiable arguments 
        # will require modifications to quadcoil.io.quadcoil_for_diff, but I think the included list
        # of differentiable is already fairly comprehensive.
        _static_attrs = [
            '_verbose',
            '_Bnormal_plasma_chunk_size'
            '_nondiff_args_name',
            '_plasma_M_theta',
            '_plasma_N_phi',
            'vacuum',
            'metric_name',
            'nfp',
            'stellsym',
            '_desc_to_vmec_surf_R'
            '_desc_to_vmec_surf_Z'
        ]
        # This will become a list of all non-differentiable 
        # attributes that will be passed into quadcoil.io.quadcoil_for_diff.
        # The current elements are the non-differentiable elements that cna
        # be extracted from DESC.
        _nondiff_args_name = [
            'nfp',
            'stellsym',
            'plasma_mpol',
            'plasma_ntor',
            'plasma_quadpoints_phi',
            'plasma_quadpoints_theta',
        ]
        # loop over all user-provided arguments for quadcoil
        for user_key, user_value in quadcoil_args.items():
            if user_key in _DESC_DERIVED_ARGNAMES:
                continue
            if user_key in QUADCOIL_STATIC_ARGNAMES:
                _static_attrs.append(user_key)
            if user_key not in _DIFF_USER_ARGNAMES:
                _nondiff_args_name.append(user_key)
            # After registering static and non-differentiable
            # attribute names, add the attributes to self.
            setattr(self, user_key, user_value)
        # This is a special exception: 'constraint_name' may not be in 
        # quadcoil_args for unconstrained problems, because by default
        # quadcoil is unconstrained. However, we still want to 
        # use the convenient scaling function, generate_desc_scaling. 
        # So we need to set a default for it. We set this default to 
        # (''), because _Objective also happens to not like empty tuple
        # attributes. We also modify constraint_value and unit to be the same 
        # length.
        if 'constraint_name' not in quadcoil_args.keys() or len(quadcoil_args['constraint_name'])==0:
            self.constraint_name = ('',)
            self.constraint_type = ('',)
            self.constraint_unit = jnp.array([1.,])
            self.constraint_value = jnp.array([0.,])
            _static_attrs.append('constraint_name')
            _static_attrs.append('constraint_type')
        self._static_attrs = _static_attrs
        # Convert the list to tuples, because lists are mutable.
        self._nondiff_args_name = tuple(_nondiff_args_name)
        
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

        # ----- Building the desc surff -> quadcoil (simsopt) surf map -----
        self._desc_to_vmec_surf_R = ptolemy_map_precomputation(eq.surface.R_basis.modes[:,1], eq.surface.R_basis.modes[:,2])
        self._desc_to_vmec_surf_Z = ptolemy_map_precomputation(eq.surface.Z_basis.modes[:,1], eq.surface.Z_basis.modes[:,2])
        
        # ----- Building grids and transforms -----
        # source_grid for Bnormal_plasma, and eval_grid.
        # source_grid will only be generated when self.vacuum == False.
        # Here, eval_grid is not only used to define Bnormal_plasma, 
        # but also used to generate plasma_quadpoints_phi and theta.
        # Therefore, it will be greated regardless self.valuum == True.
        if self.vacuum:
            source_profiles=None
            source_transforms=None
            interpolator=None
        else:
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
            try:
                interpolator = FFTInterpolator(eval_grid, source_grid, st, sz, q)
            except AssertionError as e:
                warnif(
                    True,
                    msg="Could not build fft interpolator, switching to dft which is slow."
                    "\nReason: " + str(e),
                )
                interpolator = DFTInterpolator(eval_grid, source_grid, st, sz, q)
        # eval_grid for Bnormal_plasma
        # Unlike in DESC/desc/integrals/singularities.py, 
        # we construct the eval_grid with 
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
        # The grid used to calculate net toroidal current
        net_poloidal_current_grid = LinearGrid(rho=jnp.array(1.0))
        net_poloidal_current_profiles = get_profiles(['G'], obj=eq, grid=net_poloidal_current_grid)
        net_poloidal_current_transforms = get_transforms(['G'], obj=eq, grid=net_poloidal_current_grid)
        neq = 2 # The _sheet_current == False case in 
        # Storing transforms
        # Attributes inside and outside _constants aren't really treated
        # differently, except that self._constants is traced. Because quadcoil_arg
        # is a mixture of traced and static inputs, we want to individually register
        # all the static inputs. Moreover, dicts are not hashable, so the static 
        # arguments in quadcoil_args must all be stored as individual attributes. 
        # We might as well store everything in quadcoil_args as individual attributes,\
        # and only store the transforms and profiles here in self._constants.
        self._constants = {
            'source_profiles': source_profiles,
            'source_transforms': source_transforms,
            'eval_profiles': eval_profiles,
            'eval_transforms': eval_transforms,
            'net_poloidal_current_profiles': net_poloidal_current_profiles,
            'net_poloidal_current_transforms': net_poloidal_current_transforms,
            'interpolator': interpolator,
            # Mose DESC objectives are fields, so they 
            # hard-coded the superclass to ask for a weight 
            # to integrate the field over a quadrature...
            'quad_weights': 1., 
        }
        
        # ----- Calculating DESC-derived, non-differentiable attrs -----
        self.nfp = eq.NFP
        self.stellsym = eq.sym
        self.plasma_mpol = eq.surface.M
        self.plasma_ntor = eq.surface.N
        self.plasma_quadpoints_phi = eval_grid.nodes[eval_grid.unique_zeta_idx, 2]/jnp.pi/2
        self.plasma_quadpoints_theta = eval_grid.nodes[eval_grid.unique_theta_idx, 1]/jnp.pi/2
        # This far, we have evaluated all variables that must be included in 
        # nondiff_args for quadcoil.io.quadcoil_for_diff. Because some of these 
        # are static, and a dict is not hashable, we do not construct the full dictionary
        # here, but instead in compute(), where quadcoil.io.quadcoil_for_diff is called.
        
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
        # ----- Constructing nondiff_dict -----
        # Strape up all the non-differentiable, equilibrium-independent arguments,
        # traced or static, into a dictionary to pass into quadcoil's custom_jvp wrapper.
        # We do this here because some of these quantities are static, so we can't
        # group them into one object during build.
        # Besides the ones we catalogued earlier in self._nondiff_args_name, we 
        # also need to add in the special exceptions: default values for 
        # constraints.
        nondiff_args = {
            'constraint_name': self.constraint_name,
            'constraint_type': self.constraint_type,
            'constraint_unit': self.constraint_unit,
            'constraint_value': self.constraint_value,
        }
        for nondiff_keys in self._nondiff_args_name:
            nondiff_args[nondiff_keys] = getattr(self, nondiff_keys)
        nondiff_args['verbose'] = self._verbose
        
        # ----- Constructing rz surface dofs in Simsopt convention -----
        rs_raw, rc_raw = self._desc_to_vmec_surf_R(params['Rb_lmn'])
        zs_raw, zc_raw = self._desc_to_vmec_surf_Z(params['Zb_lmn'])
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
        
        if not self.vacuum:
            # Using the stored transforms to calculate B_normal_plasma
            source_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                _BPLASMA_DATA_KEYS,
                params=eq_params,
                transforms=constants["source_transforms"],
                profiles=constants["source_profiles"],
            )
            eval_data = compute_fun(
                "desc.equilibrium.equilibrium.Equilibrium",
                _BPLASMA_DATA_KEYS,
                params=eq_params,
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
        
        # ----- Calling the quadcoil wrapper with custom_vjp -----       
        if full_mode:
            # For testing if the parameters are carried through correctly
            return quadcoil_for_diff_full(
                plasma_dofs=plasma_dofs,
                net_poloidal_current_amperes=G_data["G"][0],
                net_toroidal_current_amperes=self.net_toroidal_current_amperes, 
                Bnormal_plasma=Bnormal_plasma,
                plasma_coil_distance=self.plasma_coil_distance,
                winding_dofs=self.winding_dofs,
                objective_weight=self.objective_weight,
                constraint_value=self.constraint_value,
                nondiff_args=nondiff_args
            )
            
        metric_dict = quadcoil_for_diff(
            plasma_dofs=plasma_dofs,
            net_poloidal_current_amperes=G_data["G"][0],
            net_toroidal_current_amperes=self.net_toroidal_current_amperes, 
            Bnormal_plasma=Bnormal_plasma,
            plasma_coil_distance=self.plasma_coil_distance,
            winding_dofs=self.winding_dofs,
            objective_weight=self.objective_weight,
            constraint_value=self.constraint_value,
            nondiff_args=nondiff_args
        )

        # ----- Thresholding and weighing -----
        f_out = 0.
        for i in range(len(self.metric_name)):
            # Set during the loop through quadcoil_args
            f_name = self.metric_name[i]
            f_weight = self.metric_weight[i]
            f_target_eff = self.metric_target[i]
            f_val_eff = metric_dict[f_name]
            if self._normalize or self._normalize_target:
                f_unit = get_objective(f_name + '_desc_unit')(self.scales)
            if self._normalize:
                f_val_eff = f_val_eff / f_unit
            if self._normalize_target:
                f_target_eff = f_target_eff / f_unit
            f_out = f_out + f_weight * jnp.where(f_val_eff > f_target_eff, f_val_eff-f_target_eff, 0.)

        return f_out

# ----- Helper functions -----
def ptolemy_map_precomputation(m_1, n_1):
    """
    Precompute a map that converts from double-Fourier to double-angle form using Ptolemy's identity.
    These pre-computed maps, along with some array manipulation, convert desc surfaces
    to simsopt surfaces.

    .. code-block:: python
    
        desc_to_vmec_surf_R = ptolemy_map_precomputation(
            eq.surface.R_basis.modes[:,1], 
            eq.surface.R_basis.modes[:,2]
        )
        desc_to_vmec_surf_Z = ptolemy_map_precomputation(
            eq.surface.Z_basis.modes[:,1], 
            eq.surface.Z_basis.modes[:,2]
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
    A, vec_modes : ndarray
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

    # Precomputing map
    def transform(x):
        y = (A @ x.T).T
        cmask = vmec_modes[:, 0] == 1
        smask = vmec_modes[:, 0] == -1
    
        c_indices = np.where(cmask)[0]
        s_indices = np.where(smask)[0]
    
        c = (y.T[c_indices]).T
        s = (y.T[s_indices]).T
        
        if not len(s.T):
            s = jnp.zeros_like(c)
        elif len(s.T):  # if there are sin terms, add a zero for the m=n=0 mode
            s = jnp.concatenate([jnp.zeros_like(s.T[:1]), s.T]).T
        if not len(c.T):
            c = jnp.zeros_like(s)
        assert len(s.T) == len(c.T)
        return s, c
        
    return transform
