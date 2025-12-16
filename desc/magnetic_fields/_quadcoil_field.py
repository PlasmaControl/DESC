from ._current_potential import FourierCurrentPotentialField # CurrentPotentialField 
import inspect
import numpy as np
from functools import partial
from desc.backend import jnp
from desc.optimizable import optimizable_parameter
from desc.compute import get_profiles, get_transforms
from quadcoil.quadcoil import _input_checking, _parse_objectives, _parse_constraints
from quadcoil import (
    QuadcoilParams, 
    SurfaceRZFourierJAX,
    merge_callables, 
    gen_winding_surface_arc, 
)
from desc.vmec_utils import ptolemy_identity_fwd
from desc.grid import LinearGrid
from jax import jit, flatten_util
import warnings
from desc.objectives._quadcoil_utils import (
    _BCOIL_DATA_KEYS, _BPLASMA_DATA_KEYS,
    ptolemy_identity_rev_precompute,
    ptolemy_identity_rev_compute,
    interpolate_array,
    quadcoil_phi_to_desc_phi,
    toroidal_flip,
    compute_Bnormal_plasma,
    compute_eval_data_coils,
    compute_Bnormal_ext,
    compute_Bnormal,
)


class QuadcoilField(FourierCurrentPotentialField):
    """
    
    Attributes:
    eq
    mpol
    ntor
    quadpoints_phi
    quadpoints_theta
    plasma_quadpoints_phi
    plasma_quadpoints_theta
    winding_mpol
    winding_ntor
    winding_quadpoints_phi
    winding_quadpoints_theta
    constraint_name
    constraint_type
    constraint_unit
    constraint_value
    objective_name
    objective_unit
    objective_weight

    plasma_coil_distance: 
    A scalar. If None, then the winding surface is a reference to another optimizable 
    surface, and the accompanying QuadcoilObjective.things and QuadcoilConstraints.things
    will include an optimizable surface.

    _ptolemy_Phi_A
    _ptolemy_Phi_c_indices
    _ptolemy_Phi_s_indices
    _ptolemy_R_plasma_A
    _ptolemy_R_plasma_c_indices
    _ptolemy_R_plasma_s_indices
    _ptolemy_Z_plasma_A
    _ptolemy_Z_plasma_c_indices
    _ptolemy_Z_plasma_s_indices
    _ptolemy_R_winding_A
    _ptolemy_R_winding_c_indices
    _ptolemy_R_winding_s_indices
    _ptolemy_Z_winding_A
    _ptolemy_Z_winding_c_indices
    _ptolemy_Z_winding_s_indices
    """

    _io_attrs_ = (
        FourierCurrentPotentialField._io_attrs_
        + []
    )

    _static_attrs = (
        FourierCurrentPotentialField._static_attrs
        + [
            '_winding_surface_generator',
            '_plasma_quadpoints_phi_native',
            '_plasma_quadpoints_theta_native',
            '_plasma_quadpoints_phi',
            '_plasma_quadpoints_theta',
            '_ptolemy_R_winding_A',
            '_ptolemy_R_winding_c_indices',
            '_ptolemy_R_winding_s_indices',
            '_ptolemy_Z_winding_A',
            '_ptolemy_Z_winding_c_indices',
            '_ptolemy_Z_winding_s_indices',
            '_winding_quadpoints_phi',
            '_winding_quadpoints_theta',
            '_quadpoints_phi',
            '_quadpoints_theta',
            '_ptolemy_Phi_A',
            '_ptolemy_Phi_c_indices',
            '_ptolemy_Phi_s_indices',
            '_ptolemy_R_plasma_A',
            '_ptolemy_R_plasma_c_indices',
            '_ptolemy_R_plasma_s_indices',
            '_ptolemy_Z_plasma_A',
            '_ptolemy_Z_plasma_c_indices',
            '_ptolemy_Z_plasma_s_indices',
            'unravel_aux_dofs',
            '_aux_dofs_flat',
            '_f_quadcoil',
            '_g_quadcoil',
            '_h_quadcoil',    
            '_constants',
            '_enable_net_current_plasma',
            '_enable_Bnormal_plasma',
            '_eq_fixed',
            '_field',
            '_field_fixed',
            '_Bnormal_shape',
            '_bs_chunk_size',
            '_bplasma_chunk_size',
        ]
    )

    #  TODO: 
    # 1. find a good way to fix I and G as part of quadcoil constraints
    # 2. modify quadcoil objective and constraint to use quadcoil field's things
    # 3.  

    def __init__(
        self,
        eq,
        # By default, the plasma surface has quadpoints
        # number based on DESC quadrature points.
        plasma_M_theta:int,
        plasma_N_phi:int,
        quadpoints_phi=None,
        quadpoints_theta=None,
        # Winding surface optimization mode 
        winding_surface=None,
        # These are when winding surfaces are not provided
        plasma_coil_distance=None,
        M=6, # Number of poloidal harmonics in the winding surface. Equivalent to mpol in simsopt.
        N=5, # Number of toroidal harmonics in the winding surface. Equivalent to ntor in simsopt.
        winding_quadpoints_phi=None,
        winding_quadpoints_theta=None,
        # These are for winding surface optimization
        winding_surface_generator=gen_winding_surface_arc,
        # QUADCOIL params
        objective_name='f_B',
        objective_weight=1.,
        objective_unit=None,
        constraint_name=(),
        constraint_type=(),
        constraint_unit=(),
        constraint_value=jnp.array([]),
        # Guesses for the slack variables
        aux_dofs_vals={},
        # External coils - no external coils by default
        field=[],
        field_grid=None,
        enable_net_current_plasma=True,
        eq_fixed=False, # Whether the equilibrium are fixed
        field_fixed=True, # Whether the external fields are fixed
        enable_Bnormal_plasma=False, # Whether to enable free-boundary
        # Args for FourierCurrentPotentialField
        Phi_mn=np.array([0.0]),
        modes_Phi=np.array([[0, 0]]),
        I=0,
        G=0,
        sym_Phi=False,
        M_Phi=None,
        N_Phi=None,
        name="",
        check_orientation=True,
        smoothing='approx',
        smoothing_params={'lse_epsilon': 1e-3},
        # Misc
        bs_chunk_size=None,
        bplasma_chunk_size=None,
    ):
        self._eq = eq
        self._constants = {}
        self._field = field
        if field:
            from desc.magnetic_fields import SumMagneticField
            field = [field] if not isinstance(field, list) else field
            self._constants['sum_field'] = SumMagneticField(field)
        # Coil and things initialization
        self._enable_net_current_plasma = enable_net_current_plasma
        self._enable_Bnormal_plasma = enable_Bnormal_plasma
        self._eq_fixed = eq_fixed
        self._field_fixed = field_fixed
        self._things = []
        self._constants['field_grid'] = field_grid
        self._bs_chunk_size = bs_chunk_size
        self._bplasma_chunk_size = bplasma_chunk_size
        if not (eq_fixed or field_fixed):
            warnings.warn('Both eq_fixed and field_fixed are True. things will be empty.')
        if not eq_fixed:
            self._things += [eq]
        if not field_fixed:
            if not field:
                raise AttributeError('When field_fixed is False, field must be an object or a non-empty list.')
            self._things += [field]

        if not (G or field):
            warnings.warn('enable_net_current_plasma is false and field is empty. The problem may be trivial.')
        
        if G and field:
            warnings.warn('There are both external coils and net current. '
                          'This is very uncommon (windowpane filaments + winding surface with net current).')

        # Callculating things for QuadcoilObjective and QuadcoilConstraint
        # Checking if constraints and objective inputs are legal
        _input_checking(
            objective_name=objective_name,
            objective_weight=objective_weight,
            objective_unit=objective_unit,
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value,
        )
        self._winding_surface_generator = winding_surface_generator

        # ----- Setting plasma quantities -----
        plasma_grid = LinearGrid(
            NFP=eq.NFP,
            # If we set this to sym it'll only evaluate 
            # theta from 0 to pi.
            sym=False, 
            M=plasma_M_theta,#Poloidal grid resolution.
            N=plasma_N_phi,
            rho=1.0
        )
        self._plasma_quadpoints_phi = plasma_grid.nodes[plasma_grid.unique_zeta_idx, 2]/jnp.pi/2
        self._plasma_quadpoints_theta = plasma_grid.nodes[plasma_grid.unique_theta_idx, 1]/jnp.pi/2
        self._Bnormal_shape = (len(self._plasma_quadpoints_phi), len(self._plasma_quadpoints_theta))
        # ----- Treating the winding surface -----
        # In the default behavior of FourierCurrentPotential,
        # the winding surface is part of params and can be optimized.
        # This is not always True in QUADCOIL. Sometimes we want to 
        # use an automatically generated winding surface instead.
        # So, before initializing the superclass, we must
        # handle the winding surface first.
        self._plasma_coil_distance = plasma_coil_distance
        if winding_quadpoints_phi is None:
            winding_quadpoints_phi = jnp.linspace(0, 1, 32*eq.NFP, endpoint=False)
        if winding_quadpoints_theta is None:
            winding_quadpoints_theta = jnp.linspace(0, 1, 34, endpoint=False)
        self._winding_quadpoints_phi = winding_quadpoints_phi
        self._winding_quadpoints_theta = winding_quadpoints_theta
        if plasma_coil_distance is None:
            if winding_surface is None:
                raise ValueError('One of plasma_coil_distance and winding_surface must be provided.')
            # Plasma-blanket optimization. The winding surface and the plasma are both allowed to change.
            # ----- Building the desc surf -> quadcoil (simsopt) surf map -----
            (
                self._ptolemy_R_winding_A, 
                self._ptolemy_R_winding_c_indices, 
                self._ptolemy_R_winding_s_indices
            ) = ptolemy_identity_rev_precompute(
                winding_surface._R_basis.modes[:,1], 
                winding_surface._R_basis.modes[:,2]
            )
            (
                self._ptolemy_Z_winding_A, 
                self._ptolemy_Z_winding_c_indices, 
                self._ptolemy_Z_winding_s_indices
            ) = ptolemy_identity_rev_precompute(
                winding_surface._Z_basis.modes[:,1], 
                winding_surface._Z_basis.modes[:,2]
            )
            # ----- Initializing the superclass -----
            R_lmn = winding_surface.R_lmn
            Z_lmn = winding_surface.Z_lmn
            modes_R = winding_surface._R_basis.modes[:, 1:]
            modes_Z = winding_surface._Z_basis.modes[:, 1:]
            NFP = winding_surface.NFP
            sym = winding_surface.sym
            name = winding_surface.name
            # Not a good choice of quadrature resolution
            # because the winding surfacve can have very low M and N.
            # Tieing basis M and N to resolution is always a bad idea
            # for a quadcoil problem, because the winding surface itself 
            # does not share the same M and N as the current potential.
            # winding_grid = LinearGrid(
            #     NFP=winding_surface.NFP,
            #     # If we set this to sym it'll only evaluate 
            #     # theta from 0 to pi.
            #     sym=False, 
            #     M=winding_surface.M,#Poloidal grid resolution.
            #     N=winding_surface.N,
            #     rho=1.0
            # )
            # winding_quadpoints_phi_1fp = winding_grid.nodes[winding_grid.unique_zeta_idx, 2]/jnp.pi/2
            # # The quadpoints for the winding surface needs to cover all field periods. 
            # # this repeats the quadpoints over all NFP.
            # self._winding_quadpoints_phi = (
            #     jnp.repeat(jnp.linspace(0, 1, NFP, endpoint=False), len(winding_quadpoints_phi_1fp))
            #     + jnp.tile(winding_quadpoints_phi_1fp, NFP)
            # )
            # self._winding_quadpoints_theta = winding_grid.nodes[winding_grid.unique_theta_idx, 1]/jnp.pi/2
        else:
            self._ptolemy_R_winding_A = None
            self._ptolemy_R_winding_c_indices = None
            self._ptolemy_R_winding_s_indices = None
            self._ptolemy_Z_winding_A = None
            self._ptolemy_Z_winding_c_indices = None
            self._ptolemy_Z_winding_s_indices = None
            NFP = eq.surface.NFP
            sym = eq.surface.sym
            R_lmn = None # winding_surface.R_lmn
            Z_lmn = None # winding_surface.Z_lmn
            modes_R = None # winding_surface._R_basis.modes[:, 1:]
            modes_Z = None # winding_surface._Z_basis.modes[:, 1:]
            name = 'Automated winding surface'

        if quadpoints_phi is None or quadpoints_theta is None:
            quadpoints_phi = jnp.linspace(0, 1/NFP, 32, endpoint=False)
            quadpoints_theta = jnp.linspace(0, 1, 34, endpoint=False)
        self._quadpoints_phi, self._quadpoints_theta = quadpoints_phi, quadpoints_theta

        if sym_Phi == "auto":
            sym_Phi = "sin" if sym else False
        super().__init__(
            Phi_mn=Phi_mn,
            modes_Phi=modes_Phi,
            I=I,
            G=G,
            sym_Phi=sym_Phi,
            M_Phi=M_Phi,
            N_Phi=N_Phi,
            R_lmn=R_lmn,
            Z_lmn=Z_lmn,
            modes_R=modes_R,
            modes_Z=modes_Z,
            # NFP=NFP,
            # sym=sym,
            NFP=NFP,
            sym=sym,
            M=M,
            N=N,
            name=name,
            check_orientation=check_orientation,
        )
        # the following part of __init__ uses params_dict, 
        # which loops over all optimizable parameters. 
        # for it to work properly, we must first give a dummy 
        # value to self._aux_dofs_flat.
        self._aux_dofs_flat=jnp.array([])

        # ----- Building the operators for converting plasma Fourier coefficients -----
        (
            self._ptolemy_Phi_A, 
            self._ptolemy_Phi_c_indices, 
            self._ptolemy_Phi_s_indices
        ) = ptolemy_identity_rev_precompute(
            self.Phi_basis.modes[:,1], 
            self.Phi_basis.modes[:,2]
        )
        (
            self._ptolemy_R_plasma_A, 
            self._ptolemy_R_plasma_c_indices, 
            self._ptolemy_R_plasma_s_indices
        ) = ptolemy_identity_rev_precompute(
            eq.surface.R_basis.modes[:,1], 
            eq.surface.R_basis.modes[:,2]
        )
        (
            self._ptolemy_Z_plasma_A, 
            self._ptolemy_Z_plasma_c_indices, 
            self._ptolemy_Z_plasma_s_indices
        ) = ptolemy_identity_rev_precompute(
            eq.surface.Z_basis.modes[:,1], 
            eq.surface.Z_basis.modes[:,2]
        )

        # ----- Building the QUADCOIL callables -----
        # TODO: finish function parsing.
        f_obj, g_obj_list, h_obj_list, aux_dofs_obj = _parse_objectives(
            objective_name=objective_name, 
            objective_unit=objective_unit,
            objective_weight=objective_weight, 
            smoothing=smoothing,
            smoothing_params=smoothing_params,
        )
        g_cons_list, h_cons_list, aux_dofs_cons = _parse_constraints(
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_unit=constraint_unit,
            constraint_value=constraint_value,
            smoothing=smoothing,
            smoothing_params=smoothing_params,
        )
        # Merging constraints and aux dofs from different sources
        g_list = g_obj_list + g_cons_list
        h_list = h_obj_list + h_cons_list
        self._f_quadcoil = lambda qp, x, f_obj=f_obj: f_obj(qp, x)
        self._g_quadcoil = lambda qp, x, g_list=g_list: merge_callables(g_list)(qp, x)
        self._h_quadcoil = lambda qp, x, h_list=h_list: merge_callables(h_list)(qp, x)

        # ----- Pre-computing external-coil-related things -----
        # Now that all transforms are calculated, time to 
        # precompute quantities where applicable.
        # Precomputing transforms
        # eval_grid is used to generate quadrature points. 
        # it is also used to calculate surface Bnormal_plasma
        # when enable_Bnormal_plasma=True, along with surface_grid.
        # because we the quadrature points must be calculated before generating
        # quadcoil callable, it will be constructed here, instead of in the build().
        eval_data_keys = []
        if not self._field:
            eval_data_keys = eval_data_keys + _BCOIL_DATA_KEYS
        if not self._enable_Bnormal_plasma:
            eval_data_keys = eval_data_keys + _BPLASMA_DATA_KEYS
        eval_profiles = get_profiles(eval_data_keys, obj=eq, grid=plasma_grid)
        eval_transforms = get_transforms(eval_data_keys, obj=eq, grid=plasma_grid)
        self._constants['eval_profiles'] = eval_profiles
        self._constants['eval_transforms'] = eval_transforms

        # Precomputing magnetic-field related quantites
        if self._eq_fixed:
            # Plasma dofs
            plasma_surface = self.params_to_quadcoil_plasma_surface(eq.params_dict)
            self._constants['plasma_dofs'] = plasma_surface.dofs
            if plasma_coil_distance is not None:
                winding_dofs = self._winding_surface_generator(
                    plasma_gamma=plasma_surface.gamma(), 
                    d_expand=-self.plasma_coil_distance, # This is DESC's sign convention 
                    nfp=self.NFP, 
                    stellsym=self.sym,
                    mpol=self.M,
                    ntor=self.N,
                )
                self._constants['winding_dofs'] = winding_dofs

            # B plasma
            if self._enable_Bnormal_plasma:
                self._constants['Bnormal_plasma'] = compute_Bnormal_plasma(
                    self._constants, 
                    eq.params_dict, 
                    self._bplasma_chunk_size
                )

            # Part of external field
            if self._field:
                coils_x, coils_n_rho = compute_eval_data_coils(self._constants, eq.params_dict)
                self._constants['coils_x'] = coils_x
                self._constants['coils_n_rho'] = coils_n_rho
                if self._field_fixed:
                    self._constants['Bnormal_ext'] = compute_Bnormal_ext(
                        self._constants, 
                        self._constants['sum_field'].params_dict, # params_field, 
                        self._bs_chunk_size
                    )

        # ----- Initializing auxiliary dofs (slack variables) -----
        # aux_dofs_init is a list of callables that 
        # calculates initial guesses for the slack 
        # variables based on phi and plasma properties
        # If initial guesses for the slack variables 
        # are not provided, we use it to calculate 
        # initial guesses.
        if not aux_dofs_vals:
            aux_dofs_init = aux_dofs_obj | aux_dofs_cons
            phi_init = self.params_to_phi(self.params_dict)
            qp_init = self.quadcoil_params
            # Obtaining the initial values of the slack variables
            for key in aux_dofs_init.keys():
                if callable(aux_dofs_init[key]): 
                    # Callable(qp: QuadcoilParams, dofs: dict, f_unit: float)
                    aux_dofs_vals[key] = aux_dofs_init[key](qp_init, {'phi': phi_init})
                else:
                    try:
                        aux_dofs_vals[key] = jnp.array(aux_dofs_init[key])
                    except:
                        raise TypeError(
                            f'The auxiliary variable {key} is not a callable, '\
                            'and cannot be converted to an array. Its value is: '\
                            f'{str(aux_dofs_init[key])}. This is dur to improper '\
                            'implementation of the physical quantity. Please contact the developers.')
            # For compatibility with params parsing in DESC, we must store aux_dof as a
            # jax numpy array aux_dofs_flat.
        aux_dofs_flat, unravel_aux_dofs = flatten_util.ravel_pytree(aux_dofs_vals)
        self.unravel_aux_dofs = unravel_aux_dofs
        self._aux_dofs_flat = aux_dofs_flat

    def from_quadcoil_kwargs(
        eq, 
        quadcoil_kwargs,
        plasma_M_theta, 
        plasma_N_phi,
        quadcoil_dofs=None,
        field=[],
        field_grid=None,
        enable_net_current_plasma=True,
        eq_fixed=False, # Whether the equilibrium are fixed
        field_fixed=True, # Whether the external fields are fixed
        enable_Bnormal_plasma=False, # Whether to enable free-boundary
        # Args for FourierCurrentPotentialField
        # Phi_mn=np.array([0.0]),
        # modes_Phi=np.array([[0, 0]]),
        name="",
        check_orientation=True,
        # Misc
        bs_chunk_size=None,
        bplasma_chunk_size=None,
    ):
        '''
        An alternative constructor using quadcoil's kwargs, like QuadcoilProxy's constructor.
        '''
        filtered = {}
        source_kwargs = quadcoil_kwargs.copy()
        if 'winding_stellsym' in source_kwargs.keys():
            winding_stellsym = source_kwargs['winding_stellsym']
        else:
            winding_stellsym = eq.sym

        # Reading winding surface information.
        if 'winding_dofs' in source_kwargs.keys():
            winding_dofs = source_kwargs.pop('winding_dofs')
            quadcoil_winding_surface = SurfaceRZFourierJAX(
                nfp=eq.NFP, 
                stellsym=winding_stellsym, 
                mpol=source_kwargs['winding_mpol'], 
                ntor=source_kwargs['winding_ntor'], 
                quadpoints_phi=source_kwargs['winding_quadpoints_phi'], 
                quadpoints_theta=source_kwargs['winding_quadpoints_theta'], 
                dofs=winding_dofs
            )
            filtered['winding_surface'] = quadcoil_winding_surface.to_desc()

        if 'stellsym' in source_kwargs.keys():
            stellsym = source_kwargs.pop('stellsym')
        else:
            stellsym = eq.sym and winding_stellsym
        if stellsym:
            filtered['sym_Phi'] = 'sin'
        else:
            filtered['sym_Phi'] = False
        if 'smoothing' not in source_kwargs.keys():
            filtered['smoothing'] = 'approx'
            filtered['smoothing_params'] = {'lse_epsilon': 1e-3}
        elif source_kwargs['smoothing'] == 'slack':
            warnings.warn(
                'It is not advised to perform single-stage '\
                'optimization using the \'slack\' smoothing mode, because '\
                'DESC may hang when the constraint has large dimensions (~500), '\
                'which is common under the \'slack\' smoothing mode.'
            )
        # Initialize using a Quadcoil initial guess
        if quadcoil_dofs is not None:
            try:
                aux_dofs_vals = quadcoil_dofs.copy()
                phi_pre_flip = aux_dofs_vals.pop('phi')
                mpol = quadcoil_kwargs['mpol']
                ntor = quadcoil_kwargs['ntor']
                m, n = QuadcoilParams.make_mn_helper(mpol, ntor, stellsym)
                phi_flipped = toroidal_flip(phi_pre_flip, m, n)
                Phi_mn, modes_M, modes_N = quadcoil_phi_to_desc_phi(
                    phi_mn_quadcoil=phi_flipped, 
                    stellsym=stellsym, 
                    mpol=mpol, 
                    ntor=ntor
                )
                modes_Phi = np.stack((modes_M, modes_N)).T.astype(np.int32)
                filtered['aux_dofs_vals'] = aux_dofs_vals
                filtered['Phi_mn'] = Phi_mn
                filtered['modes_Phi'] = modes_Phi
                
            except KeyError:
                raise KeyError(
                    'When an initial guess is provided via quadcoil_dofs, '\
                    'mpol and ntor (mode numbers of the current potential) '\
                    'must also be provided.'
                )

        # Renaming arguments (quadcoil and FourierCurrentPotential
        # have some different naming conventions)
        rename_map = {
            'winding_mpol': 'M', 
            'winding_ntor': 'N', 
            'mpol': 'M_Phi', 
            'ntor': 'N_Phi', 
            'net_poloidal_current_amperes': 'G',
            'net_toroidal_current_amperes': 'I',
        }

        # Rename keys as needed
        source_kwargs = {
            rename_map.get(k, k): v
            for k, v in source_kwargs.items()
        }

        # Filter only parameters accepted by target_func
        target_params = inspect.signature(QuadcoilField).parameters
        filtered = filtered | {
            k: v
            for k, v in source_kwargs.items()
            if k in target_params
        }
        discarded_kwargs = [
            k
            for k, v in source_kwargs.items()
            if k not in target_params
        ]
        if discarded_kwargs:
            warnings.warn(
                'The following items in quadcoil_kwargs will be ignored '\
                'because they will be automatically calculated by or has alternative '\
                'definitions in QuadcoilField: '
                + str(discarded_kwargs)
            )

        return QuadcoilField(
            eq=eq,
            # These arguments can all be given by the **kwargs of QUADCOIL.
            # No refinement in plasma grid
            # plasma_quadpoints_phi_interp=1, 
            # plasma_quadpoints_theta_interp=1, 
            # quadpoints_phi=quadcoil_kwargs_temp['quadpoints_phi'],
            # quadpoints_theta=quadcoil_kwargs_temp['quadpoints_theta'],
            # # Winding surface optimization mode 
            # winding_surface=winding_surface.to_desc(),
            # # These are when winding surfaces are not provided
            # plasma_coil_distance=quadcoil_kwargs_temp['plasma_coil_distance'],
            # M=quadcoil_kwargs_temp['mpol'], # Number of poloidal harmonics in the winding surface. Equivalent to mpol in simsopt.
            # N=quadcoil_kwargs_temp['ntor'], # Number of toroidal harmonics in the winding surface. Equivalent to ntor in simsopt.
            # winding_quadpoints_phi=quadcoil_kwargs_temp['winding_quadpoints_phi'],
            # winding_quadpoints_theta=quadcoil_kwargs_temp['winding_quadpoints_theta'],
            # # These are for winding surface optimization
            # winding_surface_generator=quadcoil_kwargs_temp['winding_surface_generator'],
            # # QUADCOIL params
            # objective_name=quadcoil_kwargs_temp['objective_name'],
            # objective_weight=quadcoil_kwargs_temp['objective_weight'],
            # objective_unit=quadcoil_kwargs_temp['objective_unit'],
            # constraint_name=quadcoil_kwargs_temp['constraint_name'],
            # constraint_type=quadcoil_kwargs_temp['constraint_type'],
            # constraint_unit=quadcoil_kwargs_temp['constraint_unit'],
            # constraint_value=quadcoil_kwargs_temp['constraint_value'],
            # # Net currents
            # I=0,
            # G=0,
            # # Phi parameters
            # sym_Phi=False,
            # M_Phi=None,
            # N_Phi=None,
            # # Smoothing params
            # smoothing='approx',
            # smoothing_params={'lse_epsilon': 1e-3},
            # Args for FourierCurrentPotentialField
            # Phi_mn=Phi_mn,
            # modes_Phi=modes_Phi,
            # External coils - no external coils by default
            plasma_M_theta=plasma_M_theta, 
            plasma_N_phi=plasma_N_phi,
            field=field,
            field_grid=field_grid,
            enable_net_current_plasma=enable_net_current_plasma,
            eq_fixed=eq_fixed, # Whether the equilibrium are fixed
            field_fixed=field_fixed, # Whether the external fields are fixed
            enable_Bnormal_plasma=enable_Bnormal_plasma, # Whether to enable free-boundary
            # Args for FourierCurrentPotentialField
            name=name,
            check_orientation=check_orientation,
            # Misc
            bs_chunk_size=bs_chunk_size,
            bplasma_chunk_size=bplasma_chunk_size,
            **filtered
        )

    @property
    def params_dict(self):
        """dict: dictionary of arrays of optimizable parameters."""
        return {
            key: jnp.atleast_1d(jnp.asarray(getattr(self, key))).copy()
            for key in self.optimizable_params
        }
    
    # The surface coeffs in FourierCurrentPotentialField
    # are treated as dofs, but in QUADCOIL the winding surface
    # can also be generated from the plasma surface.
    # This method updates the surface parameter with auto-generated
    # values whenever the user requests them.
    @params_dict.setter
    def params_dict(self, d):
        for key, val in d.items():
            if jnp.asarray(val).size:
                setattr(self, key, val)
        # If winding surface is auto-generated, then 
        # override the winding surface coeffs with the 
        # auto-generated values
        if self.plasma_coil_distance is not None:
            winding_surface = self.quadcoil_winding_surface
            winding_surface_desc = winding_surface.to_desc()
            setattr(self, "R_lmn", winding_surface_desc.R_lmn)
            setattr(self, "Z_lmn", winding_surface_desc.Z_lmn)

    @optimizable_parameter
    @property
    def aux_dofs_flat(self):
        """ Flattened slack variables for QUADCOIL """
        return self._aux_dofs_flat
    
    @aux_dofs_flat.setter
    def aux_dofs_flat(self, new):
        """ Flattened slack variables for QUADCOIL """
        self._aux_dofs_flat = new
    
    @property
    def quadcoil_plasma_surface(self):
        return self.params_to_quadcoil_plasma_surface(self._eq.params_dict)
    
    @property
    def quadcoil_winding_surface(self):
        return self.params_to_quadcoil_winding_surface(self._eq.params_dict, self.params_dict)
    
    @property
    def quadcoil_dofs(self):
        return self.params_to_dofs(self.params_dict)
    
    @property
    def quadcoil_params(self):
        if self._field:
            params_field = self._constants['sum_field'].params_dict
        else:
            params_field = {}
        return self.params_to_qp(self._eq.params_dict, self.params_dict, params_field)
      
    @property
    def plasma_coil_distance(self):
        """ The plasma coil distance, if applicable. """
        return self._plasma_coil_distance
    
    # ----- Logics -----
    # In QUADCOIL, the form of the opimizable depends on the choice of 
    # objectives and constraints. This is because QUADCOIL targets non-smooth
    # objectives and constraints by automatically adding slack variables.
    # Because of this, we decided to put most logics that would usually be in 
    # Objective.build() and Objective.compute() into this Optimizable instead.
    # These includes conversion from DESC params into quadcoil objects 
    # and parsing objective and constraint functions.
    def params_to_phi(self, params_qf):
        Phi_s_raw, Phi_c_raw = ptolemy_identity_rev_compute(self._ptolemy_Phi_A, self._ptolemy_Phi_c_indices, self._ptolemy_Phi_s_indices, params_qf['Phi_mn'])
        # Stellsym SurfaceRZFourier's dofs consists of 
        # [rc, zs]
        # Non-stellsym SurfaceRZFourier's dofs consists of 
        # [rc, rs, zc, zs]
        # Because rs, zs from ptolemy_identity_rev shares the same m, n 
        # arrays as rc, zc, they both have a zero as the first element 
        # that need to be removed.
        Phi_c = Phi_c_raw.flatten()
        Phi_s = Phi_s_raw.flatten()[1:]
        if self.sym_Phi:
            return Phi_s
        else:
            return jnp.concatenate([Phi_c, Phi_s])

    def params_to_dofs(self, params_qf):
        """ Converts params (input to QuadcoilObjective.compute()) into a quadcoil dofs dictionary """
        phi_quadcoil = self.params_to_phi(params_qf)
        # Converting the flattened aux dofs dictionary back into a dict.
        return {'phi': phi_quadcoil} | self.unravel_aux_dofs(params_qf['aux_dofs_flat'])
    
    def params_to_quadcoil_plasma_surface(self, params_eq):
        """ Reads plasma surface info from params and creates a quadcoil.SurfaceRZFourierJAX object """
        # If plasma surface is fixed, then use pre-computed plasma dofs
        if self._eq_fixed and 'plasma_dofs' in self._constants.keys():
            plasma_dofs = self._constants['plasma_dofs']
        else:
            r_s_raw, r_c_raw = ptolemy_identity_rev_compute(self._ptolemy_R_plasma_A, self._ptolemy_R_plasma_c_indices, self._ptolemy_R_plasma_s_indices, params_eq['Rb_lmn'])
            z_s_raw, z_c_raw = ptolemy_identity_rev_compute(self._ptolemy_Z_plasma_A, self._ptolemy_Z_plasma_c_indices, self._ptolemy_Z_plasma_s_indices, params_eq['Zb_lmn'])
                # Stellsym SurfaceRZFourier's dofs consists of 
            # [rc, zs]
            # Non-stellsym SurfaceRZFourier's dofs consists of 
            # [rc, rs, zc, zs]
            # Because rs, zs from ptolemy_identity_rev shares the same m, n 
            # arrays as rc, zc, they both have a zero as the first element 
            # that need to be removed.
            rc = r_c_raw.flatten()
            rs = r_s_raw.flatten()[1:]
            zc = z_c_raw.flatten()
            zs = z_s_raw.flatten()[1:]
            if self._eq.surface.sym:
                plasma_dofs = jnp.concatenate([rc, zs])
            else:
                plasma_dofs = jnp.concatenate([rc, rs, zc, zs])
        
        return SurfaceRZFourierJAX(
            nfp=self.NFP, stellsym=self.sym, 
            mpol=self._eq.surface.M, ntor=self._eq.surface.N, 
            quadpoints_phi=self._plasma_quadpoints_phi, 
            quadpoints_theta=self._plasma_quadpoints_theta,
            dofs=plasma_dofs
        )

    def params_to_quadcoil_winding_surface(self, params_eq, params_qf):
        """ Reads winding surface surface info from params and creates a quadcoil.SurfaceRZFourierJAX object """
        # One mode generates the winding surface automatically from the 
        # plasma surface
        if self.plasma_coil_distance is not None:
            plasma_surface = self.params_to_quadcoil_plasma_surface(params_eq)
            if self._eq_fixed and 'winding_dofs' in self._constants.keys():
                winding_dofs = self._constants['winding_dofs']
            else:
                winding_dofs = self._winding_surface_generator(
                    plasma_gamma=plasma_surface.gamma(), 
                    d_expand=-self.plasma_coil_distance, # This is DESC's sign convention 
                    nfp=self.NFP, 
                    stellsym=self.sym,
                    mpol=self.M,
                    ntor=self.N,
                )
        # Another mode keeps the winding surface as an optimizable
        # (It will also appear in QuadcoilObjective.things)
        else:
            r_s_raw, r_c_raw = ptolemy_identity_rev_compute(self._ptolemy_R_winding_A, self._ptolemy_R_winding_c_indices, self._ptolemy_R_winding_s_indices, params_qf['R_lmn'])
            z_s_raw, z_c_raw = ptolemy_identity_rev_compute(self._ptolemy_Z_winding_A, self._ptolemy_Z_winding_c_indices, self._ptolemy_Z_winding_s_indices, params_qf['Z_lmn'])
                # Stellsym SurfaceRZFourier's dofs consists of 
            # [rc, zs]
            # Non-stellsym SurfaceRZFourier's dofs consists of 
            # [rc, rs, zc, zs]
            # Because rs, zs from ptolemy_identity_rev shares the same m, n 
            # arrays as rc, zc, they both have a zero as the first element 
            # that need to be removed.
            rc = r_c_raw.flatten()
            rs = r_s_raw.flatten()[1:]
            zc = z_c_raw.flatten()
            zs = z_s_raw.flatten()[1:]
            if self.sym:
                winding_dofs = jnp.concatenate([rc, zs])
            else:
                winding_dofs = jnp.concatenate([rc, rs, zc, zs])

        return SurfaceRZFourierJAX(
            nfp=self.NFP, stellsym=self.sym, 
            mpol=self.M, ntor=self.N, 
            quadpoints_phi=self._winding_quadpoints_phi, 
            quadpoints_theta=self._winding_quadpoints_theta,
            dofs=winding_dofs
        )

    def params_to_qp(self, params_eq, params_qf, params_field):
        """ Converts params (input to QuadcoilObjective.compute()) into a quadcoil.QuadcoilParams object"""

        plasma_surface = self.params_to_quadcoil_plasma_surface(params_eq)
        winding_surface = self.params_to_quadcoil_winding_surface(params_eq, params_qf)

        
        # Net fields  
        Bnormal = compute_Bnormal(
            field=self._field, 
            constants=self._constants, 
            Bnormal_shape=self._Bnormal_shape,
            enable_Bnormal_plasma=self._enable_Bnormal_plasma,
            eq_fixed=self._eq_fixed, 
            field_fixed=self._field_fixed, 
            params_eq=params_eq, 
            params_field=params_field, 
            bs_chunk_size=self._bs_chunk_size, 
            bplasma_chunk_size=self._bplasma_chunk_size
        )

        if self._enable_net_current_plasma:
            G = params_qf['G']
            I = params_qf['I']
        else:
            G = 0
            I = 0

        qp_out = QuadcoilParams(
            plasma_surface=plasma_surface, 
            winding_surface=winding_surface, 
            net_poloidal_current_amperes=G, 
            net_toroidal_current_amperes=I,
            Bnormal_plasma=-Bnormal, # Because DESC plasma surface is flipped.
            mpol=self._M_Phi, 
            ntor=self._N_Phi, 
            quadpoints_phi=self._quadpoints_phi,
            quadpoints_theta=self._quadpoints_theta, 
            stellsym=(self.sym_Phi == 'sin')
        )
        return(qp_out)
        