import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid
from desc.objectives.find_sour_test import bn_res
from desc.utils import Timer, errorif, safenorm, setdefault

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs

from .sources_dipoles_utils import interp_grid, alt_grid, iso_coords_interp, compute_mask

#####################
# Sources and sinks #
#####################
class SinksSourcesSurfaceQuadraticFlux(_Objective):
    """Target |B_{src} - B_s| = 0 on a surface.

    Used to find a quadratic-flux-minimizing (QFM) surface, so a
    `FourierRZToroidalSurface` should be passed to the objective.
    Should always be used along with a ``ToroidalFlux`` or ``Volume`` objective to
    ensure that the resulting QFM surface has the desired amount of
    flux enclosed and avoid trivial solutions.

    Note: Winding Surface is fixed, equilibrium is fixed.

    Parameters
    ----------
    surface : FourierRZToroidalSurface
        QFM surface upon which the normal field error will be minimized.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided QFM surface. May be fixed
        by passing in ``field_fixed=True``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=surface.M_grid,``
        ``N=surface.N_grid, NFP=surface.NFP, sym=False)``
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    field_fixed : bool
        Whether or not to fix the magnetic field's DOFs during the optimization.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_data_keys",
        "_field_keys",
        "_source_keys",
        "_contour_keys",
        "_stick_keys",
        "_iso_keys",
        "_field_keys_mod",
        "_source_keys_mod",
        '_iso_keys_mod',
        "_N",
        "_M",
        "_N_sum",
        "_eq",
    ]

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
            self,
            field,  # Field for sinks and sources
            eq,  # Equilibrium
            winding_surface,  # Winding surface
            iso_data,  # Pass a dictionary to this objective with the information about the isothermal coordinates
            N_sum,  # Nnumber of terms for the sum in the Jacobi-theta function
            d0,  # Regularization radius for Guenther's function
            target=None,
            bounds=None,
            weight=1,
            normalize=True,
            normalize_target=True,
            eval_grid=None,
            field_grid=None,
            countour_grid = None,
            stick_grid = None,
        #dt = 0, # Poloidal separation between sources
        #dz = 0, # Toroidal separation between sources
        #iso_name, # String with the location of the directory that stores the info of isothermal coordinates
        name="Sinks/Sources Quadratic flux",
        jac_chunk_size=None,
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
        **kwargs,
    ):

        if target is None and bounds is None:
            target = 0

        #self._source_grid = source_grid  # Locations of the cores of the sources/sinks
        self._eval_grid = eval_grid
        self._field_grid = field_grid
        self._contour_grid = eval_grid
        self._stick_grid = stick_grid

        self._field = field
        self._eq = eq
        self._winding_surface = (
            winding_surface
        )
        
        self._iso_data = iso_data  # Info on isothermal coordinates
        
        self._N_sum = N_sum
        self._d0 = d0

        self._bs_chunk_size = bs_chunk_size
        self._B_plasma_chunk_size = setdefault(B_plasma_chunk_size, bs_chunk_size)

        super().__init__(
            things=[field],
            # [#self._field, #self._sinks_and_sources],
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
        
        from desc.objectives.sources_dipoles_utils import iso_coords_interp

        eq = self._eq
        field = self._field

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=False,
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid
            
        self._M = field.p_M# * 2 + 1
        self._N = field.p_N# * 2 + 1
        
        field_grid = self._field_grid
        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|"]
        self._field_keys = [
            "theta", "zeta",
            "e^theta_s", "e^zeta_s",
            'x',
            '|e_theta x e_zeta|',
        ]  # Info on the winding surface
        
        self._contour_keys = ["theta", "zeta", "e_theta",'x']
        self._stick_keys = ["theta", "x"]
        self._source_keys = ["theta", "zeta", "e^theta_s", "e^zeta_s",
                             "x",]

        self._iso_keys = ["u_iso", "v_iso", 
                          #"Phi_iso", "Psi_iso",
                          #"b_iso","lambda_ratio",
                          "omega_1", "omega_2","tau","tau_1","tau_2",
                          "lambda_iso", "lambda_u","lambda_v",
                          #"u_t","u_z","v_t","v_z",
                          #"e^u_s","e^v_s",
                          "e_u", "e_v",
                          'w',
                         ]
        
        #self._field_keys_mod = self._field_keys + self._iso_keys
        #self._source_keys_mod = self._source_keys + self._iso_keys
        
        self._field_keys_mod = self._field_keys + self._iso_keys
        self._source_keys_mod = self._source_keys + self._iso_keys
        self._iso_keys_mod = ['theta', 'zeta','u_iso','v_iso',
                              'Phi_iso', 'Psi_iso',
                              'b_iso', 'lambda_ratio',
                              'omega_1', 'omega_2',
                              'tau', 'tau_1', 'tau_2',
                              'lambda_iso_u','lambda_iso_v',
                              'u_t', 'u_z', 'v_t', 'v_z']
        
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)

        eval_data = compute_fun(
            eq,
            self._data_keys,
            params=eq.params_dict,
            transforms=eval_transforms,
            profiles=eval_profiles,
        )

        # Define other grids
        field_grid2 = Grid(
            nodes=jnp.vstack(
                (
                    field_grid.nodes[:, 0],
                    field_grid.nodes[:, 1],
                    field_grid.nodes[:, 2] + (2 * jnp.pi / field.NFP) * 1,
                )
            ).T
        )
        
        field_grid3 = Grid(
            nodes=jnp.vstack(
                (
                    field_grid.nodes[:, 0],
                    field_grid.nodes[:, 1],
                    field_grid.nodes[:, 2] + (2 * jnp.pi / field.NFP) * 2,
                )
            ).T
        )

        
        # Find transforms for the grids on the winding surface
        field_transforms1 = get_transforms(
            self._field_keys,
            obj=self._winding_surface,
            grid=self._field_grid,
            has_axis=field_grid.axis.size,
        )

        # source_profiles2 = get_profiles(self._source_keys, obj=self._field, grid=field_grid2)
        field_transforms2 = get_transforms(
            self._field_keys,
            obj=self._winding_surface,
            grid=field_grid2,
            has_axis=field_grid2.axis.size,
        )

        # source_profiles3 = get_profiles(self._source_keys, obj=self._field, grid=field_grid3)
        field_transforms3 = get_transforms(
            self._field_keys,
            obj=self._winding_surface,
            grid=field_grid3,
            has_axis=field_grid3.axis.size,
        )

        # Build data on the grids on the winding surface
        field_data1 = compute_fun(
            self._winding_surface,
            self._field_keys,
            params=self._winding_surface.params_dict,
            transforms=field_transforms1,
            profiles={},  # source_profiles1,
            # has_axis=field_grid.axis.size,
        )

        field_data2 = compute_fun(
            self._winding_surface,
            self._field_keys,
            params=self._winding_surface.params_dict,
            transforms=field_transforms2,
            profiles={},
        )

        field_data3 = compute_fun(
            self._winding_surface,
            self._field_keys,
            params=self._winding_surface.params_dict,
            transforms=field_transforms3,
            profiles={},
        )

        # Now update each of the field_data dicts with the isothermal coordinates
        field_data1 = iso_coords_interp(
            self._iso_data, field_data1, self._field_grid
        )
        
        field_data2 = iso_coords_interp(
            self._iso_data, field_data2, field_grid2
        )
        
        field_data3 = iso_coords_interp(
            self._iso_data, field_data3, field_grid3
        )


        theta_sources = jnp.linspace(
            2 * jnp.pi * (1 / (self._M * 2 + 1)) * 1 / 2,
            2 * jnp.pi * (1 - 1 / (self._M * 2 + 1) * 1 / 2),
            self._M * 2 + 1,
        )
    
        zeta_sources = jnp.linspace(
            2 * jnp.pi / self._winding_surface.NFP * (1 / ( self._N * 2 + 1 )) * 1 / 2,
            2 * jnp.pi / self._winding_surface.NFP * (1 - 1 / ( self._N * 2 + 1 ) * 1 / 2),
            self._N * 2+1,
        )

        #import pdb
        #pdb.set_trace()
        ss_grid = alt_grid(theta_sources, zeta_sources)
        
        source_transforms = get_transforms(
            self._source_keys,
            obj=self._winding_surface,
            grid=ss_grid,#self._source_grid,
            has_axis=ss_grid.axis.size,
        )
        # Find info of the isothermal coordinates at the locations of the sources
        ss_data = compute_fun(
            self._winding_surface,
            self._source_keys,
            params=self._winding_surface.params_dict,
            transforms=source_transforms,
            profiles={},
        )
        
        ss_data = iso_coords_interp(self._iso_data, ss_data, ss_grid)
        
        assert (self._M * 2 + 1) * (self._N * 2 + 1) == ss_data["theta"].shape[0], "Check that the sources coincide with the number of sources/sinks"
    
        ####################################
        # Contours
        contour_transforms = get_transforms(
            self._contour_keys,
            obj=self._winding_surface,
            grid=self._contour_grid,
            has_axis=self._contour_grid.axis.size,
        )

        contour_data = compute_fun(
            self._winding_surface,
            self._contour_keys,
            params=self._winding_surface.params_dict,
            transforms=contour_transforms,
            profiles={},  # source_profiles1,
            # has_axis=field_grid.axis.size,
        )

        # Build the matrix for K from contours
        sign_vals = jnp.where(contour_data["theta"] < jnp.pi, -1, 1) #+ jnp.where(ss_data["theta"] > jnp.pi, 1, 0)
        A = compute_mask(contour_data, theta_sources, zeta_sources)
        AA = A[:, None, :] * contour_data['e_theta'][:, :, None]
        AAA = AA * ( jnp.sum(contour_data["e_theta"] * contour_data["e_theta"], axis = 1 ) ** (-1 / 2) * sign_vals )[:, None, None]
        
        ########################################
        # Sticks
        stick_transforms = get_transforms(
            self._stick_keys,
            obj=self._winding_surface,
            grid=self._stick_grid,
            has_axis=self._stick_grid.axis.size,
        )

        stick_data = compute_fun(
            self._winding_surface,
            self._stick_keys,
            params=self._winding_surface.params_dict,
            transforms=stick_transforms,
            profiles={},  # source_profiles1,
            # has_axis=field_grid.axis.size,
        )
        
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # pre-compute B_target because we are assuming eq is fixed
        B_target = self._winding_surface.compute_magnetic_field(
            x,
            source_grid=self._field_grid,
            basis="rpz",
            chunk_size=self._bs_chunk_size,
            # params=self._winding_surface.params_dict,
        )
    
        self._constants = {
            "eq": eq,
            "winding_surface": self._winding_surface,
            # "field": self._field,#SumMagneticField(self._field),
            "eval_grid": self._eval_grid,
            "field_grid": self._field_grid,
            "contour_grid": self._contour_grid,
            "stick_grid": self._stick_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "B_target": B_target,
            "sdata1": field_data1,
            "sdata2": field_data2,
            "sdata3": field_data3,
            "contour_data": contour_data,
            "stick_data": stick_data,
            "N_sum": self._N_sum,
            "d0": self._d0,
            'iso_data':self._iso_data,
            'source_data': ss_data,
            'coords':x,
            #'theta_coarse':theta_sources,
            #'zeta_coarse':zeta_sources,
            'AAA':AAA,
        }

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(
        self,
        params1,
        constants=None,
    ):
        """Compute normal field error on boundary.

        Parameters
        ----------
        field_params : dict
            Dictionary of the external field's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Bnorm from B_ext and B_plasma

        """
        if constants is None:
            constants = self.constants

        # B_plasma from equilibrium precomputed
        eval_data = constants["eval_data"]
        #import pdb
        #pdb.set_trace()

        #sdata1_arrays = {key: constants["sdata1"][key] for key in self._field_keys}
        #sdata2_arrays = {key: constants["sdata2"][key] for key in self._field_keys}
        #sdata3_arrays = {key: constants["sdata3"][key] for key in self._field_keys}
        sdata1_arrays = {key: constants["sdata1"][key] for key in self._field_keys_mod}
        sdata2_arrays = {key: constants["sdata2"][key] for key in self._field_keys_mod}
        sdata3_arrays = {key: constants["sdata3"][key] for key in self._field_keys_mod}
        
        #ss_data_arrays = {key: constants["source_data"][key] for key in self._source_keys}
        ss_data_arrays = {key: constants["source_data"][key] for key in self._source_keys_mod}
        
        contour_data_arrays = {key: constants["contour_data"][key] for key in self._contour_keys}
        stick_data_arrays = {key: constants["stick_data"][key] for key in self._stick_keys}

        # Define expected keys for iso_data

        # In compute method
        iso_data_arrays = {key: constants["iso_data"][key] for key in self._iso_keys_mod}
        #iso_data_arrays = {key: constants["iso_data"][key] for key in constants["iso_data"]}#self._stick_keys}
        
        # B from sources: B_src
        B_src = bn_res(
            #self._M,
            #self._N,
            sdata1_arrays,
            sdata2_arrays,
            sdata3_arrays,
            #constants["sdata1"],
            #constants["sdata2"],
            #constants["sdata3"],
            constants["field_grid"],
            constants["winding_surface"],
            params1["x_mn"],
            self._N_sum,
            constants["d0"],
            constants['coords'],
            iso_data_arrays,
            #constants['iso_data'],
            contour_data_arrays,
            stick_data_arrays,
            #constants["contour_data"],
            #constants["stick_data"],
            constants["contour_grid"],
            ss_data_arrays,
            #constants["source_data"],
            #constants['theta_coarse'],
            #constants['zeta_coarse'],
            constants['AAA'],
        )
        
        error = B_src - constants["B_target"]
        f = jnp.sqrt(jnp.sum(error * error, axis=1)) * jnp.sqrt(
            eval_data["|e_theta x e_zeta|"]
        )
        return f


# Sum of sinks and sources
class SinksSourcesSum2(_Objective):
    """Target Sum(Sources + Sinks) = 0 on a surface.

    Used to find a quadratic-flux-minimizing (QFM) surface, so a
    `FourierRZToroidalSurface` should be passed to the objective.
    Should always be used along with a ``ToroidalFlux`` or ``Volume`` objective to
    ensure that the resulting QFM surface has the desired amount of
    flux enclosed and avoid trivial solutions.

    Note: Winding Surface is fixed, equilibrium is fixed.

    Parameters
    ----------
    surface : FourierRZToroidalSurface
        QFM surface upon which the normal field error will be minimized.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided QFM surface. May be fixed
        by passing in ``field_fixed=True``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=surface.M_grid,``
        ``N=surface.N_grid, NFP=surface.NFP, sym=False)``
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    field_fixed : bool
        Whether or not to fix the magnetic field's DOFs during the optimization.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_data_keys",
        "_source_keys",
        "_N",
        "_M",
        "_N_sum",
        "_eq",
    ]

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        field,  # Field for sinks and sources
        eq,  # Equilibrium
        #winding_surface,  # Winding surface
        #iso_data,  # Pass a dictionary to this objective with the information about the isothermal coordinates
        #N_sum,  # Nnumber of terms for the sum in the Jacobi-theta function
        #d0,  # Regularization radius for Guenther's function
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        #source_grid=None,
        eval_grid=None,
        field_grid=None,
        name="Sinks/Sources Quadratic flux",
        jac_chunk_size=None,
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
        **kwargs,
    ):

        if target is None and bounds is None:
            target = 0

        #self._source_grid = source_grid  # Locations of the cores of the sources/sinks
        self._eval_grid = eval_grid
        #self._iso_data = iso_data  # Info on isothermal coordinates
        self._eq = eq
        self._field = field
        #self._winding_surface = (
        #    winding_surface  # Array that stores the values of sinks/sources
        #)
        self._field_grid = field_grid
        #self._N_sum = N_sum
        #self._d0 = d0

        self._bs_chunk_size = bs_chunk_size
        self._B_plasma_chunk_size = setdefault(B_plasma_chunk_size, bs_chunk_size)

        super().__init__(
            things=[field],
            # [#self._field, #self._sinks_and_sources],
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
        # from desc.magnetic_fields import SumMagneticField
        # from desc.fns_simp import _compute_magnetic_field_from_Current
        from desc.objectives.find_sour import iso_coords_interp

        eq = self._eq
        field = self._field

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=False,
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid

        #field_grid = self._field_grid
        
        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|"]
        #self._source_keys = [
        #    "theta",
        #    "zeta",
        #    "e^theta_s",
        #    "e^zeta_s",
        #    'x',
        #    '|e_theta x e_zeta|',
        #]  # Info on the winding surface

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")
        
        self._constants = {
            "eq": eq,
            "quad_weights": w,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
        }

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(
        self,
        params1,
        constants=None,
    ):
        """Compute normal field error on boundary.

        Parameters
        ----------
        field_params : dict
            Dictionary of the external field's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Bnorm from B_ext and B_plasma

        """
        if constants is None:
            constants = self.constants

        # B_plasma from equilibrium precomputed
        #eval_data = constants["eval_data"]

        #return f
        return jnp.sum(params1["x_mn"])#* jnp.sqrt( eval_data["|e_theta x e_zeta|"] )


class SinksSourcesRegularization(_Objective):
    """Target Sum(Sources^2 + Sinks^2) = 0 on a surface.

    Used to find a quadratic-flux-minimizing (QFM) surface, so a
    `FourierRZToroidalSurface` should be passed to the objective.
    Should always be used along with a ``ToroidalFlux`` or ``Volume`` objective to
    ensure that the resulting QFM surface has the desired amount of
    flux enclosed and avoid trivial solutions.

    Note: Winding Surface is fixed, equilibrium is fixed.

    Parameters
    ----------
    surface : FourierRZToroidalSurface
        QFM surface upon which the normal field error will be minimized.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided QFM surface. May be fixed
        by passing in ``field_fixed=True``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=surface.M_grid,``
        ``N=surface.N_grid, NFP=surface.NFP, sym=False)``
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    field_fixed : bool
        Whether or not to fix the magnetic field's DOFs during the optimization.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_data_keys",
        "_source_keys",
        "_N",
        "_M",
        "_N_sum",
        "_eq",
    ]

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        field,  # Field for sinks and sources
        eq,  # Equilibrium
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        #source_grid=None,
        eval_grid=None,
        field_grid=None,
        sink_source_grid = None,
        stick_grid = None,
        contour_grid = None,
        name="Sinks/Sources Quadratic flux",
        jac_chunk_size=None,
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
        **kwargs,
    ):

        if target is None and bounds is None:
            target = 0

        self._eval_grid = eval_grid
        self._eq = eq
        self._field = field
        self._field_grid = field_grid

        self._bs_chunk_size = bs_chunk_size
        self._B_plasma_chunk_size = setdefault(B_plasma_chunk_size, bs_chunk_size)

        super().__init__(
            things=[field],
            # [#self._field, #self._sinks_and_sources],
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
        #from desc.objectives.find_sour import iso_coords_interp

        eq = self._eq
        field = self._field

        if self._eval_grid is None:
            eval_grid = LinearGrid(
                rho=np.array([1.0]),
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=False,
            )
            self._eval_grid = eval_grid
        else:
            eval_grid = self._eval_grid
            
        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = eval_grid.num_nodes

        w = eval_grid.weights
        w *= jnp.sqrt(eval_grid.num_nodes)

        eval_profiles = get_profiles(self._data_keys, obj=eq, grid=eval_grid)
        eval_transforms = get_transforms(self._data_keys, obj=eq, grid=eval_grid)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")
        
        self._constants = {
            "eq": eq,
            "quad_weights": w,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
        }

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(
        self,
        params1,
        constants=None,
    ):
        """Compute normal field error on boundary.

        Parameters
        ----------
        field_params : dict
            Dictionary of the external field's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Bnorm from B_ext and B_plasma

        """
        if constants is None:
            constants = self.constants
            
        return jnp.sum(params1["x_mn"] ** 2)#* jnp.sqrt( eval_data["|e_theta x e_zeta|"] )