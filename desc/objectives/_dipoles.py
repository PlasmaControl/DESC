import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import Grid, LinearGrid
from desc.objectives.find_dips import bn_res
from desc.utils import Timer, errorif, safenorm, setdefault

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


#####################
# Sources and sinks #
#####################
class DipolesSurfaceQuadraticFlux(_Objective):
    """Target |B_{dip} - B_s| = 0 on a surface.

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
        "_stick_keys",
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
        dt, # Poloidal separation between sources
        dz, # Toroidal separation between sources
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        eval_grid=None,
        field_grid=None,
        name="Dipoles Quadratic flux",
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
        #self._stick_grid = stick_grid

        self._field = field
        self._eq = eq
        self._winding_surface = (
            winding_surface  # Array that stores the values of sinks/sources
        )
        
        self._iso_data = iso_data  # Info on isothermal coordinates

        self._N_sum = N_sum
        self._d0 = d0
        self._dt = dt
        self._dz = dz

        self._bs_chunk_size = bs_chunk_size
        self._B_plasma_chunk_size = setdefault(B_plasma_chunk_size, bs_chunk_size)

        super().__init__(
            things=[field],
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
        from desc.objectives.find_sour_test import iso_coords_interp

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

        field_grid = self._field_grid
        self._data_keys = ["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|"]
        self._source_keys = [
            "theta",
            "zeta",
            "e^theta_s",
            "e^zeta_s",
            'x',
            '|e_theta x e_zeta|',
        ]  # Info on the winding surface
        
        self._stick_keys = ["theta","zeta","e^theta_s","e^zeta_s","x"]

        self._M = field.p_M * 2 + 1
        self._N = field.p_N * 2 + 1
        
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
        # source_profiles1 = get_profiles(self._source_keys, obj=self._field, grid=self._field_grid)
        source_transforms1 = get_transforms(
            self._source_keys,
            obj=self._winding_surface,
            grid=self._field_grid,
            has_axis=field_grid.axis.size,
        )

        # source_profiles2 = get_profiles(self._source_keys, obj=self._field, grid=field_grid2)
        source_transforms2 = get_transforms(
            self._source_keys,
            obj=self._winding_surface,
            grid=field_grid2,
            has_axis=field_grid2.axis.size,
        )

        # source_profiles3 = get_profiles(self._source_keys, obj=self._field, grid=field_grid3)
        source_transforms3 = get_transforms(
            self._source_keys,
            obj=self._winding_surface,
            grid=field_grid3,
            has_axis=field_grid3.axis.size,
        )

        # Build data on the grids on the winding surface
        field_data1 = compute_fun(
            self._winding_surface,
            self._source_keys,
            params=self._winding_surface.params_dict,
            transforms=source_transforms1,
            profiles={},  # source_profiles1,
            # has_axis=field_grid.axis.size,
        )

        field_data2 = compute_fun(
            self._winding_surface,
            self._source_keys,
            params=self._winding_surface.params_dict,
            transforms=source_transforms2,
            profiles={},  # source_profiles2,
            # has_axis=field_grid2.axis.size,
        )

        field_data3 = compute_fun(
            self._winding_surface,
            self._source_keys,
            params=self._winding_surface.params_dict,
            transforms=source_transforms3,
            profiles={},  # source_profiles3,
            # has_axis=field_grid3.axis.size,
        )

        # Now update each of the field_data dicts with the isothermal coordinates
        field_data1 = iso_coords_interp(
            self._iso_data, field_data1, self._field_grid
        )
        
        field_data2 = iso_coords_interp(
            self._iso_data, field_data2, field_grid2
        )
        
        field_data3 = iso_coords_interp(
            self._iso_data, field_data3, field_grid2
        )

        # Now let's generated the grids for the returning currents
        
        # Generate the dipole center grid
        theta = jnp.linspace(
                2 * jnp.pi * (1 / (self._M * 2 + 1)) * 1 / 2,
                2 * jnp.pi * (1 - 1 / (self._M * 2 + 1) * 1 / 2),
                self._M * 2 + 1,
            )
        
        zeta = jnp.linspace(
                    2 * jnp.pi / field.NFP * ( 1 / ( self._N * 2 + 1 ) ) * 1 / 2,
                    2 * jnp.pi / field.NFP * ( 1 - 1 / ( self._N * 2 + 1 ) * 1 / 2 ),
                    self._N * 2 + 1,
                )
            
        assert (self._M * 2 + 1) * (self._N * 2 + 1) == theta.shape[0] * zeta.shape[0], "Check that the dipoles coincide with the number of points on the grid generated"

        # Now let's generate the shifted locations on the theta-zeta plane
        l_zeta = zeta - self._dz/2
        r_zeta = zeta + self._dz/2
        d_theta = theta - self._dt/2
        u_theta = theta + self._dt/2
        
        from .find_sour_test import alt_grid
        # Grids for the sticks (returning currents)
        l_grid = alt_grid(theta,l_zeta)
        r_grid = alt_grid(theta,r_zeta)
        d_grid = alt_grid(d_theta,zeta)
        u_grid = alt_grid(u_theta,zeta)
        
        # Now generate transforms for each shifted grid
        l_transforms = get_transforms(
            self._stick_keys,
            obj=self._winding_surface,
            grid=l_grid,
            has_axis=l_grid.axis.size,
        )

        l_data = compute_fun(
            self._winding_surface,
            self._stick_keys,
            params=self._winding_surface.params_dict,
            transforms=l_transforms,
            profiles={},
        )
        
        r_transforms = get_transforms(
            self._stick_keys,
            obj=self._winding_surface,
            grid=r_grid,
            has_axis=r_grid.axis.size,
        )

        r_data = compute_fun(
            self._winding_surface,
            self._stick_keys,
            params=self._winding_surface.params_dict,
            transforms=r_transforms,
            profiles={},
        )

        d_transforms = get_transforms(
            self._stick_keys,
            obj=self._winding_surface,
            grid=d_grid,
            has_axis=d_grid.axis.size,
        )

        d_data = compute_fun(
            self._winding_surface,
            self._stick_keys,
            params=self._winding_surface.params_dict,
            transforms=d_transforms,
            profiles={},
        )
        
        u_transforms = get_transforms(
            self._stick_keys,
            obj=self._winding_surface,
            grid=u_grid,
            has_axis=u_grid.axis.size,
        )

        u_data = compute_fun(
            self._winding_surface,
            self._stick_keys,
            params=self._winding_surface.params_dict,
            transforms=u_transforms,
            profiles={},
        )

        # Now update each of the field_data dicts with the isothermal coordinates
        l_data = iso_coords_interp(
            self._iso_data, l_data, l_grid
        )
        
        r_data = iso_coords_interp(
            self._iso_data, r_data, r_grid
        )
        
        d_data = iso_coords_interp(
            self._iso_data, d_data, d_grid
        )

        u_data = iso_coords_interp(
            self._iso_data, u_data, u_grid
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
        )
    
        self._constants = {
            "eq": eq,
            "winding_surface": self._winding_surface,
            "eval_grid": self._eval_grid,
            "field_grid": self._field_grid,
            "l_grid": l_grid,
            "r_grid": r_grid,
            "d_grid": d_grid,
            "u_grid": u_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "B_target": B_target,
            "sdata1": field_data1,
            "sdata2": field_data2,
            "sdata3": field_data3,
            "N_sum": self._N_sum,
            "d0": self._d0,
            "dt": self._dt,
            "dz": self._dz,
            'coords':x,
            'iso_data':self._iso_data,
            'l_data':l_data,
            'r_data':r_data,
            'd_data':d_data,
            'u_data':u_data,
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

        # B from sources: B_src
        B_dip = bn_res(
            self._M,
            self._N,
            constants["sdata1"],
            constants["sdata2"],
            constants["sdata3"],
            constants["field_grid"],
            constants["winding_surface"],
            jnp.concatenate((params1["x_mn_poloidal"],params1["x_mn_toroidal"])),
            self._N_sum,
            constants["d0"],
            constants["dt"],
            constants["dz"],
            constants['coords'],
            constants['iso_data'],
            constants['l_data'],
            constants['r_data'],
            constants['d_data'],
            constants['u_data'],
        )
            
        error = B_dip - constants["B_target"]
        f = jnp.sqrt(jnp.sum(error * error, axis=1)) * jnp.sqrt(
            eval_data["|e_theta x e_zeta|"]
        )
        return f


class DipolesRegularization(_Objective):
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
            
        return jnp.sum(params1["x_mn_poloidal"] ** 2 + params1["x_mn_toroidal"] ** 2)