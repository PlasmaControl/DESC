import numbers
import warnings

import numpy as np
from scipy.constants import mu_0

from desc.backend import jnp, tree_flatten, tree_leaves, tree_map, tree_unflatten
from desc.batching import vmap_chunked
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid, Grid, _Grid
from desc.integrals import compute_B_plasma
from desc.utils import (
    Timer,
    broadcast_tree,
    copy_rpz_periods,
    errorif,
    rpz2xyz,
    safenorm,
    setdefault,
    warnif,
)

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs
from .utils import softmax, softmin

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
        #"_B_plasma_chunk_size",
        #"_bs_chunk_size",
        #"_vacuum",
    ]

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        field, # Field for sinks and sources
        eq, # Equilibrium
        winding_surface, # Winding surface
        iso_data, # Pass a dictionary to this objective with the information about the isothermal coordinates
        N_sum, # Nnumber of terms for the sum in the Jacobi-theta function
        d0, # Regularization radius for Guenther's function
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        source_grid=None,
        eval_grid=None,
        field_grid=None,
        vacuum=False,
        name="Sinks/Sources Quadratic flux",
        jac_chunk_size=None,
        #*,
        #bs_chunk_size=None,
        #B_plasma_chunk_size=None,
        **kwargs,
    ):

        if target is None and bounds is None:
            target = 0
        
        self._source_grid = source_grid # Locations of the cores of the sources/sinks
        self._eval_grid = eval_grid
        self._iso_data = iso_data # Info on isothermal coordinates
        self._eq = eq
        self._field = [field] #if not isinstance(field, list) else field
        self._winding_surface = winding_surface # Array that stores the values of sinks/sources
        self._field_grid = field_grid
        self._N_sum = N_sum
        self._d0 = d0
        
        #self._vacuum = vacuum
        #self._bs_chunk_size = bs_chunk_size
        #self._B_plasma_chunk_size = setdefault(B_plasma_chunk_size, bs_chunk_size)

        #from desc.geometry import FourierRZToroidalSurface
        #errorif(
        #    isinstance(eq, FourierRZToroidalSurface),
        #    TypeError,
        #    "Detected FourierRZToroidalSurface object "
        #    "if attempting to find a QFM surface, please use "
        #    "SurfaceQuadraticFlux objective instead.",
        #)
        
        super().__init__(
            things= [field],
            #[#self._field, #self._sinks_and_sources],
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
        #from desc.magnetic_fields import SumMagneticField
        #from desc.fns_simp import _compute_magnetic_field_from_Current
        from desc.objectives.find_sour import bn_res, iso_coords_interp
        
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
        self._source_keys = ["theta","zeta","e^theta_s","e^zeta_s"] # Info on the winding surface
        
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
        field_grid2 = Grid(nodes = jnp.vstack((field_grid.nodes[:,0],
                                                field_grid.nodes[:,1],
                                                field_grid.nodes[:,2] + (2*jnp.pi/self._winding_surface.NFP)*1)).T
                            )
        field_grid3 = Grid(nodes = jnp.vstack((field_grid.nodes[:,0],
                                               field_grid.nodes[:,1],
                                                field_grid.nodes[:,2] + (2*jnp.pi/self._winding_surface.NFP)*2)).T
                            )

        # Find transforms for the grids on the winding surface
        #source_profiles1 = get_profiles(self._source_keys, obj=self._field, grid=self._field_grid)
        source_transforms1 = get_transforms(self._source_keys, obj=self._winding_surface, grid=self._field_grid, 
            has_axis=field_grid.axis.size,)
        
        #source_profiles2 = get_profiles(self._source_keys, obj=self._field, grid=field_grid2)
        source_transforms2 = get_transforms(self._source_keys, obj=self._winding_surface, grid=field_grid2, 
            has_axis=field_grid2.axis.size,)

        #source_profiles3 = get_profiles(self._source_keys, obj=self._field, grid=field_grid3)
        source_transforms3 = get_transforms(self._source_keys, obj=self._winding_surface, grid=field_grid3, 
            has_axis=field_grid3.axis.size,)

        # Build data on the grids on the winding surface
        field_data1 = compute_fun(
            self._winding_surface,
            self._source_keys,
            params=self._winding_surface.params_dict,
            transforms=source_transforms1,
            profiles={},#source_profiles1,
            #has_axis=field_grid.axis.size,
        )
        
        field_data2 = compute_fun(
            self._winding_surface,
            self._source_keys,
            params=self._winding_surface.params_dict,
            transforms=source_transforms2,
            profiles={},#source_profiles2,
            #has_axis=field_grid2.axis.size,
        )

        field_data3 = compute_fun(
            self._winding_surface,
            self._source_keys,
            params=self._winding_surface.params_dict,
            transforms=source_transforms3,
            profiles={},#source_profiles3,
            #has_axis=field_grid3.axis.size,
        )

        # Now update each of the field_data dicts with the isothermal coordinates
        field_data1 = iso_coords_interp(self._iso_data, field_data1, self._field_grid)#,self._winding_surface)
        field_data2 = iso_coords_interp(self._iso_data, field_data2, field_grid2)#,self._winding_surface)
        field_data3 = iso_coords_interp(self._iso_data, field_data3, field_grid2)#,self._winding_surface)
        
        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")
            
        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T
        
        # pre-compute B_target because we are assuming eq is fixed
        B_target = self._winding_surface.compute_magnetic_field(
            x,
            source_grid = self._field_grid,
            basis="rpz",
            #params=self._winding_surface.params_dict,
            #chunk_size=self._bs_chunk_size,
        )

        self._constants = {
            "eq": eq,
            "winding_surface": self._winding_surface,
            #"field": self._field,#SumMagneticField(self._field),
            "eval_grid": self._eval_grid,
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "B_target": B_target,
            'p_M': field.p_M,
            'p_N': field.p_N,
            'sdata1':sdata1,
            'sdata2':sdata2,
            'sdata3':sdata3,
            'N_sum': N_sum,
            'd0':self._d0,
        }

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, #*field_params, 
                params1,
                #params2 = None,
                #params2 = None,
                constants=None):
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
        #B_plasma = constants["B_plasma"]
        
        #x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # B from sources: B_src
        B_src = bn_res(constants['p_M']*2+1, 
                       constants['p_N']*2+1,  
                       constants['sdata1'],
                       constants['sdata2'],
                       constants['sdata3'],
                       constants['field_grid'], contants['winding_surface'], 
                       params1['x_mn'],
                       #self._sinks_and_sources.x_mn, 
                       constants['N_sum'], constants['d0'], 
                       contants['eq'], 
                       constant['eval_grid'],
                      )
        
        error = B_src - constants['B_target']
        f = jnp.sqrt( jnp.sum(error * error, axis = 1) ) * jnp.sqrt( eval_data["|e_theta x e_zeta|"] )
        return f

class SourceSinkRegularization(_Objective):
    """Target the magnitude of the sources/sinks whene minimizing the error between a target field and the field generated by sources/sinks on a winding surface.

    """

    weight_str = (
        "weight : {float, ndarray}, optional"
        "\n\tWeighting to apply to the Objective, relative to other Objectives."
        "\n\tMust be broadcastable to to ``Objective.dim_f``"
        "\n\tWhen used with QuadraticFlux objective, this acts as the regularization"
        "\n\tparameter (with w^2 = lambda), with 0 corresponding to no regularization."
        "\n\tThe larger this parameter is, the less complex the surface current will "
        "be,\n\tbut the worse the normal field."
    )
    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
        overwrite={"weight": weight_str},
    )
    _static_attrs = _Objective._static_attrs + ["_regularization"]
    _coordinates = "tz"
    _print_value_fmt = "Surface Current Regularization: "

    def __init__(
        self,
        surface_current_field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        jac_chunk_size=None,
        regularization="K",
        source_grid=None,
        name="surface-current-regularization",
    ):
        from desc.magnetic_fields import (
            CurrentPotentialField,
            FourierCurrentPotentialField,
        )

        errorif(
            regularization not in ["K", "Phi", "sqrt(Phi)"],
            ValueError,
            "regularization must be one of ['K', 'Phi', 'sqrt(Phi)'], "
            + f"got {regularization}.",
        )
        if target is None and bounds is None:
            target = 0
        assert isinstance(
            surface_current_field, (CurrentPotentialField, FourierCurrentPotentialField)
        ), (
            "surface_current_field must be a CurrentPotentialField or "
            + f"FourierCurrentPotentialField, instead got {type(surface_current_field)}"
        )
        self._regularization = regularization
        self._surface_current_field = surface_current_field
        self._source_grid = source_grid
        self._units = (
            "(A)"
            if self._regularization == "K"
            else "(A*m)" if self._regularization == "Phi" else "(sqrt(A)*m)"
        )

        super().__init__(
            things=[surface_current_field],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            jac_chunk_size=jac_chunk_size,
            name=name,
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
        from desc.magnetic_fields import FourierCurrentPotentialField

        surface_current_field = self.things[0]
        if isinstance(surface_current_field, FourierCurrentPotentialField):
            M_Phi = surface_current_field._M_Phi
            N_Phi = surface_current_field._N_Phi
        else:
            M_Phi = surface_current_field.M
            N_Phi = surface_current_field.N

        if self._source_grid is None:
            source_grid = LinearGrid(
                M=3 * M_Phi + 1,
                N=3 * N_Phi + 1,
                NFP=surface_current_field.NFP,
            )
        else:
            source_grid = self._source_grid

        if not np.allclose(source_grid.nodes[:, 0], 1):
            warnings.warn("Source grid includes off-surface pts, should be rho=1")

        # source_grid.num_nodes for the regularization cost
        self._dim_f = source_grid.num_nodes
        self._data_keys = ["Phi", "K", "|e_theta x e_zeta|"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        surface_transforms = get_transforms(
            self._data_keys,
            obj=surface_current_field,
            grid=source_grid,
            has_axis=source_grid.axis.size,
        )
        if self._normalize:
            if isinstance(surface_current_field, FourierCurrentPotentialField):
                self._normalization = np.max(
                    [abs(surface_current_field.I) + abs(surface_current_field.G), 1]
                )
            else:  # it does not have I,G bc is CurrentPotentialField
                Phi = surface_current_field.compute("Phi", grid=source_grid)["Phi"]
                self._normalization = np.max([np.mean(np.abs(Phi)), 1])

        self._constants = {
            "surface_transforms": surface_transforms,
            "quad_weights": source_grid.weights * jnp.sqrt(source_grid.num_nodes),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, surface_params=None, constants=None):
        """Compute surface current regularization.

        Parameters
        ----------
        surface_params : dict
            Dictionary of surface degrees of freedom,
            eg FourierCurrentPotential.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            The surface current density magnitude on the source surface.

        """
        if constants is None:
            constants = self.constants

        surface_data = compute_fun(
            self._surface_current_field,
            self._data_keys,
            params=surface_params,
            transforms=constants["surface_transforms"],
            profiles={},
        )

        if self._regularization == "K":
            K = safenorm(surface_data["K"], axis=-1)
        elif self._regularization == "Phi":
            K = jnp.abs(surface_data["Phi"])
        elif self._regularization == "sqrt(Phi)":
            K = jnp.sqrt(jnp.abs(surface_data["Phi"]))
        return K * jnp.sqrt(surface_data["|e_theta x e_zeta|"])