"""Objectives for targeting geometrical quantities."""

import warnings

import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms, rpz2xyz
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import safenorm
from desc.grid import LinearGrid, QuadratureGrid
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective
from .utils import softmin

    
#############################################################################################################################
class HarmonicField_to_BiotSavart(_Objective):
    """Target the difference between a Harmonic field on a flux surface and Biot-Savart.

    Computes Biot-Savart Law to find a harmonic magnetic field on a plasma boundary that 
    encloses net-zero current inside the plasma volume.

    NOTE: By default, assumes the surface is not fixed and its coordinates are computed
    at every iteration, for example if the winding surface you compare to is part of the
    optimization and thus changing.

    Other notes: 
    1. for best results, use this objective in combination with either MeanCurvature
    or PrincipalCurvature, to penalize the tendency for the optimizer to only move the
    points on surface corresponding to the grid that the plasma-vessel distance
    is evaluated at, which can cause cusps or regions of very large curvature.

    2. When use_softmin=True, ensures that alpha*values passed in is
    at least >1, otherwise the softmin will return inaccurate approximations
    of the minimum. Will automatically multiply array values by 2 / min_val if the min
    of alpha*array is <1. This is to avoid inaccuracies that arise when values <1
    are present in the softmin, which can cause inaccurate mins or even incorrect
    signs of the softmin versus the actual min.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    surface : Surface
        Bounding surface to penalize distance to.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Must be broadcastable to Objective.dim_f. Defaults to ``bounds=(1,np.inf)``.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f.
        Defaults to ``bounds=(1,np.inf)``.
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    surface_grid : Grid, optional
        Collocation grid containing the nodes to evaluate surface geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    plasma_grid : Grid, optional
        Collocation grid containing the nodes to evaluate plasma geometry at.
        Defaults to ``LinearGrid(M=eq.M_grid, N=eq.N_grid)``.
    use_softmin: bool, optional
        Use softmin or hard min.
    surface_fixed: bool, optional
        Whether the surface the distance from the plasma is computed to
        is fixed or not. If True, the surface is fixed and its coordinates are
        precomputed, which saves on computation time during optimization, and
        self.things = [eq] only.
        If False, the surface coordinates are computed at every iteration.
        False by default, so that self.things = [eq, surface]
    alpha: float, optional
        Parameter used for softmin. The larger alpha, the closer the softmin
        approximates the hardmin. softmin -> hardmin as alpha -> infinity.
        if alpha*array < 1, the underlying softmin will automatically multiply
        the array by 2/min_val to ensure that alpha*array>1. Making alpha larger
        than this minimum value will make the softmin a more accurate approximation
        of the true min.
    name : str, optional
        Name of the objective function.
    """

    _coordinates = "rtz"
    _units = "(m)"
    _print_value_fmt = "Plasma-vessel distance: {:10.3e} "

    def __init__(
        self,
        eq,
        field,
        curve,
        G,
        surface_grid=None,
        plasma_grid=None,
        curve_grid = None,
        #surface,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        use_softmin=False,
        surface_fixed=False,
        alpha=1.0,
        name="Harmonic Field",
        #vacuum=False,
    ):
        if target is None and bounds is None:
            bounds = (1, np.inf)
        self._eq = eq
        self._field = field
        self._curve = curve
        self._G = G
        #self._surface = surface
        self._surface_grid = surface_grid
        self._plasma_grid = plasma_grid
        self._curve_grid = curve_grid
        self._use_softmin = use_softmin
        self._surface_fixed = surface_fixed
        self._alpha = alpha
        super().__init__(
            #things=[eq, self._surface] if not surface_fixed else [eq],
            things=[eq] if not surface_fixed else [eq],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
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
        eq = self.things[0]
        #plasma_grid = self.things[5]
        
        #surface = self._surface if self._surface_fixed else self.things[1]
        # if things[1] is different than self._surface, update self._surface
        #if surface != self._surface:
        #    self._surface = surface
        
        if self._surface_grid is None:
            surface_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            surface_grid = self._surface_grid
        
        if self._plasma_grid is None:
            plasma_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
        else:
            plasma_grid = self._plasma_grid
        if not np.allclose(surface_grid.nodes[:, 0], 1):
            warnings.warn("Surface grid includes off-surface pts, should be rho=1")
        if not np.allclose(plasma_grid.nodes[:, 0], 1):
            warnings.warn("Plasma grid includes interior points, should be rho=1")
        
        if self._curve_grid is None:
            curve_grid = LinearGrid(M=eq.M_grid, N=0, NFP=eq.NFP)
        else:
            curve_grid = self._curve_grid
            
        self._dim_f = surface_grid.num_nodes
        #self._equil_data_keys = ["R", "phi", "Z"]
        #self._surface_data_keys = ["x"]
        
        # Keys for equilibrium surface are not required since the harmonic fields are computed in the
        # HarmonicSuperPos objective
        self._equil_data_keys = ["R","phi","Z",
                                 #"theta","zeta",
                                 #"n_rho",
                                 #"e^theta_s","e^zeta_s",
                                 #"e^theta_s_t","e^theta_s_z",
                                 #"e^zeta_s_t","e^zeta_s_z",
                                 #"nabla_s^2_theta","nabla_s^2_zeta",
                                ]
        
        self._surface_data_keys = [#"n_rho",
                                   "R","phi","Z",
                                  ]
        
        self._curve_data_keys = [#"n_rho",
                                   "R","Z",
                                  ]
        
        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        equil_transforms = get_transforms(
            self._equil_data_keys,
            obj=eq,
            grid=plasma_grid,
            has_axis=plasma_grid.axis.size,
        )
        
        # compute returns points on the grid of the surface
        # (dim_f = surface_grid.num_nodes)
        # so set quad_weights to the surface grid
        # to avoid it being incorrectly set in the super build
        w = surface_grid.weights
        w *= jnp.sqrt(surface_grid.num_nodes)

        self._constants = {
            #"equil_transforms": equil_transforms,
            #"equil_profiles": equil_profiles,
            #"surface_transforms": surface_transforms,
            "quad_weights": w,
            "G": self._G,
        }

        #if self._surface_fixed:
            # precompute the surface coordinates
            # as the surface is fixed during the optimization
        #    surface_coords = compute_fun(
        #        self._surface,
        #        self._surface_data_keys,
        #        params=self._surface.params_dict,
        #        transforms=surface_transforms,
        #        profiles={},
        #        basis="xyz",
        #    )["x"]
        #    self._constants["surface_coords"] = surface_coords

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

        
    def compute(self, equil_params, surface_params=None, constants=None):
        # In def compute(self) you take self.surf (original surface), create a winding surface wsurf 
        # and define self.CurrentPotential on wsurf
        
        """Compute plasma-surface distance.

        Parameters
        ----------
        equil_params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        surface_params : dict
            Dictionary of surface degrees of freedom, eg Surface.params_dict
            Only needed if self._surface_fixed = False
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        d : ndarray, shape(surface_grid.num_nodes,)
            For each point in the surface grid, approximate distance to plasma.

        """
        
        if constants is None:
            constants = self.constants

        # Solve for isothermal coordinates and find hamonic vector field
        #plasma_coords = rpz2xyz(jnp.array([data["R"], data["phi"], data["Z"]]).T)
        equil_profiles = get_profiles(
            self._equil_data_keys,
            obj=self._eq,
            grid=self._plasma_grid,
            has_axis=self._plasma_grid.axis.size,
        )
        
        equil_transforms = get_transforms(
            self._equil_data_keys,
            obj=self._eq,
            grid=self._plasma_grid,
            jitable=True,
        )
        
        curve_transforms = get_transforms(
            self._curve_data_keys,
            obj=self._curve,
            grid=self._curve_grid,
            jitable=True,
        )
        
        self._constants = {
            "equil_transforms": equil_transforms,
            "equil_profiles": equil_profiles,
            #"surface_transforms": surface_transforms,
        }

        edata = compute_fun(
            self._eq,
            self._equil_data_keys,
            params=equil_params,
            transforms=constants["equil_transforms"],
            profiles=constants["equil_profiles"],
            basis="xyz",
        )
        
        # Generate a winding surface as an offset surface from the plasma surface
        surf_winding = self._eq.surface.constant_offset_surface(offset=5e-2, # desired offset
                                                          M=16, # Poloidal resolution of desired offset surface
                                                          N=8, # Toroidal resolution of desired offset surface
                                                          grid=LinearGrid(M=32,
                                                                          N=16,
                                                                          NFP=self._eq.NFP)
                                                         ) # grid of points on base surface to evaluate unit normal 
                                                        # and find points on offset surface, generally should be 
                                                        # twice the desired 
                
        surface_transforms = get_transforms(
            self._surface_data_keys,
            obj=surf_winding,#self._eq,
            grid=self._surface_grid,
            jitable=True,
        )
        
        #surface_transforms = get_transforms(
        #    self._surface_data_keys,
        #    obj=surf_winding,
        #    grid=plasma_grid,
        #    jitable=True,
        #)
        
        #sdata = compute_fun(
        #    surf_winding._eq,
        #    self._surface_data_keys,
        #    params=surface_params,
        #    transforms=constants["surface_transforms"],
        #    basis="rpz",
        #)
        
        cdata = compute_fun(
            self._curve,
            self._curve_data_keys,
            params=curve_params,
            transforms=constants["curve_transforms"],
            #profiles=constants["equil_profiles"],
            basis="rpz",
        )
        
        # Define transforms for Current Potential
        field = self._field
        if hasattr(field, "Phi_mn"):
            # make the transform for the FourierCurrentPotentialField
            if self._field_grid is None:
                self._field_grid = LinearGrid(
                    M=30 + 2 * max(field.M, field.M_Phi),
                    N=30 + 2 * max(field.N, field.N_Phi),
                    NFP=field.NFP,
                )
                
            field_transforms = get_transforms(
                G = self._G,
                #["K"], 
                #["K", "x"], 
                obj = self._field, grid = self._field_grid,
            )
        else:
            field_transforms = None

        # Compute Biot-Savart's Law
        B_ = self._field.compute_magnetic_field(
            jnp.vstack([edata["R"],edata["phi"],edata["Z"]]).T, #x, # Coordinates on evaluation grid
            surface_grid=constants["field_grid"],
            basis="rpz",
            params=field_params,
            transforms=constants["field_transforms"],
        )
        
        H1 = compute_fun(self._eq,
            ["H1"],
            params=equil_params,
            transforms=constants["surface_transforms"],
            profiles=constants["equil_profiles"],
            basis="xyz",
                        )["H1"]
        
        a = cdata["R"]
        b = cdata["Z"]
        
        H_ = a*H1 + b*cross(edata["n_rho"],H1)#H2
        #find_H1(egrid,edata)
        #H2 = cross(edata["n_rho"],H1)
        
        err = B_ - H_
        
        return dot(err,err)
    
class HarmonicSuperPos(_Objective):
    """Create a superposition of harmonic vectors given a surface.

    This objective is needed when performing stage-two coil optimization on
    a vacuum equilibrium, to avoid the trivial solution of minimizing Bn
    by making the coil currents zero. Instead, this objective ensures
    the coils create the necessary toroidal flux for the equilibrium field.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium for which the toroidal flux will be calculated.
        The Equilibrium is assumed to be held fixed when using this
        objective.
    field : MagneticField
        MagneticField object, the parameters of this will be optimized
        to minimize the objective.
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if bounds is None.
        Defaults to eq.Psi. Must be broadcastable to Objective.dim_f.
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides target.
        Both bounds must be broadcastable to to Objective.dim_f
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to to Objective.dim_f
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    loss_function : {None, 'mean', 'min', 'max'}, optional
        Loss function to apply to the objective values once computed. This function
        is called on the raw compute value, before any shifting, scaling, or
        normalization. Note: has no effect for this objective
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute jacobian matrix, either forward mode or reverse mode AD.
        "auto" selects forward or reverse mode based on the size of the input and output
        of the objective. Has no effect on self.grad or self.hess which always use
        reverse mode and forward over reverse mode respectively.
    field_grid : Grid, optional
        Grid containing the nodes to evaluate field source at on
        the winding surface. (used if e.g. field is a CoilSet or
        FourierCurrentPotentialField). Defaults to the default for the
        given field, see the docstring of the field object for the specific default.
    eval_grid : Grid, optional
        Collocation grid containing the nodes to evaluate the normal magnetic field at
        plasma geometry at. Defaults to a LinearGrid(L=eq.L_grid, M=eq.M_grid,
        zeta=jnp.array(0.0), NFP=eq.NFP).
    name : str, optional
        Name of the objective function.

    """

    _coordinates = "rtz"
    _units = "(Wb)"
    #_print_value_fmt = "Toroidal Flux: {:10.3e} "

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        field_grid=None,
        eval_grid=None,
        name="toroidal-flux",
    ):
        if target is None and bounds is None:
            target = eq.Psi
        self._field = field
        self._field_grid = field_grid
        self._eval_grid = eval_grid
        self._eq = eq

        super().__init__(
            things=[field],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
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
        eq = self._eq
        if self._eval_grid is None:
            eval_grid = LinearGrid(
                L=eq.L_grid, M=eq.M_grid, zeta=jnp.array(0.0), NFP=eq.NFP
            )
            self._eval_grid = eval_grid
        eval_grid = self._eval_grid

        errorif(
            not np.allclose(eval_grid.nodes[:, 2], eval_grid.nodes[0, 2]),
            ValueError,
            "Evaluation grid should be at constant zeta",
        )
        if self._normalize:
            self._normalization = eq.Psi

        # ensure vacuum eq, as is unneeded for finite beta
        #pres = np.max(np.abs(eq.compute("p")["p"]))
        #curr = np.max(np.abs(eq.compute("current")["current"]))
        #warnif(
        #    pres > 1e-8,
        #    UserWarning,
        #    f"Pressure appears to be non-zero (max {pres} Pa), "
        #    + "this objective is unneeded at finite beta.",
        #)
        #warnif(
        #    curr > 1e-8,
        #    UserWarning,
        #    f"Current appears to be non-zero (max {curr} A), "
        #    + "this objective is unneeded at finite beta.",
        #)

        # eval_grid.num_nodes for quad flux cost,
        #self._dim_f = 1
        #timer = Timer()
        #if verbose > 0:
        #    print("Precomputing transforms")
        #timer.start("Precomputing transforms")

        #data = eq.compute(
        #    ["R", "phi", "Z", "|e_rho x e_theta|", "n_zeta"], grid=eval_grid
        #)
        
        #plasma_coords = jnp.array([data["R"], data["phi"], data["Z"]]).T

        field = self._field

        #if hasattr(field, "Phi_mn"):
            # make the transform for the FourierCurrentPotentialField
        #    if self._field_grid is None:
        #        self._field_grid = LinearGrid(
        #            M=30 + 2 * max(field.M, field.M_Phi),
        #            N=30 + 2 * max(field.N, field.N_Phi),
        #            NFP=field.NFP,
        #        )
        #    field_transforms = get_transforms(
        #        ["K", "x"], obj=field, grid=self._field_grid
        #    )
        #else:
        #    field_transforms = None

        self._constants = {
        #    "plasma_coords": plasma_coords,
        #    "equil_data": data,
            "quad_weights": 1.0,
        #    "field": self._field,
            "field_grid": self._field_grid,
            "eval_grid": eval_grid,
        #    "field_transforms": field_transforms,
        }

        #timer.stop("Precomputing transforms")
        #if verbose > 1:
        #    timer.disp("Precomputing transforms")

        #super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, field_params=None, constants=None):
        """Compute superposition of Harmonic Field.

        Parameters
        ----------
        field_params : dict
            Dictionary of field degrees of freedom,
            eg FourierCurrentPotential.params_dict or CoilSet.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : float
            Toroidal flux from coils and external field

        """
        if constants is None:
            constants = self.constants

        data = constants["equil_data"]
        plasma_coords = constants["plasma_coords"]

        bdata = compute_fun(
            self._eq,
            ["n_rho","H_1"],
            params=equil_params,
            transforms=constants["surface_transforms"],
            profiles=constants["equil_profiles"],
            basis="rtz",
            #basis="xyz",
                        )
        
        H_ = bdata["H_1"] + cross(bdata["n_rho"],bdata["H_1"])
        
        #B = constants["field"].compute_magnetic_field(
        #    plasma_coords,
        #    basis="rpz",
        #    source_grid=constants["field_grid"],
        #    params=field_params,
        #    transforms=constants["field_transforms"],
        #)
        #grid = constants["eval_grid"]

        #B_dot_n_zeta = jnp.sum(B * data["n_zeta"], axis=1)

        #Psi = jnp.sum(
        #    grid.spacing[:, 0]
        #    * grid.spacing[:, 1]
        #    * data["|e_rho x e_theta|"]
        #    * B_dot_n_zeta
        #)

        return H_