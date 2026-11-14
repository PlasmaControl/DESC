import numpy as np

from desc.backend import jnp
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.integrals import compute_B_plasma
from desc.utils import Timer, errorif, setdefault

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs


class QuadraticFluxPM(_Objective):
    """Target B*n = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_dipole + B_plasma)*n. The equilibrium is kept fixed while the
    field is unfixed.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error
        will be minimized. The equilibrium is kept fixed during the optimization
        with this objective.
    field : MagneticField
        External field produced by dipoles, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
    field_fixed : MagneticField, optional
        External field produced by coils or other source, which will be fixed during
        optimization, acting typically as a supplemental to the toroidal flux that
        dipoles cannot provide.
    source_grid : Grid, optional
        Collocation grid containing the nodes for plasma source terms.
        Default grid is detailed in the docs for ``compute_B_plasma``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=eq.M_grid, N=eq.N_grid,
        NFP=eq.NFP, sym=False)``
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    vacuum : bool
        If true, B_plasma (the contribution to the normal field on the boundary from the
        plasma currents) is set to zero.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    B_plasma_chunk_size : int or None
        Size to split singular integral computation for B_plasma into chunks.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``bs_chunk_size``.
    fixed_mask : array-like of bool, optional
        Boolean mask specifying which dipole parameters are held fixed when
        computing the normal field error. Entries set to True use the
        corresponding values in ``fixed_rho``, while False entries use the
        current optimization parameters. Only applied when using the
        precomputed induction-matrix path. Default is None.
    fixed_rho : array-like, optional
        Raw dipole parameter values to use at positions where ``fixed_mask``
        is True. The effective normalized dipole strengths are obtained
        through ``tanh(fixed_rho)``. Must have the same shape as
        ``fixed_mask`` and must be provided when ``fixed_mask`` is specified.
        Default is None.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _static_attrs = _Objective._static_attrs + [
        "_B_plasma_chunk_size",
        "_bs_chunk_size",
        "_vacuum",
        "_field_fixed",
        "_dipole_fields",
    ]

    _scalar = False
    _linear = False
    _print_value_fmt = "Boundary normal field error: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        eq,
        field,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        source_grid=None,
        eval_grid=None,
        field_grid=None,
        vacuum=False,
        name="Quadratic flux",
        field_fixed=None,
        jac_chunk_size=None,
        use_induction_matrix=True,
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
        fixed_mask=None,
        fixed_rho=None,
        **kwargs,
    ):
        from desc.geometry import FourierRZToroidalSurface

        if target is None and bounds is None:
            target = 0
        self._source_grid = source_grid
        self._eval_grid = eval_grid
        self._eq = eq
        self._field = [field] if not isinstance(field, list) else field
        self._field_grid = field_grid
        self._vacuum = vacuum
        self._field_fixed = field_fixed
        self._bs_chunk_size = bs_chunk_size
        self._B_plasma_chunk_size = setdefault(B_plasma_chunk_size, bs_chunk_size)
        self._fixed_mask = fixed_mask
        self._fixed_rho = fixed_rho
        self._use_induction_matrix = use_induction_matrix

        if fixed_mask is None:
            self._fixed_mask = None
            self._fixed_rho = None
        else:
            errorif(
                fixed_rho is None,
                ValueError,
                "fixed_rho must be supplied when fixed_mask is supplied.",
            )

            fixed_mask = np.asarray(
                fixed_mask,
                dtype=bool,
            ).copy()

            fixed_rho = np.asarray(
                fixed_rho,
            ).copy()

            errorif(
                fixed_mask.shape != fixed_rho.shape,
                ValueError,
                "fixed_mask and fixed_rho must have the same shape.",
            )

            self._fixed_mask = fixed_mask
            self._fixed_rho = fixed_rho
        errorif(
            isinstance(eq, FourierRZToroidalSurface),
            TypeError,
            "Detected FourierRZToroidalSurface object "
            "if attempting to find a QFM surface, please use "
            "SurfaceQuadraticFlux objective instead.",
        )
        super().__init__(
            things=self._field,
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
        from desc.dipole import _Dipole
        from desc.magnetic_fields import SumMagneticField

        eq = self._eq

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

        raw_weights = eval_grid.weights

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

        # pre-compute B_plasma because we are assuming eq is fixed
        Bplasma = (
            jnp.zeros(eval_grid.num_nodes)
            if self._vacuum
            else compute_B_plasma(
                eq,
                eval_grid,
                self._source_grid,
                normal_only=True,
                chunk_size=self._B_plasma_chunk_size,
            )
        )

        if self._fixed_mask is None:
            fixed_mask = None
            fixed_rho = None
        else:
            fixed_mask = jnp.asarray(self._fixed_mask, dtype=bool)
            fixed_rho = jnp.asarray(self._fixed_rho)

        Bcoils_vec = self._field_fixed.compute_magnetic_field(
            jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T,
            source_grid=self._field_grid,
            basis="rpz",
            chunk_size=self._bs_chunk_size,
        )
        Bcoils = jnp.sum(Bcoils_vec * eval_data["n_rho"], axis=-1)

        # ---- induction-matrix fast path ----
        # If EVERY field being optimized here is a fixed-geometry _Dipole/
        # DipoleSet (position/orientation/m0 are not optimizable -- only
        # rho is), B.n at each eval point is exactly linear in each
        # dipole's M0 = m0 * tanh(rho): (B.n)_i = sum_j g_ij * M0_j. g_ij
        # depends only on fixed dipole geometry and eq's (fixed) surface,
        # so it can be built once here and reused every objective call as
        # a single matrix-vector product, instead of re-running the full
        # Biot-Savart sum (dipole_field over every dipole x every eval
        # point x every symmetric/field-period image) on every iteration.
        # See DipoleSet.calc_g for the derivation and a numerical
        # verification against compute_magnetic_field.
        #
        # This is skipped (falls back to the original direct-field path)
        # if ANY field being optimized is not a _Dipole/DipoleSet -- e.g. a
        # real Coil/CoilSet whose geometry DOES change during optimization,
        # for which no fixed induction matrix exists.
        self._dipole_fields = [f for f in self._field if isinstance(f, _Dipole)]
        induction_matrices = None
        if (
            self._use_induction_matrix
            and len(self._dipole_fields) == len(self._field)
            and len(self._dipole_fields) > 0
        ):
            induction_matrices = [
                dipole_field.calc_g(eq, grid=eval_grid)
                for dipole_field in self._dipole_fields
            ]
            if verbose > 0:
                print(
                    "Using precomputed induction matrix for "
                    f"{len(self._dipole_fields)} fixed-geometry dipole field(s)"
                )

        self._constants = {
            "field": SumMagneticField(self._field),
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "B_plasma": Bplasma,
            "B_coils": Bcoils,
            "induction_matrices": induction_matrices,
            "fixed_mask": (None if fixed_mask is None else jnp.asarray(fixed_mask)),
            "fixed_rho": (None if fixed_rho is None else jnp.asarray(fixed_rho)),
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            B0 = scales["B"]

            A0 = jnp.sum(raw_weights * eval_data["|e_theta x e_zeta|"])

            self._normalization = B0 * A0

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, *field_params):
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
        constants = self.constants

        eval_data = constants["eval_data"]
        B_plasma = constants["B_plasma"]
        B_coils = constants["B_coils"]

        induction_matrices = constants.get("induction_matrices")

        if induction_matrices is not None:

            B_ext = jnp.zeros(eval_data["R"].shape[0])

            fixed_mask = constants.get("fixed_mask")
            fixed_rho = constants.get("fixed_rho")

            for g_ij, dipole_field, params in zip(
                induction_matrices,
                self._dipole_fields,
                field_params,
            ):

                rho_raw = jnp.atleast_1d(params["rho"])

                if fixed_mask is not None:
                    rho_raw = jnp.where(
                        fixed_mask,
                        fixed_rho,
                        rho_raw,
                    )

                M0 = dipole_field.m0 * jnp.tanh(rho_raw)
                B_ext = B_ext + g_ij @ M0

        f = (B_ext + B_plasma + B_coils) * jnp.sqrt(eval_data["|e_theta x e_zeta|"])

        return f


class _DipoleObjective(_Objective):
    """Base class for calculating dipole objectives.

    Parameters
    ----------
    dipole : DipoleSet or Dipole
        Dipole(s) for which the data keys will be optimized.
    data_keys : list of str
        Data keys that will be computed/optimized when this class is
        inherited.

    """

    __doc__ = __doc__.rstrip() + collect_docs(coil=True)

    def __init__(
        self,
        dipole,
        data_keys,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name=None,
        jac_chunk_size=None,
    ):
        self._data_keys = data_keys
        self._normalize = normalize
        super().__init__(
            things=[dipole],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
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
        # local import to avoid circular import
        from desc.dipole import DipoleSet, _Dipole

        dipole = self.things[0]
        errorif(
            not isinstance(dipole, _Dipole),
            TypeError,
            f"Expected object of type Dipole or DipoleSet, got {type(dipole)}",
        )

        self._num_dipoles = len(dipole) if isinstance(dipole, DipoleSet) else 1
        self._dim_f = (
            1 if getattr(self, "_reduce_residual", False) else self._num_dipoles
        )

        timer = Timer()
        if verbose > 0:
            print("Precomputing dipole parameters")
        timer.start("Precomputing dipole parameters")

        self._constants = {
            "quad_weights": jnp.ones(self._dim_f),
            "params": {key: getattr(dipole, key) for key in self._data_keys},
        }

        timer.stop("Precomputing dipole parameters")
        if verbose > 1:
            timer.disp("Precomputing dipole parameters")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params):
        """Compute data of dipole(s) for the given data key(s).

        Parameters
        ----------
        params : dict or list of dict
            Dictionary (or list, one per dipole) of the dipole's degrees of
            freedom, e.g. ``rho_tilde``, ``phi``, ``theta``, ``m0``, ...
        constants : dict
            Dictionary of constant data, e.g. precomputed params. Defaults
            to ``self._constants``.

        Returns
        -------
        data : list of dict of ndarray
            Computed data, one dict per dipole (matches
            ``DipoleSet.compute``'s return format).

        """
        constants = self._constants

        dipole = self.things[0]
        data = dipole.compute(
            self._data_keys,
            params=params if params is not None else constants["params"],
        )
        if isinstance(data, dict):
            data = [data]
        return data


class DipoleDiscreteness(_DipoleObjective):
    """Target dipole strengths to have rho=-1,0,1.

    Parameters
    ----------
    dipole : DipoleSet

    """

    _scalar = False
    _reduce_residual = True
    _units = "(dimensionless)"
    _print_value_fmt = "Dipole discreteness: "

    def __init__(
        self,
        dipole,
        target=0,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="rev",
        name="dipole-discreteness",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            bounds = (-np.inf, np.inf)
        super().__init__(
            dipole,
            data_keys=["rho"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
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
        super().build(use_jit=use_jit, verbose=verbose)
        self._normalization = 1.0

    def compute(self, params):
        """Compute dipole discreteness.

        Parameters
        ----------
        params : dict or list of dict
            Dictionary (or list, one per dipole) of the dipole's degrees of
            freedom.

        Returns
        -------
        d : ndarray, shape(num_dipoles,)

        """
        constants = self._constants

        p = params if params is not None else constants["params"]

        rho_raw = jnp.atleast_1d(p["rho"])
        rho_tilde = jnp.tanh(rho_raw)
        d = rho_tilde * (1.0 - rho_tilde**2)

        return jnp.sum(d**2).reshape(1)


class DipoleVolume(_DipoleObjective):
    """Target dipole volume, minimizing rho so some magnets have rho=0.

    Parameters
    ----------
    dipole : DipoleSet

    """

    _scalar = False
    _units = "(dimensionless)"
    _print_value_fmt = "Dipole volume: "

    def __init__(
        self,
        dipole,
        target=0,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="dipole-volume",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        super().__init__(
            dipole,
            data_keys=["rho"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
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
        super().build(use_jit=use_jit, verbose=verbose)
        self._normalization = 1.0

    def compute(self, params):
        """Compute dipole discreteness.

        Parameters
        ----------
        params : dict or list of dict
            Dictionary (or list, one per dipole) of the dipole's degrees of
            freedom.
        constants : dict
            Dictionary of constant data. Defaults to ``self._constants``.

        Returns
        -------
        d : ndarray, shape(num_dipoles,)

        """
        constants = self._constants
        p = params if params is not None else constants["params"]
        rho_raw = jnp.atleast_1d(p["rho"])
        rho_tilde = jnp.tanh(rho_raw)
        eps = 1e-6
        return (rho_tilde**2 + eps**2) ** 0.25
