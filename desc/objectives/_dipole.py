import numpy as np

from desc.backend import jnp, tree_leaves
from desc.compute import get_params
from desc.utils import Timer, errorif

from .normalization import compute_scaling_factors
from .objective_funs import _Objective, collect_docs

class QuadraticFluxPM(_Objective):
    """Target B*n = 0 on LCFS.

    Uses virtual casing to find plasma component of B and penalizes
    (B_coil + B_plasma)*n. The equilibrium is kept fixed while the
    field is unfixed.

    Note: This objective is intended for coil optimization. For finding the surface
    that minimizes the normal field error, use the SurfaceQuadraticFlux objective.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium upon whose surface the normal field error
        will be minimized. The equilibrium is kept fixed during the optimization
        with this objective.
    field : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided equilibrium's surface.
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
        *,
        bs_chunk_size=None,
        B_plasma_chunk_size=None,
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

        Bcoils = np.linalg.norm(self._field_fixed.compute_magnetic_field(
                            jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T,
                            source_grid=self._field_grid,
                            basis="rpz",
                            chunk_size=self._bs_chunk_size), axis=1)


        self._constants = {
            "field": SumMagneticField(self._field),
            "field_grid": self._field_grid,
            "quad_weights": w,
            "eval_data": eval_data,
            "eval_transforms": eval_transforms,
            "eval_profiles": eval_profiles,
            "B_plasma": Bplasma,
            "B_coils": Bcoils,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["B"] * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, *field_params, constants=None):
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
        B_plasma = constants["B_plasma"]
        B_coils = constants["B_coils"]

        x = jnp.array([eval_data["R"], eval_data["phi"], eval_data["Z"]]).T

        # B_ext is not pre-computed because field is not fixed
        B_ext = constants["field"].compute_magnetic_field(
            x,
            source_grid=constants["field_grid"],
            basis="rpz",
            params=field_params,
            chunk_size=self._bs_chunk_size,
        )
        B_ext = jnp.sum(B_ext * eval_data["n_rho"], axis=-1)
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

        dipoles = dipole.dipoles if isinstance(dipole, DipoleSet) else [dipole]
        self._num_dipoles = len(dipoles)
        self._dim_f = self._num_dipoles

        timer = Timer()
        if verbose > 0:
            print("Precomputing dipole parameters")
        timer.start("Precomputing dipole parameters")

        self._constants = {
            "quad_weights": jnp.ones(self._num_dipoles),

            "params": [get_params(self._data_keys, dip) for dip in dipoles],
        }

        timer.stop("Precomputing dipole parameters")
        if verbose > 1:
            timer.disp("Precomputing dipole parameters")

        # if self._normalize:
        #     # NOTE: compute_scaling_factors is written for coils/equilibria.
        #     # If it doesn't have a code path for _Dipole, replace this with
        #     # an explicit scale (e.g. based on m0) -- see DipoleDiscreteness
        #     # below for a subclass that just turns normalization off, since
        #     # its output is already dimensionless.
        #     self._scales = [compute_scaling_factors(dip) for dip in dipoles]

        super().build(use_jit=use_jit, verbose=verbose)


    def compute(self, params, constants=None):
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
        if constants is None:
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
    """Target dipole strengths to be maximally "on" or "off"

    Parameters
    ----------
    dipole : DipoleSet

    """

    _scalar = False
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
        deriv_mode="auto",
        name="dipole-discreteness",
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

    def compute(self, params, constants=None):
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
        data = super().compute(params, constants=constants)
        rho_raw = jnp.asarray([jnp.atleast_1d(d["rho"])[0] for d in data])
        rho_tilde = jnp.tanh(rho_raw)
        return jnp.abs(rho_tilde) * (1 - jnp.abs(rho_tilde))



# class DipoleVolume(_DipoleObjective):
#     return jnp.abs(rho_tilde)

    