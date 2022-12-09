"""Objectives for targeting geometrical quantities."""

from desc.backend import jnp
from desc.compute import (
    compute_covariant_metric_coefficients,
    compute_geometry,
    compute_jacobian,
    data_index,
)
from desc.compute.utils import compress, surface_integrals
from desc.grid import LinearGrid, QuadratureGrid
from desc.transform import Transform
from desc.utils import Timer

from .normalization import compute_scaling_factors
from .objective_funs import _Objective


class AspectRatio(_Objective):
    """Aspect ratio = major radius / minor radius.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Aspect ratio: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=2,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="aspect ratio",
    ):

        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self.grid is None:
            self.grid = QuadratureGrid(
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["V"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["V"]["R_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, **kwargs):
        """Compute aspect ratio.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        AR : float
            Aspect ratio, dimensionless.

        """
        data = compute_geometry(R_lmn, Z_lmn, self._R_transform, self._Z_transform)
        return self._shift_scale(data["R0/a"])


class Elongation(_Objective):
    """Elongation = semi-major radius / semi-minor radius.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
        Note: has no effect for this objective.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
        Note: has no effect for this objective.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False
    _units = "(dimensionless)"
    _print_value_fmt = "Elongation: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=False,
        normalize_target=False,
        grid=None,
        name="elongation",
    ):

        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )
        self._print_value_fmt = "Elongation: {:10.3e} (dimensionless)"

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self.grid is None:
            self.surf_grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
            self.vol_grid = QuadratureGrid(
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )
        elif isinstance(self.grid, LinearGrid):
            self.surf_grid = self.grid
            self.vol_grid = QuadratureGrid(
                L=eq.L_grid, M=self.grid.M, N=self.grid.N, NFP=eq.NFP
            )
        elif isinstance(self.grid, QuadratureGrid):
            self.vol_grid = self.grid
            self.surf_grid = LinearGrid(M=self.grid.M, N=self.grid.N, NFP=eq.NFP)
        else:
            raise ValueError("Grid of type {} is not allowed.".format(type(self.grid)))

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._R_transform_surf = Transform(
            self.surf_grid, eq.R_basis, derivs=data_index["V"]["R_derivs"], build=True
        )
        self._Z_transform_surf = Transform(
            self.surf_grid, eq.Z_basis, derivs=data_index["V"]["R_derivs"], build=True
        )
        self._R_transform_vol = Transform(
            self.vol_grid, eq.R_basis, derivs=data_index["V"]["R_derivs"], build=True
        )
        self._Z_transform_vol = Transform(
            self.vol_grid, eq.Z_basis, derivs=data_index["V"]["R_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, **kwargs):
        """Compute elongation.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        elongation : float
            Elongation, dimensionless.

        """
        surf_data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform_surf, self._Z_transform_surf
        )
        vol_data = compute_jacobian(
            R_lmn, Z_lmn, self._R_transform_vol, self._Z_transform_vol
        )

        P = compress(  # perimeter
            self.surf_grid,
            surface_integrals(
                self.surf_grid, jnp.sqrt(surf_data["g_tt"]), surface_label="zeta"
            ),
            surface_label="zeta",
        )
        A = compress(  # area
            self.vol_grid,
            surface_integrals(
                self.vol_grid,
                jnp.abs(vol_data["sqrt(g)"] / vol_data["R"]),
                surface_label="zeta",
            ),
            surface_label="zeta",
        )

        # derived from Ramanujan approximation for the perimeter of an ellipse
        a = (
            jnp.sqrt(3)
            * (
                jnp.sqrt(8 * jnp.pi * A + P**2)
                + jnp.sqrt(
                    2 * jnp.sqrt(3) * P * jnp.sqrt(8 * jnp.pi * A + P**2)
                    - 40 * jnp.pi * A
                    + 4 * P**2
                )
            )
            + 3 * P
        ) / (12 * jnp.pi)
        b = A / (jnp.pi * a)
        elongation = jnp.max(a / b)

        return self._shift_scale(elongation)


class Volume(_Objective):
    """Plasma volume.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective.
        len(target) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target should be normalized before comparing to computed values.
        if `normalize` is `True` and the target is in physical units, this should also
        be set to True.
    grid : Grid, ndarray, optional
        Collocation grid containing the nodes to evaluate at.
    name : str
        Name of the objective function.

    """

    _scalar = True
    _linear = False
    _units = "(m^3)"
    _print_value_fmt = "Plasma volume: {:10.3e} "

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        normalize=True,
        normalize_target=True,
        grid=None,
        name="volume",
    ):

        self.grid = grid
        super().__init__(
            eq=eq,
            target=target,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

    def build(self, eq, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if self.grid is None:
            self.grid = QuadratureGrid(
                L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["V"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["V"]["R_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["V"]

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, R_lmn, Z_lmn, **kwargs):
        """Compute plasma volume.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).

        Returns
        -------
        V : float
            Plasma volume (m^3).

        """
        data = compute_geometry(R_lmn, Z_lmn, self._R_transform, self._Z_transform)
        return self._shift_scale(data["V"])
