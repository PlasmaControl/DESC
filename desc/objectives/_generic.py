from scipy.constants import mu_0
from inspect import signature

from desc.backend import jnp
from desc.utils import Timer
from desc.grid import QuadratureGrid, ConcentricGrid, LinearGrid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
import desc.compute as compute_funs
from desc.compute import (
    arg_order,
    data_index,
    compute_covariant_metric_coefficients,
    compute_magnetic_field_magnitude,
    compute_contravariant_basis,
    compute_contravariant_current_density,
    compute_contravariant_magnetic_field,
    compute_pressure,
    compute_quasisymmetry_error,
    compute_toroidal_coords,
    dot,
)
from .objective_funs import _Objective


class GenericObjective(_Objective):
    """A generic objective that can compute any quantity from the `data_index`."""

    _scalar = False
    _linear = False

    def __init__(self, f, eq=None, target=0, weight=1, grid=None, name="generic"):
        """Initialize a Generic Objective.

        Parameters
        ----------
        f : str
            Name of the quatity to compute.
        eq : Equilibrium, optional
            Equilibrium that will be optimized to satisfy the Objective.
        target : float, ndarray, optional
            Target value(s) of the objective.
            len(target) must be equal to Objective.dim_f
        weight : float, ndarray, optional
            Weighting to apply to the Objective, relative to other Objectives.
            len(weight) must be equal to Objective.dim_f
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        name : str
            Name of the objective function.

        """
        self.f = f
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Residual: {:10.3e} (" + data_index[self.f]["units"] + ")"

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
            self.grid = QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)

        args = []
        self._dim_f = self.grid.num_nodes

        self.fun = getattr(compute_funs, data_index[self.f]["fun"])
        self.sig = signature(self.fun)
        self.inputs = {"data": None}

        for arg in self.sig.parameters.keys():
            if arg in arg_order:
                args.append(arg)
            elif arg == "R_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.R_basis,
                    derivs=data_index[self.f]["R_derivs"],
                    build=True,
                )
            elif arg == "Z_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.Z_basis,
                    derivs=data_index[self.f]["R_derivs"],
                    build=True,
                )
            elif arg == "L_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    eq.L_basis,
                    derivs=data_index[self.f]["L_derivs"],
                    build=True,
                )
            elif arg == "B_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    DoubleFourierSeries(
                        M=2 * eq.M, N=2 * eq.N, sym=eq.R_basis.sym, NFP=eq.NFP
                    ),
                    derivs=0,
                    build_pinv=True,
                )
            elif arg == "w_transform":
                self.inputs[arg] = Transform(
                    self.grid,
                    DoubleFourierSeries(
                        M=2 * eq.M, N=2 * eq.N, sym=eq.Z_basis.sym, NFP=eq.NFP
                    ),
                    derivs=1,
                )
            elif arg == "pressure":
                self.inputs[arg] = eq.pressure.copy()
                self.inputs[arg].grid = self.grid
            elif arg == "iota":
                self.inputs[arg] = eq.iota.copy()
                self.inputs[arg].grid = self.grid

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._args = args
        self._built = True

    def compute(self, **kwargs):
        """Compute the quantity.

        Parameters
        ----------
        args : list of ndarray
            Any of the arguments given in `arg_order`.

        Returns
        -------
        f : ndarray
            Computed quantity.

        """
        data = self.fun(**kwargs, **self.inputs)
        f = data[self.f]
        return self._shift_scale(f)


class ToroidalCurrent(_Objective):
    """Toroidal current enclosed by a surface."""

    _scalar = True
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="toroidal current"):
        """Initialize a ToroidalCurrent Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        name : str
            Name of the objective function.

        """
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Toroidal current: {:10.3e} (A)"

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
            self.grid = LinearGrid(
                L=1,
                M=2 * eq.M_grid + 1,
                N=2 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=1,
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["I"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["I"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["I"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute toroidal current.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        I : float
            Toroidal current (A).

        """
        data = compute_quasisymmetry_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        I = 2 * jnp.pi / mu_0 * data["I"]
        return self._shift_scale(I)


class MagneticWell(_Objective):
    """Magnetic well parameter.
    As described in https://fusion.gat.com/pubs-ext/ComPlasmaPhys/A22135.pdf, the magnetic
    well parameter is a useful figure of merit for stellarator operation. Systems with
    positive well parameters are favorable for containment.
    """

    _scalar = True
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="magnetic well"):
        """Initialize a Magnetic Well Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
            Note that MagneticWell.compute() assumes a linear grid spacing to
            evaluate the well parameter. If the provided grid spacing is
            highly nonlinear, the accuracy of the returned value may diminish.
        name : str
            Name of the objective function.

        """
        if grid is not None and not isinstance(grid, LinearGrid):
            print(
                "Warning: MagneticWell.compute() assumes a linear grid spacing to"
                "evaluate the well parameter. If the provided grid spacing is"
                "highly nonlinear, the accuracy of the returned value may diminish."
            )

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Magnetic Well: {:10.3e}"  # choose string fmt output

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
            self.grid = LinearGrid(
                L=1,
                M=2 * eq.M_grid + 1,
                N=2 * eq.N_grid + 1,
                NFP=eq.NFP,
                sym=eq.sym,
                rho=1,
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._pressure = eq.pressure.copy()
        self._iota.grid = self.grid
        self._pressure.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["B_r"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["B_r"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["B_r"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, Psi, **kwargs):
        """Compute the magnetic well parameter.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        W : float
            Magnetic well parameter. Systems with positive well
            parameters are favorable for containment.

        """
        # collect required physical quantities
        data = compute_contravariant_magnetic_field(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_contravariant_basis(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data
        )
        data = compute_toroidal_coords(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data
        )
        data = compute_pressure(p_l, self._pressure, data)

        # data["V"] is the geometry of the stellarator not the volume of the flux surface.
        # the volume of the flux surface is computed below using divergence theorem:
        # volume integral of (div [0, 0, Z] = 1) = surface integral of ([0, 0, Z] dot ds)

        jacobian = data["sqrt(g)"]
        dtdz = self.grid.spacing[:, 1] * self.grid.spacing[:, 2]
        drho_dz = data["e^rho"][:, 2]  # partial derivative
        V = jnp.sum(data["Z"] * drho_dz * jnp.abs(jacobian) * dtdz)

        # compute the well parameter
        # Note that, unlike the paper cited in the objective docstring, the
        # derivative with respect to volume is taken with respect to the radial
        # coordinate rho. chain rule: d/dv = dpsi/dv * drho/dpsi * d/drho.

        drho_dpsi = 0.5 / jnp.sqrt(data["psi"] * Psi)
        # dpsi/drho also = 2*data["rho"]*Psi
        # see D'haeseleer flux coordinates eq. 4.9.10 for dv/d(flux label) formula.
        dv_drho = jnp.sum(jacobian * dtdz)
        dpsi_dv = 1 / drho_dpsi / dv_drho
        B = data["B"]
        dBsquare_drho = 2 * dot(B, data["B_r"])

        pressure_average = MagneticWell._average(
            dpsi_dv * drho_dpsi * (2 * mu_0 * data["p_r"] + dBsquare_drho),
            jacobian,
        )
        Bsquare_average = MagneticWell._average(dot(B, B), jacobian)

        W = V * pressure_average / Bsquare_average
        # stable_criteria is: W * dpsi_dv * drho_dpsi * data["p_r"] < 0
        return self._shift_scale(W)

    @staticmethod
    def _average(f, jacobian):
        """
        Returns the magnetic surface average of the given function, f.
        Assumes linear grid spacing.

        :param f:           the given function
        :param jacobian:    jacobian of the surface integral
        :return:            the magnetic surface average of f
        """
        return jnp.mean(f * jacobian) / jnp.mean(jacobian)


class RadialCurrentDensity(_Objective):
    """Radial current density."""

    _scalar = False
    _linear = False

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        grid=None,
        norm=False,
        name="radial current density",
    ):
        """Initialize a RadialCurrentDensity Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).
        name : str
            Name of the objective function.

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(normalized)" if self.norm else "(A*m)"
        self._callback_fmt = "Radial current: {:10.3e} " + units

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
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="sin",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["J^rho"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["J^rho"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["J^rho"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute radial current density.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Radial current at each node (A*m).

        """
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^rho"] * jnp.sqrt(data["g_rr"])
        if self.norm:
            data = compute_magnetic_field_magnitude(
                R_lmn,
                Z_lmn,
                L_lmn,
                i_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self.grid.weights
        # XXX: when normalized this has units of m^3 ?
        return self._shift_scale(f)

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        assert norm in [True, False]
        self._norm = norm
        units = "(normalized)" if self.norm else "(A*m)"
        self._callback_fmt = "Radial current: {:10.3e} " + units


class PoloidalCurrentDensity(_Objective):
    """Poloidal current density."""

    _scalar = False
    _linear = False

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        grid=None,
        norm=False,
        name="poloidal current",
    ):
        """Initialize a PoloidalCurrentDensity Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).
        name : str
            Name of the objective function.

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(normalized)" if self.norm else "(A*m)"
        self._callback_fmt = "Poloidal current: {:10.3e} " + units

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
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="cos",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["J^theta"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["J^theta"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["J^theta"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute poloidal current density.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Poloidal current at each node (A*m).

        """
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^theta"] * jnp.sqrt(data["g_tt"])
        if self.norm:
            data = compute_magnetic_field_magnitude(
                R_lmn,
                Z_lmn,
                L_lmn,
                i_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self.grid.weights
        # XXX: when normalized this has units of m^3 ?
        return self._shift_scale(f)

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        assert norm in [True, False]
        self._norm = norm
        units = "(normalized)" if self.norm else "(A*m)"
        self._callback_fmt = "Poloidal current: {:10.3e} " + units


class ToroidalCurrentDensity(_Objective):
    """Toroidal current density."""

    _scalar = False
    _linear = False

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        grid=None,
        norm=False,
        name="toroidal current",
    ):
        """Initialize a ToroidalCurrentDensity Objective.

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
        grid : Grid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        norm : bool, optional
            Whether to normalize the objective values (make dimensionless).
        name : str
            Name of the objective function.

        """
        self.grid = grid
        self.norm = norm
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(normalized)" if self.norm else "(A*m)"
        self._callback_fmt = "Toroidal current: {:10.3e} " + units

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
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation="cos",
                node_pattern=eq.node_pattern,
            )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["J^zeta"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["J^zeta"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["J^zeta"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, Psi, **kwargs):
        """Compute toroidal current density.

        Parameters
        ----------
        R_lmn : ndarray
            Spectral coefficients of R(rho,theta,zeta) -- flux surface R coordinate (m).
        Z_lmn : ndarray
            Spectral coefficients of Z(rho,theta,zeta) -- flux surface Z coordinate (m).
        L_lmn : ndarray
            Spectral coefficients of lambda(rho,theta,zeta) -- poloidal stream function.
        i_l : ndarray
            Spectral coefficients of iota(rho) -- rotational transform profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Toroidal current at each node (A*m).

        """
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
        )
        data = compute_covariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data=data
        )
        f = data["J^zeta"] * jnp.sqrt(data["g_zz"])
        if self.norm:
            data = compute_magnetic_field_magnitude(
                R_lmn,
                Z_lmn,
                L_lmn,
                i_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
            )
            B = jnp.mean(data["|B|"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            R = jnp.mean(data["R"] * data["sqrt(g)"]) / jnp.mean(data["sqrt(g)"])
            f = f * mu_0 / (B * R ** 2)
        f = f * data["sqrt(g)"] * self.grid.weights
        # XXX: when normalized this has units of m^3 ?
        return self._shift_scale(f)

    @property
    def norm(self):
        """bool: Whether the objective values are normalized."""
        return self._norm

    @norm.setter
    def norm(self, norm):
        assert norm in [True, False]
        self._norm = norm
        units = "(normalized)" if self.norm else "(A*m)"
        self._callback_fmt = "Toroidal current: {:10.3e} " + units
