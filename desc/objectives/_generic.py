from scipy.constants import mu_0
from inspect import signature

from desc.backend import jnp
from desc.utils import Timer
from desc.grid import QuadratureGrid, LinearGrid, ConcentricGrid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
import desc.compute as compute_funs
from desc.compute import (
    arg_order,
    data_index,
    compute_covariant_metric_coefficients,
    compute_magnetic_field_magnitude,
    compute_contravariant_current_density,
    compute_contravariant_metric_coefficients,
    compute_pressure,
    compute_quasisymmetry_error,
    compute_toroidal_coords,
    cross,
    dot,
)
from desc.compute.utils import compress, surface_averages, surface_integrals
from .objective_funs import _Objective


class GenericObjective(_Objective):
    """A generic objective that can compute any quantity from the `data_index`.

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

    _scalar = False
    _linear = False

    def __init__(self, f, eq=None, target=0, weight=1, grid=None, name="generic"):

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
                if eq.iota is not None:
                    self.inputs[arg] = eq.iota.copy()
                    self.inputs[arg].grid = self.grid
                else:
                    self.inputs[arg] = None
            elif arg == "current":
                if eq.current is not None:
                    self.inputs[arg] = eq.current.copy()
                    self.inputs[arg].grid = self.grid
                else:
                    self.inputs[arg] = None

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


# TODO: move this class to a different file (not generic)
class ToroidalCurrent(_Objective):
    """Toroidal current enclosed by a surface.

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

    _scalar = True
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="toroidal current"):

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
            self.grid = LinearGrid(M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym)

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

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

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
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
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
        )
        I = 2 * jnp.pi / mu_0 * data["I"]
        return self._shift_scale(I)


class MagneticWell(_Objective):
    """
    The magnetic well parameter is a fast proxy for MHD stability.
    This makes it a useful figure of merit for stellarator operation.
    Systems with positive well parameters are favorable for containment.

    Landreman, M., & Jorge, R. (2020). Magnetic well and Mercier stability of
    stellarators near the magnetic axis. Journal of Plasma Physics, 86(5), 905860510.
    doi:10.1017/S002237782000121X.
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
        grid : LinearGrid, ConcentricGrid, QuadratureGrid, ndarray, optional
            Collocation grid containing the nodes to evaluate at.
        name : str
            Name of the objective function.
        """
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Magnetic Well: {:10.3e}"

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
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
            )

        self._dim_f = 1

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._pressure.grid = self.grid
        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

        R_derivs = jnp.vstack(
            (
                data_index["B_r"]["R_derivs"],
                data_index["B_theta_r"]["R_derivs"],
                data_index["J"]["R_derivs"],
            )
        )
        L_derivs = jnp.vstack(
            (
                data_index["B_r"]["L_derivs"],
                data_index["B_theta_r"]["L_derivs"],
                data_index["J"]["L_derivs"],
            )
        )
        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=R_derivs, build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=R_derivs, build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=L_derivs, build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, p_l, i_l, c_l, Psi, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        W : ndarray
            Magnetic well parameter.

            Currently, returns a dictionary of values for debugging purposes.
        """
        # data = compute_contravariant_magnetic_field(
        #     R_lmn,
        #     Z_lmn,
        #     L_lmn,
        #     i_l,
        #     c_l,
        #     Psi,
        #     self._R_transform,
        #     self._Z_transform,
        #     self._L_transform,
        #     self._iota,
        #     self._current,
        # )
        # Dcurr specific call
        data = compute_contravariant_current_density(
            R_lmn,
            Z_lmn,
            L_lmn,
            i_l,
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
        )
        # end Dcurr specific calls
        data = compute_toroidal_coords(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data
        )
        data = compute_pressure(p_l, self._pressure, data)
        data = compute_contravariant_metric_coefficients(
            R_lmn, Z_lmn, self._R_transform, self._Z_transform, data
        )

        dv_drho = self._enclosed_volume(data, dr=1)
        sqrtg = jnp.abs(data["sqrt(g)"])
        sqrtg_r = jnp.abs(data["sqrt(g)_r"])
        Bsq = dot(data["B"], data["B"])
        Bsq_av = surface_averages(self.grid, Bsq, sqrtg, denominator=dv_drho)

        # pressure = thermal + magnetic
        # The flux surface average function is an additive homomorphism
        # This means average(a + b) = average(a) + average(b).
        # Thermal pressure is constant over a rho surface.
        # Therefore average(thermal) = thermal.
        dthermal_drho = 2 * mu_0 * compress(self.grid, data["p_r"])
        dmagnetic_av_drho = (
            surface_integrals(
                self.grid, sqrtg_r * Bsq + sqrtg * 2 * dot(data["B"], data["B_r"])
            )
            - surface_integrals(self.grid, sqrtg_r) * Bsq_av
        ) / dv_drho

        unique_rho = compress(self.grid, self.grid.nodes[:, 0])
        W2 = unique_rho * (dthermal_drho + dmagnetic_av_drho) / Bsq_av

        V = self._enclosed_volume(data)
        W3 = V * (dthermal_drho + dmagnetic_av_drho) / dv_drho / Bsq_av

        return {
            "Dshear": (a := self.Dshear(data)),
            "Dcurr": (b := self.Dcurr(data)),
            "Dgeod": (c := self.Dgeod(data)),
            "1. DESC Magnetic Well: M. Landreman eq. 4.19": self._shift_scale(
                d := self.Dwell(data)
            ),
            "Mercier": a + b + c + d,
            "2. DESC Magnetic Well: M. Landreman eq. 3.2 with rho * d/drho": self._shift_scale(
                W2
            ),
            "3. DESC Magnetic Well: M. Landreman eq. 3.2": self._shift_scale(W3),
            "B square average": Bsq_av,
            "d(magnetic pressure average)/drho": dmagnetic_av_drho,
            "d(thermal pressure)/drho": dthermal_drho,
            "d(total pressure average)/drho": dthermal_drho + dmagnetic_av_drho,
            "d(total pressure average)/dvolume": (dthermal_drho + dmagnetic_av_drho)
            / dv_drho,
            "d^2(volume)/d(rho)^2": self._enclosed_volume(data, dr=2),
            "dvolume/drho": dv_drho,
            "volume": V,
            "dtdz": surface_integrals(self.grid, jnp.ones(len(self.grid.nodes))),
        }

    # TODO: should add to compute.utils
    def _enclosed_volume(self, data, dr=0):
        """
        Returns enclosed positive volume derivatives wrt rho.
        See D'haeseleer flux coordinates eq. 4.9.10 for dv/d(flux label).
        Intuitively, this formula makes sense because
        V = integral(dr dt dz * sqrt(g)) and differentiating wrt rho removes
        the dr integral. What remains is a surface integral with a 3D jacobian.
        This is the volume added by a thin shell of constant rho.

        Parameters
        ----------
        dr: int
            Derivative order

        Returns
        -------
        ndarray
            dr derivative of volume enclosed by flux surface wrt rho.
        """
        if dr == 0:
            # data["V"] is the total volume, not the volume enclosed by the flux surface.
            # enclosed volume is computed using divergence theorem:
            # volume integral(div [0, 0, Z] = 1) = surface integral([0, 0, Z] dot ds)
            sqrtg_e_sup_rho = cross(data["e_theta"], data["e_zeta"])
            return jnp.abs(
                surface_integrals(self.grid, sqrtg_e_sup_rho[:, 2] * data["Z"])
            )
        if dr == 1:
            return surface_integrals(self.grid, jnp.abs(data["sqrt(g)"]))
        if dr == 2:
            return surface_integrals(self.grid, jnp.abs(data["sqrt(g)_r"]))

    # TODO:
    #   How do we want to return these quantities?
    #   New objectives?
    def Dshear(self, data):
        """M. Landreman Equation 4.17"""
        return compress(self.grid, jnp.square(data["iota_r"] / data["psi_r"])) / (
            16 * jnp.pi ** 2
        )

    def Dcurr(self, data):
        """M. Landreman Equation 4.18"""
        # grad(psi) = grad(rho) * dpsi/drho
        # A * dtdz = |ds / grad(psi)^3| = |sqrt(g) * grad(rho) / grad(psi)^3| * dtdz
        A = jnp.abs(data["sqrt(g)"] / data["psi_r"] ** 3) / data["g^rr"]

        # TODO: confirm whether this commit
        #   https://github.com/PlasmaControl/DESC/commit/5d38b9ddb90f7efaed88447d3f19fd7a023c92cd
        #   altered the computation to just make it simpler, or because the old one gave an incorrect result.
        #   Mathematically, they should be equivalent.
        dI_dpsi = surface_averages(
            self.grid,
            data["B_theta_r"] / data["psi_r"],
            match_grid=True,
            denominator=4 * jnp.pi ** 2,
        )
        xi = mu_0 * data["J"] - jnp.atleast_2d(dI_dpsi).T * data["B"]
        # data["G"] = poloidal current
        sign_G = jnp.sign(surface_integrals(self.grid, data["B_zeta"]))

        return (
            -sign_G
            / (16 * jnp.pi ** 4)
            * compress(self.grid, data["iota_r"] / data["psi_r"])
            * surface_integrals(self.grid, A * dot(xi, data["B"]))
        )

    def Dwell(self, data):
        """M. Landreman Equation 4.19"""
        # grad(psi) = grad(rho) * dpsi/drho
        psi_r_sq = jnp.square(data["psi_r"])
        grad_psi_sq = data["g^rr"] * psi_r_sq
        # A * dtdz = |ds / grad(psi)| = |sqrt(g) * grad(rho) / grad(psi)| * dtdz
        A = jnp.abs(data["sqrt(g)"] / data["psi_r"])

        dv_drho = self._enclosed_volume(data, dr=1)
        d2v_drho2 = self._enclosed_volume(data, dr=2)
        d2v_dpsi2 = compress(self.grid, jnp.sign(data["psi"]) / psi_r_sq) * (
            d2v_drho2 - dv_drho * compress(self.grid, data["psi_rr"] / data["psi_r"])
        )
        dp_dpsi = compress(self.grid, data["p_r"] / data["psi_r"])
        Bsq = dot(data["B"], data["B"])

        return (
            mu_0
            / (64 * jnp.pi ** 6)
            * dp_dpsi
            * (d2v_dpsi2 - mu_0 * dp_dpsi * surface_integrals(self.grid, A / Bsq))
            * surface_integrals(self.grid, A / grad_psi_sq * Bsq)
        )

    def Dgeod(self, data):
        """M. Landreman Equation 4.20"""
        # grad(psi) = grad(rho) * dpsi/drho
        # A * dtdz = |ds / grad(psi)^3| = |sqrt(g) * grad(rho) / grad(psi)^3| * dtdz
        A = jnp.abs(data["sqrt(g)"] / data["psi_r"] ** 3) / data["g^rr"]

        j_dot_b = mu_0 * dot(data["J"], data["B"])
        Bsq = dot(data["B"], data["B"])

        return (
            jnp.square(surface_integrals(self.grid, A * j_dot_b))
            - surface_integrals(self.grid, A * Bsq)
            * surface_integrals(self.grid, A * jnp.square(j_dot_b) / Bsq)
        ) / (64 * jnp.pi ** 6)


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

        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

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

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
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
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
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
                c_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
                self._current,
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

        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

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

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
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
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
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
                c_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
                self._current,
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

        if eq.iota is not None:
            self._iota = eq.iota.copy()
            self._iota.grid = self.grid
            self._current = None
        else:
            self._current = eq.current.copy()
            self._current.grid = self.grid
            self._iota = None

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

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, c_l, Psi, **kwargs):
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
        c_l : ndarray
            Spectral coefficients of I(rho) -- toroidal current profile.
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
            c_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._current,
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
                c_l,
                Psi,
                self._R_transform,
                self._Z_transform,
                self._L_transform,
                self._iota,
                self._current,
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
