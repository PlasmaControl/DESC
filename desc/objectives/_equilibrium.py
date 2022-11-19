from desc.backend import use_jax, put, jnp
from desc.utils import Timer
from desc.grid import QuadratureGrid, ConcentricGrid
from desc.transform import Transform
from desc.compute import (
    data_index,
    compute_force_error,
    compute_energy,
    compute_contravariant_current_density,
)
from .objective_funs import _Objective
#from desc.objectives.utils import factorize_linear_constraints, get_fixed_boundary_constraints
if use_jax:
    import jax
from desc.derivatives import Derivative

# import numpy as np
# from scipy.linalg import block_diag

# from desc.utils import svd_inv_null
# from desc.compute import arg_order
# from .linear_objectives import (
#     FixBoundaryR,
#     FixBoundaryZ,
#     FixLambdaGauge,
#     FixPressure,
#     FixIota,
#     FixPsi,
# )

from scipy.constants import mu_0
import warnings
from termcolor import colored

class ForceBalance(_Objective):
    """Radial and helical MHD force balance.

    F_rho = sqrt(g) (B^zeta J^theta - B^theta J^zeta) - grad(p)
    f_rho = F_rho |grad(rho)| dV  (N)

    F_beta = sqrt(g) J^rho
    beta = -B^zeta grad(theta) + B^theta grad(zeta)
    f_beta = F_beta |beta| dV  (N)

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

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="force",equality = True, lb = None):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(N)"
        self._callback_fmt = "Total force: {:10.3e} " + units
        self.lb = lb
        self.equality = equality

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
            if eq.node_pattern is None or eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
                self.grid = ConcentricGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                    rotation=None,
                    node_pattern=eq.node_pattern,
                )
            elif eq.node_pattern == "quad":
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )

        self._dim_f = 2 * self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._iota = eq.iota.copy()
        self._pressure.grid = self.grid
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["F_rho"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute MHD force balance errors.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            MHD force balance error at each node (N).

        """
        data = compute_force_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
        )
        fr = data["F_rho"] * data["|grad(rho)|"]
        fr = fr * data["sqrt(g)"] * self.grid.weights

        fb = data["F_beta"] * data["|beta|"]
        fb = fb * data["sqrt(g)"] * self.grid.weights

        f = jnp.concatenate([fr, fb])
        
        if self.lb:
            return -self._shift_scale(f)
        else:
            return self._shift_scale(f)
        
class ForceBalanceGalerkin(_Objective):
    """Radial and helical MHD force balance.

    F_rho = sqrt(g) (B^zeta J^theta - B^theta J^zeta) - grad(p)
    f_rho = F_rho |grad(rho)| dV  (N)

    F_beta = sqrt(g) J^rho
    beta = -B^zeta grad(theta) + B^theta grad(zeta)
    f_beta = F_beta |beta| dV  (N)

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

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="force_gal",equality = True, lb = None):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(N)"
        self._callback_fmt = "Total force: {:10.3e} " + units
        self.lb = lb
        self.equality = equality

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
                rotation=None,
                node_pattern="jacobi",
            )

        self._dim_f = eq.R_basis.num_modes + eq.Z_basis.num_modes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._iota = eq.iota.copy()
        self._pressure.grid = self.grid
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["F_rho"]["L_derivs"], build=True
        )

        #self._R_transform.build_pinv()
        #self._Z_transform.build_pinv()
        #self._L_transform.build_pinv()

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute MHD force balance errors.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            MHD force balance error at each node (N).

        """
        data = compute_force_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
        )
        fr = data["F_rho"] * data["|grad(rho)|"]
        fr = fr * data["sqrt(g)"] * self.grid.weights

        fb = data["F_beta"] * data["|beta|"]
        fb = fb * data["sqrt(g)"] * self.grid.weights
        
        F = data["F"]
        Fr = F[:,1]*data["sqrt(g)"]*self.grid.weights
        Fp = F[:,2]*data["sqrt(g)"]*self.grid.weights
        Fz = F[:,3]*data["sqrt(g)"]*self.grid.weights

        #Fr_proj = self._R_transform.project(Fr)
        #Fp_proj = self._L_transform.project(Fp)
        #Fz_proj = self._Z_transform.project(Fz)

        fr_proj = self._R_transform.project(fr)
        fb_proj = self._Z_transform.project(fb)

        #fr_proj = self._R_transform.fit(fr)
        #fb_proj = self._Z_transform.fit(fb)


        f = jnp.concatenate([fr_proj, fb_proj])
        print("The shape of f is " + str(f.shape))

        #f = jnp.concatenate([Fr_proj,Fp_proj,Fz_proj])
        
        if self.lb:
            return -self._shift_scale(f)
        else:
            return self._shift_scale(f)


class GradientForceBalance(_Objective):
    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="force",equality=True):
        
        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(N)"
        self._callback_fmt = "Total force: {:10.3e} " + units
        self.equality = equality

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
        
        from desc.objectives.utils import factorize_linear_constraints, get_fixed_boundary_constraints
        
        if self.grid is None:
            self.grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
                rotation=None,
                node_pattern="jacobi",
            )

        #self._dim_f = 2 * self.grid.num_nodes
        
        fixed_boundary_constraints = get_fixed_boundary_constraints()
        for constraint in fixed_boundary_constraints:
            if not constraint.built:
                constraint.build(eq)
        _, _, _, _, Zfb, unfixed_idxfb, _, _ = factorize_linear_constraints(
            fixed_boundary_constraints
        )
        
        self.Zfb = Zfb
        self.unfixed_idxfb = unfixed_idxfb
        self._dim_f = Zfb.shape[1]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._iota = eq.iota.copy()
        self._pressure.grid = self.grid
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["F_rho"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True
        self.eq = eq
        
        grid = ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP)
        gradB = eq.compute('grad(|B|^2)', grid=grid)['grad(|B|^2)']/(2*mu_0)
        g = eq.compute("sqrt(g)", grid=grid)['sqrt(g)']
        gradB_mag = jnp.linalg.norm(gradB, axis=-1)
        self.gradB_avg = jnp.sum(gradB_mag*g*grid.weights)/jnp.sum(g*grid.weights)

        
        
    def compute_force_balance_scalar(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute MHD force balance errors.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            MHD force balance error at each node (N).

        """
        data = compute_force_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
        )
        fr = data["F_rho"] * data["|grad(rho)|"]
        fr = fr * data["sqrt(g)"] * self.grid.weights

        fb = data["F_beta"] * data["|beta|"]
        fb = fb * data["sqrt(g)"] * self.grid.weights

        f = jnp.concatenate([fr, fb])
        #return jnp.sum(self._shift_scale(f) ** 2) / 2
        return jnp.sum(f ** 2) / (2*self.gradB_avg)
                 
    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
       
        grad = Derivative(self.compute_force_balance_scalar, mode="grad", use_jit=True)
        #grad = jax.grad(self.compute_force_balance_scalar)
        
        #print("The dim of grad is " + str(len(grad.compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs)[self.unfixed_idxfb])))
        #print("The shape of Zfb is " + str(self.Zfb.shape))
        
        g = grad.compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs)[self.unfixed_idxfb] @ self.Zfb
        
        #return g/jnp.linalg.norm(g)
        return g
        #return grad.compute(R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs)
        
        
    
    # def get_fixed_boundary_constraints(profiles=True):
    #     """Get the constraints necessary for a typical fixed-boundary equilibrium problem.

    #     Parameters
    #     ----------
    #     profiles : bool
    #         Whether to also return constraints to fix input profiles.

    #     Returns
    #     -------
    #     constraints, tuple of _Objectives
    #         A list of the linear constraints used in fixed-boundary problems.

    #     """
    #     constraints = (
    #         FixBoundaryR(fixed_boundary=True),
    #         FixBoundaryZ(fixed_boundary=True),
    #         FixLambdaGauge(),
    #         FixPsi(),
    #     )
    #     if profiles:
    #         constraints = constraints + (FixPressure(), FixIota())
    #     return constraints

    # def factorize_linear_constraints(constraints, extra_args=[]):
    #     """Compute and factorize A to get pseudoinverse and nullspace.

    #     Given constraints of the form Ax=b, factorize A to find a particular solution xp
    #     and the null space Z st. Axp=b and AZ=0, so that the full space of solutions to
    #     Ax=b can be written as x=xp + Zy where y is now unconstrained.


    #     Parameters
    #     ----------
    #     constraints : tuple of Objectives
    #         linear objectives/constraints to factorize for projection method.
    #     extra_args : list of str
    #         names of extra arguments that are not constrained but may need to be included
    #         for indexing etc. Should generally include all args to all objectives.

    #     Returns
    #     -------
    #     xp : ndarray
    #         particular solution to Ax=b
    #     A : dict of ndarray
    #         Individual constraint matrices, keyed by argument
    #     Ainv : dict of ndarray
    #         Individual pseudoinverses of constraint matrices
    #     b : dict of ndarray
    #         Individual rhs vectors
    #     Z : ndarray
    #         Null space operator for full combined A
    #     unfixed_idx : ndarray
    #         indices of x that correspond to non-fixed values
    #     project, recover : function
    #         functions to project full vector x into reduced vector y, and recovering x from y.

    #     """
    #     # set state vector
    #     args = np.concatenate([obj.args for obj in constraints])
    #     #print(args)
    #     args = np.concatenate((args, extra_args))
    #     #print(args)
    #     args = [arg for arg in arg_order if arg in args]
    #     #print(args)
    #     dimensions = constraints[0].dimensions
    #     dim_x = 0
    #     x_idx = {}
    #     for arg in args:
    #         x_idx[arg] = np.arange(dim_x, dim_x + dimensions[arg])
    #         dim_x += dimensions[arg]

    #     A = {}
    #     b = {}
    #     Ainv = {}
    #     xp = jnp.zeros(dim_x)  # particular solution to Ax=b
    #     constraint_args = []  # all args used in constraints
    #     unfixed_args = []  # subset of constraint args for unfixed objectives

    #     # linear constraint matrices for each objective
    #     for obj in constraints:
    #         if len(obj.args) > 1:
    #             raise ValueError("Linear constraints must have only 1 argument.")
    #         arg = obj.args[0]
    #         constraint_args.append(arg)
    #         if obj.fixed and obj.dim_f == obj.dimensions[obj.target_arg]:
    #             # if all coefficients are fixed the constraint matrices are not needed
    #             xp = put(xp, x_idx[obj.target_arg], obj.target)
    #         else:
    #             unfixed_args.append(arg)
    #             A_ = obj.derivatives[arg]
    #             b_ = obj.target
    #             if A_.shape[0]:
    #                 Ainv_, Z_ = svd_inv_null(A_)
    #             else:
    #                 Ainv_ = A_.T
    #             A[arg] = A_
    #             b[arg] = b_
    #             Ainv[arg] = Ainv_

    #     # catch any arguments that are not constrained
    #     for arg in x_idx.keys():
    #         if arg not in constraint_args:
    #             unfixed_args.append(arg)
    #             A[arg] = jnp.zeros((1, constraints[0].dimensions[arg]))
    #             b[arg] = jnp.zeros((1,))

    #     # full A matrix for all unfixed constraints
    #     if len(A):
    #         unfixed_idx = jnp.concatenate(
    #             [x_idx[arg] for arg in arg_order if arg in A.keys()]
    #         )
    #         A_full = block_diag(*[A[arg] for arg in arg_order if arg in A.keys()])
    #         b_full = jnp.concatenate([b[arg] for arg in arg_order if arg in b.keys()])
            
    #         Ainv_full, Z = svd_inv_null(A_full)
    #         xp = put(xp, unfixed_idx, Ainv_full @ b_full)

    #     def project(x):
    #         """Project a full state vector into the reduced optimization vector."""
    #         print(x.shape)
    #         print(xp.shape)
    #         print(Z.T.shape)
    #         x_reduced = jnp.dot(Z.T, (x - xp)[unfixed_idx])
    #         return jnp.atleast_1d(jnp.squeeze(x_reduced))

    #     def recover(x_reduced):
    #         """Recover the full state vector from the reducted optimization vector."""
    #         dx = put(jnp.zeros(dim_x), unfixed_idx, Z @ x_reduced)
    #         return jnp.atleast_1d(jnp.squeeze(xp + dx))

    #     return xp, A, Ainv, b, Z, unfixed_idx, project, recover

    
    
class RadialForceBalance(_Objective):
    """Radial MHD force balance.

    F_rho = sqrt(g) (B^zeta J^theta - B^theta J^zeta) - grad(p)
    f_rho = F_rho |grad(rho)| dV  (N)

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

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="radial force"):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(N)"
        self._callback_fmt = "Radial force: {:10.3e} " + units

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
            if eq.node_pattern is None or eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
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
            elif eq.node_pattern == "quad":
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._pressure = eq.pressure.copy()
        self._iota = eq.iota.copy()
        self._pressure.grid = self.grid
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["F_rho"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["F_rho"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute radial MHD force balance errors.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f_rho : ndarray
            Radial MHD force balance error at each node (N).

        """
        data = compute_force_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
        )
        f = data["F_rho"] * data["|grad(rho)|"]
        f = f * data["sqrt(g)"] * self.grid.weights

        return self._shift_scale(f)


class HelicalForceBalance(_Objective):
    """Helical MHD force balance.

    F_beta = sqrt(g) J^rho
    beta = -B^zeta grad(theta) + B^theta grad(zeta)
    f_beta = F_beta |beta| dV  (N)

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

    _scalar = False
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, name="helical force"):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(N)"
        self._callback_fmt = "Helical force: {:10.3e}, " + units

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
            if eq.node_pattern is None or eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
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
            elif eq.node_pattern == "quad":
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                )

        self._dim_f = self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._pressure = eq.pressure.copy()
        self._iota.grid = self.grid
        self._pressure.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["F_beta"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["F_beta"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["F_beta"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute helical MHD force balance errors.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        f : ndarray
            Helical MHD force balance error at each node (N).

        """
        data = compute_force_error(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._pressure,
            self._iota,
        )
        f = data["F_beta"] * data["|beta|"]
        f = f * data["sqrt(g)"] * self.grid.weights

        return self._shift_scale(f)


class Energy(_Objective):
    """MHD energy.

    W = integral( B^2 / (2*mu0) + p / (gamma - 1) ) dV  (J)

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
        Collocation grid containing the nodes to evaluate at. This will default to a QuadratureGrid
    gamma : float, optional
        Adiabatic (compressional) index. Default = 0.
    name : str
        Name of the objective function.

    """

    _io_attrs_ = _Objective._io_attrs_ + ["gamma"]
    _scalar = True
    _linear = False

    def __init__(self, eq=None, target=0, weight=1, grid=None, gamma=0, name="energy"):

        self.grid = grid
        self.gamma = gamma
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        self._callback_fmt = "Total MHD energy: {:10.3e} (J)"

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
            if eq.node_pattern in [
                "jacobi",
                "cheb1",
                "cheb2",
                "ocs",
                "linear",
            ]:
                warnings.warn(
                    colored(
                        "Energy objective built using grid "
                        + "that is not the quadrature grid! "
                        + "This is not recommended and may result in poor convergence. "
                        "yellow",
                    )
                )
                self.grid = ConcentricGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
                    sym=eq.sym,
                    axis=False,
                    rotation=None,
                    node_pattern=eq.node_pattern,
                )

            else:
                self.grid = QuadratureGrid(
                    L=eq.L_grid,
                    M=eq.M_grid,
                    N=eq.N_grid,
                    NFP=eq.NFP,
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
            self.grid, eq.R_basis, derivs=data_index["W"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["W"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["W"]["L_derivs"], build=True
        )

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        self._check_dimensions()
        self._set_dimensions(eq)
        self._set_derivatives(use_jit=use_jit)
        self._built = True

    def compute(self, R_lmn, Z_lmn, L_lmn, i_l, p_l, Psi, **kwargs):
        """Compute MHD energy.

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
        p_l : ndarray
            Spectral coefficients of p(rho) -- pressure profile.
        Psi : float
            Total toroidal magnetic flux within the last closed flux surface (Wb).

        Returns
        -------
        W : float
            Total MHD energy in the plasma volume (J).

        """
        data = compute_energy(
            R_lmn,
            Z_lmn,
            L_lmn,
            p_l,
            i_l,
            Psi,
            self._R_transform,
            self._Z_transform,
            self._L_transform,
            self._iota,
            self._pressure,
            self._gamma,
        )
        return self._shift_scale(jnp.atleast_1d(data["W"]))

    @property
    def gamma(self):
        """float: Adiabatic (compressional) index."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma


class CurrentDensity(_Objective):
    """Radial, poloidal, and toroidal current density.

    Useful for solving vacuum equilibria.

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

    _scalar = False
    _linear = False

    def __init__(
        self,
        eq=None,
        target=0,
        weight=1,
        grid=None,
        name="current density",
        equality = True
    ):

        self.grid = grid
        super().__init__(eq=eq, target=target, weight=weight, name=name)
        units = "(A/m^2)"
        self._callback_fmt = "Total current density: {:10.3e} " + units
        self.equality = equality

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
                rotation=None,
                node_pattern="jacobi",
            )

        self._dim_f = 3 * self.grid.num_nodes

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._iota = eq.iota.copy()
        self._iota.grid = self.grid

        self._R_transform = Transform(
            self.grid, eq.R_basis, derivs=data_index["J"]["R_derivs"], build=True
        )
        self._Z_transform = Transform(
            self.grid, eq.Z_basis, derivs=data_index["J"]["R_derivs"], build=True
        )
        self._L_transform = Transform(
            self.grid, eq.L_basis, derivs=data_index["J"]["L_derivs"], build=True
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
        jr = data["J^rho"] * data["sqrt(g)"] * self.grid.weights
        jt = data["J^theta"] * data["sqrt(g)"] * self.grid.weights
        jz = data["J^zeta"] * data["sqrt(g)"] * self.grid.weights

        f = mu_0*jnp.concatenate([jr, jt, jz])
        return self._shift_scale(f)
