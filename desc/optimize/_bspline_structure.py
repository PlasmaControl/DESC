"""Exploit the radial locality of B-spline bases in equilibrium solves.

With a ``BSplineDoubleFourierBasis``, a force balance residual at a collocation node
only depends on the spline coefficients whose radial function is nonzero in the
radial interval that contains the node. The Jacobian is therefore block banded when
unknowns and residuals are ordered radially. This module finds that structure for a
``LinearConstraintProjection`` and uses it to

* compute the Jacobian with one JVP per group of structurally orthogonal columns
  instead of one per column ("compressed" forward mode), and
* form and factorize the normal equations in banded storage for the trust region
  subproblem of ``lsqtr``.
"""

import heapq
import warnings

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded

# equilibrium parameters whose basis functions are radially local
_LOCAL = {"R_lmn": "R_basis", "Z_lmn": "Z_basis", "L_lmn": "L_basis"}
# parameters that only enter through linear constraints, not force balance
_INERT = {"Ra_n", "Za_n", "Rb_lmn", "Zb_lmn"}


def _local_objective_types():
    from desc.objectives import ForceBalance

    return (ForceBalance,)


class BSplineStructure:
    """Sparsity structure of a force balance Jacobian in a B-spline basis.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective whose Jacobian is wanted. Must only contain radially local
        objectives (``ForceBalance``) of a single Equilibrium with a B-spline basis.
    tangents : ndarray, shape(dim_x, dim_x_reduced)
        Feasible directions of the linear constraints, one per reduced variable.

    Attributes
    ----------
    color : ndarray of int, shape(dim_x_reduced,)
        Group of each reduced variable. Variables in a group affect disjoint rows.
    num_colors : int
        Number of groups, ie number of JVPs needed for the Jacobian.
    lo, hi : ndarray of int, shape(dim_x_reduced,)
        First and last radial interval whose residuals a variable can change.
    row_elem : ndarray of int, shape(dim_f,)
        Radial interval of each residual.
    perm : ndarray of int, shape(dim_x_reduced,)
        Ordering of the variables that makes the normal equations banded.
    bandwidth : int
        Number of super-diagonals of the normal equations in that ordering.

    """

    def __init__(self, objective, tangents):
        eq = objective.things[0]
        knots = eq.R_basis.knots
        L = eq.R_basis.L

        # radial interval of each residual
        row_elem = []
        for obj in objective.objectives:
            grid = obj._constants["transforms"]["grid"]
            rho = np.asarray(grid.nodes[:, 0])
            elem = np.clip(np.searchsorted(knots, rho, side="right") - 1, 0, L - 1)
            row_elem.append(np.tile(elem, obj.dim_f // grid.num_nodes))
        self.row_elem = np.concatenate(row_elem)
        self.num_elems = L

        # radial intervals each reduced variable can influence
        T = np.asarray(tangents)
        nz = np.abs(T) > 1e-12 * np.abs(T).max()
        n = T.shape[1]
        lo = np.full(n, L, dtype=int)
        hi = np.full(n, -1, dtype=int)
        is_global = np.zeros(n, dtype=bool)
        for key, idx in eq.x_idx.items():
            idx = np.asarray(idx)
            if not idx.size or key in _INERT:
                continue
            if key in _LOCAL:
                basis = getattr(eq, _LOCAL[key])
                first, last = basis.radial_support(basis.modes[:, 0], basis.modes[:, 1])
                sub = nz[idx]
                lo = np.minimum(lo, np.where(sub, first[:, None], L).min(axis=0))
                hi = np.maximum(hi, np.where(sub, last[:, None], -1).max(axis=0))
            else:
                is_global |= nz[idx].any(axis=0)
        is_global |= hi < lo  # nothing local found: be safe and treat as dense
        lo[is_global] = 0
        hi[is_global] = L - 1
        self.lo, self.hi = lo, hi
        self.num_global = int(is_global.sum())

        # interval graph coloring: variables with disjoint supports share a JVP
        order = np.lexsort((hi, lo))
        color = np.zeros(n, dtype=int)
        free = []  # heap of (last interval used, color)
        ncolors = 0
        for j in order:
            if free and free[0][0] < lo[j]:
                _, c = heapq.heappop(free)
            else:
                c = ncolors
                ncolors += 1
            color[j] = c
            heapq.heappush(free, (hi[j], c))
        self.color = color
        self.num_colors = ncolors
        self.mask = (self.row_elem[:, None] >= lo[None, :]) & (
            self.row_elem[:, None] <= hi[None, :]
        )

        # banded ordering of the normal equations
        self.perm = order
        lo_p, hi_p = lo[order], hi[order]
        # last variable (in the new order) that shares an interval with variable i
        last = np.searchsorted(lo_p, hi_p, side="right") - 1
        self.bandwidth = int(np.max(last - np.arange(n))) if n else 0
        # for each interval: its residual rows and the window of variables it sees
        self.elem_rows = [np.where(self.row_elem == e)[0] for e in range(L)]
        self.elem_window = []
        for e in range(L):
            cols = np.where((lo_p <= e) & (hi_p >= e))[0]
            self.elem_window.append(
                (int(cols.min()), int(cols.max()) + 1) if cols.size else (0, 0)
            )

    @classmethod
    def from_objective(cls, objective, tangents):
        """Return the structure, or None if the objective does not have one."""
        try:
            things = objective.things
            if len(things) != 1:
                return None
            if getattr(things[0], "radial_basis", None) != "bspline":
                return None
            if not all(
                isinstance(obj, _local_objective_types())
                for obj in objective.objectives
            ):
                return None
            return cls(objective, tangents)
        except Exception as err:  # an optimization only, never fail a solve
            warnings.warn(f"Could not find the B-spline Jacobian structure: {err!r}")
            return None

    def compress(self, tangents):
        """Sum the feasible directions of each group, shape(dim_x, num_colors)."""
        T = np.asarray(tangents)
        out = np.zeros((T.shape[0], self.num_colors))
        np.add.at(out.T, self.color, T.T)
        return out

    def normal_equations_banded(self, J):
        """J.T @ J in the banded ordering, as LAPACK upper banded storage."""
        J = np.asarray(J)
        n, u = J.shape[1], self.bandwidth
        ab = np.zeros((u + 1, n))
        for rows, (c0, c1) in zip(self.elem_rows, self.elem_window):
            if not rows.size or c1 == c0:
                continue
            Je = J[np.ix_(rows, self.perm[c0:c1])]
            G = Je.T @ Je
            i, j = np.triu_indices(c1 - c0)
            ab[u + i - j, c0 + j] += G[i, j]
        return ab


def trust_region_step_banded(g, ab, trust_radius, initial_alpha, rtol=0.01, max_iter=10):
    """Solve (B + alpha*I) p = -g, ||p|| <= trust_radius, for banded B.

    Same algorithm as ``trust_region_step_exact_cho`` (Nocedal & Wright 4.3), with
    ``B`` given in LAPACK upper banded storage ``ab`` and ``g`` in the same ordering.
    """
    u = ab.shape[0] - 1
    diag = ab[u].copy()
    # tiny shift for safety, B is only positive semi-definite in exact arithmetic
    eps = 1e-14 * max(diag.max(), 1e-300)

    def factor_solve(alpha, rhs):
        a = ab.copy()
        a[u] = diag + alpha + eps
        c = cholesky_banded(a, lower=False, check_finite=False)
        return c, cho_solve_banded((c, False), rhs, check_finite=False)

    try:
        _, p = factor_solve(0.0, -g)
        if np.linalg.norm(p) <= trust_radius and np.all(np.isfinite(p)):
            return p, False, 0.0
    except np.linalg.LinAlgError:
        pass

    alpha_upper = np.linalg.norm(g) / trust_radius
    alpha_lower = 0.0
    alpha = float(np.clip(initial_alpha, alpha_lower, alpha_upper))
    if alpha == 0:
        alpha = 1e-3 * alpha_upper
    p = -g
    for _ in range(max_iter):
        try:
            c, p = factor_solve(alpha, -g)
        except np.linalg.LinAlgError:
            alpha_lower = alpha
            alpha = max(2 * alpha, 0.5 * (alpha_lower + alpha_upper))
            continue
        p_norm = np.linalg.norm(p)
        phi = p_norm - trust_radius
        if phi < 0:
            alpha_upper = alpha
        else:
            alpha_lower = alpha
        if abs(phi) <= rtol * trust_radius:
            break
        # ||q||^2 = p.T @ inv(B + alpha I) @ p
        q_norm2 = p @ cho_solve_banded((c, False), p, check_finite=False)
        alpha += p_norm**2 / q_norm2 * phi / trust_radius
        alpha = float(np.clip(alpha, alpha_lower, alpha_upper))
    p = p * (trust_radius / np.linalg.norm(p))
    return p, True, alpha
