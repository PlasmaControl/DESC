"""Exploit the radial locality of B-spline bases in equilibrium solves.

With a ``BSplineDoubleFourierBasis``, a force balance residual at a collocation node
only depends on the spline coefficients whose radial function is nonzero in the
radial interval that contains the node. The Jacobian is therefore block banded when
unknowns and residuals are ordered radially. This module finds that structure for a
``LinearConstraintProjection`` and uses it to

* compute the Jacobian with one JVP per group of structurally orthogonal columns
  instead of one per column, and keep it in that compressed form
  (``CompressedJacobian``, shape (dim_f, num_groups)) instead of a dense matrix, and
* form the normal equations as a block tridiagonal matrix and solve the trust region
  subproblem of ``lsqtr`` with a block tridiagonal Cholesky factorization.
"""

import functools
import heapq
import warnings
import weakref

import numpy as np

from desc.backend import jit, jnp, scan, solve_triangular, while_loop

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
    colmap : ndarray of int, shape(num_intervals, num_colors)
        Variable of each group that is active in each interval, or -1.
    blk, loc : ndarray of int, shape(dim_x_reduced,)
        Block and position within the block of each variable, in the block
        tridiagonal ordering of the normal equations.
    num_blocks, block_size : int
        Number of blocks and size of the (padded) blocks.

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
        # colmap[e, c]: the variable of group c that is active in interval e, or -1
        colmap = np.full((L, ncolors), -1, dtype=int)
        for j in range(n):
            colmap[lo[j] : hi[j] + 1, color[j]] = j
        self.colmap = colmap
        self.elem_rows = [np.where(self.row_elem == e)[0] for e in range(L)]

        # Block tridiagonal partition of the normal equations. Variables are grouped
        # by the interval where their support starts, in blocks of `span - 1`
        # intervals, where `span` is the longest support. Two variables that share an
        # interval start at most `span - 1` intervals apart, so they are in the same
        # or in neighboring blocks, and the variables active in one interval are in
        # at most two neighboring blocks.
        span = int(np.max(hi - lo + 1)) if n else 1
        self.span = span
        blk = lo // max(span - 1, 1)
        self.num_blocks = int(blk.max()) + 1 if n else 1
        loc = np.zeros(n, dtype=int)
        sizes = np.zeros(self.num_blocks, dtype=int)
        for j in order:
            loc[j] = sizes[blk[j]]
            sizes[blk[j]] += 1
        self.blk, self.loc = blk, loc
        self.block_size = int(sizes.max()) if n else 1
        self.block_sizes = sizes
        self.num_vars = n

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

    def to_blocks(self, v):
        """Vector over the variables -> padded blocks, shape(num_blocks, block_size)."""
        out = jnp.zeros((self.num_blocks, self.block_size), dtype=v.dtype)
        return out.at[self.blk, self.loc].set(v)

    def from_blocks(self, vb):
        """Padded blocks -> vector over the variables."""
        return vb[self.blk, self.loc]


class CompressedJacobian:
    """Jacobian of a radially local objective, stored as one column per group.

    ``data[r, c]`` is the derivative of residual ``r`` with respect to the only
    variable of group ``c`` whose radial support contains the interval of ``r``
    (``structure.colmap``). It supports the operations ``lsqtr`` needs without ever
    forming the dense (dim_f, dim_x) matrix.

    Parameters
    ----------
    structure : BSplineStructure
        Sparsity structure.
    data : ndarray, shape(dim_f, num_colors)
        Output of the grouped JVPs.

    """

    def __init__(self, structure, data):
        self.structure = structure
        self.data = data
        self.shape = (data.shape[0], structure.num_vars)
        self.dtype = data.dtype
        self._ops = _ops(structure)

    def block_until_ready(self):
        """Wait for the data to be computed (as for a jax array)."""
        self.data = self.data.block_until_ready()
        return self

    def matvec(self, v):
        """J @ v."""
        return self._ops["matvec"](self.data, jnp.asarray(v))

    def rmatvec(self, u):
        """J.T @ u."""
        return self._ops["rmatvec"](self.data, jnp.asarray(u))

    def __matmul__(self, v):
        return self.matvec(v)

    def column_norms(self):
        """2-norm of every column."""
        return self._ops["column_norms"](self.data)

    def scale_columns(self, d):
        """Return J * d (columns scaled by d), reusing the buffer of J."""
        data = self._ops["scale"](self.data, jnp.asarray(d))
        self.data = None  # the buffer was donated
        return CompressedJacobian(self.structure, data)

    def todense(self):
        """Dense (dim_f, dim_x) Jacobian, as a jax array (also works under jit)."""
        return self._ops["todense"](self.data)

    def normal_blocks(self):
        """J.T @ J as a block tridiagonal matrix.

        Returns
        -------
        D : ndarray, shape(num_blocks, block_size, block_size)
            Diagonal blocks. Blocks are padded to the same size with the identity.
        E : ndarray, shape(num_blocks, block_size, block_size)
            ``E[k]`` is the block below ``D[k]`` (rows of block k+1, columns of
            block k). The last one is zero.

        """
        return self._ops["normal_blocks"](self.data)


_OPS_CACHE = weakref.WeakKeyDictionary()


def _ops(st):  # noqa: C901
    """Jitted operations on the data of a CompressedJacobian, cached per structure.

    The index arrays of the structure are compile time constants of these functions.
    """
    if st in _OPS_CACHE:
        return _OPS_CACHE[st]
    L, n, nc = st.num_elems, st.num_vars, st.num_colors
    K, b = st.num_blocks, st.block_size
    row_elem = st.row_elem
    valid = st.colmap >= 0  # (L, num_colors)
    cols = np.where(valid, st.colmap, 0)
    ev, cv = np.nonzero(valid)
    target = st.colmap[ev, cv]

    def expand(v):
        # value of v for the active variable of every (interval, group)
        return jnp.where(valid, v[cols], 0.0)

    def segment(X):
        # sum of the rows of X over each interval
        return jnp.zeros((L, X.shape[1]), dtype=X.dtype).at[row_elem].add(X)

    def collect(G):
        # sum (interval, group) values onto the variables they belong to
        return jnp.zeros(n, dtype=G.dtype).at[target].add(G[ev, cv])

    @jit
    def matvec(data, v):
        return jnp.sum(data * expand(v)[row_elem], axis=1)

    @jit
    def rmatvec(data, u):
        return collect(segment(data * u[:, None]))

    @jit
    def column_norms(data):
        return collect(segment(data**2)) ** 0.5

    @functools.partial(jit, donate_argnums=0)
    def scale(data, d):
        return data * expand(d)[row_elem]

    cm = st.colmap[row_elem]  # variable of each (residual, group)
    rr, gg = np.nonzero(cm >= 0)
    cc = cm[rr, gg]

    @jit
    def todense(data):
        out = jnp.zeros((data.shape[0], n), dtype=data.dtype)
        return out.at[rr, cc].set(data[rr, gg])

    # For normal_blocks: the residual rows of each interval (padded with an out of
    # range index), the first block k_e of the variables active in the interval,
    # and the position of each group's variable in blocks (k_e, k_e + 1) stacked,
    # or 2*b if the group has no variable there.
    rmax = max((r.size for r in st.elem_rows), default=0)
    rows_pad = np.full((L, max(rmax, 1)), row_elem.size, dtype=int)
    k_e = np.zeros(L, dtype=int)
    pos = np.full((L, nc), 2 * b, dtype=int)
    for e, rows in enumerate(st.elem_rows):
        rows_pad[e, : rows.size] = rows
        act = np.nonzero(valid[e])[0]
        if act.size:
            var = st.colmap[e, act]
            k_e[e] = st.blk[var].min()
            pos[e, act] = (st.blk[var] - k_e[e]) * b + st.loc[var]
    pad = np.arange(b)[None, :] >= st.block_sizes[:, None]
    pk, pi = np.nonzero(pad)

    @jit
    def normal_blocks(data):
        def body(carry, x):
            D, E = carry
            rows, k, p = x
            X = jnp.take(data, rows, axis=0, mode="fill", fill_value=0)
            G = X.T @ X
            M = jnp.zeros((2 * b, 2 * b), dtype=data.dtype)
            M = M.at[p[:, None], p[None, :]].add(G, mode="drop")
            D = D.at[k].add(M[:b, :b])
            D = D.at[k + 1].add(M[b:, b:], mode="drop")
            E = E.at[k].add(M[b:, :b])
            return (D, E), None

        D = jnp.zeros((K, b, b), dtype=data.dtype)
        E = jnp.zeros((K, b, b), dtype=data.dtype)
        (D, E), _ = scan(body, (D, E), (rows_pad, k_e, pos))
        # identity on the padding so the blocks stay positive definite
        return D.at[pk, pi, pi].set(1.0), E

    ops = dict(
        matvec=matvec,
        rmatvec=rmatvec,
        column_norms=column_norms,
        scale=scale,
        todense=todense,
        normal_blocks=normal_blocks,
    )
    _OPS_CACHE[st] = ops
    return ops


@jit
def _block_cholesky(D, E, shift):
    """Cholesky factor of a block tridiagonal matrix with ``shift`` on the diagonal.

    Returns the diagonal blocks ``Ld[k]`` (lower triangular) and the sub-diagonal
    blocks ``Lo[k]`` (block k+1, k) of the factor.
    """
    b = D.shape[1]

    def body(C, x):
        Dk, Ek, sk = x
        # C is the sub-diagonal block of the factor to the left of this row
        Lkk = jnp.linalg.cholesky(Dk + jnp.diag(sk) - C @ C.T)
        Cn = solve_triangular(Lkk, Ek.T, lower=True).T
        return Cn, (Lkk, Cn)

    _, (Ld, Lo) = scan(body, jnp.zeros((b, b), dtype=D.dtype), (D, E, shift))
    return Ld, Lo


@jit
def _block_chol_failed(Ld, real):
    """Whether a block Cholesky factor is unusable because the matrix was singular.

    The factorization returns nan when a pivot is not positive, but a pivot at the
    level of rounding errors completes and gives a useless factor, so both count as
    a failure. Only the pivots of real (not padding) variables are checked.
    """
    d = jnp.diagonal(Ld, axis1=1, axis2=2)
    n = jnp.sum(real)
    dmax = jnp.max(jnp.where(real, jnp.abs(d), 0.0))
    dmin = jnp.min(jnp.where(real, d, jnp.inf))
    tol = n * jnp.finfo(Ld.dtype).eps
    return jnp.any(jnp.isnan(Ld)) | (dmin**2 <= tol * dmax**2)


@jit
def _block_solve(Ld, Lo, rhs):
    """Solve with the factor from ``_block_cholesky``. ``rhs`` has shape (K, b)."""
    b = Ld.shape[1]
    # Lo[k] couples block k+1 to block k: shift so row k sees its left neighbor
    Lleft = jnp.concatenate([jnp.zeros((1, b, b), dtype=Ld.dtype), Lo[:-1]])

    def fwd(yprev, x):
        Lkk, C, r = x
        y = solve_triangular(Lkk, r - C @ yprev, lower=True)
        return y, y

    _, y = scan(fwd, jnp.zeros(b, dtype=Ld.dtype), (Ld, Lleft, rhs))

    def bwd(xnext, x):
        Lkk, C, yk = x
        xk = solve_triangular(Lkk.T, yk - C.T @ xnext, lower=False)
        return xk, xk

    _, x = scan(bwd, jnp.zeros(b, dtype=Ld.dtype), (Ld, Lo, y), reverse=True)
    return x


_PCG_CACHE = weakref.WeakKeyDictionary()


def _pcg(st):
    """Jitted preconditioned CG for (J.T@J + alpha*I) p = -g, cached per structure.

    The preconditioner is the block Cholesky factor (Ld, Lo) and the products with
    J.T@J are computed through J, so the accuracy is set by the condition number of
    J rather than J.T@J. With an exact factorization this converges in one or two
    steps (like iterative refinement). After a shift by delta it still resolves the
    directions with singular values below about sqrt(delta), which the shifted
    factor alone would damp.
    """
    if st in _PCG_CACHE:
        return _PCG_CACHE[st]
    ops = _ops(st)
    blk, loc, K, b = st.blk, st.loc, st.num_blocks, st.block_size

    def normal(data, v, alpha):
        return ops["rmatvec"](data, ops["matvec"](data, v)) + alpha * v

    def apply(Ld, Lo, r):
        rb = jnp.zeros((K, b), dtype=r.dtype).at[blk, loc].set(r)
        return _block_solve(Ld, Lo, rb)[blk, loc]

    @functools.partial(jit, static_argnums=6)
    def pcg(data, Ld, Lo, g, alpha, tol, maxiter):
        p = apply(Ld, Lo, -g)
        r = -g - normal(data, p, alpha)
        z = apply(Ld, Lo, r)
        stop = tol * jnp.linalg.norm(g)

        def cond_fun(state):
            p, r, z, d, rz, k, ok = state
            return ok & (k < maxiter) & (jnp.linalg.norm(r) > stop)

        def body(state):
            p, r, z, d, rz, k, ok = state
            Ad = normal(data, d, alpha)
            dAd = d @ Ad
            ok = dAd > 0
            a = jnp.where(ok, rz / jnp.where(ok, dAd, 1.0), 0.0)
            p = p + a * d
            r = r - a * Ad
            z = apply(Ld, Lo, r)
            rz_new = r @ z
            d = z + jnp.where(ok, rz_new / rz, 0.0) * d
            return p, r, z, d, rz_new, k + 1, ok

        state = (p, r, z, z, r @ z, 0, jnp.array(True))
        p, r, _, _, _, k, _ = while_loop(cond_fun, body, state)
        return p, k, jnp.linalg.norm(r) / jnp.linalg.norm(g)

    _PCG_CACHE[st] = pcg
    return pcg


def trust_region_step_block(
    f,
    J,
    D,
    E,
    trust_radius,
    initial_alpha,
    rtol=0.01,
    max_iter=10,
    cg_tol=1e-10,
    cg_maxiter=50,
):
    """Solve min_p ||J p + f||^2, ||p|| <= trust_radius, with J.T@J block tridiagonal.

    Same algorithm as ``trust_region_step_exact_cho`` (Nocedal & Wright 4.3) with a
    block tridiagonal Cholesky factorization of ``J.T@J + alpha*I``. ``J.T@J`` is
    often singular or very ill conditioned, so the factorization is only used as the
    preconditioner of conjugate gradients whose products are computed through ``J``
    instead of ``J.T@J`` (see ``_pcg``), and the accuracy of the step depends on the
    condition number of ``J``. The factorization is shifted by at least a small
    ``delta`` just above the level of rounding errors, so it does not fail. With an
    exact factorization (``alpha >= delta``) the conjugate gradients are one or two
    steps of iterative refinement.

    Parameters
    ----------
    f : ndarray
        Residuals.
    J : CompressedJacobian
        Jacobian.
    D, E : ndarray
        Block tridiagonal ``J.T@J`` from ``J.normal_blocks()``.
    trust_radius : float
        Radius of the trust region.
    initial_alpha : float
        Initial guess for the Levenberg-Marquardt parameter.
    rtol : float
        Relative tolerance on the step length when the step is on the boundary.
    max_iter : int
        Maximum number of iterations for alpha.
    cg_tol : float
        Conjugate gradients stop when the residual of the normal equations is below
        ``cg_tol * norm(J.T@f)``.
    cg_maxiter : int
        Maximum number of conjugate gradient iterations per solve.

    Returns
    -------
    p : ndarray
        Step.
    hits_boundary : bool
        Whether the step is on the boundary of the trust region.
    alpha : float
        Levenberg-Marquardt parameter of the step.

    """
    st = J.structure
    n = st.num_vars
    g = J.rmatvec(f)
    real = jnp.asarray(np.arange(st.block_size)[None, :] < st.block_sizes[:, None])
    diag_max = jnp.max(jnp.where(real, jnp.diagonal(D, axis1=1, axis2=2), 0.0))
    delta = 4 * n * jnp.finfo(D.dtype).eps * diag_max

    def solve(alpha):
        # The factorization is only a preconditioner, so it can use a shift of at
        # least delta: B + alpha*I can be singular to working precision when alpha
        # is below delta, and the conjugate gradients make up for the difference.
        shift = jnp.maximum(alpha, delta)
        Ld, Lo = _block_cholesky(D, E, jnp.where(real, shift, 0.0))
        for _ in range(3):
            if not bool(_block_chol_failed(Ld, real)):
                break
            # can only happen through rounding errors in D, E
            shift = 100 * shift
            Ld, Lo = _block_cholesky(D, E, jnp.where(real, shift, 0.0))

        def apply(rhs):
            return st.from_blocks(_block_solve(Ld, Lo, st.to_blocks(rhs)))

        p, _, _ = _pcg(st)(J.data, Ld, Lo, g, alpha, cg_tol, cg_maxiter)
        return p, apply

    p, _ = solve(0.0)
    if bool(jnp.all(jnp.isfinite(p))) and float(jnp.linalg.norm(p)) <= trust_radius:
        return p, False, 0.0

    alpha_upper = float(jnp.linalg.norm(g)) / trust_radius
    alpha_lower = 0.0
    alpha = float(np.clip(initial_alpha, alpha_lower, alpha_upper))
    for _ in range(max_iter):
        p, apply = solve(alpha)
        p_norm = float(jnp.linalg.norm(p))
        phi = p_norm - trust_radius
        if phi < 0:
            alpha_upper = alpha
        else:
            alpha_lower = alpha
        if abs(phi) <= rtol * trust_radius:
            break
        # ||q||^2 = p.T @ inv(J.T@J + alpha*I) @ p
        q_norm2 = float(p @ apply(p))
        alpha += p_norm**2 / q_norm2 * phi / trust_radius
        alpha = float(np.clip(alpha, alpha_lower, alpha_upper))
    p = p * (trust_radius / jnp.linalg.norm(p))
    return p, True, alpha
