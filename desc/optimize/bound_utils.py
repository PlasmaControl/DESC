"""Utilities for dealing with simple bound constraints."""

from desc.backend import jnp

from .utils import evaluate_quadratic_form_hess, evaluate_quadratic_form_jac


def cl_scaling_vector(x, g, lb, ub):
    """Compute Coleman-Li scaling vector and its derivatives.

    Components of a vector v are defined as follows:
    ::
               | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
        v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
               | 1,           otherwise

    According to this definition v[i] >= 0 for all i. It differs from the
    definition in paper [1]_ (eq. (2.2)), where the absolute value of v is
    used. Both definitions are equivalent down the line.
    Derivatives of v with respect to x take value 1, -1 or 0 depending on a
    case.

    Parameters
    ----------
    x : ndarray
        state vector
    g : ndarray
        gradient of objective function
    lb, ub : ndarray
        lower and upper bounds on x

    Returns
    -------
    v : ndarray with shape of x
        Scaling vector.
    dv : ndarray with shape of x
        Derivatives of v[i] with respect to x[i], diagonal elements of v's
        Jacobian.

    References
    ----------
    .. [1] M.A. Branch, T.F. Coleman, and Y. Li, "A Subspace, Interior,
           and Conjugate Gradient Method for Large-Scale Bound-Constrained
           Minimization Problems," SIAM Journal on Scientific Computing,
           Vol. 21, Number 1, pp 1-23, 1999.
    """
    v = jnp.ones_like(x)
    dv = jnp.zeros_like(x)

    mask = (g < 0) & jnp.isfinite(ub)
    v = jnp.where(mask, ub - x, v)
    dv = jnp.where(mask, -1, dv)

    mask = (g > 0) & jnp.isfinite(lb)
    v = jnp.where(mask, x - lb, v)
    dv = jnp.where(mask, 1, dv)

    return v, dv


def step_size_to_bound(x, s, lb, ub):
    """Compute a min_step size required to reach a bound.

    The function computes a positive scalar t, such that x + s * t is on
    the bound.

    Parameters
    ----------
    x : ndarray
        state vector
    s : ndarray
        proposed step
    lb, ub : ndarray
        lower and upper bounds on x

    Returns
    -------
    step : float
        Computed step. Non-negative value.
    hits : ndarray of int with shape of x
        Each element indicates whether a corresponding variable reaches the
        bound:
             *  0 - the bound was not hit.
             * -1 - the lower bound was hit.
             *  1 - the upper bound was hit.
    """
    steps = jnp.inf * jnp.ones(x.size)
    mask = s != 0
    steps = jnp.where(mask, jnp.maximum((lb - x) / s, (ub - x) / s), steps)
    min_step = jnp.min(steps)
    return min_step, jnp.equal(steps, min_step) * jnp.sign(s).astype(int)


def find_active_constraints(x, lb, ub, rtol=1e-10):
    """Determine which constraints are active in a given point.

    The threshold is computed using `rtol` and the absolute value of the
    closest bound.

    Parameters
    ----------
    x : ndarray
        state vector
    lb, ub : ndarray
        lower and upper bounds on x

    Returns
    -------
    active : ndarray of int with shape of x
        Each component shows whether the corresponding constraint is active:
             *  0 - a constraint is not active.
             * -1 - a lower bound is active.
             *  1 - a upper bound is active.
    """
    active = jnp.zeros_like(x, dtype=int)

    if rtol == 0:
        active = jnp.where(x <= lb, -1, active)
        active = jnp.where(x >= ub, 1, active)
        return active

    lower_dist = x - lb
    upper_dist = ub - x

    lower_threshold = rtol * jnp.maximum(1, jnp.abs(lb))
    upper_threshold = rtol * jnp.maximum(1, jnp.abs(ub))

    lower_active = jnp.isfinite(lb) & (
        lower_dist <= jnp.minimum(upper_dist, lower_threshold)
    )
    active = jnp.where(lower_active, -1, active)

    upper_active = jnp.isfinite(ub) & (
        upper_dist <= jnp.minimum(lower_dist, upper_threshold)
    )
    active = jnp.where(upper_active, 1, active)

    return active


def make_strictly_feasible(x, lb, ub, rstep=1e-10):
    """Shift a point to the interior of a feasible region.

    Each element of the returned vector is at least at a relative distance
    `rstep` from the closest bound. If ``rstep=0`` then `np.nextafter` is used.
    """
    active = find_active_constraints(x, lb, ub, rstep)
    lower_mask = jnp.equal(active, -1)
    upper_mask = jnp.equal(active, 1)

    if rstep == 0:
        lwr = jnp.nextafter(lb, ub)
        upr = jnp.nextafter(ub, lb)
    else:
        lwr = lb + rstep * jnp.maximum(1, jnp.abs(lb))
        upr = ub - rstep * jnp.maximum(1, jnp.abs(ub))

    x_new = x.copy()
    x_new = jnp.where(lower_mask, lwr, x_new)
    x_new = jnp.where(upper_mask, upr, x_new)
    tight_bounds = (x_new < lb) | (x_new > ub)
    x_new = jnp.where(tight_bounds, 0.5 * (lb + ub), x_new)

    return x_new


def in_bounds(x, lb, ub):
    """Check if a point lies within bounds."""
    return jnp.all((x >= lb) & (x <= ub))


def select_step(x, JorH, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta, mode="jac"):
    """Select the best step according to Trust Region Reflective algorithm."""
    assert mode in ["jac", "hess"]

    if mode == "jac":
        evaluate_quadratic_form = evaluate_quadratic_form_jac
        build_quadratic_1d = build_quadratic_1d_jac
    else:
        evaluate_quadratic_form = evaluate_quadratic_form_hess
        build_quadratic_1d = build_quadratic_1d_hess

    if in_bounds(x + p, lb, ub):
        p_value = evaluate_quadratic_form(JorH, g_h, p_h, diag=diag_h)
        return p, p_h, -p_value

    p_stride, hits = step_size_to_bound(x, p, lb, ub)

    # Compute the reflected direction.
    r_h = jnp.copy(p_h)
    r_h = jnp.where(hits.astype(bool), r_h * -1, r_h)
    r = d * r_h

    # Restrict trust-region step, such that it hits the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Reflected direction will cross first either feasible region or trust region
    # boundary.
    _, to_tr = intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # Find lower and upper bounds on a step size along the reflected
    # direction, considering the strict feasibility requirement. There is no
    # single correct way to do that, the chosen approach seems to work best
    # on test problems.
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1

    # Check if reflection step is available.
    if r_stride_l <= r_stride_u:
        a, b, c = build_quadratic_1d(JorH, g_h, r_h, s0=p_h, diag=diag_h)
        r_stride, r_value = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = jnp.inf

    # Now correct p_h to make it strictly interior.
    p *= theta
    p_h *= theta
    p_value = evaluate_quadratic_form(JorH, g_h, p_h, diag=diag_h)

    ag_h = -g_h
    ag = d * ag_h

    to_tr = Delta / jnp.linalg.norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr:
        ag_stride = theta * to_bound
    else:
        ag_stride = to_tr

    a, b = build_quadratic_1d(JorH, g_h, ag_h, diag=diag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    elif r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    else:
        return ag, ag_h, -ag_value


def build_quadratic_1d_jac(J, g, s, diag=None, s0=None):
    """Parameterize a multivariate quadratic function along a line.

    The resulting univariate quadratic function is given as follows:
    ::
        f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) +
               g.T * (s0 + s*t)

    Parameters
    ----------
    J : ndarray, shape (m, n)
        Jacobian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (n,)
        Direction vector of a line.
    diag : None or ndarray with shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.
    s0 : None or ndarray with shape (n,), optional
        Initial point. If None, assumed to be 0.

    Returns
    -------
    a : float
        Coefficient for t**2.
    b : float
        Coefficient for t.
    c : float
        Free term. Returned only if `s0` is provided.
    """
    v = J.dot(s)
    a = jnp.dot(v, v)
    if diag is not None:
        a += jnp.dot(s * diag, s)
    a *= 0.5

    b = jnp.dot(g, s)

    if s0 is not None:
        u = J.dot(s0)
        b += jnp.dot(u, v)
        c = 0.5 * jnp.dot(u, u) + jnp.dot(g, s0)
        if diag is not None:
            b += jnp.dot(s0 * diag, s)
            c += 0.5 * jnp.dot(s0 * diag, s0)
        return a, b, c
    else:
        return a, b


def build_quadratic_1d_hess(H, g, s, diag=None, s0=None):
    """Parameterize a multivariate quadratic function along a line.

    The resulting univariate quadratic function is given as follows:
    ::
        f(t) = 0.5 * (s0 + s*t).T * (H + diag) * (s0 + s*t) +
               g.T * (s0 + s*t)

    Parameters
    ----------
    H : ndarray, shape (n, n)
        Hessian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (n,)
        Direction vector of a line.
    diag : None or ndarray with shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.
    s0 : None or ndarray with shape (n,), optional
        Initial point. If None, assumed to be 0.

    Returns
    -------
    a : float
        Coefficient for t**2.
    b : float
        Coefficient for t.
    c : float
        Free term. Returned only if `s0` is provided.
    """
    a = H.dot(s).dot(s)
    if diag is not None:
        a += jnp.dot(s * diag, s)
    a *= 0.5

    b = jnp.dot(g, s)

    if s0 is not None:
        u = H.dot(s0)
        b += jnp.dot(u, s)
        c = 0.5 * jnp.dot(u, s0) + jnp.dot(g, s0)
        if diag is not None:
            b += jnp.dot(s0 * diag, s)
            c += 0.5 * jnp.dot(s0 * diag, s0)
        return a, b, c
    else:
        return a, b


def minimize_quadratic_1d(a, b, lb, ub, c=0):
    """Minimize a 1-D quadratic function subject to bounds.

    The free term `c` is 0 by default. Bounds must be finite.

    Returns
    -------
    t : float
        Minimum point.
    y : float
        Minimum value.
    """
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = jnp.asarray(t)
    y = t * (a * t + b) + c
    min_index = jnp.argmin(y)
    return t[min_index], y[min_index]


def intersect_trust_region(x, s, Delta):
    """Find the intersection of a line with the boundary of a trust region.

    This function solves the quadratic equation with respect to t

    ||(x + s*t)||**2 = Delta**2.

    Returns
    -------
    t_neg, t_pos : tuple of float
        Negative and positive roots.

    Raises
    ------
    AssertionError
        If `s` is zero or `x` is not within the trust region.
    """
    a = jnp.dot(s, s)
    assert a != 0, "`s` is zero."

    b = jnp.dot(x, s)

    c = jnp.dot(x, x) - Delta**2
    assert c <= 0, f"`x` is not within the trust region, c={c}"

    d = jnp.sqrt(b * b - a * c)  # Root from one fourth of the discriminant.

    q = -(b + jnp.sign(b) * jnp.abs(d))
    t1 = q / a
    t2 = c / q

    if t1 < t2:
        return t1, t2
    else:
        return t2, t1
