import numpy as np
from desc.backend import jnp, cho_factor, cho_solve, qr


def solve_trust_region_dogleg(g, hess, scale, trust_radius, f=None):
    """
    Solve trust region subproblem the dog-leg method.

    Parameters
    ----------
    g : ndarray
        gradient of objective function
    hess : Hessian
        Hessian with dot and solve methods
    scale : ndarray
        scaling array for gradient and hessian
    trust_radius : float
        We are allowed to wander only this far away from the origin.
    f : ndarray, optional
        function values for least squares dogleg step

    Returns
    -------
    p : ndarray
        The proposed step.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.

    Notes
    -----
    The Hessian is required to be positive definite.

    """

    # This is the optimum for the quadratic model function.
    # If it is inside the trust radius then return this point.
    if f is None:
        p_newton = -1 / scale * hess.solve(g)
    else:
        p_newton = -1 / scale * hess.solve(f)
    if jnp.linalg.norm(p_newton) < trust_radius:
        hits_boundary = False
        return p_newton, hits_boundary

    # This is the predicted optimum along the direction of steepest descent.
    gBg = hess.quadratic(scale ** 2 * g, scale ** 2 * g)
    p_cauchy = -(jnp.dot(scale * g, scale * g) / gBg) * scale * g

    # If the Cauchy point is outside the trust region,
    # then return the point where the path intersects the boundary.
    p_cauchy_norm = jnp.linalg.norm(p_cauchy)
    if p_cauchy_norm >= trust_radius:
        p_boundary = p_cauchy * (trust_radius / p_cauchy_norm)
        hits_boundary = True
        return p_boundary, hits_boundary

    # Compute the intersection of the trust region boundary
    # and the line segment connecting the Cauchy and Newton points.
    # This requires solving a quadratic equation.
    # ||p_u + t*(p_best - p_u)||**2 == trust_radius**2
    # Solve this for positive time t using the quadratic formula.
    delta = p_newton - p_cauchy
    _, tb = get_boundaries_intersections(p_cauchy, delta, trust_radius)
    p_boundary = p_cauchy + tb * delta
    hits_boundary = True

    return p_boundary, hits_boundary


def solve_trust_region_2d_subspace(grad, hess, scale, trust_radius, f=None):
    """Solve a trust region problem over the 2d subspace spanned by the gradient
    and Newton direction

    Parameters
    ----------
    grad : ndarray
        gradient of objective function
    hess : OptimizerDerivative
        hessian of objective function
    scale : ndarray
        scaling array for gradient and hessian
    trust_radius : float
        We are allowed to wander only this far away from the origin.
    f : ndarray, optional
        function values for least squares subspace step

    Returns
    -------
    p : ndarray
        The proposed step.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.

    """
    if hess.is_pos_def and f is None:
        p_newton = -1 / scale * hess.solve(grad)
    elif f is None:
        p_newton = hess.negative_curvature_direction
    else:
        p_newton = -1 / scale * hess.solve(f)

    S = np.vstack([grad, p_newton]).T
    S, _ = qr(S, mode="economic")
    g = S.T.dot(scale * grad)
    B = S.T.dot(scale[:, jnp.newaxis] * hess.dot(scale[:, jnp.newaxis] * S))

    # B = [a b]  g = [d f]
    #     [b c]  q = [x y]
    # p = Sq

    try:
        R, lower = cho_factor(B)
        q = -cho_solve((R, lower), g)
        if np.dot(q, q) <= trust_radius ** 2:
            return S.dot(q), True
    except np.linalg.linalg.LinAlgError:
        pass

    a = B[0, 0] * trust_radius ** 2
    b = B[0, 1] * trust_radius ** 2
    c = B[1, 1] * trust_radius ** 2

    d = g[0] * trust_radius
    f = g[1] * trust_radius

    coeffs = np.array([-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = np.roots(coeffs)  # Can handle leading zeros.
    t = np.real(t[np.isreal(t)])

    q = trust_radius * np.vstack((2 * t / (1 + t ** 2), (1 - t ** 2) / (1 + t ** 2)))
    value = 0.5 * np.sum(q * B.dot(q), axis=0) + np.dot(g, q)
    i = np.argmin(value)
    q = q[:, i]
    p = S.dot(q)

    return p, False


# not used yet, need to get some other stuff working first
# TODO: give this the same signature as the others?


def solve_lsq_trust_region(
    n, m, uf, s, V, trust_radius, initial_alpha=None, rtol=0.01, max_iter=10
):  # pragma: no cover
    """Solve a trust-region problem arising in least-squares minimization.
    This function implements a method described by J. J. More [1]_ and used
    in MINPACK, but it relies on a single SVD of Jacobian instead of series
    of Cholesky decompositions. Before running this function, compute:
    ``U, s, VT = svd(J, full_matrices=False)``.
    Parameters
    ----------
    n : int
        Number of variables.
    m : int
        Number of residuals.
    uf : ndarray
        Computed as U.T.dot(f).
    s : ndarray
        Singular values of J.
    V : ndarray
        Transpose of VT.
    trust_radius : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy ``abs(norm(p) - trust_radius) < rtol * trust_radius``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.
    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    alpha : float
        Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
        Sometimes called Levenberg-Marquardt parameter.
    n_iter : int
        Number of iterations made by root-finding procedure. Zero means
        that Gauss-Newton step was selected as the solution.
    References
    ----------
    .. [1] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
           and Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes
           in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    """

    def phi_and_derivative(alpha, suf, s, trust_radius):
        """Function of which to find zero.
        It is defined as "norm of regularized (by alpha) least-squares
        solution minus `trust_radius`". Refer to [1]_.
        """
        denom = s ** 2 + alpha
        p_norm = np.linalg.norm(suf / denom)
        phi = p_norm - trust_radius
        phi_prime = -np.sum(suf ** 2 / denom ** 3) / p_norm
        return phi, phi_prime

    suf = s * uf

    # Check if J has full rank and try Gauss-Newton step.
    threshold = EPS * m * s[0]
    full_rank = s[-1] > threshold
    if full_rank:
        p = -V.dot(uf / s)
        if norm(p) <= trust_radius:
            return p, 0.0, 0

    alpha_upper = norm(suf) / trust_radius

    if full_rank:
        phi, phi_prime = phi_and_derivative(0.0, suf, s, trust_radius)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0

    if initial_alpha is None or not full_rank and initial_alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
    else:
        alpha = initial_alpha

    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)

        phi, phi_prime = phi_and_derivative(alpha, suf, s, trust_radius)

        if phi < 0:
            alpha_upper = alpha

        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + trust_radius) * ratio / trust_radius

        if np.abs(phi) < rtol * trust_radius:
            break

    p = -V.dot(suf / (s ** 2 + alpha))

    # Make the norm of p equal to trust_radius, p is changed only slightly during
    # this. It is done to prevent p lie outside the trust region (which can
    # cause problems later).
    p *= trust_radius / norm(p)

    return p, alpha, it + 1


def update_tr_radius(
    trust_radius,
    actual_reduction,
    predicted_reduction,
    step_norm,
    bound_hit,
    max_tr=np.inf,
    min_tr=0,
    increase_threshold=0.75,
    increase_ratio=2,
    decrease_threshold=0.25,
    decrease_ratio=0.25,
    ga_ratio=0,
    ga_accept_threshold=1,
):
    """Update the radius of a trust region based on the cost reduction.

    Parameters
    ----------
    trust_radius : float
        current trust region radius
    actual_reduction : float
        actual cost reduction from the proposed step
    predicted_reduction : float
        cost reduction predicted by quadratic model
    step_norm : float
        size of the proposed step
    bound_hit : bool
        whether the current step hits the trust region bound
    max_tr : float
        maximum allowed trust region radius
    min_tr : float
        minimum allowed trust region radius
    increase_threshold, increase_ratio : float
        if ratio > inrease_threshold, trust radius is increased by a factor of increase_ratio
    decrease_threshold, decrease_ratio : float
        if ratio < decrease_threshold, trust radius is decreased by a factor of decrease_ratio
    ga_ratio : float
        ratio of geodesic acceleration step size to original step size
    ga_accept_threshold: float
        only accept step if ga_ratio < ga_accept_threshold


    Returns
    -------
    trust_radius : float
        New radius.
    ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0

    if ratio < decrease_threshold:
        trust_radius = decrease_ratio * step_norm
    elif ratio > increase_threshold and bound_hit:
        trust_radius *= increase_ratio

    trust_radius = np.clip(trust_radius, min_tr, max_tr)

    return trust_radius, ratio


def get_boundaries_intersections(z, d, trust_radius):
    """
    Solve the scalar quadratic equation ||z + t d|| == trust_radius.
    This is like a line-sphere intersection.
    Return the two values of t, sorted from low to high.
    """
    a = jnp.dot(d, d)
    b = 2 * jnp.dot(z, d)
    c = jnp.dot(z, z) - trust_radius ** 2
    sqrt_discriminant = jnp.sqrt(b * b - 4 * a * c)

    # The following calculation is mathematically
    # equivalent to:
    # ta = (-b - sqrt_discriminant) / (2*a)
    # tb = (-b + sqrt_discriminant) / (2*a)
    # but produce smaller round off errors.
    # Look at Matrix Computation p.97
    # for a better justification.
    aux = b + jnp.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    return np.sort(np.array([ta, tb]))
