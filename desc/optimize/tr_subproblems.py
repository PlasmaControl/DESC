"""Functions for solving subproblems arising in trust region methods."""

import numpy as np

from desc.backend import cho_factor, cho_solve, jnp, qr, solve_triangular

from .utils import chol


def solve_trust_region_dogleg(g, H, trust_radius, initial_alpha=None, **kwargs):
    """Solve trust region subproblem using the dog-leg method.

    Parameters
    ----------
    g : ndarray
        gradient of objective function
    H : ndarray
        Hessian matrix
    trust_radius : float
        We are allowed to wander only this far away from the origin.
    initial_alpha : float
        initial guess for levenberg-marquadt parameter - unused by this method.

    Returns
    -------
    p : ndarray
        The proposed step.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        "levenberg-marquadt" parameter - unused by this method.

    """
    L = chol(H)
    # This is the optimum for the quadratic model function.
    # If it is inside the trust radius then return this point.
    p_newton = -cho_solve((L, True), g)
    if jnp.linalg.norm(p_newton) < trust_radius:
        hits_boundary = False
        return p_newton, hits_boundary, initial_alpha

    # This is the predicted optimum along the direction of steepest descent.
    gBg = g @ H @ g
    p_cauchy = -(jnp.dot(g, g) / gBg) * g

    # If the Cauchy point is outside the trust region,
    # then return the point where the path intersects the boundary.
    p_cauchy_norm = jnp.linalg.norm(p_cauchy)
    if p_cauchy_norm >= trust_radius:
        p_boundary = p_cauchy * (trust_radius / p_cauchy_norm)
        hits_boundary = True
        return p_boundary, hits_boundary, initial_alpha

    # Compute the intersection of the trust region boundary
    # and the line segment connecting the Cauchy and Newton points.
    # This requires solving a quadratic equation.
    # ||p_u + t*(p_best - p_u)||**2 == trust_radius**2
    # Solve this for positive time t using the quadratic formula.
    delta = p_newton - p_cauchy
    _, tb = get_boundaries_intersections(p_cauchy, delta, trust_radius)
    p_boundary = p_cauchy + tb * delta
    hits_boundary = True

    return p_boundary, hits_boundary, initial_alpha


def solve_trust_region_2d_subspace(g, H, trust_radius, initial_alpha=None, **kwargs):
    """Solve a trust region problem using 2d subspace method.

    Minimizes model function over subspace spanned by the gradient
    and Newton direction

    Parameters
    ----------
    g : ndarray
        gradient of objective function
    H : ndarray
        Hessian of objective function
    trust_radius : float
        We are allowed to wander only this far away from the origin.
    initial_alpha : float
        initial guess for levenberg-marquadt parameter - unused by this method

    Returns
    -------
    p : ndarray
        The proposed step.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        "levenberg-marquadt" parameter - unused by this method

    """
    L = chol(H)
    # This is the optimum for the quadratic model function.
    p_newton = -cho_solve((L, True), g)

    S = np.vstack([g, p_newton]).T
    S, _ = qr(S, mode="economic")
    g = S.T @ g
    B = S.T @ H @ S

    # B = [a b]  g = [d f]
    #     [b c]  q = [x y]
    # p = Sq                    # noqa: E800

    try:
        R, lower = cho_factor(B)
        q = -cho_solve((R, lower), g)
        if np.dot(q, q) <= trust_radius**2:
            return S.dot(q), True, initial_alpha
    except np.linalg.linalg.LinAlgError:
        pass

    a = B[0, 0] * trust_radius**2
    b = B[0, 1] * trust_radius**2
    c = B[1, 1] * trust_radius**2

    d = g[0] * trust_radius
    f = g[1] * trust_radius

    coeffs = np.array([-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = np.roots(coeffs)  # Can handle leading zeros.
    t = np.real(t[np.isreal(t)])

    q = trust_radius * np.vstack((2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)))
    value = 0.5 * np.sum(q * B.dot(q), axis=0) + np.dot(g, q)
    i = np.argmin(value)
    q = q[:, i]
    p = S.dot(q)

    return p, False, initial_alpha


def trust_region_step_exact_svd(
    f, u, s, v, trust_radius, initial_alpha=None, rtol=0.01, max_iter=10, threshold=None
):
    """Solve a trust-region problem using a semi-exact method.

    Solves problems of the form
        min_p ||J*p + f||^2,  ||p|| < trust_radius

    Parameters
    ----------
    f : ndarray
        Vector of residuals
    u : ndarray
        Left singular vectors of J.
    s : ndarray
        Singular values of J.
    v : ndarray
        Right singular vectors of J (eg transpose of VT).
    trust_radius : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy
        ``abs(norm(p) - trust_radius) < rtol * trust_radius``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.
    threshold : float
        relative cutoff for small singular values

    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
        Sometimes called Levenberg-Marquardt parameter.

    """
    uf = u.T.dot(f)
    suf = s * uf

    def phi_and_derivative(alpha, suf, s, trust_radius):
        """Function of which to find zero.

        It is defined as "norm of regularized (by alpha) least-squares
        solution minus `trust_radius`".
        """
        denom = s**2 + alpha
        p_norm = np.linalg.norm(suf / denom)
        phi = p_norm - trust_radius
        phi_prime = -np.sum(suf**2 / denom**3) / p_norm
        return phi, phi_prime

    # Check if J has full rank and try Gauss-Newton step.
    if threshold is None:
        threshold = np.finfo(s.dtype).eps * f.size * s[0]
    else:
        threshold *= s[0]
    large = s > threshold
    s_inv = np.divide(1, s, where=large)
    s_inv[(~large,)] = 0

    p = -v.dot(uf * s_inv)
    if np.linalg.norm(p) <= trust_radius:
        return p, False, 0.0

    alpha_upper = np.linalg.norm(suf) / trust_radius
    alpha_lower = 0.0

    if initial_alpha is None or initial_alpha == 0:
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

    p = -v.dot(suf / (s**2 + alpha))

    # Make the norm of p equal to trust_radius; p is changed only slightly during this.
    # This is done to prevent p from lying outside the trust region
    # (which can cause problems later).
    p *= trust_radius / np.linalg.norm(p)

    return p, True, alpha


def trust_region_step_exact_cho(
    g, B, trust_radius, initial_alpha=None, rtol=0.01, max_iter=10
):
    """Solve a trust-region problem using a semi-exact method.

    Solves problems of the form
        (B + alpha*I)*p = -g,  ||p|| < trust_radius
    for symmetric B. A modified Cholesky factorization is used to deal with indefinite B

    Parameters
    ----------
    g : ndarray
        gradient vector
    B : ndarray
        Hessian or approximate Hessian
    trust_radius : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy
        ``abs(norm(p) - trust_radius) < rtol * trust_radius``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.

    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        Positive value such that (B + alpha*I)*p = -g.
        Sometimes called Levenberg-Marquardt parameter.

    """
    # try full newton step
    R = chol(B)
    p = cho_solve((R, True), -g)
    if np.linalg.norm(p) <= trust_radius:
        return p, False, 0.0

    alpha_upper = np.linalg.norm(g) / trust_radius
    alpha_lower = 0.0

    if initial_alpha is None or initial_alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
    else:
        alpha = initial_alpha

    # algorithm 4.3 from Nocedal & Wright
    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)

        Bi = B + alpha * jnp.eye(B.shape[0])
        R = chol(Bi)
        p = cho_solve((R, True), -g)
        p_norm = np.linalg.norm(p)
        phi = p_norm - trust_radius
        if phi < 0:
            alpha_upper = alpha
        if phi > 0:
            alpha_lower = alpha

        q = solve_triangular(R.T, p, lower=False)
        q_norm = np.linalg.norm(q)

        alpha += (p_norm / q_norm) ** 2 * phi / trust_radius
        if np.abs(phi) < rtol * trust_radius:
            break

    Bi = B + alpha * jnp.eye(B.shape[0])
    R = chol(Bi)
    p = cho_solve((R, True), -g)

    # Make the norm of p equal to trust_radius; p is changed only slightly during this.
    # This is done to prevent p from lying outside the trust region
    # (which can cause problems later).
    p *= trust_radius / np.linalg.norm(p)

    return p, True, alpha


def update_tr_radius(
    trust_radius,
    actual_reduction,
    predicted_reduction,
    step_norm,
    bound_hit,
    max_tr=np.inf,
    increase_threshold=0.75,
    increase_ratio=2,
    decrease_threshold=0.25,
    decrease_ratio=0.25,
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
    increase_threshold, increase_ratio : float
        if ratio > inrease_threshold, trust radius is increased by a factor
        of increase_ratio
    decrease_threshold, decrease_ratio : float
        if ratio < decrease_threshold, trust radius is decreased by a factor
        of decrease_ratio

    Returns
    -------
    trust_radius : float
        New radius.
    reduction_ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:
        reduction_ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        reduction_ratio = 1
    else:
        reduction_ratio = 0

    if reduction_ratio < decrease_threshold:
        trust_radius = decrease_ratio * step_norm
    elif reduction_ratio > increase_threshold:
        trust_radius = max(step_norm * increase_ratio, trust_radius)

    trust_radius = np.clip(trust_radius, 0, max_tr)

    return trust_radius, reduction_ratio


def get_boundaries_intersections(z, d, trust_radius):
    """Solve the scalar quadratic equation ||z + t d|| == trust_radius.

    This is like a line-sphere intersection.
    Return the two values of t, sorted from low to high.
    """
    a = jnp.dot(d, d)
    b = 2 * jnp.dot(z, d)
    c = jnp.dot(z, z) - trust_radius**2
    sqrt_discriminant = jnp.sqrt(b * b - 4 * a * c)

    # The following calculation is mathematically
    # equivalent to:
    # ta = (-b - sqrt_discriminant) / (2*a)    # noqa: E800
    # tb = (-b + sqrt_discriminant) / (2*a)    # noqa: E800
    # but produce smaller round off errors.
    # Look at Matrix Computation p.97
    # for a better justification.
    aux = b + jnp.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    return np.sort(np.array([ta, tb]))
