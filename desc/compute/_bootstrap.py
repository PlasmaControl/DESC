"""Compute functions for bootstrap current."""

from scipy.special import roots_legendre

from desc.backend import jnp, put

from .utils import check_derivs


def trapped_fraction(modB, sqrtg, n_gauss=20):
    r"""Evaluate the effective trapped particle fraction.

    Compute the effective fraction of trapped particles, which enters
    several formulae for neoclassical transport, as well as several
    quantities that go into its calculation.  The input data can be
    provided on a uniform grid of arbitrary toroidal and poloidal
    angles that need not be straight-field-line angles.

    The trapped fraction ``f_t`` has a standard definition in neoclassical theory:

    .. math::
        f_t = 1 - \frac{3}{4} \left< B^2 \right> \int_0^{1/Bmax}
            \frac{\lambda\; d\lambda}{\left< \sqrt{1 - \lambda B} \right>}

    where :math:`\left< \ldots \right>` is a flux surface average.

    The effective inverse aspect ratio epsilon is defined by

    .. math::
        \frac{Bmax}{Bmin} = \frac{1 + \epsilon}{1 - \epsilon}

    This definition is motivated by the fact that this formula would
    be true in the case of circular cross-section surfaces in
    axisymmetry with :math:`B \propto 1/R` and :math:`R = (1 +
    \epsilon \cos\theta) R_0`.

    Parameters
    ----------
    modB : array of size (ntheta, nphi, ns)
        :math:`|B|` on the grid points.
    sqrtg: array of size (ntheta, nphi, ns)
        The Jacobian :math:`1/(\nabla\rho\times\nabla\theta\cdot\nabla\zeta)`
        on the grid points.
    n_gauss: int
        Number of Gauss-Legendre integration points for the lambda integral.

    Returns
    -------
    data : dict
        Dictionary containing the following:
        - ``"Bmin"``: A 1D array, with the minimum of :math:`|B|` on each surface.
        - ``"Bmax"``: A 1D array, with the maximum of :math:`|B|` on each surface.
        - ``"epsilon"``: A 1D array, with the effective inverse aspect ratio on each surface.
        - ``"<1/B>"``: A 1D array with :math:`\left<B^2\right>` on each surface,
          where :math:`\left< \ldots \right>` denotes a flux surface average.
        - ``"<1/B>"``: A 1D array with :math:`\left<1/B\right>` on each surface,
          where :math:`\left< \ldots \right>` denotes a flux surface average.
        - ``"f_t"``: A 1D array, with the effective trapped fraction on each surface.
    """
    assert modB.shape == sqrtg.shape
    assert len(modB.shape) == 3
    nr = modB.shape[2]

    fourpisq = 4 * jnp.pi * jnp.pi
    d_V_d_rho = jnp.mean(sqrtg, axis=(0, 1)) / fourpisq
    fsa_B2 = jnp.mean(modB * modB * sqrtg, axis=(0, 1)) / (fourpisq * d_V_d_rho)
    fsa_1overB = jnp.mean(sqrtg / modB, axis=(0, 1)) / (fourpisq * d_V_d_rho)

    Bmax = jnp.max(modB, axis=(0, 1))
    Bmin = jnp.min(modB, axis=(0, 1))
    w = Bmax / Bmin
    epsilon = (w - 1) / (w + 1)

    # Get nodes and weights for Gauss-Legendre integration:
    base_nodes, base_weights = roots_legendre(n_gauss)
    
    f_t = jnp.zeros(nr)
    for jr in range(nr):
        # Shift and scale integration nodes and weights for the interval
        # [0, 1 / Bmax]:
        lambd = (base_nodes + 1) * 0.5 / Bmax[jr]
        weights = base_weights * 0.5 / Bmax[jr]
        
        # Evaluate <sqrt(1 - lambda B)>:
        flux_surf_avg_term = (jnp.mean(
            jnp.sqrt(1 - lambd[None, None, :] * modB[:, :, jr, None]) * sqrtg[:, :, jr, None],
            axis=(0, 1)) 
            / (fourpisq * d_V_d_rho[jr]))
        
        integrand = lambd / flux_surf_avg_term
        
        integral = jnp.sum(weights * integrand)
        
        f_t = put(f_t, jr, 1 - 0.75 * fsa_B2[jr] * integral)

    results = {
        "<B**2>": fsa_B2,
        "<1/B>": fsa_1overB,
        "Bmin": Bmin,
        "Bmax": Bmax,
        "epsilon": epsilon,
        "f_t": f_t,
    }
    return results
