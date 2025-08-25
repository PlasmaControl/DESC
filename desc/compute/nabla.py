"""
Functions for taking the curl and divergence in cylindrical coordinates.

Partial derivatives are calculated using spectral methods. Both functions
have an identical API that requires as inputs the input and output Transform
objects to and from spectral space, assumed to be for the same basis.
"""

from desc.backend import jnp


def curl_cylindrical(A, in_R, out_R, in_transform, out_transform, scales):
    """Take the curl of A in cylindrical coordinates.

    Uses spectral methods to take partial derivatives of A and calculate
    its curl on the grid associated with out_transform.

    Parameters
    ----------
    A : ndarray, shape(n,3)
        vector field, in cylindrical (R,phi,Z) form
    in_R : ndarray, shape(n,)
        radial distance for each point of A
    out_R : ndarray, shape(n,)
        radial location for each point at which to calculate
        the curl. out_R should never be 0.
    in_transform: Transform
        transform from the real grid on which A is defined to
        a spectral basis in which partial derivatives of A can be evaluated.
        the pseudoinverse of the transform must have been built.
    out_transform: Transform
        transform from the spectral basis on which A is calculated to the
        real grid on which the curl is to be evaluated.
        the transform must have been built, with derivs>=1.
    scales: jnp.ndarray, shape (3,)
        This parameter adjusts the dimensions of the partial derivatives so they are
        taken with respect to the normal coordinates, not the coordinates in the
        transform.

    Returns
    -------
    curl_A : ndarray, shape(n,3)
        The curl of the vector field, in cylindrical coordinates.
    """
    A_coeff = in_transform.fit(A)
    RA_phi_coeff = in_transform.fit(in_R * A[:, 1])
    return _curl_cylindrical(A_coeff, RA_phi_coeff, out_R, out_transform, scales)


def _curl_cylindrical(A_coeff, RA_phi_coeff, out_R, out_transform, scales):
    """Take the curl of A, using precomputed spectral coeffs.

    Parameters
    ----------
    A_coeff : ndarray, shape(num_modes,3)
        Spectral coefficients for A.
    RA_phi_coeff : ndarray, shape(num_modes,)
        Spectral coefficients for R*A_phi.
    out_R : ndarray, shape(n,)
        radial location for each point at which to calculate
        the curl. out_R should never be zero.
    out_transform: Transform
        transform from the spectral basis on which A is calculated to the
        real grid on which the curl is to be evaluated.
        the transform must have been built, with derivs>=1.
    scales: jnp.ndarray, shape (3,)
        This parameter adjusts the dimensions of the partial derivatives so they are
        taken with respect to the normal coordinates, not the coordinates in the
        transform.

    Returns
    -------
    curl_A : ndarray, shape(n,3)
        The curl of the vector field, in cylindrical coordinates.
    """
    # Calculate matrix of terms for the curl
    # (dims: datapoint, component index, derivative index)
    terms = jnp.zeros((out_R.shape[0], 3, 3))
    for c in range(3):
        for d in range(3):
            if c == 1 and d == 0:
                # partial (R*A_phi)/partial R instead of partial A_phi/partial R
                term_cd = out_transform.transform(RA_phi_coeff, dr=1)
                terms = terms.at[:, c, d].set(term_cd)
            elif c != d:
                # partial A_c/partial r_d
                term_cd = out_transform.transform(
                    A_coeff[:, c], dr=(d == 0), dt=(d == 1), dz=(d == 2)
                )
                terms = terms.at[:, c, d].set(term_cd)
    # Rescale derivatives
    terms = terms / scales.reshape(1, 1, -1)

    # Calculate curl from the partial derivatives
    # (curl(A))_R = 1/R partial A_z/partial A_phi - partial A_phi/partial z
    curl_A_R = 1 / out_R * terms[:, 2, 1] - terms[:, 1, 2]

    # (curl(A))_phi = partial A_R/partial A_z - partial A_z/partial R
    curl_A_phi = terms[:, 0, 2] - terms[:, 2, 0]

    # (curl(A))_z = 1/R(partial(R A_phi)/partial R - partial A_R/partial phi)
    curl_A_z = 1 / out_R * (terms[:, 1, 0] - terms[:, 0, 1])

    curl_A = jnp.stack([curl_A_R, curl_A_phi, curl_A_z], axis=-1)
    return curl_A


def div_cylindrical(A, in_R, out_R, in_transform, out_transform, scales):
    """
    Take the divergence of A in cylindrical coordinates.

    Uses spectral methods to take partial derivatives of A and calculate
    its divergence on the grid associated with out_transform.

    Parameters
    ----------
    A : ndarray, shape(n,3)
        vector field, in cylindrical (R,phi,Z) form
    in_R : ndarray, shape(n,)
        radial distance for each point of A
    out_R : ndarray, shape(n,)
        radial location for each point at which to calculate
        the curl. out_R should never be 0.
    in_transform: Transform
        transform from the real grid on which A is defined to
        a spectral basis in which partial derivatives of A can be evaluated.
        the pseudoinverse of the transform must have been built.
    out_transform: Transform
        transform from the spectral basis on which A is calculated to the
        real grid on which the divergence is to be evaluated.
        the transform must have been built, with derivs>=1.
    scales: jnp.ndarray, shape (3,)
        This parameter adjusts the dimensions of the partial derivatives so they
        are taken with respect to the normal coordinates, not the coordinates
        in the transform.

    Returns
    -------
    div_A : ndarray, shape(n,)
        The divergence of the vector field, in cylindrical coordinates.
    """
    A_coeff = in_transform.fit(
        A * jnp.stack([in_R, jnp.ones_like(in_R), jnp.ones_like(in_R)], axis=-1)
    )
    # Calculate matrix of terms for the curl
    # (dims: datapoint, component index, derivative index)
    terms = jnp.zeros((out_R.shape[0], 3))
    for c in range(3):
        # partial A_c/partial r_d
        term_c = out_transform.transform(
            A_coeff[:, c], dr=(c == 0), dt=(c == 1), dz=(c == 2)
        )
        terms = terms.at[:, c].set(term_c)
    # Rescale derivatives
    terms = (
        terms
        * jnp.stack([1 / out_R, 1 / out_R, jnp.ones_like(out_R)], axis=-1)
        / scales.reshape(1, -1)
    )

    # Calculate divergence from the partial derivatives
    div = terms.sum(axis=1)
    return div
