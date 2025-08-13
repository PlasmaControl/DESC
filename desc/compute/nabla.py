import jax.numpy as jnp

def curl_cylindrical(A, in_R, out_R, in_transform, out_transform, scales):
    """
    Take the curl of A in cylindrical coordinates,
    given Transforms to and from spectral coordinates.

    Parameters
    ----------
    A : ndarray, shape(n,3)
        vector field, in cylindrical (R,phi,Z) form
    in_R : ndarray, shape(n,)
        radial distance for each point of A
    out_R : ndarray, shape(n,)
        radial location for each point at which to calculate
        the curl
    in_transform: Transform
        transform from the real grid on which A is defined to
        a spectral basis in which partial derivatives of A can be evaluated.
        the pseudoinverse of the transform must have been built.
    out_transform: Transform
        transform from the spectral basis on which A is calculated to the
        real grid on which the curl is to be evaluated.
        the transform must have been built, with derivs>=1.
    scales: jnp.ndarray, shape (3,)
        If the real coordinates in the transform object are scaled to be dimensionless,
        this parameter adjusts the dimensions of the partial derivatives
        so they are taken with the normal coordinates, not the coordinates in the transform.

    Returns
    -------
    curl_A : ndarray, shape(n,3)
        The curl of the vector field, in cylindrical coordinates.
    """
    A_coeff = in_transform.fit(A)
    # Calculate matrix of terms for the curl
    # (dims: datapoint, component index, derivative index)
    terms = jnp.zeros((out_R.shape[0], 3, 3))
    for c in range(3):
        for d in range(3):
            if c == 1 and d == 0:
                # partial (R*A_phi)/partial R instead of partial A_phi/partial R
                RA_phi_coeff = in_transform.fit(in_R * A[:, 1])
                terms[:, c, d] = out_transform.transform(RA_phi_coeff, dr=1)
            elif c != d:
                # partial A_c/partial r_d
                terms[:, c, d] = out_transform.transform(
                    A_coeff[:, c], dr=(d == 0), dt=(d == 1), dz=(d == 2)
                )
    # Rescale derivatives
    terms = terms / scales.reshape(1, 1, -1)

    # Calculate curl from the partial derivatives
    # (curl(A))_R = 1/R partial A_z/partial A_phi - partial A_phi/partial z
    curl_A_R = 1 / out_R * terms[:, 2, 1] - terms[:, 1, 2]

    # (curl(A))_phi = partial A_R/partial A_z - partial A_z/partial R
    curl_A_phi = terms[:, 0, 2] - terms[:, 2, 0]

    # (curl(A))_z = 1/R(partial(R A_phi)/partial R - partial A_R/partial phi)
    curl_A_z = 1 / out_R * (terms[:, 1, 0] - terms[:, 0, 1])

    curl_A = jnp.vstack([curl_A_R, curl_A_phi, curl_A_z]).T
    return curl_A


def div_cylindrical(A, in_R, out_R, in_transform, out_transform, scales):
    """
    Take the divergence of A in cylindrical coordinates,
    given Transforms to and from spectral coordinates.

    Parameters
    ----------
    A : ndarray, shape(n,3)
        vector field, in cylindrical (R,phi,Z) form
    in_R : ndarray, shape(n,)
        radial distance for each point of A
    out_R : ndarray, shape(n,)
        radial location for each point at which to calculate
        the curl
    in_transform: Transform
        transform from the real grid on which A is defined to
        a spectral basis in which partial derivatives of A can be evaluated.
        the pseudoinverse of the transform must have been built.
    out_transform: Transform
        transform from the spectral basis on which A is calculated to the
        real grid on which the divergence is to be evaluated.
        the transform must have been built, with derivs>=1.
    scales: jnp.ndarray, shape (3,)
        If the real coordinates in the transform object are scaled to be dimensionless,
        this parameter adjusts the dimensions of the partial derivatives
        so they are taken with the normal coordinates, not the coordinates in the transform.

    Returns
    -------
    div_A : ndarray, shape(n,3)
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
        terms[:, c] = out_transform.transform(
            A_coeff[:, c], dr=(c == 0), dt=(c == 1), dz=(c == 2)
        )
    # Rescale derivatives
    terms = (
        terms
        * jnp.stack([1 / out_R, 1 / out_R, jnp.ones_like(out_R)], axis=-1)
        / scales.reshape(1, -1)
    )

    # Calculate curl from the partial derivatives
    div = terms.sum(axis=1)
    return div
