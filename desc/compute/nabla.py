from desc.transform import Transform
import numpy as np
import jax.numpy as jnp
from desc.grid import Grid, _Grid
from desc.basis import DoubleChebyshevFourierBasis
from desc.transform import Transform


def curl_cylindrical(A, in_coords, out_coords=None, L=None, M=8, N=None, NFP=1):
    """
    Take the curl of A in cylindrical coordinates.

    Parameters
    ----------
    A : ndarray, shape(n,3)
        Vector field, in cylindrical (R,phi,Z) form
    in_coords : ndarray, shape(n,3)
        Coordinates for each point of A, corresponding to (R,phi,Z)
    out_coords : ndarray, shape(m,3), optional
        Coordinates at which to evaluate the curl of A, corresponding
        to (R,phi,Z). Defaults to in_coords.
    L: integer
        Radial resolution to use for the spectral decomposition.
        Default is M
    M: integer
        Toroidal resolution to use for the spectral decomposition.
        Default is 8
    N: integer
        Vertical resolution to use for the spectral decomposition.
        Default is M
    NFP: integer
        Number of field periods. Default is 1
    Returns
    -------
    curl_A : ndarray, shape(n,3)
        The curl of the vector field, in cylindrical coordinates.
    """
    # Take the curl of A
    return _curl_cylindrical(A, *_get_del_inputs(in_coords, out_coords, L, M, N, NFP))


def div_cylindrical(A, in_coords, out_coords=None, L=None, M=8, N=None, NFP=1):
    """
    Take the divergence of A in cylindrical coordinates.

    Parameters
    ----------
    A : ndarray, shape(n,3)
        Vector field, in cylindrical (R,phi,Z) form
    in_coords : ndarray, shape(n,3)
        Coordinates for each point of A, corresponding to (R,phi,Z)
    out_coords : ndarray, shape(m,3), optional
        Coordinates at which to evaluate the divergence of A, corresponding
        to (R,phi,Z). Defaults to in_coords.
    L: integer
        Radial resolution to use for the spectral decomposition.
        Default is M
    M: integer
        Toroidal resolution to use for the spectral decomposition.
        Default is 8
    N: integer
        Vertical resolution to use for the spectral decomposition.
        Default is M
    NFP: integer
        Number of field periods. Default is 1
    Returns
    -------
    div_A : ndarray, shape(n,3)
        The divergence of the vector field, in cylindrical coordinates.
    """
    # Take the curl of A
    return _div_cylindrical(A, *_get_del_inputs(in_coords, out_coords, L, M, N, NFP))


def _get_del_inputs(in_coords, out_coords, L, M, N, NFP):
    """
    Get inputs for _curl_cylindrical and _div_cylindrical.

    Parameters
    ----------
    in_coords : ndarray, shape(n,3)
        Coordinates for each point of A, corresponding to (R,phi,Z)
    out_coords : ndarray, shape(m,3), optional
        Coordinates at which to evaluate the curl of A, corresponding
        to (R,phi,Z). Defaults to in_coords.
    L: integer
        Radial resolution to use for the spectral decomposition.
        Default is M
    M: integer
        Toroidal resolution to use for the spectral decomposition.
        Default is 8
    N: integer
        Vertical resolution to use for the spectral decomposition.
        Default is M
    NFP: integer
        Number of field periods. Default is 1
    Returns
    -------
    in_R, out_R, in_transform, out_transform, scales
        Outputs, in order, for _curl_cylindrical and _div_cylindrical (except for A)
    """
    # Default spectral resolution parameters
    if L is None:
        L = M
    if N is None:
        N = M

    # Normalize R and Z to [alpha/2,1-alpha/2] so they can be used in the Chebyshev basis
    if out_coords is None:
        out_coords = in_coords

    Rs = np.unique(np.hstack([in_coords[:, 0], out_coords[:, 0]]))
    Zs = np.unique(np.hstack([in_coords[:, 2], out_coords[:, 2]]))
    shifts, scales = _normalize_rpz(Rs, Zs)

    # Create the grids for the in and out coordinates
    source_grid = Grid(nodes=(in_coords - shifts) / scales)
    destination_grid = Grid(nodes=(out_coords - shifts) / scales)

    # Create the transform to the 3D spectral basis
    basis_obj = DoubleChebyshevFourierBasis(L, M, N, NFP)
    in_transform = Transform(
        source_grid, basis_obj, derivs=0, build=False, build_pinv=True
    )

    # Create the transform from the 3D spectral basis to the out grid
    out_transform = Transform(
        destination_grid, basis_obj, derivs=1, build=True, build_pinv=False
    )

    # Return the necessary inputs for del functions
    return in_coords[:, 0], out_coords[:, 0], in_transform, out_transform, scales


def _curl_cylindrical(A, in_R, out_R, in_transform, out_transform, scales):
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
    scales: np.ndarray, shape (3,)
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
    terms = np.zeros((out_R.shape[0], 3, 3))
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

    curl_A = np.vstack([curl_A_R, curl_A_phi, curl_A_z]).T
    return curl_A


def _div_cylindrical(A, in_R, out_R, in_transform, out_transform, scales):
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
    scales: np.ndarray, shape (3,)
        If the real coordinates in the transform object are scaled to be dimensionless,
        this parameter adjusts the dimensions of the partial derivatives
        so they are taken with the normal coordinates, not the coordinates in the transform.

    Returns
    -------
    div_A : ndarray, shape(n,3)
        The divergence of the vector field, in cylindrical coordinates.
    """
    A_coeff = in_transform.fit(
        A * np.vstack([in_R, np.ones_like(in_R), np.ones_like(in_R)]).T
    )
    # Calculate matrix of terms for the curl
    # (dims: datapoint, component index, derivative index)
    terms = np.zeros((out_R.shape[0], 3))
    for c in range(3):
        # partial A_c/partial r_d
        terms[:, c] = out_transform.transform(
            A_coeff[:, c], dr=(c == 0), dt=(c == 1), dz=(c == 2)
        )
    # Rescale derivatives
    terms = (
        terms
        * np.vstack([1 / out_R, 1 / out_R, np.ones_like(out_R)]).T
        / scales.reshape(1, -1)
    )

    # Calculate curl from the partial derivatives
    div = terms.sum(axis=1)
    return div


def _normalize_rpz(Rs, Zs, alpha=1e-5):
    """
    Convenience function to calculate the shifts and scales to normalize R and Z.
    ([R,phi,Z].T - shifts)/scales will leave phi untouched while
    rescaling R and Z to between [alpha/2,1-alpha/2].

    Parameters
    ----------
    Rs : array-like
        Array of R coordinates to normalize.
    Zs : array-like
        Array of Z coordinates to normalize.
    alpha : array-like
        R and Z will be rescaled to between [alpha/2,1-alpha/2].
        Prevents numerical instabilities at 0 and 1.

    Returns
    -------
    shifts : np.ndarray
        Values to shift R, phi, and Z into the desired range
        shifts[1]=0 always.
    scales : np.ndarray
        Values to scale R, phi, and Z into the desired range.
        scales[1]=1 always.
    """
    Rs, Zs = np.atleast_1d(Rs), np.atleast_1d(Zs)
    shifts = np.array([Rs.min(), 0, Zs.min()])
    scales = np.array(
        [
            (1 / (1 - alpha)) * ((Rs - shifts[0]).max()),
            1,
            (1 / (1 - alpha)) * ((Zs - shifts[2]).max()),
        ]
    )
    shifts -= np.array([alpha / 2, 0, alpha / 2]) * scales

    # If the minimum R and Z are the same as the max, shift to 0.5
    shifts -= np.where(scales == 0, 0.5, 0)
    scales = np.where(scales == 0, 1, scales)

    return shifts, scales
