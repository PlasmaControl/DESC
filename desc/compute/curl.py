from desc.transform import Transform
import numpy as np
import jax.numpy as jnp
from desc.grid import Grid
from desc.basis import DoubleChebyshevFourierBasis
from desc.transform import Transform

def curl_cylindrical(A,coords,L=None,M=8,N=None,NFP=1):
    """
    Take the curl of A in cylindrical coordinates.

    Parameters
    ----------
    A : ndarray, shape(n,3)
        Vector field, in cylindrical (R,phi,Z) form
    coords : ndarray, shape(n,3)
        Coordinates for each point of A, corresponding to (R,phi,Z)
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
    # Default spectral resolution parameters
    if L is None:
        L = M
    if N is None:
        N = M

    # Normalize R and Z to [0,1] so they can be used in the Chebyshev basis
    Rs = coords[:,0]
    Zs = coords[:,2]
    shifts = np.array([Rs.min(),0,Zs.min()])
    scales = np.array([(Rs-shifts[0]).max(),1,(Zs-shifts[2]).max()])
    destination_grid = Grid(nodes = (coords-shifts)/scales)
    
    # Create the transform to the 3D spectral basis    
    basis_obj = DoubleChebyshevFourierBasis(L,M,N,NFP)
    transform = Transform(destination_grid,
        basis_obj, derivs=1, build=True, build_pinv=True)
    
    # Take the curl of A
    return _curl_cylindrical(A,Rs,transform,scales)

def _curl_cylindrical(A,R,transform,scales=np.array([1,1,1])):
    """
    Take the curl of A in cylindrical coordinates,
    given a Transform to spectral coordinates.

    Parameters
    ----------
    A : ndarray, shape(n,3)
        vector field, in cylindrical (R,phi,Z) form
    R : ndarray, shape(n,)
        radial distance for each point of A
    transform: Transform
        transform from the real basis of A to a spectral basis in which
        partial derivatives of A can be evaluated
    scales: np.ndarray, shape (3,)
        If the real coordinates in the transform object are scaled to be dimensionless,
        this parameter adjusts the dimensions of the partial derivatives
        so they are taken with the normal coordinates, not the coordinates in the transform.

    Returns
    -------
    curl_A : ndarray, shape(n,3)
        The curl of the vector field, in cylindrical coordinates.
    """
    A_coeff = transform.fit(A)
    # Calculate matrix of terms for the curl
    # (dims: datapoint, component index, derivative index)
    terms = np.zeros((R.shape[0],3,3))
    for c in range(3):
        for d in range(3):
            if c ==1 and d==0:
                # partial (R*A_phi)/partial R instead of partial A_phi/partial R 
                RA_phi_coeff = transform.fit(R * A[:,1])
                terms[:,c,d] = transform.transform(RA_phi_coeff, dr = 1)
            elif c!=d:
                # partial A_c/partial r_d
                terms[:,c,d] = transform.transform(A_coeff[:,c],
                            dr = (d==0), dt = (d==1), dz = (d==2))
    # Rescale derivatives
    terms = terms / scales.reshape(1,1,-1)

    # Calculate curl from the partial derivatives
    # (curl(A))_R = 1/R partial A_z/partial A_phi - partial A_phi/partial z
    curl_A_R = 1/R * terms[:,2,1] - terms[:,1,2]
    
    # (curl(A))_phi = partial A_R/partial A_z - partial A_z/partial R
    curl_A_phi = terms[:,0,2] - terms[:,2,0]

    # (curl(A))_z = 1/R(partial(R A_phi)/partial R - partial A_R/partial phi)
    curl_A_z = 1/R * (terms[:,1,0]-terms[:,0,1])

    curl_A = np.vstack([curl_A_R,curl_A_phi,curl_A_z]).T
    return curl_A