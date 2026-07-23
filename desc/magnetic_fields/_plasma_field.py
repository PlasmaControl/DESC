"""
_MagneticField subclass to quickly compute the plasma current magnetic field.

Uses the eq.compute_vector_potential method to precompute the vector potential,
and then quickly computes the magnetic field from the plasma using spectral
differentiation.
"""

from desc.backend import jnp
from desc.basis import DoubleChebyshevFourierBasis
from desc.compute.nabla import _curl_cylindrical
from desc.grid import CylindricalGrid, Grid, QuadratureGrid
from desc.magnetic_fields import _MagneticField
from desc.transform import Transform
from desc.utils import rpz2xyz_vec, xyz2rpz


class PlasmaField(_MagneticField):
    """_MagneticField subclass to compute the plasma current magnetic field.

    Uses the eq.compute_vector_potential method to precompute the vector potential,
    and then quickly computes the magnetic field from the plasma using spectral
    differentiation.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium object which creates the magnetic field
    source_grid : Grid, int or None or array-like, optional
        Grid used to discretize Biot-Savart vector potential integral.
    R_bounds : array-like shape (2,)
        Minimum and maximum R at which the magnetic field will need to
        be calculated. Used to normalize R to between [0,1].
    Z_bounds : array-like shape (2,)
        Minimum and maximum Z at which the magnetic field will need to
        be calculated. Used to normalize Z to between [0,1].
    A_res : int
        The resolution in R and Z of the collocation grid on which
        A will be evaluated. The resolution in phi is fixed by
        source_grid.N, which defaults to 64.
    chunk_size : int or None
        Size to split computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``. Default is ``None``.
    """

    _static_attrs = _MagneticField._static_attrs
    _io_attrs_ = ["_A_coeff", "_RA_phi_coeff", "_scales", "_shifts", "_basis", "_NFP"]

    def __init__(
        self,
        eq,
        source_grid=None,
        R_bounds=[5, 11],
        Z_bounds=[-5, 5],
        A_res=256,
        chunk_size=50,
    ):
        if source_grid is None:
            source_grid = QuadratureGrid(64, 64, 64, eq.NFP)

        # Build a DoubleChebyshevFourierBasis and a corresponding grid
        A_grid = CylindricalGrid(
            L=A_res,
            M=source_grid.N,
            N=A_res,
            NFP=eq.NFP,
        )
        basis = DoubleChebyshevFourierBasis(A_grid.L, A_grid.M, A_grid.N, NFP=eq.NFP)
        in_transform = Transform(
            A_grid, basis, build_pinv=True, build=False, method="rpz"
        )

        # We fixed the A grid resolution in phi to be the same as the source grid,
        # so we can rotate the A grid to be as far away fom the source grid as possible,
        # to avoid the singularities associated with discretizing the integral
        # near the first-order pole
        phi = source_grid.nodes[source_grid.unique_zeta_idx, 2]

        # We need to shift and scale the grid coordinates to R, phi, Z coordinates
        shifts = jnp.array([R_bounds[0], (phi[1] - phi[0]) / 2, Z_bounds[0]])
        scales = jnp.array([R_bounds[1] - R_bounds[0], 1, Z_bounds[1] - Z_bounds[0]])

        # Coordinates on which to calculate the vector potential
        A_coords = A_grid.nodes * scales + shifts

        # Compute A on the optimal collocation points
        A = eq.compute_magnetic_vector_potential(
            A_coords,
            source_grid=source_grid,
            chunk_size=chunk_size,
        )

        self._A_coeff = in_transform.fit(A)
        self._RA_phi_coeff = in_transform.fit(A_coords[:, 0] * A[:, 1])
        self._scales = scales
        self._shifts = shifts
        self._basis = basis
        self._NFP = eq.NFP
        self._R_bounds = R_bounds
        self._Z_bounds = Z_bounds

    def compute_magnetic_field(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute magnetic field at a set of points.

        If saving a very large number of points on a meshgrid
        (e.g. for saving an mgrid), use compute_magnetic_grid.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate field at in [R,phi,Z] or [X,Y,Z] coordinates.
            R and Z must be in R_bounds and Z_bounds, and R cannot be 0.
        params : dict or array-like of dict, optional
            Not used for this subclass.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic field.
        source_grid : Grid, int or None or array-like, optional
            Not used for this subclass.
        transforms : dict of Transform
            Not used for this subclass.
        chunk_size : int or None
            Not used for this subclass.

        Returns
        -------
        field : ndarray, shape(n,3)
            Magnetic field at specified points

        """
        shifts = self._shifts
        scales = self._scales
        basis_obj = self._basis

        # Convert to RPZ if necessary
        coords = jnp.atleast_2d(coords)
        coords = xyz2rpz(coords) if basis.lower() == "xyz" else coords

        # Normalize the coordinates
        out_grid = Grid((coords - shifts) / scales, jitable=True)
        out_transform = Transform(
            out_grid, basis_obj, build_pinv=False, build=True, derivs=1
        )

        # Calculate B as the curl of A
        B = _curl_cylindrical(
            self._A_coeff, self._RA_phi_coeff, coords[:, 0], out_transform, scales
        )

        # Convert back to XYZ if necessary
        if basis.lower == "xyz":
            B = rpz2xyz_vec(B, phi=coords[:, 1])

        return B

    def compute_magnetic_grid(self, R, phi, Z, NFP=None):
        """
        Compute magnetic field on a full 3D grid.

        Works more quickly than compute_magnetic_field and uses less memory,
        since it transforms from spectral coordinates using partial sums,
        instead of the direct1 transform method.

        Parameters
        ----------
        R : array-like shape(l,)
            Unique R coordinates. Must be monotonically increasing,
            within self.R_bounds, and never 0.
        phi : array-like shape(m,)
            Unique phi coordinates. Must be within [0,2pi/NFP] and
            monotonically increasing.
        Z : array-like shape(n,)
            Unique Z coordinates. Must be within self.Z_bounds, excluding the
            endpoints, and monotonically increasing.
        NFP : int, optional
            Number of field periods in phi. Defaults to the number of
            field periods in the equilibrium used to generate this object.

        Returns
        -------
        field : ndarray, shape(l,m,n,3)
            Magnetic field at specified points
        """
        if NFP is None:
            NFP = self._NFP
        shifts = self._shifts
        scales = self._scales
        basis_obj = self._basis

        # Normalize to be within the R and Z bounds
        RPZ = [R, phi, Z]
        RPZ = [jnp.atleast_1d(arr) for arr in RPZ]
        R, phi, Z = tuple([(RPZ[i] - shifts[i]) / scales[i] for i in jnp.arange(3)])

        # Create a grid with the normalized coordinates
        B_grid = Grid.create_meshgrid((R, phi, Z), jitable=True, coordinates="rpz")
        B_coords = B_grid.nodes * scales + shifts

        # Create a transform object to pass to _curl_cylindrical
        out_transform = Transform(
            B_grid,
            basis_obj,
            build_pinv=False,
            build=True,
            derivs=1,
            method="directrpz",
        )

        # Calculate B as the curl of A
        B = _curl_cylindrical(
            self._A_coeff, self._RA_phi_coeff, B_coords[:, 0], out_transform, scales
        )

        # Reshape B to 4D and make indexing B[r,phi,z]
        B = B.reshape(Z.shape[0], R.shape[0], phi.shape[0], -1)
        B = B.transpose(1, 2, 0, 3)

        return B

    def compute_magnetic_vector_potential(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        chunk_size=None,
    ):
        """Compute magnetic vector potential at a set of points.

        Uses spectral methods to interpolate precomputed magnetic vector potential
        onto the given coordinates.

        Parameters
        ----------
        coords : array-like shape(n,3)
            Nodes to evaluate vector potential at in [R,phi,Z] or [X,Y,Z] coordinates.
        params : dict or array-like of dict, optional
            Dict of values for B0.
        basis : {"rpz", "xyz"}
            Basis for input coordinates and returned magnetic vector potential.
        source_grid : Grid, int or None or array-like, optional
            Unused by this MagneticField class, only kept for API compatibility.
        chunk_size : int or None
            Size to split computation into chunks of evaluation points.
            If no chunking should be done or the chunk size is the full input
            then supply ``None``. Default is ``None``.

        Returns
        -------
        A : ndarray, shape(N,3)
            magnetic vector potential at specified points

        """
        shifts = self._shifts
        scales = self._scales
        basis_obj = self._basis

        # Convert to RPZ if necessary
        coords = jnp.atleast_2d(coords)
        coords = xyz2rpz(coords) if basis.lower() == "xyz" else coords

        # Normalize the coordinates
        out_grid = Grid((coords - shifts) / scales)
        out_transform = Transform(
            out_grid, basis_obj, build_pinv=False, build=True, derivs=0
        )

        # Interpolate the A grid
        A = jnp.stack(
            [
                out_transform.transform(self._A_coeff[:, 0]),
                out_transform.transform(self._A_coeff[:, 1]),
                out_transform.transform(self._A_coeff[:, 2]),
            ],
            axis=-1,
        )

        if basis.lower == "xyz":
            A = rpz2xyz_vec(A, phi=coords[:, 1])

        return A

    @property
    def R_bounds(self):
        """The minimum and maximum R at which the field can be evaluated."""
        return self._R_bounds

    @property
    def Z_bounds(self):
        """The minimum and maximum Z at which the field can be evaluated."""
        return self._Z_bounds
