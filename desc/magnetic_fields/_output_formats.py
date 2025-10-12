"""Converts coil and equilibrium objects to BMW and FIELDINES-style files."""

import numpy as np

from desc.backend import jnp
from desc.batching import batch_map
from desc.equilibrium.coords import map_coordinates
from desc.grid import ConcentricGrid, Grid, LinearGrid, QuadratureGrid
from desc.io import write_bmw_file, write_fieldlines_file
from desc.utils import rpz2xyz, safenorm

from ._plasma_field import PlasmaField


def save_bmw_format(
    path,
    eq,
    Rmin=5,
    Rmax=11,
    Zmin=-5,
    Zmax=5,
    nR=101,
    nZ=101,
    nphi=90,
    save_vector_potential=True,
    chunk_size=50,
    source_grid=None,
    A_source_grid=None,
    method="vector potential",
    A_res=256,
    series=3,
):
    """
    Save the plasma magnetic field in the same format as BMW.

    Parameters
    ----------
    path : str
        The filepath to save the magnetic field. Ends with .nc.
    eq : Equilibrium
        The equilibrium to be saved as a BMW-style file.
    Rmin, Rmax, Zmin, Zmax : float, optional
        Bounds for the R and Z coordinates of the desired evaluation points
    nR, nZ, nphi : int, optional
        Desired number of evaluation points in the radial, vertical, and toroidal
        directions.
    save_vector_potential : bool, optional
        Whether to also calculate the vector potential and save it as well
    chunk_size : int or None
        Size to split computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    source_grid : Grid or None, optional
        Grid used to discretize Equilibrium object. Should be a surface grid for
        virtual casing method (i.e. rho=[1.0]) and a 3D grid otherwise.
    A_source_grid : Grid, int or None or array-like, optional
        Grid used to discretize MagneticField object for calculating A.
        Defaults to the source_grid unless method == 'virtual casing'.
        Should NOT include endpoint at 2pi.
    method: string, optional
        "biot-savart" or "virtual casing" or "vector potential". "biot-savart"
        and "virtual casing" calculates the magnetic field directly from the
        current density, whereas "vector potential" first calculates A, and then
        takes curl(A) to ensure a divergence-free field. 'virtual casing' is the
        fastest method and accurate at high resolution, but doesn't work inside the
        plasma.
        if passing method='virtual casing' and save_vector_potential=False, the
        final dataset will not contain the source current density values (which BMW
        normally contains) since the current density is not directly computed in the
        virtual casing method.
    series: int, optional
        The BMW series tag that is saved.
    """
    R = np.linspace(Rmin, Rmax, nR)
    Z = np.linspace(Zmin, Zmax, nZ)
    phi = np.linspace(0, 2 * np.pi / eq.NFP, nphi, endpoint=False)

    # Default source grid values
    if source_grid is None:
        if method == "virtual casing":
            source_grid = LinearGrid(
                rho=jnp.array([1.0]),
                M=256,
                N=256,
                NFP=eq.NFP if eq.N > 0 else 64,
                sym=False,
            )
        else:
            source_grid = QuadratureGrid(L=64, M=64, N=64, NFP=eq.NFP)

    # For the direct methods, just use compute_magnetic_field
    [RR, PHI, ZZ] = np.meshgrid(R, phi, Z, indexing="ij")
    coords = np.array([RR.flatten(), PHI.flatten(), ZZ.flatten()]).T

    if method in ["biot-savart", "virtual casing"]:
        B, data = eq.compute_magnetic_field(
            coords,
            source_grid=source_grid,
            chunk_size=chunk_size,
            method=method,
            return_data=True,
        )
    else:
        # To calculate field from vector potential, create a PlasmaField object
        R_bounds = [R.min() - 0.1, R.max() + 0.1]
        Z_bounds = [Z.min() - 0.1, Z.max() + 0.1]
        field = PlasmaField(
            eq,
            source_grid,
            R_bounds,
            Z_bounds,
            A_res,
            chunk_size,
            return_data=True,
        )
        B = field.compute_magnetic_grid(R, phi, Z, eq.NFP).reshape(-1, 3)
        data = field._data

    # Compute the vector potential at the same grid, if necessary
    if save_vector_potential:
        if method != "virtual casing" and A_source_grid is None:
            A_source_grid = source_grid
        A, data = eq.compute_magnetic_vector_potential(
            coords,
            chunk_size=chunk_size,
            source_grid=A_source_grid,
            return_data=True,
        )
    else:
        A = None

    # Pass data through to write_bmw_file for formatting
    write_bmw_file(
        path,
        B=B,
        Rmin=Rmin,
        Rmax=Rmax,
        Zmin=Zmin,
        Zmax=Zmax,
        source_data=data,
        source_grid=source_grid,
        nR=nR,
        nZ=nZ,
        nphi=nphi,
        NFP=eq.NFP,
        A=A,
        series=series,
    )


def save_fieldlines_format(
    path,
    eq=None,
    coils=None,
    Rmin=5,
    Rmax=11,
    Zmin=-5,
    Zmax=5,
    nR=101,
    nZ=101,
    nphi=90,
    save_pressure=True,
    chunk_size=50,
    coil_grid=None,
    source_grid=None,
    method="vector potential",
    NFP=None,
    replace_in_plasma=True,
    A_res=256,
):
    """
    Save the total magnetic field in the FIELDLINES format.

    Parameters
    ----------
    path : str
        The filepath to save the magnetic field. Ends with .h5.
    eq : Equilibrium, optional
        The equilibrium which contributes to the magnetic field and from which
        the pressure and magnetic field inside the plasma is calculated.
    coils : _MagneticField, optional
        The coils, which will create the magnetic field in addition to
        the Equilibrium.
    Rmin, Rmax, Zmin, Zmax : float, optional
        Bounds for the R and Z coordinates of the desired evaluation points
    nR, nZ, nphi : int, optional
        Desired number of evaluation points in the radial, vertical, and toroidal
        directions.
    save_pressure : bool, optional
        Whether to also calculate the pressure and save it as well. Ignored if
        eq=None.
    chunk_size : int or None, optional
        Size to split computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.
    coil_grid : Grid, int or None or array-like, optional
        Grid used to discretize _MagneticField object. Should NOT include
        endpoint at 2pi.
    source_grid : Grid or None, optional
        Grid used to discretize Equilibrium object. Should be a surface grid for
        virtual casing method (i.e. rho=[1.0]) and a 3D grid otherwise.
    method: string, optional
        "biot-savart", "virtual casing" or "vector potential". "biot-savart"
        and "virtual casing" calculates the magnetic field directly from the
        current density, whereas "vector potential" first calculates A, and then
        takes curl(A) to ensure a divergence-free field. 'virtual casing' is the
        fastest method and accurate at high resolution, but doesn't work inside the
        plasma.
        if passing method='virtual casing', replace_in_plasma is highly recommended.
    NFP : int, optional
        The NFP input only defines the maximum phi of the evaluation grid, i.e.
        phimax = 2pi/NFP. Defaults to eq.NFP
    replace_in_plasma: bool, optional
        If True, the magnetic field computations as given by compute_magnetic_field
        will be replaced by the equilibrium's internal magnetic field, as given by
        the equilibrium solve and assuming nested flux surfaces. Ignored if eq=None.
    A_res : int, optional
        If using method="vector potential", the resolution of the grid on which A
        is evaluated.
    """
    assert method in ["biot-savart", "virtual casing", "vector potential"]

    # Evenly spaced meshgrid with an endpoint at phi=2pi/NFP
    if (NFP is None) and (eq is None):
        NFP = 1
    elif NFP is None:
        NFP = eq.NFP
    R = np.linspace(Rmin, Rmax, nR)
    Z = np.linspace(Zmin, Zmax, nZ)
    phi = np.linspace(0, 2 * np.pi / NFP, nphi, endpoint=True)
    [RR, PHI, ZZ] = np.meshgrid(R, phi, Z, indexing="ij")
    coords = np.array([RR.flatten(), PHI.flatten(), ZZ.flatten()]).T

    if eq is not None:
        # Direct methods for computing magnetic field
        if method in ["biot-savart", "virtual casing"]:
            B, data = eq.compute_magnetic_field(
                coords,
                source_grid=source_grid,
                chunk_size=chunk_size,
                method=method,
                return_data=True,
            )
        elif method == "vector potential":
            # curl(A) method
            R_bounds = [R.min() - 0.1, R.max() + 0.1]
            Z_bounds = [Z.min() - 0.1, Z.max() + 0.1]
            field = PlasmaField(
                eq,
                source_grid,
                R_bounds,
                Z_bounds,
                A_res,
                chunk_size,
            )
            B = field.compute_magnetic_grid(R, phi, Z, NFP).reshape(-1, 3)
    else:
        B = np.zeros_like(coords)

    # Add magnetic field from coils
    if coils is not None:
        B += coils.compute_magnetic_field(
            coords, source_grid=coil_grid, basis="rpz", chunk_size=chunk_size
        )

    # eq.compute requires flux coordinates, so we need to map coordinates
    if (eq is not None) and (save_pressure or replace_in_plasma):
        # Determine which coordinates are in the plasma
        plasma_mask = eq.in_plasma(coords.reshape(nR, nphi, nZ, 3)).flatten()
        plasma_coords = coords[plasma_mask]

        # Inputs for map_coordinates
        inbasis = ("R", "phi", "Z")
        period = (np.inf, 2 * np.pi / NFP, np.inf)

        if method == "biot-savart":
            # We already have to calculate distance from source grid points
            # So we have a good initial guess for the (rho,theta,zeta) coordinates
            guess = data["src_rtz"]
            guess = jnp.asarray(guess[plasma_mask])

        else:
            # For the other two methods, have to do a nearest neighbor search
            grid = ConcentricGrid(eq.L_grid, eq.M_grid, max(eq.N_grid, eq.M_grid))
            yg = jnp.array(grid.nodes)
            xg = eq.compute("x", grid, basis="xyz")["x"]
            eval_xyz = rpz2xyz(plasma_coords)

            def _distance_body(x):
                distance = safenorm(x - xg, axis=-1)
                return jnp.argmin(distance, axis=-1)

            idx = batch_map(_distance_body, eval_xyz[..., jnp.newaxis, :], chunk_size)
            guess = yg[idx]

        # We know zeta = phi, so we can swap that out
        guess = guess.at[:, 2].set(plasma_coords[:, 1])

        rtz = map_coordinates(
            eq,
            plasma_coords,
            inbasis,
            ("rho", "theta", "zeta"),
            guess=guess,
            period=period,
        )

        if save_pressure:
            # Calculate the pressure for points inside the plasma
            p_plasma = eq.compute("p", grid=Grid(rtz, NFP=eq.NFP))["p"]

            # Set points outside the plasma to have p=0
            pressure = np.zeros_like(plasma_mask, dtype=p_plasma.dtype)
            pressure[plasma_mask] = p_plasma
        else:
            pressure = None
        if replace_in_plasma:
            # Manual chunking using a for loop to prevent memory problems
            B_plasma = np.zeros_like(plasma_coords, dtype=B.dtype)
            for index in np.arange(0, rtz.shape[0], step=chunk_size):
                end_index = np.minimum(index + chunk_size, rtz.shape[0])
                rtz_chunk = Grid(rtz[index:end_index, :], NFP=eq.NFP)
                B_chunk = eq.compute("B", grid=rtz_chunk, basis="rpz")["B"]
                B_plasma[index:end_index, :] = B_chunk
            B = B.at[plasma_mask, :].set(B_plasma)

    write_fieldlines_file(
        path=path,
        B=B,
        Rmin=Rmin,
        Rmax=Rmax,
        Zmin=Zmin,
        Zmax=Zmax,
        phi_min=phi.min(),
        phi_max=phi.max(),
        nR=nR,
        nZ=nZ,
        nphi=nphi,
        pressure=pressure,
    )
