"""Utility functions to export DESC objects to Paraview."""

import warnings

import numpy as np

try:
    import pyvista as pv
except ImportError:
    warnings.warn(
        "DESC objects are exported to Paraview using `pyvista`"
        "package which is an optional dependency. Please pip install "
        "`pyvista` to your environment."
    )

from desc.coils import CoilSet, _Coil
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid


def export_surface_to_paraview(
    obj, filename, res=(100, 100), keys=[], rho=1.0, return_mesh=False
):
    """Export a constant rho surface data for Paraview as VTK file.

    Parameters
    ----------
    obj : Equilibrium or FourierRZToroidalSurface
        Object to be exported. Different flux surfaces of the Equilibrium
        can be exported by supplying `rho`.
    filename : str
        Name for the saved file. The file extension will be `.vtk`.
    res : tuple, list
        The resolution used to create the mesh. Must be in the form (Np, Nt),
        number of points in poloidal and toroidal direction, respectively.
        Default is (100, 100).
    keys : list, optional
        The names of the quantities to be computed on the grid points. Defaults to
        empty list.
    rho : float
        The flux surface to be exported if `obj` is an Equilibrium. Defaults to 1.
    return_mesh : bool
        If True, return the created pyvista StructuredGrid object. Defaults to False.

    Returns
    -------
    mesh : pyvista.StructuredGrid
        Created structured grid object. With this object one can compute more
        quantities on `LinearGrid(rho=rho, theta=Np, zeta=Nt, NFP=1, endpoint=True)`
        and add it to the mesh by `mesh['name'] = value`. Once the mesh is changed,
        the user has to save it again `mesh.save(filename)`.
    """
    if not isinstance(obj, (Equilibrium, FourierRZToroidalSurface)):
        raise ValueError(
            "This function only support Equilibrium or FourierRZToroidalSurface "
            f"objects but {type(obj)} is given."
        )

    Np, Nt = res
    grid = LinearGrid(rho=rho, theta=Np, zeta=Nt, NFP=1, endpoint=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unequal number of field")
        data = obj.compute(["x"] + keys, grid=grid, basis="xyz")
    nodes = np.asarray(data.pop("x"))

    # Create the structured mesh
    mesh = pv.StructuredGrid()
    mesh.points = nodes
    mesh.dimensions = (Np, Nt, 1)

    # Additional data
    for key in keys:
        value = data[key]
        if value.size == Nt * Np or value.shape == (Nt * Np, 3):
            # this will handle both 1D and 3D
            mesh[key] = value
        else:
            raise ValueError(f"Data {key} has an unimplemented shape ({value.shape})")

    mesh.save(filename + ".vtk")
    print(f"File is saved as {filename}.vtk")
    if return_mesh:
        return mesh


def export_coils_to_paraview(coils, filename, res=100, keys=[]):
    """Export coils data for Paraview as VTP files.

    Parameters
    ----------
    coils : Coil, CoilSet, MixedCoilSet
        Object to be exported. Can be a single coil or multiple coils given
        as CoilSet or MixedCoilSet.
    filename : str
        Name for the saved file. The file extension will be `.vtp`.
        If there are multiple coils, an index will be appended to the given name.
    res : int
        The resolution to use for all coils. Defaults to 100 points around a
        single coil.
    keys : list of str
        Additional data to store in the file. By default, only the current
        corresponding to each coil is stored.
    """
    if not isinstance(coils, _Coil):
        raise ValueError(
            "This function only support classes inherited from _Coil "
            f"but {type(coils)} is given."
        )

    def flatten_coils(coilset):
        if hasattr(coilset, "__len__"):
            if hasattr(coilset, "_NFP") and hasattr(coilset, "_sym"):
                if coilset.NFP > 1 or coilset.sym:
                    # plot all coils for symmetric coil sets
                    coilset = CoilSet.from_symmetry(
                        coilset,
                        NFP=coilset.NFP,
                        sym=coilset.sym,
                        check_intersection=False,
                    )
            return [a for i in coilset for a in flatten_coils(i)]
        else:
            return [coilset]

    coils_list = flatten_coils(coils)
    grid = LinearGrid(zeta=res, endpoint=True)

    for i, coil in enumerate(coils_list):
        data = coil.compute(["x"] + keys, grid=grid, basis="xyz")
        points = np.asarray(data["x"])
        current = getattr(coil, "current", np.nan)

        # Create PolyData object
        poly = pv.PolyData()
        poly.points = points
        # Connectivity of the points
        poly.lines = np.roll(np.arange(len(points) + 1), shift=1)

        # Add current as a scalar field
        current_array = np.full(len(points), current)
        poly["current"] = current_array

        # Additional data
        for key in keys:
            value = data[key]
            if value.size == res or value.shape == (res, 3):
                # this will handle both 1D and 3D
                poly[key] = value
            else:
                raise ValueError(
                    f"Data {key} has an unimplemented shape ({value.shape})"
                )

        poly.save(f"{filename}_{i}.vtp")

    print(f"Saved {len(coils_list)} coils with name format `{filename}_i.vtp`")


def export_volume_to_paraview(
    eq, filename, res=(20, 100, 100), keys=[], return_mesh=False
):
    """Export equilibrium volume data for Paraview as VTK file.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium to be exported.
    filename : str
        Name for the saved file. The file extension will be `.vtk`.
    res : tuple of ints, (Nr, Np, Nt)
        The resolutions to use. Defaults to (20, 100, 100) points in radial, poloidal
        and toroidal directions, respectively.
    keys : list of str
        Additional data to store in the file. Defaults to empty list.
    return_mesh : bool
        If True, return the created pyvista StructuredGrid object. Defaults to False.

    Returns
    -------
    mesh : pyvista.StructuredGrid
        Created structured grid object. With this object one can compute more
        quantities on `LinearGrid(rho=Nr, theta=Np, zeta=Nt, NFP=1, endpoint=True)`
        and add it to the mesh by `mesh['name'] = value`. Once the mesh
        is changed, the user has to save it again `mesh.save(filename)`.
    """
    if not isinstance(eq, Equilibrium):
        raise ValueError(
            "This function only support classes of type Equilibrium "
            f"but {type(eq)} is given."
        )
    Nr, Np, Nt = res
    grid = LinearGrid(rho=Nr, theta=Np, zeta=Nt, NFP=1, endpoint=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unequal number of field")
        data = eq.compute(["x"] + keys, grid=grid, basis="xyz")
    nodes = np.asarray(data.pop("x"))

    # Create the structured mesh
    mesh = pv.StructuredGrid()
    mesh.points = nodes
    mesh.dimensions = (Np, Nr, Nt)

    # Additional data
    for key in keys:
        value = data[key]
        if value.size == Nr * Nt * Np or value.shape == (Nr * Nt * Np, 3):
            # this will handle both 1D and 3D
            mesh[key] = value
        else:
            raise ValueError(f"Data {key} has an unimplemented shape ({value.shape})")

    mesh.save(filename + ".vtk")
    print(f"File is saved as {filename}.vtk")
    if return_mesh:
        return mesh
