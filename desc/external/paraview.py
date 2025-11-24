"""Utility functions to export DESC objects to Paraview."""

import warnings

import numpy as np

try:
    import pyvista as pv
    from pyvista import CellType
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
    obj, grid_shape=(100, 100), keys=[], rho=1.0, filename="surface"
):
    """Export a constant rho surface as VTU file.

    Parameters
    ----------
    obj : Equilibrium or FourierRZToroidalSurface
        Object to be exported. Different flux surfaces of the Equilibrium
        can be exported by supplying `rho`.
    grid_res : tuple, list
        The resolution used to create the mesh. Must be in the form
        (N_toroidal, N_poloidal). Default is (100, 100).
    keys : list, optional
        The names of the quantities to be computed on the grid points. Defaults to
        empty list.
    rho : float
        The flux surface to be exported if `obj` is an Equilibrium. Defaults to 1.
    filename : str
        Name for the saved file. The file extension will be `.vtu`. Default name will be
        `surface.vtu`
    """
    if not isinstance(obj, (Equilibrium, FourierRZToroidalSurface)):
        raise ValueError(
            "This function only support Equilibrium or FourierRZToroidalSurface "
            f"objects but {type(obj)} is given."
        )

    Nt, Np = grid_shape
    grid = LinearGrid(rho=rho, theta=Np, zeta=Nt, NFP=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unequal number of field")
        data = obj.compute(["x"] + keys, grid=grid, basis="xyz")
    nodes = np.asarray(data.pop("x"))

    # Build connectivity: each quad is made of 4 points
    cells = []
    celltypes = []

    for i in range(Nt):
        for j in range(Np):
            # Current point index
            p0 = i * Np + j
            # Neighbor indices (with wrap-around)
            p1 = i * Np + (j + 1) % Np
            p2 = ((i + 1) % Nt) * Np + (j + 1) % Np
            p3 = ((i + 1) % Nt) * Np + j

            # Append one quad: format = [4, pt0, pt1, pt2, pt3]
            cells.extend([4, p0, p1, p2, p3])
            celltypes.append(CellType.QUAD)

    # Convert to numpy arrays
    cells = np.array(cells)
    celltypes = np.array(celltypes, dtype=np.uint8)

    # Create the unstructured mesh
    mesh = pv.UnstructuredGrid(cells, celltypes, nodes)

    # Additional data
    for key in keys:
        value = data[key]
        if len(value) == Nt * Np or value.shape == (Nt * Np, 3):
            # this will handle both 1D and 3D
            mesh[key] = value
        else:
            raise ValueError(f"Data {key} has an unimplemented shape ({value.shape})")

    mesh.save(filename + ".vtu")
    print(f"File is saved as {filename}.vtu")


def export_coils_to_paraview(coils, res=100, filename="coil"):
    """Export coils as VTP files.

    Parameters
    ----------
    coils : Coil, CoilSet, MixedCoilSet
        Object to be exported. Can be a single coil or multiple coils given
        as CoilSet or MixedCoilSet.
    res : int
        The resolution to use. Defaults to 100.
    filename : str
        Name for the saved file. The file extension will be `.vtp`. If there
        are multiple coils, an index will be appended to the given name.
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
    grid = LinearGrid(N=res, endpoint=True)

    for i, coil in enumerate(coils_list):
        points = coil.compute("x", grid=grid, basis="xyz")["x"]
        points = np.asarray(points)
        current = getattr(coil, "current", np.nan)

        # Create PolyData object
        poly = pv.PolyData()
        poly.points = points
        # Connectivity of the points
        poly.lines = np.roll(np.arange(len(points) + 1), shift=1)

        # Add current as a scalar field
        current_array = np.full(len(points), current)
        poly["current"] = current_array

        # Save to VTP
        poly.save(filename + f"_{i}.vtp")

    print(f"Saved {len(coils_list)} coils with name format `{filename}_i.vtp`")
