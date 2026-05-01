import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from desc.dipole import _Dipole, DipoleSet, export_dipoles, import_dipoles
from desc.grid import LinearGrid
from desc.integrals import compute_B_plasma
from desc.magnetic_fields import _MagneticField
from desc.plotting import plot_dipoles, plot_3d, plot_comparison, plot_coils2, poincare_plot
from desc.io import load


class CombinedField(_MagneticField):

    def __init__(self, dipoles, coils, NFP=1):
        self._dipoles = dipoles
        self._coils = coils
        self.NFP = NFP
        super().__init__()

    def compute_magnetic_field(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        B_dipoles = self._dipoles.compute_magnetic_field(coords, params=params, basis=basis, source_grid=source_grid, transforms=transforms, chunk_size=chunk_size)
        B_coils   = self._coils.compute_magnetic_field(coords, params=params, basis=basis, source_grid=source_grid, transforms=transforms, chunk_size=chunk_size)
        return B_dipoles + B_coils

    def compute_magnetic_vector_potential(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        A_dipoles = self._dipoles.compute_magnetic_vector_potential(coords, params=params, basis=basis, source_grid=source_grid, transforms=transforms, chunk_size=chunk_size)
        A_coils   = self._coils.compute_magnetic_vector_potential(coords, params=params, basis=basis, source_grid=source_grid, transforms=transforms, chunk_size=chunk_size)
        return A_dipoles + A_coils

    def tree_flatten(self):
        return (self._dipoles, self._coils), {"NFP": self.NFP}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1], **aux_data)


def compute_average_normalized_field(
    field, coils, eq, p, vacuum=False, chunk_size=None, B_plasma_chunk_size=None
):
    if B_plasma_chunk_size is None:
        B_plasma_chunk_size = chunk_size

    grid = LinearGrid(M=20, N=20, NFP=eq.NFP, endpoint=True, sym=eq.sym)

    surf_coords = eq.surface.compute(['R', 'phi', 'Z'], grid=grid)
    surf_coords_array = np.column_stack([surf_coords['R'], surf_coords['phi'], surf_coords['Z']])
    n_surf = eq.surface.compute(['n_rho'], grid=grid)['n_rho']

    B_PM    = field.compute_magnetic_field(surf_coords_array)
    B_TF    = coils.compute_magnetic_field(surf_coords_array)
    B_total = B_PM + B_TF

    Bn = np.sum(B_total * n_surf, axis=1)

    normalizing_field_vec = B_total.copy()
    if not vacuum:
        normalizing_field_vec += compute_B_plasma(
            eq, eval_grid=grid, chunk_size=B_plasma_chunk_size
        )

    normalizing_field = np.mean(np.linalg.norm(normalizing_field_vec, axis=1))

    data = eq.surface.compute(['X', 'Y', 'Z'], grid=grid)
    X = np.asarray(data['X']).reshape(41, 21)
    Y = np.asarray(data['Y']).reshape(41, 21)
    Z = np.asarray(data['Z']).reshape(41, 21)

    pgrid = pv.StructuredGrid(X, Y, Z)
    pgrid.point_data["bn"] = Bn / normalizing_field
    surf = pgrid.extract_surface()
    p.add_mesh(surf, scalars="bn", smooth_shading=True, show_edges=False)

    return np.mean(np.abs(Bn)) / normalizing_field


eq = load('input.muse-fixedb_output.h5')[-1]
coilset = load('tf_coils_desc.h5')
one_period = import_dipoles(eq, 'muse_dipoles_desc.csv')

combined_field = CombinedField(one_period, coilset, NFP=eq.NFP)

R0 = np.linspace(33.3, 33.7, 10)
Z0 = np.zeros_like(R0)

fig_poincare, ax_poincare = poincare_plot(combined_field, R0, Z0, ntransit=100, phi=6, NFP=eq.NFP, grid=None, return_data=False)
fig_poincare.show()