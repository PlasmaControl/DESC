# %%
import numpy as np
from matplotlib import pyplot as ply
from desc.io import load
from desc.plotting import plot_3d, plot_section, plot_surfaces
from desc.grid import LinearGrid, Grid
from desc.equilibrium import Equilibrium

# %% Load Data
eq = load("./mirror_test.h5")

# %% Generate mesh for LCFS

zeta = np.linspace(0, 2 * np.pi, 100, endpoint=True)
theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)

zeta_mesh, theta_mesh = np.meshgrid(zeta, theta, indexing="ij")
shape = zeta_mesh.shape
grid = Grid(
    nodes=np.stack(
        [
            np.ones_like(zeta_mesh.reshape(-1)),
            theta_mesh.reshape(-1),
            zeta_mesh.reshape(-1),
        ]
    ).T,
    sort=False,
)
data = eq.compute(["R", "Z", "zeta", "|B|"], grid=grid)

mesh_LCFS = [
    data["R"].reshape(shape),
    data["zeta"].reshape(shape),
    data["Z"].reshape(shape),
    data["|B|"].reshape(shape),
]

np.save("mesh_LCFS", mesh_LCFS)

# %% Generate mesh for end cap 0

rho = np.linspace(0, 1, 10, endpoint=True)
theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)

rho_mesh, theta_mesh = np.meshgrid(rho, theta, indexing="ij")
shape = rho_mesh.shape
grid = Grid(
    nodes=np.stack(
        [
            rho_mesh.reshape(-1),
            theta_mesh.reshape(-1),
            np.zeros_like(rho_mesh.reshape(-1)),
        ]
    ).T,
    sort=False,
)
data = eq.compute(["R", "Z", "zeta", "p"], grid=grid)

mesh_end0 = [
    data["R"].reshape(shape),
    data["zeta"].reshape(shape),
    data["Z"].reshape(shape),
    data["p"].reshape(shape),
]

np.save("mesh_end0", mesh_end0)

# %% Generate mesh for end cap 1

rho = np.linspace(0, 1, 10, endpoint=True)
theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)

rho_mesh, theta_mesh = np.meshgrid(rho, theta, indexing="ij")
shape = rho_mesh.shape
grid = Grid(
    nodes=np.stack(
        [
            rho_mesh.reshape(-1),
            theta_mesh.reshape(-1),
            2 * np.pi * np.ones_like(rho_mesh.reshape(-1)),
        ]
    ).T,
    sort=False,
)
data = eq.compute(["R", "Z", "zeta", "p"], grid=grid)

mesh_end1 = [
    data["R"].reshape(shape),
    data["zeta"].reshape(shape),
    data["Z"].reshape(shape),
    data["p"].reshape(shape),
]

np.save("mesh_end1", mesh_end1)

# %% Get fieldlines on the LCFS
assert isinstance(eq, Equilibrium)

theta = np.linspace(0, 2 * np.pi, 16)
zeta = np.linspace(0, 2 * np.pi, 100, endpoint=True)

theta_mesh, zeta_mesh = np.meshgrid(theta, zeta, indexing="ij")
shape = theta_mesh.shape
coords_sfl = np.stack(
    [
        np.ones_like(theta_mesh.reshape(-1)),
        theta_mesh.reshape(-1),
        zeta_mesh.reshape(-1),
    ],
    axis=-1,
)

coords_geo = eq.compute_theta_coords(coords_sfl)
coords_data = eq.compute(["R", "Z", "zeta"], grid=Grid(nodes=coords_geo, sort=False))


lines = np.array([coords_data["R"], coords_data["zeta"], coords_data["Z"]])
lines = lines.reshape(3, shape[0], shape[1])
lines = np.moveaxis(lines, [0, 1], [1, 0])
np.save("lines", lines)

# %%

surfs: list = []
cons_theta: list = []

rho_surf = np.linspace(0.1, 1, 10, endpoint=True)
theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)
theta_sfl_surf = np.linspace(0, 2 * np.pi, 16)
rho = np.linspace(0, 1, 50, endpoint=True)


for zeta0 in [0, 2 * np.pi]:
    for rho0 in rho_surf:
        data = eq.compute(
            ["R", "Z", "zeta"],
            grid=Grid(
                nodes=np.stack(
                    [rho0 * np.ones_like(theta), theta, zeta0 * np.ones_like(theta)],
                    axis=-1,
                ),
                sort=False,
            ),
        )
        surfs.append([data["R"], data["zeta"], data["Z"]])
    for theta_sfl in theta_sfl_surf:
        nodes_geo = eq.compute_theta_coords(
            np.stack([
                rho,
                theta_sfl * np.ones_like(rho),
                zeta0 * np.ones_like(rho),
                ], axis=-1)
        )
        data = eq.compute(
            ["R", "Z", "zeta"],
            grid = Grid(nodes=nodes_geo, sort=False)
        )
        cons_theta.append([data["R"], data["zeta"], data["Z"]])

np.save("surfs", surfs)
np.save("constant_theta", cons_theta)