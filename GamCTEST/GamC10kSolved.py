# running a job array with SLURM
import os

import matplotlib.pyplot as plt
import numpy as np

import desc.io
from desc.examples import get
from desc.grid import LinearGrid

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
s = ((idx % 50) + 1) / 50

if idx < 50:
    name = "QA1"
elif idx < 100 and idx >= 50:
    name = "QA2"
elif idx < 150 and idx >= 100:
    name = "QA3"
elif idx < 200 and idx >= 150:
    name = "QA4"
elif idx < 250 and idx >= 200:
    name = "QA4n"
elif idx < 300 and idx >= 250:
    name = "QA5"
elif idx < 350 and idx >= 300:
    name = "QA4-1"
elif idx < 400 and idx >= 350:
    name = "QA4-2"
elif idx < 450 and idx >= 400:
    name = "QA4-3"
elif idx < 500 and idx >= 450:
    name = "QA4-4"
elif idx < 550 and idx >= 500:
    name = "QA5-1"
elif idx < 600 and idx >= 550:
    name = "QA5-2"
elif idx < 650 and idx >= 600:
    name = "QA5-3"
elif idx < 700 and idx >= 650:
    name = "QA5-4"

eq = desc.io.load(name + "_solved.h5")

# from desc.vmec import VMECIO
# eq = VMECIO.load("wout_QA4n.nc")
# eq.save("wout_QA4n.h5")
# print('loaded eq')

# eq = get("W7-X")

grid = LinearGrid(rho=1, M=40, N=41, axis=False, NFP=eq.NFP, sym=False)
alpha = eq.compute("alpha", grid=grid)["alpha"] % (2 * np.pi)


from desc.compute.utils import cross, dot
from desc.grid import Grid

# get iota at this surface to use for initial guess
iota = eq.compute("iota", grid=Grid(np.array([[np.sqrt(s), 0, 0]])))["iota"]


stepswithin2pi = 100
nfulltransits = 100

coords = np.ones((stepswithin2pi * nfulltransits, 3))
coords[:, 0] = coords[:, 0] * np.sqrt(s)
coords[:, 2] = np.linspace(0, nfulltransits * 2 * np.pi, stepswithin2pi * nfulltransits)
guess = coords.copy()

alpha = 0
coords[:, 1] = coords[:, 1] * alpha  # set which field line we want

# for initial guess, alpha = zeta + iota*theta*
# rearrange for theta* and approx theta* ~ theta
# theta = (alpha - zeta) / iota
# as initial guess
guess[:, 1] = (alpha - guess[:, 2]) / iota
print(guess)

t1 = time()
print("starting map coords")
coords1 = eq.map_coordinates(
    coords=coords,
    inbasis=["rho", "alpha", "zeta"],
    outbasis=["rho", "theta", "zeta"],
    period=[np.inf, 2 * np.pi, np.inf],
    guess=guess,
)  # (2 * np.pi / eq.NFP)],
# )
print(f"map coords finished, took {time() - t1} seconds")
# reason is alpha = zeta + iota * theta*
# if zeta is modded by 2pi/NFP, then after each field period, it is as if we are trying to
# find the theta* for the point (alpha, zeta=0), which is DIFFERENT from (alpha,zeta=2pi/NFP)

# print(coords1)

coords1 = coords1.at[:, 2].set(coords[:, 2])

# print('mapped coords')

# print(np.any(np.isnan(coords1)))
# print(coords1)
# print(np.where(np.isnan(coords1)))

grid2 = Grid(coords1)
# print(grid2)

B = eq.compute("|B|", grid2)["|B|"]
# print(B)

# plt.plot(coords1[:,2],B)
# plt.ylabel("|B|")
# plt.xlabel("zeta")

maxB = np.nanmax(B)
# print(maxB)
minB = np.nanmin(np.abs(B))
# print(minB)

bpstep = 80  # iterations through b'
nsteps = len(
    B
)  # steps along each field line (equal to number of B values we have, for now)
bp = np.zeros(bpstep)
deltabp = (maxB - minB) / (minB * bpstep)

wellGamma_c = 0
bigGamma_c = 0

# compute important quantities in DESC.

data_names = [
    "|grad(psi)|",
    "grad(psi)",
    "|grad(zeta)|",
    "e^zeta",
    "grad(|B|)",
    "e_theta",
    "kappa_g",
    "psi",
    "B^zeta",
    "B_R",
    "B_phi",
    "B_Z",
    "iota_r",
    "X",
    "Y",
    "Z",
]
data = eq.compute(data_names, grid2)

grad_psi_mag = data["|grad(psi)|"]
grad_psi = data["grad(psi)"]
grad_zeta_mag = data["|grad(zeta)|"]
grad_zeta = data["e^zeta"]
grad_B = data["grad(|B|)"]
e_theta = np.linalg.norm(data["e_theta"], axis=-1)
kappa_g = data["kappa_g"]
psi = data["psi"]
Bsupz = data["B^zeta"]
dBsupzdpsi = grad_B[:, 2] * 2 * np.pi / psi
dBdpsi = grad_B[:, 2] * 2 * np.pi / psi

Br = data["B_R"]
Bphi = data["B_phi"]
zeta = coords1[:, 2]
Bxyz = np.zeros((len(B), 3))
Bxyz[:, 0] = Br * np.cos(zeta) - Bphi * np.sin(zeta)
Bxyz[:, 1] = Br * np.sin(zeta) + Bphi * np.cos(zeta)
Bxyz[:, 2] = data["B_Z"]

dVdb_t1 = data["iota_r"] * dot(cross(grad_psi, Bxyz), grad_zeta) / B

# finding basic arc length of each segment
x = data["X"]
y = data["Y"]
z = data["Z"]
ds = np.sqrt(
    np.add(np.square(np.diff(x)), np.square(np.diff(y)), np.square(np.diff(z)))
)

# integrating dl/b
dloverb = 0
for j in range(0, nsteps - 1):
    dloverb += ds[j] / B[j]

# making the b prime array
for i in range(0, bpstep):
    bp[i] = 1 + ((maxB - minB) * (i - 0.5) / (minB * bpstep))

for i in range(0, bpstep):
    B_reflect = minB * bp[i]
    in_well = 0
    well_start = [0]
    well_end = [0]
    cur_well = 0

    grad_psi_min = 1e10
    grad_psi_i = np.ones(len(B)) * 1e10
    e_theta_min = 0
    e_theta_i = np.zeros(len(B))
    curB_min = B_reflect

    where_above_strength = np.where(B > B_reflect)[0]

    well_start_inds = np.where(np.diff(where_above_strength) > 1)[0]
    well_start = (
        where_above_strength[well_start_inds] + 1 + 1
    )  # second +1 is to match stellopt
    well_end = (
        where_above_strength[well_start_inds + 1] - 2
    )  # the -2 is not supposed to be here IDT, but stellopt has it like this
    # B[well_end] > B_reflect actually, so that B[well_start:well_end] is the whole well
    assert (
        well_start.size == well_end.size
    )  # make sure same number of starts and stops have been found
    # assert np.all(
    #     well_end - well_start >= 0
    # )  # make sure ends are before or equal to starts
    # remove ones ends which are the same as starts
    equal_inds = well_end == well_start
    np.delete(well_end, equal_inds)
    np.delete(well_start, equal_inds)

    total_wells = well_start.size

    # also need to find the grad_psi and e_theta at the local B min of each B well
    for start, end in zip(well_start, well_end):
        if start >= end:
            continue
        cur_well_e_theta = e_theta[start:end]
        cur_well_grad_psi_mag = grad_psi_mag[start:end]
        cur_well_B = B[start:end]
        B_min_index = np.argmin(cur_well_B)
        grad_psi_i[start:end] = cur_well_grad_psi_mag[B_min_index]
        e_theta_i[start:end] = cur_well_e_theta[B_min_index]

    # print(total_wells)
    # print(B_reflect)
    # print(well_start)
    # print(well_end)
    # print(grad_psi_i)

    vrovervt = 0

    # loop to compute important quantities at each step of b'
    for k in range(total_wells):

        in_well_inds = np.arange(well_start[k], well_end[k])
        sqrt_bbb = np.sqrt(1 - B[in_well_inds] / B_reflect)
        ds_in_well = ds[in_well_inds]
        bp_in_well = bp[i]
        bp_in_well_sqrd = bp[i] ** 2

        dIdb = np.sum(ds_in_well / bp_in_well_sqrd / sqrt_bbb) / 2 / minB

        dgdb = np.sum(
            ds_in_well
            * grad_psi_mag[in_well_inds]
            * kappa_g[in_well_inds]
            / bp_in_well_sqrd
            / 2
            / B[in_well_inds]
            * (sqrt_bbb + 1 / sqrt_bbb)
        )
        dbigGdb = np.sum(
            dBdpsi[in_well_inds]
            * ds_in_well
            / B_reflect
            / bp_in_well
            / B[in_well_inds]
            / 2
            * (sqrt_bbb + 1 / sqrt_bbb)
        )

        dVdb = np.sum(
            (
                dVdb_t1[in_well_inds]
                - (
                    2 * dBdpsi[in_well_inds]
                    - B[in_well_inds] / Bsupz[in_well_inds] * dBsupzdpsi[in_well_inds]
                )
            )
            * 1.5
            * ds_in_well
            / B[in_well_inds]
            / B_reflect
            * sqrt_bbb
        )

        # if the well start and well end are not the same point, compute vr/vt (radial drift / torodial drift)
        if well_start[k] < well_end[k]:
            temp = (
                dgdb
                / grad_psi_i[well_start[k]]
                / dIdb
                / minB
                / e_theta_i[well_start[k]]
            )
            temp = temp / (dbigGdb / dIdb + 2 / 3 * dVdb / dIdb)
            vrovervt = temp
        else:
            vrovervt = 0

        gamma_c = 2 * np.arctan(vrovervt) / np.pi

        wellGamma_c += gamma_c * gamma_c * dIdb

    bigGamma_c += wellGamma_c * np.pi / 2 / np.sqrt(2) * deltabp

bigGamma_c = bigGamma_c / dloverb

print(bigGamma_c)

file = name + "_10kSolved.txt"
f = open(file, "a")
f.write(f"{s:1.2f}, {bigGamma_c:1.3e}\n")
f.close()
