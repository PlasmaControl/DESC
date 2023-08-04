import os
import jax.numpy as jnp
import numpy as np
import desc.io
import time
from desc.compute.utils import cross, dot
from desc.grid import Grid
from desc.examples import get
from desc.grid import LinearGrid

def GammaC(name, s, nfulltransits, stepswithin1FP, bpstep, alpha = 0):
   
    start_time = time.time()
   
    eq = desc.io.load(name + "_solved.h5")
    
    # get iota at this surface to use for initial guess
    iota = eq.compute("iota", grid=Grid(jnp.array([[jnp.sqrt(s), 0, 0]])))["iota"]
    
    # keep steps within one field period consistent by multiplying by NFP
    stepswithin2pi = stepswithin1FP * eq.NFP
    
    coords = jnp.ones((stepswithin2pi * nfulltransits, 3))
    coords = coords.at[:, 0].set(coords[:, 0] * jnp.sqrt(s))
    coords = coords.at[:, 2].set(jnp.linspace(0, nfulltransits * 2 * jnp.pi, stepswithin2pi * nfulltransits))
    guess = coords.copy()
    
    coords = coords.at[:, 1].set(coords[:, 1] * alpha)  # set which field line we want
    
    # for initial guess, alpha = zeta + iota*theta*
    # rearrange for theta* and approx theta* ~ theta
    # theta = (alpha - zeta) / iota
    # as initial guess
    guess = guess.at[:, 1].set((alpha - guess[:, 2]) / iota)
    
    print("starting map coords")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    coords1 = eq.map_coordinates(
        coords=coords,
        inbasis=["rho", "alpha", "zeta"],
        outbasis=["rho", "theta", "zeta"],
        period=[jnp.inf, 2 * jnp.pi, jnp.inf],
        guess=guess,
    )  # (2 * jnp.pi / eq.NFP)],
    # )
    # reason is alpha = zeta + iota * theta*
    # if zeta is modded by 2pi/NFP, then after each field period, it is as if we are trying to
    # find the theta* for the point (alpha, zeta=0), which is DIFFERENT from (alpha,zeta=2pi/NFP)
    
    # print(coords1)
    
    coords1 = coords1.at[:, 2].set(coords[:, 2])
    
    print('mapped coords')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    grid2 = Grid(coords1)
    # print(grid2)
    
    
    wellGamma_c = 0
    bigGamma_c = 0
    
    # compute important quantities in DESC.
    psi = eq.Psi  # might need to be normalized by 2pi
    
    data_names = [
        "|grad(psi)|",
        "grad(psi)",
        "|grad(zeta)|",
        "e^zeta",
        "|B|",
        "|B|_r",
        "e_theta",
        "kappa_g",
        "B^zeta",
        "B^zeta_r",
        "B_R",
        "psi_r",
        "B_phi",
        "B_Z",
        "iota_r",
        "X",
        "Y",
        "Z",
    ]
    data = eq.compute(data_names, grid2)
    
    print("eq.compute done")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    grad_psi_mag = data["|grad(psi)|"]
    grad_psi = data["grad(psi)"]
    grad_zeta_mag = data["|grad(zeta)|"]
    grad_zeta = data["e^zeta"]
    e_theta = jnp.linalg.norm(data["e_theta"], axis=-1)
    kappa_g = data["kappa_g"]
    Bsupz = data["B^zeta"]
    dBsupzdpsi = data["B^zeta_r"] * 2 * jnp.pi / data["psi_r"]
    dBdpsi = data["|B|_r"] * 2 * jnp.pi / data["psi_r"]  # might need 2pi
    
    Br = data["B_R"]
    Bphi = data["B_phi"]
    zeta = coords1[:, 2]
    B = data["|B|"]
    Bxyz = jnp.zeros((len(B), 3))
    Bxyz = Bxyz.at[:, 0].set(Br * jnp.cos(zeta) - Bphi * jnp.sin(zeta))
    Bxyz = Bxyz.at[:, 1].set(Br * jnp.sin(zeta) + Bphi * jnp.cos(zeta))
    Bxyz = Bxyz.at[:, 2].set(data["B_Z"])
    
    dVdb_t1 = data["iota_r"] * dot(cross(grad_psi, Bxyz), grad_zeta) / B
    
    # finding basic arc length of each segment
    x = data["X"]
    y = data["Y"]
    z = data["Z"]
    ds = jnp.sqrt(
        jnp.add(jnp.add(jnp.square(jnp.diff(x)), jnp.square(jnp.diff(y))), jnp.square(jnp.diff(z)))
    )
    
    maxB = jnp.nanmax(B)
    minB = jnp.nanmin(jnp.abs(B))
    
    # integrating dl/b
    dloverb = jnp.sum(ds/B[:-1])
    ds = jnp.append(ds, 0) # need to look into this
    
    deltabp = (maxB - minB) / (minB * bpstep)
    bp = 1+ (deltabp*jnp.linspace(-0.5, bpstep-1.5, bpstep))
    
    B_reflect = minB * bp
    
    
    B_array = jnp.tile(B, bpstep)
    bp_array = jnp.repeat(bp, len(B))
    B_reflect_array = minB * bp_array
    
    # array with values 0 or 1 and length bpstep*len(B)
    # 0 means not in well, 1 is in well 
    # First len(B) values are for bp = lowest, etc
    in_well = jnp.where(B_array < B_reflect_array, jnp.ones(len(B)*bpstep), jnp.zeros(len(B)*bpstep))
    
    
    # flattened grad_psi array with first len(B) values being bp = lowest etc
    # "flattened" means every value in each well is reduced to the value with the index corresponding to the minimum in B along that well.
    # print("starting flattening")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # grad_psi_flat, e_theta_flat = array_flatten_op(jnp.tile(grad_psi_mag, bpstep)*in_well, jnp.tile(e_theta, bpstep)*in_well, B_array)
    # print("flattened")
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    ds_array = jnp.tile(ds, bpstep)
    sqrt_bbb_array = jnp.sqrt(1 - B_array / B_reflect_array)
    
    dIdb = jnp.nansum(((in_well * ds_array / bp_array / sqrt_bbb_array)).reshape([bpstep,len(B)]), axis = 1) / 2 / minB
    
    dgdb = jnp.nansum((in_well * ds_array * jnp.tile(grad_psi_mag,bpstep) * jnp.tile(kappa_g, bpstep) / bp_array / B_array * (sqrt_bbb_array + 1 / sqrt_bbb_array)
    ).reshape([bpstep,len(B)]),axis = 1) / 2
    
    dBdpsi_array = jnp.tile(dBdpsi, bpstep)
    
    dbigGdb = jnp.nansum((
        in_well * dBdpsi_array * ds_array / B_reflect_array / bp_array / B_array * (sqrt_bbb_array + 1 / sqrt_bbb_array)
            ).reshape([bpstep,len(B)]), axis = 1) / 2
    
    dVdb = jnp.nansum((
        (in_well * jnp.tile(dVdb_t1, bpstep) - (2 * dBdpsi_array - B_array / jnp.tile(Bsupz, bpstep) * jnp.tile(dBsupzdpsi, bpstep)))
        * ds_array / B_array / B_reflect_array * sqrt_bbb_array
            ).reshape([bpstep,len(B)]), axis = 1) * 1.5
    
    print("entering for loop")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    start = jnp.where(jnp.diff(in_well) == 1)[0] + 1
    end = jnp.where(jnp.diff(in_well) == -1)[0] + 1
    
    if in_well[-1] == 1:
        start = start[:-1]
    if in_well[0] == 1:
        end = end[1:]
    assert len(start) == len(end)
    
    grad_psi_mag = jnp.tile(grad_psi_mag, bpstep)
    e_theta = jnp.tile(e_theta, bpstep)
    
    B_len = len(B)
    bp_ind = start // B_len
    vrovervt = jnp.zeros_like(start)
    wellGamma_c = jnp.zeros_like(start)
    for i in range(len(start)):
        min_B_ind = jnp.argmin(B_array[start[i]:end[i]])+start[i]
        temp = dgdb[bp_ind[i]] / grad_psi_mag[min_B_ind] / dIdb[bp_ind[i]] / minB / e_theta[min_B_ind]
        vrovervt = temp / (dbigGdb[bp_ind[i]] / dIdb[bp_ind[i]] + 2 / 3 * dVdb[bp_ind[i]] / dIdb[bp_ind[i]])
    
        gamma_c = 2 * jnp.arctan(vrovervt) / jnp.pi
        wellGamma_c = wellGamma_c.at[i].set(gamma_c * gamma_c * dIdb[bp_ind[i]])
    
    bigGamma_c = jnp.sum(wellGamma_c) * jnp.pi / 2 / jnp.sqrt(2) * deltabp / dloverb
    
    print("--- %s seconds ---" % (time.time() - start_time))

    return bigGamma_c