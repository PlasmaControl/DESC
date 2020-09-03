import re
import numpy as np
import os
import subprocess
import pickle
import time as pytime
import scipy.integrate

colorblind_colors = [(0.0000, 0.4500, 0.7000), # blue
                     (0.8359, 0.3682, 0.0000), # vermillion
                     (0.0000, 0.6000, 0.5000), # bluish green
                     (0.9500, 0.9000, 0.2500), # yellow
                     (0.3500, 0.7000, 0.9000), # sky blue
                     (0.8000, 0.6000, 0.7000), # reddish purple
                     (0.9000, 0.6000, 0.0000), # orange
                     (0.5000, 0.5000, 0.5000)] # grey

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
matplotlib.rcdefaults()
rcParams['font.family'] = 'DejaVu Serif'
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 10
rcParams['figure.facecolor'] = (1,1,1,1)
rcParams['figure.figsize'] = (8,6)
rcParams['figure.dpi'] = 141
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.labelsize'] =  'small'
rcParams['axes.titlesize'] = 'medium'
rcParams['lines.linewidth'] = 1.5
rcParams['lines.solid_capstyle'] = 'round'
rcParams['lines.dash_capstyle'] = 'round'
rcParams['lines.dash_joinstyle'] = 'round'
rcParams['xtick.labelsize'] = 'x-small'
rcParams['ytick.labelsize'] = 'x-small'
rcParams['legend.fontsize'] = 'small'
color_cycle = cycler(color=colorblind_colors)
rcParams['axes.prop_cycle'] =  color_cycle

labelsize=10
ticksize=8

def plot_gfile(g,bf=None, figsize=(8,6)):

    if isinstance(g,str):
        g = read_gfile(g)
    R = np.linspace(g['rleft'],g['rleft']+g['rdim'],g['nw'])
    Z = np.linspace(-g['zdim']/2,g['zdim']/2,g['nh'])
    psio = g['boundary_flux'] - g['axis_flux']
    psi_grid = np.linspace(0,1,g['nw'])
    psirz_arr = g['boundary_flux'] - g['psirz']
    if psio<0:
        psio = -psio
        psirz_arr = -psirz_arr


    fig = plt.figure(figsize=figsize)
    fig.suptitle("Shot #{} at t={} ms".format(g['shot'],g['time']))
    gs = matplotlib.gridspec.GridSpec(5, 4, width_ratios=[1,1,1,1]) 
    ax0 = plt.subplot(gs[:,:2])
    ax1 = plt.subplot(gs[0,2:])
    ax2 = plt.subplot(gs[1,2:])
    ax3 = plt.subplot(gs[2,2:])
    ax4 = plt.subplot(gs[3,2:])
    ax5 = plt.subplot(gs[4,2:])
    axes = np.array([ax0,ax1,ax2,ax3,ax4,ax5])

    im = ax0.contourf(R,Z,psirz_arr,levels=20)
    if g.get('rbbbs') is not None and g.get('zbbbs') is not None:
        ax0.plot(g['rbbbs'],g['zbbbs'],c=colorblind_colors[1])
    if g.get('rlimitr') is not None and g.get('zlimitr') is not None:
        ax0.plot(g['rlimitr'],g['zlimitr'],c='w')
    ax0.scatter(g['rmaxis'],g['zmaxis'],c=[colorblind_colors[5]])
    ax0.set_xlabel('R')
    ax0.set_ylabel('Z')
    fig.colorbar(im, ax=ax0, orientation='vertical',fraction=.1)

    ax0.set_xlim((R.min(),R.max()))
    ax0.set_ylim((Z.min(),Z.max()))
    ax0.axis('equal')
    ax0.set_title('$\psi(R,Z)$')

    ax1.plot(psi_grid,g['pres'],lw=1)
    ax1.set_ylabel('$P$')
    ax1.xaxis.set_ticklabels([])
    ax1.grid(True, which='both', lw=.3)

    ax2.plot(psi_grid,g['pprime'],lw=1)
    ax2.set_ylabel("$P'$")
    ax2.xaxis.set_ticklabels([])
    ax2.grid(True, which='both', lw=.3)

    ax3.plot(psi_grid,g['fpol'],lw=1)
    ax3.set_ylabel('$F$')
    ax3.xaxis.set_ticklabels([])
    ax3.grid(True, which='both', lw=.3)

    ax4.plot(psi_grid,g['ffprime'],lw=1)
    ax4.set_ylabel("$FF'$")
    ax4.xaxis.set_ticklabels([])
    ax4.grid(True, which='both', lw=.3)

    ax5.plot(psi_grid,g['qpsi'],lw=1)
    ax5.set_yticks([0,5,10], minor=False)
    ax5.set_yticks(np.arange(10), minor=True)
    ax5.set_ylim(0,None)
    ax5.set_ylabel('$q$')
    ax5.set_xlabel('$\psi$')
    ax5.grid(True, which='both', lw=.3)

    plt.subplots_adjust(wspace=2)

    if bf is not None:
        Br = bf.Br(R[::5],Z[::5],grid=True).T
        Bz = bf.Bz(R[::5],Z[::5],grid=True).T
        ax0.quiver(R[::5],Z[::5],Br,Bz)
    return fig, axes

def write_gfile(filename, date, shot, time, efit, nw, nh, rdim, zdim, rcentr, rleft, zmid,
                rmaxis, zmaxis, axis_flux, boundary_flux, bcentr, current,
                fpol, pres, ffprime, pprime, psirz, qpsi, rbbbs=None, zbbbs=None, rlimitr=None, zlimitr=None, **kwargs):

    s = '  EFITD    {}    #{:06d}{:>6d}ms  {:<8} 0{:>4d}{:>4d}\n'.format(pytime.strftime("%m/%d/%Y",date),
        int(shot), int(time), efit if efit else "", int(nw), int(nh))
    s += '{: 16.8e}{: 16.8e}{: 16.8e}{: 16.8e}{: 16.8e} \n'.format(
        rdim, zdim, rcentr, rleft, zmid)
    s += '{: 16.8e}{: 16.8e}{: 16.8e}{: 16.8e}{: 16.8e} \n'.format(
        rmaxis, zmaxis, axis_flux, boundary_flux, bcentr)
    s += '{: 16.8e}{: 16.8e}{: 16.8e}{: 16.8e}{: 16.8e} \n'.format(
        current, axis_flux, 0, rmaxis, 0)
    s += '{: 16.8e}{: 16.8e}{: 16.8e}{: 16.8e}{: 16.8e} \n'.format(
        zmaxis, 0, boundary_flux, 0, 0)
    for i, elem in enumerate(fpol.flatten()):
        s += '{: 16.8e}'.format(elem)
        if (i+1) % 5 == 0 or (i+1) == fpol.size:
            s += ' \n'
    for i, elem in enumerate(pres.flatten()):
        s += '{: 16.8e}'.format(elem)
        if (i+1) % 5 == 0 or (i+1) == pres.size:
            s += ' \n'
    for i, elem in enumerate(ffprime.flatten()):
        s += '{: 16.8e}'.format(elem)
        if (i+1) % 5 == 0 or (i+1) == ffprime.size:
            s += ' \n'
    for i, elem in enumerate(pprime.flatten()):
        s += '{: 16.8e}'.format(elem)
        if (i+1) % 5 == 0 or (i+1) == pprime.size:
            s += ' \n'
    for i, elem in enumerate(psirz.T.flatten()):
        s += '{: 16.8e}'.format(elem)
        if (i+1) % 5 == 0 or (i+1) == psirz.size:
            s += ' \n'
    for i, elem in enumerate(qpsi.flatten()):
        s += '{: 16.8e}'.format(elem)
        if (i+1) % 5 == 0 or (i+1) == qpsi.size:
            s += ' \n'
    if rbbbs is not None and zbbbs is not None:
        nbbbs = len(rbbbs)
        if nbbbs != len(zbbbs):
            raise ValueError('unequal number of boundary pts for R,Z')
    else:
        nbbbs = 0
        rbbbs = np.array([])
        zbbbs = np.array([])
    if rlimitr is not None and zlimitr is not None:
        nlimitr = len(rlimitr)
        if nlimitr != len(zlimitr):
            raise ValueError('unequal number of limiter points for R,Z')
    else:
        nlimitr = 0
        rlimitr = np.array([])
        zlimitr = np.array([])
    s += ' {:>3d}  {:>3d}'.format(nbbbs,nlimitr)
    rzbbbs = np.stack([rbbbs.flatten(),zbbbs.flatten()]).T.flatten()
    rzlimitr = np.stack([rlimitr.flatten(),zlimitr.flatten()]).T.flatten
    
    for i, elem in enumerate(rzbbbs):
        s += '{: 16.8e}'.format(elem)
        if (i+1) % 5 == 0 or (i+1) == rzbbbs.size:
            s += ' \n'
    for i, elem in enumerate(rzlimitr):
        s += '{: 16.8e}'.format(elem)
        if (i+1) % 5 == 0 or (i+1) == rzlimitr.size:
            s += ' \n'
    with open(filename, 'w+') as f:
        f.write(s)
        f.close()


def read_gfile(filename, **kwargs):

    lines = open(filename, 'r').readlines()
    g = {}

    has_spaces = all([elem == 5 for elem in [len(lines[i].split())
                                             for i in range(1, 5)]])
    if has_spaces:
        def splitline(s):
            return s.split()

        def splitarr(l):
            return ' '.join(l).split()
    else:
        def splitline(s):
            ss = s.split('\n')[0]
            idx = np.arange(0, len(ss)+1, 16)
            return [ss[i:j] for i, j in zip(idx[:-1], idx[1:])]

        def splitarr(l):
            return splitline(''.join([line.split('\n')[0] for line in l]))

    line0 = lines[0].split()
    g['date'] = pytime.strptime(line0[1],"%m/%d/%Y")
    g['shot'] = int(''.join([c for c in line0[2] if c.isdigit()]))
    g['time'] = int(''.join([c for c in line0[3] if c.isdigit()]))
    g['nw'] = int(''.join([c for c in line0[-2] if c.isdigit()]))
    g['nh'] = int(''.join([c for c in line0[-1] if c.isdigit()]))
    if len(line0) == 8:
        g['efit'] = line0[4]
    else:
        g['efit'] = None
    
    [g['rdim'], g['zdim'], g['rcentr'], g['rleft'], g['zmid']] = [
        float(foo) for foo in splitline(lines[1])]
    [g['rmaxis'], g['zmaxis'], g['axis_flux'], g['boundary_flux'],
     g['bcentr']] = [float(foo) for foo in splitline(lines[2])]
    [g['current'], g['axis_flux'], _, g['rmaxis'], _] = [
        float(foo) for foo in splitline(lines[3])]
    [g['zmaxis'], _, g['boundary_flux'], _, _] = [
        float(foo) for foo in splitline(lines[4])]
    lines_per_profile = int(np.ceil(g['nw']/5))
    lines_psirz = int(np.ceil(g['nw']*g['nh']/5))
    lines = lines[5:]
    g['fpol'] = np.array([float(foo)
                          for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]
    g['pres'] = np.array([float(foo)
                          for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]
    g['ffprime'] = np.array([float(foo)
                             for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]
    g['pprime'] = np.array([float(foo)
                            for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]
    g['psirz'] = np.array([float(foo) for foo in splitarr(
        lines[:lines_psirz])]).reshape((g['nh'], g['nw']))
    lines = lines[lines_psirz:]
    g['qpsi'] = np.array([float(foo)
                          for foo in splitarr(lines[:lines_per_profile])])
    lines = lines[lines_per_profile:]

    if len(lines)>0:
        [g['nbbbs'], g['limitr']] = [int(foo) for foo in lines[0].split()]
        lines = lines[1:]
        lines_nbbbs = int(np.ceil(2*g['nbbbs']/5))
        lines_limitr = int(np.ceil(2*g['limitr']/5))
    if len(lines)>=lines_nbbs:
        rzbbbs = np.array([float(foo) for foo in splitarr(
            lines[:lines_nbbbs])]).reshape((g['nbbbs'], 2))
        g['rbbbs'], g['zbbbs'] = rzbbbs[:, 0], rzbbbs[:, 1]
        lines = lines[lines_nbbbs:]
    if len(lines)>=lines_limitr:
        rzlimitr = np.array([float(foo) for foo in splitarr(
            lines[:lines_limitr])]).reshape((g['limitr'], 2))
        g['rlimitr'], g['zlimitr'] = rzlimitr[:, 0], rzlimitr[:, 1]
    lines = lines[lines_limitr:]
    
    # fix zero pressure
    if np.max(np.abs(g['pres'])) == 0:
        psio =  g['boundary_flux'] - g['axis_flux']
        psi = np.linspace(0,1,g['nw'])
        pp = g['pprime']
        g['pres'] = scipy.integrate.cumtrapz(pp[::-1],psi[::-1],initial=0)[::-1]/psio


    return g
