import numpy as np
import _booz_xform as bx
import matplotlib.pyplot as plt

from desc.utils import sign
from desc.grid import LinearGrid
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
from desc.equilibrium import EquilibriaFamily


def transform(Cmn, Smn, xm, xn, theta=np.zeros((1, 0)), zeta=np.zeros((1, 0)), si=-1):
    f = np.zeros_like(theta)
    if np.sum(Cmn.shape):
        f += np.sum(
            np.cos(
                xm[np.newaxis] * theta[:, np.newaxis]
                - xn[np.newaxis] * zeta[:, np.newaxis]
            )
            * Cmn[:, si],
            axis=-1,
        )
    if np.sum(Smn.shape):
        f += np.sum(
            np.sin(
                xm[np.newaxis] * theta[:, np.newaxis]
                - xn[np.newaxis] * zeta[:, np.newaxis]
            )
            * Smn[:, si],
            axis=-1,
        )
    return f


def ptolemy(modes, xm, xn, mns, mnc):
    f = np.zeros((modes.shape[0],))
    for i in range(len(xm)):
        m = xm[i]
        n = xn[i]
        if np.sum(mns.shape):
            if m != 0:  # sin(m*t) * cos(n*p)
                idx = np.where((modes == [0, -np.abs(m), np.abs(n)]).all(axis=1))[0]
                f[idx] += np.sign(m) * mns[i]
            if n != 0:  # cos(m*t) * sin(n*p)
                idx = np.where((modes == [0, np.abs(m), -np.abs(n)]).all(axis=1))[0]
                f[idx] += -np.sign(n) * mns[i]
        if np.sum(mnc.shape):
            # cos(m*t) * cos(n*p)
            idx = np.where((modes == [0, np.abs(m), np.abs(n)]).all(axis=1))[0]
            f[idx] += mnc[i]
            if m != 0 and n != 0:  # sin(m*t) * sin(n*p)
                idx = np.where((modes == [0, -np.abs(m), -np.abs(n)]).all(axis=1))[0]
                f[idx] += np.sign(m) * np.sign(n) * mnc[i]
    return f


# DESC
# fam = EquilibriaFamily.load("examples/DESC/DSHAPE_output.h5")
fam = EquilibriaFamily.load("examples/STELLOPT_QS_RBC-2E-02_ZBS+2E-02.h5")
eq = fam[-1]
grid = LinearGrid(M=2 * eq.M_grid + 1, N=2 * eq.N_grid + 1, NFP=eq.NFP, rho=1.0)
data = eq.compute("|B|_mn", grid)
theta = data["theta"]
zeta = data["zeta"]
theta_B = data["theta_B"]
zeta_B = data["zeta_B"]
B_theta_desc = data["B_theta"]
B_zeta_desc = data["B_zeta"]
lambda_desc = data["lambda"]
nu_desc = data["nu"]

# BOOZ_XFORM
b = bx.Booz_xform()
# b.read_wout("examples/VMEC/wout_DSHAPE.nc")
b.read_wout("examples/wout_STELLOPT_QS_RBC-2E-02_ZBS+2E-02.nc")
b.verbose = 0
b.mboz = 12  # 14
b.nboz = 10  # 0
b.run()
g_booz = transform(b.gmnc_b, b.gmns_b, b.xm_b, b.xn_b, theta, zeta)
B_theta_booz = transform(b.bsubumnc, b.bsubumns, b.xm_nyq, b.xn_nyq, theta, zeta)
B_zeta_booz = transform(b.bsubvmnc, b.bsubvmns, b.xm_nyq, b.xn_nyq, theta, zeta)
lambda_booz = transform(b.lmnc, b.lmns, b.xm, b.xn, theta, zeta)
nu_booz = transform(b.numnc_b, b.numns_b, b.xm_b, b.xn_b, theta, zeta)
lambda_mn = ptolemy(
    data["lambda modes"],
    b.xm,
    b.xn / eq.NFP,
    b.lmns[:, -1],
    b.lmnc,
)
B_theta_mn = ptolemy(
    data["Boozer modes"],
    b.xm_nyq,
    b.xn_nyq / eq.NFP,
    b.bsubumns,
    b.bsubumnc[:, -1],
)
B_zeta_mn = ptolemy(
    data["Boozer modes"],
    b.xm_nyq,
    b.xn_nyq / eq.NFP,
    b.bsubvmns,
    b.bsubvmnc[:, -1],
)
B_mag_mn = ptolemy(
    data["Boozer modes"],
    b.xm_b,
    b.xn_b / eq.NFP,
    b.bmns_b,
    b.bmnc_b[:, -1],
)
# np.savez("B_mn.pny", B_theta_mn=B_theta_mn, B_zeta_mn=B_zeta_mn, B_mag_mn=B_mag_mn)

# B_theta
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(theta, np.abs(B_theta_booz), "b-", label="BOOZ_XFORM")
ax1.plot(theta, np.abs(B_theta_desc), "r:", label="DESC")
ax1.set_ylabel(r"$B_\theta$")
ax1.set_xlabel(r"$\theta$")
ax1.legend()
ax2.plot(zeta, np.abs(B_theta_booz), "b-", label="BOOZ_XFORM")
ax2.plot(zeta, np.abs(B_theta_desc), "r:", label="DESC")
ax2.set_xlabel(r"$\zeta$")
ax2.legend()
plt.savefig("B_theta.png")

# B_zeta
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(theta, np.abs(B_zeta_booz), "b-", label="BOOZ_XFORM")
ax1.plot(theta, np.abs(B_zeta_desc), "r:", label="DESC")
ax1.set_ylabel(r"$B_\zeta$")
ax1.set_xlabel(r"$\theta$")
ax1.legend()
ax2.plot(zeta, np.abs(B_zeta_booz), "b-", label="BOOZ_XFORM")
ax2.plot(zeta, np.abs(B_zeta_desc), "r:", label="DESC")
ax2.set_xlabel(r"$\zeta$")
ax2.legend()
plt.savefig("B_zeta.png")

# lambda
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(theta, lambda_booz, "b-", label="BOOZ_XFORM")
ax1.plot(theta, lambda_desc, "r:", label="DESC")
ax1.set_ylabel(r"$\lambda$")
ax1.set_xlabel(r"$\theta$")
ax1.legend()
ax2.plot(zeta, lambda_booz, "b-", label="BOOZ_XFORM")
ax2.plot(zeta, lambda_desc, "r:", label="DESC")
ax2.set_xlabel(r"$\zeta$")
ax2.legend()
plt.savefig("lambda.png")

# nu
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(theta, nu_booz, "b-", label="BOOZ_XFORM")
ax1.plot(theta_B, nu_desc, "r:", label="DESC")
ax1.set_ylabel(r"$\nu$")
ax1.set_xlabel(r"$\theta_{B}$")
ax1.legend()
ax2.plot(zeta, nu_booz, "b-", label="BOOZ_XFORM")
ax2.plot(zeta_B, nu_desc, "r:", label="DESC")
ax2.set_xlabel(r"$\zeta_{B}$")
ax2.legend()
plt.savefig("nu.png")

# lambda_mn
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.semilogy(data["lambda modes"][:, 1], np.abs(lambda_mn), "bs", label="BOOZ_XFORM")
ax1.semilogy(data["lambda modes"][:, 1], np.abs(data["L_mn"]), "ro", label="DESC")
ax1.set_ylabel(r"$\lambda_{mn}$")
ax1.set_xlabel(r"$m$")
ax1.legend()
ax2.semilogy(data["lambda modes"][:, 2], np.abs(lambda_mn), "bs", label="BOOZ_XFORM")
ax2.semilogy(data["lambda modes"][:, 2], np.abs(data["L_mn"]), "ro", label="DESC")
ax2.set_xlabel(r"$n$")
ax2.legend()
plt.savefig("lambda_mn.png")


# B_theta_mn
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.semilogy(data["Boozer modes"][:, 1], np.abs(B_theta_mn), "bs", label="BOOZ_XFORM")
ax1.semilogy(data["Boozer modes"][:, 1], np.abs(data["B_theta_mn"]), "ro", label="DESC")
ax1.set_ylabel(r"${B_\theta}_{mn}$")
ax1.set_xlabel(r"$m$")
ax1.legend()
ax2.semilogy(data["Boozer modes"][:, 2], np.abs(B_theta_mn), "bs", label="BOOZ_XFORM")
ax2.semilogy(data["Boozer modes"][:, 2], np.abs(data["B_theta_mn"]), "ro", label="DESC")
ax2.set_xlabel(r"$n$")
ax2.legend()
plt.savefig("B_theta_mn.png")

# B_zeta_mn
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.semilogy(data["Boozer modes"][:, 1], np.abs(B_zeta_mn), "bs", label="BOOZ_XFORM")
ax1.semilogy(data["Boozer modes"][:, 1], np.abs(data["B_zeta_mn"]), "ro", label="DESC")
ax1.set_ylabel(r"${B_\zeta}_{mn}$")
ax1.set_xlabel(r"$m$")
ax1.legend()
ax2.semilogy(data["Boozer modes"][:, 2], np.abs(B_zeta_mn), "bs", label="BOOZ_XFORM")
ax2.semilogy(data["Boozer modes"][:, 2], np.abs(data["B_zeta_mn"]), "ro", label="DESC")
ax2.set_xlabel(r"$n$")
ax2.legend()
plt.savefig("B_zeta_mn.png")

# |B|_mn
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.semilogy(data["Boozer modes"][:, 1], np.abs(B_mag_mn), "bs", label="BOOZ_XFORM")
ax1.semilogy(data["Boozer modes"][:, 1], np.abs(data["|B|_mn"]), "ro", label="DESC")
ax1.set_ylabel(r"$|B|_{mn}$")
ax1.set_xlabel(r"$m$")
ax1.legend()
ax2.semilogy(data["Boozer modes"][:, 2], np.abs(B_mag_mn), "bs", label="BOOZ_XFORM")
ax2.semilogy(data["Boozer modes"][:, 2], np.abs(data["|B|_mn"]), "ro", label="DESC")
ax2.set_xlabel(r"$n$")
ax2.legend()
plt.savefig("B_mn.png")

print("B_theta BOOZ: {}".format(sign(b.bsubumnc[0, 0])[0]))
print("B_theta DESC: {}".format(sign(np.mean(data["B_theta"]))[0]))

print("B_zeta BOOZ: {}".format(sign(b.bsubvmnc[0, 0])[0]))
print("B_zeta DESC: {}".format(sign(np.mean(data["B_zeta"]))[0]))

print("iota BOOZ: {}".format(sign(b.iota[-1])[0]))
print("iota DESC: {}".format(sign(data["iota"][0])[0]))

print("sqrt(g) BOOZ: {}".format(sign(np.mean(g_booz))[0]))
print("sqrt(g) DESC: {}".format(sign(np.mean(data["sqrt(g)"]))[0]))

print("I BOOZ = {}".format(b.Boozer_I[-1]))
print("I DESC = {}".format(data["I"][0]))
print("G BOOZ = {}".format(b.Boozer_G[-1]))
print("G DESC = {}".format(data["G"][0]))

print("# BOOZ modes: {}".format(len(b.bmnc_b)))
print("# DESC modes: {}".format(len(data["|B|_mn"])))

print(np.flipud(np.sort(np.abs(B_mag_mn)))[0:10])
print(np.flipud(np.sort(np.abs(data["|B|_mn"])))[0:10])
