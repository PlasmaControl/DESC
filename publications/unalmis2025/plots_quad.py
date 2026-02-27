"""Quadrature benchmarking."""

import pickle

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.special import ellipe, ellipk

from desc.integrals.quad_utils import (  # automorphism_arcsin,; grad_automorphism_arcsin,
    automorphism_sin,
    bijection_from_disc,
    chebgauss1,
    chebgauss2,
    get_quadrature,
    grad_automorphism_sin,
    grad_bijection_from_disc,
    leggauss_lob,
    simpson2,
    tanh_sinh,
    uniform,
)
from desc.utils import safediv

apprx_err = 0
noise_level = 0

n = np.arange(7, 202, 2)
n[-1] = 200
try:
    with open("legs_quad.pkl", "rb") as file:
        leg_gaus, leg_lobs = pickle.load(file)
except FileNotFoundError:
    leg_gaus = {k: leggauss(k) for k in n}
    # I enabled the tridiagonal solver in backend.
    leg_lobs = {k: leggauss_lob(k, interior_only=True) for k in n}
    with open("legs_quad.pkl", "wb") as file:
        pickle.dump((leg_gaus, leg_lobs), file)


def leggauss_sin(m):
    if m in leg_gaus:
        m = leg_gaus[m]
    else:
        m = leggauss(m)
    return get_quadrature(m, (automorphism_sin, grad_automorphism_sin))


def leggauss_lob_sin(m):
    if m in leg_lobs:
        m = leg_lobs[m]
    else:
        m = leggauss_lob(m, interior_only=True)
    return get_quadrature(m, (automorphism_sin, grad_automorphism_sin))


def get_quadratures_to_test(is_F):
    if is_F:
        cheb = chebgauss1
        cheb_name = r"GC$_{1}$"
        legs = leggauss_sin
        legs_name = r"GL$_{1}$ & $\sin$"
    else:
        cheb = chebgauss2
        cheb_name = r"GC$_{2}$"
        legs = leggauss_lob_sin
        legs_name = r"GL$_{2}$ & $\sin$"

    # def cheb_arcsin(n):
    #     # Kosloff and Tal-Ezer almost-equispaced grid where γ = 1−β = cos(0.5).
    #     # Spectrally convergent with almost uniformly spaced nodes.
    #     return get_quadrature(cheb(n), (automorphism_arcsin, grad_automorphism_arcsin))

    # cheb_arcsin_name = cheb_name + r" & $\arcsin$"

    quad_funs = [
        uniform,
        simpson2,
        tanh_sinh,
        cheb,
        # cheb_arcsin,
        legs,
    ]
    names = [
        "Midpoint",
        "Simpson",
        "DE",
        cheb_name,
        # cheb_arcsin_name,
        legs_name,
    ]

    return quad_funs, names


def plot_quadratures(
    truth,
    fun,
    n,
    quad_funs,
    names,
    interval,
    filename,
    include_legend=True,
    include_mach_eps=True,
    simpson_lw=None,
):
    eps_label = "Mach. prec.\n" + r"$5 \times 10^{-16}$"
    # Free to increase eps as we please for plots so long as eps <= 1e^{-precision}.
    eps = np.finfo(np.array(1.0).dtype).eps
    eps = max(eps, 5e-16)
    print("eps =", eps)
    print("precision =", np.finfo(np.array(1.0).dtype).precision)

    fig, ax = plt.subplots(figsize=(10 if include_legend else 6.75, 6))

    for j, quad_fun in enumerate(quad_funs):
        abs_error = np.zeros(n.size)
        for i, n_i in enumerate(n):
            x, w = quad_fun(n_i)
            x = bijection_from_disc(x, *interval)
            result = fun(x).dot(w) * grad_bijection_from_disc(*interval)
            abs_error[i] = np.abs(result - truth)

        linewidth = 6
        markersize = 12
        if names[j] == "Simpson":
            markersize = 0
            if simpson_lw is not None:
                linewidth = simpson_lw
        if names[j] == "Midpoint":
            markersize = 0

        ax.semilogy(
            n,
            abs_error,
            label=names[j],
            marker="o",
            linestyle="-",
            markevery=slice(0, 10, 2),
            markersize=markersize,
            linewidth=linewidth,
        )

    if include_mach_eps:
        ax.axhline(y=eps, color="black", linestyle="--", lw=5, label=eps_label)

    ax.set_xlabel(r"$N_q$", fontsize=28)
    ax.set_ylabel("Abs. error", fontsize=28, labelpad=-3)
    ax.set_xticks([7, 25, 50, 100, 150, 200])
    ax.tick_params(which="both", labelsize=26)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=1)
    ax.xaxis.get_major_ticks()[-1].gridline.set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    if include_legend:
        fig.tight_layout(rect=[0.3, 0, 1, 1])
        fig.legend(loc="center left", frameon=False, fontsize=24)

    fig.savefig(f"{filename}_quad_compare.pdf")
    return fig


def plot_B_and_fun(B, fun, fun_latex="", filename="example"):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel(r"$\zeta$", fontsize=28)
    ax1.set_ylabel(r"$\vert B \vert$", fontsize=28)
    color1, color2 = "tab:blue", "k"
    ax1.tick_params(axis="x", labelsize=26)
    ax1.tick_params(axis="y", labelcolor=color1, labelsize=26)
    for spine in ax1.spines.values():
        spine.set_linewidth(3)

    ax2 = ax1.twinx()
    ax2.set_ylabel(fun_latex, fontsize=28)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=26)

    x = np.linspace(-1, 1, 1000)[1:-1]
    ax1.plot(x, B(x), color=color1, label=r"$\vert B \vert$", linewidth=5)
    ax2.plot(x, fun(x), "--", color=color2, label=r"$f$", linewidth=5)

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.legend(loc="center right", frameon=False, fontsize=26)
    fig.savefig(f"{filename}.pdf")
    return fig


class EllipticIntegral:
    """Elliptic integral quadrature plotter."""

    @staticmethod
    def integrand_F(z, k):
        return np.reciprocal(np.sqrt(k**2 - np.sin(z) ** 2))

    @staticmethod
    def integrand_E(z, k):
        return np.sqrt(k**2 - np.sin(z) ** 2)

    @staticmethod
    def analytic_F(k):
        """Incomplete elliptic F(arcsin k, 1/k) / k."""
        # ellipkm1 only useful when k close to 1 s.t. 1-k is wrong
        return ellipk(k**2)

    @staticmethod
    def analytic_E(k):
        """Incomplete elliptic E(arcsin k, 1/k) k."""
        return ellipe(k**2) + (k**2 - 1) * ellipk(k**2)

    @staticmethod
    def fixed(fun, k, quad_fun, resolution):
        k = np.atleast_1d(k)
        b = np.arcsin(k)
        a = -b
        x, w = quad_fun(resolution)
        z = bijection_from_disc(x, a[..., np.newaxis], b[..., np.newaxis])
        k = k[..., np.newaxis]
        result = fun(z, k).dot(w) * grad_bijection_from_disc(a, b)
        return result / 2

    @staticmethod
    def plot_vs_k(is_F, k, resolution, quad_fun, quad_fun_name, color=None):
        fig, ax = plt.subplots(figsize=(6, 6))

        if is_F:
            fun = EllipticIntegral.integrand_F
            filename = "Incomplete_elliptic_F_k"
        else:
            fun = EllipticIntegral.integrand_E
            filename = "Incomplete_elliptic_kE"

        ax.plot(
            k,
            EllipticIntegral.fixed(fun, k, quad_fun, resolution),
            label=quad_fun_name + rf" $(N_q = {resolution})$",
            color="tab:orange" if color is None else color,
            linewidth=7.5,
        )

        if is_F:
            ax.plot(
                k,
                EllipticIntegral.analytic_F(k),
                label=r"$k^{-1} F(\arcsin k, k^{-1})$",
                color="black",
                linewidth=5,
                linestyle="--",
            )
        else:
            ax.plot(
                k,
                EllipticIntegral.analytic_E(k),
                label=r"$k E(\arcsin k, k^{-1})$",
                color="black",
                linestyle="--",
                linewidth=5,
            )

        ax.set_xlabel(r"$k$", fontsize=30)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(which="both", labelsize=26)
        for spine in ax.spines.values():
            spine.set_linewidth(3)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            [handles[1], handles[0]],
            [labels[1], labels[0]],
            fontsize=28,
            loc="upper left",
            frameon=False,
        )

        fig.savefig(filename + f"_vs_{quad_fun_name[:2]}_plot_vs_k.pdf")
        return fig

    @staticmethod
    def plot_vs_quad(is_F, k, include_legend=True):
        if is_F:
            fun = EllipticIntegral.integrand_F
            filename = f"Incomplete_elliptic_F_k_k={k}"
            truth = 2 * EllipticIntegral.analytic_F(k)
        else:
            fun = EllipticIntegral.integrand_E
            filename = f"Incomplete_elliptic_kE_k={k}"
            truth = 2 * EllipticIntegral.analytic_E(k)

        z2 = np.arcsin(k)
        z1 = -z2

        quad_funs, names = get_quadratures_to_test(is_F)
        plot_quadratures(
            truth=truth,
            fun=lambda z: fun(z, k),
            n=n,
            quad_funs=quad_funs,
            names=names,
            interval=(z1 + apprx_err, z2 - apprx_err),
            filename=filename,
            include_legend=include_legend,
        )


class BumpyWell:
    """Bounce integral on W shaped well."""

    def bump(x, h):
        """Well with bump of height h in [0, 1 - epsilon small] in middle"""
        if noise_level > 0:
            x = x + noise_level * np.random.randn(*np.atleast_1d(x).shape)
        return h * (1 - x**2) ** 2 + x**2 + 1

    def plot(weak, B, fun_latex, filename, h):
        """Compare quadratures in W-shaped wells."""

        def fun(x):
            w = np.sqrt(np.abs(2 - B(x)))
            if weak:
                return safediv(1, w)
            return w

        quad_funs, names = get_quadratures_to_test(weak)
        m = h / (h - 1)
        if weak:
            true_anal = 2 / np.sqrt(1 - h) * ellipk(m)
        else:
            true_anal = (2 / (3 * h)) * ((2 * h - 1) * ellipe(h) + (1 - h) * ellipk(h))

        true1, err1 = integrate.quad(
            fun,
            -1,
            0,
            points=(-1, 0),
            epsabs=1e-14,
            epsrel=1e-13,
        )
        true2, err2 = integrate.quad(
            fun,
            0,
            1,
            points=(0, 1),
            epsabs=1e-14,
            epsrel=1e-13,
        )
        err = err1 + err2
        assert err < 1e-12
        np.testing.assert_allclose(
            true1 + true2, true_anal, err_msg="Analytic result wrong."
        )

        plot_B_and_fun(B, fun, fun_latex, filename=f"{filename}_B")
        plot_quadratures(
            truth=true_anal,
            fun=fun,
            n=n,
            quad_funs=quad_funs,
            names=names,
            interval=(-1 + apprx_err, 1 - apprx_err),
            filename=filename,
            include_mach_eps="0p999" not in filename,
            simpson_lw=3.5 if ("0p999" in filename) else None,
        )

    @staticmethod
    def run_W_well():
        """W shaped well with different quadratures."""
        hs = [0.85, 0.999, 0.85, 0.999]
        examples = [
            (
                False,
                lambda x: BumpyWell.bump(x, hs[0]),
                r"$f = (2 - \vert B \vert)^{1/2}$",
                "W_shaped_0p85",
            ),
            (
                False,
                lambda x: BumpyWell.bump(x, hs[1]),
                r"$f = (2 - \vert B \vert)^{1/2}$",
                "W_shaped_0p999",
            ),
            (
                True,
                lambda x: BumpyWell.bump(x, hs[2]),
                r"$f = (2 - \vert B \vert)^{-1/2}$",
                "W_shaped_0p85_weak",
            ),
            (
                True,
                lambda x: BumpyWell.bump(x, hs[3]),
                r"$f = (2 - \vert B \vert)^{-1/2}$",
                "W_shaped_0p999_weak",
            ),
        ]
        for i, example in enumerate(examples):
            BumpyWell.plot(*example, h=hs[i])


if __name__ == "__main__":
    plt.rcParams["figure.constrained_layout.use"] = True

    k1 = np.array([0.25, 0.999])
    k2 = np.linspace(1e-3, 1, 1000, endpoint=False)
    resolution = n[0]

    EllipticIntegral.plot_vs_k(
        False, k2, resolution, chebgauss2, r"GC$_2$", color="tab:red"
    )
    EllipticIntegral.plot_vs_k(
        True, k2, resolution, leggauss_sin, r"GL$_{1}$ & $\sin$", color="tab:purple"
    )

    for is_F in (True, False):
        EllipticIntegral.plot_vs_quad(is_F, k1[0], include_legend=True)
        EllipticIntegral.plot_vs_quad(is_F, k1[1], include_legend=False)

    BumpyWell.run_W_well()
