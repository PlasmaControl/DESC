"""Quadrature plotting.

In this file, "strong singularity" means weakly singular,
while "weak singularity" means weakly singular derivative.
"""

from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy import integrate

from desc.integrals.quad_utils import (
    automorphism_arcsin,
    automorphism_sin,
    bijection_from_disc,
    chebgauss1,
    chebgauss2,
    get_quadrature,
    grad_automorphism_arcsin,
    grad_automorphism_sin,
    grad_bijection_from_disc,
    leggauss_lob,
    simpson2,
    tanh_sinh,
    uniform,
)

# quadrature resolutions
n = np.arange(7, 201, 2)
leggauss_vals = [leggauss(k) for k in n]
leggauss_lob_vals = [leggauss_lob(k) for k in n]


def leggauss_with_sin(m):
    # don't want to resolve eigenvalue problem each run so store values
    auto_sin = (automorphism_sin, grad_automorphism_sin)
    vals = leggauss_vals[np.nonzero(m == n)[0].item()]
    x, w = get_quadrature(vals, auto_sin)
    return x, w


def leggauss_lob_with_sin(m):
    # don't want to resolve eigenvalue problem each run so store values
    auto_sin = (automorphism_sin, grad_automorphism_sin)
    vals = leggauss_lob_vals[np.nonzero(m == n)[0].item()]
    x, w = get_quadrature(vals, auto_sin)
    return x, w


def get_quadratures_to_test(is_strong):
    """Returns list of quad functions and their names."""
    auto_arcsin = (automorphism_arcsin, grad_automorphism_arcsin)

    if is_strong:
        cheb = chebgauss1
        cheb_name = r"GC$_{1}$"
        legs = leggauss_with_sin
    else:
        cheb = chebgauss2
        cheb_name = r"GC$_{2}$"
        legs = leggauss_lob_with_sin

    # Kosloff and Tal-Ezer almost-equispaced grid where γ = 1−β = cos(0.5).
    # Spectrally convergent with almost uniformly spaced nodes.
    cheb_arcsin = lambda n: get_quadrature(cheb(n), auto_arcsin)
    # cheb_arcsin_name = cheb_name + r" & $\sin^{-1}$"
    cheb_arcsin_name = cheb_name

    quad_funs = [
        uniform,
        simpson2,
        tanh_sinh,
        cheb,
        # cheb_arcsin,
        legs,
    ]
    names = [
        "Uniform",
        "Simpson",
        "DE",
        cheb_name,
        # cheb_arcsin_name,
        r"GL & $\sin$",
    ]

    return quad_funs, names


def plot_quadratures(
    truth,
    fun,
    n,
    quad_funs,
    names,
    interval=(-1, 1),
    filename="",
    include_legend=True,
    **kwargs,
):

    kwargs.setdefault("ylabel1", "Abs. error")
    kwargs.setdefault("ylabel2", "Rel. error")
    kwargs.setdefault("title", kwargs.get("filename", "Quadrature_comparison"))

    fig, ax1 = plt.subplots(figsize=(10 if include_legend else 6.75, 6))
    ax1.tick_params(axis="both", labelsize=26)
    ax1.set_xlabel(r"$N_q$", fontsize=28)
    ax1.set_xticks([0, 50, 100, 150, 200])
    ax1.set_ylabel(kwargs["ylabel1"], fontsize=28, labelpad=-3)

    for j, quad_fun in enumerate(quad_funs):
        abs_error = np.zeros(n.size)
        rel_error = np.zeros(n.size)
        max_index = n.size
        for i, n_i in enumerate(n):
            x, w = quad_fun(n_i)
            x = bijection_from_disc(x, interval[0], interval[1])
            w *= grad_bijection_from_disc(interval[0], interval[1])
            result = fun(x).dot(w)
            abs_error[i] = np.abs(result - truth)
            rel_error[i] = np.abs(1 - result / truth)

        ax1.semilogy(
            n[:max_index:2],
            abs_error[:max_index:2],
            label=names[j],
            marker="o",
            linestyle="-",
            markersize=6,
            linewidth=4,
        )

    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=1)
    if include_legend:
        fig.tight_layout(rect=[0.27, 0, 1, 1])
        fig.legend(loc="center left", frameon=False, fontsize=24)
    else:
        fig.tight_layout()
    fig.savefig(f"{filename}_plot_quad.pdf")
    return fig


def plot_B_and_fun(
    B,
    fun,
    B_latex="",
    fun_latex="",
    **kwargs,
):
    kwargs.setdefault("xlabel", r"$\zeta$")
    kwargs.setdefault("ylabel1", B_latex)
    kwargs.setdefault("ylabel2", fun_latex)
    kwargs.setdefault(
        "title", kwargs.get("filename", "Example_well_and_bounce_integrand")
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel(kwargs["xlabel"], fontsize=28)
    ax1.set_ylabel(r"$|B|$", fontsize=28)
    color1, color2 = "tab:blue", "tab:red"
    ax1.minorticks_on()
    ax1.tick_params(axis="x", labelsize=26)
    ax1.tick_params(axis="y", labelcolor=color1, labelsize=26)

    ax2 = ax1.twinx()
    ax2.set_ylabel(kwargs["ylabel2"], fontsize=28)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=26)

    x = np.linspace(-1, 1, 1000)[1:-1]
    ax1.plot(x, B(x), color=color1, label=r"$\vert B \vert$", linewidth=5)
    ax2.plot(x, fun(x), color=color2, label=r"$f$", linewidth=5)

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.legend(loc="center right", frameon=False, fontsize=26)
    fig.savefig(f"{kwargs['filename']}.pdf")
    return fig


class EllipticBounceIntegral:
    """Elliptic bounce integral quadrature plotter."""

    strong_fun = lambda z, k: 1 / np.sqrt(k**2 - np.sin(z) ** 2)
    weak_fun = lambda z, k: np.sqrt(k**2 - np.sin(z) ** 2)

    @staticmethod
    @partial(np.vectorize, excluded={0})
    def adaptive(fun, k):
        """Compute true value.

        Parameters
        ----------
        fun : callable
        k : ndarray or float

        """
        a = -np.arcsin(k)
        b = -a
        # Scipy's elliptic integrals are broken.
        # https://github.com/scipy/scipy/issues/20525.
        result = integrate.quad(
            fun,
            a,
            b,
            args=(k,),
            points=(a, b),
            # can't go below 1e-10 for above k=0.999
            epsabs=1e-10,
            epsrel=1e-10,
        )[0]
        return result

    def fixed(fun, k, quad_fun, resolution):
        """Integrate with given quadrature.

        Parameters
        ----------
        fun : callable
        k : ndarray or float
        quad_fun : callable
        resolution : int

        """
        k = np.atleast_1d(k)
        a = -np.arcsin(k)
        b = -a
        x, w = quad_fun(resolution)
        Z = bijection_from_disc(x, a[..., np.newaxis], b[..., np.newaxis])
        k = k[..., np.newaxis]
        return fun(Z, k).dot(w) * grad_bijection_from_disc(a, b)

    def plot_vs_k(is_strong, k, resolutions, quad_fun, quad_fun_name="", **kwargs):
        """This plots multiples curves of different resolution for integral value vs pitch.

        Parameters
        ----------
        is_strong : bool
        k : ndarray
        resolutions : ndarray
        quad_fun : callable

        """
        if is_strong:
            fun = EllipticBounceIntegral.strong_fun
            title = "Strong_elliptic_integral"
        else:
            fun = EllipticBounceIntegral.weak_fun
            title = "Weak_elliptic_integral"

        kwargs.setdefault("xlabel", r"$k$")
        kwargs.setdefault("title", title + f"_vs_{quad_fun_name}_quadrature")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(kwargs["xlabel"], fontsize=30)
        ax.set_ylabel("", fontsize=28)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.minorticks_on()
        ax.tick_params(which="both", labelsize=26)

        if is_strong:
            ax.plot(
                k,
                0.5 * EllipticBounceIntegral.adaptive(fun, k),
                label=r"$k^{-1} F(\arcsin k, k^{-1})$",
                color="black",
                linewidth=7,
            )
        else:
            ax.plot(
                k,
                0.5 * EllipticBounceIntegral.adaptive(fun, k),
                label=r"$k E(\arcsin k, k^{-1})$",
                color="black",
                linewidth=7,
            )

        for i, resolution in enumerate(resolutions):
            if quad_fun_name == "GC2":
                label = r"GC$_2$ $(N_q = 7)$"
            elif quad_fun_name == "GL":
                label = r"GL & $\sin$ $(N_q = 7)$"
            ax.plot(
                k,
                0.5 * EllipticBounceIntegral.fixed(fun, k, quad_fun, resolution),
                label=label,
                color="tab:orange",
                linewidth=4.5,
                linestyle="--",
            )

        ax.legend(fontsize=27, loc="upper left", frameon=False)
        plt.tight_layout()
        plt.savefig(f"{title}_plot_vs_k.pdf")
        return fig

    def plot_vs_quad(is_strong, k, plot_integrand=False):
        """This compares all quadratures at the given pitch angle variables k.

        Parameters
        ----------
        is_strong : bool
        k : float
        plot_integrand : bool

        """
        if is_strong:
            fun = EllipticBounceIntegral.strong_fun
            title = f"Strong_elliptic_integral_{k}"
        else:
            fun = EllipticBounceIntegral.weak_fun
            title = f"Weak_elliptic_integral_{k}"

        z1 = -np.arcsin(k)
        z2 = -z1
        if plot_integrand:
            z = np.linspace(z1, z2, 1000)
            plt.plot(z, fun(z, k))

        quad_funs, names = get_quadratures_to_test(is_strong)
        plot_quadratures(
            truth=EllipticBounceIntegral.adaptive(fun, k),
            fun=lambda z: fun(z, k),
            n=n,
            quad_funs=quad_funs,
            names=names,
            interval=(z1, z2),
            title=title,
            filename=title,
        )

    def run_quad_compare(
        is_strong,
        k1=np.array([0.25, 0.999]),
        k2=np.linspace(0, 1, 1000, endpoint=False),
        resolutions=n[:1],
        quad_fun=chebgauss2,
        quad_fun_name="GC2",
    ):
        """Elliptic bounce integrals of first and second kind with different quadratures.

        Parameters
        ----------
        is_strong : bool
            Whether to compute weak or strongly singular integrals.
        k1 : ndarray
            k values to compare different quadratures at
        k2 : ndarray
            k values to compare a particular quadrature against the true integral.
        resolutions : ndarray
            Number of different resolution to include for plots with k2.
        quad_fun : callable
            Quadrature to plot with k2.
        quad_fun_name : str
            Name of quad fun to plot with k2.

        Returns
        -------

        """
        for k in k1:
            EllipticBounceIntegral.plot_vs_quad(is_strong, k)
        EllipticBounceIntegral.plot_vs_k(
            is_strong, k2, resolutions, quad_fun, quad_fun_name
        )


class BumpyWell:
    """Bounce integral on W shaped well."""

    def bump(x, h):
        """Well with bump of height h in [0, 1 - epsilon small] in middle"""
        return h * (1 - x**2) ** 2 + x**2 + 1

    def gaussians(x, x_peak=(-0.5, 0, 0.5), h_peak=(0.5, 0.75, 0.25), sigma=0.125):
        """Make |B| with humps.

        Parameters
        ----------
        x : np.ndarray
            Points to evaluate.
        x_peak : np.ndarray
            Peak centers in [-1, 1], excluding endpoints.
        h_peak : np.ndarray
            Peak heights in [0, 1], excluding endpoints.
        sigma : float
            Standard deviation of Gaussian.

        """
        x, x_peak, h_peak = np.atleast_1d(x, x_peak, h_peak)
        x_peak = np.hstack([-1, x_peak, 1])
        h_peak = np.hstack([1, h_peak, 1])
        basis = np.exp(-np.square((x[:, np.newaxis] - x_peak) / sigma))
        return basis.dot(h_peak).squeeze() + 1

    def plot(
        is_strong,
        B,
        B_latex,
        fun_latex,
        filename,
    ):
        """Compare quadratures in W-shaped wells."""

        def fun(x):
            w = np.sqrt(2 - B(x))
            if is_strong:
                return 1 / w
            return w

        quad_funs, names = get_quadratures_to_test(is_strong)
        truth = (
            integrate.quad(
                fun,
                -1,
                0,
                epsabs=1e-10,
                epsrel=1e-10,
            )[0]
            + integrate.quad(
                fun,
                0,
                1,
                epsabs=1e-10,
                epsrel=1e-10,
            )[0]
        )

        fig1 = plot_B_and_fun(B, fun, B_latex, fun_latex, filename=filename)
        fig2 = plot_quadratures(
            truth=truth,
            fun=fun,
            n=n,
            quad_funs=quad_funs,
            names=names,
            filename=filename,
            title="Integration of " + fun_latex + " where " + B_latex,
        )

    @staticmethod
    def run_W_well():
        """W shaped well with different quadratures."""
        examples = [
            (
                False,
                lambda x: BumpyWell.bump(x, 0.75),
                r"$\vert B \vert (\zeta, h=0.75)$",
                r"$f = (2 - \vert B \vert)^{1/2}$",
                "W_shaped_0p75_weak_singularity",
            ),
            (
                False,
                lambda x: BumpyWell.bump(x, 0.999),
                r"$\vert B \vert (\zeta, h=0.999)$",
                r"$f = (2 - \vert B \vert)^{1/2}$",
                "W_shaped_0p999_weak_singularity",
            ),
            (
                True,
                lambda x: BumpyWell.bump(x, 0.75),
                r"$\vert B \vert (\zeta, h=0.75)$",
                r"$f = (2 - \vert B \vert)^{-1/2}$",
                "W_shaped_0p75_strong_singularity",
            ),
            (
                True,
                lambda x: BumpyWell.bump(x, 0.999),
                r"$\vert B \vert (\zeta, h=0.999)$",
                r"$f = (2 - \vert B \vert)^{-1/2}$",
                "W_shaped_0p999_strong_singularity",
            ),
            # (
            #     # not sure how realistic
            #     False,
            #     BumpyWell.gaussians,
            #     "Bumpy well, Gaussian peak basis functions",
            #     r"$f(\zeta) = \sqrt{2 - \vert B \vert}$",
            #     "Gaussian peak bumpy wells, weak singularity",
            # ),
            # (
            #     True,
            #     BumpyWell.gaussians,
            #     "Bumpy well, Gaussian peak basis functions",
            #     r"$f(\zeta) = 1 / \sqrt{2 - \vert B \vert}$",
            #     "Gaussian peak bumpy wells, strong singularity",
            # ),
        ]
        for example in examples:
            BumpyWell.plot(*example)


if __name__ == "__main__":
    BumpyWell.run_W_well()
    EllipticBounceIntegral.run_quad_compare(False)
    EllipticBounceIntegral.run_quad_compare(
        True,
        quad_fun=leggauss_with_sin,
        quad_fun_name=r"GL",
    )
