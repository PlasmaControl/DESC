# TODO: Remove this once 117 is merged.
#   https://github.com/f0uriest/interpax/pull/117

from interpax.utils import asarray_inexact, wrap_jit

from desc.backend import jnp


@wrap_jit(static_argnames=["n"])
def fft_interp1d(
    f,
    n: int,
    sx=None,
    dx: float = 1.0,
):
    """Interpolation of a real-valued 1D periodic function via FFT.

    Parameters
    ----------
    f : ndarray, shape(nx, ...)
        Source data. Assumed to cover 1 full period, excluding the endpoint.
    n : int
        Number of desired interpolation points.
    sx : ndarray or None
        Shift in x to evaluate at. If original data is f(x), interpolates to f(x + sx).
    dx : float
        Spacing of source points.

    Returns
    -------
    fi : ndarray, shape(n, ..., len(sx))
        Interpolated (and possibly shifted) data points.

    """
    f = asarray_inexact(f)
    return ifft_interp1d(jnp.fft.rfft(f, axis=0, norm="forward"), f.shape[0], n, sx, dx)


def ifft_interp1d(
    c,
    nx: int,
    n: int,
    sx=None,
    dx: float = 1.0,
):
    """Interpolation of a 1D Hermitian Fourier series via FFT.

    Parameters
    ----------
    c : ndarray, shape(nx // 2 + 1, ...)
        Fourier coefficients ``jnp.fft.rfft(f,axis=0,norm="forward")``.
    nx : bool
        Number of sample points e.g. ``f.shape[0]``.
    n : int
        Number of desired interpolation points.
    sx : ndarray or None
        Shift in x to evaluate at. If original data is f(x), interpolates to f(x + sx).
    dx : float
        Spacing of source points.

    Returns
    -------
    fi : ndarray, shape(n, ..., len(sx))
        Interpolated (and possibly shifted) data points.

    """
    nx_half = c.shape[0]
    if n < nx_half:
        # truncate early to reduce computation when shifting
        c = c[:n]

    if sx is not None:
        sx = asarray_inexact(sx)
        sx = jnp.exp(1j * _rfftfreq(c.shape[0], nx, dx)[:, None] * sx)
        c = (c[None].T * sx).T
        c = jnp.moveaxis(c, 0, -1)

    if n >= nx:
        return jnp.fft.irfft(c, n, axis=0, norm="forward")

    if (n >= nx_half) and (nx % 2 == 0):
        # then we had not truncated, and we need to half the top frequency
        c = c.at[-1].divide(2)
    c = c.at[0].divide(2) * 2

    x = jnp.linspace(0, dx * nx, n, endpoint=False)
    x = jnp.exp(1j * (c.shape[0] // 2) * x).reshape(n, *((1,) * (c.ndim - 1)))

    c = _fft_pad(c, n, 0)
    return (jnp.fft.ifft(c, axis=0, norm="forward") * x).real


@wrap_jit(static_argnames=["n1", "n2"])
def fft_interp2d(
    f,
    n1: int,
    n2: int,
    sx=None,
    sy=None,
    dx: float = 1.0,
    dy: float = 1.0,
):
    """Interpolation of a real-valued 2D periodic function via FFT.

    Parameters
    ----------
    f : ndarray, shape(nx, ny, ...)
        Source data. Assumed to cover 1 full period, excluding the endpoint.
    n1, n2 : int
        Number of desired interpolation points in x and y directions.
    sx, sy : ndarray or None
        Shift in x and y to evaluate at. If original data is f(x,y), interpolates to
        f(x + sx, y + sy). Both must be provided or None.
    dx, dy : float
        Spacing of source points in x and y.

    Returns
    -------
    fi : ndarray, shape(n1, n2, ..., len(sx))
        Interpolated (and possibly shifted) data points.

    """
    nx, ny = f.shape[:2]

    # https://github.com/f0uriest/interpax/pull/117
    if (sx is None or jnp.size(sx) == 1) and (sy is None or jnp.size(sy) == 1):
        if n1 < nx:
            return fft_interp1d(
                fft_interp1d(f, n1, sx, dx).squeeze(-1).swapaxes(0, 1),
                n2,
                sy,
                dy,
            ).swapaxes(0, 1)
        if n2 < ny:
            return fft_interp1d(
                fft_interp1d(f.swapaxes(0, 1), n2, sy, dy).squeeze(-1).swapaxes(0, 1),
                n1,
                sx,
                dx,
            )

    return ifft_interp2d(
        jnp.fft.rfft2(asarray_inexact(f), axes=(0, 1), norm="forward"),
        ny,
        n1,
        n2,
        sx,
        sy,
        dx,
        dy,
    )


def ifft_interp2d(
    c,
    ny: int,
    n1: int,
    n2: int,
    sx=None,
    sy=None,
    dx: float = 1.0,
    dy: float = 1.0,
):
    """Interpolation of 2D Hermitian Fourier series via FFT.

    Parameters
    ----------
    c : ndarray, shape(nx, ny // 2 + 1, ...)
        Fourier coefficients ``jnp.fft.rfft2(f,axes=(0,1),norm="forward")``.
    ny : bool
        Number of sample points in y coordinate, e.g. ``f.shape[1]``.
    n1, n2 : int
        Number of desired interpolation points in x and y directions.
    sx, sy : ndarray or None
        Shift in x and y to evaluate at. If original data is f(x,y), interpolates to
        f(x + sx, y + sy). Both must be provided or None.
    dx, dy : float
        Spacing of source points in x and y.

    Returns
    -------
    fi : ndarray, shape(n1, n2, ..., len(sx))
        Interpolated (and possibly shifted) data points.

    """
    nx = c.shape[0]
    ny_half = c.shape[1]
    if n2 < ny_half:
        # truncate early to reduce computation when shifting
        c = c[:, :n2]

    if (sx is not None) and (sy is not None):
        sx = asarray_inexact(sx)
        sy = asarray_inexact(sy)
        tau = 2 * jnp.pi
        sx = jnp.exp(1j * jnp.fft.fftfreq(nx, dx / tau)[:, None] * sx)
        sy = jnp.exp(1j * _rfftfreq(c.shape[1], ny, dy)[:, None] * sy)
        c = (c[None].T * (sx[None] * sy[:, None])).T
        c = jnp.moveaxis(c, 0, -1)

    c = _fft_pad(jnp.fft.fftshift(c, 0), n1, 0)
    if n2 >= ny:
        return jnp.fft.irfft2(c, (n1, n2), axes=(0, 1), norm="forward")

    if (n2 >= ny_half) and (ny % 2 == 0):
        # then we had not truncated, and we need to half the top frequency
        c = c.at[:, -1].divide(2)
    c = c.at[:, 0].divide(2) * 2

    y = jnp.linspace(0, dy * ny, n2, endpoint=False)
    y = jnp.exp(1j * (c.shape[1] // 2) * y).reshape(1, n2, *((1,) * (c.ndim - 2)))

    c = jnp.fft.ifft(c, axis=0, norm="forward")
    c = _fft_pad(c, n2, 1)
    return (jnp.fft.ifft(c, axis=1, norm="forward") * y).real


def _rfftfreq(n, nx, dx):
    return jnp.arange(n) * (2 * jnp.pi / (nx * dx))


def _fft_pad(c_shift, n_out, axis):
    n_in = c_shift.shape[axis]
    p = n_out - n_in
    p = (p // 2, p - p // 2)
    if n_in % 2 != 0:
        p = p[::-1]
    return jnp.fft.ifftshift(_pad_along_axis(c_shift, p, axis), axis)


def _pad_along_axis(array, pad: tuple = (0, 0), axis: int = 0):
    """Pad with zeros or truncate a given dimension."""
    index = [slice(None)] * array.ndim
    pad_width = [(0, 0)] * array.ndim
    start = stop = None

    if pad[0] < 0:
        start = -pad[0]
        pad = (0, pad[1])
    if pad[1] < 0:
        stop = pad[1]
        pad = (pad[0], 0)

    index[axis] = slice(start, stop)
    pad_width[axis] = pad
    return jnp.pad(array[tuple(index)], pad_width)


def _test_fft_interp2d():
    """Test for 2d Fourier interpolation."""
    import numpy as np

    def fun2(x, y):
        return (
            2 * np.sin(1 * x[:, None])
            - 1.2 * np.cos(2 * x[:, None])
            + 3 * np.cos(3 * y[None])
            - 2 * np.cos(5 * y[None])
            + 1
        )

    x = {"o": {}, "e": {}}
    y = {"o": {}, "e": {}}
    x["o"][1] = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    x["e"][1] = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    x["o"][2] = np.linspace(0, 2 * np.pi, 133, endpoint=False)
    x["e"][2] = np.linspace(0, 2 * np.pi, 132, endpoint=False)
    y["o"][1] = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    y["e"][1] = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    y["o"][2] = np.linspace(0, 2 * np.pi, 133, endpoint=False)
    y["e"][2] = np.linspace(0, 2 * np.pi, 132, endpoint=False)

    f2 = {}
    for xp in ["o", "e"]:
        f2[xp] = {}
        for yp in ["o", "e"]:
            f2[xp][yp] = {}
            for i in [1, 2]:
                f2[xp][yp][i] = {}
                for j in [1, 2]:
                    f2[xp][yp][i][j] = fun2(x[xp][i], y[yp][j])

    shiftx = 0.2
    shifty = 0.3
    for spx in ["o", "e"]:  # source parity x
        for spy in ["o", "e"]:  # source parity y
            fi = f2[spx][spy][1][1]
            fs = fun2(x[spx][1] + shiftx, y[spy][1] + shifty)
            np.testing.assert_allclose(
                fft_interp2d(
                    fi,
                    *fi.shape,
                    shiftx,
                    shifty,
                    dx=np.diff(x[spx][1])[0],
                    dy=np.diff(y[spy][1])[0]
                ).squeeze(),
                fs,
            )
            for epx in ["o", "e"]:  # eval parity x
                for epy in ["o", "e"]:  # eval parity y
                    for sx in ["up", "down"]:  # up or downsample x
                        if sx == "up":
                            xs = 1
                            xe = 2
                        else:
                            xs = 2
                            xe = 1
                        for sy in ["up", "down"]:  # up or downsample y
                            if sy == "up":
                                ys = 1
                                ye = 2
                            else:
                                ys = 2
                                ye = 1

                            true = fun2(x[epx][xe] + shiftx, y[epy][ye] + shifty)
                            interp = fft_interp2d(
                                f2[spx][spy][xs][ys],
                                x[epx][xe].size,
                                y[epy][ye].size,
                                shiftx,
                                shifty,
                                dx=x[spx][xs][1] - x[spx][xs][0],
                                dy=y[spy][ys][1] - y[spy][ys][0],
                            ).squeeze()
                            np.testing.assert_allclose(
                                interp, true, atol=1e-12, rtol=1e-12
                            )
