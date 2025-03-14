# Adapted from interpax.
# to support https://github.com/f0uriest/interpax/issues/53
# TODO: Use rfft2 and irfft2.
from desc.backend import jnp


def fft_interp2d(
    f,
    n1,
    n2,
    sx=None,
    sy=None,
    dx=1.0,
    dy=1.0,
    is_fourier=False,
):
    """Interpolation of a 2d periodic function via FFT.

    Parameters
    ----------
    f : ndarray, shape(nx, ny, ...)
        Source data. Assumed to cover 1 full period, excluding the endpoint.
    n1, n2 : int
        Number of desired interpolation points in x and y directions
    sx, sy : ndarray or None
        Shift in x and y to evaluate at. If original data is f(x,y), interpolates to
        f(x + sx, y + sy). Both must be provided or None
    dx, dy : float
        Spacing of source points in x and y

    Returns
    -------
    fi : ndarray, shape(n1, n2, ..., len(sx))
        Interpolated (and possibly shifted) data points
    """
    c = f if is_fourier else jnp.fft.ifft2(f, axes=(0, 1))
    nx, ny = c.shape[:2]
    if (sx is not None) and (sy is not None):
        sx = jnp.exp(-1j * 2 * jnp.pi * jnp.fft.fftfreq(nx)[:, None] * sx / dx)
        sy = jnp.exp(-1j * 2 * jnp.pi * jnp.fft.fftfreq(ny)[:, None] * sy / dy)
        c = (c[None].T * sx[None, :, :] * sy[:, None, :]).T
        c = jnp.moveaxis(c, 0, -1)
    padx = ((n1 - nx) // 2, n1 - nx - (n1 - nx) // 2)
    pady = ((n2 - ny) // 2, n2 - ny - (n2 - ny) // 2)
    if nx % 2 != 0:
        padx = padx[::-1]
    if ny % 2 != 0:
        pady = pady[::-1]

    c = jnp.fft.ifftshift(
        _pad_along_axis(jnp.fft.fftshift(c, axes=0), padx, axis=0), axes=0
    )
    c = jnp.fft.ifftshift(
        _pad_along_axis(jnp.fft.fftshift(c, axes=1), pady, axis=1), axes=1
    )

    return jnp.fft.fft2(c, axes=(0, 1)).real


def _pad_along_axis(array, pad=(0, 0), axis=0):
    """Pad with zeros or truncate a given dimension."""
    array = jnp.moveaxis(array, axis, 0)

    if pad[0] < 0:
        array = array[abs(pad[0]) :]
        pad = (0, pad[1])
    if pad[1] < 0:
        array = array[: -abs(pad[1])]
        pad = (pad[0], 0)

    npad = [(0, 0)] * array.ndim
    npad[0] = pad

    array = jnp.pad(array, pad_width=npad, mode="constant", constant_values=0)
    return jnp.moveaxis(array, 0, axis)
