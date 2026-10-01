"""Methods for computing the critical gradient and effective radius of curvature."""

import functools

from desc.backend import jit, jnp


@functools.partial(jit, static_argnames=["n_wells", "order"])
def extract_Kd_wells(Kd, n_wells=5, order=False):
    """Extract wells from the Kd array by identifying sign changes.

    This function detects regions where the Kd values transition between
    positive and negative, creating a specified number of wells.
    It can also sort the wells by their lengths if desired.

    Parameters
    ----------
    Kd : jnp.ndarray
        1D array of Kd values.
    n_wells : int
        Number of wells to extract from Kd.
    order : bool
        Whether to sort the wells by their lengths in descending order.

    Returns
    -------
    Kd_wells : jnp.ndarray
        2D array where each row corresponds to the Kd values of an extracted well.
    lengths_wells : jnp.ndarray
        1D array containing the lengths of each extracted well.
    masks_wells : jnp.ndarray
        2D boolean array indicating which Kd values belong to each well.
    """
    # Step 1: Identify sign changes in Kd
    signs = jnp.sign(Kd)

    # Create masks for positive and negative crossings of the same size as Kd
    positive_crossings = jnp.zeros_like(Kd, dtype=bool)
    negative_crossings = jnp.zeros_like(Kd, dtype=bool)

    # Set negative crossings (from positive to negative)
    negative_crossings = negative_crossings.at[:-1].set(
        (signs[:-1] == 1) & (signs[1:] == -1)
    )

    # Set positive crossings (from negative to positive)
    positive_crossings = positive_crossings.at[:-1].set(
        (signs[:-1] == -1) & (signs[1:] == 1)
    )

    # Create cumulative sums for positive and negative crossings
    cumulative_positive = jnp.cumsum(positive_crossings)
    cumulative_negative = jnp.cumsum(negative_crossings)

    Kd_wells = jnp.zeros(
        (n_wells, Kd.shape[0]), dtype=Kd.dtype
    )  # Initialize with zeros
    lengths_wells = jnp.zeros(n_wells, dtype=int)
    masks_wells = jnp.zeros((n_wells, Kd.shape[0]), dtype=Kd.dtype)

    # Use a loop to fill the lengths array
    for i in range(1, n_wells + 1):
        # Create well masks
        well_mask = (cumulative_negative == i) & (
            cumulative_negative == cumulative_positive
        )
        # Fill the corresponding row in the masks array
        well_values = jnp.where(well_mask, Kd, 0)
        # Store the well values in the corresponding row
        Kd_wells = Kd_wells.at[i - 1, : well_values.size].set(well_values)
        masks_wells = masks_wells.at[i - 1, : well_values.size].set(
            well_mask.astype(Kd.dtype)
        )  # Store mask as row
        lengths_wells = lengths_wells.at[i - 1].set(well_mask.sum())

    if order:
        # Sort wells by lengths
        sort_indices = jnp.argsort(lengths_wells)[::-1]  # Descending order
        Kd_wells = Kd_wells[sort_indices]
        masks_wells = masks_wells[sort_indices]
        lengths_wells = lengths_wells[sort_indices]

    return Kd_wells, lengths_wells, masks_wells


@functools.partial(jit, static_argnames=["n_wells"])
def extract_Kd_wells_and_peaks(Kd, n_wells=5):
    """Extract wells from the Kd array by identifying sign changes.

    This function detects regions where the Kd values transition between
    positive and negative, creating a specified number of wells.
    It can also sort the wells by their lengths if desired.

    Parameters
    ----------
    Kd : jnp.ndarray
        1D array of Kd values.
    n_wells : int
        Number of wells to extract from Kd.

    Returns
    -------
    Kd_wells : jnp.ndarray
        2D array where each row corresponds to the Kd values of an extracted well.
    lengths_wells : jnp.ndarray
        1D array containing the lengths of each extracted well.
    masks_wells : jnp.ndarray
        2D boolean array indicating which Kd values belong to each well.
    """
    # Step 1: Identify sign changes in Kd
    signs = jnp.sign(Kd)

    # Create masks for positive and negative crossings of the same size as Kd
    positive_crossings = jnp.zeros_like(Kd, dtype=bool)
    negative_crossings = jnp.zeros_like(Kd, dtype=bool)

    # Set negative crossings (from positive to negative)
    negative_crossings = negative_crossings.at[:-1].set(
        (signs[:-1] == 1) & (signs[1:] == -1)
    )

    # Set positive crossings (from negative to positive)
    positive_crossings = positive_crossings.at[:-1].set(
        (signs[:-1] == -1) & (signs[1:] == 1)
    )

    # Create cumulative sums for positive and negative crossings
    cumulative_positive = jnp.cumsum(positive_crossings)
    cumulative_negative = jnp.cumsum(negative_crossings)

    Kd_wells = jnp.zeros(
        (n_wells, Kd.shape[0]), dtype=Kd.dtype
    )  # Initialize with zeros
    lengths_wells = jnp.zeros(n_wells, dtype=int)
    masks_wells = jnp.zeros((n_wells, Kd.shape[0]), dtype=Kd.dtype)

    Kd_peaks = jnp.zeros(
        (n_wells, Kd.shape[0]), dtype=Kd.dtype
    )  # Initialize with zeros
    lengths_peaks = jnp.zeros(n_wells, dtype=int)
    masks_peaks = jnp.zeros((n_wells, Kd.shape[0]), dtype=Kd.dtype)

    # Use a loop to fill the lengths array
    for i in range(n_wells):

        well_mask = (cumulative_negative == i + 1) & (
            cumulative_negative == cumulative_positive
        )
        peak_mask = (cumulative_positive == i + 1) & (
            cumulative_negative == cumulative_positive - 1
        )
        lengths_wells = lengths_wells.at[i].set(well_mask.sum())
        lengths_peaks = lengths_peaks.at[i].set(peak_mask.sum())

        # Fill the corresponding row in the masks array
        masks_wells = masks_wells.at[i].set(
            well_mask.astype(Kd.dtype)
        )  # Store mask as row
        well_values = jnp.where(well_mask, Kd, 0)
        # Store the well values in the corresponding row
        Kd_wells = Kd_wells.at[i, : well_values.size].set(well_values)

        # Fill the corresponding row in the masks array
        masks_peaks = masks_peaks.at[i].set(
            peak_mask.astype(Kd.dtype)
        )  # Store mask as row
        peak_values = jnp.where(peak_mask, Kd, 0)
        # Store the well values in the corresponding row
        Kd_peaks = Kd_peaks.at[i, : peak_values.size].set(peak_values)

    Kd_cut = {
        "Kd_wells": Kd_wells,
        "Kd_peaks": Kd_peaks,
    }

    masks = {
        "masks_wells": masks_wells,
        "masks_peaks": masks_peaks,
    }

    lengths = {
        "lengths_wells": lengths_wells,
        "lengths_peaks": lengths_peaks,
    }

    return Kd_cut, masks, lengths


@jit
def weighted_least_squares(l, Kd, mask):
    """Perform a weighted least-squares quadratic fit of Kd.

    Performs a fit of the form Kd(l) = R_eff_inv * (1 - (l - lc)^2 / ln^2)
    using only the values where the mask is True.

    Parameters
    ----------
    l : jnp.ndarray
        The coordinate array for the well.
    Kd : jnp.ndarray
        The Kd values along the well.
    mask : jnp.ndarray (bool)
        A mask indicating the valid part of the well.

    Returns
    -------
    R_eff_inv, ln_squared : float
        Fitted parameters for the quadratic fit.
    """
    # Apply mask to select only valid entries (replace invalid ones with 0)
    weights = mask.astype(float)  # Weight of 1 for valid entries, 0 for invalid

    # Calculate the center of the well (l_c) using only valid entries
    lc = jnp.sum(l * weights) / jnp.sum(weights)

    # Shift the coordinates around the center (l - lc)
    l_shifted = l - lc

    # Build the design matrix A
    A = jnp.stack([-jnp.ones_like(l), l_shifted**2], axis=-1)

    # Apply weights to both A and Kd to exclude invalid entries
    A_weighted = A * weights[:, None]
    Kd_weighted = Kd * weights

    # Solve the least-squares problem: A @ [R_eff_inv, R_eff_inv/ln^2] = Kd
    coeffs, _, _, _ = jnp.linalg.lstsq(A_weighted, Kd_weighted, rcond=None)

    R_eff_inv = coeffs[0]
    ln_squared = R_eff_inv / coeffs[1]
    return R_eff_inv, ln_squared


@functools.partial(jit, static_argnames="n_wells")
def fit_Kd_wells(l, Kd_wells, masks, n_wells=5):
    """Fit the quadratic function to each well using masks.

    Parameters
    ----------
    l : jnp.ndarray
        The coordinate array (same for all wells).
    Kd_wells : jnp.ndarray
        2D array containing the Kd values for each well.
    masks : jnp.ndarray
        2D boolean array containing the mask for each well.
    n_wells : int
        Number of wells to fit.

    Returns
    -------
    R_eff_inv_array, ln_squared_array : jnp.ndarray
        Arrays containing the fitted parameters for each well.
    """
    R_eff_inv_array = jnp.zeros(n_wells)
    ln_squared_array = jnp.zeros(n_wells)

    for i in range(n_wells):
        # Extract the mask for the current well
        well_mask = masks[i]

        # Perform the weighted least-squares fit using the mask
        R_eff_inv, ln_squared = weighted_least_squares(l, Kd_wells[i], well_mask)

        # Store the results
        R_eff_inv_array = R_eff_inv_array.at[i].set(R_eff_inv)
        ln_squared_array = ln_squared_array.at[i].set(ln_squared)

    R_eff_array = jnp.abs(1 / R_eff_inv_array)

    return R_eff_inv_array, ln_squared_array, R_eff_array
