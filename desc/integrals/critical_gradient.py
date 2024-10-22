'''Methods for computing the critical gradient and effective radius of curvature'''

from desc.backend import jnp, jit
import functools


# @functools.partial(jit,static_argnames=["n_wells","order"])
def extract_Kd_wells(Kd,n_wells=7,order=False):
    # Step 1: Identify sign changes in Kd
    signs = jnp.sign(Kd)
    
    # Create masks for positive and negative crossings of the same size as Kd
    positive_crossings = jnp.zeros_like(Kd, dtype=bool)
    negative_crossings = jnp.zeros_like(Kd, dtype=bool)

    # Set negative crossings (from positive to negative)
    negative_crossings = negative_crossings.at[:-1].set((signs[:-1] == 1) & (signs[1:] == -1))

    # Set positive crossings (from negative to positive)
    positive_crossings = positive_crossings.at[:-1].set((signs[:-1] == -1) & (signs[1:] == 1))

    # Create cumulative sums for positive and negative crossings
    cumulative_positive = jnp.cumsum(positive_crossings)
    cumulative_negative = jnp.cumsum(negative_crossings)

    Kd_wells = jnp.zeros((n_wells, Kd.shape[0]), dtype=Kd.dtype)  # Initialize with zeros
    lengths_wells = jnp.zeros(n_wells, dtype=int)
    masks_wells = jnp.zeros((n_wells, Kd.shape[0]), dtype=Kd.dtype)

    # Use a loop to fill the lengths array
    for i in range(1,n_wells+1):
        # Create well masks
        well_mask = (cumulative_negative == i) & (cumulative_negative == cumulative_positive)        
        # Fill the corresponding row in the masks array
        well_values = jnp.where(well_mask,Kd,0)
        # Store the well values in the corresponding row
        Kd_wells = Kd_wells.at[i-1, :well_values.size].set(well_values)
        masks_wells = masks_wells.at[i-1, :well_values.size].set(well_mask.astype(Kd.dtype))  # Store mask as row
        lengths_wells = lengths_wells.at[i-1].set(well_mask.sum())

    if order : 
        # Sort wells by lengths
        sort_indices = jnp.argsort(lengths_wells)[::-1]  # Descending order
        Kd_wells = Kd_wells[sort_indices]
        lengths_wells = lengths_wells[sort_indices]

    # return Kd_wells[0:n_return], lengths_wells[0:n_return], masks_wells[0:n_return]
    return Kd_wells, lengths_wells, masks_wells



# @jit
def weighted_least_squares(l, Kd, mask):
    """
    Perform a weighted least-squares quadratic fit:
    Kd(l) = R_eff_inv * (1 - (l - lc)^2 / ln^2)
    using only the values where the mask is True.

    Parameters:
    -----------
    l : jnp.ndarray
        The coordinate array for the well.
    Kd : jnp.ndarray
        The Kd values along the well.
    mask : jnp.ndarray (bool)
        A mask indicating the valid part of the well.

    Returns:
    --------
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



# @functools.partial(jit, static_argnames="n_wells")
def fit_Kd_wells(l, Kd_wells, masks, n_wells=7):
    """
    Fit the quadratic function to each well using masks.

    Parameters
    -----------
    l : jnp.ndarray
        The coordinate array (same for all wells).
    Kd_wells : jnp.ndarray
        2D array containing the Kd values for each well.
    masks : jnp.ndarray
        2D boolean array containing the mask for each well.
    n_wells : int
        Number of wells to fit.

    Returns:
    --------
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
    
    R_eff_array = jnp.abs(1/R_eff_inv_array)

    return R_eff_inv_array, ln_squared_array, R_eff_array



### OLD ###

# from ..equilibrium.coords import get_rtz_grid TODO why can't I import this?

# TODO can't import from desc.equilibrium because of a circular import
# def get_field_line_grid(eq,rho=0.5,alpha=0,n_pol = 4,n_points = 200):
#     '''Creates a field line aligned grid for chosen value of rho and alpha in rtz coordinates
    
#     Notes : Toroidal grid in zeta is defined between 0 and 2*(n_pol/(iota*NFP))*jnp.pi

#     Parameters
#         ----------
#         eq : equilibrium
#             A DESC equilibrium object
#         rho : float
#             DESC rho coordinate, has to be between 0 and 1
#             default : rho=0.5 surface
#         alpha : float
#             field line label
#             default : alpha = 0
#         n_pol : int
#             number of poloidal turns for the grid
#             default : 4
#         n_points : int
#             number of points for each poloidal turn
#             default : 200

#         Returns
#         -------
#        grid : Grid
#             grid in rtz coordinates for a chosen flux surface and field line label    
#     '''
#     # Get initial grid to get iota value on the chosen field line
#     initial_grid = get_rtz_grid(
#         eq,
#         jnp.array(rho),
#         jnp.array(alpha),
#         jnp.array(0),
#         coordinates="raz",
#         period=(jnp.inf,2*jnp.pi,jnp.inf)
#     )
#     iota = jnp.abs(eq.compute("iota",grid=initial_grid)["iota"])
#     NFP = eq.NFP
#     n_tor = n_pol/(iota*NFP)
#     zeta = jnp.linspace(0,2*n_tor*jnp.pi,n_points*n_pol)

#     # Create output grid
#     grid = get_rtz_grid(
#         eq,
#         jnp.array(rho),
#         jnp.array(alpha),
#         zeta,
#         coordinates="raz",
#         period=(jnp.inf,2*jnp.pi,jnp.inf),
#     )
#     return grid

# def Kd_quadratic(l, Reff_inv, ln):
#     '''Quadratic function for fitting the drift curvature profile

#     This function models the drift curvature as a quadratic profile, where the 
#     peak of the well is centered at the midpoint of the arc length (lc). It 
#     fits the profile based on two parameters:
#         - Reff_inv: Inverse of the effective radius of curvature.
#         - ln: Characteristic length scale of the drift well.

#     Notes : 
#         - Uses R_eff_inv instead of 1/R_eff for the fitting as the function is used to fit
#         both wells and peaks and curve_fit has trouble going through R_eff = 0 whe optimizing

#     Parameters:
#     l : array-like
#         Arc length values.
#     Reff_inv : float
#         Inverse of the effective radius of curvature (1/Reff).
#     ln : float
#         Characteristic length of the drift well.

#     Returns:
#     array-like
#         The quadratic drift profile values at each point in l
#     '''
#     lc = (l[0]+l[-1])/2
#     return Reff_inv * (1 - (l - lc)**2 / ln**2)

# def fit_drift_peaks(l,Kd):
#     '''Fits all regions of bad curvatude of the drift curvature with a desired quadratic fit of 
#     the form Kd(l) = R_eff_inv*(1-(l-l_c)^2/l_n^2) with R_eff and l_n free fitting parameters
#     For more information go to https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.L032028 
    
#     Notes : 
#     - Considers the drift curvature is bad at the outboard midplane (zeta=0,theta=0,alpha=0) and fits
#     only the curvature peaks/valleys that have the same sign as the value at tht point
#     - Drift curvature convention follows the paper above Kd = a^2*∇α·(b × κ) which is equal to 
#     a^2*|B|*cvdrift where cvdrift is the drift curvature defined in DESC


#     Parameters
#         ----------
#         l : numpy.ndarray
#             array of toroidal coordinates along the field line
#         Kd : numpy.ndarray
#             drift curvature 

#     Returns:
#     data : dict
#         A dictionary containing the following keys:
#         - 'peaks': List of tuples where each tuple contains:
#             - `l_peak`  : Arc length values corresponding to the drift peak region.
#             - `Kd_peak` : Curvature drift values in that region.

#         - 'fits': List of fitting parameters for each drift peak. The parameters 
#           are from fitting a quadratic function to each peak.

#         - 'values': List of computed values for each peak:
#             - `R_eff`: The effective radius of curvature obtained from the fit.
#             - `L_par`: The parallel length of the identified peak.

#     '''

#     # Initialize the lists
#     peaks = []
#     fits = []
#     values = []

#     # Find the value at theta = 0
#     val_0 = Kd[0]
    
#     # Find indices where Kd changes sign (crosses zero)
#     zero_crossings = jnp.where(jnp.diff(jnp.sign(Kd)))[0]
    
#     # Initialize lists to store valid peak intervals
#     peak_indices = []
    
#     # TODO define a better way for the len_thres, maybe remove
#     len_thres = 50 

#     # Loop over zero crossing pairs to check if it's a peak or a valley
#     for i in range(0, len(zero_crossings)-1):
#         l_min_idx = zero_crossings[i]
#         l_max_idx = zero_crossings[i + 1]
#         len_peak = l_max_idx-l_min_idx

#         # # Skip if there's insufficient data points
#         if len_peak < len_thres:
#             continue  
        
#         # Check the midpoint value of Kd
#         mid_idx = (l_min_idx + l_max_idx) // 2
#         if Kd[mid_idx]*val_0 > 0:  # Keep the range if it's abad curvature (same as initial)
#             peak_indices.append((l_min_idx, l_max_idx))
#         else : 
#             continue
    
#     # Define p0 for fitting, the importance is on the sign as the curve_fit has trouble when it has to change sign
#     p0 = jnp.sign(val_0)*jnp.array([1,1])

#     # Loop through valid peak indices and perform fitting
#     for l_min_idx, l_max_idx in peak_indices:
#         # Extract the arc length and Kd values within the peak
#         l_peak = l[l_min_idx:l_max_idx+1]
#         Kd_peak = Kd[l_min_idx:l_max_idx+1]     
        
#         # Fit the quadratic curve to the peak
#         popt,_ = curve_fit(Kd_quadratic,l_peak,Kd_peak,p0=p0)

#         R_eff = jnp.abs(1/popt[0])
#         L_par = l[l_max_idx] - l[l_min_idx]
        
#         # Store the peak data and fitting parameters
#         peaks.append((l_peak, Kd_peak))
#         fits.append(popt)
#         values.append([R_eff,L_par])

#     data = {
#         "peaks" :       peaks,
#         "fits"  :       fits,
#         "values":       values,
#     }
#     return data
