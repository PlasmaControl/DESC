'''Methods for computing the critical gradient and effective radius of curvature'''

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit

# from ..equilibrium.coords import get_rtz_grid TODO why can't I import this?

# TODO can't import from desc.equilibrium because of a circular import
# def get_field_line_grid(eq,rho=0.5,alpha=0,n_pol = 4,n_points = 200):
#     '''Creates a field line aligned grid for chosen value of rho and alpha in rtz coordinates
    
#     Notes : Toroidal grid in zeta is defined between 0 and 2*(n_pol/(iota*NFP))*np.pi

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
#         np.array(rho),
#         np.array(alpha),
#         np.array(0),
#         coordinates="raz",
#         period=(np.inf,2*np.pi,np.inf)
#     )
#     iota = np.abs(eq.compute("iota",grid=initial_grid)["iota"])
#     NFP = eq.NFP
#     n_tor = n_pol/(iota*NFP)
#     zeta = np.linspace(0,2*n_tor*np.pi,n_points*n_pol)

#     # Create output grid
#     grid = get_rtz_grid(
#         eq,
#         np.array(rho),
#         np.array(alpha),
#         zeta,
#         coordinates="raz",
#         period=(np.inf,2*np.pi,np.inf),
#     )
#     return grid

def Kd_quadratic(l, Reff_inv, ln):
    '''Quadratic function for fitting the drift curvature profile

    This function models the drift curvature as a quadratic profile, where the 
    peak of the well is centered at the midpoint of the arc length (lc). It 
    fits the profile based on two parameters:
        - Reff_inv: Inverse of the effective radius of curvature.
        - ln: Characteristic length scale of the drift well.

    Notes : 
        - Uses R_eff_inv instead of 1/R_eff for the fitting as the function is used to fit
        both wells and peaks and curve_fit has trouble going through R_eff = 0 whe optimizing

    Parameters:
    l : array-like
        Arc length values.
    Reff_inv : float
        Inverse of the effective radius of curvature (1/Reff).
    ln : float
        Characteristic length of the drift well.

    Returns:
    array-like
        The quadratic drift profile values at each point in l
    '''
    lc = (l[0]+l[-1])/2
    return Reff_inv * (1 - (l - lc)**2 / ln**2)


def fit_drift_peaks(l,Kd):
    '''Fits all regions of bad curvatude of the drift curvature with a desired quadratic fit of 
    the form Kd(l) = R_eff_inv*(1-(l-l_c)^2/l_n^2) with R_eff and l_n free fitting parameters
    For more information go to https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.L032028 
    
    Notes : 
    - Considers the drift curvature is bad at the outboard midplane (zeta=0,theta=0,alpha=0) and fits
    only the curvature peaks/valleys that have the same sign as the value at tht point
    - Drift curvature convention follows the paper above Kd = a^2*∇α·(b × κ) which is equal to 
    a^2*|B|*cvdrift where cvdrift is the drift curvature defined in DESC


    Parameters
        ----------
        l : numpy.ndarray
            array of toroidal coordinates along the field line
        Kd : numpy.ndarray
            drift curvature 

    Returns:
    data : dict
        A dictionary containing the following keys:
        - 'peaks': List of tuples where each tuple contains:
            - `l_peak`  : Arc length values corresponding to the drift peak region.
            - `Kd_peak` : Curvature drift values in that region.

        - 'fits': List of fitting parameters for each drift peak. The parameters 
          are from fitting a quadratic function to each peak.

        - 'values': List of computed values for each peak:
            - `R_eff`: The effective radius of curvature obtained from the fit.
            - `L_par`: The parallel length of the identified peak.

    '''

    # Initialize the lists
    peaks = []
    fits = []
    values = []

    # Find the value at theta = 0
    val_0 = Kd[0]
    
    # Find indices where Kd changes sign (crosses zero)
    zero_crossings = np.where(np.diff(np.sign(Kd)))[0]
    
    # Initialize lists to store valid peak intervals
    peak_indices = []
    
    # TODO define a better way for the len_thres, maybe remove
    len_thres = 50 

    # Loop over zero crossing pairs to check if it's a peak or a valley
    for i in range(0, len(zero_crossings)-1):
        l_min_idx = zero_crossings[i]
        l_max_idx = zero_crossings[i + 1]
        len_peak = l_max_idx-l_min_idx

        # # Skip if there's insufficient data points
        if len_peak < len_thres:
            continue  
        
        # Check the midpoint value of Kd
        mid_idx = (l_min_idx + l_max_idx) // 2
        if Kd[mid_idx]*val_0 > 0:  # Keep the range if it's abad curvature (same as initial)
            peak_indices.append((l_min_idx, l_max_idx))
        else : 
            continue
    
    # Define p0 for fitting, the importance is on the sign as the curve_fit has trouble when it has to change sign
    p0 = np.sign(val_0)*np.array([1,1])

    # Loop through valid peak indices and perform fitting
    for l_min_idx, l_max_idx in peak_indices:
        # Extract the arc length and Kd values within the peak
        l_peak = l[l_min_idx:l_max_idx+1]
        Kd_peak = Kd[l_min_idx:l_max_idx+1]     
        
        # Fit the quadratic curve to the peak
        popt,_ = curve_fit(Kd_quadratic,l_peak,Kd_peak,p0=p0)

        R_eff = np.abs(1/popt[0])
        L_par = l[l_max_idx] - l[l_min_idx]
        
        # Store the peak data and fitting parameters
        peaks.append((l_peak, Kd_peak))
        fits.append(popt)
        values.append([R_eff,L_par])

    data = {
        "peaks" :       peaks,
        "fits"  :       fits,
        "values":       values,
    }
    return data
