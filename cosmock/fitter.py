import numpy as np
from .Gn import Gn
from .structs import NonlinParameters
from scipy.optimize import minimize
from scipy.stats import norm

def variance_from_Cl(Cl):
    """
    Computes the variance of the field
    as predicted by the cls.
    
    Parameters
    ----------
    Cl : array
             Array of cls
    ell_min : float 
             First ell to start calculation. 
             Defaults to ell=0. 
    
    Returns
    -------
    variance : float
             Variance of the field as predicted by cls.
    """
    Cl = np.asarray(Cl)
    ell = np.arange(len(Cl))
    return np.sum((2*ell + 1) * Cl) / (4*np.pi)

def get_binned_data(x_data, y_data, x_range, n_bins):
    """
    Bin x values and compute mean y within each bin
    
    Parameters
    ----------
    x_data : array of Gaussianized x values
    y_data : array of corresponding y values
    n_bins : number of bins
    x_range : tuple (min, max) to keep 
    
    Returns
    -------
    x_bin_centers : x value at center of each bin
    y_bin_means : mean y value in each bin
    """

    mask = (x_data >= x_range[0]) & (x_data <= x_range[1])
    x_filtered = x_data[mask]
    y_filtered = y_data[mask]
    
    # Create bins
    bin_edges = np.linspace(x_filtered.min(), x_filtered.max(), n_bins + 1)
    
    # Find which bin each x value belongs to
    bin_indices = np.digitize(x_filtered, bin_edges) - 1
    
    # Clip to valid range
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    x_bin_centers = []
    y_bin_means   = []
    
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        # We only want to include bins with data
        if np.sum(bin_mask) > 0:  
            x_bin_centers.append(bin_edges[i:i+2].mean())  
            y_bin_means.append(y_filtered[bin_mask].mean())
    
    return np.array(x_bin_centers), np.array(y_bin_means)

def empirical_cdf(map):
    """
    Computes the CDF of a map
   
    Parameters
    ----------
    map : array of that represents map
   
    Returns
    -------
    sorted_map : x-coordinates of the CDF
    cdf_values : y-coordinates of the CDF
    """
    sorted_map = np.sort(map)
    cdf_values = np.arange(1, len(sorted_map) + 1) / len(sorted_map)
    cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)
   
    return sorted_map, cdf_values

def histogramer2d(map,Nbins,x_range=(-4.5,4.5)):
    """
    Given a NL field, computes the black triangles in
    FIG 1 in 2411.04759
    
    Parameters
    ----------
    map : field that we want to model
    
    Returns
    -------
    x_avg : x value of triangle (standard normal)
    y_avg : y value of triangle (NL field)
    """
    y_data, cdf  = empirical_cdf(map)
    x_gaussianized = norm.ppf(cdf)
    x_avg,y_avg = get_binned_data(x_gaussianized,y_data,x_range,Nbins)
    return x_avg, y_avg


def fit_gn_with_constraint(x_data, y_data, N, cls, initial_lbda = None):
    """
    Fit a Gn transformation to (x, y) data points 
    in a self-consistent manner by also including
    the variance as predicted by the power spectrum.
    
    Parameters
    ----------
    x_data : array
             array of x-values (standard normal)
    y_data : array 
             array of y-values (NL field)
    N : str
             Which G function to use ('2' and '3' are the 
             only supported)
    cls : array
             The power spectrum of the field.
    initial_lbda : array
             Initialization. If None, defaults to ones.
    
    Returns
    -------
    fitted_lbda : array
             Best fit parameters.
    """
    var = variance_from_Cl(cls)

    def calc_constrained_lbda(unconstrained_lbda, var, N):
        if N == 2:
            beta = unconstrained_lbda[0]
            alpha = np.sqrt(np.log(1 + var / beta**2))
            return np.array([alpha, beta])
        elif N == 3:
            a, b = unconstrained_lbda
            c = np.sqrt((np.exp(a**2) - 1 + 2*a*b + b**2) / var) - 1
            return np.array([a, b, c])
        else:
            raise ValueError(f"Cannot do a constrained fit for G{N}.")
        
    def cost_function(unconstrained_lbda):
        """Least squares cost"""
        lbda = calc_constrained_lbda(unconstrained_lbda, var, N)

        params = NonlinParameters(lbda = lbda, N = N,Cl_Gauss=None)

        try:
            y_pred = Gn(x_data, params)
            return np.sum((y_pred - y_data)**2)
        except:
            return np.inf
    if initial_lbda is None:
        if N == 2:
            beta_grid = np.geomspace(1e-6, 1, 50)
            cost_grid = np.array([cost_function([b]) for b in beta_grid])
            beta_init = beta_grid[np.argmin(cost_grid)]
            initial_lbda = np.array([beta_init, np.nan])
        elif N == 3:
            a_grid = np.linspace(0.01, 1.7, 25)
            b_grid = np.linspace(0.01, 2.7, 25)
            cost_grid = np.array([
                [cost_function([aa, bb]) for aa in a_grid]
                for bb in b_grid
            ])
            j_min, i_min = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
            initial_lbda = np.array([a_grid[i_min], b_grid[j_min], np.nan])
        else:
            initial_lbda = np.ones(int(N))
    initial_unconstrained_lbda = initial_lbda[:int(N)-1]

    result = minimize(
        fun=cost_function,
        x0=initial_unconstrained_lbda,
        method='BFGS'
    )
    return calc_constrained_lbda(result.x, var, N)

#a