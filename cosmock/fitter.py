"""Fitting utilities for point-transformation parameters."""

import numpy as np
from .Gn import Gn
from scipy.optimize import minimize
from scipy.stats import norm

def variance_from_Cl(Cl):
    """Compute field variance from an angular power spectrum.

    Parameters
    ----------
    Cl : array_like
        One-dimensional angular power spectrum indexed by multipole ``ell``.

    Returns
    -------
    float
        Variance implied by ``Cl`` using
        ``sum((2 * ell + 1) * C_ell) / (4 * pi)``.
    """
    Cl = np.asarray(Cl)
    ell = np.arange(len(Cl))
    return np.sum((2*ell + 1) * Cl) / (4*np.pi)

def get_binned_data(x_data, y_data, x_range, n_bins):
    """Bin Gaussianized values and average field values in each bin.

    Parameters
    ----------
    x_data : array_like
        Gaussianized coordinates.
    y_data : array_like
        Field values paired with ``x_data``.
    x_range : tuple
        Inclusive ``(min, max)`` range of ``x_data`` to keep before binning.
    n_bins : int
        Number of bins.

    Returns
    -------
    x_bin_centers : numpy.ndarray
        Center of each populated bin.
    y_bin_means : numpy.ndarray
        Mean field value in each populated bin.
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
    """Compute the empirical cumulative distribution of a map.

    Parameters
    ----------
    map : array_like
        One-dimensional map values.

    Returns
    -------
    sorted_map : numpy.ndarray
        Sorted field values.
    cdf_values : numpy.ndarray
        Clipped empirical CDF values in the open interval ``(0, 1)``.
    """
    sorted_map = np.sort(map)
    cdf_values = np.arange(1, len(sorted_map) + 1) / len(sorted_map)
    cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)
   
    return sorted_map, cdf_values

def histogramer2d(map,Nbins,x_range=(-4.5,4.5)):
    """Compute binned Gaussianized map values for transformation fitting.

    The output corresponds to the binned points used to fit the
    point-transformation relation between a standard-normal variable and the
    non-Gaussian kappa field.

    Parameters
    ----------
    map : array_like
        One-dimensional field values to model.
    Nbins : int
        Number of Gaussianized bins.
    x_range : tuple, optional
        Inclusive range of Gaussianized values to keep. The default is
        ``(-4.5, 4.5)``.

    Returns
    -------
    x_avg : numpy.ndarray
        Standard-normal bin centers.
    y_avg : numpy.ndarray
        Mean field values.
    """
    y_data, cdf  = empirical_cdf(map)
    x_gaussianized = norm.ppf(cdf)
    x_avg,y_avg = get_binned_data(x_gaussianized,y_data,x_range,Nbins)
    return x_avg, y_avg


def fit_gn_with_constraint(x_data, y_data, N, cls, initial_lbda = None):
    """Fit constrained ``G_N`` parameters to Gaussianized field data.

    The fit minimizes squared residuals between the transformation and the
    binned field values while enforcing the variance implied by the input
    power spectrum.

    Parameters
    ----------
    x_data : array_like
        Standard-normal coordinates.
    y_data : array_like
        Non-Gaussian field values paired with ``x_data``.
    N : int
        Transformation order. The current implementation supports ``2`` and
        ``3``.
    cls : array_like
        One-dimensional angular power spectrum for the field being fit.
    initial_lbda : array_like, optional
        Initial parameter values. If omitted, a coarse grid search provides
        the initialization.

    Returns
    -------
    numpy.ndarray
        Best-fit constrained transformation parameters.

    Raises
    ------
    ValueError
        If the constrained fit is requested for an unsupported
        transformation order.
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

        try:
            y_pred = Gn(x_data, N, lbda)
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
