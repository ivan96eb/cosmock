import numpy as np 
from numpy.polynomial.hermite import hermgauss
from scipy.special import eval_legendre
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from .Gn import Gn

def get_gh_nodes_weights(n_nodes):
    """
    Computes the quadrature points and weights to 
    solve integrals for the form I = E[f(x)] where x
    is standard normal. So, to solve integral
    simply do np.sum(y_weights * f(y_nodes)).
    
    Parameters
    ----------
    x : int
            Number of quadrature points
    
    Returns
    -------
    points_weights : tuple
            Returns the quadrature points, weights.
    """
    t, w = hermgauss(n_nodes)
    y_nodes = np.sqrt(2.0) * t
    y_weights = w / np.sqrt(np.pi)
    return y_nodes, y_weights


def F_gauss_hermite_single(n, params_i, params_j, xi_g,
                           n_nodes=40,
                           precomputed=None):
    if precomputed is None:
        y_nodes, y_weights = get_gh_nodes_weights(n_nodes)
    else:
        y_nodes, y_weights = precomputed

    cov = np.array([[1.0, xi_g],
                    [xi_g, 1.0]])
    L = np.linalg.cholesky(cov)  

    # We build tensor product nodes and weights 
    YI, YJ = np.meshgrid(y_nodes, y_nodes, indexing='ij')
    WI, WJ = np.meshgrid(y_weights, y_weights, indexing='ij')

    # We map to X coordinates
    Ystack = np.stack([YI.ravel(), YJ.ravel()], axis=1)   
    Xstack = (L @ Ystack.T).T                             

    xi_vals = Xstack[:, 0]
    xj_vals = Xstack[:, 1]

    Gn_i = Gn(xi_vals, params_i)
    Gn_j = Gn(xj_vals, params_j)

    integrand = Gn_i * Gn_j * (WI * WJ).ravel()
    result = integrand.sum()
    return result

def build_lookup_table(n, params_i,params_j, xi_g_values, pre,nnodes=20):
    results = []
    for xi_g in xi_g_values:
        result = F_gauss_hermite_single(n, params_i,params_j, xi_g, nnodes,pre)
        results.append(result)
    return np.array(results)


def C_NG_to_C_G(cl_NG, fitted_params, N_bins, N, 
                          xig_grid_size=75, quad_order=3, Nnodes=16, 
                          n_jobs=4, v=0):
    """
    Converts non-Gaussian power spectrum to a gaussianized version
    given the parameters
    
    Parameters
    ----------
    cl_NG : array
        Non-Gaussian power spectrum
    fitted_params : array
        Parameters for Gn function for each z-bin
    N_bins : int
        Number of redshift bins
    N : int
        Parameter for Gn function
    xig_grid_size : int, optional
        Number of points in xi_g interpolation grid (default: 75)
    quad_order : int, optional
        Gauss-Legendre quadrature order multiplier (default: 3)
        Total points = quad_order * lmax
    Nnodes : int, optional
        Number of Gauss-Hermite nodes for F integral (default: 16)
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores (default: 4)
    v : int, optional
        Verbosity level for joblib (default: 0)
    
    Returns
    -------
    cl_G : array
        Gaussian power spectrum
    """    
    cl_G = np.zeros_like(cl_NG)
    
    # Determine number of quadrature points needed
    lmax_cl = cl_NG.shape[-1] - 1
    n_quad  = quad_order * lmax_cl
    
    # We compute Gauss-Legendre quadrature points and weights
    mu, w = np.polynomial.legendre.leggauss(n_quad)
    
    # Pre-compute Legendre polynomials, instead of computing them
    # each time 
    ell_array = np.arange(lmax_cl + 1)
    P_ell     = np.array([eval_legendre(ell, mu) for ell in ell_array])

    # Setup for lookup table
    xi_g_grid = np.linspace(-0.99999, 0.99999, xig_grid_size)
    pre       = get_gh_nodes_weights(Nnodes)
    
    def process_pair_optimized(i, j):
        params_i = fitted_params[i]
        params_j = fitted_params[j]
                
        # Cl_NG -> xi_NG 
        ell_col = ell_array[:, np.newaxis]
        arg = (2*ell_col + 1) * P_ell * cl_NG[i, j, :, np.newaxis]
        xi_NG = np.sum(arg, axis=0) / (4*np.pi)
        
        # xi_NG -> xi_G
        # Only in the case N = 2 we have an analitycal relation.
        # In the other cases, we have to build lookup table.
        if N == 2:
            alpha_i, beta_i = params_i
            alpha_j, beta_j = params_j
            xi_G = np.log(1 + xi_NG / (beta_i * beta_j)) / (alpha_i * alpha_j)
        else:
            F_values = build_lookup_table(N, params_i, params_j, xi_g_grid, pre, Nnodes)
            F_to_xi_g = interp1d(F_values, xi_g_grid, kind='linear',
                                fill_value='extrapolate')
            xi_G = F_to_xi_g(xi_NG)
        
        # xi_G -> Cl_G 
        integrand = P_ell * xi_G[np.newaxis, :]
        clG_ij = 2 * np.pi * np.sum(w[np.newaxis, :] * integrand, axis=1)
        clG_ij[:2] = 1e-20
        
        return i, j, clG_ij
    
    # Generate list of pairs to process
    pairs = [(i, j) for i in range(N_bins) for j in range(i + 1)]
    
    # Parallel computation
    results = Parallel(n_jobs=n_jobs, verbose=v)(
        delayed(process_pair_optimized)(i, j) for i, j in pairs
    )
    
    # We fill the cl_G array with our results.
    for i, j, clG_ij in results:
        cl_G[i, j] = clG_ij
        cl_G[j, i] = clG_ij
    
    # ell = 0,1 are zero. This gives problems
    # Instead, we make a positive definite matrix
    # with very small eigenvalues. 
    for l in [0, 1]:
        cl_G[:, :, l] = 1e-20 * np.eye(N_bins)
    
    return cl_G