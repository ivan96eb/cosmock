"""Angular-spectrum transformations and variance helpers."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import eval_legendre

from .quadrature import get_gh_nodes_weights
from .transforms import Gn
from .validation import spectrum_diagnostics, validate_cls

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None


def F_gauss_hermite_single(n, params_i, params_j, xi_g, n_nodes=40, precomputed=None):
    """Map latent Gaussian correlation to transformed-field correlation."""

    if precomputed is None:
        y_nodes, y_weights = get_gh_nodes_weights(n_nodes)
    else:
        y_nodes, y_weights = precomputed

    cov = np.array([[1.0, xi_g], [xi_g, 1.0]])
    L = np.linalg.cholesky(cov)

    yi, yj = np.meshgrid(y_nodes, y_nodes, indexing="ij")
    wi, wj = np.meshgrid(y_weights, y_weights, indexing="ij")
    ystack = np.stack([yi.ravel(), yj.ravel()], axis=1)
    xstack = (L @ ystack.T).T

    gn_i = Gn(xstack[:, 0], n, params_i)
    gn_j = Gn(xstack[:, 1], n, params_j)
    return np.sum(gn_i * gn_j * (wi * wj).ravel())


def build_lookup_table(n, params_i, params_j, xi_g_values, pre, nnodes=20):
    """Build a lookup table for the nonlinear correlation mapping."""

    results = []
    for xi_g in xi_g_values:
        result = F_gauss_hermite_single(n, params_i, params_j, xi_g, nnodes, pre)
        results.append(result)
    return np.array(results)


def C_NG_to_C_G(
    cl_NG,
    fitted_params,
    N_bins,
    N,
    xig_grid_size=75,
    quad_order=3,
    Nnodes=16,
    n_jobs=4,
    v=0,
):
    """Convert target non-Gaussian spectra to latent Gaussian spectra."""

    N = str(N)
    cl_NG = validate_cls(cl_NG, n_bins=N_bins, name="cl_NG")
    fitted_params = np.asarray(fitted_params, dtype=float)
    cl_G = np.zeros_like(cl_NG)

    lmax_cl = cl_NG.shape[-1] - 1
    n_quad = max(quad_order * lmax_cl, 1)
    mu, w = np.polynomial.legendre.leggauss(n_quad)

    ell_array = np.arange(lmax_cl + 1)
    P_ell = np.array([eval_legendre(ell, mu) for ell in ell_array])
    xi_g_grid = np.linspace(-0.99999, 0.99999, xig_grid_size)
    pre = get_gh_nodes_weights(Nnodes)

    def process_pair_optimized(i, j):
        params_i = fitted_params[i]
        params_j = fitted_params[j]

        ell_col = ell_array[:, np.newaxis]
        arg = (2 * ell_col + 1) * P_ell * cl_NG[i, j, :, np.newaxis]
        xi_NG = np.sum(arg, axis=0) / (4 * np.pi)

        if N == "2":
            alpha_i, beta_i = params_i
            alpha_j, beta_j = params_j
            xi_G = np.log(1 + xi_NG / (beta_i * beta_j)) / (alpha_i * alpha_j)
        else:
            F_values = build_lookup_table(N, params_i, params_j, xi_g_grid, pre, Nnodes)
            F_to_xi_g = interp1d(F_values, xi_g_grid, kind="linear", fill_value="extrapolate")
            xi_G = F_to_xi_g(xi_NG)

        integrand = P_ell * xi_G[np.newaxis, :]
        clG_ij = 2 * np.pi * np.sum(w[np.newaxis, :] * integrand, axis=1)
        clG_ij[:2] = 1e-20
        return i, j, clG_ij

    pairs = [(i, j) for i in range(N_bins) for j in range(i + 1)]
    if Parallel is None or n_jobs == 1:
        results = [process_pair_optimized(i, j) for i, j in pairs]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=v)(
            delayed(process_pair_optimized)(i, j) for i, j in pairs
        )

    for i, j, clG_ij in results:
        cl_G[i, j] = clG_ij
        cl_G[j, i] = clG_ij

    for ell in [0, 1]:
        if ell < cl_G.shape[-1]:
            cl_G[:, :, ell] = 1e-20 * np.eye(N_bins)

    return cl_G


def target_cls_to_latent_cls(cl_target, transform_params, *, order=3, n_jobs=4, **kwargs):
    """Convert target kappa/density spectra to latent Gaussian spectra ``C^x``."""

    params = np.asarray(transform_params, dtype=float)
    if params.ndim != 2:
        raise ValueError("transform_params must have shape (n_bins, n_params).")
    n_bins = params.shape[0]
    cl_target = validate_cls(cl_target, n_bins=n_bins, name="cl_target")
    cl_x = C_NG_to_C_G(
        cl_target,
        params,
        n_bins,
        str(order),
        n_jobs=n_jobs,
        **kwargs,
    )
    return validate_cls(cl_x, n_bins=n_bins, name="cl_x")


def diagnose_cl_G(cl_G):
    """Print positive-definiteness diagnostics for latent spectra."""

    diagnostics = spectrum_diagnostics(cl_G, name="cl_G", psd_atol=0.0)
    if diagnostics.positive_semidefinite:
        print("All matrices are positive definite.")
        print(f"Minimum eigenvalue across all ell: {diagnostics.min_eigenvalue:.6e}")
    else:
        for ell in diagnostics.problematic_ells[:10]:
            eigvals = np.linalg.eigvalsh(cl_G[:, :, ell])
            print(f"l={ell}: min eigenvalue = {np.min(eigvals):.6e}")
        print(f"Found {len(diagnostics.problematic_ells)} problematic ell values")
        print(f"First few: {diagnostics.problematic_ells[:10]}")
    return diagnostics


def integrand(x, N, params):
    """Integrand used for transformed PDF variance."""

    return Gn(x, N, params) ** 2


def var_pdf(N, params, n_nodes=10):
    """Compute transformed-field variance by Gauss-Hermite quadrature."""

    quads, weights = get_gh_nodes_weights(n_nodes)
    integral = weights * integrand(quads, N, params)
    return integral.sum()


def var_cl(cl):
    """Compute field variance from one angular power spectrum."""

    ell = np.arange(cl.shape[0])
    return np.sum((2 * ell + 1) * cl) / (4 * np.pi)


def compute_A(cl, N, fitted_params, N_bins):
    """Compute variance mismatch correction terms."""

    variance_pdf = np.array([var_pdf(N, fitted_params[i]) for i in range(N_bins)])
    variance_cl = np.array([var_cl(cl[i, i]) for i in range(N_bins)])
    return variance_pdf - variance_cl


def compute_alpha_ij(Ai, Aj, c_ii, c_jj):
    """Compute multiplicative variance correction for a pair of spectra."""

    arg1 = Ai / c_ii
    arg2 = Aj / c_jj
    arg3 = (Ai * Aj) / (c_ii * c_jj)
    return np.sqrt(1 + arg1 + arg2 + arg3)


def correct_cl(cl, N, fitted_params, N_bins, A=None, diag_only=True):
    """Apply a variance correction to target spectra."""

    if A is None:
        A = compute_A(cl, N, fitted_params, N_bins)
    cl_corrected = cl.copy()

    if diag_only:
        for i in range(N_bins):
            alpha_ii = compute_alpha_ij(A[i], A[i], cl[i, i], cl[i, i])
            cl_corrected[i, i] = alpha_ii * cl[i, i]
    else:
        for i in range(N_bins):
            for j in range(i + 1):
                alpha_ij = compute_alpha_ij(A[i], A[j], cl[i, i], cl[j, j])
                cl_corrected_ij = alpha_ij * cl[i, j]
                cl_corrected[i, j] = cl_corrected_ij
                cl_corrected[j, i] = cl_corrected_ij

    for ell in [0, 1]:
        if ell < cl_corrected.shape[-1]:
            cl_corrected[:, :, ell] = 1e-20 * np.eye(N_bins)
    return cl_corrected
