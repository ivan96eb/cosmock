"""Angular-spectrum transformations and variance helpers."""

from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre

from .transforms import gptg_transform
from .util.quadrature import get_gh_nodes_weights
from .util.validation import spectrum_diagnostics, validate_cls

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None


_LOW_ELL_IDENTITY_VALUE = 1e-20
_SUPPORTED_SPECTRA_ORDERS = {"2", "3"}


def _validate_spectra_order(order) -> str:
    order = str(order)
    if order not in _SUPPORTED_SPECTRA_ORDERS:
        raise ValueError(f"cosmock v1 supports latent spectra conversion for GPTG orders 2 and 3; got {order}.")
    return order


def _force_low_ell_identity(cl, n_bins):
    """Apply the v1 convention that ell=0,1 are tiny diagonal spectra."""

    for ell in (0, 1):
        if ell < cl.shape[-1]:
            cl[:, :, ell] = _LOW_ELL_IDENTITY_VALUE * np.eye(n_bins)
    return cl


def _gauss_hermite_tensor_product(precomputed):
    y_nodes, y_weights = precomputed
    yi, yj = np.meshgrid(y_nodes, y_nodes, indexing="ij")
    wi, wj = np.meshgrid(y_weights, y_weights, indexing="ij")
    return yi.ravel(), yj.ravel(), (wi * wj).ravel()


def _F_gauss_hermite_grid(n, params_i, params_j, xi_g_values, precomputed):
    """Vectorized nonlinear correlation lookup on a grid of latent correlations."""

    xi_g_values = np.asarray(xi_g_values, dtype=float)
    if np.any((xi_g_values <= -1.0) | (xi_g_values >= 1.0)):
        raise ValueError("xi_g lookup values must be strictly inside (-1, 1).")

    yi, yj, weights = _gauss_hermite_tensor_product(precomputed)
    sqrt_term = np.sqrt(np.maximum(1.0 - xi_g_values**2, 0.0))
    xj = xi_g_values[:, np.newaxis] * yi[np.newaxis, :] + sqrt_term[:, np.newaxis] * yj

    gn_i = gptg_transform(yi, n, params_i)
    gn_j = gptg_transform(xj, n, params_j)
    return np.sum(gn_j * (gn_i * weights)[np.newaxis, :], axis=1)


def _invert_lookup_values(F_values, xi_g_grid, xi_NG):
    F_values = np.asarray(F_values, dtype=float)
    xi_g_grid = np.asarray(xi_g_grid, dtype=float)
    if not np.all(np.isfinite(F_values)):
        raise ValueError("G3 correlation lookup table contains non-finite values.")

    diffs = np.diff(F_values)
    increasing = bool(np.all(diffs > 0.0))
    decreasing = bool(np.all(diffs < 0.0))
    if not (increasing or decreasing):
        raise ValueError("G3 correlation lookup table must be monotone to invert.")
    if decreasing:
        F_values = F_values[::-1]
        xi_g_grid = xi_g_grid[::-1]

    lower = float(F_values[0])
    upper = float(F_values[-1])
    margin = 100.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
    out_of_range = (xi_NG < lower - margin) | (xi_NG > upper + margin)
    if np.any(out_of_range):
        bad = np.asarray(xi_NG)[out_of_range]
        raise ValueError(
            "xi_NG is outside the G3 correlation lookup range; "
            f"range=({lower:.6e}, {upper:.6e}), "
            f"first offending value={float(bad.flat[0]):.6e}."
        )

    return np.interp(np.clip(xi_NG, lower, upper), F_values, xi_g_grid)


def F_gauss_hermite_single(n, params_i, params_j, xi_g, n_nodes=40, precomputed=None):
    """Map latent Gaussian correlation to transformed-field correlation."""

    if precomputed is None:
        precomputed = get_gh_nodes_weights(n_nodes)

    values = _F_gauss_hermite_grid(n, params_i, params_j, np.asarray([xi_g]), precomputed)
    return float(values[0])


def build_lookup_table(n, params_i, params_j, xi_g_values, pre, nnodes=20):
    """Build a lookup table for the nonlinear correlation mapping."""

    if pre is None:
        pre = get_gh_nodes_weights(nnodes)
    return _F_gauss_hermite_grid(n, params_i, params_j, xi_g_values, pre)


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

    N = _validate_spectra_order(N)
    cl_NG = validate_cls(cl_NG, n_bins=N_bins, name="cl_NG")
    fitted_params = np.asarray(fitted_params, dtype=float)
    if fitted_params.shape != (N_bins, int(N)):
        raise ValueError(f"fitted_params must have shape ({N_bins}, {int(N)}) for G{N}.")
    cl_G = np.zeros_like(cl_NG)

    lmax_cl = cl_NG.shape[-1] - 1
    n_quad = max(quad_order * lmax_cl, 1)
    mu, w = np.polynomial.legendre.leggauss(n_quad)

    ell_array = np.arange(lmax_cl + 1)
    P_ell = np.array([eval_legendre(ell, mu) for ell in ell_array])
    forward_basis = ((2 * ell_array + 1)[:, np.newaxis] * P_ell) / (4 * np.pi)
    inverse_basis = 2 * np.pi * (P_ell * w[np.newaxis, :])
    xi_g_grid = np.linspace(-0.99999, 0.99999, xig_grid_size)
    pre = get_gh_nodes_weights(Nnodes)

    def process_pair_optimized(i, j):
        params_i = fitted_params[i]
        params_j = fitted_params[j]

        xi_NG = cl_NG[i, j] @ forward_basis

        if N == "2":
            alpha_i, beta_i = params_i
            alpha_j, beta_j = params_j
            denominator = beta_i * beta_j
            alpha_product = alpha_i * alpha_j
            if denominator == 0.0 or alpha_product == 0.0:
                raise ValueError("G2 inverse requires non-zero alpha and beta products.")
            log_arg = 1.0 + xi_NG / denominator
            if np.any(log_arg <= 0.0):
                bad = log_arg[log_arg <= 0.0]
                raise ValueError(
                    "G2 inverse log-domain is invalid; "
                    f"first non-positive argument={float(bad.flat[0]):.6e}."
                )
            xi_G = np.log(log_arg) / alpha_product
        else:
            F_values = build_lookup_table(N, params_i, params_j, xi_g_grid, pre, Nnodes)
            xi_G = _invert_lookup_values(F_values, xi_g_grid, xi_NG)

        clG_ij = inverse_basis @ xi_G
        clG_ij[:2] = _LOW_ELL_IDENTITY_VALUE
        return i, j, clG_ij

    pairs = [(i, j) for i in range(N_bins) for j in range(i + 1)]
    if Parallel is None or n_jobs == 1:
        results = [process_pair_optimized(i, j) for i, j in pairs]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=v, prefer="threads")(
            delayed(process_pair_optimized)(i, j) for i, j in pairs
        )

    for i, j, clG_ij in results:
        cl_G[i, j] = clG_ij
        cl_G[j, i] = clG_ij

    return _force_low_ell_identity(cl_G, N_bins)


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

    return gptg_transform(x, N, params) ** 2


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

    c_ii, c_jj = np.broadcast_arrays(
        np.asarray(c_ii, dtype=float),
        np.asarray(c_jj, dtype=float),
    )
    alpha = np.ones_like(c_ii, dtype=float)
    valid = (c_ii != 0.0) & (c_jj != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = (1.0 + Ai / c_ii[valid]) * (1.0 + Aj / c_jj[valid])
    if np.any((arg < 0.0) | ~np.isfinite(arg)):
        raise ValueError("Variance correction produced a non-finite or negative scale factor.")
    alpha[valid] = np.sqrt(arg)
    return float(alpha) if alpha.ndim == 0 else alpha


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

    return _force_low_ell_identity(cl_corrected, N_bins)
