"""Angular power-spectrum conversion utilities."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from numpy.polynomial.hermite import hermgauss
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from scipy.special import eval_legendre

from .transforms import evaluate_transform, validate_order


def gauss_hermite_nodes(
    n_nodes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return nodes and weights for standard-normal expectations."""

    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError("n_nodes must be a positive integer.")
    nodes, weights = hermgauss(n_nodes)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def _transformed_correlation_single(
    order: int,
    params_i: NDArray[np.float64],
    params_j: NDArray[np.float64],
    xi_latent: float,
    precomputed: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> float:
    y_nodes, y_weights = precomputed
    xi_clipped = float(np.clip(xi_latent, -0.99999, 0.99999))
    covariance = np.asarray([[1.0, xi_clipped], [xi_clipped, 1.0]])
    factor = np.linalg.cholesky(covariance)

    y_i, y_j = np.meshgrid(y_nodes, y_nodes, indexing="ij")
    w_i, w_j = np.meshgrid(y_weights, y_weights, indexing="ij")
    y_stack = np.stack([y_i.ravel(), y_j.ravel()], axis=1)
    x_stack = (factor @ y_stack.T).T

    transformed_i = evaluate_transform(x_stack[:, 0], order, params_i)
    transformed_j = evaluate_transform(x_stack[:, 1], order, params_j)
    return float(np.sum(transformed_i * transformed_j * (w_i * w_j).ravel()))


def _build_correlation_lookup(
    order: int,
    params_i: NDArray[np.float64],
    params_j: NDArray[np.float64],
    xi_latent_grid: NDArray[np.float64],
    precomputed: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    return np.asarray(
        [
            _transformed_correlation_single(
                order,
                params_i,
                params_j,
                xi_latent,
                precomputed,
            )
            for xi_latent in xi_latent_grid
        ]
    )


def field_cl_to_latent_cl(
    cl_field: ArrayLike,
    transform_params: ArrayLike,
    order: int,
    *,
    xig_grid_size: int = 75,
    quad_order: int = 3,
    n_nodes: int = 16,
    n_jobs: int = 1,
    joblib_verbosity: int = 0,
) -> NDArray[np.float64]:
    """Convert target scalar-field spectra into latent Gaussian spectra.

    Parameters
    ----------
    cl_field : array_like of shape ``(n_bins, n_bins, n_ell)``
        Target non-Gaussian scalar-field angular power spectra.
    transform_params : array_like of shape ``(n_bins, order)``
        Fitted point-transformation parameters.
    order : {2, 3}
        Transformation order.
    xig_grid_size : int, optional
        Number of latent-correlation grid points for lookup inversion.
    quad_order : int, optional
        Multiplier for the Gauss-Legendre quadrature size.
    n_nodes : int, optional
        Number of Gauss-Hermite nodes.
    n_jobs : int, optional
        Number of joblib workers. The default is serial execution.
    joblib_verbosity : int, optional
        Joblib verbosity level.

    Returns
    -------
    ndarray
        Latent Gaussian spectra with shape ``(n_bins, n_bins, n_ell)``.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If inputs are malformed or contain non-finite values.
    """

    order = validate_order(order)
    cl_array = np.asarray(cl_field, dtype=float)
    params_array = np.asarray(transform_params, dtype=float)

    if cl_array.ndim != 3:
        raise ValueError("cl_field must have shape (n_bins, n_bins, n_ell).")
    n_bins, n_bins_2, n_ell = cl_array.shape
    if n_bins != n_bins_2:
        raise ValueError("cl_field must be square in its first two axes.")
    if n_ell < 3:
        raise ValueError("cl_field must contain at least ell=0, 1, and 2.")
    if params_array.shape != (n_bins, order):
        raise ValueError(
            "transform_params must have shape (n_bins, order) matching cl_field."
        )
    if not np.all(np.isfinite(cl_array)):
        raise ValueError("cl_field must contain only finite values.")
    if not np.all(np.isfinite(params_array)):
        raise ValueError("transform_params must contain only finite values.")
    if not isinstance(xig_grid_size, int) or xig_grid_size < 3:
        raise ValueError("xig_grid_size must be an integer greater than 2.")
    if not isinstance(quad_order, int) or quad_order <= 0:
        raise ValueError("quad_order must be a positive integer.")
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError("n_nodes must be a positive integer.")
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer.")

    cl_latent = np.zeros_like(cl_array)
    lmax_cl = n_ell - 1
    n_quad = quad_order * lmax_cl
    mu, weights = np.polynomial.legendre.leggauss(n_quad)

    ell_array = np.arange(lmax_cl + 1)
    p_ell = np.asarray([eval_legendre(ell, mu) for ell in ell_array])
    xi_latent_grid = np.linspace(-0.99999, 0.99999, xig_grid_size)
    precomputed = gauss_hermite_nodes(n_nodes)

    def process_pair(i: int, j: int) -> tuple[int, int, NDArray[np.float64]]:
        params_i = params_array[i]
        params_j = params_array[j]

        ell_column = ell_array[:, np.newaxis]
        integrand = (2 * ell_column + 1) * p_ell * cl_array[i, j, :, np.newaxis]
        xi_field = np.sum(integrand, axis=0) / (4 * np.pi)

        if order == 2:
            alpha_i, beta_i = params_i
            alpha_j, beta_j = params_j
            domain = 1.0 + xi_field / (beta_i * beta_j)
            if np.any(domain <= 0.0):
                raise ValueError(
                    "order=2 spectra conversion encountered a non-positive "
                    "logarithm domain."
                )
            xi_latent = np.log(domain) / (alpha_i * alpha_j)
        else:
            transformed_values = _build_correlation_lookup(
                order,
                params_i,
                params_j,
                xi_latent_grid,
                precomputed,
            )
            interpolator = interp1d(
                transformed_values,
                xi_latent_grid,
                kind="linear",
                fill_value="extrapolate",
            )
            xi_latent = interpolator(xi_field)

        cl_pair = 2 * np.pi * np.sum(
            weights[np.newaxis, :] * p_ell * xi_latent[np.newaxis, :],
            axis=1,
        )
        cl_pair[:2] = 1e-20
        return i, j, cl_pair

    pairs = [(i, j) for i in range(n_bins) for j in range(i + 1)]
    results = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
        delayed(process_pair)(i, j) for i, j in pairs
    )

    for i, j, cl_pair in results:
        cl_latent[i, j] = cl_pair
        cl_latent[j, i] = cl_pair

    for ell_index in (0, 1):
        cl_latent[:, :, ell_index] = 1e-20 * np.eye(n_bins)

    return cl_latent
