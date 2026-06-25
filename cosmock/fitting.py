"""Fitting utilities for scalar-field point-transformation models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import norm

from .transforms import evaluate_transform, validate_order


def variance_from_cl(cl: ArrayLike) -> float:
    """Compute field variance from an angular power spectrum.

    Parameters
    ----------
    cl : array_like of shape ``(n_ell,)``
        Angular power spectrum indexed by multipole ``ell``.

    Returns
    -------
    float
        Variance implied by ``sum((2 * ell + 1) * C_ell) / (4 * pi)``.

    Raises
    ------
    ValueError
        If ``cl`` is not one-dimensional, finite, and non-empty.
    """

    cl_array = np.asarray(cl, dtype=float)
    if cl_array.ndim != 1 or cl_array.size == 0:
        raise ValueError("cl must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(cl_array)):
        raise ValueError("cl must contain only finite values.")

    ell = np.arange(cl_array.size)
    return float(np.sum((2 * ell + 1) * cl_array) / (4 * np.pi))


def empirical_cdf(values: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute sorted values and their clipped empirical CDF.

    Parameters
    ----------
    values : array_like of shape ``(n_pix,)``
        One-dimensional scalar-field map values.

    Returns
    -------
    sorted_values, cdf_values : ndarray
        Sorted field values and CDF values clipped into ``(0, 1)``.

    Raises
    ------
    ValueError
        If values are not one-dimensional, finite, and non-empty.
    """

    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 1 or value_array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(value_array)):
        raise ValueError("values must contain only finite entries.")

    sorted_values = np.sort(value_array)
    cdf_values = np.arange(1, sorted_values.size + 1) / sorted_values.size
    return sorted_values, np.clip(cdf_values, 1e-10, 1.0 - 1e-10)


def binned_means(
    x_data: ArrayLike,
    y_data: ArrayLike,
    x_range: tuple[float, float],
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin paired data and return populated-bin centers and means."""

    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    if len(x_range) != 2 or x_range[0] >= x_range[1]:
        raise ValueError("x_range must be an increasing (min, max) pair.")

    x_array = np.asarray(x_data, dtype=float)
    y_array = np.asarray(y_data, dtype=float)
    if x_array.shape != y_array.shape:
        raise ValueError("x_data and y_data must have the same shape.")
    if x_array.ndim != 1 or x_array.size == 0:
        raise ValueError("x_data and y_data must be non-empty one-dimensional arrays.")

    mask = (x_array >= x_range[0]) & (x_array <= x_range[1])
    if not np.any(mask):
        raise ValueError("x_range excludes all input samples.")

    x_filtered = x_array[mask]
    y_filtered = y_array[mask]
    bin_edges = np.linspace(x_filtered.min(), x_filtered.max(), n_bins + 1)
    bin_indices = np.clip(np.digitize(x_filtered, bin_edges) - 1, 0, n_bins - 1)

    x_bin_centers = []
    y_bin_means = []
    for bin_index in range(n_bins):
        bin_mask = bin_indices == bin_index
        if np.any(bin_mask):
            x_bin_centers.append(bin_edges[bin_index : bin_index + 2].mean())
            y_bin_means.append(y_filtered[bin_mask].mean())

    return np.asarray(x_bin_centers), np.asarray(y_bin_means)


def histogram_gaussianized(
    values: ArrayLike,
    n_bins: int,
    x_range: tuple[float, float] = (-4.5, 4.5),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gaussianize scalar-field values by rank and return binned fit points."""

    sorted_values, cdf_values = empirical_cdf(values)
    x_gaussianized = norm.ppf(cdf_values)
    return binned_means(x_gaussianized, sorted_values, x_range, n_bins)


def fit_transform_params(
    x_data: ArrayLike,
    y_data: ArrayLike,
    order: int,
    cl_field: ArrayLike,
    initial_params: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Fit constrained transformation parameters for one scalar-field map.

    Parameters
    ----------
    x_data, y_data : array_like of shape ``(n_samples,)``
        Gaussianized coordinates and matching field values.
    order : {2, 3}
        Transformation order.
    cl_field : array_like of shape ``(n_ell,)``
        Auto-spectrum for the fitted field bin.
    initial_params : array_like, optional
        Optional starting parameters. Only the first ``order - 1`` values are
        optimized directly because the final constrained parameter is derived
        from the target variance.

    Returns
    -------
    ndarray of shape ``(order,)``
        Fitted transformation parameters.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If inputs are malformed, unsupported, or imply non-positive variance.
    """

    order = validate_order(order)
    x_array = np.asarray(x_data, dtype=float)
    y_array = np.asarray(y_data, dtype=float)
    if x_array.shape != y_array.shape:
        raise ValueError("x_data and y_data must have the same shape.")
    if x_array.ndim != 1 or x_array.size == 0:
        raise ValueError("x_data and y_data must be non-empty one-dimensional arrays.")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("x_data and y_data must contain only finite values.")

    variance = variance_from_cl(cl_field)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("cl_field must imply a positive finite variance.")

    def constrained_params(unconstrained_params: ArrayLike) -> NDArray[np.float64]:
        params = np.asarray(unconstrained_params, dtype=float)
        if order == 2:
            beta = params[0]
            if beta == 0.0:
                return np.full(2, np.nan)
            alpha = np.sqrt(np.log1p(variance / beta**2))
            return np.asarray([alpha, beta])

        a, b = params
        numerator = np.exp(a**2) - 1.0 + 2.0 * a * b + b**2
        if numerator < 0.0:
            return np.full(3, np.nan)
        c = np.sqrt(numerator / variance) - 1.0
        return np.asarray([a, b, c])

    def cost_function(unconstrained_params: ArrayLike) -> float:
        params = constrained_params(unconstrained_params)
        if not np.all(np.isfinite(params)):
            return np.inf
        try:
            y_pred = evaluate_transform(x_array, order, params)
        except (FloatingPointError, OverflowError, ValueError):
            return np.inf
        if not np.all(np.isfinite(y_pred)):
            return np.inf
        return float(np.sum((y_pred - y_array) ** 2))

    if initial_params is None:
        if order == 2:
            beta_grid = np.geomspace(1e-6, 1.0, 50)
            cost_grid = np.asarray([cost_function([beta]) for beta in beta_grid])
            beta_init = beta_grid[int(np.argmin(cost_grid))]
            initial_params_array = np.asarray([beta_init, np.nan])
        else:
            a_grid = np.linspace(0.01, 1.7, 25)
            b_grid = np.linspace(0.01, 2.7, 25)
            cost_grid = np.asarray(
                [[cost_function([a, b]) for a in a_grid] for b in b_grid]
            )
            b_index, a_index = np.unravel_index(
                int(np.argmin(cost_grid)), cost_grid.shape
            )
            initial_params_array = np.asarray(
                [a_grid[a_index], b_grid[b_index], np.nan]
            )
    else:
        initial_params_array = np.asarray(initial_params, dtype=float)
        if initial_params_array.shape != (order,):
            raise ValueError(f"initial_params must have shape ({order},).")

    result = minimize(
        fun=cost_function,
        x0=initial_params_array[: order - 1],
        method="BFGS",
    )
    fitted_params = constrained_params(result.x)
    if not np.all(np.isfinite(fitted_params)):
        raise ValueError("fit did not produce finite transformation parameters.")
    return fitted_params
