"""Empirical distribution and histogram helpers."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import norm


def get_binned_data(x_data, y_data, x_range, n_bins):
    """Bin Gaussianized ``x`` values and compute mean kappa within each bin."""

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    mask = (x_data >= x_range[0]) & (x_data <= x_range[1])
    x_filtered = x_data[mask]
    y_filtered = y_data[mask]
    if x_filtered.size == 0:
        raise ValueError("No Gaussianized samples fall within x_range.")

    bin_edges = np.linspace(x_filtered.min(), x_filtered.max(), n_bins + 1)
    bin_indices = np.digitize(x_filtered, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    x_bin_centers = []
    y_bin_means = []
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            x_bin_centers.append(bin_edges[i : i + 2].mean())
            y_bin_means.append(y_filtered[bin_mask].mean())

    return np.array(x_bin_centers), np.array(y_bin_means)


def empirical_cdf(map_values):
    """Compute sorted values and clipped empirical CDF values for a map."""

    sorted_map = np.sort(np.asarray(map_values))
    cdf_values = np.arange(1, len(sorted_map) + 1) / len(sorted_map)
    cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)
    return sorted_map, cdf_values


def histogramer2d(map_values, Nbins, x_range=(-4.5, 4.5)):
    """Return Gaussianized samples plus binned transform data."""

    y_data, cdf = empirical_cdf(map_values)
    x_gaussianized = norm.ppf(cdf)
    x_avg, y_avg = get_binned_data(x_gaussianized, y_data, x_range, Nbins)
    return x_gaussianized, x_avg, y_avg


def histogram_pdf_spline(y_samples, x_eval, bins, smoothing=0.1, k=3):
    """Histogram PDF estimate with spline interpolation."""

    hist, bin_edges = np.histogram(y_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    spline = UnivariateSpline(bin_centers, hist, k=k, s=smoothing)
    result = spline(x_eval)
    if np.isscalar(x_eval):
        return max(float(result), 1e-10)
    return np.maximum(result, 1e-10)


def histogram_pdf_linear(y_samples, x_eval, bins):
    """Histogram PDF estimate with linear interpolation."""

    hist, bin_edges = np.histogram(y_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    interpolator = interp1d(bin_centers, hist, kind="linear", bounds_error=False, fill_value=1e-10)
    result = interpolator(x_eval)
    return float(result) if np.isscalar(x_eval) else result


def histogram_pdf_quadratic(y_samples, x_eval, bins):
    """Histogram PDF estimate with quadratic interpolation."""

    hist, bin_edges = np.histogram(y_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    interpolator = interp1d(
        bin_centers, hist, kind="quadratic", bounds_error=False, fill_value=1e-10
    )
    result = interpolator(x_eval)
    if np.isscalar(x_eval):
        return max(float(result), 1e-10)
    return np.maximum(result, 1e-10)


def histogram_pdf_cubic(y_samples, x_eval, bins):
    """Histogram PDF estimate with cubic interpolation."""

    hist, bin_edges = np.histogram(y_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    interpolator = interp1d(bin_centers, hist, kind="cubic", bounds_error=False, fill_value=1e-10)
    result = interpolator(x_eval)
    if np.isscalar(x_eval):
        return max(float(result), 1e-10)
    return np.maximum(result, 1e-10)
