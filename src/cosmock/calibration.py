"""Calibration objects and empirical map statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import norm

from ._optional import require_healpy
from .validation import validate_cls, validate_kappa_maps, validate_pixwin


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


def estimate_cls_from_maps(kappa_maps, *, nside: int | None = None, lmax: int | None = None):
    """Estimate auto/cross spectra from kappa maps using healpy."""

    hp = require_healpy("Estimating spectra from maps")
    maps = validate_kappa_maps(kappa_maps, nside=nside)
    if nside is None:
        nside = hp.npix2nside(maps.shape[1])
    if lmax is None:
        lmax = 3 * int(nside) - 1

    n_bins = maps.shape[0]
    cl_ng = np.zeros((n_bins, n_bins, lmax + 1), dtype=float)
    for i in range(n_bins):
        for j in range(i + 1):
            cl_ij = hp.anafast(maps[i], maps[j], lmax=lmax)
            cl_ng[i, j] = cl_ij
            cl_ng[j, i] = cl_ij
    return cl_ng


@dataclass
class KappaCalibration:
    """Calibration data derived from one or more input kappa maps."""

    maps: np.ndarray
    nside: int | None
    n_bins: int
    n_pix: int
    gaussianized_samples: list[np.ndarray]
    binned_x: list[np.ndarray]
    binned_y: list[np.ndarray]
    cl_ng: np.ndarray | None = None
    pixwin: np.ndarray | None = None
    x_range: tuple[float, float] = (-10.0, 4.5)
    n_fit_bins: int = 1000

    @classmethod
    def from_maps(
        cls,
        kappa_maps,
        *,
        nside: int | None = None,
        cl_ng=None,
        pixwin=None,
        x_range=(-10.0, 4.5),
        n_fit_bins: int = 1000,
        lmax: int | None = None,
    ) -> "KappaCalibration":
        """Build calibration statistics from input kappa maps."""

        maps = validate_kappa_maps(kappa_maps, nside=nside)
        n_bins, n_pix = maps.shape

        if cl_ng is None:
            cl_ng_arr = estimate_cls_from_maps(maps, nside=nside, lmax=lmax)
        else:
            cl_ng_arr = validate_cls(cl_ng, n_bins=n_bins, name="cl_ng")

        pixwin_arr = None
        if pixwin is not None:
            required_lmax = cl_ng_arr.shape[-1] - 1 if cl_ng_arr is not None else lmax
            pixwin_arr = validate_pixwin(pixwin, lmax=required_lmax)

        gaussianized_samples: list[np.ndarray] = []
        binned_x: list[np.ndarray] = []
        binned_y: list[np.ndarray] = []
        for map_values in maps:
            x_gaussianized, x_avg, y_avg = histogramer2d(
                map_values, n_fit_bins, x_range=x_range
            )
            gaussianized_samples.append(x_gaussianized)
            binned_x.append(x_avg)
            binned_y.append(y_avg)

        return cls(
            maps=maps,
            nside=nside,
            n_bins=n_bins,
            n_pix=n_pix,
            gaussianized_samples=gaussianized_samples,
            binned_x=binned_x,
            binned_y=binned_y,
            cl_ng=cl_ng_arr,
            pixwin=pixwin_arr,
            x_range=tuple(x_range),
            n_fit_bins=int(n_fit_bins),
        )

