"""Debiasing utilities for mock spectra."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from ._optional import require_healpy
from .generation import get_kappa_lm_pixwin, get_y_maps
from .spectra import C_NG_to_C_G, diagnose_cl_G


def correct_mult_cl(cl, A):
    """Apply a multiplicative correction to all spectra."""

    N_bins = cl.shape[0]
    cl_correct = np.zeros_like(cl)
    for i in range(N_bins):
        for j in range(i + 1):
            cl_ij = np.sqrt(A[i] * A[j]) * cl[i, j]
            cl_correct[i, j] = cl_ij
            cl_correct[j, i] = cl_ij
    return cl_correct


def cl_mock_avg(
    cl_NG,
    cl_G,
    fitted_params,
    pixwin,
    pixwin_ell_filter,
    N,
    Nside,
    N_bins,
    auto=True,
    N_mocks=200,
):
    """Average mock-to-target power-spectrum ratios over generated mocks."""

    hp = require_healpy("Averaging mock spectra")
    gen_lmax = 3 * Nside - 1
    lmax = 2 * Nside
    cl_arr = np.zeros((N_mocks, N_bins, N_bins, lmax + 1))
    for mock in range(N_mocks):
        if mock % 50 == 0:
            print(f"Working on mock {mock}")
        y_maps, _ = get_y_maps(cl_G, Nside, N_bins, gen_lmax)
        kappa_lm_mock = get_kappa_lm_pixwin(
            y_maps, N_bins, N, fitted_params, Nside, pixwin_ell_filter
        )
        for i in range(N_bins):
            for j in range(i + 1):
                if auto and i != j:
                    continue
                c_ij = hp.alm2cl(kappa_lm_mock[i], kappa_lm_mock[j], lmax=lmax)
                cl_arr[mock, i, j] = c_ij
                cl_arr[mock, j, i] = c_ij

    perdiff_arr = np.zeros_like(cl_arr)
    for mock in range(N_mocks):
        perdiff_arr[mock] = cl_arr[mock] / (cl_NG[:, :, : lmax + 1] * pixwin[: lmax + 1] ** 2)
    return np.average(perdiff_arr, axis=0)


def Acoeff(average_ratio):
    """Estimate per-bin multiplicative correction coefficients."""

    Nbins = average_ratio.shape[0]
    beta = np.zeros(Nbins)
    for i in range(Nbins):
        beta[i] = np.average(average_ratio[i, i, 10:300])
    return 1 / beta


def debiaser(cl_NG, N, params, pixwin, pixwinellfilter, *, Nside, N_iter=3, Nmocks=200):
    """Iteratively debias target spectra using generated mocks."""

    Nbins = cl_NG.shape[0]
    cl_NG_corr = cl_NG
    for i in range(N_iter):
        print(f"Iteration {i}")
        cl_G = C_NG_to_C_G(cl_NG_corr, params, Nbins, N)
        diagnose_cl_G(cl_G)
        avg_ratio = cl_mock_avg(
            cl_NG, cl_G, params, pixwin, pixwinellfilter, N, Nside, Nbins, N_mocks=Nmocks
        )
        A = Acoeff(avg_ratio)
        print("beta=", 1 / A)
        cl_NG_corr = correct_mult_cl(cl_NG_corr, A)
    return cl_NG_corr


def smooth_pixwin_savgol(pixwin, window_length=11, polyorder=3):
    """Smooth a pixel-window correction with a Savitzky-Golay filter."""

    window_length = min(window_length, len(pixwin))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        return pixwin

    if np.any(~np.isfinite(pixwin)):
        mask = np.isfinite(pixwin)
        if np.sum(mask) < window_length:
            return pixwin
        pixwin_interp = np.interp(np.arange(len(pixwin)), np.arange(len(pixwin))[mask], pixwin[mask])
        return savgol_filter(pixwin_interp, window_length, polyorder)
    return savgol_filter(pixwin, window_length, polyorder)


def debiaser_premium(cl_NG, N, params, pixwin, pixwinellfilter, *, Nside, N_iter=3, Nmocks=200):
    """Iteratively debias target spectra using a smoothed ell-dependent correction."""

    Nbins = cl_NG.shape[0]
    cl_NG_corr = cl_NG
    for i in range(N_iter):
        print(f"Iteration {i}")
        cl_G = C_NG_to_C_G(cl_NG_corr, params, Nbins, N)
        diagnose_cl_G(cl_G)
        avg_ratio = cl_mock_avg(
            cl_NG,
            cl_G,
            params,
            pixwin,
            pixwinellfilter,
            N,
            Nside,
            Nbins,
            N_mocks=Nmocks,
        )
        smooth_bias = np.ones_like(avg_ratio)
        for j in range(Nbins):
            ratio_ii = smooth_pixwin_savgol(avg_ratio[j, j, 2 : 2 * 256], window_length=50)
            smooth_bias[j, j, 2 : 2 * 256] = ratio_ii
        print("beta=", 1 / Acoeff(avg_ratio))
        beta = np.array([smooth_bias[j, j] for j in range(Nbins)])
        A = 1 / beta
        cl_NG_corr = correct_mult_cl(cl_NG_corr, A)
    return cl_NG_corr


def ccl_biaser(cls_ccl, bias):
    """Bias CCL spectra to match simulated auto-spectra while preserving correlations."""

    N_cosmo = cls_ccl.shape[0]
    N_bins = cls_ccl.shape[1]
    corrected_cl = np.zeros_like(cls_ccl)
    corr_coeff = np.zeros_like(cls_ccl)
    biased_cl = np.zeros_like(cls_ccl)

    for i in range(N_bins):
        for j in range(N_bins):
            if i == j:
                corr_coeff[:, i, j] = 1.0
            else:
                corr_coeff[:, i, j] = cls_ccl[:, i, j] / np.sqrt(
                    cls_ccl[:, i, i] * cls_ccl[:, j, j]
                )

    for cosmo in range(N_cosmo):
        for i in range(N_bins):
            corrected_cl[cosmo, i, i] = cls_ccl[cosmo, i, i] * bias[i, i]

    for cosmo in range(N_cosmo):
        for i in range(N_bins):
            for j in range(N_bins):
                biased_cl[cosmo, i, j] = corr_coeff[cosmo, i, j] * np.sqrt(
                    corrected_cl[cosmo, i, i] * corrected_cl[cosmo, j, j]
                )
    mask = np.isnan(biased_cl)
    biased_cl[mask] = cls_ccl[mask]
    return biased_cl

