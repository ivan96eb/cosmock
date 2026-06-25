"""HEALPix sampling utilities for scalar-field mock generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .transforms import evaluate_transform, validate_order


def _healpy():
    import healpy as hp

    return hp


def pack_standard_alm(
    xlm_real: NDArray[np.float64],
    xlm_imag: NDArray[np.float64],
    generation_lmax: int,
    n_bins: int,
) -> NDArray[np.complex128]:
    """Pack real and imaginary standard-normal draws into HEALPix alm arrays."""

    hp = _healpy()
    ell, emm = hp.Alm.getlm(generation_lmax)
    alm_real = np.zeros((n_bins, ell.size))
    alm_imag = np.zeros_like(alm_real)
    alm_real[:, ell > 1] = xlm_real
    alm_imag[:, (ell > 1) & (emm > 0)] = xlm_imag
    return alm_real + 1j * alm_imag


def draw_standard_alm(
    n_bins: int,
    generation_lmax: int,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Draw standard-normal harmonic coefficients for latent fields."""

    hp = _healpy()
    ell, emm = hp.Alm.getlm(generation_lmax)
    xlm_real = rng.normal(size=(n_bins, np.count_nonzero(ell > 1)))
    xlm_imag = rng.normal(size=(n_bins, np.count_nonzero((ell > 1) & (emm > 0))))
    return pack_standard_alm(xlm_real, xlm_imag, generation_lmax, n_bins)


def _covariance_factor(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as exc:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        tolerance = max(float(np.max(np.abs(eigenvalues))) * 1e-12, 1e-30)
        if np.min(eigenvalues) < -tolerance:
            raise np.linalg.LinAlgError(
                "latent spectra contain a non-positive-semidefinite covariance "
                "matrix."
            ) from exc
        return eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))


def _spectra_square_root_by_ell(
    cl_latent: NDArray[np.float64],
) -> NDArray[np.float64]:
    factors_by_ell = np.asarray(
        [_covariance_factor(cl_latent[:, :, ell]) for ell in range(cl_latent.shape[2])]
    )
    return np.moveaxis(factors_by_ell, 0, 2)


def apply_latent_spectra(
    standard_alm: NDArray[np.complex128],
    cl_latent: ArrayLike,
    generation_lmax: int,
) -> NDArray[np.complex128]:
    """Apply latent spectra to standard-normal alm coefficients."""

    cl_array = np.asarray(cl_latent, dtype=float)
    if cl_array.ndim != 3:
        raise ValueError("cl_latent must have shape (n_bins, n_bins, n_ell).")
    n_bins, n_bins_2, n_ell = cl_array.shape
    if n_bins != n_bins_2:
        raise ValueError("cl_latent must be square in its first two axes.")
    if n_ell <= generation_lmax:
        raise ValueError("cl_latent does not cover generation_lmax.")
    hp = _healpy()
    if standard_alm.shape != (n_bins, hp.Alm.getsize(generation_lmax)):
        raise ValueError("standard_alm shape does not match cl_latent and lmax.")
    if not np.all(np.isfinite(cl_array)):
        raise ValueError("cl_latent must contain only finite values.")

    ell, emm = hp.Alm.getlm(generation_lmax)
    factors = _spectra_square_root_by_ell(cl_array[:, :, : generation_lmax + 1])
    factors_expanded = factors[:, :, ell]

    alm_real = np.einsum("ijm,jm->im", factors_expanded, standard_alm.real)
    alm_imag = np.einsum("ijm,jm->im", factors_expanded, standard_alm.imag)
    alm_real = alm_real / np.sqrt(2.0)
    alm_imag = alm_imag / np.sqrt(2.0)
    alm_real = np.where(emm == 0, alm_real * np.sqrt(2.0), alm_real)
    alm_imag = np.where(emm == 0, 0.0, alm_imag)
    return alm_real + 1j * alm_imag


def sample_latent_maps(
    cl_latent: ArrayLike,
    nside: int,
    generation_lmax: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate correlated latent Gaussian HEALPix maps."""

    hp = _healpy()
    cl_array = np.asarray(cl_latent, dtype=float)
    n_bins = cl_array.shape[0]
    standard_alm = draw_standard_alm(n_bins, generation_lmax, rng)
    latent_alm = apply_latent_spectra(standard_alm, cl_array, generation_lmax)
    return np.asarray(
        [
            hp.alm2map(
                np.ascontiguousarray(latent_alm[bin_index]),
                nside,
                lmax=generation_lmax,
                pol=False,
            )
            for bin_index in range(n_bins)
        ]
    )


def transform_latent_maps(
    latent_maps: ArrayLike,
    order: int,
    transform_params: ArrayLike,
) -> NDArray[np.float64]:
    """Transform latent Gaussian maps into scalar-field maps."""

    order = validate_order(order)
    latent_array = np.asarray(latent_maps, dtype=float)
    params_array = np.asarray(transform_params, dtype=float)
    if latent_array.ndim != 2:
        raise ValueError("latent_maps must have shape (n_bins, n_pix).")
    if params_array.shape != (latent_array.shape[0], order):
        raise ValueError("transform_params must have shape (n_bins, order).")

    return np.asarray(
        [
            evaluate_transform(latent_array[bin_index], order, params_array[bin_index])
            for bin_index in range(latent_array.shape[0])
        ]
    )


def apply_pixel_window(
    field_maps: ArrayLike,
    nside: int,
    pixwin_by_ell: ArrayLike,
    pixwin_lmax: int,
) -> NDArray[np.float64]:
    """Apply a HEALPix pixel window to scalar-field maps."""

    maps = np.asarray(field_maps, dtype=float)
    pixwin = np.asarray(pixwin_by_ell, dtype=float)
    if maps.ndim != 2:
        raise ValueError("field_maps must have shape (n_bins, n_pix).")
    hp = _healpy()
    if pixwin.shape != (hp.Alm.getsize(pixwin_lmax),):
        raise ValueError("pixwin_by_ell must be indexed at each alm mode.")
    if not np.all(np.isfinite(pixwin)):
        raise ValueError("pixwin values must be finite.")

    windowed_maps = []
    for bin_index in range(maps.shape[0]):
        field_alm = hp.map2alm(maps[bin_index], lmax=pixwin_lmax)
        field_alm = field_alm * pixwin
        windowed_maps.append(hp.alm2map(field_alm, nside, lmax=pixwin_lmax))
    return np.asarray(windowed_maps)
