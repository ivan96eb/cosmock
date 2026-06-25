"""Public function-first workflow for scalar-field mock generation."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fitting import fit_transform_params, histogram_gaussianized
from .generation import apply_pixel_window, sample_latent_maps, transform_latent_maps
from .spectra import field_cl_to_latent_cl
from .transforms import validate_order
from .types import FieldMockFit

logger = logging.getLogger(__name__)


def _healpy():
    import healpy as hp

    return hp


def _validate_field_maps(field_maps: ArrayLike) -> tuple[NDArray[np.float64], int, int]:
    maps = np.asarray(field_maps, dtype=float)
    if maps.ndim != 2:
        raise ValueError("field_maps must have shape (n_bins, n_pix).")
    if maps.shape[0] < 1 or maps.shape[1] < 1:
        raise ValueError("field_maps must contain at least one bin and one pixel.")
    if not np.all(np.isfinite(maps)):
        raise ValueError("field_maps must contain only finite values.")

    hp = _healpy()
    try:
        nside = hp.npix2nside(maps.shape[1])
    except ValueError as exc:
        message = "field_maps second axis must be a valid HEALPix npix."
        raise ValueError(message) from exc
    if hp.nside2npix(nside) != maps.shape[1]:
        raise ValueError("field_maps second axis must be a valid HEALPix npix.")
    return maps, maps.shape[0], nside


def _validate_cl_field(
    cl_field: ArrayLike,
    n_bins: int,
    min_n_ell: int,
) -> NDArray[np.float64]:
    cl_array = np.asarray(cl_field, dtype=float)
    if cl_array.ndim != 3:
        raise ValueError("cl_field must have shape (n_bins, n_bins, n_ell).")
    if cl_array.shape[0] != n_bins or cl_array.shape[1] != n_bins:
        raise ValueError("cl_field first two axes must match field_maps bins.")
    if cl_array.shape[2] < min_n_ell:
        raise ValueError(
            f"cl_field must contain at least {min_n_ell} multipoles for this nside."
        )
    if not np.all(np.isfinite(cl_array)):
        raise ValueError("cl_field must contain only finite values.")
    if not np.allclose(cl_array, np.swapaxes(cl_array, 0, 1)):
        raise ValueError("cl_field must be symmetric across its first two axes.")
    return cl_array


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def fit_field_model(
    field_maps: ArrayLike,
    cl_field: ArrayLike,
    order: int,
    *,
    n_histogram_bins: int = 1000,
    x_range: tuple[float, float] = (-4.5, 4.5),
    xig_grid_size: int = 75,
    quad_order: int = 3,
    n_nodes: int = 16,
    n_jobs: int = 1,
) -> FieldMockFit:
    """Fit a scalar-field mock-generation model.

    Parameters
    ----------
    field_maps : array_like of shape ``(n_bins, n_pix)``
        HEALPix scalar-field maps with one row per field/bin.
    cl_field : array_like of shape ``(n_bins, n_bins, n_ell)``
        Target non-Gaussian scalar-field angular power spectra.
    order : {2, 3}
        Point-transformation order.
    n_histogram_bins : int, optional
        Number of Gaussianized bins used for each one-point fit.
    x_range : tuple of float, optional
        Standard-normal range retained for transformation fitting.
    xig_grid_size, quad_order, n_nodes : int, optional
        Numerical controls for the spectra conversion.
    n_jobs : int, optional
        Number of joblib workers for spectra conversion. The default is serial.

    Returns
    -------
    FieldMockFit
        Fitted state for :func:`generate_field_mocks`.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If shapes, values, HEALPix resolution, or numerical controls are invalid.
    """

    order = validate_order(order)
    n_histogram_bins = _validate_positive_int(n_histogram_bins, "n_histogram_bins")
    maps, n_bins, nside = _validate_field_maps(field_maps)
    generation_lmax = 3 * nside - 1
    pixwin_lmax = 2 * nside
    cl_array = _validate_cl_field(cl_field, n_bins, generation_lmax + 1)

    transform_params = np.zeros((n_bins, order), dtype=float)
    for bin_index in range(n_bins):
        x_fit, y_fit = histogram_gaussianized(
            maps[bin_index],
            n_histogram_bins,
            x_range=x_range,
        )
        transform_params[bin_index] = fit_transform_params(
            x_fit,
            y_fit,
            order,
            cl_array[bin_index, bin_index, : generation_lmax + 1],
        )
    logger.debug("Fitted order-%s transform parameters.", order)

    cl_latent_full = field_cl_to_latent_cl(
        cl_array,
        transform_params,
        order,
        xig_grid_size=xig_grid_size,
        quad_order=quad_order,
        n_nodes=n_nodes,
        n_jobs=n_jobs,
    )
    cl_latent = cl_latent_full[:, :, : generation_lmax + 1]
    logger.debug("Converted target spectra to latent Gaussian spectra.")

    return FieldMockFit(
        transform_params=transform_params,
        order=order,
        cl_latent=cl_latent,
        map_shape=tuple(maps.shape),
        nside=nside,
        generation_lmax=generation_lmax,
        pixwin_lmax=pixwin_lmax,
    )


def _validate_fit(fit: FieldMockFit) -> FieldMockFit:
    if not isinstance(fit, FieldMockFit):
        raise TypeError("fit must be a FieldMockFit returned by fit_field_model.")
    validate_order(fit.order)
    if len(fit.map_shape) != 2:
        raise ValueError("fit.map_shape must be a two-element tuple.")
    n_bins, n_pix = fit.map_shape
    hp = _healpy()
    if hp.nside2npix(fit.nside) != n_pix:
        raise ValueError("fit.nside is inconsistent with fit.map_shape.")
    if fit.transform_params.shape != (n_bins, fit.order):
        raise ValueError("fit.transform_params has an invalid shape.")
    if fit.cl_latent.shape[0] != n_bins or fit.cl_latent.shape[1] != n_bins:
        raise ValueError("fit.cl_latent first two axes must match fit.map_shape.")
    if fit.cl_latent.shape[2] <= fit.generation_lmax:
        raise ValueError("fit.cl_latent does not cover fit.generation_lmax.")
    if not np.all(np.isfinite(fit.transform_params)):
        raise ValueError("fit.transform_params must contain only finite values.")
    if not np.all(np.isfinite(fit.cl_latent)):
        raise ValueError("fit.cl_latent must contain only finite values.")
    return fit


def _pixwin_by_alm(
    nside: int,
    pixwin_lmax: int,
    pixwin: ArrayLike | None,
) -> NDArray[np.float64]:
    hp = _healpy()
    if pixwin is None:
        pixwin_array = hp.pixwin(nside, pol=False, lmax=pixwin_lmax)
    else:
        pixwin_array = np.asarray(pixwin, dtype=float)
    if pixwin_array.ndim != 1 or pixwin_array.size <= pixwin_lmax:
        raise ValueError("pixwin must be one-dimensional and cover pixwin_lmax.")
    if not np.all(np.isfinite(pixwin_array)):
        raise ValueError("pixwin must contain only finite values.")

    ell, _ = hp.Alm.getlm(pixwin_lmax)
    return pixwin_array[ell]


def generate_field_mocks(
    fit: FieldMockFit,
    n_mocks: int,
    *,
    seed: int | np.random.SeedSequence | np.random.Generator | None = None,
    pixwin: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Generate scalar-field mock-map cubes from a fitted model.

    Parameters
    ----------
    fit : FieldMockFit
        Fitted state returned by :func:`fit_field_model`.
    n_mocks : int
        Number of mock cubes to generate.
    seed : int, SeedSequence, Generator, or None, optional
        Random seed or generator used for deterministic sampling.
    pixwin : array_like, optional
        HEALPix pixel window indexed by multipole. If omitted, the window is
        computed with ``healpy.pixwin`` for ``fit.nside``.

    Returns
    -------
    ndarray of shape ``(n_mocks, n_bins, n_pix)``
        Mock scalar-field maps.

    Raises
    ------
    TypeError
        If ``fit`` is not a :class:`FieldMockFit`.
    ValueError
        If ``n_mocks`` or fitted state values are invalid.
    """

    fit = _validate_fit(fit)
    n_mocks = _validate_positive_int(n_mocks, "n_mocks")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    pixwin_at_ell = _pixwin_by_alm(fit.nside, fit.pixwin_lmax, pixwin)

    mocks = np.zeros((n_mocks, *fit.map_shape), dtype=float)
    for mock_index in range(n_mocks):
        latent_maps = sample_latent_maps(
            fit.cl_latent,
            fit.nside,
            fit.generation_lmax,
            rng,
        )
        field_maps = transform_latent_maps(
            latent_maps,
            fit.order,
            fit.transform_params,
        )
        mocks[mock_index] = apply_pixel_window(
            field_maps,
            fit.nside,
            pixwin_at_ell,
            fit.pixwin_lmax,
        )
    return mocks
