"""Typed containers for the public cosmock workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FieldMockFit:
    """Fitted state needed to generate scalar-field mock maps.

    Parameters
    ----------
    transform_params : ndarray of shape ``(n_bins, order)``
        Fitted point-transformation parameters for each tomographic bin.
    order : {2, 3}
        Point-transformation order.
    cl_latent : ndarray of shape ``(n_bins, n_bins, n_ell)``
        Angular power spectra for the latent Gaussian fields.
    map_shape : tuple of int
        Original field-map shape ``(n_bins, n_pix)``.
    nside : int
        HEALPix nside resolution inferred from ``n_pix``.
    generation_lmax : int
        Maximum multipole used to sample latent Gaussian maps.
    pixwin_lmax : int
        Maximum multipole used when applying the HEALPix pixel window.
    """

    transform_params: NDArray[np.float64]
    order: int
    cl_latent: NDArray[np.float64]
    map_shape: tuple[int, int]
    nside: int
    generation_lmax: int
    pixwin_lmax: int
