"""Structured containers returned by the public cosmock workflow."""

from typing import NamedTuple
import numpy as np

class NonlinParameters(NamedTuple):
    """Fitted parameters needed to generate mock kappa maps.

    Notes
    -----
    ``lbda`` is a ``numpy.ndarray`` containing fitted ``G_N``
    transformation parameters with shape ``(N_bins, N)``.
    ``N`` is the transformation order used during fitting, currently ``2``
    or ``3``.
    ``Cl_Gauss`` is a ``numpy.ndarray`` containing latent Gaussian angular
    power spectra with shape ``(N_bins, N_bins, N_ell)``.
    ``mapshape`` is the original kappa-map shape ``(N_bins, N_pix)``.
    """

    lbda: np.ndarray
    N: int
    Cl_Gauss: np.ndarray
    mapshape: tuple
