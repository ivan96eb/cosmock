"""Public workflow functions for fitting and sampling cosmock maps."""

import numpy as np
import healpy as hp
from .structs import NonlinParameters
from .fitter import histogramer2d, fit_gn_with_constraint
from .Cls import C_NG_to_C_G
from .mocker import get_y_maps, get_kappa_pixwin


def fit_parameters(maps, Cl_delta, N):
    """Fit transformation parameters for a kappa-map data set.

    This is the first step in the standard ``cosmock`` workflow. It fits a
    ``G_N`` point-transformation model for each tomographic bin and converts
    the target non-Gaussian kappa spectra into latent Gaussian spectra.

    Parameters
    ----------
    maps : numpy.ndarray
        Dimensionless kappa maps with shape ``(N_bins, N_pix)``.
    Cl_delta : numpy.ndarray
        Target non-Gaussian angular power spectra with shape
        ``(N_bins, N_bins, N_ell)``.
    N : int
        Transformation order. The current implementation supports ``2`` and
        ``3``.

    Returns
    -------
    NonlinParameters
        Fitted nonlinear parameters, latent Gaussian
        spectra, transformation order, and the original map shape.

    Raises
    ------
    AssertionError
        If ``N`` is not ``2`` or ``3``.
    """

    assert N == 2 or N == 3, f'N={N} is not a supported transformation'

    Nbins = maps.shape[0]
    Nside = hp.npix2nside(maps.shape[1])

    lbda = np.zeros((Nbins, N))
    for i in range(Nbins):
        x, y = histogramer2d(maps[i], 1000)
        lbda[i] = fit_gn_with_constraint(x, y, N, Cl_delta[i,i,:3*Nside])
    
    print(f'Done fitting the G{N} parameters')

    Cl_G = C_NG_to_C_G(Cl_delta, lbda, Nbins, N)
    Cl_G = Cl_G[:,:,:3*Nside]

    print('Done finding the power spectrum of the underlying gaussian random field')

    params = NonlinParameters(lbda = lbda, N = N, Cl_Gauss = Cl_G, mapshape = maps.shape)

    return params


def generate_mocks(params, Nmocks, pixwin=None):
    """Generate mock kappa-map cubes from fitted parameters.

    This is the second step in the standard ``cosmock`` workflow. It samples
    latent Gaussian HEALPix maps, applies the fitted nonlinear transform, and
    optionally applies a HEALPix pixel window.

    Parameters
    ----------
    params : NonlinParameters
        Parameters returned by :func:`fit_parameters`.
    Nmocks : int
        Number of mock cubes to generate.
    pixwin : numpy.ndarray, optional
        HEALPix pixel window values indexed by multipole. If ``None``,
        ``healpy.pixwin`` is computed for the fitted map resolution.

    Returns
    -------
    numpy.ndarray
        Mock kappa maps with shape
        ``(Nmocks, N_bins, N_pix)``. Kappa is dimensionless lensing
        convergence.
    """

    lbda = params.lbda
    N = params.N
    Cl_G = params.Cl_Gauss
    mapshape = params.mapshape
    Nbins    = mapshape[0]
    Npix     = mapshape[1]
    Nside    = hp.npix2nside(Npix)
    gen_lmax = 3*Nside-1
    lmax     = 2*Nside
    ell, emm = hp.Alm.getlm(lmax)

    if pixwin is None:
        pixwin = hp.pixwin(Nside, False, gen_lmax)

    mocks = np.zeros((Nmocks, *mapshape))
    for i in range(Nmocks):
        y_map = get_y_maps(Cl_G, Nside, Nbins, gen_lmax)
        mocks[i] = get_kappa_pixwin(y_map, Nbins, N, lbda, Nside, pixwin[ell])

    return mocks
