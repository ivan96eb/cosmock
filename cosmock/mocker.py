"""Low-level HEALPix helpers for generating mock kappa maps."""

import numpy as np 
import healpy as hp 
from .Gn import Gn

def get_xlm(xlm_real, xlm_imag,gen_lmax,nbins):
    """Pack real and imaginary Gaussian coefficients into HEALPix alm arrays.

    Parameters
    ----------
    xlm_real : numpy.ndarray
        Real Gaussian coefficients with shape ``(nbins, n_real_modes)`` for
        multipoles ``ell > 1``.
    xlm_imag : numpy.ndarray
        Imaginary Gaussian coefficients with shape ``(nbins, n_imag_modes)``
        for multipoles ``ell > 1`` and ``m > 0``.
    gen_lmax : int
        Maximum multipole for the generated alm arrays.
    nbins : int
        Number of tomographic bins.

    Returns
    -------
    numpy.ndarray
        Complex alm coefficients with shape
        ``(nbins, hp.Alm.getsize(gen_lmax))``.
    """
    ell, emm = hp.Alm.getlm(gen_lmax)
    #==============================
    _xlm_real = np.zeros((nbins, len(ell)))
    _xlm_imag = np.zeros_like(_xlm_real)
    _xlm_real[:,ell > 1] = xlm_real
    _xlm_imag[:,(ell > 1) & (emm > 0)] = xlm_imag
    xlm = _xlm_real + 1j * _xlm_imag
    #==============================
    return xlm

def generate_xlm(nbins,gen_lmax):
    """Draw standard-normal harmonic coefficients for latent fields.

    Parameters
    ----------
    nbins : int
        Number of tomographic bins.
    gen_lmax : int
        Maximum multipole for the generated alm arrays.

    Returns
    -------
    xlm : numpy.ndarray
        Complex alm array with one row per bin.
    raw_coefficients : list of numpy.ndarray
        Real and imaginary normal draws used to build ``xlm``.
    """
    ell, emm = hp.Alm.getlm(gen_lmax)
    xlm_real = np.random.normal(size=(nbins, (ell > 1).sum()))
    xlm_imag = np.random.normal(size=(nbins, ((ell > 1) & (emm > 0)).sum()))
    xlm = get_xlm(xlm_real, xlm_imag, gen_lmax, nbins)
    return xlm, [xlm_real,xlm_imag]

def apply_cl_G(xlm, Cl_G, gen_lmax):
    """Apply latent Gaussian spectra to standard-normal alm coefficients.

    Parameters
    ----------
    xlm : numpy.ndarray
        Complex standard-normal alm coefficients with shape
        ``(N_bins, hp.Alm.getsize(gen_lmax))``.
    Cl_G : numpy.ndarray
        Latent Gaussian angular power spectra with shape
        ``(N_bins, N_bins, N_ell)``.
    gen_lmax : int
        Maximum multipole used for generation.

    Returns
    -------
    numpy.ndarray
        Correlated latent Gaussian alm coefficients with the same shape as
        ``xlm``.
    """
    Cl_G_T = np.moveaxis(Cl_G, 2, 0)
    L_T = np.linalg.cholesky(Cl_G_T)
    L_G = np.moveaxis(L_T, 0, 2)
    gen_ell, gen_emm = hp.Alm.getlm(gen_lmax)
    L_expanded = L_G[:, :, gen_ell]
    ylm_real = np.einsum('ijm,jm->im', L_expanded, xlm.real) / np.sqrt(2)
    ylm_imag = np.einsum('ijm,jm->im', L_expanded, xlm.imag) / np.sqrt(2)
    ylm_real = np.where(gen_emm == 0, ylm_real * np.sqrt(2), ylm_real)
    ylm_imag = np.where(gen_emm == 0, 0.0, ylm_imag)
    return ylm_real + 1j * ylm_imag 

def get_y_maps(cl,nside,nbins,gen_lmax,xlms=None):
    """Generate latent Gaussian HEALPix maps.

    Parameters
    ----------
    cl : numpy.ndarray
        Latent Gaussian spectra with shape ``(N_bins, N_bins, N_ell)``.
    nside : int
        HEALPix ``nside`` resolution of the output maps.
    nbins : int
        Number of tomographic bins.
    gen_lmax : int
        Maximum multipole used for generation.
    xlms : numpy.ndarray, optional
        Precomputed standard-normal alm coefficients. If omitted, new
        coefficients are drawn.

    Returns
    -------
    numpy.ndarray
        Latent Gaussian maps with shape
        ``(N_bins, hp.nside2npix(nside))``.
    """
    if xlms is not None:
        xlm = xlms
        _xlm = None
    else:
        xlm, _xlm = generate_xlm(nbins,gen_lmax)
    y_lm, xlm = apply_cl_G(xlm, cl,gen_lmax), _xlm
    y_maps = []
    for i in range(nbins):
        y_map = hp.alm2map(np.ascontiguousarray(y_lm[i]), nside, lmax=gen_lmax, pol=False)
        y_maps.append(y_map)    
    return np.array(y_maps)    

def get_kappa(y_maps,nbins,N,fitted_params):
    """Transform latent Gaussian maps into kappa maps.

    Parameters
    ----------
    y_maps : numpy.ndarray
        Latent Gaussian maps with shape ``(N_bins, N_pix)``.
    nbins : int
        Number of tomographic bins.
    N : int
        Transformation order, currently ``2`` or ``3``.
    fitted_params : numpy.ndarray
        Fitted transform parameters with one row per tomographic bin.

    Returns
    -------
    numpy.ndarray
        Dimensionless kappa maps with shape
        ``(N_bins, N_pix)``.
    """
    k_list = []
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        k = k_nf
        k_list.append(k)  
    k_arr  = np.array(k_list)
    return k_arr  

def get_kappa_pixwin(y_maps,nbins,N,fitted_params,nside,pixwinatell):
    """Transform latent maps into kappa maps and apply a pixel window.

    Parameters
    ----------
    y_maps : numpy.ndarray
        Latent Gaussian maps with shape ``(N_bins, N_pix)``.
    nbins : int
        Number of tomographic bins.
    N : int
        Transformation order, currently ``2`` or ``3``.
    fitted_params : numpy.ndarray
        Fitted transform parameters with one row per tomographic bin.
    nside : int
        HEALPix ``nside`` resolution.
    pixwinatell : numpy.ndarray
        Pixel-window values evaluated at each alm multipole up to
        ``2 * nside``.

    Returns
    -------
    numpy.ndarray
        Pixel-windowed dimensionless kappa maps with shape
        ``(N_bins, hp.nside2npix(nside))``.
    """
    k_list = []
    lmax = 2*nside
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        k = k_nf
        klm = hp.map2alm(k,lmax=lmax)
        klm = klm * pixwinatell 
        k = hp.alm2map(klm,nside)
        k_list.append(k)  
    k_arr  = np.array(k_list)

    return k_arr  

def get_kappa_lm_pixwin(y_maps,nbins,N,fitted_params,nside,pixwinatell):
    """Transform latent maps and return pixel-windowed kappa alm arrays.

    Parameters
    ----------
    y_maps : numpy.ndarray
        Latent Gaussian maps with shape ``(N_bins, N_pix)``.
    nbins : int
        Number of tomographic bins.
    N : int
        Transformation order, currently ``2`` or ``3``.
    fitted_params : numpy.ndarray
        Fitted transform parameters with one row per tomographic bin.
    nside : int
        HEALPix ``nside`` resolution.
    pixwinatell : numpy.ndarray
        Pixel-window values evaluated at each alm multipole up to
        ``2 * nside``.

    Returns
    -------
    numpy.ndarray
        Complex alm coefficients for the transformed kappa fields with one
        row per tomographic bin.
    """
    k_lm_list = []
    lmax = 2*nside
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        k = k_nf
        klm = hp.map2alm(k,lmax=lmax)
        klm = klm * pixwinatell 
        k_lm_list.append(klm)  
    k_lm_arr  = np.array(k_lm_list)

    return k_lm_arr  
