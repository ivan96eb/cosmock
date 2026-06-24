import numpy as np 
import healpy as hp 
from .Gn import Gn

def get_xlm(xlm_real, xlm_imag,gen_lmax,nbins):
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
    ell, emm = hp.Alm.getlm(gen_lmax)
    xlm_real = np.random.normal(size=(nbins, (ell > 1).sum()))
    xlm_imag = np.random.normal(size=(nbins, ((ell > 1) & (emm > 0)).sum()))
    xlm = get_xlm(xlm_real, xlm_imag, gen_lmax, nbins)
    return xlm, [xlm_real,xlm_imag]

def apply_cl_G(xlm, Cl_G, gen_lmax):
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
    return np.array(y_maps),xlm    

def get_kappa(y_maps,nbins,N,fitted_params):
    k_list = []
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        k = k_nf
        k_list.append(k)  
    k_arr  = np.array(k_list)
    return k_arr  

def get_kappa_pixwin(y_maps,nbins,N,fitted_params,nside,pixwinatell):
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
