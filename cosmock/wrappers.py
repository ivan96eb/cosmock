import numpy as np
import healpy as hp
from structs import NonlinParameters
from fitter import histogramer2d, fit_gn_with_constraint
from Cls import C_NG_to_C_G
from mocker import get_y_maps, get_kappa_pixwin


def fit_parameters(maps, Cl_delta, N):

    Nbins = maps.shape[0]
    Nside = hp.npix2nside(maps.shape[1])

    lbda = np.zeros((Nbins, N))
    for i in range(Nbins):
        x, y = histogramer2d(maps[i], 1000)
        lbda[i] = fit_gn_with_constraint(x, y, N, Cl_delta[i,i,3*Nside])
    
    print(f'Done fitting the G{N} parameters')

    Cl_G = C_NG_to_C_G(Cl_delta, lbda, Nbins, N)
    Cl_G = Cl_G[:,:,:3*Nside]

    print('Done finding the power spectrum of the underlying gaussian random field')

    params = NonlinParameters(lbda = lbda, N = N, Cl_Gauss = Cl_G, mapshape = maps.shape)

    return params


def generate_mocks(params, Nmocks, pixwin=None):

    lbda = params.lbda
    N = params.N
    Cl_G = params.Cl_Gauss
    mapshape = params.mapshape
    Nbins = mapshape[0]
    Npix  = mapshape[1]
    Nside = hp.npix2nside(Npix)
    gen_lmax = 3*Nside-1
    ell, emm = hp.Alm.getlm(gen_lmax)

    if pixwin is None:
        pixwin = hp.pixwin(Nside, False, gen_lmax)

    mocks = np.zeros((Nmocks, *mapshape))
    for i in range(Nmocks):
        y_map = get_y_maps(Cl_G, Nside, Nbins, gen_lmax)
        mocks[i] = get_kappa_pixwin(y_map, Nbins, N, lbda, Nside, pixwin[ell])

    return mocks