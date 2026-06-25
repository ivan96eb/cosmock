from cosmock import generate_mocks
from cosmock.structs import NonlinParameters
import numpy as np
import pytest

def test_generate_mocks():

    path_to_test_map = '../data/Kappa_Gower_St_ID_44_DESy3_tomography_Nside_256.npy'
    path_to_test_CL_G_2 = '../data/Cl_Gauss_G2.npy'
    path_to_test_CL_G_3 = '../data/Cl_Gauss_G3.npy'
    path_to_fit_params_G2 = '../data/fitted_params_G2.npy'
    path_to_fit_params_G3 = '../data/fitted_params_G3.npy'
    path_to_pixwin = '../data/pixwin_256.npy'

    testmap = np.load(path_to_test_map)
    Cl_G2 = np.load(path_to_test_CL_G_2)
    Cl_G3 = np.load(path_to_test_CL_G_3)
    lbda_G2 = np.load(path_to_fit_params_G2)
    lbda_G3 = np.load(path_to_fit_params_G3)
    pixwin = np.load(path_to_pixwin)

    params_G2 = NonlinParameters(lbda_G2, 2, Cl_G2, testmap.shape)
    params_G3 = NonlinParameters(lbda_G3, 3, Cl_G3, testmap.shape)

    generate_mocks(params_G2, 1, pixwin)
    generate_mocks(params_G2, 1)
    generate_mocks(params_G3, 1, pixwin)
    generate_mocks(params_G3, 1)