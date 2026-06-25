from cosmock import fit_parameters
import numpy as np
import pytest


def test_fit_parameters_N():

    path_to_test_map = './data/Kappa_Gower_St_ID_44_DESy3_tomography_Nside_256.npy'
    path_to_test_Cl = './data/UNBIASED_3point75nsideminus1_Cls_NG_Gower_St_ID_44.npy'

    testmap = np.load(path_to_test_map)
    testCl = np.load(path_to_test_Cl)

    for N in [0, 1, 4]:
        with pytest.raises(AssertionError):
            fit_parameters(testmap, testCl, N)

def test_fit_parameters_Cl():

    path_to_test_map = './data/Kappa_Gower_St_ID_44_DESy3_tomography_Nside_256.npy'
    path_to_test_Cl = './data/UNBIASED_3point75nsideminus1_Cls_NG_Gower_St_ID_44.npy'
    path_to_test_CL_G_2 = './data/Cl_Gauss_G2.npy'
    path_to_test_CL_G_3 = './data/Cl_Gauss_G3.npy'

    testmap = np.load(path_to_test_map)
    testCl = np.load(path_to_test_Cl)
    Cl_G2 = np.load(path_to_test_CL_G_2)
    Cl_G3 = np.load(path_to_test_CL_G_3)

    N = 2
    params = fit_parameters(testmap, testCl, N)
    params.Cl_Gauss == pytest.approx(Cl_G2)

    N = 3
    params = fit_parameters(testmap, testCl, N)
    params.Cl_Gauss == pytest.approx(Cl_G3)