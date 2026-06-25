import numpy as np

from cosmock import FieldMockFit, fit_field_model

FIELD_MAP_PATH = "data/Kappa_Gower_St_ID_44_DESy3_tomography_Nside_256.npy"
CL_FIELD_PATH = "data/UNBIASED_3point75nsideminus1_Cls_NG_Gower_St_ID_44.npy"
CL_LATENT_G2_PATH = "data/Cl_Gauss_G2.npy"
CL_LATENT_G3_PATH = "data/Cl_Gauss_G3.npy"
TRANSFORM_PARAMS_G2_PATH = "data/fitted_params_G2.npy"
TRANSFORM_PARAMS_G3_PATH = "data/fitted_params_G3.npy"

PARAM_RTOL = 2.0e-6
SPECTRA_RTOL = 1.0e-7


def test_fit_field_model_matches_bundled_scalar_field_regression_data():
    field_maps = np.load(FIELD_MAP_PATH)
    cl_field = np.load(CL_FIELD_PATH)

    fit_g2 = fit_field_model(field_maps, cl_field, 2, n_jobs=4)
    assert isinstance(fit_g2, FieldMockFit)
    np.testing.assert_allclose(
        fit_g2.transform_params,
        np.load(TRANSFORM_PARAMS_G2_PATH),
        rtol=PARAM_RTOL,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        fit_g2.cl_latent,
        np.load(CL_LATENT_G2_PATH),
        rtol=SPECTRA_RTOL,
        atol=1.0e-12,
    )

    fit_g3 = fit_field_model(field_maps, cl_field, 3, n_jobs=4)
    assert isinstance(fit_g3, FieldMockFit)
    np.testing.assert_allclose(
        fit_g3.transform_params,
        np.load(TRANSFORM_PARAMS_G3_PATH),
        rtol=PARAM_RTOL,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        fit_g3.cl_latent,
        np.load(CL_LATENT_G3_PATH),
        rtol=SPECTRA_RTOL,
        atol=1.0e-12,
    )
