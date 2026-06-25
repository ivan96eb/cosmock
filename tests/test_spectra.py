import numpy as np

from cosmock.spectra import field_cl_to_latent_cl


def test_field_cl_to_latent_cl_returns_symmetric_finite_spectra():
    cl_field = np.zeros((2, 2, 4))
    cl_field[0, 0, 2:] = [1.0e-4, 8.0e-5]
    cl_field[1, 1, 2:] = [1.2e-4, 9.0e-5]
    cl_field[0, 1, 2:] = [2.0e-5, 1.0e-5]
    cl_field[1, 0] = cl_field[0, 1]
    transform_params = np.array([[0.2, 1.0], [0.25, 1.1]])

    cl_latent = field_cl_to_latent_cl(
        cl_field,
        transform_params,
        2,
        quad_order=2,
        n_jobs=1,
    )

    assert cl_latent.shape == cl_field.shape
    assert np.all(np.isfinite(cl_latent))
    np.testing.assert_allclose(cl_latent, np.swapaxes(cl_latent, 0, 1))
    np.testing.assert_allclose(cl_latent[:, :, 0], 1.0e-20 * np.eye(2))
    np.testing.assert_allclose(cl_latent[:, :, 1], 1.0e-20 * np.eye(2))
