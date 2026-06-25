import numpy as np
import pytest

from cosmock import FieldMockFit, generate_field_mocks


def small_fit():
    cl_latent = np.zeros((1, 1, 3))
    cl_latent[0, 0] = [1.0e-20, 1.0e-20, 1.0e-4]
    return FieldMockFit(
        transform_params=np.array([[0.1, 1.0]]),
        order=2,
        cl_latent=cl_latent,
        map_shape=(1, 12),
        nside=1,
        generation_lmax=2,
        pixwin_lmax=2,
    )


def test_generate_field_mocks_is_deterministic_with_seed():
    fit = small_fit()
    pixwin = np.ones(3)

    first = generate_field_mocks(fit, 2, seed=123, pixwin=pixwin)
    second = generate_field_mocks(fit, 2, seed=123, pixwin=pixwin)

    assert first.shape == (2, 1, 12)
    np.testing.assert_allclose(first, second)


def test_generate_field_mocks_rejects_invalid_n_mocks():
    with pytest.raises(ValueError, match="n_mocks"):
        generate_field_mocks(small_fit(), 0, pixwin=np.ones(3))
