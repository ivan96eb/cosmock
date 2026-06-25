import numpy as np
import pytest

from cosmock import fit_field_model


def valid_small_inputs():
    field_maps = np.linspace(-1.0, 1.0, 12).reshape(1, 12)
    cl_field = np.zeros((1, 1, 3))
    cl_field[0, 0] = [1.0e-20, 1.0e-20, 1.0e-4]
    return field_maps, cl_field


def test_fit_field_model_rejects_unsupported_order():
    field_maps, cl_field = valid_small_inputs()

    with pytest.raises(ValueError, match="order"):
        fit_field_model(field_maps, cl_field, 1)


def test_fit_field_model_rejects_non_healpix_map_size():
    field_maps, cl_field = valid_small_inputs()

    with pytest.raises(ValueError, match="HEALPix"):
        fit_field_model(field_maps[:, :10], cl_field, 2)


def test_fit_field_model_rejects_mismatched_spectra_shape():
    field_maps, cl_field = valid_small_inputs()

    with pytest.raises(ValueError, match="first two axes"):
        fit_field_model(field_maps, np.repeat(cl_field, 2, axis=0), 2)
